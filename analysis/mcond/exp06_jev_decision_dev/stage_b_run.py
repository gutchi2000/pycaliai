# -*- coding: utf-8 -*-
"""
stage_b_run.py — Stage B主評価+感度分析の実API実行 (2026-09-20)
================================================================================
実行順6番目。主評価: 2024-2025年の全適格レース、1レース1回のみ問い合わせ
(jev_client.query_jevのキャッシュが単発応答ポリシーを強制)。感度分析: 固定200レース
(stage_b_sample.py、結果ラベル不使用で抽出済み)を各3回uncachedで問い合わせる。

安全上限: 累積input_tokensが20,000,000へ到達したら新規呼び出しを停止し、
それまでの結果を保存する(このスクリプトが強制する)。

匿名化: Jevへ送るstateにrid16(レースID)そのものは含めない。代わりに連番の
anon_race_idを発行し、rid16との対応表(out/stage_b_race_id_map.json)はローカルにのみ
保存する(Jevには一切送らない)。結果ラベル(top3/win等)もJevへは送らない
(判断時点より後の情報を入力しない、というspec絶対条件)。

再実行時の挙動: query_jevは既にキャッシュ済みのinput_hashを再送しない(課金されない)。
このスクリプトを中断後に再実行すれば、未処理分から自動的に再開する。

実行: python -m analysis.mcond.exp06_jev_decision_dev.stage_b_run [--sensitivity-only]
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.exp06_jev_decision_dev import historical_state as HS  # noqa: E402
from analysis.mcond.exp06_jev_decision_dev import jev_client as JC  # noqa: E402
from analysis.mcond.exp06_jev_decision_dev import stage_b_sample as SBS  # noqa: E402
from analysis.mcond.exp06_jev_decision_dev.ood_support import (  # noqa: E402
    SupportModel, CONTINUOUS_COLS, CATEGORICAL_COLS)

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / "spec.json").read_text(encoding="utf-8"))
PROMPT_SCHEMA_HASH = "exp06_v2_20260920"
RACE_ID_MAP_PATH = HERE / "out" / "stage_b_race_id_map.json"
PRIMARY_LOG_PATH = HERE / "out" / "stage_b_primary_progress.jsonl"
SENSITIVITY_LOG_PATH = HERE / "out" / "stage_b_sensitivity_results.jsonl"
MAX_INPUT_TOKENS_CAP = 20_000_000
LOG_EVERY = 100

# data_availability_audit(spec.json)により2023-2025で取得できないフィールド
_UNAVAILABLE_FIELDS = {
    "odds_t20": None, "odds_t10": None, "odds_change_rate": None, "popularity_rank_change": None,
    "candidate_ticket_count": None, "candidate_ticket_probs": None, "current_odds": None,
    "predicted_confirmed_odds": None, "ev_point_estimate": None, "ev_lower_bound": None,
    "min_payout": None, "same_horse_concentration_rate": None,
}


def _load_race_id_map() -> dict:
    if RACE_ID_MAP_PATH.exists():
        return json.loads(RACE_ID_MAP_PATH.read_text(encoding="utf-8"))
    return {}


def _save_race_id_map(m: dict) -> None:
    RACE_ID_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    RACE_ID_MAP_PATH.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")


def _anon_id_for(rid16: str, race_id_map: dict, reverse_map: dict) -> str:
    if rid16 in reverse_map:
        return reverse_map[rid16]
    anon = f"HIST_{len(race_id_map):06d}"
    race_id_map[anon] = rid16
    reverse_map[rid16] = anon
    return anon


def build_jev_state(row, in_dist_support: float, similar_count: int) -> dict:
    """rid16そのものは含めない(呼び出し側でanon_race_idに差し替える)。
    結果ラベルも含めない。data_availability_auditのフィールドはNone+availabilityで明示。"""
    state = {
        "venue": row["venue"], "surface": row["surface"], "distance_band": row["distance_band"],
        "class_band": row["class_band"], "field_size": int(row["field_size"]),
        "m1_top_prob": row["m1_top_prob"], "m3_top_prob": row["m3_top_prob"],
        "m4_top_prob": row["m4_top_prob"],
        "entropy_m1": row["entropy_m1"], "entropy_m3": row["entropy_m3"], "entropy_m4": row["entropy_m4"],
        "model_rank_disagreement": row["model_rank_disagreement"],
        "model_prob_variance": row["model_prob_variance"],
        "market_prob": row["market_prob"], "market_entropy": row["market_entropy"],
        "ai_market_divergence": row["ai_market_divergence"],
        "popularity_band": row["popularity_band"],
        "feature_missing_rate": row["feature_missing_rate"],
        "unknown_category_rate": row["unknown_category_rate"],
        "in_distribution_support": float(in_dist_support),
        "similar_past_case_count": int(similar_count),
    }
    state.update(_UNAVAILABLE_FIELDS)
    state["availability"] = {"odds_trajectory": False, "ticket_candidates": False}
    return state


def _cumulative_input_tokens() -> int:
    total = 0
    if not JC.CACHE_DIR.exists():
        return 0
    for p in JC.CACHE_DIR.glob("*.json"):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
            usage = rec.get("usage") or {}
            total += int(usage.get("input_tokens") or 0)
        except Exception:
            continue
    return total


def run_primary(state_df, support_result, race_id_map, reverse_map) -> dict:
    n_done, n_skipped_cap, n_fail = 0, 0, 0
    t0 = time.time()
    PRIMARY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    for i, (rid16, row) in enumerate(state_df.iterrows()):
        cum_tokens = _cumulative_input_tokens()
        if cum_tokens >= MAX_INPUT_TOKENS_CAP:
            n_skipped_cap = len(state_df) - i
            msg = f"[STOP] 累積input_tokens={cum_tokens}が上限{MAX_INPUT_TOKENS_CAP}へ到達、新規呼び出しを停止"
            print(msg)
            with open(HERE / "out" / "stage_b_errors.log", "a", encoding="utf-8") as f:
                f.write(msg + "\n")
            break
        anon_id = _anon_id_for(rid16, race_id_map, reverse_map)
        support = float(support_result.loc[rid16, "in_distribution_support"])
        count = int(support_result.loc[rid16, "similar_past_case_count"])
        state = build_jev_state(row, support, count)
        result = JC.query_jev(anon_id, PROMPT_SCHEMA_HASH, state, SPEC["questions"],
                              exp05_model_hash="exp05_oos_safe_m1m3m4", market_snapshot_time="historical_tanpuk_pre")
        if not result.get("ok"):
            n_fail += 1
        else:
            n_done += 1
        with open(PRIMARY_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"rid16": rid16, "anon_id": anon_id, "ok": result.get("ok"),
                                "from_cache": result.get("from_cache"),
                                "input_hash": result.get("input_hash")}, ensure_ascii=False) + "\n")
        if (i + 1) % LOG_EVERY == 0:
            _save_race_id_map(race_id_map)
            print(f"[progress] {i+1}/{len(state_df)} done={n_done} fail={n_fail} "
                 f"cum_tokens={cum_tokens} elapsed={time.time()-t0:.0f}s")
    _save_race_id_map(race_id_map)
    return {"n_total": len(state_df), "n_done": n_done, "n_fail": n_fail, "n_skipped_cap": n_skipped_cap}


def run_sensitivity(state_df) -> dict:
    """固定200レースを各3回uncachedで問い合わせる。主評価キャッシュは変更しない
    (JC._call_jev_api_rawを直接呼びquery_jevのキャッシュを経由しない)。"""
    sample = SBS.load_or_build_sample()
    race_ids = [r for r in sample["race_ids"] if r in state_df.index]
    SENSITIVITY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_calls, n_fail = 0, 0
    for i, rid16 in enumerate(race_ids):
        row = state_df.loc[rid16]
        support = float(row.get("_support", 0.5))
        count = int(row.get("_similar_count", 0))
        state = build_jev_state(row, support, count)
        for rep in range(3):
            try:
                raw = JC._call_jev_api_raw(PROMPT_SCHEMA_HASH + "_sensitivity", state, SPEC["questions"])
                ok = True
            except Exception as exc:
                raw = {"error": str(exc)}
                ok = False
                n_fail += 1
            n_calls += 1
            with open(SENSITIVITY_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({"rid16": rid16, "rep": rep, "ok": ok,
                                    "answers": raw.get("answers"), "error": raw.get("error")},
                                   ensure_ascii=False) + "\n")
        if (i + 1) % 20 == 0:
            print(f"[sensitivity] {i+1}/{len(race_ids)} races, {n_calls} calls, {n_fail} failed")
    return {"n_races": len(race_ids), "n_calls": n_calls, "n_fail": n_fail}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensitivity-only", action="store_true")
    ap.add_argument("--primary-only", action="store_true", help="感度分析をスキップする(動作確認用)")
    ap.add_argument("--limit", type=int, default=0, help="テスト用: 主評価の対象レース数を制限")
    args = ap.parse_args()

    print("[stage_b_run] loading historical state...")
    df, feature_lists = HS.load_design_and_features()
    preds = HS.fit_oos_safe_predictions(df, feature_lists)
    state = HS.build_race_state_vectors(df, preds)

    fit_cols = CONTINUOUS_COLS + CATEGORICAL_COLS
    d2023 = state[state["year"] == 2023]
    sm = SupportModel().fit(d2023[fit_cols])
    d2425 = state[state["year"].isin([2024, 2025])].copy()
    support_result = sm.score(d2425[fit_cols])

    race_id_map = _load_race_id_map()
    reverse_map = {v: k for k, v in race_id_map.items()}

    if args.limit:
        d2425 = d2425.iloc[:args.limit]
        support_result = support_result.loc[d2425.index]

    if not args.sensitivity_only:
        print(f"[stage_b_run] primary evaluation: {len(d2425)} races (2024+2025)")
        primary_summary = run_primary(d2425, support_result, race_id_map, reverse_map)
        print("[stage_b_run] primary summary:", json.dumps(primary_summary, ensure_ascii=False))

    if args.primary_only:
        return 0

    d2425["_support"] = support_result["in_distribution_support"]
    d2425["_similar_count"] = support_result["similar_past_case_count"]
    print("[stage_b_run] sensitivity analysis: 200 fixed races x 3 uncached repeats")
    sens_summary = run_sensitivity(d2425)
    print("[stage_b_run] sensitivity summary:", json.dumps(sens_summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
