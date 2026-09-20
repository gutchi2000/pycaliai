# -*- coding: utf-8 -*-
"""
stage_b_run.py — Stage B主評価+感度分析の実API実行 (2026-09-20、state_schema_v3)
================================================================================
2026-09-20夜、state_schema_v2(unknown_category_rate=0埋め)は誤りと判明し、
`out/quarantine_20260920_old_schema/`へ隔離した(MANIFEST.json参照)。本ファイルは
state_schema_v3(unknown_category_rateをstateから完全に省略しavailability=falseで
明示)へ修正済み。旧schemaの応答は削除せず、しかし新評価には一切再利用しない
(prompt_schema_hashが変わったためcompute_input_hashが自動的に別キーになる)。

対象:
  development_2023: 2023年、support reference/development/単純対照モデルfit用。
    Jevへは問い合わせるが主成績には混ぜない(専用ログファイルで物理的に分離)。
    support指標はSupportModel.score_reference_self()でLOOスコアする(自己一致を防ぐ)。
  primary_2024_2025: 2024-2025年、主評価対象。SupportModel.score()で2023年参照集合に
    対するスコアを使う(2023年でfit・再fitしない)。
  sensitivity: 固定200レース(stage_b_sample.py)を各3回uncachedで問い合わせる。
    主評価キャッシュには一切触れない。

安全上限: 累積input_tokensには**旧schema(quarantine済み)の実消費分も含める**
(2026-09-20ユーザー指定)。合計が20,000,000へ到達したら新規呼び出しを停止する。

匿名化: Jevへ送るstateにrid16は含めない。anon_race_idとrid16の対応表は
out/stage_b_race_id_map.json にローカル保存のみ(Jevには送らない)。

実行: python -m analysis.mcond.exp06_jev_decision_dev.stage_b_run [--development-only|--primary-only|--sensitivity-only]
"""
from __future__ import annotations
import argparse
import json
import os
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
STATE_SCHEMA_VERSION = "v3"
PROMPT_SCHEMA_HASH = "exp06_v3_20260920"
RACE_ID_MAP_PATH = HERE / "out" / "stage_b_race_id_map.json"
DEVELOPMENT_LOG_PATH = HERE / "out" / "stage_b_development_2023_progress.jsonl"
PRIMARY_LOG_PATH = HERE / "out" / "stage_b_primary_2024_2025_progress.jsonl"
SENSITIVITY_LOG_PATH = HERE / "out" / "stage_b_sensitivity_results.jsonl"
QUARANTINE_DIR = HERE / "out" / "quarantine_20260920_old_schema"
# 2026-09-20夜、budget_amendment_20260920(spec.json v1.3.1)によりcorrected design
# (2023 development + 2024/2025 primary + sensitivity 600、削減なし)を完遂するため
# 20M→25Mへ引き上げ。quarantine済みstate_schema_v2実消費分を含む累積へ適用する。
MAX_INPUT_TOKENS_CAP = 25_000_000
LOG_EVERY = 100

# data_availability_audit(spec.json)により2023-2025で取得できないフィールド。
# unknown_category_rateも2026-09-20夜にここへ追加(0埋めは誤りだったため)。
_UNAVAILABLE_FIELDS = {
    "odds_t20": None, "odds_t10": None, "odds_change_rate": None, "popularity_rank_change": None,
    "candidate_ticket_count": None, "candidate_ticket_probs": None, "current_odds": None,
    "predicted_confirmed_odds": None, "ev_point_estimate": None, "ev_lower_bound": None,
    "min_payout": None, "same_horse_concentration_rate": None,
}
_AVAILABILITY_FLAGS = {"odds_trajectory": False, "ticket_candidates": False,
                       "unknown_category_rate": False}


def _load_race_id_map() -> dict:
    if RACE_ID_MAP_PATH.exists():
        return json.loads(RACE_ID_MAP_PATH.read_text(encoding="utf-8"))
    return {}


def _save_race_id_map(m: dict) -> None:
    """atomic checkpoint (2026-09-20夜、ユーザー指定「100件ごとにatomic checkpoint」):
    一時ファイルへ書いてからos.replaceで置き換える(書き込み途中でのプロセス中断でも
    既存ファイルを壊さない)。"""
    RACE_ID_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = RACE_ID_MAP_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, RACE_ID_MAP_PATH)


def _anon_id_for(rid16: str, race_id_map: dict, reverse_map: dict) -> str:
    if rid16 in reverse_map:
        return reverse_map[rid16]
    anon = f"HISTV3_{len(race_id_map):06d}"
    race_id_map[anon] = rid16
    reverse_map[rid16] = anon
    return anon


def build_jev_state(row, in_dist_support: float, similar_count: int) -> dict:
    """rid16そのものは含めない。結果ラベルも含めない。unknown_category_rateは
    stateから完全に省略し、availabilityでfalseと明示する(0埋めしない、
    2026-09-20夜の訂正、state_schema_v3)。"""
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
        # unknown_category_rate: 意図的に省略 (state_schema_v3)
        "in_distribution_support": float(in_dist_support),
        "similar_past_case_count": int(similar_count),
    }
    state.update(_UNAVAILABLE_FIELDS)
    state["availability"] = dict(_AVAILABILITY_FLAGS)
    return state


def _tokens_in_dir(cache_dir: Path) -> int:
    total = 0
    if not cache_dir.exists():
        return 0
    for p in cache_dir.glob("*.json"):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
            usage = rec.get("usage") or {}
            total += int(usage.get("input_tokens") or 0)
        except Exception:
            continue
    return total


def _quarantined_tokens() -> int:
    return _tokens_in_dir(QUARANTINE_DIR / "jev_cache")


def _cumulative_input_tokens_including_quarantine() -> int:
    """現行(state_schema_v3)キャッシュの消費量 + 隔離済み旧schemaの実消費量。
    2026-09-20ユーザー指定: 20M上限は両方を合算した「実消費量」に対して適用する。"""
    return _tokens_in_dir(JC.CACHE_DIR) + _quarantined_tokens()


def run_batch(state_df, support_result, race_id_map, reverse_map, log_path: Path, role: str) -> dict:
    n_done, n_skipped_cap, n_fail = 0, 0, 0
    t0 = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for i, (rid16, row) in enumerate(state_df.iterrows()):
        cum_tokens = _cumulative_input_tokens_including_quarantine()
        if cum_tokens >= MAX_INPUT_TOKENS_CAP:
            n_skipped_cap = len(state_df) - i
            msg = (f"[STOP] 累積input_tokens(旧schema隔離分+現行分)={cum_tokens}が上限"
                  f"{MAX_INPUT_TOKENS_CAP}へ到達、新規呼び出しを停止 (role={role})")
            print(msg)
            with open(HERE / "out" / "stage_b_errors.log", "a", encoding="utf-8") as f:
                f.write(msg + "\n")
            break
        anon_id = _anon_id_for(rid16, race_id_map, reverse_map)
        support = float(support_result.loc[rid16, "in_distribution_support"])
        count = int(support_result.loc[rid16, "similar_past_case_count"])
        state = build_jev_state(row, support, count)
        result = JC.query_jev(anon_id, PROMPT_SCHEMA_HASH, state, SPEC["questions"],
                              exp05_model_hash="exp05_oos_safe_m1m3m4",
                              market_snapshot_time="historical_tanpuk_pre")
        if not result.get("ok"):
            n_fail += 1
        else:
            n_done += 1
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"rid16": rid16, "anon_id": anon_id, "role": role,
                                "ok": result.get("ok"), "from_cache": result.get("from_cache"),
                                "input_hash": result.get("input_hash")}, ensure_ascii=False) + "\n")
        if (i + 1) % LOG_EVERY == 0:
            _save_race_id_map(race_id_map)
            print(f"[progress:{role}] {i+1}/{len(state_df)} done={n_done} fail={n_fail} "
                 f"cum_tokens(incl_quarantine)={cum_tokens} elapsed={time.time()-t0:.0f}s")
    _save_race_id_map(race_id_map)
    return {"role": role, "n_total": len(state_df), "n_done": n_done, "n_fail": n_fail,
           "n_skipped_cap": n_skipped_cap}


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
    ap.add_argument("--development-only", action="store_true")
    ap.add_argument("--primary-only", action="store_true")
    ap.add_argument("--sensitivity-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="テスト用: 各対象群のレース数を制限")
    args = ap.parse_args()

    print(f"[stage_b_run] state_schema={STATE_SCHEMA_VERSION} prompt_schema_hash={PROMPT_SCHEMA_HASH}")
    print("[stage_b_run] loading historical state...")
    df, feature_lists = HS.load_design_and_features()
    preds = HS.fit_oos_safe_predictions(df, feature_lists)
    state = HS.build_race_state_vectors(df, preds)

    fit_cols = CONTINUOUS_COLS + CATEGORICAL_COLS
    d2023 = state[state["year"] == 2023].copy()
    sm = SupportModel().fit(d2023[fit_cols])
    d2425 = state[state["year"].isin([2024, 2025])].copy()

    support_2023 = sm.score_reference_self()
    support_2425 = sm.score(d2425[fit_cols])

    race_id_map = _load_race_id_map()
    reverse_map = {v: k for k, v in race_id_map.items()}

    if args.limit:
        d2023 = d2023.iloc[:args.limit]
        d2425 = d2425.iloc[:args.limit]

    run_dev = not (args.primary_only or args.sensitivity_only)
    run_pri = not (args.development_only or args.sensitivity_only)
    run_sens = not (args.development_only or args.primary_only)

    if run_dev:
        print(f"[stage_b_run] development (2023): {len(d2023)} races, role=development_2023")
        dev_summary = run_batch(d2023, support_2023, race_id_map, reverse_map,
                                DEVELOPMENT_LOG_PATH, "development_2023")
        print("[stage_b_run] development summary:", json.dumps(dev_summary, ensure_ascii=False))

    if run_pri:
        print(f"[stage_b_run] primary evaluation (2024+2025): {len(d2425)} races, role=primary_2024_2025")
        primary_summary = run_batch(d2425, support_2425, race_id_map, reverse_map,
                                    PRIMARY_LOG_PATH, "primary_2024_2025")
        print("[stage_b_run] primary summary:", json.dumps(primary_summary, ensure_ascii=False))

    if run_sens:
        d2425["_support"] = support_2425["in_distribution_support"]
        d2425["_similar_count"] = support_2425["similar_past_case_count"]
        print("[stage_b_run] sensitivity analysis: 200 fixed races x 3 uncached repeats")
        sens_summary = run_sensitivity(d2425)
        print("[stage_b_run] sensitivity summary:", json.dumps(sens_summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
