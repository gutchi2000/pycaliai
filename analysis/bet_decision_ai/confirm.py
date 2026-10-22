# -*- coding: utf-8 -*-
"""
confirm.py — 確認期間 2023-2025 を 1 回だけ開けて仕様 v1.0 §7 で判定する
========================================================================
**このスクリプトは開発者が実行してはいけない。実行はユーザーの判断で 1 回だけ。**

安全装置
  - `--confirm-once` が無ければ何もしない
  - out/confirm_results.json（結果）または out/confirm_STARTED（開始印）が既にあれば拒否（再実行・やり直し不可）
  - 仕様 docs/research/BET_DECISION_AI_SPEC.md の sha256 が凍結版（5cc64e6adfbd8769…）か検査
  - models/manifest.json に記録した全ファイル（NN 10 本・GBM 5 本・標準化・config）の sha256 を検査。
    `--expect-manifest-sha` を渡すと manifest.json 自体の sha256 も照合（DEV_REPORT.md に記載の値）
  - 2023-25 の読み込みは data.py の唯一の入口に CONFIRM_TOKEN を渡したときだけ可能

判定（§7 の機械的実装、予算 B = 10,000 円。B = 2,000 円は記述のみ）
  各方策（5 本の平均方策 = 主、種 0-4 は個別）について
    購入レース < 300R                                  → FAIL
    PASS  : 回収率 CI 下限 > 100% かつ 上位 10 払戻除外の点推定 ≥ 100% かつ B0〜B4 全部との差の CI 下限 > 0
    SIGNAL: B0〜B4 全部との差の CI 下限 > 0 かつ 回収率 CI が 100% をまたぐ（下限 ≤ 100% ≤ 上限）
    FAIL  : それ以外
  最終判定: 平均方策の判定 V が種 5 本中 4 本以上と一致すれば V。
           一致しない場合、V=PASS かつ種 4 本以上が PASS か SIGNAL なら SIGNAL、それ以外は FAIL（保守側に倒す）。
  B4 は市場版判断 AI の 5 本の平均方策。差は開催日ブロック bootstrap（同じ抽出で対比較）。

実行（ユーザーのみ）: python -m analysis.bet_decision_ai.confirm --confirm-once [--expect-manifest-sha <sha>]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.bet_decision_ai import data as D

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models"
OUT_DIR = HERE / "out"
RESULT = OUT_DIR / "confirm_results.json"
STARTED = OUT_DIR / "confirm_STARTED"
COMPARE = ["B0_null", "B1_a95", "B2_market_fav", "B3_prob_first", "B4_market_AI"]
MIN_RACES = 300


def verdict(m: dict, diffs: dict) -> tuple[str, dict]:
    why = {"n_bet_races": m["n_bet_races"], "roi": m["roi"], "roi_ci": m["roi_ci"],
           "roi_ex_top10": m.get("roi_ex_top10"),
           "diff_ci_lo": {c: diffs[c]["roi_diff_ci"][0] for c in COMPARE}}
    if m["n_bet_races"] < MIN_RACES or m["roi"] is None:
        return "FAIL", why | {"reason": f"購入 {m['n_bet_races']}R < {MIN_RACES}R"}
    diffs_ok = all(np.isfinite(diffs[c]["roi_diff_ci"][0]) and diffs[c]["roi_diff_ci"][0] > 0 for c in COMPARE)
    lo, hi = m["roi_ci"]
    if lo > 1.0 and m["roi_ex_top10"] >= 1.0 and diffs_ok:
        return "PASS", why
    if diffs_ok and lo <= 1.0 <= hi:
        return "SIGNAL", why
    return "FAIL", why


def final_verdict(v_mean: str, v_seeds: list[str]) -> tuple[str, str]:
    n_same = sum(v == v_mean for v in v_seeds)
    if n_same >= 4:
        return v_mean, f"平均方策 {v_mean}、種 {n_same}/5 が一致"
    if v_mean == "PASS" and sum(v in ("PASS", "SIGNAL") for v in v_seeds) >= 4:
        return "SIGNAL", f"平均方策 PASS だが PASS 一致は種 {n_same}/5。種 4 本以上が PASS/SIGNAL なので SIGNAL に降格"
    return "FAIL", f"平均方策 {v_mean}、一致は種 {n_same}/5（4 本未満）→ FAIL"


def verify_models(expect_manifest_sha: str | None) -> dict:
    man_p = MODEL_DIR / "manifest.json"
    if not man_p.exists():
        raise SystemExit("models/manifest.json が無い")
    man_sha = D.sha256(man_p)
    if expect_manifest_sha and man_sha != expect_manifest_sha:
        raise SystemExit(f"manifest.json の sha256 不一致: {man_sha} != {expect_manifest_sha}")
    man = json.loads(man_p.read_text(encoding="utf-8"))
    for name, sha in man["files"].items():
        got = D.sha256(MODEL_DIR / name)
        if got != sha:
            raise SystemExit(f"モデルファイルの sha256 不一致: {name} {got} != {sha}")
    return {"manifest_sha256": man_sha, "n_files": len(man["files"])}


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-once", action="store_true", help="確認期間 2023-25 を 1 回だけ開ける（ユーザー判断）")
    ap.add_argument("--expect-manifest-sha", default=None)
    ap.add_argument("--dry-run-dev", action="store_true",
                    help="動作確認: 同じ処理を選択期間 2021-22（開発データ）で流す。確認期間は読まない・開始印も作らない")
    args = ap.parse_args()
    if args.dry_run_dev:
        spec_sha = D.check_spec()
        mv = verify_models(args.expect_manifest_sha)
        run(D.SELECT_YEARS, None, OUT_DIR / "confirm_dryrun_dev.json", spec_sha, mv)
        return
    if not args.confirm_once:
        print("確認期間は --confirm-once を付けたときだけ開ける（1 回限り）。何もしないで終了。")
        sys.exit(2)
    if RESULT.exists() or STARTED.exists():
        raise SystemExit(f"確認期間は既に開けられている（{RESULT.name} / {STARTED.name} が存在）。再実行は禁止（仕様 §7）。")
    spec_sha = D.check_spec()
    mv = verify_models(args.expect_manifest_sha)
    OUT_DIR.mkdir(exist_ok=True)
    STARTED.write_text(datetime.now().isoformat(timespec="seconds"), encoding="utf-8")   # 以降は再実行不可
    run(D.CONFIRM_YEARS, D.CONFIRM_TOKEN, RESULT, spec_sha, mv)


def run(years: tuple, tok: str | None, result_path: Path, spec_sha: str, mv: dict) -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="X does not have valid feature names")

    import lightgbm as lgb
    import torch

    from analysis.bet_decision_ai import baselines as BL
    from analysis.bet_decision_ai import evaluate as E
    from analysis.bet_decision_ai import policy as P

    lo, hi = years
    tag = "confirm" if tok else "dryrun_dev"
    cfg = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cand = D.load_candidates(lo, hi, _confirm_token=tok)
    races = D.load_races(2015, hi, _confirm_token=tok)
    base = D.load_base_market(2015, hi, _confirm_token=tok)
    mkt_ctx = D.market_race_context(base, races)
    std_m = D.Standardizer.from_dict(json.loads((MODEL_DIR / "standardizer_model.json").read_text(encoding="utf-8")))
    std_q = D.Standardizer.from_dict(json.loads((MODEL_DIR / "standardizer_market.json").read_text(encoding="utf-8")))
    fm = D.ticket_features(cand, "model")
    fq = D.ticket_features(cand, "market", mkt_ctx)
    assert list(fm.columns) == cfg["feature_cols_model"] and list(fq.columns) == cfg["feature_cols_market"]
    Tm = D.RaceTensor(cand, std_m.transform(fm))
    Tq = D.RaceTensor(cand, std_q.transform(fq))
    assert (Tm.races == Tq.races).all()

    def load_nn(key, tag):
        c = cfg[key]["cfg"]
        out = []
        for s in cfg[key]["seeds"]:
            m = P.PolicyNet(len(cfg["feature_cols_model"]), c["width"], c["depth"], c.get("abstain_bias", 6.0),
                            c.get("dropout", 0.0), c.get("stake_mode", "softmax_abstain")).to(device)
            m.load_state_dict(torch.load(MODEL_DIR / s["path"], map_location=device, weights_only=True))
            out.append(m)
        return out

    nn_m = load_nn("nn_model", "nn_model")
    nn_q = load_nn("nn_market_B4", "nn_market")
    sm = [P.predict_nn(m, Tm.X, Tm.mask, device) for m in nn_m]
    sq = [P.predict_nn(m, Tq.X, Tq.mask, device) for m in nn_q]
    sg = []
    for s in cfg["gbm_model"]["seeds"]:
        bst = lgb.Booster(model_file=str(MODEL_DIR / s["path"]))
        pred = np.full(Tm.mask.shape, -np.inf)
        pred[Tm.mask] = bst.predict(Tm.X[Tm.mask])
        sg.append(P.gbm_stakes(pred, Tm.mask, cfg["gbm_model"]["cfg"]["thr"]))
    fr = {"AI": np.mean(sm, 0), "B4_market_AI": np.mean(sq, 0), "GBM": np.mean(sg, 0)}
    for i, s in enumerate(sm):
        fr[f"AI_seed{i}"] = s

    waku_map = D.load_waku_map(lo, hi, _confirm_token=tok)
    tan_snap = D.load_tan_snapshot_full_field(lo, hi, _confirm_token=tok)
    wak = D.load_wakuren_for_b1(lo, hi, _confirm_token=tok)
    bcf = base[base["year"].between(lo, hi)]
    universe = pd.DataFrame({"rid16": Tm.races, "date": Tm.dates})
    edges = {k: v for k, v in cfg["quintile_edges_lpq_train"].items()}
    out = {"spec_sha256": spec_sha, "models": mv, "run_at": datetime.now().isoformat(timespec="seconds"),
           "period": [lo, hi], "n_races": int(len(universe)),
           "n_unsettleable_excluded": cand.attrs["n_unsettleable_races"], "budgets": {}}
    seeds = [f"AI_seed{i}" for i in range(len(sm))]
    for B in (10000, 2000):
        bb = {"B0_null": BL.b0_null(cand, B), "B3_prob_first": BL.b3_prob_first(cand, B)}
        bb["B1_a95"], i1 = BL.b1_a95(cand, bcf, wak, waku_map, B)
        bb["B2_market_fav"], i2 = BL.b2_market_fav(cand, bcf, tan_snap, B)
        res = E.evaluate_period(Tm, fr, bb, universe, B, edges, mains=["AI", "GBM"] + seeds,
                                compare_to=COMPARE + ["GBM"])
        res["books"]["AI"].to_parquet(OUT_DIR / f"{tag}_books_AI_B{B}.parquet", index=False)
        res.pop("books")
        res["baseline_info"] = {"B1": i1, "B2": i2}
        v_mean, why = verdict(res["policies"]["AI"], res["diff_vs"]["AI"])
        v_seeds = [verdict(res["policies"][s], res["diff_vs"][s])[0] for s in seeds]
        res["verdict_mean_policy"] = {"verdict": v_mean, "detail": why}
        res["verdict_seeds"] = v_seeds
        if B == 10000:
            fv, note = final_verdict(v_mean, v_seeds)
            out["FINAL_VERDICT"] = fv
            out["final_note"] = note
        out["budgets"][f"B{B}"] = res
    out["mode"] = tag
    result_path.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"[{tag}] FINAL_VERDICT (B=10,000): {out['FINAL_VERDICT']} / {out['final_note']}")
    print(f"結果: {result_path}")


if __name__ == "__main__":
    main()
