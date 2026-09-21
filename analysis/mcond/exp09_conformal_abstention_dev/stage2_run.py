# -*- coding: utf-8 -*-
"""
stage2_run.py — EXP09 Stage2実装順(spec.json implementation_order)の
ドライバ。ステップ1-9を厳密にこの順序で実行する:

  1. eligible race集合の生成
  2. 2023 calibration score生成
  3. q_hat固定
  4. 2023を再利用せず2024・2025 prediction set生成
  5. 全対象上のempirical coverage確認(Gate1)
  6. APS-derived abstention score生成
  7. 同一participation_rate比較(7方式)
  8. Gate1〜4
  9. 通過した場合だけGate5 ROI(本ドライバはGate5を実装しない、Gate1-4の
     結果を見てから別途判断する)

実行: PYTHONUTF8=1 ./venv311/Scripts/python.exe -m analysis.mcond.exp09_conformal_abstention_dev.stage2_run
出力: analysis/mcond/exp09_conformal_abstention_dev/out/STAGE2_GATE_REPORT.json
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
HERE = Path(__file__).resolve().parent

from analysis.mcond.exp09_conformal_abstention_dev.eligible_races import (  # noqa: E402
    build_eligible_races,
)
from analysis.mcond.exp09_conformal_abstention_dev.aps import (  # noqa: E402
    build_calibration_scores, compute_q_hat, build_prediction_sets, empirical_coverage,
    NOMINAL_COVERAGE, ALPHA,
)
from analysis.mcond.exp09_conformal_abstention_dev.comparators import (  # noqa: E402
    score_max_probability, score_entropy, fit_ood_model, score_ood_support,
    score_feature_missing, fit_lr_control, score_lr_control, score_current_gate,
    _race_level_stats,
)
from analysis.mcond.exp09_conformal_abstention_dev.evaluate import (  # noqa: E402
    race_level_metrics, gate2_same_participation_rate, gate3_full_control,
)

PARTICIPATION_RATE_POINTS = [0.90, 0.75, 0.50, 0.25]
PRIMARY_PARTICIPATION_RATE = 0.75


def main():
    t0 = time.time()
    report = {"steps": {}}

    # --- 1. eligible race集合の生成 ---
    print("[stage2] 1. eligible race集合の生成...")
    elig = build_eligible_races([2023, 2024, 2025])
    frames = elig["race_frames"]
    frames_2023 = {r: f for r, f in frames.items() if r.startswith("2023")}
    frames_2024 = {r: f for r, f in frames.items() if r.startswith("2024")}
    frames_2025 = {r: f for r, f in frames.items() if r.startswith("2025")}
    frames_2425 = {**frames_2024, **frames_2025}
    report["steps"]["1_eligible_races"] = {
        "n_2023": len(frames_2023), "n_2024": len(frames_2024), "n_2025": len(frames_2025),
        "exclusion_counts": elig["exclusion_counts"],
    }
    print(f"  2023={len(frames_2023)} 2024={len(frames_2024)} 2025={len(frames_2025)}")

    # --- 2. 2023 calibration score生成 ---
    print("[stage2] 2. 2023 calibration score生成...")
    calib_scores_df = build_calibration_scores(frames_2023)
    report["steps"]["2_calibration_scores"] = {"n": len(calib_scores_df)}

    # --- 3. q_hat固定 ---
    print("[stage2] 3. q_hat固定...")
    q_res = compute_q_hat(calib_scores_df["nonconformity_score"].to_numpy(), alpha=ALPHA)
    report["steps"]["3_q_hat"] = {
        "q_hat": q_res.q_hat, "n": q_res.n, "k": q_res.k, "fail_closed": q_res.fail_closed,
    }
    print(f"  q_hat={q_res.q_hat:.6f} n={q_res.n} k={q_res.k} fail_closed={q_res.fail_closed}")
    if q_res.fail_closed:
        print("[stage2] Gate0 WARNING: q_hat fail-closed (k>n)")

    # --- 4. 2023を再利用せず2024・2025 prediction set生成 ---
    print("[stage2] 4. 2024・2025 prediction set生成(2023データは再利用しない、q_hatのみ)...")
    pred_2024 = build_prediction_sets(frames_2024, q_res.q_hat)
    pred_2025 = build_prediction_sets(frames_2025, q_res.q_hat)
    pred_all = pd.concat([pred_2024, pred_2025], ignore_index=True)

    # --- 5. 全対象上のempirical coverage確認(Gate1) ---
    print("[stage2] 5. empirical coverage確認(Gate1、all_eligible_racesのみ)...")
    cov_2024_all = empirical_coverage(pred_2024)
    cov_2025_all = empirical_coverage(pred_2025)
    gate1 = {
        "nominal_conformal_coverage": NOMINAL_COVERAGE,
        "empirical_coverage_all_eligible_races": {"2024": cov_2024_all, "2025": cov_2025_all},
        "verdict": "PASS" if (cov_2024_all >= NOMINAL_COVERAGE and cov_2025_all >= NOMINAL_COVERAGE)
                   else "FAIL",
    }
    report["gate1"] = gate1
    print(f"  2024 coverage={cov_2024_all:.4f}  2025 coverage={cov_2025_all:.4f}  "
         f"nominal={NOMINAL_COVERAGE}  verdict={gate1['verdict']}")

    # --- 6. APS-derived abstention score生成(すでにpred_*に格納済み、conformalの
    #     参加スコアとして抽出) ---
    print("[stage2] 6. APS-derived abstention score(既にstep4で計算済み、抽出)...")
    conformal_score_2425 = pred_all[["rid16", "abstention_score"]].copy()
    conformal_score_2425["score"] = conformal_score_2425["abstention_score"].apply(
        lambda t: (t[0], t[1], t[2]))
    # ソート用にタプルのまま比較可能な複合キーへ(pandasの安定ソートで辞書式比較させる)
    conformal_score_2425["_k0"] = conformal_score_2425["score"].apply(lambda t: t[0])
    conformal_score_2425["_k1"] = conformal_score_2425["score"].apply(lambda t: t[1])
    conformal_score_2425["_k2"] = conformal_score_2425["score"].apply(lambda t: t[2])

    # 診断値: 参加/見送り部分集合のcoverage(nominal保証の対象外、primary participation_rateのみ)
    n_select_primary = max(1, round(PRIMARY_PARTICIPATION_RATE * len(pred_all)))
    ordered = conformal_score_2425.sort_values(["_k0", "_k1", "_k2"], kind="mergesort")
    participating_rid = set(ordered["rid16"].iloc[:n_select_primary])
    abstained_rid = set(ordered["rid16"]) - participating_rid
    cov_participating = empirical_coverage(pred_all, participating_rid)
    cov_abstained = empirical_coverage(pred_all, abstained_rid)
    report["steps"]["6_coverage_diagnostics_primary_rate"] = {
        "empirical_coverage_participating_races": cov_participating,
        "empirical_coverage_abstained_races": cov_abstained,
        "note": "診断値のみ、Gate1の保証判定には使わない",
    }
    print(f"  [診断] participation_rate={PRIMARY_PARTICIPATION_RATE}時点: "
         f"participating coverage={cov_participating:.4f}  abstained coverage={cov_abstained:.4f}")

    # --- 7. 同一participation_rate比較(7方式) ---
    print("[stage2] 7. 同一participation_rate比較(7方式構築)...")
    metrics_2425 = race_level_metrics(frames_2425)

    max_prob_score = score_max_probability(frames_2425)
    entropy_score = score_entropy(frames_2425)
    ood_model = fit_ood_model(frames_2023)
    ood_score = score_ood_support(ood_model, frames_2425)
    feature_missing_score = score_feature_missing([2024, 2025])
    lr_clf, lr_scaler = fit_lr_control(frames_2023)
    lr_score = score_lr_control(lr_clf, lr_scaler, frames_2425)
    current_gate_score = score_current_gate(frames_2425)

    conformal_for_selection = conformal_score_2425[["rid16", "_k0", "_k1", "_k2"]].copy()
    conformal_for_selection = conformal_for_selection.sort_values(
        ["_k0", "_k1", "_k2"], kind="mergesort").reset_index(drop=True)
    conformal_for_selection["score"] = np.arange(len(conformal_for_selection), dtype=float)

    method_scores = {
        "conformal": conformal_for_selection[["rid16", "score"]],
        "max_probability": max_prob_score,
        "entropy": entropy_score,
        "ood_support": ood_score,
        "feature_missing": feature_missing_score,
        "lr_control_2023_only": lr_score,
        "current_gate": current_gate_score,
    }
    # 全方式が同じeligible race集合(rid16)をカバーしているか確認
    # (2026-09-21実装中に発覚: score_feature_missingはmaster_v2から直接読むため、
    # eligible_races.pyの例外処理で除外されたレース(同着等)も含んでしまい、
    # eligible_setの**superset**になっていた。unionでfillするだけでは不十分で、
    # 先にeligible_setへ**intersect**しないと、後段のselect_top_nがeligible_set外の
    # rid16を選んでしまい、metrics(eligible_setのみ)への.locでKeyErrorになる
    # 実害バグだった。)
    eligible_set = set(frames_2425.keys())
    for name, df in method_scores.items():
        df = df[df["rid16"].isin(eligible_set)]  # eligible_set外を除外(superset対策)
        missing = eligible_set - set(df["rid16"])
        if missing:
            print(f"  WARNING: {name} missing {len(missing)} eligible races, filling with worst score")
            fill = pd.DataFrame({"rid16": list(missing), "score": [df["score"].max() + 1] * len(missing)})
            df = pd.concat([df, fill], ignore_index=True)
        assert set(df["rid16"]) == eligible_set, f"{name}: eligible_setと不一致"
        method_scores[name] = df

    gate2_by_rate = {}
    for rate in PARTICIPATION_RATE_POINTS:
        gate2_by_rate[rate] = gate2_same_participation_rate(
            metrics_2425, method_scores, rate, len(eligible_set))
    report["gate2_by_participation_rate"] = gate2_by_rate
    primary_gate2 = gate2_by_rate[PRIMARY_PARTICIPATION_RATE]
    print(f"  [primary participation_rate={PRIMARY_PARTICIPATION_RATE}] "
         f"conformal mean logloss={primary_gate2['conformal_mean_logloss']:.4f}")
    for name, v in primary_gate2["vs"].items():
        print(f"    vs {name}: conformal - other = {v['conformal_minus_other_logloss']:+.6f}")

    # --- 8. Gate3(full-control)・Gate4(安定性、簡易版) ---
    print("[stage2] 8. Gate3(full-control)...")
    stats_2425 = _race_level_stats(frames_2425)
    controls = stats_2425[["rid16", "n_field", "max_prob", "entropy"]].copy()
    gate3 = gate3_full_control(metrics_2425, conformal_for_selection[["rid16", "score"]], controls)
    report["gate3_full_control"] = gate3
    print(f"  conformal coefficient={gate3['conformal_coefficient']:.4f} "
         f"CI95={gate3['ci95']} survives={gate3['survives_full_control']}")

    print("[stage2] 8b. Gate4(年度別安定性、primary participation_rate)...")
    gate4 = {}
    for year, frames_y in [("2024", frames_2024), ("2025", frames_2025)]:
        elig_y = set(frames_y.keys())
        n_sel_y = max(1, round(PRIMARY_PARTICIPATION_RATE * len(elig_y)))
        conf_y = conformal_for_selection[conformal_for_selection["rid16"].isin(elig_y)]
        sel_y = set(conf_y.sort_values("score").iloc[:n_sel_y]["rid16"])
        m_y = metrics_2425[metrics_2425["rid16"].isin(sel_y)]
        gate4[year] = {"n": len(sel_y), "mean_logloss": float(m_y["logloss"].mean())}
    report["gate4_year_stability"] = gate4
    print(f"  2024 conformal mean logloss={gate4['2024']['mean_logloss']:.4f}  "
         f"2025={gate4['2025']['mean_logloss']:.4f}")

    report["elapsed_seconds"] = time.time() - t0
    out_dir = HERE / "out"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "STAGE2_GATE_REPORT.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str),
                        encoding="utf-8")
    print(f"[stage2] wrote {out_path}")
    print(f"[stage2] total elapsed: {report['elapsed_seconds']:.0f}s")


if __name__ == "__main__":
    main()
