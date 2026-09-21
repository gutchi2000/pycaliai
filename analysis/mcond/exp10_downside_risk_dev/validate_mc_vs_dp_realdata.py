# -*- coding: utf-8 -*-
"""
validate_mc_vs_dp_realdata.py
==============================
EXP10 Stage 1。合成oracle(test_pl_rank_distribution.py)に加え、実際の2023レース
(スコア分布が合成テストより歪んでいる可能性がある)についても、モンテカルロ推定と
厳密bitmask DPを層化サンプルで突合する。頭数n=5..18の層から各3レースを抽出。
2024・2025年のデータは使わない。性能・ROI評価ではなく、エンジンの数値的正しさの
検証のみ。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pl_rank_distribution import mc_rank_distribution, exact_rank_distribution, quantile_rank  # noqa: E402
from eligible_races import build_target_rows  # noqa: E402

MC_DRAWS = 50_000
MC_GLOBAL_SEED = 20261010

targets, race_horses, meta = build_target_rows([2023])
rng = np.random.default_rng(999)

max_abs_diff_overall = 0.0
max_q90_diff = 0
n_checked = 0
print("n   race_id            max|MC-DP|   q90_MC  q90_DP  diff")
for n_field, sub in race_horses.groupby("n"):
    rids = sub["rid"].unique()
    sample = rng.choice(rids, size=min(3, len(rids)), replace=False)
    for rid in sample:
        g = race_horses[race_horses["rid"] == rid].sort_values("pos_in_group").reset_index(drop=True)
        finish = g["finish"].values.astype(float)
        vals = finish[~np.isnan(finish)]
        if len(vals) and (np.unique(vals, return_counts=True)[1] > 1).any():
            continue  # 同着は主解析同様スキップ
        focal_pos = int(g[g["is_focal"]].index[0])
        scores = g["score"].values.astype(float)
        mc = mc_rank_distribution(scores, focal_pos, MC_DRAWS, MC_GLOBAL_SEED, str(rid))
        dp = exact_rank_distribution(scores, focal_pos)
        diff = np.max(np.abs(mc - dp))
        q90_mc = quantile_rank(mc, 0.90)
        q90_dp = quantile_rank(dp, 0.90)
        max_abs_diff_overall = max(max_abs_diff_overall, diff)
        max_q90_diff = max(max_q90_diff, abs(q90_mc - q90_dp))
        n_checked += 1
        print(f"{n_field:<3d} {rid:<18s} {diff:.5f}      {q90_mc:<7d} {q90_dp:<7d} {q90_mc-q90_dp}")

print(f"\n[summary] n_checked={n_checked}  max|MC-DP| overall={max_abs_diff_overall:.5f}  "
      f"max|q90_MC-q90_DP|={max_q90_diff}")
