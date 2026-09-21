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
from pl_rank_distribution import (  # noqa: E402
    mc_rank_distribution, exact_rank_distribution, quantile_rank, resolve_q90_label,
)
from eligible_races import build_target_rows  # noqa: E402

MC_DRAWS = 50_000
MC_GLOBAL_SEED = 20261010

targets, race_horses, meta = build_target_rows([2023])
rng = np.random.default_rng(999)

max_abs_diff_overall = 0.0
max_q90_diff_fixed_k = 0
n_checked = 0
n_final_method_matches = 0
n_ladder_forced_matches = 0
print("n   race_id            max|MC-DP|   q90_MC  q90_DP  diff  final_method  ladder_forced_q90")
sample_rows = []
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

        # (a) 固定K=50,000のMC vs 厳密DP (旧検証、参考として残す)
        mc = mc_rank_distribution(scores, focal_pos, MC_DRAWS, MC_GLOBAL_SEED, str(rid))
        dp = exact_rank_distribution(scores, focal_pos)
        diff = np.max(np.abs(mc - dp))
        q90_mc_fixed = quantile_rank(mc, 0.90)
        q90_dp = quantile_rank(dp, 0.90)
        max_abs_diff_overall = max(max_abs_diff_overall, diff)
        max_q90_diff_fixed_k = max(max_q90_diff_fixed_k, abs(q90_mc_fixed - q90_dp))

        # (b) 本番方式(resolve_q90_label, n<=18なので常にexact_dp) vs 厳密DP直接計算
        final = resolve_q90_label(scores, focal_pos, str(rid))
        final_match = (final["q90_rank"] == q90_dp)
        n_final_method_matches += int(final_match)

        # (c) MC梯子を強制発火(exact_dp_max_n=0)させても厳密DPと一致するか
        #     (本番ではn<=18なので発火しないが、梯子の正しさを実データでも確認する)
        ladder_forced = resolve_q90_label(scores, focal_pos, str(rid), exact_dp_max_n=0)
        ladder_match = (ladder_forced["q90_rank"] == q90_dp)
        n_ladder_forced_matches += int(ladder_match)

        n_checked += 1
        print(f"{n_field:<3d} {rid:<18s} {diff:.5f}      {q90_mc_fixed:<7d} {q90_dp:<7d} "
              f"{q90_mc_fixed-q90_dp:<5d} {final['method']:<14s} "
              f"{ladder_forced['q90_rank']}({ladder_forced['method']})")

print(f"\n[summary/固定K=50,000のMC] n_checked={n_checked}  "
      f"max|MC-DP| overall={max_abs_diff_overall:.5f}  "
      f"max|q90_MC-q90_DP|={max_q90_diff_fixed_k}")
print(f"[summary/本番方式resolve_q90_label] {n_final_method_matches}/{n_checked} が厳密DPと一致 "
      f"(n<=18は常にexact_dpを使うため理論上{n_checked}/{n_checked}必達)")
print(f"[summary/MC梯子を強制発火させた場合] {n_ladder_forced_matches}/{n_checked} が厳密DPと一致 "
      f"(adaptive Wilson CI機構の実データでの正しさの確認)")
