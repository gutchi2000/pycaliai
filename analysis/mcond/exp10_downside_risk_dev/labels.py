# -*- coding: utf-8 -*-
"""
labels.py
=========
EXP10 Stage 1. 主目的変数 catastrophic_downside と副次診断ラベルの計算。

主ラベル(ユーザー指定、spec.jsonに凍結):
    q90_predicted_rank = quantile_rank(predicted_rank_dist, 0.90)
    catastrophic_downside = 1[ observed_rank > q90_predicted_rank ]

predicted_rank_dist は対象馬(◎、他の任意の focal horseにも汎用)の raw v6 score から
Plackett-Luceで導かれるレース内順位の周辺分布（pl_rank_distribution.py、本番は
Gumbel-max モンテカルロ、seed/draw数固定）。

同着(dead heat)はPLの前提(全順序の確率分布)外なので主解析から除外し件数を記録する。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pl_rank_distribution import (  # noqa: E402
    mc_rank_distribution, quantile_rank, expected_rank, rank_percentile,
)
from eligible_races import build_target_rows  # noqa: E402

MC_DRAWS = 50_000
MC_GLOBAL_SEED = 20261010
Q90 = 0.90


def _has_dead_heat(finish_vals: np.ndarray) -> bool:
    """数値着順に重複があれば同着とみなす(NaNは除く)。"""
    vals = finish_vals[~np.isnan(finish_vals)]
    if len(vals) == 0:
        return False
    u, c = np.unique(vals, return_counts=True)
    return bool((c > 1).any())


def compute_labels(years: list[int]) -> tuple[pd.DataFrame, dict]:
    targets, race_horses, meta = build_target_rows(years)

    rows = []
    n_dead_heat_excluded = 0
    n_focal_finish_nan = 0

    for rid, g in race_horses.groupby("rid", sort=False):
        g = g.sort_values("pos_in_group").reset_index(drop=True)
        n = len(g)
        finish = g["finish"].values.astype(float)
        if _has_dead_heat(finish):
            n_dead_heat_excluded += 1
            continue
        focal_row = g[g["is_focal"]]
        if len(focal_row) != 1:
            raise AssertionError(f"rid={rid}: is_focalが1でない({len(focal_row)})")
        focal_pos = int(focal_row.index[0])
        if np.isnan(finish[focal_pos]):
            n_focal_finish_nan += 1
            continue

        scores = g["score"].values.astype(float)
        dist = mc_rank_distribution(scores, focal_pos, MC_DRAWS, MC_GLOBAL_SEED, str(rid))
        q90_rank = quantile_rank(dist, Q90)
        exp_rank = expected_rank(dist)
        observed_rank = int(finish[focal_pos])

        catastrophic = int(observed_rank > q90_rank)
        obs_pct = rank_percentile(observed_rank, n)
        exp_pct = rank_percentile(exp_rank, n)
        residual = obs_pct - exp_pct if n > 1 else np.nan

        # 副次診断
        top3_failure = int(observed_rank > 3)
        bottom_half = int(obs_pct > 0.5) if n > 1 else np.nan
        p1 = np.exp(scores - scores.max())
        p1 = p1 / p1.sum()
        focal_win_prob = float(p1[focal_pos])

        rows.append(dict(
            rid=str(rid), year=int(g["year"].iloc[0]), n=n,
            focal_ban=int(g.loc[focal_pos, "ban"]),
            observed_rank=observed_rank, q90_predicted_rank=q90_rank,
            expected_rank=exp_rank, observed_rank_pct=obs_pct, expected_rank_pct=exp_pct,
            downside_residual=residual, catastrophic_downside=catastrophic,
            top3_failure=top3_failure, bottom_half=bottom_half,
            focal_win_prob=focal_win_prob,
        ))

    labels = pd.DataFrame(rows)
    diag = dict(
        **meta,
        n_dead_heat_excluded=n_dead_heat_excluded,
        n_focal_finish_nan=n_focal_finish_nan,
        n_labeled=len(labels),
    )
    return labels, diag


if __name__ == "__main__":
    labels, diag = compute_labels([2023])
    print("[2023 development, catastrophic_downside 主ラベル・構造的件数のみ]")
    for k, v in diag.items():
        print(f"  {k}: {v:,}")
    n_pos = int(labels["catastrophic_downside"].sum())
    n_tot = len(labels)
    print(f"  catastrophic_downside 陽性数: {n_pos:,} / {n_tot:,} "
          f"({n_pos/n_tot*100:.2f}%)")
    print(f"  top3_failure 陽性率: {labels['top3_failure'].mean()*100:.2f}%")
    print(f"  bottom_half 陽性率: {labels['bottom_half'].mean()*100:.2f}%")
    print("\n  q90_predicted_rank 分布:")
    print(labels["q90_predicted_rank"].value_counts().sort_index().to_string())
    print("\n  downside_residual 記述統計:")
    print(labels["downside_residual"].describe().to_string())
    labels.to_parquet(Path(__file__).parent / "out" / "labels_2023.parquet")
    print("\n[saved] out/labels_2023.parquet")
