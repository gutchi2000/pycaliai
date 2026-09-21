# -*- coding: utf-8 -*-
"""
labels.py
=========
EXP10 Stage 1. 主目的変数 catastrophic_downside と副次診断ラベルの計算。

主ラベル(ユーザー指定、spec.jsonに凍結):
    q90_predicted_rank = quantile_rank(predicted_rank_dist, 0.90)
    catastrophic_downside = 1[ observed_rank > q90_predicted_rank ]

predicted_rank_dist は対象馬(◎、他の任意の focal horseにも汎用)の raw v6 score から
Plackett-Luceで導かれるレース内順位の周辺分布（pl_rank_distribution.resolve_q90_label、
n<=18は常に厳密bitmask DP、n>18はWilson信頼区間で確定するadaptiveモンテカルロ梯子。
モンテカルロ誤差だけで教師ラベルが決まることはない）。

主解析からの除外カテゴリ(dead_heat_audit.pyで内訳を分解済み):
  - 任意順位の同着(PLは全順序前提のため)
  - 着順番号に欠番があるレース(データ整合性異常、2023年に1件発見)
  - 同一race_id内の重複行(2023年は0件だが安全のため検査を残す)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pl_rank_distribution import (  # noqa: E402
    resolve_q90_label, expected_rank, rank_percentile,
)
from eligible_races import build_target_rows  # noqa: E402

Q90 = 0.90


def _has_dead_heat(finish_vals: np.ndarray) -> bool:
    """数値着順に重複があれば同着とみなす(NaNは除く)。"""
    vals = finish_vals[~np.isnan(finish_vals)]
    if len(vals) == 0:
        return False
    u, c = np.unique(vals, return_counts=True)
    return bool((c > 1).any())


def _has_missing_finish_number(finish_vals: np.ndarray) -> bool:
    """観測されたユニーク着順が1..len(unique)の連番になっていなければ欠番とみなす。
    2023年で1件(rid=2023102204040405)発見済み。フィールドの一部が構造上不明の
    第三者(TARGET側の行欠落等)によって占められている可能性があり、n(頭数)と
    観測着順の対応が信頼できないため除外する。"""
    vals = finish_vals[~np.isnan(finish_vals)]
    if len(vals) == 0:
        return False
    u = np.unique(vals)
    expected = set(range(1, len(u) + 1))
    observed = set(int(x) for x in u)
    return observed != expected


def _has_duplicate_row(ban_vals: np.ndarray) -> bool:
    return bool(pd.Series(ban_vals).duplicated().any())


def compute_labels(years: list[int]) -> tuple[pd.DataFrame, dict]:
    targets, race_horses, meta = build_target_rows(years)

    rows = []
    n_dead_heat_excluded = 0
    n_missing_finish_number_excluded = 0
    n_duplicate_row_excluded = 0
    n_focal_finish_nan = 0
    n_unresolved = 0
    method_counts: dict[str, int] = {}

    for rid, g in race_horses.groupby("rid", sort=False):
        g = g.sort_values("pos_in_group").reset_index(drop=True)
        n = len(g)
        finish = g["finish"].values.astype(float)
        ban = g["ban"].values

        if _has_dead_heat(finish):
            n_dead_heat_excluded += 1
            continue
        if _has_missing_finish_number(finish):
            n_missing_finish_number_excluded += 1
            continue
        if _has_duplicate_row(ban):
            n_duplicate_row_excluded += 1
            continue

        focal_row = g[g["is_focal"]]
        if len(focal_row) != 1:
            raise AssertionError(f"rid={rid}: is_focalが1でない({len(focal_row)})")
        focal_pos = int(focal_row.index[0])
        if np.isnan(finish[focal_pos]):
            n_focal_finish_nan += 1
            continue

        scores = g["score"].values.astype(float)
        result = resolve_q90_label(scores, focal_pos, str(rid), q=Q90)
        method_counts[result["method"]] = method_counts.get(result["method"], 0) + 1
        if not result["resolved"]:
            n_unresolved += 1
            continue

        q90_rank = result["q90_rank"]
        dist = result["dist"]
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
            q90_method=result["method"],
            expected_rank=exp_rank, observed_rank_pct=obs_pct, expected_rank_pct=exp_pct,
            downside_residual=residual, catastrophic_downside=catastrophic,
            top3_failure=top3_failure, bottom_half=bottom_half,
            focal_win_prob=focal_win_prob,
        ))

    labels = pd.DataFrame(rows)
    diag = dict(
        **meta,
        n_dead_heat_excluded=n_dead_heat_excluded,
        n_missing_finish_number_excluded=n_missing_finish_number_excluded,
        n_duplicate_row_excluded=n_duplicate_row_excluded,
        n_focal_finish_nan=n_focal_finish_nan,
        n_q90_unresolved=n_unresolved,
        q90_method_counts=method_counts,
        n_labeled=len(labels),
    )
    return labels, diag


if __name__ == "__main__":
    labels, diag = compute_labels([2023])
    print("[2023 development, catastrophic_downside 主ラベル・構造的件数のみ]")
    for k, v in diag.items():
        print(f"  {k}: {v}")
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
