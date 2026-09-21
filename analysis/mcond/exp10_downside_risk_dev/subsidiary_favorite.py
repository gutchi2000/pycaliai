# -*- coding: utf-8 -*-
"""
subsidiary_favorite.py
=======================
EXP10 Stage 1副次診断。「1番人気のcatastrophic downside」を計算する。

1番人気の同定: 外部kekkaファイル(E:\\競馬過去走データ\\raw_data\\
kekka_2010_2025_fix_raceid_v2__keyed.csv)の単勝オッズが最小の馬(オッズ同値の場合は
馬番昇順、JRA公表の人気表示と同じ慣行)。この馬についても、AI(v6)のraw scoreから
導かれる同じPL順位分布を使ってq90_predicted_rank/catastrophic_downsideを計算する
(=「市場の一番人気が、AI自身の予測分布からみても異常に悪い着順だったか」)。

主判定を救済する目的ではなく、副次診断としてのみ扱う(spec.json §3)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pl_rank_distribution import resolve_q90_label  # noqa: E402
from eligible_races import build_target_rows  # noqa: E402
from labels import _has_missing_finish_number, _has_duplicate_row  # noqa: E402

KEKKA_EXT = Path(r"E:\競馬過去走データ\raw_data\kekka_2010_2025_fix_raceid_v2__keyed.csv")
MC_GLOBAL_SEED = 20261011  # labels.pyとは独立(favorite用と明示的に分離)
Q90 = 0.90


def load_favorites(year: int) -> pd.DataFrame:
    df = pd.read_csv(KEKKA_EXT, encoding="utf-8-sig",
                      usecols=["race_id", "umaban", "単勝オッズ"], low_memory=False)
    df["race_id"] = df["race_id"].astype(str)
    df = df[df["race_id"].str.startswith(str(year))].copy()
    df = df.dropna(subset=["単勝オッズ"])
    df = df.sort_values(["race_id", "単勝オッズ", "umaban"])
    fav = df.groupby("race_id", as_index=False).first()[["race_id", "umaban", "単勝オッズ"]]
    fav = fav.rename(columns={"umaban": "fav_ban", "単勝オッズ": "fav_odds"})
    return fav


def compute_favorite_labels(year: int) -> tuple[pd.DataFrame, dict]:
    targets, race_horses, _meta = build_target_rows([year])
    fav = load_favorites(year)

    rows = []
    n_no_odds = 0
    n_dead_heat_excluded = 0
    n_missing_finish_number_excluded = 0
    n_duplicate_row_excluded = 0
    n_unresolved = 0
    method_counts: dict[str, int] = {}
    for rid, g in race_horses.groupby("rid", sort=False):
        g = g.sort_values("pos_in_group").reset_index(drop=True)
        n = len(g)
        finish = g["finish"].values.astype(float)
        ban_all = g["ban"].values
        vals = finish[~np.isnan(finish)]
        if len(vals) and (np.unique(vals, return_counts=True)[1] > 1).any():
            n_dead_heat_excluded += 1
            continue
        if _has_missing_finish_number(finish):
            n_missing_finish_number_excluded += 1
            continue
        if _has_duplicate_row(ban_all):
            n_duplicate_row_excluded += 1
            continue
        f = fav[fav["race_id"] == rid]
        if len(f) == 0:
            n_no_odds += 1
            continue
        fav_ban = int(f["fav_ban"].iloc[0])
        match = g[g["ban"] == fav_ban]
        if len(match) == 0:
            n_no_odds += 1
            continue
        fav_pos = int(match.index[0])
        if np.isnan(finish[fav_pos]):
            n_no_odds += 1
            continue

        scores = g["score"].values.astype(float)
        result = resolve_q90_label(scores, fav_pos, str(rid), q=Q90, global_seed=MC_GLOBAL_SEED)
        method_counts[result["method"]] = method_counts.get(result["method"], 0) + 1
        if not result["resolved"]:
            n_unresolved += 1
            continue
        q90_rank = result["q90_rank"]
        observed_rank = int(finish[fav_pos])
        catastrophic = int(observed_rank > q90_rank)
        is_also_ai_focal = bool(g.loc[fav_pos, "is_focal"])

        rows.append(dict(
            rid=str(rid), n=n, fav_ban=fav_ban, observed_rank=observed_rank,
            q90_predicted_rank=q90_rank, catastrophic_downside=catastrophic,
            fav_is_ai_focal=is_also_ai_focal,
        ))

    out = pd.DataFrame(rows)
    diag = dict(
        n_races_with_favorite_label=len(out),
        n_no_odds_or_unmatched=n_no_odds,
        n_dead_heat_excluded=n_dead_heat_excluded,
        n_missing_finish_number_excluded=n_missing_finish_number_excluded,
        n_duplicate_row_excluded=n_duplicate_row_excluded,
        n_q90_unresolved=n_unresolved,
        q90_method_counts=method_counts,
        favorite_is_ai_focal_rate=float(out["fav_is_ai_focal"].mean()) if len(out) else float("nan"),
        catastrophic_positive_count=int(out["catastrophic_downside"].sum()) if len(out) else 0,
        catastrophic_positive_rate=float(out["catastrophic_downside"].mean()) if len(out) else float("nan"),
    )
    return out, diag


if __name__ == "__main__":
    out, diag = compute_favorite_labels(2023)
    print("[2023 development, 1番人気catastrophic_downside 副次診断・構造的件数のみ]")
    for k, v in diag.items():
        print(f"  {k}: {v}")
    out.to_parquet(Path(__file__).parent / "out" / "favorite_labels_2023.parquet")
    print("[saved] out/favorite_labels_2023.parquet")
