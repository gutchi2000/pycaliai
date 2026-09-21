# -*- coding: utf-8 -*-
"""
stage1_5_features.py
=====================
EXP10 Stage 1.5. B0〜B3・R1(最小構成)の特徴量をレース単位(=catastrophic_downside
ラベルと同じ対象馬1頭/レース)で構築する。全て予測時点で計算可能な情報のみ使用
(今回のレースの結果は一切使わない、DATA_AUDIT.md §2 / STAGE1_DESIGN.md §4参照)。

R1は「時点安全監査で"既存列で対応可"と判定された特徴のみ」を使う最小構成
(spec.jsonの10候補中: 過去着順分散・振幅・距離/馬場/競馬場変更)。Stage2で
"要構築"の残り5候補(タイム残差下方分位・休養明け成績分散・馬体重履歴変動・
出遅れ頻度・騎手調教師変更)を追加した完全版R1を評価する。Stage1.5はこの
最小構成でさえB3に対する限界的寄与がゼロなら、フル構築のコストをかける前に
立ち止まるための安価な早期ゲートとして機能する。

B1の市場特徴: 外部kekkaファイルの単勝オッズ(発走前最終)由来。
  market_implied_prob = 1/odds (正規化なし、単純な逆数)
  popularity_rank = レース内でオッズ昇順の順位(1=一番人気)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eligible_races import build_target_rows, score_population  # noqa: E402
from labels import compute_labels  # noqa: E402
from pl_rank_distribution import exact_rank_distribution  # noqa: E402
from backtest_pl_ev import COL_RID, COL_BAN  # noqa: E402

KEKKA_EXT = Path(r"E:\競馬過去走データ\raw_data\kekka_2010_2025_fix_raceid_v2__keyed.csv")

B3_COLS = [
    "tail_prob_gt3",       # B0: v6自身の下方尾部確率 P(rank>3)
    "market_implied_prob", "popularity_rank",             # B1
    "n", "max_prob", "entropy", "gap_12",                  # B2
    "kako5_race_count", "interval_days",                   # B3(出走回数・休養日数)
]
R1_EXTRA_COLS = [
    "kako5_std_pos", "kako5_amplitude",
    "distance_change_abs", "going_changed", "venue_changed",
]


def _load_market(year: int) -> pd.DataFrame:
    df = pd.read_csv(KEKKA_EXT, encoding="utf-8-sig",
                      usecols=["race_id", "umaban", "単勝オッズ"], low_memory=False)
    df["race_id"] = df["race_id"].astype(str)
    df = df[df["race_id"].str.startswith(str(year))].copy()
    df = df.dropna(subset=["単勝オッズ"])
    df["popularity_rank"] = df.groupby("race_id")["単勝オッズ"].rank(method="first", ascending=True)
    df["market_implied_prob"] = 1.0 / df["単勝オッズ"]
    return df.rename(columns={"umaban": "ban"})[["race_id", "ban", "market_implied_prob", "popularity_rank"]]


def build_stage1_5_table(years: list[int]) -> tuple[pd.DataFrame, dict]:
    labels, label_diag = compute_labels(years)
    targets, race_horses, _meta = build_target_rows(years)
    te = score_population(years)
    te = te[[COL_RID, COL_BAN, "kako5_race_count", "間隔", "kako5_std_pos",
             "kako5_best_pos", "kako5_avg_pos", "前距離", "距離",
             "前走馬場状態", "馬場状態", "前走場所", "場所"]].copy()
    te[COL_RID] = te[COL_RID].astype(str)

    # レース単位の集計(n, max_prob, entropy, gap_12, B0=focal自身の下方尾部確率)
    # — race_horses(全馬)から計算。全レースn<=18(実測)のためexact DPを直接使う。
    race_agg = {}
    for rid, g in race_horses.groupby("rid", sort=False):
        g = g.sort_values("pos_in_group").reset_index(drop=True)
        scores = g["score"].values.astype(float)
        w = np.exp(scores - scores.max())
        p = w / w.sum()
        entropy = float(-(p * np.log(np.clip(p, 1e-12, 1))).sum())
        ss = np.sort(scores)[::-1]
        gap_12 = float(ss[0] - ss[1]) if len(ss) >= 2 else np.nan
        n = len(g)
        tail_prob_gt3 = np.nan
        if n > 3:
            focal_rows = g.index[g["is_focal"]]
            if len(focal_rows) == 1:
                dist = exact_rank_distribution(scores, int(focal_rows[0]))
                tail_prob_gt3 = float(1.0 - np.cumsum(dist)[2])  # 1 - CDF(3)
        elif n <= 3:
            tail_prob_gt3 = 0.0  # 頭数3以下は3着超がありえない
        race_agg[rid] = dict(n=n, max_prob=float(p.max()), entropy=entropy, gap_12=gap_12,
                              tail_prob_gt3=tail_prob_gt3)

    market = _load_market(years[0] if len(years) == 1 else years)  # Stage1.5は2023単独

    rows = []
    n_market_missing = 0
    for _, lab in labels.iterrows():
        rid = lab["rid"]
        focal_ban = int(lab["focal_ban"])
        agg = race_agg.get(rid)
        if agg is None:
            continue
        te_row = te[(te[COL_RID] == rid) & (te[COL_BAN] == focal_ban)]
        if len(te_row) == 0:
            continue
        te_row = te_row.iloc[0]

        mkt = market[(market["race_id"] == rid) & (market["ban"] == focal_ban)]
        if len(mkt) == 0:
            n_market_missing += 1
            market_implied_prob = np.nan
            popularity_rank = np.nan
        else:
            market_implied_prob = float(mkt["market_implied_prob"].iloc[0])
            popularity_rank = float(mkt["popularity_rank"].iloc[0])

        distance_change_abs = abs(float(te_row["距離"]) - float(te_row["前距離"])) \
            if not pd.isna(te_row["前距離"]) else np.nan
        going_changed = float(te_row["前走馬場状態"] != te_row["馬場状態"]) \
            if not pd.isna(te_row["前走馬場状態"]) else np.nan
        venue_changed = float(te_row["前走場所"] != te_row["場所"]) \
            if not pd.isna(te_row["前走場所"]) else np.nan
        kako5_amplitude = (float(te_row["kako5_best_pos"]) - float(te_row["kako5_avg_pos"])) \
            if not pd.isna(te_row["kako5_best_pos"]) else np.nan

        rows.append(dict(
            rid=rid, catastrophic_downside=int(lab["catastrophic_downside"]),
            tail_prob_gt3=agg["tail_prob_gt3"],
            market_implied_prob=market_implied_prob, popularity_rank=popularity_rank,
            n=agg["n"], max_prob=agg["max_prob"], entropy=agg["entropy"], gap_12=agg["gap_12"],
            kako5_race_count=float(te_row["kako5_race_count"]),
            interval_days=float(te_row["間隔"]) if not pd.isna(te_row["間隔"]) else np.nan,
            kako5_std_pos=float(te_row["kako5_std_pos"]) if not pd.isna(te_row["kako5_std_pos"]) else np.nan,
            kako5_amplitude=kako5_amplitude,
            distance_change_abs=distance_change_abs,
            going_changed=going_changed, venue_changed=venue_changed,
        ))

    table = pd.DataFrame(rows)
    table["meeting_day"] = table["rid"].str[:10]
    diag = dict(n_rows=len(table), n_market_missing=n_market_missing, **label_diag)
    return table, diag


if __name__ == "__main__":
    table, diag = build_stage1_5_table([2023])
    print("[Stage1.5 特徴テーブル構築]")
    print(f"  行数: {len(table):,}")
    print(f"  市場オッズ欠損: {diag['n_market_missing']:,}")
    print(table.isna().mean().round(4).to_string())
    table.to_parquet(Path(__file__).parent / "out" / "stage1_5_features_2023.parquet")
    print("[saved] out/stage1_5_features_2023.parquet")
