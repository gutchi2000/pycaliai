# -*- coding: utf-8 -*-
"""
asof_features.py
==================
EXP11 Stage1修正2。低頻度騎手×調教師ペアの判定を「日初時点固定」の完全
as-of方式で再計算する。全期間集計(Stage0の暫定版)は使わない。

方式(ユーザー指定、日初時点固定を主解析として採用):
  各レース日dについて、pair_prior_count(as-of) = dより前の日に成立した
  (騎手コード,調教師コード)ペアの累積出現回数。同日内の他レース(dより前でも
  後でも)は一切カウントしない=日初時点でスナップショット固定。

  ※「同日内で判断時点より前に終了したレースを加える」方式(逐次更新)は
  実装が複雑な割に効果が薄い(同日内の発走順は数分〜数十分差で、走行間隔の
  実質的な情報価値は低い)と判断し、主解析では不採用。将来必要なら
  add_pair_prior_count(..., mode="sequential_same_day")として拡張可能な
  設計にしてある。

同じ方式でjockey_prior_count/trainer_prior_count(騎手・調教師それぞれの
as-ofキャリア出走数)も計算し、未知カテゴリの4分類(項目6)に使う。

削除不変性テスト: test_asof_features.pyで、未来年のデータを削除しても
過去行のpair_prior_count/jockey_prior_count/trainer_prior_countが変化
しないことを確認する。
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def add_asof_prior_counts(
    df: pd.DataFrame,
    date_col: str = "日付",
    jockey_col: str = "騎手コード",
    trainer_col: str = "調教師コード",
) -> pd.DataFrame:
    """日初時点固定のas-of pair/jockey/trainer prior countを追加する。

    df は既に日付昇順にソートされている必要はない(内部でソートする)。
    戻り値は元の行順を保持したコピー。
    """
    work = df.copy()
    work["_orig_order"] = np.arange(len(work))
    work = work.sort_values(date_col, kind="mergesort")  # 安定ソート、同日内は元の順序を保持

    pair_counts: dict[tuple, int] = defaultdict(int)
    jockey_counts: dict = defaultdict(int)
    trainer_counts: dict = defaultdict(int)

    pair_prior = np.zeros(len(work), dtype=np.int64)
    jockey_prior = np.zeros(len(work), dtype=np.int64)
    trainer_prior = np.zeros(len(work), dtype=np.int64)

    dates = work[date_col].values
    jockeys = work[jockey_col].values
    trainers = work[trainer_col].values

    pos = 0
    n = len(work)
    while pos < n:
        d = dates[pos]
        end = pos
        while end < n and dates[end] == d:
            end += 1
        # このdの全行に「dより前」の状態(スナップショット)を割り当てる
        for i in range(pos, end):
            key = (jockeys[i], trainers[i])
            pair_prior[i] = pair_counts[key]
            jockey_prior[i] = jockey_counts[jockeys[i]]
            trainer_prior[i] = trainer_counts[trainers[i]]
        # その後にdの分を加算(次の日以降のスナップショットに反映される)
        for i in range(pos, end):
            key = (jockeys[i], trainers[i])
            pair_counts[key] += 1
            jockey_counts[jockeys[i]] += 1
            trainer_counts[trainers[i]] += 1
        pos = end

    work["pair_prior_count_asof"] = pair_prior
    work["jockey_prior_count_asof"] = jockey_prior
    work["trainer_prior_count_asof"] = trainer_prior
    work = work.sort_values("_orig_order").drop(columns=["_orig_order"])
    return work


def classify_unknown_category(row) -> str:
    """項目6の6分類。prior_count==0を「未知」とみなす(そのカテゴリでの
    as-of実績がゼロ=このモデルにとって初見)。"""
    jp = row["jockey_prior_count_asof"]
    tp = row["trainer_prior_count_asof"]
    pp = row["pair_prior_count_asof"]
    jockey_known = jp > 0
    trainer_known = tp > 0
    if jockey_known and trainer_known:
        cat = "known_jockey_known_trainer_new_pair" if pp == 0 else "known_pair"
    elif jockey_known and not trainer_known:
        cat = "known_jockey_new_trainer"
    elif not jockey_known and trainer_known:
        cat = "new_jockey_known_trainer"
    else:
        cat = "both_new"
    return cat


def add_unknown_category_class(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["unknown_category_class"] = df.apply(classify_unknown_category, axis=1)
    df["pair_prior_bucket"] = pd.cut(
        df["pair_prior_count_asof"], [-1, 0, 4, np.inf],
        labels=["0", "1-4", "5+"],
    )
    return df
