# -*- coding: utf-8 -*-
"""
eligible_races.py — Stage2実装順ステップ1: eligible race集合の生成。

spec.json `exception_handling` に固定した例外処理カテゴリを適用する。結果ラベルは
使うが(同着検出等に必須)、2024・2025年の**性能**(coverage/logloss/ROI等)は
一切計算・出力しない(この段階は集合を確定するだけ)。全7方式(Stage2以降)は
このモジュールが返す同一のeligible race集合を使う。

データ源: data/_research/mcond/exp05_design.parquet の rid16/ban/year/v6_score/
n_field/fin/win/date(EXP07で確立済み・cross-check済みのv6生スコア源、2016-2025網羅)。

実行: 単体実行しない。aps.py / comparators.py から import して使う。
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
HERE = Path(__file__).resolve().parent

PROB_SUM_TOLERANCE = 1e-6
MIN_FIELD_SIZE = 3


def load_raw(years: list[int]) -> pd.DataFrame:
    df = pd.read_parquet(
        BASE / "data/_research/mcond/exp05_design.parquet",
        columns=["rid16", "ban", "year", "v6_score", "n_field", "fin", "win", "date"],
    )
    return df[df["year"].isin(years)].copy()


def build_eligible_races(years: list[int]) -> dict:
    """例外処理カテゴリを適用し、eligible race集合(rid16のset、年別)と
    除外理由別カウント(spec.json exception_handling準拠)を返す。

    戻り値: {
      "eligible_rid16": set[str],
      "exclusion_counts": {year: {category: count}},
      "race_frames": {rid16: DataFrame(ban, v6_score, fin, win)},  # eligibleのみ
    }
    """
    df = load_raw(years)
    exclusion_counts = {y: {
        "dead_heat": 0, "duplicate_race_id": 0, "nan_score": 0,
        "field_too_small": 0, "prob_sum_anomaly": 0, "missing_result": 0,
    } for y in years}

    eligible_rid16 = set()
    race_frames = {}

    for rid16, g in df.groupby("rid16", sort=False):
        year = int(g["year"].iloc[0])
        if year not in exclusion_counts:
            continue

        # 結果欠損(finが非数値/NaN)
        if g["fin"].isna().any():
            exclusion_counts[year]["missing_result"] += 1
            continue

        # race_id重複(同一rid16内で同一banが複数行)
        if g["ban"].duplicated().any():
            exclusion_counts[year]["duplicate_race_id"] += 1
            continue

        # 同着(1着(fin==1)が複数)
        if (g["fin"] == 1).sum() > 1:
            exclusion_counts[year]["dead_heat"] += 1
            continue

        # 極端な少頭数
        n_field = int(g["n_field"].iloc[0])
        if len(g) < MIN_FIELD_SIZE or n_field < MIN_FIELD_SIZE:
            exclusion_counts[year]["field_too_small"] += 1
            continue

        # v6_scoreがNaNの馬が1頭でもいる
        scores = g["v6_score"].to_numpy(dtype=float)
        if np.isnan(scores).any():
            exclusion_counts[year]["nan_score"] += 1
            continue

        # 単勝確率(pl_probs.all_tansho(pl_probs.pl_weights(scores))と等価)の
        # 確率和チェック(許容誤差1e-6、超過分は除外・丸めのみ再正規化)
        p = _tansho_probs(scores)
        s = float(p.sum())
        if abs(s - 1.0) > PROB_SUM_TOLERANCE:
            exclusion_counts[year]["prob_sum_anomaly"] += 1
            continue

        eligible_rid16.add(rid16)
        race_frames[rid16] = g[["ban", "v6_score", "fin", "win"]].reset_index(drop=True)

    return {
        "eligible_rid16": eligible_rid16,
        "exclusion_counts": exclusion_counts,
        "race_frames": race_frames,
    }


def _tansho_probs(scores: np.ndarray) -> np.ndarray:
    """pl_probs.all_tansho(pl_probs.pl_weights(scores))と数値的に同一
    (独自実装、テストで既存pl_probs.pyとの一致を検証する)。
    pl_weights: w_i=exp(s_i-max) は正規化前の重みであることに注意
    (pl_probs.all_tanshoがw/w.sum()で正規化する)。"""
    w = np.exp(scores - np.max(scores))
    return w / w.sum()
