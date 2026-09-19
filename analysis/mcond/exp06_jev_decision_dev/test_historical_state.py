# -*- coding: utf-8 -*-
"""
test_historical_state.py — historical_state.py の純粋関数の検定 (実データ・実モデル
fit不要、高速)。フルパイプライン(models.pyのfit_m0_m3等、~1分)はpytestに含めない
(実データでの検証はSTAGE_B_DATA_AUDIT.mdの§7に記録済み: 34,545レース全件で欠損0件、
分布妥当性を確認済み)。
実行: python -m pytest analysis/mcond/exp06_jev_decision_dev/test_historical_state.py -q
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.exp06_jev_decision_dev import historical_state as HS  # noqa: E402


def test_entropy_uniform_is_max():
    uniform = np.array([0.25, 0.25, 0.25, 0.25])
    concentrated = np.array([0.97, 0.01, 0.01, 0.01])
    assert HS._entropy(uniform) > HS._entropy(concentrated)


def test_entropy_handles_empty_and_zeros():
    assert np.isnan(HS._entropy(np.array([])))
    assert np.isnan(HS._entropy(np.array([0.0, 0.0])))


def test_venue_name_detects_onehot_and_unknown():
    row = pd.Series({c: 0 for c in HS._VENUE_COLS})
    assert HS._venue_name(row) == "unknown"
    row["c1__場所__東京"] = 1
    assert HS._venue_name(row) == "東京"


def test_distance_band_boundaries_fixed_not_data_driven():
    """distance_band境界は固定値であり、標本分位点から動的に決めていないことを確認する
    (結果を見て変更しない設計の裏付け)。"""
    assert HS._DISTANCE_BINS == [0, 1400, 1800, 2200, 2800, 99999]
    assert HS._DISTANCE_LABELS == ["sprint", "mile", "intermediate", "long", "extended"]
    binned = pd.cut([1000, 1500, 2000, 2500, 3000], HS._DISTANCE_BINS, labels=HS._DISTANCE_LABELS)
    assert list(binned.astype(str)) == ["sprint", "mile", "intermediate", "long", "extended"]


def test_class_band_boundaries_are_thirds_of_fixed_ordinal_range():
    """クラス帯はordinalの固定レンジ(0-9)の三等分であり、標本分位点ではない。"""
    assert HS._CLASS_BINS == [-0.1, 3, 6, 9.1]
    binned = pd.cut([0, 2, 4, 5, 7, 9], HS._CLASS_BINS, labels=HS._CLASS_LABELS)
    assert list(binned.astype(str)) == ["low", "low", "mid", "mid", "high", "high"]


def test_popularity_band_boundaries():
    assert HS._POPULARITY_BINS == [0, 3, 6, 999]
    binned = pd.cut([1, 3, 4, 6, 10], HS._POPULARITY_BINS, labels=HS._POPULARITY_LABELS)
    assert list(binned.astype(str)) == ["1-3", "1-3", "4-6", "4-6", "7+"]
