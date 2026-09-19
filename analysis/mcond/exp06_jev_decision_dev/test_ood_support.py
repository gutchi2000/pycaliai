# -*- coding: utf-8 -*-
"""
test_ood_support.py — ood_support.SupportModel の合成データ検定 (実データ不要、高速)。
実データでの検証はstage_b_dry_run.py実行時のログ/STAGE_B_DATA_AUDIT.mdを参照
(2023年fit→2024/2025年scoreでin_distribution_supportが概ね一様分布になることを確認済み)。
実行: python -m pytest analysis/mcond/exp06_jev_decision_dev/test_ood_support.py -q
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.exp06_jev_decision_dev.ood_support import (  # noqa: E402
    SupportModel, CONTINUOUS_COLS, CATEGORICAL_COLS, K_NEIGHBORS, RADIUS_PERCENTILE,
)


def _synthetic_reference(n=500, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n) for c in CONTINUOUS_COLS})
    df["venue"] = rng.choice(["A", "B", "C"], size=n)
    df["surface"] = rng.choice(["turf", "dirt"], size=n)
    df["distance_band"] = rng.choice(["short", "mid", "long"], size=n)
    df["class_band"] = rng.choice(["low", "mid", "high"], size=n)
    df["popularity_band"] = rng.choice(["1-3", "4-6", "7+"], size=n)
    return df


def test_in_distribution_query_gets_moderate_to_high_support():
    """参照集合そのものから抜き出した点は明らかに分布内だが、個々の点はスパースな
    裾に当たることもあるため、平均でチェックする(単点ごとの厳密な下限は課さない)。"""
    ref = _synthetic_reference()
    sm = SupportModel().fit(ref)
    query = ref.iloc[:30]
    result = sm.score(query)
    assert result["in_distribution_support"].mean() > 0.3
    assert (result["similar_past_case_count"] > 0).any()


def test_clear_outlier_gets_zero_support():
    ref = _synthetic_reference()
    sm = SupportModel().fit(ref)
    rng = np.random.default_rng(1)
    outlier = pd.DataFrame({c: rng.normal(loc=15, scale=1, size=5) for c in CONTINUOUS_COLS})
    outlier["venue"] = "NEVER_SEEN_VENUE"
    outlier["surface"] = "turf"
    outlier["distance_band"] = "short"
    outlier["class_band"] = "low"
    outlier["popularity_band"] = "1-3"
    result = sm.score(outlier)
    assert (result["in_distribution_support"] == 0.0).all()
    assert (result["similar_past_case_count"] == 0).all()


def test_support_bounded_0_1():
    ref = _synthetic_reference()
    sm = SupportModel().fit(ref)
    rng = np.random.default_rng(2)
    mixed = pd.concat([ref.iloc[:10], _synthetic_reference(n=10, seed=99)], ignore_index=True)
    result = sm.score(mixed)
    assert (result["in_distribution_support"] >= 0.0).all()
    assert (result["in_distribution_support"] <= 1.0).all()


def test_unseen_category_treated_as_unk_not_crash():
    """未知カテゴリ(2023年で観測されなかった値)は全列0(__UNK__相当)として扱われ、
    例外を出さないこと。"""
    ref = _synthetic_reference()
    sm = SupportModel().fit(ref)
    query = ref.iloc[:3].copy()
    query["venue"] = "TOTALLY_NEW_VENUE"
    result = sm.score(query)
    assert len(result) == 3
    assert result["in_distribution_support"].notna().all()


def test_fit_never_refits_on_score_call():
    """score()は2023年で凍結したパラメータを変えない(呼び出し前後でmedian/iqr/PCAが
    同一であることを確認、再fit禁止の実装確認)。"""
    ref = _synthetic_reference()
    sm = SupportModel().fit(ref)
    median_before = dict(sm.median_)
    pca_components_before = sm.pca_components_.copy()
    sm.score(_synthetic_reference(n=50, seed=7))
    assert sm.median_ == median_before
    assert np.array_equal(sm.pca_components_, pca_components_before)


def test_similar_radius_matches_p95_of_d20_ref():
    ref = _synthetic_reference()
    sm = SupportModel().fit(ref)
    assert sm.similar_radius_ == pytest.approx(np.percentile(sm.d20_ref_, RADIUS_PERCENTILE))


def test_pca_dims_capped_at_20():
    ref = _synthetic_reference(n=300)
    sm = SupportModel().fit(ref)
    assert sm.n_pca_dims_ <= 20


def test_k_neighbors_constant_is_20():
    assert K_NEIGHBORS == 20
