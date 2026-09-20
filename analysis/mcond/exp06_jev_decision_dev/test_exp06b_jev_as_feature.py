# -*- coding: utf-8 -*-
"""test_exp06b_jev_as_feature.py — 純粋ロジック検定 (実データ不要、高速)。"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.exp06_jev_decision_dev import exp06b_jev_as_feature as E6B  # noqa: E402


def test_jev_feature_row_extracts_all_expected_keys():
    answers = {
        "Q1": {"noul": 0.6},
        "Q2": {"score": 1.2, "probabilities": {"0": 0.1, "1": 0.5, "2": 0.2, "3": 0.1, "4": 0.1}},
        "Q4": {"probabilities": {"AI": 0.2, "MARKET": 0.3, "BLEND": 0.4, "ABSTAIN": 0.1}},
        "Q5": {"probabilities": {"BET": 0.3, "PASS_NO_EDGE": 0.4, "PASS_UNCERTAIN": 0.1,
                                 "PASS_OOD": 0.1, "PASS_PRICE_RISK": 0.1}},
    }
    row = E6B._jev_feature_row(answers)
    assert row["jev_q1_noul"] == pytest.approx(0.6)
    assert row["jev_q2_score"] == pytest.approx(1.2)
    assert row["jev_q2_p1"] == pytest.approx(0.5)
    assert row["jev_q4_p_BLEND"] == pytest.approx(0.4)
    assert row["jev_q5_p_PASS_NO_EDGE"] == pytest.approx(0.4)
    assert len(row) == 2 + 5 + 4 + 5  # q1 + q2_score + q2 levels + q4 opts + q5 opts


def test_jev_feature_row_missing_questions_default_to_zero_or_nan():
    row = E6B._jev_feature_row({})
    assert np.isnan(row["jev_q1_noul"])
    assert row["jev_q4_p_AI"] == 0.0


def test_bootstrap_diff_zero_when_predictions_identical():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.1, 0.9, size=500)
    y = (rng.uniform(size=500) < p).astype(int)
    result = E6B._bootstrap_diff(p, p.copy(), y, lambda pp, yy: (pp - yy) ** 2, n=200, seed=1)
    assert result["point_r1_minus_r0"] == pytest.approx(0.0)
    assert result["ci_lower_2.5"] <= 0.0 <= result["ci_upper_97.5"]


def test_bootstrap_diff_detects_clear_improvement():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=500).astype(float)
    p0 = np.full(500, 0.5)  # 情報なし
    p1 = y * 0.9 + (1 - y) * 0.1  # ほぼ完璧
    result = E6B._bootstrap_diff(p0, p1, y, lambda pp, yy: (pp - yy) ** 2, n=200, seed=2)
    assert result["ci_upper_97.5"] < 0  # p1(R1)がp0(R0)よりbrierが低い=改善
