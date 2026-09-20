# -*- coding: utf-8 -*-
"""
test_stage_b_gate_eval.py — stage_b_gate_eval.py の純粋ロジック検定 (実データ不要、高速)。
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.exp06_jev_decision_dev import stage_b_gate_eval as GE  # noqa: E402


def test_brier_perfect_prediction_is_zero():
    p = np.array([1.0, 0.0, 1.0])
    y = np.array([1, 0, 1])
    assert np.allclose(GE._brier(p, y), 0.0)


def test_brier_worst_prediction_is_one():
    p = np.array([0.0, 1.0])
    y = np.array([1, 0])
    assert np.allclose(GE._brier(p, y), 1.0)


def test_logloss_clips_extremes_no_inf():
    p = np.array([0.0, 1.0])
    y = np.array([1, 0])
    ll = GE._logloss(p, y)
    assert np.all(np.isfinite(ll))
    assert np.all(ll > 0)


def test_q5_risk_prob_sums_pass_uncertain_and_pass_ood():
    answers = {"Q5": {"type": "choice", "choice": "BET",
                      "probabilities": {"BET": 0.5, "PASS_UNCERTAIN": 0.2, "PASS_OOD": 0.1,
                                        "PASS_NO_EDGE": 0.15, "PASS_PRICE_RISK": 0.05}}}
    assert GE._q5_risk_prob(answers) == pytest.approx(0.3)


def test_q5_risk_prob_missing_q5_is_zero():
    assert GE._q5_risk_prob({}) == 0.0


def test_q5_is_bet_true_only_for_bet_choice():
    assert GE._q5_is_bet({"Q5": {"choice": "BET"}}) is True
    assert GE._q5_is_bet({"Q5": {"choice": "PASS_NO_EDGE"}}) is False
    assert GE._q5_is_bet({}) is False


def test_bootstrap_ci_point_matches_direct_stat():
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 0.1, size=200)
    b = rng.normal(0.5, 0.1, size=200)
    result = GE._bootstrap_ci([a, b], lambda x, y: float(np.mean(x) - np.mean(y)), n=500, seed=1)
    assert result["point"] == pytest.approx(np.mean(a) - np.mean(b))
    assert result["ci_lower_2.5"] < result["point"] < result["ci_upper_97.5"]
    assert result["n_per_group"] == [200, 200]


def test_bootstrap_ci_detects_no_real_difference_when_identical():
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 1.0, size=300)
    result = GE._bootstrap_ci([a, a.copy()], lambda x, y: float(np.mean(x) - np.mean(y)), n=500, seed=2)
    assert result["ci_lower_2.5"] <= 0.0 <= result["ci_upper_97.5"]
