# -*- coding: utf-8 -*-
"""test_stage_b_sensitivity_stability.py — 純粋ロジック検定 (実データ不要、高速)。"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.exp06_jev_decision_dev import stage_b_sensitivity_stability as ST  # noqa: E402


def test_pairwise_mean_abs_diff_identical_vectors_is_zero():
    vecs = [np.array([0.5, 0.3]), np.array([0.5, 0.3]), np.array([0.5, 0.3])]
    assert ST._pairwise_mean_abs_diff(vecs) == pytest.approx(0.0)


def test_pairwise_mean_abs_diff_known_value():
    vecs = [np.array([1.0]), np.array([0.0])]
    assert ST._pairwise_mean_abs_diff(vecs) == pytest.approx(1.0)


def test_pairwise_max_abs_diff_picks_largest_pair():
    vecs = [np.array([0.0]), np.array([0.1]), np.array([0.9])]
    assert ST._pairwise_max_abs_diff(vecs) == pytest.approx(0.9)


def test_icc_oneway_perfect_agreement_is_one():
    scores = [[1.0, 1.0, 1.0], [3.0, 3.0, 3.0], [0.0, 0.0, 0.0]]
    assert ST._icc_oneway(scores) == pytest.approx(1.0, abs=1e-6)


def test_icc_oneway_pure_noise_is_near_zero():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(200, 3)).tolist()
    icc = ST._icc_oneway(scores)
    assert -0.3 < icc < 0.3


def test_vec_probs_extracts_in_fixed_option_order():
    answers = {"Q5": {"probabilities": {"BET": 0.7, "PASS_NO_EDGE": 0.3}}}
    v = ST._vec_probs(answers, "Q5", ["BET", "PASS_NO_EDGE", "PASS_UNCERTAIN"])
    assert np.allclose(v, [0.7, 0.3, 0.0])


def test_vec_probs_missing_question_returns_none():
    assert ST._vec_probs({}, "Q5", ["BET"]) is None
