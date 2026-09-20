# -*- coding: utf-8 -*-
"""test_uncertainty_scenarios.py — 不確実性集合生成の純粋ロジック検定(実データ不要)。"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.exp07_robust_portfolio_dev import uncertainty_scenarios as US  # noqa: E402
import pl_probs as PL  # noqa: E402


def test_score_noise_scale_zero_when_raw_equals_calibrated():
    df = pd.DataFrame({"raw_p": [0.1, 0.2, 0.3], "cal_p": [0.1, 0.2, 0.3]})
    assert US.estimate_score_noise_scale_from_2023(df) == pytest.approx(0.0, abs=1e-9)


def test_score_noise_scale_positive_when_calibration_shifts_probabilities():
    rng = np.random.default_rng(0)
    raw = rng.uniform(0.05, 0.5, size=200)
    cal = np.clip(raw * 1.3, 0.01, 0.99)  # 系統的にずれた較正
    df = pd.DataFrame({"raw_p": raw, "cal_p": cal})
    scale = US.estimate_score_noise_scale_from_2023(df)
    assert scale > 0.0


def test_perturbation_scenarios_count_and_shape():
    scores = np.array([0.5, -0.2, 1.1, 0.0])
    draws = US.generate_score_perturbation_scenarios(scores, n_draws=10, sigma=0.3, seed=1)
    assert len(draws) == 10
    assert all(d.shape == scores.shape for d in draws)


def test_perturbation_zero_sigma_returns_identical_scores():
    scores = np.array([0.5, -0.2, 1.1])
    draws = US.generate_score_perturbation_scenarios(scores, n_draws=5, sigma=0.0, seed=2)
    assert all(np.allclose(d, scores) for d in draws)


def test_build_uncertainty_scenarios_all_pass_gate_j0():
    rng = np.random.default_rng(3)
    scores = rng.normal(size=8)
    result = US.build_uncertainty_state_probability_scenarios(
        scores, n_draws=15, sigma=0.4, seed=4,
    )
    assert result["all_draws_passed_gate_j0"] is True
    assert result["n_gate_j0_failures"] == 0
    assert len(result["scenarios"]) == 15
    # 全シナリオが同じ状態順序(states)を共有し、各シナリオの確率和が1であること
    for probs in result["scenarios"]:
        assert abs(sum(probs) - 1.0) < 1e-6


def test_build_uncertainty_scenarios_wider_sigma_gives_more_dispersed_top1_prob():
    rng = np.random.default_rng(5)
    scores = rng.normal(size=6)
    narrow = US.build_uncertainty_state_probability_scenarios(scores, n_draws=200, sigma=0.05, seed=6)
    wide = US.build_uncertainty_state_probability_scenarios(scores, n_draws=200, sigma=1.0, seed=6)

    w0 = PL.pl_weights(scores)
    top1_narrow = [
        sum(p for s, p in zip(narrow["base_states"], scen) if s[0] == 0)
        for scen in narrow["scenarios"]
    ]
    top1_wide = [
        sum(p for s, p in zip(wide["base_states"], scen) if s[0] == 0)
        for scen in wide["scenarios"]
    ]
    assert np.std(top1_wide) > np.std(top1_narrow)


def test_odds_drift_distribution_from_synthetic_tanpuk_frame():
    df = pd.DataFrame({
        "レースID": ["r1", "r1", "r2", "r2"],
        "区分": [1, 4, 1, 4],
        "月日時分": ["0101090000", "0101100000", "0101090000", "0101100000"],
        "1単": [5.0, 6.0, 3.0, 2.5],
    })
    result = US.estimate_odds_drift_distribution_from_2023(df)
    assert result["n"] == 2
    assert result["mean_ratio"] == pytest.approx((6.0 / 5.0 + 2.5 / 3.0) / 2, abs=1e-6)
