# -*- coding: utf-8 -*-
"""test_gate_j1_calibration.py — Gate J1関連の純粋ロジック検定(実データ不要、高速)。"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.exp07_robust_portfolio_dev import gate_j1_calibration as GJ1  # noqa: E402


def test_brier_perfect_is_zero():
    assert GJ1._brier(np.array([1.0, 0.0]), np.array([1, 0])) == pytest.approx(0.0)


def test_logloss_finite_at_extremes():
    ll = GJ1._logloss(np.array([0.0, 1.0]), np.array([1, 0]))
    assert np.isfinite(ll) and ll > 0


def test_actual_umaren_pairs_basic():
    fin = pd.Series([1, 2, 3, 4])
    ban = pd.Series([5, 7, 2, 9])
    pairs = GJ1._actual_umaren_pairs(fin, ban)
    assert pairs == {(5, 7)}


def test_actual_umaren_pairs_empty_when_no_top2():
    fin = pd.Series([3, 4, 5])
    ban = pd.Series([1, 2, 3])
    assert GJ1._actual_umaren_pairs(fin, ban) == set()


def test_calibration_slope_intercept_perfect_calibration():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, size=5000)
    y = (rng.uniform(size=5000) < p).astype(int)
    result = GJ1._calibration_slope_intercept(p, y)
    assert result["slope"] == pytest.approx(1.0, abs=0.15)
    assert result["intercept"] == pytest.approx(0.0, abs=0.15)


def test_calibration_slope_intercept_single_class_returns_none():
    p = np.array([0.1, 0.2, 0.3])
    y = np.array([0, 0, 0])
    result = GJ1._calibration_slope_intercept(p, y)
    assert result["slope"] is None


def test_adaptive_bin_ece_perfect_calibration_near_zero():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.1, 0.9, size=5000)
    y = (rng.uniform(size=5000) < p).astype(int)
    result = GJ1._adaptive_bin_ece(p, y, n_bins=10)
    assert result["ece"] < 0.05


def test_adaptive_bin_ece_detects_systematic_overconfidence():
    # 予測は常に0.5だが実際の的中率は0.1しかない = 明確な過大評価
    p = np.full(2000, 0.5)
    rng = np.random.default_rng(2)
    y = (rng.uniform(size=2000) < 0.1).astype(int)
    result = GJ1._adaptive_bin_ece(p, y, n_bins=5)
    assert result["ece"] > 0.3


def test_low_bin_oe_skips_bins_with_insufficient_expected_events():
    # 最初の帯は期待イベント数<10(ノイズ)、2番目の帯は十分
    fake_ev = {
        "adaptive_bin_ece": {
            "reliability_table": [
                {"n": 1000, "predicted_hit_probability": 0.005, "observed_over_expected": 3.0},  # exp=5
                {"n": 1000, "predicted_hit_probability": 0.02, "observed_over_expected": 0.9},   # exp=20
            ]
        }
    }
    result = GJ1._low_bin_oe(fake_ev)
    assert result is not None
    assert result["observed_over_expected"] == pytest.approx(0.9)
    assert result["skipped_lower_bins_for_low_event_count"] == 1


def test_low_bin_oe_returns_none_when_no_bin_has_enough_events():
    fake_ev = {
        "adaptive_bin_ece": {
            "reliability_table": [
                {"n": 100, "predicted_hit_probability": 0.001, "observed_over_expected": 2.0},  # exp=0.1
            ]
        }
    }
    assert GJ1._low_bin_oe(fake_ev) is None


def test_gate_j1_verdict_flags_clear_tail_overconfidence():
    evaluations = {
        "toy": {
            "raw_pl": {
                "calibration": {"slope": 1.0, "intercept": 0.0},
                "adaptive_bin_ece": {"reliability_table": [
                    {"n": 10000, "predicted_hit_probability": 0.01, "observed_over_expected": 0.4},  # exp=100
                ]},
            },
            "existing_calibrated": {
                "calibration": {"slope": 1.0, "intercept": 0.0},
                "adaptive_bin_ece": {"reliability_table": [
                    {"n": 10000, "predicted_hit_probability": 0.008, "observed_over_expected": 1.0},  # exp=80
                ]},
            },
        }
    }
    verdicts = GJ1.gate_j1_verdict(evaluations)
    assert verdicts["toy"]["raw_pl_overconfident_at_low_probability_tail"] is True
    assert verdicts["toy"]["existing_calibrator_acceptable"] is True
    assert verdicts["toy"]["usable"] is True
