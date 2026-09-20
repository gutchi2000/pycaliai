# -*- coding: utf-8 -*-
"""
test_evaluate.py — evaluate.pyの単体テスト(ロジスティック回帰・bootstrap・
permutation placeboのシャッフル機構)。2024・2025年の実データは一切使わない。
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.exp08_online_track_state_dev.evaluate import (  # noqa: E402
    fit_offset_logistic, predict_offset_logistic, logloss,
    paired_bootstrap_delta_logloss, shuffle_observations_within_unit, meeting_day_key,
)
from analysis.mcond.exp08_online_track_state_dev.online_state import STATE_DIMS  # noqa: E402


def test_fit_offset_logistic_recovers_known_coefficient():
    """既知のbeta_trueで生成した合成データから、fit_offset_logisticがおおよそ
    正しいbetaを復元できること(offsetは固定でfitしないことの確認も兼ねる)。"""
    rng = np.random.default_rng(0)
    n = 5000
    X = rng.normal(size=(n, 1))
    offset = rng.normal(scale=0.5, size=n)
    beta_true = np.array([1.5])
    from scipy.special import expit
    p_true = expit(offset + X @ beta_true)
    y = rng.binomial(1, p_true)
    beta_hat = fit_offset_logistic(X, y, offset, l2=0.001)
    assert beta_hat[0] == pytest.approx(1.5, abs=0.15)


def test_fit_offset_logistic_empty_features_returns_empty_beta():
    X = np.zeros((100, 0))
    y = np.zeros(100)
    offset = np.zeros(100)
    beta = fit_offset_logistic(X, y, offset)
    assert beta.shape == (0,)


def test_predict_offset_logistic_matches_manual_sigmoid():
    X = np.array([[1.0], [2.0]])
    offset = np.array([0.0, 0.0])
    beta = np.array([0.5])
    p = predict_offset_logistic(X, offset, beta)
    from scipy.special import expit
    expected = expit(np.array([0.5, 1.0]))
    np.testing.assert_allclose(p, expected)


def test_logloss_perfect_prediction_near_zero():
    y = np.array([1.0, 0.0, 1.0, 0.0])
    p = np.array([0.999, 0.001, 0.999, 0.001])
    assert logloss(y, p) < 0.01


def test_paired_bootstrap_identical_models_gives_zero_point():
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    p = np.array([0.6, 0.3, 0.7, 0.2, 0.5, 0.4, 0.6, 0.3])
    rid16 = np.array([f"202401060{i}010101" for i in range(1, 5)] * 2)
    out = paired_bootstrap_delta_logloss(y, p, p, rid16, n_boot=200)
    assert out["point"] == pytest.approx(0.0, abs=1e-12)


def test_paired_bootstrap_detects_clear_improvement():
    """model_aが明確に良い(y=1でp大きい、y=0でp小さい)ケースでpoint<0
    (a-bで改善=負)になること。"""
    rng = np.random.default_rng(1)
    n_races = 300
    y = rng.binomial(1, 0.22, size=n_races).astype(float)
    p_a = np.where(y == 1, 0.5, 0.15)  # 良いモデル
    p_b = np.full(n_races, 0.22)       # ベースライン
    rid16 = np.array([f"2024{(i%200)+100:04d}0601{ (i%12)+1:02d}01" for i in range(n_races)])
    out = paired_bootstrap_delta_logloss(y, p_a, p_b, rid16, n_boot=500)
    assert out["point"] < 0


def test_meeting_day_key_uses_first_10_chars():
    rid = np.array(["2024010606010101", "2024010606010102", "2024010706010101"])
    keys = meeting_day_key(rid)
    assert keys[0] == keys[1] == "2024010606"
    assert keys[2] == "2024010706"


def _mk_obs_for_shuffle():
    rows = []
    rng = np.random.default_rng(0)
    for i in range(6):
        rows.append({
            "rid16": f"r{i}", "date": "20240106", "venue": "05", "surface": "芝",
            "speed_signal": rng.normal() if i % 3 != 0 else np.nan,  # 一部欠損
            "agari_signal": rng.normal(),
            "pace_signal": rng.normal(),
        })
    return pd.DataFrame(rows)


def test_shuffle_preserves_missing_pattern():
    obs = _mk_obs_for_shuffle()
    shuffled = shuffle_observations_within_unit(obs, seed=1)
    for d in STATE_DIMS:
        orig_missing = obs[d].isna()
        new_missing = shuffled[d].isna()
        pd.testing.assert_series_equal(orig_missing, new_missing, check_names=False)


def test_shuffle_preserves_race_count_and_multiset_of_values():
    obs = _mk_obs_for_shuffle()
    shuffled = shuffle_observations_within_unit(obs, seed=2)
    assert len(shuffled) == len(obs)
    for d in STATE_DIMS:
        orig_vals = np.sort(obs[d].dropna().to_numpy())
        new_vals = np.sort(shuffled[d].dropna().to_numpy())
        np.testing.assert_allclose(orig_vals, new_vals)


def test_shuffle_does_not_touch_non_signal_columns():
    obs = _mk_obs_for_shuffle()
    obs["decision_timestamp"] = pd.Timestamp("2024-01-06 10:00")
    shuffled = shuffle_observations_within_unit(obs, seed=3)
    pd.testing.assert_series_equal(obs["decision_timestamp"], shuffled["decision_timestamp"])
    pd.testing.assert_series_equal(obs["rid16"], shuffled["rid16"])


def test_shuffle_actually_changes_the_assignment():
    """シャッフルが自明な恒等置換(たまたま毎回同じ)になっていないことの確認
    (統計的な健全性チェック、稀に一致することはあるがseedを変えれば必ずどこかで動く)。"""
    obs = _mk_obs_for_shuffle()
    changed = False
    for seed in range(20):
        shuffled = shuffle_observations_within_unit(obs, seed=seed)
        if not shuffled["speed_signal"].equals(obs["speed_signal"]):
            changed = True
            break
    assert changed, "20回シャッフルしても一度も変化しなかった(シャッフルが機能していない可能性)"
