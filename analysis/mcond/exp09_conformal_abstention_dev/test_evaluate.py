# -*- coding: utf-8 -*-
"""
test_evaluate.py — evaluate.pyの合成テスト。実データ(2024-2025)は使わない。
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.exp09_conformal_abstention_dev.evaluate import (  # noqa: E402
    race_level_metrics, select_top_n, meeting_day_key, paired_bootstrap_delta,
    gate2_same_participation_rate, gate3_full_control,
)


def _mk_race(rid16, scores, winner_idx):
    n = len(scores)
    fin = [2] * n
    fin[winner_idx] = 1
    win = [0] * n
    win[winner_idx] = 1
    return rid16, pd.DataFrame({"ban": range(1, n + 1), "v6_score": scores, "fin": fin, "win": win})


def test_race_level_metrics_perfect_prediction_low_logloss():
    """真の勝ち馬がargmaxで、スコア差が大きいほどlogloss/brierが小さいこと。"""
    rid, df = _mk_race("r1", [10.0, -10.0, -10.0], 0)  # ほぼ確実に0番が勝つ予測
    metrics = race_level_metrics({rid: df})
    assert metrics.iloc[0]["logloss"] < 0.01
    assert metrics.iloc[0]["brier"] < 0.01


def test_race_level_metrics_wrong_favorite_high_loss():
    rid, df = _mk_race("r1", [10.0, -10.0, -10.0], 2)  # 予測は0番だが実際は2番が勝つ
    metrics = race_level_metrics({rid: df})
    assert metrics.iloc[0]["logloss"] > 5.0
    assert metrics.iloc[0]["brier"] == pytest.approx(1.0, abs=0.01)


def test_select_top_n_picks_smallest_scores():
    df = pd.DataFrame({"rid16": ["a", "b", "c", "d"], "score": [3, 1, 4, 2]})
    sel = select_top_n(df, 2)
    assert sel == {"b", "d"}


def test_meeting_day_key_uses_first_10_chars():
    assert meeting_day_key("2024010606010101") == "2024010606"


def test_paired_bootstrap_zero_when_identical():
    rid = np.array([f"202401060601{str(i).zfill(2)}" for i in range(1, 11)])
    metric = np.array([0.5] * 10)
    out = paired_bootstrap_delta(metric, metric, rid)
    assert out["point"] == pytest.approx(0.0, abs=1e-9)


def test_paired_bootstrap_detects_clear_difference():
    rid = np.array([f"202401060601{str(i).zfill(2)}" for i in range(1, 11)])
    a = np.array([0.1] * 10)
    b = np.array([0.5] * 10)
    out = paired_bootstrap_delta(a, b, rid)
    assert out["point"] < 0  # aがbより改善(小さい)


def test_gate2_same_participation_rate_selects_equal_counts():
    rng = np.random.default_rng(0)
    n_races = 40
    frames = {}
    for i in range(n_races):
        n = 8
        scores = rng.normal(size=n)
        winner = int(rng.integers(0, n))
        rid, df = _mk_race(f"r{i}", scores, winner)
        frames[rid] = df
    metrics = race_level_metrics(frames)

    method_scores = {}
    for name in ["conformal", "max_prob", "entropy"]:
        s = rng.permutation(n_races).astype(float)
        method_scores[name] = pd.DataFrame({"rid16": list(frames.keys()), "score": s})

    result = gate2_same_participation_rate(metrics, method_scores, 0.75, n_races)
    assert result["n_select"] == round(0.75 * n_races)
    assert "max_prob" in result["vs"]
    assert "entropy" in result["vs"]


def test_gate3_full_control_runs_and_returns_ci():
    rng = np.random.default_rng(1)
    n_races = 200
    frames = {}
    for i in range(n_races):
        n = 8
        scores = rng.normal(size=n)
        winner = int(rng.integers(0, n))
        rid, df = _mk_race(f"r{i}", scores, winner)
        frames[rid] = df
    metrics = race_level_metrics(frames)
    conformal_score = pd.DataFrame({"rid16": list(frames.keys()),
                                    "score": rng.normal(size=n_races)})
    controls = pd.DataFrame({"rid16": list(frames.keys()),
                             "n_field": [8] * n_races,
                             "max_prob": rng.uniform(0.1, 0.9, size=n_races)})
    out = gate3_full_control(metrics, conformal_score, controls)
    assert "conformal_coefficient" in out
    assert len(out["ci95"]) == 2
    assert isinstance(out["survives_full_control"], bool)
