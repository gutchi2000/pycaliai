# -*- coding: utf-8 -*-
"""test_stage_b_gate_eval_extended.py — 純粋ロジック検定 (実データ不要、高速)。"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.exp06_jev_decision_dev import stage_b_gate_eval_extended as GEE  # noqa: E402


def test_onehot_known_categories():
    d = pd.DataFrame({"surface": ["turf", "dirt", "turf"]})
    X, vocab = GEE._onehot(d, ["surface"])
    assert vocab["surface"] == ["dirt", "turf"]
    assert X.shape == (3, 2)
    assert np.array_equal(X[0], [0, 1])  # turf
    assert np.array_equal(X[1], [1, 0])  # dirt


def test_onehot_unseen_category_becomes_all_zero_with_fixed_vocab():
    d_fit = pd.DataFrame({"surface": ["turf", "dirt"]})
    _, vocab = GEE._onehot(d_fit, ["surface"])
    d_new = pd.DataFrame({"surface": ["turf", "NEVER_SEEN"]})
    X_new, _ = GEE._onehot(d_new, ["surface"], vocab=vocab)
    assert np.array_equal(X_new[1], [0, 0])


def test_build_full_X_shape_matches_controls_plus_categoricals():
    n = 5
    d = pd.DataFrame({
        "m4_top_prob": np.linspace(0.1, 0.9, n), "entropy_m4": np.ones(n),
        "market_prob": np.linspace(0.1, 0.9, n), "market_entropy": np.ones(n),
        "ai_market_divergence": np.zeros(n), "model_rank_disagreement": np.zeros(n),
        "model_prob_variance": np.zeros(n), "in_distribution_support": np.ones(n) * 0.5,
        "similar_past_case_count": np.ones(n) * 10, "feature_missing_rate": np.zeros(n),
        "field_size": np.ones(n) * 10,
        "popularity_band": ["1-3"] * n, "venue": ["tokyo"] * n, "surface": ["turf"] * n,
        "distance_band": ["mile"] * n, "class_band": ["low"] * n,
        "year": [2024, 2024, 2025, 2025, 2025],
    })
    X, vocab, year_vocab = GEE._build_full_X(d, include_year=True)
    n_cont = len(GEE.FULL_CONT_CONTROLS)
    n_cat = sum(1 for _ in d["popularity_band"].unique()) + sum(1 for _ in d["venue"].unique()) + \
        sum(1 for _ in d["surface"].unique()) + sum(1 for _ in d["distance_band"].unique()) + \
        sum(1 for _ in d["class_band"].unique())
    n_year = 2
    assert X.shape == (n, n_cont + n_cat + n_year)
