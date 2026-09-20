# -*- coding: utf-8 -*-
"""test_build_scenarios.py — Gate J0関連の純粋ロジック検定(実データ不要、高速)。"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.exp07_robust_portfolio_dev import build_scenarios as BS  # noqa: E402
import pl_probs as PL  # noqa: E402


def _toy_weights(n=6, seed=0):
    rng = np.random.default_rng(seed)
    return PL.pl_weights(rng.normal(size=n))


def test_state_count_matches_permutation_formula():
    states, probs = BS.build_top3_states(_toy_weights(n=6))
    assert len(states) == 6 * 5 * 4
    assert len(probs) == len(states)


def test_probabilities_sum_to_one():
    _, probs = BS.build_top3_states(_toy_weights(n=8, seed=1))
    assert abs(probs.sum() - 1.0) < 1e-9


def test_no_state_has_duplicate_horse():
    states, _ = BS.build_top3_states(_toy_weights(n=8, seed=2))
    assert all(len(set(s)) == 3 for s in states)


def test_tansho_wins_only_when_first():
    assert BS.ticket_wins_in_state("tansho", (2,), (2, 0, 1)) is True
    assert BS.ticket_wins_in_state("tansho", (2,), (0, 2, 1)) is False


def test_fukusho_wins_if_in_top3_any_position():
    assert BS.ticket_wins_in_state("fukusho", (2,), (2, 0, 1)) is True
    assert BS.ticket_wins_in_state("fukusho", (2,), (0, 1, 2)) is True
    assert BS.ticket_wins_in_state("fukusho", (2,), (0, 1, 3)) is False


def test_umaren_ignores_order_within_top2_but_not_third():
    assert BS.ticket_wins_in_state("umaren", (0, 2), (2, 0, 1)) is True
    assert BS.ticket_wins_in_state("umaren", (0, 2), (0, 2, 1)) is True
    assert BS.ticket_wins_in_state("umaren", (0, 1), (2, 0, 1)) is False  # 1は3着、対象外


def test_wide_wins_if_both_in_top3_regardless_of_order():
    assert BS.ticket_wins_in_state("wide", (0, 3), (2, 0, 3)) is True
    assert BS.ticket_wins_in_state("wide", (0, 3), (3, 2, 0)) is True
    assert BS.ticket_wins_in_state("wide", (0, 4), (2, 0, 3)) is False


def test_umatan_requires_exact_order():
    assert BS.ticket_wins_in_state("umatan", (2, 0), (2, 0, 1)) is True
    assert BS.ticket_wins_in_state("umatan", (0, 2), (2, 0, 1)) is False


def test_sanrenpuku_ignores_order():
    assert BS.ticket_wins_in_state("sanrenpuku", (0, 1, 2), (2, 0, 1)) is True
    assert BS.ticket_wins_in_state("sanrenpuku", (0, 1, 3), (2, 0, 1)) is False


def test_sanrentan_requires_exact_order():
    assert BS.ticket_wins_in_state("sanrentan", (2, 0, 1), (2, 0, 1)) is True
    assert BS.ticket_wins_in_state("sanrentan", (0, 2, 1), (2, 0, 1)) is False


def test_unknown_ticket_type_raises():
    with pytest.raises(ValueError):
        BS.ticket_wins_in_state("wakuren", (0, 1), (2, 0, 1))


def test_state_payoffs_zero_when_losing():
    states, _ = BS.build_top3_states(_toy_weights(n=5, seed=3))
    payoffs = BS.state_payoffs_for_ticket("tansho", (0,), states, odds_floor=3.5)
    winning = [p for p, s in zip(payoffs, states) if s[0] == 0]
    losing = [p for p, s in zip(payoffs, states) if s[0] != 0]
    assert all(p == pytest.approx(3.5) for p in winning)
    assert all(p == 0.0 for p in losing)


def test_gate_j0_passes_on_synthetic_scores():
    result = BS.gate_j0_checks(_toy_weights(n=10, seed=4))
    assert result["overall_pass"] is True
    assert result["marginal_consistency_pass"] is True
    assert result["used_result_labels"] is False


def test_gate_j0_marginal_errors_at_machine_precision():
    result = BS.gate_j0_checks(_toy_weights(n=18, seed=5))
    for key, value in result["marginal_consistency"].items():
        if key.endswith("_error"):
            assert value < 1e-6, f"{key} too large: {value}"
    assert result["n_states"] == 18 * 17 * 16
