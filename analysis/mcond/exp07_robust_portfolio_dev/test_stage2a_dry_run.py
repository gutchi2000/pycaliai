# -*- coding: utf-8 -*-
"""test_stage2a_dry_run.py — Stage 2Aドライランの純粋ロジック検定。"""
from __future__ import annotations
import sys
from math import comb
from pathlib import Path

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.exp07_robust_portfolio_dev import stage2a_dry_run as DR  # noqa: E402


def test_non_result_cols_excludes_all_known_result_columns():
    result_cols = {"fin", "top3", "win", "fpay", "tan_final_odds"}
    assert result_cols.isdisjoint(set(DR._NON_RESULT_COLS))


def test_fixed_params_ticket_scope_matches_gate_j1_decision():
    assert DR.STAGE2A_FIXED_PARAMS["ticket_types"] == ["tansho", "fukusho"]
    assert "umaren" not in DR.STAGE2A_FIXED_PARAMS["ticket_types"]


def test_fixed_params_budget_and_unit_are_consistent():
    b = DR.STAGE2A_FIXED_PARAMS["budget"]
    assert b["budget_yen_per_race"] % b["unit_yen"] == 0
    assert b["bankroll_yen"] > b["budget_yen_per_race"]


def test_exposure_cap_is_within_budget():
    cap = DR.STAGE2A_FIXED_PARAMS["same_horse_exposure_cap"]
    budget = DR.STAGE2A_FIXED_PARAMS["budget"]["budget_yen_per_race"]
    assert 0 < cap["cap_yen_per_race"] <= budget
    assert cap["cap_yen_per_race"] == int(budget * cap["cap_fraction_of_budget"])


def test_expected_grid_size_matches_documented_bound():
    n_candidates = DR.STAGE2A_FIXED_PARAMS["candidate_generation"]["max_candidates_per_race"]
    budget_units = (DR.STAGE2A_FIXED_PARAMS["budget"]["budget_yen_per_race"]
                    // DR.STAGE2A_FIXED_PARAMS["budget"]["unit_yen"])
    grid_size = comb(n_candidates + budget_units - 1, n_candidates - 1)
    assert grid_size == 3003
    assert grid_size < DR.STAGE2A_FIXED_PARAMS["solver_limits"]["full_spend_search_max_portfolios"]


def test_anomaly_fail_conditions_are_non_empty_and_documented():
    conditions = DR.STAGE2A_FIXED_PARAMS["anomaly_fail_conditions"]
    assert len(conditions) >= 3
    assert all(isinstance(c, str) and len(c) > 5 for c in conditions)
