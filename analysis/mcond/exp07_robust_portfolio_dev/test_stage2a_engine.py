# -*- coding: utf-8 -*-
"""
test_stage2a_engine.py — 高速ベクトル化エンジンの正しさ検証。
既存の検証済み実装(policies.full_spend_search、test_oracle.pyの独立参照実装と
同型の定義)と突き合わせ、目的値が一致することを確認する。
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.exp07_robust_portfolio_dev import stage2a_engine as ENG  # noqa: E402
from analysis.mcond.exp07_robust_portfolio_dev import build_scenarios as BS  # noqa: E402
from analysis.mcond.exp07_robust_portfolio_dev.policies import (  # noqa: E402
    CandidateTicket, full_spend_search,
)
import pl_probs as PL  # noqa: E402


def test_fast_build_top3_states_matches_reference_implementation():
    rng = np.random.default_rng(42)
    for n in (5, 10, 16):
        w = PL.pl_weights(rng.normal(size=n))
        ref_states, ref_probs = BS.build_top3_states(w)
        fast_states, fast_probs = ENG.fast_build_top3_states(w)
        assert fast_states == ref_states
        assert np.allclose(fast_probs, ref_probs, atol=1e-12)


def test_fast_build_top3_states_sums_to_one():
    rng = np.random.default_rng(7)
    w = PL.pl_weights(rng.normal(size=12))
    _, probs = ENG.fast_build_top3_states(w)
    assert probs.sum() == pytest.approx(1.0, abs=1e-9)


def test_reduced_states_preserve_total_probability_mass():
    rng = np.random.default_rng(0)
    w = PL.pl_weights(rng.normal(size=8))
    states, probs = BS.build_top3_states(w)
    reduced_states, reduced_probs = ENG.build_reduced_states(states, probs, (0, 1, 2))
    assert reduced_probs.sum() == pytest.approx(1.0, abs=1e-9)


def test_reduced_states_preserve_marginal_tansho_probability():
    rng = np.random.default_rng(1)
    w = PL.pl_weights(rng.normal(size=10))
    states, probs = BS.build_top3_states(w)
    candidate_indices = (2, 5, 7)
    reduced_states, reduced_probs = ENG.build_reduced_states(states, probs, candidate_indices)

    # 縮約後、候補馬0(元の添字2)が1着になる確率の和は、直接計算したp_tanshoと一致するはず
    direct_p = PL.p_tansho(w, candidate_indices[0])
    reduced_p = sum(
        p for labels, p in zip(reduced_states, reduced_probs) if labels[0] == ENG._LABEL_1ST
    )
    assert reduced_p == pytest.approx(direct_p, abs=1e-9)


def test_vectorized_engine_matches_reference_full_spend_search():
    rng = np.random.default_rng(2)
    n_horses = 8
    w = PL.pl_weights(rng.normal(size=n_horses))
    states, probs = BS.build_top3_states(w)
    candidate_indices = (0, 1, 2)  # 3頭、単勝+複勝=6候補

    reduced_states, reduced_probs = ENG.build_reduced_states(states, probs, candidate_indices)

    odds_floor = [2.5, 3.0, 4.0]
    ticket_specs = []
    candidates = []
    for slot, ban in enumerate(candidate_indices):
        ticket_specs.append(("tansho", slot, odds_floor[slot]))
        candidates.append(CandidateTicket(
            f"tansho:{ban}", horses=(ban,), probability=0.1, odds_floor=odds_floor[slot],
        ))
        ticket_specs.append(("fukusho", slot, odds_floor[slot] * 0.4))
        candidates.append(CandidateTicket(
            f"fukusho:{ban}", horses=(ban,), probability=0.3, odds_floor=odds_floor[slot] * 0.4,
        ))

    payoff_matrix = ENG.reduced_state_payoff_matrix(reduced_states, ticket_specs)

    # 参照実装(policies.full_spend_search)用にstate_payoffsを候補ごとに構築
    for c, (ttype, slot, odds) in zip(candidates, ticket_specs):
        state_payoffs = tuple(payoff_matrix[:, ticket_specs.index((ttype, slot, odds))])
        object.__setattr__(c, "state_payoffs", state_payoffs)

    budget_yen, bankroll_yen = 500, 20_000
    ref = full_spend_search(
        candidates, budget_yen=budget_yen, bankroll_yen=bankroll_yen,
        state_probability_scenarios=[reduced_probs.tolist()],
    )

    fast = ENG.vectorized_full_spend_search(
        payoff_matrix, reduced_probs.reshape(1, -1),
        budget_yen=budget_yen, bankroll_yen=bankroll_yen,
        solver_cvar_alpha=0.90, cvar_penalty=0.0,
    )

    assert fast["objective"] == pytest.approx(ref["objective"], abs=1e-6)
    assert fast["total_stake_yen"] == ref["total_stake_yen"] == budget_yen


def test_vectorized_engine_respects_exposure_cap():
    rng = np.random.default_rng(9)
    n_horses = 6
    w = PL.pl_weights(rng.normal(size=n_horses))
    states, probs = BS.build_top3_states(w)
    candidate_indices = (0, 1, 2)
    reduced_states, reduced_probs = ENG.build_reduced_states(states, probs, candidate_indices)

    odds_floor = [3.0, 3.0, 3.0]
    # tickets: [tansho0, fukusho0, tansho1, fukusho1, tansho2, fukusho2]
    ticket_specs = []
    for slot in range(3):
        ticket_specs.append(("tansho", slot, odds_floor[slot]))
        ticket_specs.append(("fukusho", slot, odds_floor[slot] * 0.4))
    payoff_matrix = ENG.reduced_state_payoff_matrix(reduced_states, ticket_specs)

    exposure_groups = [[0, 1], [2, 3], [4, 5]]  # horse0=tickets0,1; horse1=tickets2,3; horse2=tickets4,5
    budget_yen = 1000
    cap_yen = 400

    result = ENG.vectorized_full_spend_search(
        payoff_matrix, reduced_probs.reshape(1, -1),
        budget_yen=budget_yen, bankroll_yen=50_000,
        solver_cvar_alpha=0.90, cvar_penalty=0.0,
        exposure_groups=exposure_groups, exposure_cap_yen=cap_yen,
    )
    units = result["best_units"]
    stakes = [u * 100 for u in units]
    for group in exposure_groups:
        group_exposure = sum(stakes[i] for i in group)
        assert group_exposure <= cap_yen + 1e-6


def test_vectorized_engine_exposure_cap_raises_when_infeasible():
    rng = np.random.default_rng(10)
    w = PL.pl_weights(rng.normal(size=5))
    states, probs = BS.build_top3_states(w)
    reduced_states, reduced_probs = ENG.build_reduced_states(states, probs, (0,))
    ticket_specs = [("tansho", 0, 2.5)]
    payoff_matrix = ENG.reduced_state_payoff_matrix(reduced_states, ticket_specs)
    with pytest.raises(ValueError, match="exposure cap"):
        ENG.vectorized_full_spend_search(
            payoff_matrix, reduced_probs.reshape(1, -1),
            budget_yen=1000, bankroll_yen=50_000,
            exposure_groups=[[0]], exposure_cap_yen=1,  # 予算全額(1000円)が1円上限を必ず超える
        )


def test_vectorized_engine_matches_reference_with_uncertainty_scenarios():
    rng = np.random.default_rng(3)
    n_horses = 6
    w = PL.pl_weights(rng.normal(size=n_horses))
    states, probs = BS.build_top3_states(w)
    candidate_indices = (0, 1, 2)
    reduced_states, reduced_probs = ENG.build_reduced_states(states, probs, candidate_indices)

    # 摂動シナリオを2つ追加(base + 2 draws)
    w2 = PL.pl_weights(rng.normal(size=n_horses))
    states2, probs2 = BS.build_top3_states(w2)
    _, reduced_probs2 = ENG.build_reduced_states(states2, probs2, candidate_indices)
    assert states2 == states  # 頭数不変なので状態列挙順は同一のはず

    scenarios = np.vstack([reduced_probs, reduced_probs2])

    odds_floor = [2.2, 2.8, 3.5]
    ticket_specs = [("tansho", i, odds_floor[i]) for i in range(3)]
    candidates = []
    for slot in range(3):
        c = CandidateTicket(f"t{slot}", horses=(slot,), probability=0.2, odds_floor=odds_floor[slot])
        payoff_col = ENG.reduced_state_payoff_matrix(reduced_states, [ticket_specs[slot]])[:, 0]
        object.__setattr__(c, "state_payoffs", tuple(payoff_col))
        candidates.append(c)

    payoff_matrix = ENG.reduced_state_payoff_matrix(reduced_states, ticket_specs)

    budget_yen, bankroll_yen = 300, 15_000
    ref = full_spend_search(
        candidates, budget_yen=budget_yen, bankroll_yen=bankroll_yen,
        state_probability_scenarios=[reduced_probs.tolist(), reduced_probs2.tolist()],
        cvar_penalty=2.0,
    )
    fast = ENG.vectorized_full_spend_search(
        payoff_matrix, scenarios, budget_yen=budget_yen, bankroll_yen=bankroll_yen,
        solver_cvar_alpha=0.90, cvar_penalty=2.0,
    )
    assert fast["objective"] == pytest.approx(ref["objective"], abs=1e-6)
    assert fast["worst_cvar_loss_yen"] == pytest.approx(ref["worst_cvar_loss_yen"], abs=1e-4)
