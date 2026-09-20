# -*- coding: utf-8 -*-
"""
test_oracle.py — Solverの全列挙oracleテスト(仕様書追加指示3、2026-09-20夜)。

analysis/robust_ticket_portfolio.py の optimise_portfolio() 自体が全列挙ソルバーだが、
「ソルバーの内部関数を再利用しない、独立に書かれた参照実装」で目的値・制約充足を
再計算し、solverの出力と突き合わせる。ロジックを一切共有しないことがオラクルテスト
の価値なので、_reference_* 関数はrobust_ticket_portfolio.py/build_scenarios.pyの
関数を一切importしない(意図的な独立実装)。
"""
from __future__ import annotations
import sys
from itertools import product
from math import log1p
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.robust_ticket_portfolio import Ticket, optimise_portfolio  # noqa: E402

_TOL = 1e-6


def _reference_enumerate_all_allocations(n_tickets: int, budget_units: int) -> list[tuple[int, ...]]:
    """予算(100円単位)以内の全ての非負整数配分を独立に列挙する
    (robust_ticket_portfolio._unit_vectorsは一切使わない)。"""
    allocations = []
    for combo in product(range(budget_units + 1), repeat=n_tickets):
        if sum(combo) <= budget_units:
            allocations.append(combo)
    return allocations


def _reference_profit(units: tuple[int, ...], payoff_matrix: list[list[float]], unit_yen: int) -> list[float]:
    """状態ごとの純損益を独立に計算する。payoff_matrix[state][ticket] = gross multiple。"""
    stakes = [u * unit_yen for u in units]
    profits = []
    for state_row in payoff_matrix:
        total = 0.0
        for stake, mult in zip(stakes, state_row):
            total += stake * (mult - 1.0)
        profits.append(total)
    return profits


def _reference_cvar(profits: list[float], probs: list[float], alpha: float) -> float:
    """upper-tail CVaR を独立実装(ソート方式を変え、累積質量ベースで計算)。"""
    tail_mass = 1.0 - alpha
    losses = sorted(
        ((max(0.0, -p), q) for p, q in zip(profits, probs)),
        key=lambda pair: -pair[0],
    )
    acc_mass = 0.0
    acc_loss = 0.0
    for loss, mass in losses:
        take = min(mass, tail_mass - acc_mass)
        if take <= 0:
            break
        acc_loss += loss * take
        acc_mass += take
    return acc_loss / tail_mass if tail_mass > 0 else 0.0


def _reference_objective(
    units: tuple[int, ...], payoff_matrix: list[list[float]], prob_scenarios: list[list[float]],
    unit_yen: int, bankroll_yen: int, cvar_alpha_solver_convention: float, cvar_penalty: float,
) -> tuple[float, float, float, int]:
    """(objective, worst_expected_profit, worst_cvar, total_stake) を独立計算する。"""
    profits = _reference_profit(units, payoff_matrix, unit_yen)
    total_stake = sum(u * unit_yen for u in units)

    expected_profits = [
        sum(p * q for p, q in zip(profits, probs)) for probs in prob_scenarios
    ]
    worst_expected_profit = min(expected_profits)

    cvars = [_reference_cvar(profits, probs, cvar_alpha_solver_convention) for probs in prob_scenarios]
    worst_cvar = max(cvars)

    log_growths = [
        sum(log1p(p / bankroll_yen) * q for p, q in zip(profits, probs))
        for probs in prob_scenarios
    ]
    objective = min(log_growths) - cvar_penalty * worst_cvar / bankroll_yen
    return objective, worst_expected_profit, worst_cvar, total_stake


def _reference_best(
    n_tickets: int, budget_yen: int, payoff_matrix: list[list[float]], prob_scenarios: list[list[float]],
    unit_yen: int, bankroll_yen: int, cvar_alpha_solver_convention: float, cvar_penalty: float,
    min_worst_expected_profit_yen: float = 0.0, max_cvar_loss_yen: float | None = None,
) -> dict:
    budget_units = budget_yen // unit_yen
    best_obj = None
    best_units = None
    all_best_units = []
    for units in _reference_enumerate_all_allocations(n_tickets, budget_units):
        obj, worst_ep, worst_cvar, total_stake = _reference_objective(
            units, payoff_matrix, prob_scenarios, unit_yen, bankroll_yen,
            cvar_alpha_solver_convention, cvar_penalty,
        )
        if total_stake > 0 and worst_ep + _TOL < min_worst_expected_profit_yen:
            continue
        if max_cvar_loss_yen is not None and worst_cvar > max_cvar_loss_yen + _TOL:
            continue
        if best_obj is None or obj > best_obj + _TOL:
            best_obj = obj
            best_units = units
            all_best_units = [units]
        elif abs(obj - best_obj) <= _TOL:
            all_best_units.append(units)
    return {"best_objective": best_obj, "best_units": best_units, "all_tied_units": all_best_units}


def _solve(tickets, **kwargs):
    return optimise_portfolio(tickets, **kwargs)


# ---- ケース1: 2馬券、予算300円、相互排他的payoff ----
def test_oracle_2_tickets_budget_300_mutually_exclusive():
    tickets = [
        Ticket("a", probability=0.5, odds_floor=2.5, state_payoffs=(2.5, 0.0)),
        Ticket("b", probability=0.5, odds_floor=2.2, state_payoffs=(0.0, 2.2)),
    ]
    payoff_matrix = [[2.5, 0.0], [0.0, 2.2]]
    probs = [0.5, 0.5]
    bankroll = 10_000
    result = _solve(tickets, budget_yen=300, bankroll_yen=bankroll, state_probabilities=probs)
    ref = _reference_best(2, 300, payoff_matrix, [probs], 100, bankroll,
                          cvar_alpha_solver_convention=0.95, cvar_penalty=0.0)
    assert result.objective == pytest.approx(ref["best_objective"], abs=1e-6)


# ---- ケース2: 3馬券、予算500円、同時的中可能 ----
def test_oracle_3_tickets_budget_500_simultaneous():
    # 3状態: 全員的中/2人的中/誰も的中せず、のような同時的中パターン
    tickets = [
        Ticket("a", probability=0.4, odds_floor=1.8, state_payoffs=(1.8, 1.8, 0.0)),
        Ticket("b", probability=0.4, odds_floor=1.9, state_payoffs=(1.9, 0.0, 1.9)),
        Ticket("c", probability=0.4, odds_floor=2.0, state_payoffs=(0.0, 2.0, 2.0)),
    ]
    payoff_matrix = [[1.8, 1.9, 0.0], [1.8, 0.0, 2.0], [0.0, 1.9, 2.0]]
    probs = [0.34, 0.33, 0.33]
    bankroll = 10_000
    result = _solve(tickets, budget_yen=500, bankroll_yen=bankroll, state_probabilities=probs)
    ref = _reference_best(3, 500, payoff_matrix, [probs], 100, bankroll,
                          cvar_alpha_solver_convention=0.95, cvar_penalty=0.0)
    assert result.objective == pytest.approx(ref["best_objective"], abs=1e-6)


# ---- ケース3: 4馬券、予算1000円、点推定CVaR ----
def test_oracle_4_tickets_budget_1000_point_estimate_cvar():
    tickets = [
        Ticket(f"t{i}", probability=0.3 + 0.05 * i, odds_floor=2.0 + 0.3 * i)
        for i in range(4)
    ]
    # marginal(独立ベルヌーイ)を使うケース: state_paymentsなし
    bankroll = 20_000
    result = _solve(tickets, budget_yen=1000, bankroll_yen=bankroll, cvar_alpha=0.90)
    # 独立ベルヌーイ展開は2^4=16状態、solverの_independent_statesと同じ規則を
    # 独立に再実装して照合する。
    n = 4
    states = list(product((0, 1), repeat=n))
    payoff_matrix = []
    probs = []
    for hits in states:
        row = []
        p = 1.0
        for hit, ticket in zip(hits, tickets):
            row.append(ticket.odds_floor if hit else 0.0)
            p *= ticket.probability if hit else (1.0 - ticket.probability)
        payoff_matrix.append(row)
        probs.append(p)
    total_p = sum(probs)
    probs = [p / total_p for p in probs]
    ref = _reference_best(4, 1000, payoff_matrix, [probs], 100, bankroll,
                          cvar_alpha_solver_convention=0.90, cvar_penalty=0.0)
    assert result.objective == pytest.approx(ref["best_objective"], abs=1e-6)


# ---- ケース4: robust CVaR(複数シナリオ) ----
def test_oracle_robust_cvar_multiple_scenarios():
    tickets = [
        Ticket("a", probability=0.5, odds_floor=2.3, state_payoffs=(2.3, 0.0)),
        Ticket("b", probability=0.5, odds_floor=2.1, state_payoffs=(0.0, 2.1)),
    ]
    payoff_matrix = [[2.3, 0.0], [0.0, 2.1]]
    base_probs = [0.5, 0.5]
    scenario2 = [0.35, 0.65]
    bankroll = 10_000
    result = _solve(
        tickets, budget_yen=400, bankroll_yen=bankroll, state_probabilities=base_probs,
        state_probability_scenarios=(scenario2,), cvar_penalty=5.0,
    )
    ref = _reference_best(2, 400, payoff_matrix, [base_probs, scenario2], 100, bankroll,
                          cvar_alpha_solver_convention=0.95, cvar_penalty=5.0)
    assert result.objective == pytest.approx(ref["best_objective"], abs=1e-6)


# ---- ケース5: 露出上限あり(=max_units_per_ticketで近似) ----
def test_oracle_with_max_units_per_ticket_cap():
    tickets = [
        Ticket("a", probability=0.55, odds_floor=2.3),
        Ticket("b", probability=0.55, odds_floor=2.3),
    ]
    bankroll = 10_000
    result = _solve(tickets, budget_yen=600, bankroll_yen=bankroll, max_units_per_ticket=2)
    # 独立ベルヌーイ展開を再現、ただし各ticketの上限2単位(200円)を尊重した参照列挙。
    n = 2
    states = list(product((0, 1), repeat=n))
    payoff_matrix = []
    probs = []
    for hits in states:
        row = []
        p = 1.0
        for hit, ticket in zip(hits, tickets):
            row.append(ticket.odds_floor if hit else 0.0)
            p *= ticket.probability if hit else (1.0 - ticket.probability)
        payoff_matrix.append(row)
        probs.append(p)
    total_p = sum(probs)
    probs = [p / total_p for p in probs]

    budget_units = 6
    best_obj = None
    for units in product(range(3), repeat=2):  # 各ticket最大2単位
        if sum(units) > budget_units:
            continue
        obj, worst_ep, worst_cvar, total_stake = _reference_objective(
            units, payoff_matrix, [probs], 100, bankroll, 0.95, 0.0,
        )
        if best_obj is None or obj > best_obj:
            best_obj = obj
    assert result.objective == pytest.approx(best_obj, abs=1e-6)
    assert all(stake <= 200 for stake in result.stakes_yen.values())


# ---- ケース6: 同値最適解あり(対称ticket) ----
def test_oracle_tied_optimal_solutions_symmetric_tickets():
    tickets = [
        Ticket("a", probability=0.5, odds_floor=2.4, state_payoffs=(2.4, 0.0)),
        Ticket("b", probability=0.5, odds_floor=2.4, state_payoffs=(0.0, 2.4)),
    ]
    payoff_matrix = [[2.4, 0.0], [0.0, 2.4]]
    probs = [0.5, 0.5]
    bankroll = 10_000
    result = _solve(tickets, budget_yen=200, bankroll_yen=bankroll, state_probabilities=probs)
    ref = _reference_best(2, 200, payoff_matrix, [probs], 100, bankroll,
                          cvar_alpha_solver_convention=0.95, cvar_penalty=0.0)
    assert result.objective == pytest.approx(ref["best_objective"], abs=1e-6)
    # solverが選んだ配分は、独立参照実装が見つけた同値最適解集合のいずれかに含まれること
    solver_units = tuple(
        result.stakes_yen.get(t.name, 0) // 100 for t in tickets
    )
    assert solver_units in ref["all_tied_units"], (
        f"solverの解{solver_units}が独立参照実装の同値最適解集合{ref['all_tied_units']}に含まれない"
    )


# ---- 丸め・制約の健全性 ----
def test_oracle_100_yen_rounding_never_creates_constraint_violation():
    tickets = [Ticket("a", probability=0.6, odds_floor=2.0)]
    result = _solve(tickets, budget_yen=350, bankroll_yen=10_000)
    assert result.total_stake_yen <= 350
    assert result.total_stake_yen % 100 == 0


def test_oracle_rounding_does_not_unfairly_improve_objective():
    """100円丸め後の目的値が、実数配分(丸めなし)での理論最大値を超えないことを確認する
    (丸めは常に不利側にのみ働くべき、丸めバグで実数最適を上回ることがあってはならない)。"""
    tickets = [Ticket("a", probability=0.6, odds_floor=2.0)]
    result = _solve(tickets, budget_yen=350, bankroll_yen=10_000)
    # 実数配分(350円全額)での目的値を独立計算
    p, odds = 0.6, 2.0
    profit_win = 350 * (odds - 1.0)
    profit_lose = -350.0
    log_growth_continuous = p * log1p(profit_win / 10_000) + (1 - p) * log1p(profit_lose / 10_000)
    assert result.objective <= log_growth_continuous + 1e-6
