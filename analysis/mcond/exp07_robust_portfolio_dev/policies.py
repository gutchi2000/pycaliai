# -*- coding: utf-8 -*-
"""
policies.py — P0-P7比較政策の実装。既存ソルバー(analysis/robust_ticket_portfolio.py)を
そのままimportして使い、コピー・改変はしない(spec.json "existing_module_reuse"参照)。

このファイルが吸収する既知のギャップ(PRIOR_ART_AUDIT.md §14/§18/§20参照):
  1. Ticketに構成馬identityがない → CandidateTicketに`horses`を追加し、
     同一馬露出上限を事後検証+反復再solveで実装する。
  2. cvar_alphaの規約が仕様書と既存コードで逆 → _to_solver_cvar_alpha()で変換する
     (spec.json "cvar_alpha_convention_warning"参照。仕様書alpha=0.10[最悪10%平均]は
     既存コードへcvar_alpha=1-0.10=0.90として渡す)。
  3. ValueError/RuntimeErrorを自動でno-betへ変換する層がない →
     _solve_or_no_bet()でtry/exceptラップする。

実行: このファイルは単体実行を想定しない。test_synthetic.py等から import して使う。
"""
from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.robust_ticket_portfolio import (  # noqa: E402
    PortfolioResult, Ticket, optimise_portfolio,
)

MAX_EXPOSURE_REBALANCE_ROUNDS = 20


@dataclass(frozen=True)
class CandidateTicket:
    """EXP07レベルの候補馬券。horsesが同一馬露出上限の計算に使われる。"""

    name: str
    horses: tuple[int, ...]
    probability: float
    odds_floor: float
    probability_floor: float | None = None
    state_payoffs: tuple[float, ...] | None = None

    def to_solver_ticket(self, *, max_units_per_ticket: int | None = None) -> Ticket:
        return Ticket(
            name=self.name,
            probability=self.probability,
            odds_floor=self.odds_floor,
            probability_floor=self.probability_floor,
            state_payoffs=self.state_payoffs,
        )


def _to_solver_cvar_alpha(spec_alpha: float) -> float:
    """仕様書の『最悪alpha%平均』規約を既存コードの『信頼水準』規約へ変換する。
    spec.json "cvar_alpha_convention_warning" 参照。"""
    if not 0.0 < spec_alpha < 1.0:
        raise ValueError(f"spec_alpha must be in (0,1), got {spec_alpha}")
    return 1.0 - spec_alpha


def _empty_result() -> PortfolioResult:
    return PortfolioResult({}, 0, 0.0, 0.0, (0.0,), 0.0, (0.0,), 0)


def _solve_or_no_bet(tickets: Sequence[Ticket], **kwargs) -> PortfolioResult:
    """ソルバー呼び出しをラップし、ValueError/RuntimeErrorをno-betへ変換する
    (仕様書Stage1合成テスト項目13『ソルバー失敗時はno-bet』)。"""
    if not tickets:
        return _empty_result()
    try:
        return optimise_portfolio(tickets, **kwargs)
    except (ValueError, RuntimeError):
        return _empty_result()


def _horse_exposure_yen(stakes_yen: dict[str, int], candidates: Sequence[CandidateTicket]) -> dict[int, int]:
    by_name = {c.name: c for c in candidates}
    exposure: dict[int, int] = {}
    for name, stake in stakes_yen.items():
        cand = by_name.get(name)
        if cand is None:
            continue
        for horse in cand.horses:
            exposure[horse] = exposure.get(horse, 0) + stake
    return exposure


def robust_cvar_portfolio(
    candidates: Sequence[CandidateTicket],
    *,
    budget_yen: int,
    bankroll_yen: int,
    spec_cvar_alpha: float = 0.10,
    cvar_penalty: float = 0.0,
    max_cvar_loss_yen: float | None = None,
    min_worst_expected_profit_yen: float = 0.0,
    state_probabilities: Sequence[float] | None = None,
    state_probability_scenarios: Sequence[Sequence[float]] = (),
    unit_yen: int = 100,
    max_units_per_ticket: int | None = None,
    max_exposure_per_horse_yen: int | None = None,
) -> PortfolioResult:
    """P6 ROBUST_CVAR (state_probability_scenariosを渡した場合)、
    またはP5 CVAR (scenariosが空の場合)。同一馬露出上限は反復再solveで実装する
    (既存ソルバーはticket単位のmax_units_per_ticketしか持たないため、
    露出超過が起きたticketの上限を段階的に絞って再solveする簡易ヒューリスティック)。"""
    if not candidates:
        return _empty_result()

    solver_alpha = _to_solver_cvar_alpha(spec_cvar_alpha)
    per_ticket_caps: dict[str, int] = {}
    budget_units = budget_yen // unit_yen

    for _ in range(MAX_EXPOSURE_REBALANCE_ROUNDS):
        solver_tickets = [c.to_solver_ticket() for c in candidates]
        # per_ticket_capsが設定されているチケットにはmax_units_per_ticketを個別適用できない
        # (既存ソルバーは全体で1つのスカラーしか受けないため)、キャップに達したチケットは
        # 候補から一時的に除外して再solveする方式で近似する。
        active = [c for c in candidates if per_ticket_caps.get(c.name, 1) > 0]
        if not active:
            return _empty_result()
        solver_tickets = [c.to_solver_ticket() for c in active]

        result = _solve_or_no_bet(
            solver_tickets,
            budget_yen=budget_yen,
            bankroll_yen=bankroll_yen,
            unit_yen=unit_yen,
            state_probabilities=state_probabilities,
            state_probability_scenarios=state_probability_scenarios,
            min_worst_expected_profit_yen=min_worst_expected_profit_yen,
            cvar_alpha=solver_alpha,
            max_cvar_loss_yen=max_cvar_loss_yen,
            cvar_penalty=cvar_penalty,
            max_units_per_ticket=max_units_per_ticket,
        )

        if max_exposure_per_horse_yen is None or result.is_no_bet:
            return result

        exposure = _horse_exposure_yen(result.stakes_yen, candidates)
        violators = {h for h, yen in exposure.items() if yen > max_exposure_per_horse_yen}
        if not violators:
            return result

        # 露出超過を引き起こしたチケットのうち、購入されているものを次回除外する。
        removed_any = False
        for name in list(result.stakes_yen.keys()):
            cand = next((c for c in active if c.name == name), None)
            if cand is not None and any(h in violators for h in cand.horses):
                per_ticket_caps[name] = 0
                removed_any = True
        if not removed_any:
            # 除外できるチケットがないのに露出超過が続く(理論上到達しないはず)。安全側でno-bet。
            return _empty_result()

    return _empty_result()


def flat_portfolio(
    candidates: Sequence[CandidateTicket], *, budget_yen: int, unit_yen: int = 100,
) -> dict[str, int]:
    """P1 FLAT: 固定された候補へ均等配分(100円単位、端数は切り捨て)。"""
    if not candidates:
        return {}
    n = len(candidates)
    budget_units = budget_yen // unit_yen
    per_ticket_units = budget_units // n
    if per_ticket_units <= 0:
        return {}
    return {c.name: per_ticket_units * unit_yen for c in candidates}


def prob_proportional_portfolio(
    candidates: Sequence[CandidateTicket], *, budget_yen: int, unit_yen: int = 100,
) -> dict[str, int]:
    """P3 PROB: 的中確率比例配分。"""
    if not candidates:
        return {}
    total_p = sum(c.probability for c in candidates)
    if total_p <= 0:
        return {}
    budget_units = budget_yen // unit_yen
    stakes: dict[str, int] = {}
    for c in candidates:
        units = int(budget_units * (c.probability / total_p))
        if units > 0:
            stakes[c.name] = units * unit_yen
    return stakes


def ev_max_portfolio(
    candidates: Sequence[CandidateTicket], *, budget_yen: int, unit_yen: int = 100,
) -> dict[str, int]:
    """P4 EV: 点推定EV最大の1候補へ全額(保守EV<=1の候補は除外)。"""
    positive_ev = [c for c in candidates if c.probability * c.odds_floor > 1.0]
    if not positive_ev:
        return {}
    best = max(positive_ev, key=lambda c: c.probability * c.odds_floor)
    budget_units = budget_yen // unit_yen
    if budget_units <= 0:
        return {}
    return {best.name: budget_units * unit_yen}


def no_bet_portfolio(candidates: Sequence[CandidateTicket], **_kwargs) -> dict[str, int]:
    """P0 NO_BET: 常に購入しない。"""
    return {}


# ============================================================================
# Stage 2A: 予算完全消化の主比較(仕様書追加指示6、2026-09-20夜)
# ============================================================================
# 既存ソルバー(optimise_portfolio)は budget_yen 以下の任意額を許容し、EVが無ければ
# no-betを積極的に選ぶ設計(仕様書7.3のno-bet優先規律に合わせた正しい挙動)。しかし
# Stage 2Aの主比較は「同じ参加・同じ候補・同じ総投資額」で配分方式だけを比べる実験
# であり、sum(w_j)=Bを強制する必要がある。既存ソルバーにこの制約を後付けする安全な
# 方法がない(内部列挙を改変せず=既存コードをコピー・改変しない、という制約と両立
# しない)ため、予算完全消化探索は独立実装する(既存ソルバーの内部関数は一切再利用
# しない、意図的な非共有)。


class Stage2ABudgetAnomaly(Exception):
    """Stage 2A主比較で予算を完全消化できない(候補が薄すぎる等)場合に送出する。
    no-betで黙って逃げず、異常件数としてFAIL判定に計上するための例外。"""


def _full_spend_enumerate(n: int, budget_units: int):
    from itertools import product
    for combo in product(range(budget_units + 1), repeat=n):
        if sum(combo) == budget_units:
            yield combo


def _full_spend_objective(units, candidates: Sequence[CandidateTicket], unit_yen: int,
                          bankroll_yen: int, solver_cvar_alpha: float, cvar_penalty: float,
                          state_probability_scenarios) -> tuple[float, float, float]:
    from math import log1p
    stakes = [u * unit_yen for u in units]

    def _payoff_row(cand: CandidateTicket, state_idx: int | None) -> float:
        if cand.state_payoffs is not None:
            return cand.state_payoffs[state_idx]
        raise ValueError("full_spend_search requires explicit state_payoffs on all candidates")

    n_states = len(candidates[0].state_payoffs)
    profits = []
    for s in range(n_states):
        total = 0.0
        for stake, cand in zip(stakes, candidates):
            total += stake * (_payoff_row(cand, s) - 1.0)
        profits.append(total)

    base_probs = state_probability_scenarios[0] if state_probability_scenarios else None
    if base_probs is None:
        raise ValueError("full_spend_search requires at least one probability scenario")
    scenarios = list(state_probability_scenarios)

    expected_profits = [sum(p * q for p, q in zip(profits, probs)) for probs in scenarios]
    worst_expected_profit = min(expected_profits)

    def _cvar(probs):
        tail_mass = 1.0 - solver_cvar_alpha
        losses = sorted(((max(0.0, -p), q) for p, q in zip(profits, probs)), key=lambda x: -x[0])
        acc_mass, acc_loss = 0.0, 0.0
        for loss, mass in losses:
            take = min(mass, tail_mass - acc_mass)
            if take <= 0:
                break
            acc_loss += loss * take
            acc_mass += take
        return acc_loss / tail_mass if tail_mass > 0 else 0.0

    worst_cvar = max(_cvar(probs) for probs in scenarios)
    log_growths = [
        sum(log1p(p / bankroll_yen) * q for p, q in zip(profits, probs)) for probs in scenarios
    ]
    objective = min(log_growths) - cvar_penalty * worst_cvar / bankroll_yen
    return objective, worst_expected_profit, worst_cvar


def full_spend_search(
    candidates: Sequence[CandidateTicket], *, budget_yen: int, bankroll_yen: int,
    unit_yen: int = 100, spec_cvar_alpha: float = 0.10, cvar_penalty: float = 0.0,
    state_probability_scenarios: Sequence[Sequence[float]] = (),
    max_portfolios: int = 200_000,
) -> dict:
    """Stage 2A主比較専用: sum(w_j)==budget_yen を必須制約とし、候補が
    explicit state_payoffsを持つ前提で独立に全列挙する(既存ソルバーは再利用しない)。
    予算を完全消化できる非負整数配分が一つも無い場合はStage2ABudgetAnomalyを送出する
    (no-betで黙って逃げない、仕様書追加指示6の要求通り)。"""
    if not candidates:
        raise Stage2ABudgetAnomaly("no candidates supplied")
    if any(c.state_payoffs is None for c in candidates):
        raise Stage2ABudgetAnomaly("full_spend_search requires explicit state_payoffs on all candidates")

    budget_units = budget_yen // unit_yen
    n = len(candidates)
    from math import comb
    upper_bound = comb(n + budget_units - 1, n - 1) if n > 0 else 0
    if upper_bound > max_portfolios:
        raise Stage2ABudgetAnomaly(
            f"full-spend grid has up to {upper_bound} points; max_portfolios={max_portfolios}"
        )

    solver_alpha = _to_solver_cvar_alpha(spec_cvar_alpha)
    scenarios = list(state_probability_scenarios) or [None]
    if scenarios[0] is None:
        raise Stage2ABudgetAnomaly("state_probability_scenarios is required for full_spend_search")

    best = None
    evaluated = 0
    for units in _full_spend_enumerate(n, budget_units):
        evaluated += 1
        obj, worst_ep, worst_cvar = _full_spend_objective(
            units, candidates, unit_yen, bankroll_yen, solver_alpha, cvar_penalty, scenarios,
        )
        key = (obj, worst_ep, -worst_cvar, units)
        if best is None or key > best[0]:
            best = (key, units)

    if best is None:
        raise Stage2ABudgetAnomaly("no full-spend allocation found despite passing enumeration bound check")

    units = best[1]
    stakes = {c.name: u * unit_yen for c, u in zip(candidates, units) if u > 0}
    return {
        "stakes_yen": stakes, "total_stake_yen": budget_yen,
        "objective": best[0][0], "worst_expected_profit_yen": best[0][1],
        "worst_cvar_loss_yen": -best[0][2], "evaluated_portfolios": evaluated,
    }


__all__ = [
    "CandidateTicket", "robust_cvar_portfolio", "flat_portfolio",
    "prob_proportional_portfolio", "ev_max_portfolio", "no_bet_portfolio",
    "Stage2ABudgetAnomaly", "full_spend_search",
]
