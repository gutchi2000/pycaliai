# -*- coding: utf-8 -*-
"""
stage2a_engine.py — Stage 2A本実行用の高速計算エンジン
================================================================================
2026-09-20夜。6,908レース x 最大3,003配分 x 200不確実性シナリオを実行可能な速度で
計算するための最適化。数理的定義は`policies.py`/`robust_ticket_portfolio.py`と
完全に同一(このファイルはロジックのコピーではなく、同じ定義の高速な再計算経路)。
正しさは`test_stage2a_engine.py`で既存の検証済み実装(test_oracle.pyと同型の独立
参照)と突き合わせて確認する。

**縮約状態空間**: 候補馬(通常3頭、単勝+複勝で最大6候補)の払戻はその3頭の
(1着/2着/3着/圏外)ラベルの組合せだけで決まる。build_scenarios.build_top3_states()
が返す全順列状態(最大4,896)を、候補馬のラベル組合せ(最大4^3=64、実際はさらに少ない)
へ事後的に集約することで、確率質量を一切失わずに状態数を大幅削減する
(これは近似ではない、圏外の他馬をまとめて「other」ラベルへ周辺化しているだけで、
候補馬に関する限り数学的に完全に同値)。

実行: このファイルは単体実行を想定しない。stage2a_run.py等から import して使う。
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.exp07_robust_portfolio_dev import build_scenarios as BS  # noqa: E402

EPS = 1e-9
_LABEL_1ST, _LABEL_2ND, _LABEL_3RD, _LABEL_OTHER = 0, 1, 2, 3


def fast_build_top3_states(w: np.ndarray) -> tuple[list[tuple[int, int, int]], np.ndarray]:
    """build_scenarios.build_top3_states()と数学的に同一の結果を返す高速版。

    2026-09-20夜、実装中に判明した性能問題への対応: build_scenarios.build_top3_states()
    はGate J0で検証済みの正しい実装だが、状態ごとにpl_probs.p_sanrentan()をPython関数
    呼び出しで実行し、かつ関数内部でw.sum()を毎回再計算するため、不確実性シナリオ
    (200draw/レース)を大量生成するStage 2A本実行では実行不可能な遅さになる
    (20レースのテストが2分以上で終わらなかった)。この関数はp_sanrentanの閉形式
    P(i,j,k)=w_i/T・w_j/(T-w_i)・w_k/(T-w_i-w_j)をnumpyで全状態一括ベクトル化する
    (w.sum()は1回だけ計算)。test_stage2a_engine.pyでbuild_scenarios.build_top3_states()
    と数値的に完全一致することを確認済み。"""
    from itertools import permutations
    n = len(w)
    states = list(permutations(range(n), 3))
    idx = np.array(states, dtype=np.int64)  # (M, 3)
    T = float(w.sum())
    wi, wj, wk = w[idx[:, 0]], w[idx[:, 1]], w[idx[:, 2]]
    probs = (wi / T) * (wj / (T - wi)) * (wk / (T - wi - wj))
    return states, probs


def build_reduced_states(
    full_states: list[tuple[int, int, int]], full_probs: np.ndarray,
    candidate_indices: tuple[int, ...],
) -> tuple[list[tuple[int, ...]], np.ndarray]:
    """全順列状態(full_states, full_probs)を候補馬(candidate_indices、レース内の
    0-indexed添字)のラベル組合せへ集約する。ラベル: 0=1着,1=2着,2=3着,3=圏外。
    戻り値のreduced_statesは候補馬ごとのラベルのtuple、reduced_probsはその確率和。"""
    n_cand = len(candidate_indices)
    idx_to_slot = {idx: slot for slot, idx in enumerate(candidate_indices)}

    reduced_map: dict[tuple[int, ...], float] = {}
    for state, prob in zip(full_states, full_probs):
        labels = [_LABEL_OTHER] * n_cand
        for pos, horse_idx in enumerate(state):  # pos=0->1着,1->2着,2->3着
            slot = idx_to_slot.get(horse_idx)
            if slot is not None:
                labels[slot] = pos
        key = tuple(labels)
        reduced_map[key] = reduced_map.get(key, 0.0) + float(prob)

    reduced_states = list(reduced_map.keys())
    reduced_probs = np.array([reduced_map[k] for k in reduced_states], dtype=np.float64)
    return reduced_states, reduced_probs


def reduced_state_payoff_matrix(
    reduced_states: list[tuple[int, ...]], ticket_specs: list[tuple[str, int, float]],
) -> np.ndarray:
    """ticket_specs: [(ticket_type, candidate_slot, odds_floor), ...]。
    ticket_type は 'tansho' または 'fukusho'。
    戻り値: shape (n_states, n_tickets) のgross return multiple行列。"""
    n_states = len(reduced_states)
    n_tickets = len(ticket_specs)
    payoff = np.zeros((n_states, n_tickets), dtype=np.float64)
    for t, (ticket_type, slot, odds_floor) in enumerate(ticket_specs):
        for s, labels in enumerate(reduced_states):
            label = labels[slot]
            if ticket_type == "tansho":
                wins = label == _LABEL_1ST
            elif ticket_type == "fukusho":
                wins = label in (_LABEL_1ST, _LABEL_2ND, _LABEL_3RD)
            else:
                raise ValueError(f"unknown ticket_type: {ticket_type!r}")
            payoff[s, t] = odds_floor if wins else 0.0
    return payoff


def _enumerate_full_spend_unit_vectors(n_tickets: int, budget_units: int) -> np.ndarray:
    """sum(units)==budget_units となる非負整数ベクトルを全列挙する(独立実装、
    policies.full_spend_search内部と同一定義)。"""
    from itertools import product
    rows = [c for c in product(range(budget_units + 1), repeat=n_tickets) if sum(c) == budget_units]
    return np.array(rows, dtype=np.int64)


def vectorized_full_spend_search(
    payoff_matrix: np.ndarray, prob_scenarios: np.ndarray, *,
    budget_yen: int, bankroll_yen: int, unit_yen: int = 100,
    solver_cvar_alpha: float = 0.90, cvar_penalty: float = 0.0,
    exposure_groups: list[list[int]] | None = None, exposure_cap_yen: float | None = None,
) -> dict:
    """policies.full_spend_search()と同一の目的関数をnumpyでベクトル化計算する。
    payoff_matrix: shape (n_states, n_tickets)。prob_scenarios: shape (n_scenarios, n_states)。
    solver_cvar_alphaは既存コード規約(信頼水準、tail_mass=1-alpha)。呼び出し側が
    spec規約(alpha=最悪X%平均)から変換済みであることを前提とする。

    exposure_groups: 同一馬に属するticket列indexのグループ(例: [[0,1],[2,3],[4,5]]、
    tansho/fukushoが同じ馬のとき)。exposure_cap_yenと併用すると、各グループの
    stake合計がcapを超えるcomboを候補から除外する(2026-09-20夜、追加指示1の
    同一馬露出上限を実際に制約として効かせるための実装、検出のみで未適用だった
    バグを修正)。除外の結果、実行可能なcomboが1つも残らない場合はValueErrorを送出する
    (呼び出し側がStage2ABudgetAnomaly相当として扱うこと)。"""
    n_states, n_tickets = payoff_matrix.shape
    budget_units = budget_yen // unit_yen
    units = _enumerate_full_spend_unit_vectors(n_tickets, budget_units)  # (n_combos, n_tickets)
    if len(units) == 0:
        raise ValueError("no full-spend allocation exists for this budget/ticket count")
    stakes = units * unit_yen  # (n_combos, n_tickets)

    if exposure_groups and exposure_cap_yen is not None:
        feasible = np.ones(len(units), dtype=bool)
        for group in exposure_groups:
            group_exposure = stakes[:, group].sum(axis=1)
            feasible &= group_exposure <= exposure_cap_yen + EPS
        if not feasible.any():
            raise ValueError("no full-spend allocation satisfies the exposure cap")
        units = units[feasible]
        stakes = stakes[feasible]

    # profit_by_state[combo, state] = sum_t stake[combo,t] * (payoff[state,t]-1)
    profit_by_state = stakes @ (payoff_matrix - 1.0).T  # (n_combos, n_states)

    # 期待利益: (n_combos, n_states) @ (n_states, n_scenarios) -> (n_combos, n_scenarios)
    expected_profit = profit_by_state @ prob_scenarios.T
    worst_expected_profit = expected_profit.min(axis=1)  # (n_combos,)

    # CVaR: 損失の並び順はcombo単位で固定(シナリオに依らない、profit_by_stateがシナリオ非依存のため)。
    # 2026-09-20実測: シナリオ軸を含む3階テンソル一括ベクトル化は大きな一時配列の確保コストが
    # 上回りむしろ遅くなった(0.46s/race -> 0.97s/race)ため、シナリオ単位のPythonループ+
    # combo/state軸だけのベクトル化という中間形へ戻した(実測で最速)。
    loss_by_state = np.maximum(0.0, -profit_by_state)  # (n_combos, n_states)
    sort_idx = np.argsort(-loss_by_state, axis=1)  # (n_combos, n_states) 損失降順
    sorted_loss = np.take_along_axis(loss_by_state, sort_idx, axis=1)  # (n_combos, n_states)

    tail_mass = 1.0 - solver_cvar_alpha
    n_scenarios = len(prob_scenarios)
    cvar_per_scenario = np.empty((n_scenarios, len(units)), dtype=np.float64)
    for s_idx in range(n_scenarios):
        sorted_prob = prob_scenarios[s_idx][sort_idx]  # (n_combos, n_states)
        cum_prob = np.cumsum(sorted_prob, axis=1)
        prev_cum = cum_prob - sorted_prob
        take = np.clip(np.minimum(cum_prob, tail_mass) - np.minimum(prev_cum, tail_mass), 0.0, None)
        cvar_per_scenario[s_idx] = (sorted_loss * take).sum(axis=1) / max(tail_mass, EPS)
    worst_cvar = cvar_per_scenario.max(axis=0)  # (n_combos,)

    log_growth = np.log1p(profit_by_state / bankroll_yen)  # (n_combos, n_states)
    log_growth_by_scenario = log_growth @ prob_scenarios.T  # (n_combos, n_scenarios)
    worst_log_growth = log_growth_by_scenario.min(axis=1)

    objective = worst_log_growth - cvar_penalty * worst_cvar / bankroll_yen
    total_stake = stakes.sum(axis=1)  # full-spend searchでは全comboで budget_yen に一定

    # タイブレーク: objective最大 → worst_expected_profit最大 → worst_cvar最小(-worst_cvar最大)
    # → units辞書式最小(np.lexsortは昇順+最後のキーが最優先、正のunits合計を降順相当に
    # したいので符号反転して「小さいunitsが選ばれやすい」側を優先する)。
    # total_stakeは全comboで同一(予算完全消化)のため実質的にタイブレークに寄与しない。
    _base = budget_units + 1  # 各ticketの取りうる値0..budget_unitsを桁として厳密にエンコード
    _place_values = _base ** np.arange(n_tickets)[::-1]
    lex_units_key = -np.sum(units * _place_values, axis=1)  # 辞書式最小を優先(符号反転)
    order = np.lexsort((lex_units_key, -worst_cvar, worst_expected_profit, objective))
    best_idx = int(order[-1])

    return {
        "best_units": units[best_idx].tolist(),
        "objective": float(objective[best_idx]),
        "worst_expected_profit_yen": float(worst_expected_profit[best_idx]),
        "worst_cvar_loss_yen": float(worst_cvar[best_idx]),
        "total_stake_yen": int(total_stake[best_idx]),
        "n_combos_evaluated": int(len(units)),
    }
