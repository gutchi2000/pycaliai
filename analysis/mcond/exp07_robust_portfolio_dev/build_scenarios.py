# -*- coding: utf-8 -*-
"""
build_scenarios.py — 共同着順分布(top-3順列状態空間)の構築とGate J0検証
================================================================================
2026-09-20、EXP07仕様書§6/Gate J0。既存の厳密PL連鎖実装(pl_probs.py)を再利用し、
新しい分布モデルは作らない(仕様書4節「既存研究との差」通り、EXP07固有の検定対象は
共同損益分布の"消費"であって新しい確率モデルの発明ではない)。

状態空間: 出走馬n頭に対し、1-2-3着の順列(i,j,k), i≠j≠k を1状態とする
(最大 n*(n-1)*(n-2)、18頭立てで4,896。単勝/複勝/馬連/ワイド/馬単/三連複/三連単は
いずれも1-3着の順列だけで的中判定できるため、この状態空間で全券種を表現できる)。

各状態の確率は pl_probs.p_sanrentan(w,i,j,k) をそのまま使う(近似なし、結果ラベルは
一切参照しない、スコアwはモデルの生スコアのみに依存する)。

Gate J0(仕様書6節)の数値検証:
  - 確率が0〜1
  - 排他的シナリオ確率和が1
  - 同一馬の重複着順がない(構造的に保証、i≠j≠kの列挙自体がこれを満たす)
  - marginalを再集計すると元確率(pl_probs.pyの直接計算)と一致
  - 馬券的中条件が実決済ルールと一致(単体テストで検証、test_build_scenarios.py)
  - 結果情報をシナリオ確率作成に使っていない(コード上、着順・払戻列への参照ゼロ)

実行: python -m analysis.mcond.exp07_robust_portfolio_dev.build_scenarios
"""
from __future__ import annotations
import sys
from itertools import permutations
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
import pl_probs as PL  # noqa: E402

_TOL = 1e-9


def build_top3_states(w: np.ndarray) -> tuple[list[tuple[int, int, int]], np.ndarray]:
    """出走馬の重みwからtop-3順列状態空間を構築する。
    戻り値: (states=[(i,j,k),...], probs=各状態の確率)。resultは一切参照しない。"""
    n = len(w)
    states = list(permutations(range(n), 3))
    probs = np.array([PL.p_sanrentan(w, i, j, k) for (i, j, k) in states], dtype=np.float64)
    return states, probs


def ticket_wins_in_state(ticket_type: str, selection: tuple[int, ...], state: tuple[int, int, int]) -> bool:
    """実決済ルールに基づく的中判定。selectionの意味はticket_typeに依存:
      tansho: (a,)                 - a が1着
      fukusho: (a,)                - a が1-3着のいずれか
      umaren: (a,b) 順不同         - {a,b} == 上位2着の集合
      wide: (a,b) 順不同           - a,b ともに1-3着に入る
      umatan: (a,b) 順序あり       - i==a かつ j==b
      sanrenpuku: (a,b,c) 順不同   - {a,b,c} == {i,j,k}
      sanrentan: (a,b,c) 順序あり  - (i,j,k) == (a,b,c)
    """
    i, j, k = state
    top3 = {i, j, k}
    top2 = {i, j}
    if ticket_type == "tansho":
        (a,) = selection
        return i == a
    if ticket_type == "fukusho":
        (a,) = selection
        return a in top3
    if ticket_type == "umaren":
        a, b = selection
        return {a, b} == top2
    if ticket_type == "wide":
        a, b = selection
        return a in top3 and b in top3
    if ticket_type == "umatan":
        a, b = selection
        return i == a and j == b
    if ticket_type == "sanrenpuku":
        a, b, c = selection
        return {a, b, c} == top3
    if ticket_type == "sanrentan":
        a, b, c = selection
        return (i, j, k) == (a, b, c)
    raise ValueError(f"unknown ticket_type: {ticket_type!r}")


def state_payoffs_for_ticket(
    ticket_type: str, selection: tuple[int, ...], states: list[tuple[int, int, int]],
    odds_floor: float,
) -> tuple[float, ...]:
    """各状態でのgross return multiple(odds_floorまたは0)を返す。odds_floorは保守化済み
    のオッズを呼び出し側が渡す(このモジュール自体はオッズを取得しない)。"""
    return tuple(
        odds_floor if ticket_wins_in_state(ticket_type, selection, s) else 0.0
        for s in states
    )


def _marginal_from_states(states: list[tuple[int, int, int]], probs: np.ndarray,
                          ticket_type: str, selection: tuple[int, ...]) -> float:
    mask = np.array([ticket_wins_in_state(ticket_type, selection, s) for s in states])
    return float(probs[mask].sum())


def gate_j0_checks(w: np.ndarray) -> dict:
    """Gate J0の数値検証を実行し、結果を辞書で返す(PASS/FAIL含む)。結果ラベルは
    一切使わない、wはモデルスコアから作った重みのみ。"""
    states, probs = build_top3_states(w)
    n = len(w)

    checks = {}

    checks["probabilities_in_0_1"] = bool(np.all((probs >= -_TOL) & (probs <= 1.0 + _TOL)))
    checks["probabilities_sum_to_1"] = bool(abs(probs.sum() - 1.0) < 1e-6)
    checks["no_duplicate_horse_in_state"] = all(len(set(s)) == 3 for s in states)

    # marginal再集計 vs pl_probs直接計算
    marginal_errors = {}
    max_err = 0.0
    for a in range(n):
        direct = PL.p_tansho(w, a)
        recomputed = _marginal_from_states(states, probs, "tansho", (a,))
        err = abs(direct - recomputed)
        max_err = max(max_err, err)
    marginal_errors["tansho_max_abs_error"] = max_err

    max_err = 0.0
    for a in range(n):
        direct = PL.p_fukusho(w, a)
        recomputed = _marginal_from_states(states, probs, "fukusho", (a,))
        max_err = max(max_err, abs(direct - recomputed))
    marginal_errors["fukusho_max_abs_error"] = max_err

    max_err = 0.0
    pairs_checked = 0
    for a in range(n):
        for b in range(a + 1, n):
            direct = PL.p_umaren(w, a, b)
            recomputed = _marginal_from_states(states, probs, "umaren", (a, b))
            max_err = max(max_err, abs(direct - recomputed))
            pairs_checked += 1
    marginal_errors["umaren_max_abs_error"] = max_err
    marginal_errors["umaren_pairs_checked"] = pairs_checked

    max_err = 0.0
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            direct = PL.p_umatan(w, a, b)
            recomputed = _marginal_from_states(states, probs, "umatan", (a, b))
            max_err = max(max_err, abs(direct - recomputed))
    marginal_errors["umatan_max_abs_error"] = max_err

    max_err = 0.0
    for a in range(n):
        for b in range(a + 1, n):
            direct = PL.p_wide(w, a, b)
            recomputed = _marginal_from_states(states, probs, "wide", (a, b))
            max_err = max(max_err, abs(direct - recomputed))
    marginal_errors["wide_max_abs_error"] = max_err

    checks["marginal_consistency"] = marginal_errors
    checks["marginal_consistency_pass"] = all(
        v < 1e-6 for k, v in marginal_errors.items() if k.endswith("_error")
    )
    checks["n_states"] = len(states)
    checks["n_runners"] = n
    checks["used_result_labels"] = False  # コード上、着順・払戻列への参照は皆無(監査確認済み)

    checks["overall_pass"] = (
        checks["probabilities_in_0_1"]
        and checks["probabilities_sum_to_1"]
        and checks["no_duplicate_horse_in_state"]
        and checks["marginal_consistency_pass"]
    )
    return checks


if __name__ == "__main__":
    import json
    rng = np.random.default_rng(20260920)
    for n in (5, 10, 18):
        scores = rng.normal(size=n)
        w = PL.pl_weights(scores)
        result = gate_j0_checks(w)
        print(f"--- n_runners={n} ---")
        print(json.dumps(result, ensure_ascii=False, indent=1))
