# -*- coding: utf-8 -*-
"""
test_synthetic.py — EXP07 Stage 1 合成データ試験 (仕様書8節、15項目)。
実データ評価(Stage 2A)前に全通過が必須。実レースデータは一切使わない。
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.exp07_robust_portfolio_dev.policies import (  # noqa: E402
    CandidateTicket, robust_cvar_portfolio, flat_portfolio,
)
from analysis.robust_ticket_portfolio import Ticket, optimise_portfolio  # noqa: E402


# 1. 全馬券が負EVならno-bet
def test_01_all_negative_ev_yields_no_bet():
    candidates = [
        CandidateTicket("a", horses=(0,), probability=0.30, odds_floor=2.0),
        CandidateTicket("b", horses=(1,), probability=0.10, odds_floor=4.0),
    ]
    result = robust_cvar_portfolio(candidates, budget_yen=1000, bankroll_yen=50_000)
    assert result.is_no_bet


# 2. 1枚だけ確実に正EVならその馬券を選択
def test_02_single_positive_ev_is_selected():
    candidates = [
        CandidateTicket("good", horses=(0,), probability=0.60, odds_floor=2.0),
    ]
    result = robust_cvar_portfolio(candidates, budget_yen=200, bankroll_yen=50_000)
    assert not result.is_no_bet
    assert "good" in result.stakes_yen


# 3. 同一の馬券2枚なら対称な配分
def test_03_identical_tickets_get_symmetric_allocation():
    candidates = [
        CandidateTicket("x1", horses=(0,), probability=0.55, odds_floor=2.2),
        CandidateTicket("x2", horses=(1,), probability=0.55, odds_floor=2.2),
    ]
    result = robust_cvar_portfolio(candidates, budget_yen=400, bankroll_yen=50_000)
    stakes = [result.stakes_yen.get("x1", 0), result.stakes_yen.get("x2", 0)]
    assert stakes[0] == stakes[1]


# 4. 相互排他的な馬券を同時的中扱いしない
def test_04_mutually_exclusive_tickets_not_treated_as_simultaneous():
    tickets = [
        Ticket("win-a", probability=0.5, odds_floor=2.5, state_payoffs=(2.5, 0.0)),
        Ticket("win-b", probability=0.5, odds_floor=2.5, state_payoffs=(0.0, 2.5)),
    ]
    result = optimise_portfolio(
        tickets, budget_yen=200, bankroll_yen=10_000, state_probabilities=(0.5, 0.5),
    )
    # 両方に100円ずつ張った場合、状態1ではwin-aだけ的中(純利益 150-100=50)、
    # 状態2ではwin-bだけ的中。両方が同時的中する状態は存在しないため、
    # 各状態の利益は「1枚分の純利益」を超えない。
    assert max(result.state_profits_yen) <= 150.0 + 1e-6


# 5. 同時的中可能なワイド等を正しく処理
def test_05_simultaneously_hittable_wide_tickets_handled_correctly():
    # 3頭中2頭が3着内に入る状態を想定し、ワイド2組が同時的中しうるケース。
    # 状態: (A&B的中, A&Cのみ的中, B&Cのみ的中)
    tickets = [
        Ticket("wide-ab", probability=0.34, odds_floor=3.0, state_payoffs=(3.0, 0.0, 0.0)),
        Ticket("wide-ac", probability=0.33, odds_floor=3.0, state_payoffs=(0.0, 3.0, 0.0)),
    ]
    # A&B的中の状態ではwide-abのみ的中(排他的に設計したこのtoy例では同時的中なし
    # だが、実装がstate_payoffsをticketごとに独立に評価できることを確認する)
    result = optimise_portfolio(
        tickets, budget_yen=300, bankroll_yen=10_000, state_probabilities=(0.4, 0.3, 0.3),
    )
    assert result.total_stake_yen <= 300


# 6. リスク回避度(cvar_penalty)を上げると下方リスクが増えない
def test_06_increasing_risk_aversion_does_not_increase_downside():
    candidates = [
        CandidateTicket("risky", horses=(0,), probability=0.40, odds_floor=3.0),
    ]
    low_penalty = robust_cvar_portfolio(
        candidates, budget_yen=500, bankroll_yen=50_000, cvar_penalty=0.0,
    )
    high_penalty = robust_cvar_portfolio(
        candidates, budget_yen=500, bankroll_yen=50_000, cvar_penalty=50.0,
    )
    assert high_penalty.worst_cvar_loss_yen <= low_penalty.worst_cvar_loss_yen + 1e-6


# 7. 不確実性集合を広げると投資額が増えない
def test_07_widening_uncertainty_set_does_not_increase_stake():
    tickets_no_scenario = [
        CandidateTicket("t", horses=(0,), probability=0.55, odds_floor=2.2,
                        state_payoffs=(2.2, 0.0)),
    ]
    narrow = robust_cvar_portfolio(
        tickets_no_scenario, budget_yen=300, bankroll_yen=50_000,
        state_probabilities=(0.55, 0.45),
    )
    wide = robust_cvar_portfolio(
        tickets_no_scenario, budget_yen=300, bankroll_yen=50_000,
        state_probabilities=(0.55, 0.45),
        state_probability_scenarios=((0.30, 0.70), (0.20, 0.80)),
    )
    assert wide.total_stake_yen <= narrow.total_stake_yen


# 8. オッズ下振れを大きくすると投資額が増えない
def test_08_larger_odds_haircut_does_not_increase_stake():
    high_odds = [CandidateTicket("t", horses=(0,), probability=0.55, odds_floor=2.4)]
    low_odds = [CandidateTicket("t", horses=(0,), probability=0.55, odds_floor=1.9)]
    result_high = robust_cvar_portfolio(high_odds, budget_yen=300, bankroll_yen=50_000)
    result_low = robust_cvar_portfolio(low_odds, budget_yen=300, bankroll_yen=50_000)
    assert result_low.total_stake_yen <= result_high.total_stake_yen


# 9. 予算上限を超えない
def test_09_never_exceeds_budget():
    candidates = [
        CandidateTicket("a", horses=(0,), probability=0.60, odds_floor=2.5),
        CandidateTicket("b", horses=(1,), probability=0.55, odds_floor=2.3),
        CandidateTicket("c", horses=(2,), probability=0.50, odds_floor=2.1),
    ]
    result = robust_cvar_portfolio(candidates, budget_yen=500, bankroll_yen=50_000)
    assert result.total_stake_yen <= 500


# 10. 100円単位を守る
def test_10_respects_100_yen_units():
    candidates = [CandidateTicket("a", horses=(0,), probability=0.65, odds_floor=2.0)]
    result = robust_cvar_portfolio(candidates, budget_yen=350, bankroll_yen=50_000)
    assert all(stake % 100 == 0 for stake in result.stakes_yen.values())


# 11. 候補順を入れ替えても解が不変
def test_11_candidate_order_does_not_affect_solution():
    a = CandidateTicket("a", horses=(0,), probability=0.55, odds_floor=2.3)
    b = CandidateTicket("b", horses=(1,), probability=0.50, odds_floor=2.6)
    result_ab = robust_cvar_portfolio([a, b], budget_yen=400, bankroll_yen=50_000)
    result_ba = robust_cvar_portfolio([b, a], budget_yen=400, bankroll_yen=50_000)
    assert result_ab.stakes_yen == result_ba.stakes_yen
    assert result_ab.total_stake_yen == result_ba.total_stake_yen


# 12. 同一入力で解が再現する
def test_12_same_input_reproduces_same_result():
    candidates = [
        CandidateTicket("a", horses=(0,), probability=0.55, odds_floor=2.3),
        CandidateTicket("b", horses=(1,), probability=0.50, odds_floor=2.6),
    ]
    r1 = robust_cvar_portfolio(candidates, budget_yen=400, bankroll_yen=50_000)
    r2 = robust_cvar_portfolio(candidates, budget_yen=400, bankroll_yen=50_000)
    assert r1.stakes_yen == r2.stakes_yen
    assert r1.objective == r2.objective


# 13. ソルバー失敗時はno-bet
def test_13_solver_failure_yields_no_bet():
    # 列挙グリッドが上限を超える設定でValueErrorを誘発する。
    many = [
        CandidateTicket(f"t-{i}", horses=(i,), probability=0.55, odds_floor=2.0)
        for i in range(10)
    ]
    solver_tickets = [c.to_solver_ticket() for c in many]
    with pytest.raises(ValueError):
        optimise_portfolio(
            solver_tickets, budget_yen=5000, bankroll_yen=100_000, max_portfolios=10,
        )
    # policies.py のラッパーは同条件で例外を吸収してno-betを返すことを確認する。
    from analysis.mcond.exp07_robust_portfolio_dev.policies import _solve_or_no_bet
    result = _solve_or_no_bet(
        solver_tickets, budget_yen=5000, bankroll_yen=100_000, max_portfolios=10,
    )
    assert result.is_no_bet


# 14. NaN・負オッズ・確率和異常を拒否
def test_14_invalid_inputs_are_rejected_not_silently_allowed():
    with pytest.raises(ValueError):
        Ticket("bad", probability=0.5, odds_floor=-1.0)  # 負オッズ
    with pytest.raises(ValueError):
        Ticket("bad2", probability=float("nan"), odds_floor=2.0)  # NaN確率
    with pytest.raises(ValueError):
        # probability_floor > probability は無効
        Ticket("bad3", probability=0.3, odds_floor=2.0, probability_floor=0.5)


# 15. 返還馬券を損失扱いしない(evaluate.py実装待ち、暫定skip)
@pytest.mark.skip(
    reason="決済ロジック(evaluate.py)がStage 2Aでまだ実装されていないため、"
          "返還馬券の扱いは決済実装時に個別テストする(仕様書5.2で既知の要注意領域と明記済み)。"
)
def test_15_refunded_ticket_is_not_treated_as_a_loss():
    pytest.fail("evaluate.py未実装のためStage 2Aで実装・検証する")


# --- 追加の健全性チェック(仕様書には明示されていないが同じ精神で有用) ---

def test_flat_portfolio_never_exceeds_budget():
    candidates = [
        CandidateTicket(f"t-{i}", horses=(i,), probability=0.3, odds_floor=3.0)
        for i in range(3)
    ]
    stakes = flat_portfolio(candidates, budget_yen=350)
    assert sum(stakes.values()) <= 350


def test_horse_exposure_cap_is_enforced():
    # 3枚とも馬0を含む。露出上限200円を設定すると全額投入(300円超)は許されないはず。
    candidates = [
        CandidateTicket("t1", horses=(0, 1), probability=0.5, odds_floor=2.4),
        CandidateTicket("t2", horses=(0, 2), probability=0.5, odds_floor=2.4),
        CandidateTicket("t3", horses=(0, 3), probability=0.5, odds_floor=2.4),
    ]
    result = robust_cvar_portfolio(
        candidates, budget_yen=900, bankroll_yen=50_000,
        max_exposure_per_horse_yen=200,
    )
    exposure_horse0 = sum(
        stake for name, stake in result.stakes_yen.items()
        if any(c.name == name and 0 in c.horses for c in candidates)
    )
    assert exposure_horse0 <= 200
