# -*- coding: utf-8 -*-
"""test_evaluate.py — 決済ロジックの合成テスト(仕様書§9/追加指示5、11項目)。"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.exp07_robust_portfolio_dev.evaluate import (  # noqa: E402
    RaceOutcome, SettlementError, settle_portfolio, settle_ticket,
)

RID = "2023010401010101"


def _outcome(**kwargs) -> RaceOutcome:
    defaults = dict(race_id=RID)
    defaults.update(kwargs)
    return RaceOutcome(**defaults)


# 1. 通常的中
def test_normal_hit():
    outcome = _outcome(payout_per_100_yen={("tansho", (3,)): 350})
    r = settle_ticket("tansho", (3,), 200, outcome)
    assert r["status"] == "hit"
    assert r["payout_yen"] == 700
    assert r["profit_yen"] == 500


# 2. 通常不的中
def test_normal_miss():
    outcome = _outcome(payout_per_100_yen={("tansho", (3,)): 350})
    r = settle_ticket("tansho", (7,), 200, outcome)
    assert r["status"] == "miss"
    assert r["payout_yen"] == 0
    assert r["profit_yen"] == -200


# 3. 取消返還
def test_cancellation_refund():
    outcome = _outcome(scratched_horses=frozenset({5}), payout_per_100_yen={})
    r = settle_ticket("tansho", (5,), 300, outcome)
    assert r["status"] == "refunded"
    assert r["payout_yen"] == 300
    assert r["profit_yen"] == 0  # 損失扱いにしない(仕様書項目15の核心)


def test_cancellation_refund_applies_if_any_selected_horse_scratched_umaren():
    outcome = _outcome(scratched_horses=frozenset({5}), payout_per_100_yen={})
    r = settle_ticket("umaren", (5, 8), 300, outcome)
    assert r["status"] == "refunded"
    assert r["profit_yen"] == 0


# 4. 全額返還
def test_full_race_void_refund():
    outcome = _outcome(race_void=True, payout_per_100_yen={})
    r = settle_ticket("tansho", (3,), 500, outcome)
    assert r["status"] == "void"
    assert r["payout_yen"] == 500
    assert r["profit_yen"] == 0


# 5. 同着(公式払戻テーブルの値をそのまま使い、独自計算しないことを確認)
def test_dead_heat_uses_official_split_payout_verbatim():
    # 同着による分割で通常より低い払戻(公式テーブル値)を想定
    outcome = _outcome(payout_per_100_yen={("tansho", (1,)): 120, ("tansho", (2,)): 120})
    r1 = settle_ticket("tansho", (1,), 100, outcome)
    r2 = settle_ticket("tansho", (2,), 100, outcome)
    assert r1["payout_yen"] == 120
    assert r2["payout_yen"] == 120


# 6. 複勝払戻レンジ(Lo/Hiではなく確定した実払戻を使うことを確認)
def test_fukusho_uses_confirmed_payout_not_a_range():
    outcome = _outcome(payout_per_100_yen={("fukusho", (4,)): 180})
    r = settle_ticket("fukusho", (4,), 400, outcome)
    assert r["payout_yen"] == 720
    assert r["profit_yen"] == 320


# 7. ワイド複数的中(同一レース内で複数のワイド券が独立に的中できる)
def test_wide_multiple_simultaneous_hits():
    # 公式払戻テーブルには的中組合せのみが載る(非的中の組合せはキーとして存在しない、
    # 実際のJRA払戻表と同じ構造)。(2,3)は非的中のためテーブルに含めない。
    outcome = _outcome(payout_per_100_yen={
        ("wide", (1, 2)): 250, ("wide", (1, 3)): 400,
    })
    stakes = {
        ("wide", (1, 2)): 100, ("wide", (1, 3)): 100, ("wide", (2, 3)): 100,
    }
    result = settle_portfolio(stakes, outcome)
    assert result["per_ticket"][("wide", (1, 2))]["status"] == "hit"
    assert result["per_ticket"][("wide", (1, 3))]["status"] == "hit"
    assert result["per_ticket"][("wide", (2, 3))]["status"] == "miss"
    assert result["total_payout_yen"] == 250 + 400 + 0
    assert result["total_stake_yen"] == 300


def test_explicit_zero_payout_is_treated_as_miss_not_hit():
    # 呼び出し側が非的中組合せを明示的に0で埋めた場合でも、財務的にmissと同一に扱う
    # (防御的措置、payout tableの構造に依存しすぎない)。
    outcome = _outcome(payout_per_100_yen={("wide", (2, 3)): 0})
    r = settle_ticket("wide", (2, 3), 100, outcome)
    assert r["status"] == "miss"
    assert r["profit_yen"] == -100


def test_wide_selection_order_does_not_matter():
    outcome = _outcome(payout_per_100_yen={("wide", (1, 2)): 250})
    r = settle_ticket("wide", (2, 1), 100, outcome)
    assert r["status"] == "hit"
    assert r["payout_yen"] == 250


# 8. 100円単位
def test_stake_must_be_multiple_of_100():
    outcome = _outcome(payout_per_100_yen={("tansho", (1,)): 200})
    with pytest.raises(SettlementError):
        settle_ticket("tansho", (1,), 150, outcome)
    with pytest.raises(SettlementError):
        settle_ticket("tansho", (1,), 0, outcome)
    with pytest.raises(SettlementError):
        settle_ticket("tansho", (1,), -100, outcome)


# 9. 払戻端数(100円単位以外のstakeでも整数除算で正確に計算されること)
def test_payout_remainder_is_exact_integer_arithmetic():
    outcome = _outcome(payout_per_100_yen={("tansho", (1,)): 235})  # 端数のある払戻
    r = settle_ticket("tansho", (1,), 300, outcome)
    assert r["payout_yen"] == 300 // 100 * 235
    assert r["payout_yen"] == 705


# 10. 欠損結果
def test_missing_result_is_not_settled_as_win_or_loss():
    outcome = _outcome(is_missing=True, payout_per_100_yen={})
    r = settle_ticket("tansho", (1,), 200, outcome)
    assert r["status"] == "missing"
    assert r["payout_yen"] is None
    assert r["profit_yen"] is None


def test_settle_portfolio_excludes_missing_from_totals():
    outcome = _outcome(is_missing=True, payout_per_100_yen={})
    stakes = {("tansho", (1,)): 100}
    result = settle_portfolio(stakes, outcome)
    assert result["any_missing"] is True
    assert result["total_stake_yen"] == 0  # missingは集計へ含めない


# 11. 不正なrace_id拒否
def test_invalid_race_id_is_rejected():
    with pytest.raises(SettlementError):
        RaceOutcome(race_id="not-a-valid-id")
    with pytest.raises(SettlementError):
        RaceOutcome(race_id="12345")  # 桁数不足


def test_settle_ticket_rejects_outcome_bypassing_dataclass_validation():
    # RaceOutcomeを直接構築せず__post_init__を回避しても、settle_ticket側で再検証する
    outcome = RaceOutcome.__new__(RaceOutcome)
    object.__setattr__(outcome, "race_id", "bad-id")
    object.__setattr__(outcome, "race_void", False)
    object.__setattr__(outcome, "scratched_horses", frozenset())
    object.__setattr__(outcome, "payout_per_100_yen", {})
    object.__setattr__(outcome, "is_missing", False)
    with pytest.raises(SettlementError):
        settle_ticket("tansho", (1,), 100, outcome)
