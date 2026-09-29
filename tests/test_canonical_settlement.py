# -*- coding: utf-8 -*-
"""Regression tests for strict result normalization and settlement."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from canonical_settlement import (  # noqa: E402
    SettlementError,
    canonical_combo,
    normalize_result,
    settle_bets,
)


def _row(
    horse,
    position,
    *,
    win=None,
    place=None,
    quinella=None,
    exacta=None,
    trio=None,
    trifecta=None,
):
    return {
        "馬番": horse,
        "確定着順": position,
        "単勝配当": win,
        "複勝配当": place,
        "馬連": quinella,
        "馬単": exacta,
        "３連複": trio,
        "３連単": trifecta,
    }


def _ordinary_rows():
    return [
        _row(5, 1, win=820, place=210, quinella=1810, exacta=4020, trio=2400, trifecta=15620),
        _row(3, 2, win="(4.6)", place=170, quinella=1810, exacta=4020, trio=2400, trifecta=15620),
        _row(16, 3, win="(4.2)", place=150, trio=2400, trifecta=15620),
        _row(4, 4, win="(15.7)"),
        _row(6, 5, win="(18.2)"),
        _row(7, 6, win="(22.0)"),
        _row(8, 7, win="(7.0)"),
        _row(9, 8, win="(9.0)"),
    ]


def test_normalizes_all_seven_markets_and_ignores_parenthesized_win_odds():
    wide = {"3-5": 250, "5-16": 310, "3-16": 190}
    result = normalize_result(_ordinary_rows(), payout_maps={"ワイド": wide})

    assert result.status == "confirmed"
    assert result.n_starters == 8
    assert result.market("単勝").winning_combos == ((5,),)
    assert dict(result.market("単勝").payout_per100) == {(5,): 820}
    assert dict(result.market("複勝").payout_per100) == {
        (3,): 170,
        (5,): 210,
        (16,): 150,
    }
    assert dict(result.market("ワイド").payout_per100) == {
        (3, 5): 250,
        (5, 16): 310,
        (3, 16): 190,
    }
    assert dict(result.market("馬連").payout_per100) == {(3, 5): 1810}
    assert dict(result.market("馬単").payout_per100) == {(5, 3): 4020}
    assert dict(result.market("三連複").payout_per100) == {(3, 5, 16): 2400}
    assert dict(result.market("三連単").payout_per100) == {(5, 3, 16): 15620}
    assert result.ok


def test_third_place_dead_heat_maps_each_branch_payout():
    rows = [
        _row(4, 1, win=1620, place=390, quinella=32380, exacta=54080, trio=203420, trifecta=1565830),
        _row(8, 2, place=920, quinella=32380, exacta=54080, trio=203420, trifecta=1565830),
        _row(5, 3, place=1010, trio=203420, trifecta=1565830),
        _row(12, 3, place=420, trio=68430, trifecta=408530),
        *[_row(h, pos) for pos, h in enumerate(range(13, 18), start=5)],
    ]
    placed = (4, 5, 8, 12)
    # 権威払戻マップ側も実際のJRA払戻に合わせる: 同着3着どうし (5,12) は発生しない。
    wide = {combo: 100 + i
            for i, combo in enumerate(__import__("itertools").combinations(placed, 2))
            if combo != (5, 12)}

    result = normalize_result(rows, payout_maps={"wide": wide})

    assert result.market("place").winning_combos == ((4,), (5,), (8,), (12,))
    assert dict(result.market("trio").payout_per100) == {
        (4, 5, 8): 203420,
        (4, 8, 12): 68430,
    }
    assert dict(result.market("trifecta").payout_per100) == {
        (4, 8, 5): 1565830,
        (4, 8, 12): 408530,
    }
    # 3着同着では「同着3着どうしの組」は払われない (C(4,2)-1 = 5組)。
    # 実測 (data/kekka/wide_kekka.csv 2026, 該当7R すべて5組):
    #   2026070410020302 は本ケースと同形 (1着4/2着8/3着同着 5,12) で払戻は5組、
    #   欠落は 5-12。他 2026011806010710 / 2026041103010110 / 2026051005020609 も同様。
    assert len(result.market("wide").winning_combos) == 5
    assert (5, 12) not in result.market("wide").winning_combos
    assert result.ok


def test_second_place_dead_heat_resolves_pair_branches_but_not_ordered_triple():
    rows = [
        _row(1, 1, win=690, place=250, quinella=1040, exacta=2010, trio=7180, trifecta=18250),
        _row(3, 2, place=190, quinella=1040, exacta=2010, trio=7180, trifecta=18250),
        _row(7, 2, place=300, quinella=1820, exacta=3520, trio=7180, trifecta=18250),
        *[_row(h, pos) for pos, h in enumerate(range(8, 14), start=4)],
    ]
    wide = {(1, 3): 200, (1, 7): 300, (3, 7): 400}

    result = normalize_result(rows, payout_maps={"wide": wide})

    assert dict(result.market("quinella").payout_per100) == {
        (1, 3): 1040,
        (1, 7): 1820,
    }
    assert dict(result.market("exacta").payout_per100) == {
        (1, 3): 2010,
        (1, 7): 3520,
    }
    assert dict(result.market("trio").payout_per100) == {(1, 3, 7): 7180}
    assert result.market("trifecta").winning_combos == ((1, 3, 7), (1, 7, 3))
    assert result.market("trifecta").status == "unresolved"
    assert "ambiguous" in " ".join(result.market("trifecta").issues)

    resolved = normalize_result(
        rows,
        payout_maps={
            "wide": wide,
            "trifecta": {(1, 3, 7): 18250, (1, 7, 3): 24110},
        },
    )
    assert resolved.market("trifecta").status == "confirmed"
    assert dict(resolved.market("trifecta").payout_per100) == {
        (1, 3, 7): 18250,
        (1, 7, 3): 24110,
    }


def test_first_place_dead_heat_keeps_both_win_tickets_and_fails_closed_on_exact_order():
    rows = [
        _row(5, 1, win=1340, place=690, quinella=60460, exacta=57520, trio=219000, trifecta=867600),
        _row(15, 1, win=1960, place=1010, quinella=60460, exacta=57520, trio=219000, trifecta=867600),
        _row(12, 3, place=380, trio=219000, trifecta=867600),
        *[_row(h, pos) for pos, h in enumerate(range(16, 22), start=4)],
    ]
    wide = {(5, 15): 500, (5, 12): 600, (12, 15): 700}

    result = normalize_result(rows, payout_maps={"wide": wide})

    assert dict(result.market("win").payout_per100) == {(5,): 1340, (15,): 1960}
    assert dict(result.market("quinella").payout_per100) == {(5, 15): 60460}
    assert result.market("exacta").winning_combos == ((5, 15), (15, 5))
    assert result.market("exacta").status == "unresolved"
    assert dict(result.market("trio").payout_per100) == {(5, 12, 15): 219000}
    assert result.market("trifecta").status == "unresolved"


def test_position_zero_is_never_implicitly_refunded():
    rows = _ordinary_rows() + [_row(18, 0)]
    wide = {"3-5": 250, "5-16": 310, "3-16": 190}

    unknown = normalize_result(rows, payout_maps={"wide": wide})
    assert unknown.status == "unresolved"
    assert unknown.refunded_horses == frozenset()
    assert unknown.market("win").status == "unresolved"
    assert "explicit refund classification" in " ".join(unknown.issues)

    refunded = normalize_result(rows, refunded_horses={18}, payout_maps={"wide": wide})
    assert refunded.status == "confirmed"
    refund_settlement = settle_bets("win", {(18,): 300, (5,): 100}, refunded)
    assert refund_settlement["detail"][(18,)]["state"] == "refund"
    assert refund_settlement["refund"] == 300
    assert refund_settlement["total_back"] == 1120

    dnf = normalize_result(rows, non_refunded_zero_horses={18}, payout_maps={"wide": wide})
    assert dnf.status == "confirmed"
    dnf_settlement = settle_bets("win", {(18,): 300}, dnf)
    assert dnf_settlement["detail"][(18,)]["state"] == "lose"
    assert dnf_settlement["refund"] == 0


def test_void_refunds_every_ticket_even_without_payouts():
    result = normalize_result([], race_status="void")
    settled = settle_bets("三連複", {"1-2-3": 400, "4-5-6": 600}, result)

    assert settled["ok"]
    assert settled["stake"] == 1000
    assert settled["refund"] == 1000
    assert settled["total_back"] == 1000
    assert settled["pl"] == 0
    assert settled["roi"] == 1.0


def test_missing_winning_payout_is_unresolved_not_zero_return():
    rows = _ordinary_rows()
    rows[0]["単勝配当"] = None
    result = normalize_result(rows, payout_maps={"wide": {"3-5": 250, "5-16": 310, "3-16": 190}})

    settled = settle_bets("単勝", {(5,): 500}, result)

    assert not settled["ok"]
    assert settled["market_status"] == "unresolved"
    assert settled["total_back"] is None
    assert settled["pl"] is None
    assert settled["roi"] is None
    assert settled["detail"][(5,)]["state"] == "unresolved"


def test_unordered_combos_are_sorted_but_ordered_combos_are_not():
    assert canonical_combo("馬連", "9-2") == (2, 9)
    assert canonical_combo("ワイド", (11, 4)) == (4, 11)
    assert canonical_combo("三連複", "8,1,5") == (1, 5, 8)
    assert canonical_combo("馬単", "9>2") == (9, 2)
    assert canonical_combo("三連単", (8, 1, 5)) == (8, 1, 5)

    with pytest.raises(SettlementError):
        canonical_combo("馬単", "2-2")
    with pytest.raises(SettlementError):
        canonical_combo("三連複", "1-2")


def test_stake_must_be_positive_hundred_yen_increment():
    result = normalize_result(
        _ordinary_rows(),
        payout_maps={"wide": {"3-5": 250, "5-16": 310, "3-16": 190}},
    )
    with pytest.raises(SettlementError):
        settle_bets("win", {(5,): 550}, result)
    with pytest.raises(SettlementError):
        settle_bets("win", {(5,): 0}, result)


def test_wide_pays_three_pairs_with_seven_starters():
    """ワイドは複勝と違い7頭以下でも3着までの組を払う。

    実測 (data/kekka/wide_kekka.csv 2026): 5頭3R / 6頭12R / 7頭26R すべて3組。
    place_slots(=2) を流用すると当たりが1組になり unresolved になっていた。
    例は 2026010408010106 (京都6R, 7頭, 1着=2 2着=4 3着=3, 払戻 2-4 / 2-3 / 3-4)。
    """
    rows = [
        _row(2, 1, win=1000, place=200),
        _row(4, 2, place=300),
        _row(3, 3, place=400),
        *[_row(h, pos) for pos, h in enumerate((1, 5, 6, 7), start=4)],
    ]
    wide = {(2, 4): 2790, (2, 3): 4330, (3, 4): 590}

    result = normalize_result(rows, payout_maps={"wide": wide})

    assert result.market("place").winning_combos == ((2,), (4,))   # 複勝は2着まで
    assert result.market("wide").winning_combos == ((2, 3), (2, 4), (3, 4))
    assert result.market("wide").status != "unresolved"
    assert settle_bets("wide", {"2-3": 100}, result)["hit_return"] == 4330
