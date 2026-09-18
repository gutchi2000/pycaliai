# -*- coding: utf-8 -*-
"""学生大会 自動投票 (masters_vote.py) のペイロード生成・検証テスト。

公式仕様の要点をここで固定する:
  - race_id は netkeiba 12 桁 (年4 + 場2 + 回2 + 日2 + R2)
  - ワイド バラ買いは b5_c0_{小}_{大} (数字の小さい方が前)
  - mark は {"馬番": 印番号} で 1=◎ 2=〇 3=▲ 4=△、◎必須、◎〇▲は各1頭
  - 投票締切は発走 3 分前
"""
from __future__ import annotations

import pytest

import masters_vote as mv


def horse(ban, rank, mark=""):
    return {"umaban": ban, "ai_rank": rank, "mark": mark}


HORSES = [
    horse(1, 1, "◎"), horse(2, 2, "〇"), horse(3, 3, "▲"),
    horse(4, 4, "△"), horse(5, 5, "△"), horse(6, 6, ""), horse(7, 7, ""),
]
ALL = {1, 2, 3, 4, 5, 6, 7}


# ------------------------------------------------------------ race_id
def test_netkeiba_race_id_drops_monthday():
    # 2026 / 0801 / 04(新潟) / 02(回) / 03(日) / 01(R)
    assert mv.netkeiba_race_id("2026080104020301") == "202604020301"


def test_netkeiba_race_id_passes_through_12digit():
    # netkeiba の race_id をそのまま貼った場合 (中京1R 2026-08-29 で実地確認)
    assert mv.netkeiba_race_id("202607030301") == "202607030301"


def test_netkeiba_race_id_rejects_short():
    with pytest.raises(mv.VoteError):
        mv.netkeiba_race_id("20260801")


def test_payload_uses_api_confirmed_bet_key():
    """API が受理したのは "bet"。"bet_list" だと「bet項目不明」で全件拒否される。"""
    assert mv.BET_KEY == "bet"


# ------------------------------------------------------------ bet_id
@pytest.mark.parametrize("sel,expect", [
    ("3-7", "b5_c0_3_7"),
    ("14-16", "b5_c0_14_16"),
    ("10-1", "b5_c0_1_10"),      # 小さい方を前にソートする
])
def test_wide_bet_id(sel, expect):
    assert mv.wide_bet_id(sel) == expect


def test_wide_bet_id_rejects_same_horse():
    with pytest.raises(mv.VoteError):
        mv.wide_bet_id("5-5")


# ------------------------------------------------------------ mark
def test_build_marks_uses_bundle_marks():
    assert mv.build_marks(HORSES, ALL) == {"1": 1, "2": 2, "3": 3, "4": 4, "5": 4}


def test_build_marks_promotes_when_hon_scratched():
    """◎ が取消なら 〇→◎、▲→〇 と繰り上げる (印なし馬をいきなり ◎ にしない)。"""
    marks = mv.build_marks(HORSES, ALL - {1})
    assert marks["2"] == 1 and marks["3"] == 2
    assert sorted(marks.values()).count(1) == 1


def test_build_marks_always_has_hon():
    for scratched in ({1}, {1, 2}, {1, 2, 3}):
        marks = mv.build_marks(HORSES, ALL - scratched)
        assert 1 in marks.values(), scratched
        for code in (1, 2, 3):
            assert list(marks.values()).count(code) == 1


def test_build_marks_needs_active_horses():
    with pytest.raises(mv.VoteError):
        mv.build_marks(HORSES, set())


# ------------------------------------------------------------ bet_list
def test_build_bet_list_ok():
    bets = mv.build_bet_list([{"selection": "3-7"}, {"selection": "1-2"}],
                             5000, ALL, 200)
    assert bets == [{"bet_id": "b5_c0_3_7", "money": 5000},
                    {"bet_id": "b5_c0_1_2", "money": 5000}]


def test_build_bet_list_rejects_scratched_horse():
    with pytest.raises(mv.VoteError):
        mv.build_bet_list([{"selection": "3-7"}], 5000, ALL - {7}, 200)


@pytest.mark.parametrize("stake", [0, 50, 150, 99])
def test_build_bet_list_rejects_bad_stake(stake):
    with pytest.raises(mv.VoteError):
        mv.build_bet_list([{"selection": "3-7"}], stake, ALL, 200)


def test_build_bet_list_dedupes():
    bets = mv.build_bet_list([{"selection": "3-7"}, {"selection": "7-3"}],
                             1000, ALL, 200)
    assert len(bets) == 1


def test_build_bet_list_enforces_point_cap():
    with pytest.raises(mv.VoteError):
        mv.build_bet_list([{"selection": "1-2"}, {"selection": "3-4"}],
                          1000, ALL, 1)


# ------------------------------------------------------------ アーム選択
def _shadow(passed=True, triggered=False, arm=None, m1=None):
    return {"hard_gate_passed": passed, "hard_gate_reasons": [] if passed else ["chaos"],
            "triggered": triggered, "arm_a": arm or [], "control_m1": m1 or []}


def test_select_tickets_uses_arm_a_when_triggered():
    rows, why = mv.select_tickets(_shadow(triggered=True, arm=[{"selection": "1-2"}]),
                                  {"filler": None})
    assert why == "residual" and len(rows) == 1


def test_select_tickets_skips_when_gate_blocked():
    rows, why = mv.select_tickets(_shadow(passed=False), {"filler": "model_top2"})
    assert rows == [] and why.startswith("hard_gate:")


def test_select_tickets_ignore_hard_gate_uses_no_gate_candidates():
    rows, why = mv.select_tickets(
        _shadow(passed=False),
        {"ignore_hard_gate": True, "_no_gate_candidates": [{"selection": "4-9"}]})
    assert why == "residual(no-gate)" and rows == [{"selection": "4-9"}]


def test_select_tickets_ignore_hard_gate_still_needs_residual_band():
    rows, why = mv.select_tickets(_shadow(passed=False),
                                  {"ignore_hard_gate": True, "_no_gate_candidates": []})
    assert rows == [] and "gate外" in why


def test_select_tickets_filler_off_by_default():
    rows, why = mv.select_tickets(_shadow(m1=[{"selection": "1-2"}]), {"filler": None})
    assert rows == [] and "残差" in why


def test_select_tickets_filler_uses_m1():
    rows, why = mv.select_tickets(_shadow(m1=[{"selection": "1-2"}]),
                                  {"filler": "model_top2"})
    assert why == "filler:model_top2" and len(rows) == 1


# ------------------------------------------------------------ 投票後確認
# 買い目キーは 2026-08-29 に実 API で確定 ("bet"。"bet_list" は拒否された)
PAYLOAD = {"race_id": "202604020301", "mark": {"1": 1},
           mv.BET_KEY: [{"bet_id": "b5_c0_1_10", "money": 5000}]}


def test_verify_accepts_matching_response():
    rows = [{"race_id": "202604020301",
             "bet": [{"bet_id": "b5_c0_1_10", "money": "5000"}]}]
    assert mv.verify(PAYLOAD, rows) is True


@pytest.mark.parametrize("rows", [
    [],                                                                    # 未反映
    [{"race_id": "202604020302",                                           # 別レース
      "bet": [{"bet_id": "b5_c0_1_10", "money": "5000"}]}],
    [{"race_id": "202604020301",                                           # 金額違い
      "bet": [{"bet_id": "b5_c0_1_10", "money": "1000"}]}],
    [{"race_id": "202604020301",                                           # 買い目違い
      "bet": [{"bet_id": "b5_c0_1_9", "money": "5000"}]}],
    [{"race_id": "202604020301",                                           # 余分な券
      "bet": [{"bet_id": "b5_c0_1_10", "money": "5000"},
              {"bet_id": "b5_c0_2_3", "money": "5000"}]}],
])
def test_verify_rejects_mismatch(rows):
    assert mv.verify(PAYLOAD, rows) is False


# ------------------------------------------------------------ 締切
def test_post_datetime_parses_override():
    assert mv.post_datetime("20260829", "2026082905030201", "15:40").hour == 15


def test_default_config_matches_official_deadline():
    cfg = mv.DEFAULT_CONFIG
    # 発走 4 分前に取得し、締切 (発走 3 分前) より前に送信し切る余裕があること
    assert cfg["fetch_lead_min"] > 3.0
    assert 0 < cfg["submit_safety_sec"] < 60
