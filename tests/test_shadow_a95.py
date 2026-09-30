# -*- coding: utf-8 -*-
"""shadow_a95 (a95_equal_v1 前向き shadow 台帳) の単体テスト。実データ・実ネットワーク不要。"""
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import shadow_a95 as m


@pytest.fixture()
def pol():
    return m.load_policy()


def test_policy_sha_self_check_and_tamper(tmp_path, pol):
    assert pol["policy_id"] == "a95_equal_v1" and pol["real_money"] is False
    bad = dict(pol); bad["cap_per_race_yen"] = 20000
    p = tmp_path / "p.json"; p.write_text(json.dumps(bad, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(SystemExit):
        m.load_policy(p)


def test_equal_stakes_sum_and_grid(pol):
    for band, kinds in pol["plan"].items():
        st = m.equal_stakes(kinds, pol["cap_per_race_yen"], pol["unit_yen"])
        assert sum(st.values()) == 10000, band
        assert all(v % 100 == 0 and v >= 100 for v in st.values())
        assert max(st.values()) - min(st.values()) <= 100      # 均等 (端数だけ差)
    assert m.equal_stakes(["tan", "umaren", "umatan"], 10000, 100) == {"tan": 3400, "umaren": 3300, "umatan": 3300}


def test_band_edges(pol):
    f = lambda o: m.band_of(o, pol)
    assert [f(1.9), f(2.0), f(2.99), f(3.0), f(5.0), f(7.9), f(8.0), f(14.9), f(15.0), f(300.0)] == \
        ["<2倍", "2-3倍", "2-3倍", "3-5倍", "5-8倍", "5-8倍", "8-15倍", "8-15倍", "15倍〜", "15倍〜"]


def test_waku_rule():
    assert [m.waku_of(i, 8) for i in range(1, 9)] == list(range(1, 9))
    assert [m.waku_of(i, 9) for i in (7, 8, 9)] == [7, 8, 8]
    assert [m.waku_of(i, 10) for i in (6, 7, 8, 9, 10)] == [6, 7, 7, 8, 8]
    assert [m.waku_of(i, 16) for i in (1, 2, 15, 16)] == [1, 1, 8, 8]
    assert [m.waku_of(i, 17) for i in (13, 14, 15, 16, 17)] == [7, 7, 8, 8, 8]
    assert [m.waku_of(i, 18) for i in (12, 13, 15, 16, 18)] == [6, 7, 7, 8, 8]


def _k(first=3, second=5, third=1, wide=None):
    return dict(valid=True, first=first, second=second, third=third, w1=2, w2=3,
                tan=450.0, fuku={3: 160.0, 5: 210.0, 1: 300.0}, wakuren=900.0, umaren=1200.0,
                umatan=2500.0, sanpuku=3000.0, sanrentan=15000.0,
                wide=({"3-5": 400, "1-3": 700, "1-5": 900} if wide is None else wide))


def _d(a1, a2, a3, tickets, band="3-5倍"):
    return dict(rid="2026100301010101", valid=True, a1=a1, a2=a2, a3=a3, waku1=2, waku2=3, band=band,
                o1=4.0, tickets=tickets)


def test_payout_exact_order_vs_set():
    k = _k()
    hit = _d(3, 5, 1, {})
    assert m.payout_per_100("umatan", hit, k) == 2500.0
    assert m.payout_per_100("sanrentan", hit, k) == 15000.0
    rev = _d(5, 3, 1, {})                      # 1-2 着が逆
    assert m.payout_per_100("umatan", rev, k) == 0.0
    assert m.payout_per_100("umaren", rev, k) == 1200.0
    assert m.payout_per_100("sanpuku", rev, k) == 3000.0
    assert m.payout_per_100("sanrentan", rev, k) == 0.0
    assert m.payout_per_100("wide", rev, k) == 400.0
    assert m.payout_per_100("tan", rev, k) == 0.0 and m.payout_per_100("fuku", rev, k) == 210.0


def test_missing_wide_invalidates_only_arms_that_need_it(pol):
    k = _k(); k["wide"] = None
    d = _d(3, 5, 1, m.equal_stakes(pol["plan"]["5-8倍"], 10000, 100), band="5-8倍")
    r = m.settle_race(d, k, pol)
    assert r["arms"]["a95_equal"]["valid"] is False          # 5-8倍 はワイドを買う
    assert r["arms"]["b3"]["valid"] is False
    assert r["arms"]["umatan_1pt"]["valid"] is True and r["arms"]["umatan_1pt"]["ret"] == 2500.0


def test_settle_scales_by_stake(pol):
    d = _d(3, 5, 1, m.equal_stakes(pol["plan"]["3-5倍"], 10000, 100))
    r = m.settle_race(d, _k(), pol)
    a = r["arms"]["a95_equal"]
    assert a["stake"] == 10000 and a["hit"] is True
    # 馬連 1200 + 馬単 2500 + 三連複 3000 + 三連単 15000 を各 2,500 円
    assert a["ret"] == pytest.approx(25 * (1200 + 2500 + 3000 + 15000))


def test_dead_heat_and_invalid_decision(pol):
    assert m.settle_race(_d(3, 5, 1, {"umaren": 10000}), dict(valid=False, reason="dead_heat_or_missing_top3"), pol)["valid"] is False
    assert m.settle_race(dict(rid="x", valid=False, reason="no_odds"), _k(), pol)["reason"] == "no_odds"
    assert m.settle_race(_d(3, 5, 1, {"umaren": 10000}), None, pol)["reason"] == "no_kekka"


def _bundle(odds1=4.0):
    hs = [dict(umaban=i + 1, p_win=p, tansho_odds=o) for i, (p, o) in enumerate(
        [(0.30, odds1), (0.20, 6.0), (0.15, 9.0), (0.10, 12.0), (0.08, 20.0), (0.07, 30.0)])]
    return {"races": [{"race_id": "2026100301010101", "horses": hs}]}


def test_build_decisions_uses_raw_pwin_order_and_band(pol):
    d = m.build_decisions(_bundle(4.0), pol)[0]
    assert (d["a1"], d["a2"], d["a3"]) == (1, 2, 3) and d["band"] == "3-5倍"
    assert d["tickets"] == {"umaren": 2500, "umatan": 2500, "sanpuku": 2500, "sanrentan": 2500}
    assert d["band_o2"] == "5-8倍" and d["band_o3"] == "8-15倍"      # 記述列
    nod = m.build_decisions(_bundle(None), pol)[0]
    assert nod["valid"] is False and nod["reason"] == "no_odds"


def test_decide_never_overwrites_and_flags_late(tmp_path, monkeypatch, pol):
    monkeypatch.setattr(m, "LEDGER", tmp_path / "ledger")
    monkeypatch.setattr(m, "BUNDLE_DIR", tmp_path)
    past = (datetime.now(m.JST) - timedelta(days=3)).strftime("%Y%m%d")
    future = (datetime.now(m.JST) + timedelta(days=3)).strftime("%Y%m%d")
    for dt in (past, future):
        (tmp_path / f"{dt}_bundle.json").write_text(json.dumps(_bundle()), encoding="utf-8")
        assert m.cmd_decide(dt, pol) == 0
    late = json.loads((tmp_path / "ledger" / "decisions" / f"{past}.json").read_text(encoding="utf-8"))
    pre = json.loads((tmp_path / "ledger" / "decisions" / f"{future}.json").read_text(encoding="utf-8"))
    assert late["preregistered"] is False and pre["preregistered"] is True
    # bundle を書き換えても decisions は変わらない
    (tmp_path / f"{future}_bundle.json").write_text(json.dumps(_bundle(20.0)), encoding="utf-8")
    assert m.cmd_decide(future, pol) == 0
    again = json.loads((tmp_path / "ledger" / "decisions" / f"{future}.json").read_text(encoding="utf-8"))
    assert again == pre
