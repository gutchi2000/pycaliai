# -*- coding: utf-8 -*-
"""2026-11-13 監査の修正: 着順 0 (取消/除外/中止)・同着・返還・凍結版の扱い。"""
import json
from datetime import datetime

import build_site as bs
import bundle_provenance as bp


def _res(order, scratched=(), no_finish=(), pays=None, by_row=None, fuku=None):
    top = sorted(((int(u), p) for u, p in order.items() if p), key=lambda t: t[1])
    p = {"tan": None, "fuku": fuku or {}, "wide": {}, "by_row": by_row or {}}
    p.update(pays or {})
    return {"order": {str(k): v for k, v in order.items()}, "top3": [u for u, q in top if q <= 3],
            "scratched": list(scratched), "no_finish": list(no_finish), "pays": p}


def test_scratched_horse_is_not_first():
    res = _res({9: 0, 3: 1, 5: 2, 7: 3}, scratched=[9], pays={"tan": 480, "tan_by": {"3": 480}})
    assert res["top3"] == [3, 5, 7]
    # 取消馬の単勝は返還 (以前は 1 着扱いで配当が付いていた)
    r = bs.settle_bet("単勝", "9", 1000, res)
    assert r["settled"] and not r["is_win"] and r["received"] == 1000 and r["profit"] == 0
    r = bs.settle_bet("単勝", "3", 1000, res)
    assert r["is_win"] and r["received"] == 4800


def test_unknown_zero_horse_is_unsettled():
    res = _res({4: 0, 3: 1, 5: 2, 7: 3}, no_finish=[4])
    r = bs.settle_bet("複勝", "4", 1000, res)
    assert r["settled"] is False and r["reason"] == "refund_unknown"
    # 中止馬を含まない買い目は普通に決済
    r = bs.settle_bet("馬連", "3-5", 1000, _res({4: 0, 3: 1, 5: 2, 7: 3}, no_finish=[4],
                                                    pays={"umaren": 1230}))
    assert r["is_win"] and r["received"] == 12300


def test_umaren_partial_refund():
    res = _res({9: 0, 3: 1, 5: 2, 7: 3}, scratched=[9], pays={"umaren": 1000})
    r = bs.settle_bet("馬連", "3-9,3-5", 1000, res)
    # 3-9 は返還 (500)、3-5 的中 (500 × 10.0)
    assert r["is_win"] and r["received"] == 500 + 5000 and r["refunded"] == 1


def test_dead_heat_first_both_win_tansho_and_umaren():
    res = _res({2: 1, 6: 1, 8: 3}, pays={"tan_by": {"2": 350, "6": 410}, "umaren": 900},
               by_row={"umaren": {"2": 900, "6": 900}})
    assert bs.settle_bet("単勝", "6", 100, res)["received"] == 410
    assert bs.settle_bet("単勝", "2", 100, res)["received"] == 350
    assert bs.settle_bet("馬連", "2-6", 100, res)["is_win"]
    assert not bs.settle_bet("馬連", "2-8", 100, res)["is_win"]
    assert bs._winning_combos(res, 2, True) == {(2, 6), (6, 2)}


def test_dead_heat_second_two_umaren_winners_with_row_payouts():
    res = _res({1: 1, 4: 2, 5: 2}, pays={"umaren": 700},
               by_row={"umaren": {"4": 700, "5": 1500}})
    assert bs._winning_combos(res, 2, False) == {(1, 4), (1, 5)}
    assert bs.settle_bet("馬連", "1-5", 100, res)["received"] == 1500
    assert bs.settle_bet("馬連", "1-4", 100, res)["received"] == 700


def test_fukusho_needs_payout():
    # 7 頭以下は 3 着に複勝が付かない
    res = _res({1: 1, 2: 2, 3: 3}, fuku={"1": 110, "2": 150})
    assert not bs.settle_bet("複勝", "3", 100, res)["is_win"]
    assert bs.settle_bet("複勝", "2", 100, res)["received"] == 150


def test_model_era():
    assert bp.model_era("20260418", {"model": "v5"}) == "v5"
    assert bp.model_era("20260906", {"model": "v6"}) == "v6-C0"
    assert bp.model_era("20260913", {"model": "v6"}) == "v6-C1"
    assert bp.model_era("20261004", {"model": "v6"}) == "v6-C0"
    assert bp.model_era("20261025", {"model": "v6"}) == "v6-C1"
    sha = "db2ffe65ee27fb9014dcd78724851e77d292fcaf5d77aa33e40b83f3b186d0d5"
    sp = {"model": "v6", "serve_path": {"model": {"sha256": sha}}}
    assert bp.model_era("20261004", sp) == "v6-C1"   # sha が日付表より優先


def test_freeze_only_before_first_post(tmp_path):
    b = tmp_path / "20261115_bundle.json"
    b.write_text(json.dumps({"v": 1}), encoding="utf-8")
    fz = tmp_path / "frozen"
    fp = bp.first_post_time("20261115", ["10:05", "09:50", "nan"])
    assert fp == datetime(2026, 11, 15, 9, 50)
    r = bp.freeze_if_pre_race("20261115", b, fp, now=datetime(2026, 11, 14, 22, 0), frozen_dir=fz)
    assert r["frozen"]
    b.write_text(json.dumps({"v": 2}), encoding="utf-8")
    r = bp.freeze_if_pre_race("20261115", b, fp, now=datetime(2026, 11, 15, 11, 0), frozen_dir=fz)
    assert not r["frozen"] and r["frozen_exists"]
    assert json.loads((fz / "20261115_bundle.json").read_text(encoding="utf-8")) == {"v": 1}
    path, prov = bp.resolve_for_scoring("20261115", live_dir=tmp_path, frozen_dir=fz)
    assert path == fz / "20261115_bundle.json" and prov["verified_pre_race"] and prov["differs_from_live"]
