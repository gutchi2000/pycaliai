# -*- coding: utf-8 -*-
"""a95_equal 本線配線 (compute_bets a95 エンジン / 検証ガード / サイト決済 / T-20 速報) のテスト。
実データ・実ネットワーク不要。本番 policy の engine には依存せず、engine を明示して呼ぶ。"""
import json

import pytest

import a95_engine
import build_site
import validate_cowork_bets as v


def _horses(o1=4.0, n=12):
    ps = [0.30, 0.20, 0.14, 0.09, 0.07, 0.05, 0.04, 0.03, 0.03, 0.02, 0.02, 0.01][:n]
    os_ = [o1, 6.0, 9.0, 12.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 99.0][:n]
    return [dict(umaban=i + 1, p_win=p, p_sho=min(3 * p, 0.9), p_plc=min(2 * p, 0.8),
                 tansho_odds=o, fuku_odds_low=1.2, fuku_odds_high=1.8,
                 mark=("◎" if i == 0 else "〇" if i == 1 else "▲" if i == 2 else ""))
            for i, (p, o) in enumerate(zip(ps, os_))]


# ---------------------------------------------------------------- engine
def test_engine_tickets_and_selection_formats():
    plan = a95_engine.build(_horses(4.0))
    assert plan["band"] == "3-5倍" and plan["band_code"] == "C"
    got = {t["kind_jp"]: (t["selection"], t["stake"]) for t in plan["tickets"]}
    assert got == {"馬連": ("1-2", 2500), "馬単": ("1→2", 2500),
                   "三連複": ("1-2-3", 2500), "三連単": ("1→2→3", 2500)}
    assert sum(t["stake"] for t in plan["tickets"]) == 10000


def test_engine_bands_budget_and_errors():
    assert [t["kind_jp"] for t in a95_engine.build(_horses(1.5))["tickets"]] == ["馬連"]
    five8 = a95_engine.build(_horses(6.0))
    assert [t["kind_jp"] for t in five8["tickets"]] == ["馬単", "ワイド", "枠連", "三連複", "三連単"]
    wk = next(t for t in five8["tickets"] if t["kind_jp"] == "枠連")["selection"]
    assert wk == "1-2"          # 12 頭立て: 馬番1→1枠, 馬番2→2枠
    small = a95_engine.build(_horses(6.0), budget=300)      # 低予算は後ろの券種から落とす
    assert sum(t["stake"] for t in small["tickets"]) == 300 and len(small["tickets"]) == 3
    with pytest.raises(a95_engine.A95Error):
        a95_engine.build(_horses(4.0, n=4))
    hs = _horses(4.0); hs[0]["tansho_odds"] = None
    with pytest.raises(a95_engine.A95Error):
        a95_engine.build(hs)


def test_order_is_raw_pwin_not_marks():
    hs = _horses(4.0)
    hs[0]["mark"], hs[3]["mark"] = "", "◎"       # 印を動かしても p_win 順は変わらない
    plan = a95_engine.build(hs)
    assert (plan["a1"], plan["a2"], plan["a3"]) == (1, 2, 3)


# ---------------------------------------------------------------- compute_bets
def _race(chaos=0.50):
    return {"race_id": "2026100305040101",
            "race_meta": {"field_size": 12, "place": "東京", "R": 1, "course": "芝1600"},
            "race_confidence": {"field_chaos_score": chaos, "top1_dominance": 0.1,
                                "top2_concentration": 0.5, "ai_market_agreement": 0.5},
            "eligibility": None, "horses": _horses(4.0)}


def test_compute_bets_a95_engine_keeps_participation_guards(monkeypatch):
    import compute_bets as cb
    import race_eligibility as re_
    ok = {"bet_eligible": True, "is_jump": False, "reason": "", "determination": "test"}
    monkeypatch.setattr(re_, "verify_metadata", lambda meta, rid: ok)
    from production_policy import load_policy, raw_at_percentile
    limit = float(load_policy()["chaos_reference"]["skip_percentile"])
    low = raw_at_percentile(max(limit - 0.30, 0.01))
    out = cb.compute_race_bets({**_race(), "race_confidence": {**_race()["race_confidence"],
                                                               "field_chaos_score": low}},
                               budget=10000, engine="a95")
    assert out["race_nature"] == "a95"
    assert {b["馬券種"] for b in out["bets"]} == {"馬連", "馬単", "三連複", "三連単"}
    assert sum(b["購入額"] for b in out["bets"]) == 10000
    assert all("倍" not in b["理由"] for b in out["bets"]) and "倍" not in out["race_reason"]
    # 混戦度の hard 見送りは a95 でも維持される (参戦ガードは topdown と同一)
    high = raw_at_percentile(min(limit + 0.20, 0.99))
    skip = cb.compute_race_bets({**_race(), "race_confidence": {**_race()["race_confidence"],
                                                                "field_chaos_score": high}},
                                budget=10000, engine="a95")
    assert skip["race_nature"] == "見送り" and skip["bets"] == []


def test_jump_gate_still_wins_over_a95(monkeypatch):
    import compute_bets as cb
    import race_eligibility as re_
    ng = {"bet_eligible": False, "is_jump": True, "reason": "jump", "determination": "test"}
    monkeypatch.setattr(re_, "verify_metadata", lambda meta, rid: ng)
    monkeypatch.setattr(re_, "log_exclusions", lambda *a, **k: None)
    out = cb.compute_race_bets(_race(), budget=10000, engine="a95")
    assert out["bets"] == [] and out["race_nature"] == "見送り"


# ---------------------------------------------------------------- validator
def test_validator_kinds_follow_policy_engine(monkeypatch):
    valid = set(range(1, 13))
    monkeypatch.setattr(v, "kind_sets", lambda: (set(v.A95_KINDS), set()))
    assert v.content_issues({"馬券種": "馬単", "買い目": "1→2", "購入額": 2500}, valid) == []
    assert v.content_issues({"馬券種": "三連単", "買い目": "1→2→3", "購入額": 2500}, valid) == []
    assert v.content_issues({"馬券種": "枠連", "買い目": "1-2", "購入額": 2000}, valid) == []
    assert v.content_issues({"馬券種": "枠連", "買い目": "3-3", "購入額": 2000}, valid) == []
    assert v.content_issues({"馬券種": "枠連", "買い目": "1-9", "購入額": 2000}, valid)       # 枠番は 1-8
    assert v.content_issues({"馬券種": "馬単", "買い目": "1→1", "購入額": 2500}, valid)       # 重複
    assert v.content_issues({"馬券種": "三連単", "買い目": "1→2→30", "購入額": 2500}, valid)  # 不在馬番
    assert v.content_issues({"馬券種": "馬連", "買い目": "1-2", "購入額": 10100}, valid)      # 1点上限超
    # a95 以外の engine では従来どおり廃止券種を拒否
    monkeypatch.setattr(v, "kind_sets", lambda: (set(v.ALLOWED_KINDS), set(v.REJECTED_KINDS)))
    assert v.content_issues({"馬券種": "馬単", "買い目": "1→2", "購入額": 2500}, valid)
    assert v.content_issues({"馬券種": "枠連", "買い目": "1-2", "購入額": 2000}, valid)


# ---------------------------------------------------------------- site settle
RES = {"top3": [7, 3, 1], "waku": {"7": 4, "3": 2, "1": 1},
       "pays": {"tan": 450, "fuku": {"7": 160, "3": 210, "1": 300}, "wakuren": 900, "umaren": 1200,
                "umatan": 2500, "sanrenpuku": 3000, "sanrentan": 15000, "wide": {"3-7": 400}}}


def test_site_settle_arrow_selections_and_wakuren():
    s = build_site.settle_bet
    assert s("馬単", "7→3", 2500, RES)["received"] == 2500 * 25
    assert s("馬単", "3→7", 2500, RES)["is_win"] is False
    assert s("三連単", "7→3→1", 2500, RES)["received"] == 2500 * 150
    assert s("三連単", "7→1→3", 2500, RES)["is_win"] is False
    assert s("三連複", "1-3-7", 2500, RES)["received"] == 2500 * 30
    assert s("枠連", "2-4", 2000, RES)["received"] == 2000 * 9
    assert s("枠連", "1-4", 2000, RES)["is_win"] is False
    no_waku = dict(RES, waku={})
    assert s("枠連", "2-4", 2000, no_waku)["settled"] is False          # 枠番未取込は集計外


# ---------------------------------------------------------------- T-20 preview
def test_t20_a95_public_tickets_have_no_odds_or_amounts(monkeypatch):
    import t20_site_bets as t20
    import race_eligibility as re_
    import production_policy as pp
    ok = {"bet_eligible": True, "is_jump": False, "reason": "", "determination": "test"}
    monkeypatch.setattr(re_, "verify_metadata", lambda meta, rid: ok)
    monkeypatch.setattr(pp, "hard_skip_reasons", lambda rm, rc, hon: [])
    race = _race()
    market = {"tansho": {str(h["umaban"]): (2.5 if h["umaban"] == 1 else h["tansho_odds"])
                         for h in race["horses"]}}
    bets, why, done = t20.a95_site_tickets(race, market, race["race_id"])
    assert done and [b["type"] for b in bets] == ["馬連", "馬単", "三連複", "三連単"]   # T-20 の 2.5 倍 → 帯B
    assert all(set(b) == {"type", "selection", "reason"} for b in bets)
    blob = json.dumps(bets, ensure_ascii=False) + why
    assert "倍" not in blob and "¥" not in blob
    monkeypatch.setattr(pp, "hard_skip_reasons", lambda rm, rc, hon: ["chaos"])
    bets, why, done = t20.a95_site_tickets(race, market, race["race_id"])
    assert bets == [] and done and "見送り" in why


# ---------------------------------------------------------------- 全レース版 (検証用の併記)
def test_guard_skipped_race_records_would_have_tickets_but_no_bets(monkeypatch):
    import compute_bets as cb
    import race_eligibility as re_
    ok = {"bet_eligible": True, "is_jump": False, "reason": "", "determination": "test"}
    monkeypatch.setattr(re_, "verify_metadata", lambda meta, rid: ok)
    from production_policy import load_policy, raw_at_percentile
    limit = float(load_policy()["chaos_reference"]["skip_percentile"])
    hi = {**_race(), "race_confidence": {**_race()["race_confidence"],
                                         "field_chaos_score": raw_at_percentile(min(limit + 0.2, 0.99))}}
    out = cb.compute_race_bets(hi, budget=10000, engine="a95")
    assert out["bets"] == [] and out["race_nature"] == "見送り"          # ガードはそのまま効く
    aa = out["a95_all"]
    assert aa["guard_passed"] is False and sum(t["stake"] for t in aa["tickets"]) == 10000
    assert [t["kind_jp"] for t in aa["tickets"]] == ["馬連", "馬単", "三連複", "三連単"]
    lo = {**_race(), "race_confidence": {**_race()["race_confidence"],
                                         "field_chaos_score": raw_at_percentile(max(limit - 0.3, 0.01))}}
    out2 = cb.compute_race_bets(lo, budget=10000, engine="a95")
    assert out2["a95_all"]["guard_passed"] is True
    assert {(b["馬券種"], b["買い目"], b["購入額"]) for b in out2["bets"]} == \
        {(t["kind_jp"], t["selection"], t["stake"]) for t in out2["a95_all"]["tickets"]}
    # 他エンジンでは併記しない
    assert "a95_all" not in cb.compute_race_bets(hi, budget=10000, engine="topdown")


def test_jump_race_has_no_would_have_tickets(monkeypatch):
    import compute_bets as cb
    import race_eligibility as re_
    ng = {"bet_eligible": False, "is_jump": True, "reason": "jump", "determination": "test"}
    monkeypatch.setattr(re_, "verify_metadata", lambda meta, rid: ng)
    monkeypatch.setattr(re_, "log_exclusions", lambda *a, **k: None)
    assert "a95_all" not in cb.compute_race_bets(_race(), budget=10000, engine="a95")


def test_wakuren_not_bought_when_not_sold():
    """枠連は 9 頭以上でのみ発売。8 頭以下では落として残りへ均等配分する。"""
    eight = a95_engine.build(_horses(6.0, n=8))
    assert [t["kind_jp"] for t in eight["tickets"]] == ["馬単", "ワイド", "三連複", "三連単"]
    assert [t["stake"] for t in eight["tickets"]] == [2500, 2500, 2500, 2500]
    nine = a95_engine.build(_horses(6.0, n=9))
    assert "枠連" in [t["kind_jp"] for t in nine["tickets"]]
    import shadow_a95 as sa
    pol = sa.load_policy()
    d = sa.build_decisions({"races": [{"race_id": "2026100305040101", "horses": _horses(6.0, n=8)}]}, pol)[0]
    assert "wakuren" not in d["tickets"] and sum(d["tickets"].values()) == 10000
