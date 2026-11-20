"""board_combo (サイト公開用 GO × ボード一致 × ◎オッズ帯, 2026-11-21) のテスト。"""
import json

import board_combo as bc
import build_site as B


def _race(cat="go", hard="固い", marks=("◎", "〇", "▲", "△", "△", "")):
    hs = [{"umaban": i + 1, "mark": m, "p_win": 0.3 - 0.03 * i} for i, m in enumerate(marks)]
    return {"race_id": "2026112105050101", "buy_judgment": {"category": cat, "hardness": hard}, "horses": hs}


def _site(si=1, pw=1, er=1, n=6):
    hs = []
    for b in range(1, n + 1):
        hs.append({"umaban": b,
                   "speed": {"v": 90.0 if b == si else 80.0, "er": 1 if b == er else 5},
                   "pw": 99.0 if b == pw else 50.0})
    return {"race_id": "2026112105050101", "horses": hs}


def test_policy_hash_and_cells():
    pol = bc.load_policy()
    assert len(pol["cells"]) == 27
    assert all(c["band"] != "15倍〜" for c in pol["cells"])


def test_agree3_tan_in_band_b():
    out = bc.build(_race(cat="none", hard="荒れ"), 2.5, _site(), "T-10")
    # 一致3（GO不問）× 単勝 × 2-3倍 だけ当たる
    assert [b["type"] for b in out["bets"]] == ["単勝"]
    assert out["bets"][0]["selection"] == "1"
    assert out["agree"] == 3 and out["band_code"] == "B"


def test_agree3_tan_band_c_is_skip():
    out = bc.build(_race(cat="none", hard="荒れ"), 3.4, _site(), "T-10")
    assert out["bets"] == [] and out["skip"] and "対象外" in out["skip_reason"]


def test_go_agree2_hard_low_odds_dedup_and_points():
    # ◎ が速度と PW の 1 位、推定人気は別馬 → 一致 2。GO∧固い・<2倍
    out = bc.build(_race(), 1.6, _site(si=1, pw=1, er=2), "T-20")
    types = [b["type"] for b in out["bets"]]
    assert types.count("馬連") == 4          # 2 セルが馬連を指しても 1 回だけ
    assert types.count("三連複") == 6
    assert "ワイド" not in types              # ワイドの <2倍 は 90% 未満で入っていない
    assert set(b["selection"] for b in out["bets"] if b["type"] == "馬連") == {"1-2", "1-3", "1-4", "1-5"}


def test_missing_inputs_skip():
    assert bc.build(_race(), 2.5, None, "T-10")["skip_reason"].startswith("ボードの指数")
    assert bc.build(_race(), None, _site(), "T-10")["skip_reason"].startswith("◎の単勝オッズ")
    bad = _race(marks=("◎", "〇", "▲", "△", "", ""))
    assert bc.build(bad, 2.5, _site(), "T-10")["skip_reason"].startswith("印")


def test_combo_tact_and_scrub_keeps_combo():
    combo = bc.build(_race(cat="none", hard="荒れ"), 2.5, _site(), "T-10")
    t = B.combo_tact(combo)
    assert t["engine"] == "board_combo" and t["bets"][0]["type"] == "単勝"
    pv = B.combo_tact({"bets": [], "skip": True, "skip_reason": "x"}, {"expires_at": "2026-11-21T10:00:00+09:00"})
    assert pv["is_preview"] and pv["skip_reason"] == "x"
    day = {"date": "20261121", "races": [{"tact": t, "tact_settled": [{"settled": True}],
                                          "bets_settled": [1], "cowork": {"bets": [{"type": "複勝"}]}}]}
    r = B.scrub_public(json.loads(json.dumps(day)))["races"][0]
    assert r["tact"]["engine"] == "board_combo" and "tact_settled" in r
    assert "bets_settled" not in r and "bets" not in r["cowork"]       # 本線 sniper は出さない
    old = {"date": "20261121", "races": [{"tact": {"bets": [{"type": "複勝"}]}, "cowork": {}}]}
    assert "tact" not in B.scrub_public(old)["races"][0]                 # sniper の tact は今までどおり外す
