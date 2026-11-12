# -*- coding: utf-8 -*-
"""power_rating.py（強さスコア）の検算。合成データのみ使用、実データ不要。"""
from __future__ import annotations

from datetime import date

import power_rating as pr


def test_recovers_true_strength_on_mock() -> None:
    races, truth = pr.make_mock_races(seed=42)
    horses, acc = pr.run_ratings(races, pr.Params(), races[len(races) * 7 // 10].day)
    names = [n for n in truth if n in horses and horses[n].n >= 5]
    rho = pr.spearman([truth[n] for n in names], [horses[n].rating for n in names])
    assert rho > 0.85
    assert acc.summary()["pair_concordance"] > 0.55


def test_update_is_zero_sum_and_ordered() -> None:
    race = pr.Race(date(2026, 1, 4), "R1", "1勝", "芝", [("A", 1), ("B", 2), ("C", 3)])
    horses, _ = pr.run_ratings([race], pr.Params())
    assert horses["A"].rating > horses["B"].rating > horses["C"].rating
    assert abs(sum(h.rating for h in horses.values()) - 3 * pr.CLASS_INIT["1勝"]) < 1e-6


def test_eval_uses_pre_race_ratings_only() -> None:
    """検証は更新前の値で行う: 1 レース目は全馬未出走なので検証対象に入らない。"""
    r1 = pr.Race(date(2026, 1, 4), "R1", "1勝", "芝", [("A", 1), ("B", 2)])
    r2 = pr.Race(date(2026, 1, 5), "R2", "1勝", "芝", [("A", 2), ("B", 1)])
    _, acc = pr.run_ratings([r1, r2], pr.Params(), date(2026, 1, 1))
    assert acc.races == 1
    assert acc.top_win == 0  # R1 の結果だけで A を上に見ていた → R2 は外れ


def test_name_reuse_resets_horse() -> None:
    r1 = pr.Race(date(2015, 1, 4), "R1", "G1", "芝", [("A", 1), ("B", 2)])
    r2 = pr.Race(date(2026, 1, 4), "R2", "未勝利", "芝", [("A", 2), ("C", 1)])
    horses, _ = pr.run_ratings([r1, r2], pr.Params())
    assert horses["A"].n == 1  # 11 年あいた同名は別馬


def test_score_scale_and_win_prob() -> None:
    assert pr.to_score(2000.0, 1000.0, 2000.0) == 100.0
    assert pr.to_score(1000.0, 1000.0, 2000.0) == 0.0
    assert abs(pr.win_prob(50.0, 50.0, 10.0) - 0.5) < 1e-12
    assert pr.win_prob(60.0, 50.0, 10.0) > 0.5


def test_top3_probs_sum_to_three_and_ordered() -> None:
    p = pr.top3_probs([1700.0, 1600.0, 1500.0, 1500.0, 1400.0, 1300.0])
    assert abs(sum(p) - 3.0) < 1e-9
    assert p[0] > p[1] > p[2] == p[3] > p[4] > p[5]
    assert pr.top3_probs([1500.0, 1500.0]) == [1.0, 1.0]


def test_pre_features_none_for_debut_and_no_future_use() -> None:
    assert pr.pre_features(pr.Horse(rating=1500.0), date(2026, 1, 4), "1勝", "芝", 1600) is None
    h = pr.Horse(rating=1500.0, n=2, last_day=date(2026, 1, 4), last_class="未勝利",
                 last_surface="芝", last_dist=1600, resids=[0.3, -0.1])
    ft = pr.pre_features(h, date(2026, 2, 1), "1勝", "ダ", 1200)
    assert ft[:7] == ("lay:15-28", "cls:up1", "surf:芝>ダ", "dist:short", "r1:good", "r3:mid",
                      "n:2")
    # 騎手・斤量・場所を渡していない → 不明 / 実績なし
    assert ft[7:] == ("jk:unk", "kin:unk", "hs:none", "hd:none", "hp:unk")


def test_pre_features_class_steps_jockey_weight_aptitude() -> None:
    h = pr.Horse(rating=1500.0, n=3, last_day=date(2026, 1, 4), last_class="3勝",
                 last_surface="芝", last_dist=2000, resids=[0.0],
                 last_jockey="n:武豊", last_kinryo=56.0,
                 apt={"s:芝": [0.6, 3], "d:中": [-0.45, 3], "p:東京": [0.0, 1]})
    ft = pr.pre_features(h, date(2026, 2, 1), "G1", "芝", 2000, "東京", "n:ルメール", 58.0)
    assert "cls:up2" in ft and "jk:change" in ft and "kin:up" in ft
    assert "hs:good" in ft and "hd:bad" in ft and "hp:mid" in ft
    # master の騎手コードと出走表の騎手名は比べない（境目で偽の乗り替わりを出さない）
    ft2 = pr.pre_features(h, date(2026, 2, 1), "3勝", "芝", 2000, "東京", "c:01234", 56.5)
    assert "jk:unk" in ft2 and "kin:same" in ft2 and "cls:same" in ft2


def test_run_ratings_tracks_jockey_weight_and_aptitude() -> None:
    r1 = pr.Race(date(2026, 1, 4), "R1", "1勝", "芝", [("A", 1), ("B", 2)], 1600, "中山",
                 {"A": ("n:X", 55.0), "B": ("n:Y", 57.0)})
    r2 = pr.Race(date(2026, 2, 1), "R2", "1勝", "芝", [("A", 2), ("B", 1)], 1600, "中山",
                 {"A": ("n:Z", 56.0), "B": ("n:Y", 57.0)})
    hist: list[tuple] = []
    horses, _ = pr.run_ratings([r1, r2], pr.Params(), None, hist)
    assert horses["A"].last_jockey == "n:Z" and horses["A"].last_kinryo == 56.0
    assert horses["A"].apt["s:芝"][1] == 2 and horses["A"].apt["p:中山"][1] == 2
    assert hist[0][10] == ""                       # 1 走目は条件なし
    a2 = hist[2][10].split("|")                     # A の 2 走目は発走前の状態だけで作る
    assert "jk:change" in a2 and "kin:up" in a2 and "hs:good" in a2


def test_fit_form_prefix_filter() -> None:
    rows = [(("cls:up1", "hp:good"), 0.1), (("cls:same", "hp:bad"), -0.1)] * 50
    m = pr.fit_form(rows, ("cls",))
    assert all(k.startswith("cls:") for k in m["coef"])
    assert pr.form_pred(m, ("cls:up1", "hp:bad")) == pr.form_pred(m, ("cls:up1", "hp:good"))


def test_top3_probs_matches_bruteforce() -> None:
    import itertools

    r = [1650.0, 1580.0, 1500.0, 1470.0, 1400.0]
    st = [10 ** (x / 400.0) for x in r]
    want = [0.0] * len(r)
    for a, b, c in itertools.permutations(range(len(r)), 3):
        tot = sum(st)
        pr_ = st[a] / tot * st[b] / (tot - st[a]) * st[c] / (tot - st[a] - st[b])
        for i in (a, b, c):
            want[i] += pr_
    got = pr.top3_probs(r)
    assert all(abs(x - y) < 1e-9 for x, y in zip(got, want))


def test_form_model_recovers_planted_effect() -> None:
    """昇級で下振れ・降級で上振れする合成データ → 係数の向きとラベルが合う。"""
    import random

    rng = random.Random(0)
    rows = []
    for _ in range(4000):
        c = rng.choices(["cls:up", "cls:same", "cls:down"], weights=[2, 6, 2])[0]
        eff = {"cls:up": -0.05, "cls:same": 0.0, "cls:down": 0.05}[c]
        rows.append(((c,), eff + rng.gauss(0.0, 0.2)))
    m = pr.fit_form(rows)
    assert m["coef"]["cls:down"] > m["coef"]["cls:same"] > m["coef"]["cls:up"]
    assert pr.form_label(m, pr.form_pred(m, ("cls:down",))) == "好走気配"
    assert pr.form_label(m, pr.form_pred(m, ("cls:same",))) == ""
    assert pr.form_label(m, pr.form_pred(m, ("cls:up",))) == "凡走注意"
    assert pr.form_label(m, None) == ""


def test_site_refuses_public_site_dir(tmp_path, monkeypatch) -> None:
    """非公開ページを公開サイト (site/) の中へ書こうとしたら、何も書かずに止まる。"""
    import power_rating_site as ps

    target = pr.BASE / "site" / "_power_rating_should_not_exist"
    assert ps.build(None, target) == 2
    assert not target.exists()


def test_site_page_has_no_external_data_fetch() -> None:
    """ページは data.js を script で読むだけ（file:// で開ける・外部へデータを送らない）。"""
    import power_rating_site as ps

    assert "fetch(" not in ps.HTML and "XMLHttpRequest" not in ps.HTML
    assert 'content="noindex,nofollow"' in ps.HTML


def test_turf_and_dirt_have_separate_ratings() -> None:
    """芝とダートは別レーティング: 芝の結果はダートの値を動かさない。"""
    turf = pr.Race(date(2026, 1, 4), "T1", "1勝", "芝", [("A", 1), ("B", 2)])
    dirt = pr.Race(date(2026, 1, 11), "D1", "1勝", "ダ", [("A", 2), ("B", 1)])
    horses, _ = pr.run_ratings([turf, dirt], pr.Params())
    a, b = horses["A"], horses["B"]
    assert a.sr["芝"][0] > b.sr["芝"][0] and a.sr["ダ"][0] < b.sr["ダ"][0]
    assert a.sr["芝"][1] == 1 and a.sr["ダ"][1] == 1 and a.n == 2
    # ダート初出走の時点では芝の勝ち負けを持ち込まない（2 頭とも同じ初期値から）
    only_turf, _ = pr.run_ratings([turf], pr.Params())
    assert only_turf["A"].sr["芝"][0] == a.sr["芝"][0]
    assert "ダ" not in only_turf["A"].sr


def test_unknown_surface_race_is_skipped() -> None:
    r = pr.Race(date(2026, 1, 4), "X1", None, None, [("A", 1), ("B", 2)])
    horses, _ = pr.run_ratings([r], pr.Params())
    assert horses == {}


def test_scale_anchors_per_surface() -> None:
    races = [pr.Race(date(2026, 1, 4), "T1", "G1", "芝", [("A", 1), ("B", 2)]),
             pr.Race(date(2026, 1, 4), "D1", "未勝利", "ダ", [("C", 1), ("D", 2)])]
    horses, _ = pr.run_ratings(races, pr.Params())
    for sf, top in (("芝", "A"), ("ダ", "C")):
        lo, hi = pr.scale_anchors(horses, date(2026, 1, 4), sf)
        assert abs(pr.to_score(horses[top].sr[sf][0], lo, hi) - 100.0) < 1e-9


def test_retired_on_handles_name_reuse_and_asof() -> None:
    h = pr.Horse(rating=1500.0, peak=1500.0)
    h.last_day = date(2026, 3, 29)
    ret = {"A": date(2026, 9, 17), "B": date(2020, 1, 1), "C": None}
    assert pr.retired_on(ret, "A", h, date(2026, 11, 8)) == date(2026, 9, 17)
    assert pr.retired_on(ret, "A", h, date(2026, 6, 1)) is None      # 基準日より後の抹消
    assert pr.retired_on(ret, "B", h, date(2026, 11, 8)) is None     # 最終出走より前 = 同名の別馬
    assert pr.retired_on(ret, "C", h, date(2026, 11, 8)) == date(2026, 11, 8)
    assert pr.retired_on(ret, "Z", h, date(2026, 11, 8)) is None


def test_scale_anchors_skip_retired() -> None:
    races = [pr.Race(date(2026, 1, d), f"T{d}", "G1", "芝",
                     [(nm, i + 1) for i, nm in enumerate("ABCDEFGHIJKLMNOPQRSTUVWX")])
             for d in (4, 5, 6)]
    horses, _ = pr.run_ratings(races, pr.Params())
    asof = date(2026, 1, 6)
    _, hi_all = pr.scale_anchors(horses, asof, "芝")
    _, hi_ret = pr.scale_anchors(horses, asof, "芝", {"A": date(2026, 1, 6)})
    assert hi_all == horses["A"].sr["芝"][0]
    assert hi_ret == horses["B"].sr["芝"][0]


def test_parse_um_record() -> None:
    import power_rating_retired as rt
    name = "サトノレーヴ".ljust(18, "　")
    buf = "UM1" + "20260918" + "2019101700" + "1" + "20210610" + "20260917" + "20190327" + name + "X" * 50
    assert rt.parse_um(buf) == ("2019101700", "サトノレーヴ", "1", "20260917")
    active = buf[:21] + "0" + buf[22:30] + "00000000" + buf[38:]
    assert rt.parse_um(active) == ("2019101700", "サトノレーヴ", "0", "")
    assert rt.parse_um("RA" + buf[2:]) is None


def test_parse_um_repairs_mojibake() -> None:
    import power_rating_retired as rt
    name = "ビダーヤ".ljust(18, "　")
    good = "UM9" + "20261105" + "2021105537" + "1" + "20230928" + "20261104" + "20210303" + name + "ﾋﾞ"
    bad = "".join(bytes([b]).decode("cp1252") if b not in (0x81, 0x8D, 0x8F, 0x90, 0x9D) else chr(b)
                  for b in good.encode("cp932"))
    assert bad != good
    assert rt.parse_um(bad) == rt.parse_um(good) == ("2021105537", "ビダーヤ", "1", "20261104")
