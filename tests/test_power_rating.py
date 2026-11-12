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
    assert ft == ("lay:15-28", "cls:up", "surf:芝>ダ", "dist:short", "r1:good", "r3:mid", "n:2")


def test_form_model_recovers_planted_effect() -> None:
    """昇級で下振れ・降級で上振れする合成データ → 係数の向きとラベルが合う。"""
    import random

    rng = random.Random(0)
    rows = []
    for _ in range(4000):
        c = rng.choice(["cls:up", "cls:same", "cls:down"])
        eff = {"cls:up": -0.05, "cls:same": 0.0, "cls:down": 0.05}[c]
        rows.append(((c, rng.choice(["n:1", "n:2"])), eff + rng.gauss(0.0, 0.2)))
    m = pr.fit_form(rows)
    assert m["coef"]["cls:down"] > m["coef"]["cls:same"] > m["coef"]["cls:up"]
    assert pr.form_label(m, pr.form_pred(m, ("cls:down", "n:1"))) == "好走気配"
    assert pr.form_label(m, pr.form_pred(m, ("cls:up", "n:1"))) == "凡走注意"
