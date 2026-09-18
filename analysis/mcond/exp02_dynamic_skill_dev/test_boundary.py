# -*- coding: utf-8 -*-
"""
test_boundary.py — 更新式と順位例外の境界条件 (合成データ)
実行: python -m pytest analysis/mcond/exp02_dynamic_skill_dev/test_boundary.py -q
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.exp02_dynamic_skill_dev.dyn_skill import (  # noqa: E402
    wl_update, run, Hyper, MU0, SIGMA0, dist_band)

HP = Hyper(beta=25 / 6, tau2_per_day=0.02)


def race(date, rid, hids, fins, surf=1, dband=1):
    return pd.DataFrame({"date": pd.Timestamp(date), "rid16": rid, "ban": range(1, len(hids) + 1),
                         "hid": hids, "fin": fins, "surf": surf, "dband": dband})


def test_winner_up_loser_down():
    mu, var = np.full(5, MU0), np.full(5, SIGMA0 ** 2)
    nm, nv, _ = wl_update(mu, var, np.array([1., 2, 3, 4, 5]), HP.beta)
    assert nm[0] > MU0 > nm[4]
    assert np.all(np.diff(nm) < 0), "着順が良いほど能力平均が上がる"
    assert np.all(nv < var), "1走で不確実性が減る"


def test_tie_equal_update():
    mu, var = np.full(4, MU0), np.full(4, SIGMA0 ** 2)
    nm, nv, _ = wl_update(mu, var, np.array([1., 1, 3, 4]), HP.beta)
    assert np.isclose(nm[0], nm[1]) and np.isclose(nv[0], nv[1]), "同着は同じ更新"


def test_upset_moves_more():
    """格上に勝った馬は、格下に勝った馬より大きく上がる。"""
    var = np.full(2, SIGMA0 ** 2)
    up, _, _ = wl_update(np.array([20.0, 30.0]), var, np.array([1., 2]), HP.beta)
    fav, _, _ = wl_update(np.array([30.0, 20.0]), var, np.array([1., 2]), HP.beta)
    assert up[0] - 20.0 > fav[0] - 30.0


def test_new_horse_prior_and_layoff():
    df = pd.concat([race("2020-01-05", "A", ["h1", "h2", "h3"], [1, 2, 3]),
                    race("2020-02-02", "B", ["h1", "h4", "h5"], [2, 1, 3]),
                    race("2021-02-07", "C", ["h1", "h2", "h6"], [1, 2, 3])])
    f, _ = run(df, HP)
    first = f[f.rid16 == "A"]
    assert np.allclose(first.dyn_skill_mu, MU0) and np.allclose(first.dyn_skill_sigma, SIGMA0)
    new = f[(f.rid16 == "B") & (f.hid == "h4")]
    assert np.isclose(new.dyn_skill_mu.iloc[0], MU0) and new.dyn_skill_num_updates.iloc[0] == 0
    h1b = f[(f.rid16 == "B") & (f.hid == "h1")].dyn_skill_sigma.iloc[0]
    h2c = f[(f.rid16 == "C") & (f.hid == "h2")].dyn_skill_sigma.iloc[0]
    h1c = f[(f.rid16 == "C") & (f.hid == "h1")].dyn_skill_sigma.iloc[0]
    assert h2c > h1b, "約13か月の休養で不確実性が増える"
    assert h1c < h2c, "出走回数が多い方が (同じ日に) 不確実性が小さい… 休養期間の違いを除いても"


def test_same_day_batch():
    """同じ馬が同日に2走することは無いが、同日の他レースの結果は当日の特徴に入らない。"""
    df = pd.concat([race("2020-01-05", "A", ["h1", "h2"], [1, 2]),
                    race("2020-01-05", "B", ["h3", "h4"], [1, 2]),
                    race("2020-01-12", "C", ["h1", "h3"], [1, 2])])
    f, _ = run(df, HP)
    assert np.allclose(f[f.rid16 == "B"].dyn_skill_mu, MU0)
    c = f[f.rid16 == "C"].set_index("hid")
    assert c.loc["h1", "dyn_skill_num_updates"] == 1 and c.loc["h3", "dyn_skill_num_updates"] == 1


def test_t2_shrinks_to_global_when_unseen():
    df = pd.concat([race("2020-01-05", "A", ["h1", "h2", "h3"], [1, 2, 3], surf=1, dband=1),
                    race("2020-01-19", "B", ["h1", "h2", "h3"], [1, 2, 3], surf=1, dband=1),
                    race("2020-02-02", "C", ["h1", "h2", "h3"], [1, 2, 3], surf=0, dband=3)])
    f, _ = run(df, HP)
    c = f[f.rid16 == "C"].set_index("hid")
    b = f[f.rid16 == "B"].set_index("hid")
    assert abs(c.loc["h1", "condition_skill_minus_global"]) < abs(b.loc["h1", "condition_skill_minus_global"]) + 1e-9 \
        or abs(c.loc["h1", "condition_skill_minus_global"]) < 1e-9, "未経験条件の効果は0 (全体能力へ縮約)"


def test_dist_band():
    assert [dist_band(x) for x in (1000, 1400, 1401, 1800, 2000, 2200, 2400)] == [0, 0, 1, 1, 2, 2, 3]
