# -*- coding: utf-8 -*-
"""
test_state_identification.py — 状態方程式の性質 (合成データ)
  - 長い休養で短期状態が0へ近づく / 短い間隔では多く保持される
  - 休養で状態の不確実性は定常値へ戻る
  - 好走で a と s の両方が上がり、s の取り分は var_s/var_eff
  - 新馬は a=25, s=0
  - innovation は期待より良い着順で正
実行: python -m pytest analysis/mcond/exp03_latent_state_dev/test_state_identification.py -q
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.exp03_latent_state_dev.latent_state import run, Hyper, pl_score, MU0  # noqa: E402

HP = Hyper(90, 0.005, 3.0, 2.0833)


def race(date, rid, hids, fins):
    return pd.DataFrame({"date": pd.Timestamp(date), "rid16": rid, "ban": range(1, len(hids) + 1),
                         "hid": hids, "fin": fins, "surf": 1, "dband": 1})


def seq(gap_days):
    """h1 が2回連続で勝ち、gap_days 後に3走目。3走目直前の状態を返す。"""
    d0 = pd.Timestamp("2020-01-05")
    df = pd.concat([race(d0, "A", ["h1", "x1", "x2", "x3"], [1, 2, 3, 4]),
                    race(d0 + pd.Timedelta(days=14), "B", ["h1", "x4", "x5", "x6"], [1, 2, 3, 4]),
                    race(d0 + pd.Timedelta(days=14 + gap_days), "C", ["h1", "x7", "x8", "x9"], [1, 2, 3, 4])])
    f, _, _ = run(df, HP)
    return f[(f.rid16 == "C") & (f.hid == "h1")].iloc[0]


def test_state_decays_with_layoff():
    short, long_ = seq(14), seq(365)
    assert short.latent_short_state_mu > 0
    assert abs(long_.latent_short_state_mu) < abs(short.latent_short_state_mu), "長い休養で状態は0へ近づく"
    assert long_.latent_state_decay_ratio < short.latent_state_decay_ratio
    assert abs(long_.latent_short_state_sigma - HP.qs) < abs(short.latent_short_state_sigma - HP.qs) + 1e-12, \
        "長い休養で状態の不確実性は定常値へ戻る"
    # 能力 a は休養で平均が戻らない
    assert np.isclose(long_.persistent_ability_mu, short.persistent_ability_mu)


def test_new_horse_prior():
    f, _, _ = run(race("2020-01-05", "A", ["h1", "h2", "h3"], [1, 2, 3]), HP)
    assert np.allclose(f.persistent_ability_mu, MU0) and np.allclose(f.latent_short_state_mu, 0.0)


def test_split_by_variance_ratio():
    d0 = pd.Timestamp("2020-01-05")
    df = pd.concat([race(d0, "A", ["h1", "h2", "h3"], [1, 2, 3]),
                    race(d0 + pd.Timedelta(days=1), "B", ["h1", "h4"], [1, 2])])
    f, _, _ = run(df, HP)
    b = f[(f.rid16 == "B") & (f.hid == "h1")].iloc[0]
    da = b.persistent_ability_mu - MU0
    ds_prior = b.latent_short_state_mu / b.latent_state_decay_ratio      # 減衰前の状態
    va0, vs0 = (25 / 3) ** 2, HP.qs ** 2
    assert da > 0 and ds_prior > 0
    assert np.isclose(ds_prior / da, vs0 / va0, rtol=1e-6), "更新は分散比で配分"


def test_innovation_sign():
    mu, var = np.array([30.0, 20.0, 20.0]), np.full(3, 9.0)
    _, _, om_up, de, _ = pl_score(mu, var, np.array([3.0, 1, 2]), HP.beta)   # 強い馬が負ける
    _, _, om_ok, _, _ = pl_score(mu, var, np.array([1.0, 2, 3]), HP.beta)    # 強い馬が勝つ
    assert om_up[0] < 0 and om_up[1] > 0
    assert om_ok[0] > 0 and om_ok[0] < om_up[1], "期待どおりの勝ちは番狂わせの勝ちより小さい"
