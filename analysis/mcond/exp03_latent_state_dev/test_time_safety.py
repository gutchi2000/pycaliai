# -*- coding: utf-8 -*-
"""
test_time_safety.py — 仕様 §8 の時点安全テスト (2013-2014 の実データ)
  1. 将来データを削除しても過去の特徴が不変
  2. 対象レースの結果を変えても、そのレース直前の特徴が不変
  3. 同日の先行レース結果が同日の後続レースの特徴に入らない
  4. レース結果の更新は、その馬の次回出走以降の特徴にだけ反映される
  5. 同じ入力に対して決定論的 (2回実行で完全一致)
実行: python -m analysis.mcond.exp03_latent_state_dev.test_time_safety
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.exp03_latent_state_dev.latent_state import load_runs, run, Hyper  # noqa: E402

HP = Hyper(90, 0.005, 3.0, 2.0833)
FEATS = ["persistent_ability_mu", "latent_short_state_mu", "latent_short_state_sigma", "innovation_ewma",
         "last_performance_innovation", "combined_ability_state", "state_minus_field_mean",
         "latent_state_decay_ratio", "persistent_ability_change"]
_DF = None


def data():
    global _DF
    if _DF is None:
        _DF = load_runs(max_date="2014-12-31")
    return _DF.copy()


def feats(df):
    return run(df, HP)[0].set_index(["rid16", "ban"]).sort_index()


def same(a, b, rows):
    return np.allclose(a.loc[rows, FEATS].to_numpy(float), b.loc[rows, FEATS].to_numpy(float),
                       rtol=1e-12, atol=1e-12, equal_nan=True)


def busy_day(df):
    d = df[df.date.dt.year == 2014].groupby("date").rid16.nunique().idxmax()
    return d, sorted(df[df.date == d].rid16.unique())


def reversed_result(df, rid):
    m = df.rid16 == rid
    out = df.copy()
    out.loc[m, "fin"] = out.loc[m, "fin"].to_numpy()[::-1]
    return out


def test_future_deletion():
    full = feats(data())
    cut = "2014-06-30"
    part = feats(data()[lambda x: x.date <= cut])
    rows = full.index[full["date"] <= pd.Timestamp(cut)]
    assert same(full, part, rows)


def test_target_result_not_used():
    df = data()
    d, rids = busy_day(df)
    tgt = rids[len(rids) // 2]
    full, alt = feats(df), feats(reversed_result(df, tgt))
    rows = full.index[full.index.get_level_values(0) == tgt]
    assert same(full, alt, rows)


def test_same_day_earlier_result_not_used():
    df = data()
    d, rids = busy_day(df)
    full, alt = feats(df), feats(reversed_result(df, rids[0]))
    rows = full.index[(full["date"] == d) & (full.index.get_level_values(0) != rids[0])]
    assert same(full, alt, rows)


def test_update_only_affects_later_runs():
    """あるレースの結果を変えると、その出走馬の次走以降は変わり、それ以前と他馬の同日以前は変わらない。"""
    df = data()
    d, rids = busy_day(df)
    tgt = rids[len(rids) // 2]
    full, alt = feats(df), feats(reversed_result(df, tgt))
    before = full.index[full["date"] <= d]
    assert same(full, alt, before), "結果変更が当日以前の特徴に入った"
    horses = set(df.loc[df.rid16 == tgt, "hid"])
    later = full[(full["date"] > d) & full["hid"].isin(horses)].index
    assert len(later) > 0
    assert not same(full, alt, later), "結果変更が次回出走以降に反映されていない"


def test_deterministic():
    a, b = feats(data()), feats(data())
    assert same(a, b, a.index)


if __name__ == "__main__":
    for t in [test_future_deletion, test_target_result_not_used, test_same_day_earlier_result_not_used,
              test_update_only_affects_later_runs, test_deterministic]:
        t()
        print(f"{t.__name__}: PASS")
