# -*- coding: utf-8 -*-
"""
test_time_safety.py — 仕様 §7 の4テスト (2013-2014 の実データで実行)
  1. 将来期間を削除しても過去の特徴が変わらない
  2. 同日の後続レースを削除しても、その日の他レースの特徴が変わらない
  3. 同日の先行レースの結果を変えても、同日の後続レースの特徴が変わらない
  4. 対象レースの結果を変えても、そのレース直前の特徴が変わらない
実行: python -m pytest analysis/mcond/exp02_dynamic_skill_dev/test_time_safety.py -q
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.exp02_dynamic_skill_dev.dyn_skill import load_runs, run, Hyper  # noqa: E402

HP = Hyper(beta=25 / 6, tau2_per_day=0.02)
FEATS = ["dyn_skill_mu", "dyn_skill_sigma", "dyn_skill_num_updates", "dyn_skill_days_since_update",
         "field_skill_mean", "horse_skill_minus_field", "race_difficulty",
         "condition_skill_mu", "condition_skill_sigma", "condition_skill_minus_global"]
_DF = None


def data():
    global _DF
    if _DF is None:
        _DF = load_runs(max_date="2014-12-31")
    return _DF.copy()


def feats(df):
    f, _ = run(df, HP)
    return f.set_index(["rid16", "ban"]).sort_index()


def same(a, b, rows):
    x = a.loc[rows, FEATS].to_numpy(float)
    y = b.loc[rows, FEATS].to_numpy(float)
    return np.allclose(x, y, rtol=1e-12, atol=1e-12, equal_nan=True)


def busy_day(df):
    """2014年で最もレース数の多い日と、その日のレースID (ID順)。"""
    d = df[df.date.dt.year == 2014].groupby("date").rid16.nunique().idxmax()
    return d, sorted(df[df.date == d].rid16.unique())


def test_future_deletion():
    full = feats(data())
    cut = "2014-06-30"
    part = feats(data()[lambda x: x.date <= cut])
    rows = full.index[full["date"] <= pd.Timestamp(cut)]
    assert part.index.equals(rows) or set(part.index) == set(rows)
    assert same(full, part, rows), "将来期間の削除で過去の特徴が変わった"


def test_same_day_later_race_deletion():
    df = data()
    d, rids = busy_day(df)
    full = feats(df)
    drop = rids[-1]
    part = feats(df[df.rid16 != drop])
    rows = full.index[(full["date"] == d) & (full.index.get_level_values(0) != drop)]
    assert same(full, part, rows), "同日の後続レース削除で他レースの特徴が変わった"


def test_same_day_earlier_result_not_used():
    df = data()
    d, rids = busy_day(df)
    first = rids[0]
    full = feats(df)
    m = df.rid16 == first
    df2 = df.copy()
    df2.loc[m, "fin"] = df2.loc[m, "fin"].to_numpy()[::-1]          # 着順を逆転
    alt = feats(df2)
    rows = full.index[(full["date"] == d) & (full.index.get_level_values(0) != first)]
    assert same(full, alt, rows), "同日の先行レース結果が後続レースの特徴に入った"


def test_target_result_not_used():
    df = data()
    d, rids = busy_day(df)
    tgt = rids[len(rids) // 2]
    full = feats(df)
    m = df.rid16 == tgt
    df2 = df.copy()
    df2.loc[m, "fin"] = df2.loc[m, "fin"].to_numpy()[::-1]
    alt = feats(df2)
    rows = full.index[full.index.get_level_values(0) == tgt]
    assert same(full, alt, rows), "対象レースの結果がそのレースの特徴に入った"


if __name__ == "__main__":
    for t in [test_future_deletion, test_same_day_later_race_deletion,
              test_same_day_earlier_result_not_used, test_target_result_not_used]:
        t()
        print(f"{t.__name__}: PASS")
