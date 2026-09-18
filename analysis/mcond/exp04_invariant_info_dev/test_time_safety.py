# -*- coding: utf-8 -*-
"""
test_time_safety.py — 環境ラベル生成の時点安全性
  1. confirm/exploratory 期間の行を削除しても、train期間の環境ラベル (統合後) が変わらない
     (統合の要否は train_mask のみで決まるため)
  2. 環境ラベルは rid16/surface/distance/year だけの決定論的関数 (同じ入力→同じ出力)
実行: python -m analysis.mcond.exp04_invariant_info_dev.test_time_safety
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.exp04_invariant_info_dev import environments  # noqa: E402


def _synth(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    years = rng.integers(2016, 2026, n)
    courses = rng.integers(1, 11, n)
    rid = [f"{y}0101{c:02d}010101"[:16] for y, c in zip(years, courses)]
    surf = rng.choice(["芝", "ダ"], n)
    dist = rng.choice([1200, 1600, 2000, 2400], n)
    return pd.Series(rid), pd.Series(surf), pd.Series(dist, dtype=float), pd.Series(years)


def test_future_deletion_does_not_change_train_labels():
    rid, surf, dist, year = _synth()
    train = (year >= 2016) & (year <= 2021)
    full, _ = environments.build(rid, surf, dist, year, train.to_numpy())

    keep = train | (year == 2022)  # confirm 以降 (ここでは2023-25) を削除した想定
    rid2, surf2, dist2, year2 = rid[keep], surf[keep], dist[keep], year[keep]
    train2 = (year2 >= 2016) & (year2 <= 2021)
    part, _ = environments.build(rid2.reset_index(drop=True), surf2.reset_index(drop=True),
                                 dist2.reset_index(drop=True), year2.reset_index(drop=True),
                                 train2.to_numpy())
    a = full[train.to_numpy()].reset_index(drop=True)
    b = part[train2.to_numpy()].reset_index(drop=True)
    assert a.equals(b), "未来期間の削除で train 期間の環境ラベルが変わった"


def test_deterministic_and_no_future_columns():
    rid, surf, dist, year = _synth()
    train = (year >= 2016) & (year <= 2021)
    a, _ = environments.build(rid, surf, dist, year, train.to_numpy())
    b, _ = environments.build(rid, surf, dist, year, train.to_numpy())
    assert a.equals(b)
    # 環境ラベルは日付順に並べ替えても (行の順序を変えても) 各行の値は変わらない
    perm = np.random.default_rng(1).permutation(len(rid))
    c, _ = environments.build(rid.iloc[perm].reset_index(drop=True), surf.iloc[perm].reset_index(drop=True),
                              dist.iloc[perm].reset_index(drop=True), year.iloc[perm].reset_index(drop=True),
                              train.to_numpy()[perm])
    assert c.reset_index(drop=True).equals(a.iloc[perm].reset_index(drop=True))


if __name__ == "__main__":
    test_future_deletion_does_not_change_train_labels()
    print("test_future_deletion_does_not_change_train_labels: PASS")
    test_deterministic_and_no_future_columns()
    print("test_deterministic_and_no_future_columns: PASS")
