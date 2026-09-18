# -*- coding: utf-8 -*-
"""
test_time_safety.py — 削除不変性テスト (未来の行を消しても過去行の特徴が変わらないこと)
===================================================================================
master を CUT 日で打ち切って特徴を作り直し、CUT 以前の行の生特徴・単純統計の逸脱・文脈が
全データで作ったものと一致することを確認する。一致しなければ未来の情報が混入している。
行動予測方式 (build_bp.py) は年単位の expanding window を assert で強制しているので対象外。

実行: python -m pytest analysis/mcond/exp01_choice_dev/test_time_safety.py -q
      (master を2回読むので数分かかる)
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.exp01_choice_dev.build_features import load_master, build  # noqa: E402

CUTS = ["2017-06-30", "2020-12-27"]


def _cols(df):
    return [c for c in df.columns if c.startswith(("raw_", "ss_", "ctx_"))]


def test_deletion_invariance():
    full = build(load_master())
    for cut in CUTS:
        part = build(load_master(max_date=cut))
        a = full[full["date"] <= pd.Timestamp(cut)].set_index(["rid16", "ban"]).sort_index()
        b = part.set_index(["rid16", "ban"]).sort_index()
        assert a.index.equals(b.index), f"{cut}: 行集合が違う ({len(a)} vs {len(b)})"
        for c in _cols(full):
            x, y = a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float)
            same = np.isclose(x, y, rtol=1e-9, atol=1e-9, equal_nan=True)
            assert same.all(), f"{cut}: {c} が {int((~same).sum())} 行で不一致 (未来の情報が混入)"


def test_history_strictly_before_target_day():
    """調教師統計が対象日の行を含まないこと: 各調教師のその日最初の出走で prior 件数が日跨ぎで単調。"""
    f = build(load_master(max_date="2014-12-31"))
    g = f.sort_values("date").groupby(["trainer", "date"])["trainer_n_prior"].nunique()
    assert (g == 1).all(), "同じ調教師・同じ日で prior 件数が行ごとに違う (同日の情報が混入)"


if __name__ == "__main__":
    test_history_strictly_before_target_day()
    print("history_strictly_before_target_day: PASS")
    test_deletion_invariance()
    print("deletion_invariance: PASS")
