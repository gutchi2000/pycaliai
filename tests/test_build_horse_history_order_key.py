# -*- coding: utf-8 -*-
"""build_horse_history の順序キー列 (発走時刻 HHMM・馬番) の生成。

仕様: docs/research/SERVE_REPRODUCIBILITY_FIX_SPEC_20261004.md §2.2 (発走時刻の欠損は 0 件しか許さない)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import build_horse_history as bhh
import serve_history_feats as shf


def test_parse_post_time_accepts_only_hh_mm():
    s = pd.Series(["09:50", "10:00", "9:05", "16:25", "", None, "25:00", "10:61", "1000", "abc"])
    got = bhh.parse_post_time(s)
    assert got.iloc[:4].tolist() == [950.0, 1000.0, 905.0, 1625.0]
    assert got.iloc[4:].isna().all()


def _hist(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame({"date": np.int32(20261003), "post_time": [1000, 1030, 1100][:n],
                         "race_id": [f"20261003050101{k + 1:02d}" for k in range(n)],
                         "umaban": [1] * n})


def test_validate_history_rejects_missing_post_time_value():
    h = _hist().astype({"post_time": float})
    h.loc[1, "post_time"] = np.nan
    with pytest.raises(shf.HistorySchemaError, match="post_time 1"):
        shf.validate_history(h)


def test_validate_history_rejects_malformed_post_time():
    h = _hist()
    h.loc[0, "post_time"] = 961
    with pytest.raises(shf.HistorySchemaError, match="HHMM"):
        shf.validate_history(h)


def test_validate_history_accepts_unique_order_key():
    assert shf.validate_history(_hist())["order_key_duplicates"] == 0
