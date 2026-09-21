# -*- coding: utf-8 -*-
"""
test_eligible_races.py — eligible_races.pyの合成テスト(例外処理カテゴリ)。
実データ(2024-2025)は使わない。
"""
from __future__ import annotations
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.exp09_conformal_abstention_dev.eligible_races import (  # noqa: E402
    build_eligible_races, _tansho_probs, MIN_FIELD_SIZE, PROB_SUM_TOLERANCE,
)
import pl_probs as PL  # noqa: E402


def _mk_rows(rid16, year, ban_score_fin_win_list):
    rows = []
    for ban, score, fin, win in ban_score_fin_win_list:
        rows.append({"rid16": rid16, "ban": ban, "year": year, "v6_score": score,
                     "n_field": len(ban_score_fin_win_list), "fin": fin, "win": win,
                     "date": f"{year}0101"})
    return rows


def _synthetic_df(rows):
    return pd.DataFrame(rows)


def _patch_and_run(df, years):
    with patch("analysis.mcond.exp09_conformal_abstention_dev.eligible_races.load_raw",
              return_value=df):
        return build_eligible_races(years)


def test_normal_race_is_eligible():
    rows = _mk_rows("r1", 2023, [
        (1, 1.0, 1, 1), (2, 0.5, 2, 0), (3, 0.2, 3, 0), (4, -0.1, 4, 0), (5, -0.5, 5, 0),
    ])
    out = _patch_and_run(_synthetic_df(rows), [2023])
    assert "r1" in out["eligible_rid16"]
    assert sum(out["exclusion_counts"][2023].values()) == 0


def test_dead_heat_excluded():
    """1着(fin==1)が2頭 → dead_heatで除外。"""
    rows = _mk_rows("r_dh", 2023, [
        (1, 1.0, 1, 1), (2, 0.9, 1, 1), (3, 0.2, 3, 0), (4, -0.1, 4, 0), (5, -0.5, 5, 0),
    ])
    out = _patch_and_run(_synthetic_df(rows), [2023])
    assert "r_dh" not in out["eligible_rid16"]
    assert out["exclusion_counts"][2023]["dead_heat"] == 1


def test_duplicate_ban_excluded():
    rows = _mk_rows("r_dup", 2023, [
        (1, 1.0, 1, 1), (1, 0.9, 2, 0), (3, 0.2, 3, 0), (4, -0.1, 4, 0), (5, -0.5, 5, 0),
    ])
    out = _patch_and_run(_synthetic_df(rows), [2023])
    assert "r_dup" not in out["eligible_rid16"]
    assert out["exclusion_counts"][2023]["duplicate_race_id"] == 1


def test_nan_score_excluded():
    rows = _mk_rows("r_nan", 2023, [
        (1, np.nan, 1, 1), (2, 0.9, 2, 0), (3, 0.2, 3, 0), (4, -0.1, 4, 0), (5, -0.5, 5, 0),
    ])
    out = _patch_and_run(_synthetic_df(rows), [2023])
    assert "r_nan" not in out["eligible_rid16"]
    assert out["exclusion_counts"][2023]["nan_score"] == 1


def test_missing_result_excluded():
    rows = _mk_rows("r_missing", 2023, [
        (1, 1.0, np.nan, 0), (2, 0.9, 2, 0), (3, 0.2, 3, 0), (4, -0.1, 4, 0), (5, -0.5, 5, 0),
    ])
    out = _patch_and_run(_synthetic_df(rows), [2023])
    assert "r_missing" not in out["eligible_rid16"]
    assert out["exclusion_counts"][2023]["missing_result"] == 1


def test_field_too_small_excluded():
    rows = _mk_rows("r_small", 2023, [(1, 1.0, 1, 1), (2, 0.5, 2, 0)])  # n=2 < MIN_FIELD_SIZE
    out = _patch_and_run(_synthetic_df(rows), [2023])
    assert "r_small" not in out["eligible_rid16"]
    assert out["exclusion_counts"][2023]["field_too_small"] == 1
    assert MIN_FIELD_SIZE == 3


def test_all_categories_independent_and_summable():
    """複数レースが別々の理由で除外されても、他の正常レースには影響しない。"""
    rows = (
        _mk_rows("ok1", 2023, [(1, 1.0, 1, 1), (2, 0.5, 2, 0), (3, 0.1, 3, 0)])
        + _mk_rows("dh1", 2023, [(1, 1.0, 1, 1), (2, 0.9, 1, 1), (3, 0.1, 3, 0)])
        + _mk_rows("small1", 2023, [(1, 1.0, 1, 1), (2, 0.5, 2, 0)])
    )
    out = _patch_and_run(_synthetic_df(rows), [2023])
    assert "ok1" in out["eligible_rid16"]
    assert "dh1" not in out["eligible_rid16"]
    assert "small1" not in out["eligible_rid16"]
    assert out["exclusion_counts"][2023]["dead_heat"] == 1
    assert out["exclusion_counts"][2023]["field_too_small"] == 1


def test_tansho_probs_matches_pl_probs():
    rng = np.random.default_rng(3)
    scores = rng.normal(size=8)
    ours = _tansho_probs(scores)
    ref = PL.all_tansho(PL.pl_weights(scores))
    np.testing.assert_allclose(ours, ref, atol=1e-12)


def test_no_result_performance_is_computed_here():
    """このモジュールはeligible集合の構築のみで、性能指標(coverage/logloss/ROI等)
    を一切計算しない設計であることの構造的確認(戻り値のキーを検査)。"""
    rows = _mk_rows("r1", 2023, [
        (1, 1.0, 1, 1), (2, 0.5, 2, 0), (3, 0.2, 3, 0), (4, -0.1, 4, 0), (5, -0.5, 5, 0),
    ])
    out = _patch_and_run(_synthetic_df(rows), [2023])
    forbidden_keys = {"coverage", "logloss", "brier", "roi", "accuracy"}
    assert forbidden_keys.isdisjoint(set(out.keys()))
