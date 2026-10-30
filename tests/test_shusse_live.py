# -*- coding: utf-8 -*-
"""
tests/test_shusse_live.py — 出世予測 AI v2 の 2026 延長・現役馬推論の検査
=========================================================================
(a) 2026 延長後の starts_ext で、馬ごとの k が 1 から連番で日付が単調非減少
(b) 2025 年末に基準点を持つ既知馬で、master だけ（starts_v2/horses_v2）から作った特徴行と
    2026 延長込み（starts_ext/horses_ext）で作った特徴行が同一（基準日が 2025 内なら 2026 行は影響しない）
(c) predict_live の出力で P(L0..L4) の和 = 1、全確率が [0,1]、各馬の is_latest が 1 行

実行:
    ./venv311/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/test_shusse_live.py
analysis/shusse/out_v2/live/ が無ければ skip（extend_2026 → predict_live を先に実行）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "analysis" / "shusse" / "out_v2"
LIVE = OUT / "live"
pytestmark = pytest.mark.skipif(not (LIVE / "starts_ext.parquet").exists(), reason="run analysis.shusse.extend_2026 first")

from analysis.shusse.build_career import COL_HORSE  # noqa: E402
from analysis.shusse.features_v2 import build_features, load_asof_tables  # noqa: E402

K = 3
N_SAMPLE = 300
SEED = 11


@pytest.fixture(scope="module")
def ext():
    return pd.read_parquet(LIVE / "starts_ext.parquet"), pd.read_parquet(LIVE / "horses_ext.parquet")


@pytest.fixture(scope="module")
def master():
    return pd.read_parquet(OUT / "starts_v2.parquet"), pd.read_parquet(OUT / "horses_v2.parquet")


def test_k_contiguous_and_dates_monotone(ext):
    starts, _ = ext
    s = starts.sort_values([COL_HORSE, "k"])
    g = s.groupby(COL_HORSE)
    # k は 1..n の連番
    expect = g.cumcount().values + 1
    assert (s["k"].values == expect).all(), "k is not contiguous from 1"
    # 日付は k の順に単調非減少、同日重複なし
    prev = g["date"].shift(1)
    ok = prev.isna() | (s["date"] > prev)
    assert ok.all(), f"{(~ok).sum()} rows with non-increasing date"
    # 2026 行が実際に入っている
    assert (s["date"].dt.year == 2026).sum() > 30000
    assert s["src"].isin(["master", "weekly2026"]).all()


def test_master_rows_unchanged_values(ext, master):
    starts_ext, _ = ext
    starts_v2, _ = master
    m = starts_ext[starts_ext["src"] == "master"].drop(columns=["src"]).reset_index(drop=True)
    assert len(m) == len(starts_v2)
    for c in starts_v2.columns:
        a, b = m[c], starts_v2[c]
        eq = (a == b) | (a.isna() & b.isna())
        assert eq.all(), f"master rows changed in column {c}: {(~eq).sum()} rows"


def test_features_identical_for_2025_base_dates(ext, master):
    starts_ext, horses_ext = ext
    starts_v2, horses_v2 = master
    spec = json.loads((OUT / f"model_spec_k{K}.json").read_text(encoding="utf-8"))
    tables = load_asof_tables(OUT, K)
    priors = spec["te_priors"]
    base = starts_v2[(starts_v2["k"] == K) & (starts_v2["date"] >= "2025-07-01")]
    hv = horses_v2.set_index(COL_HORSE)
    cand = base[COL_HORSE]
    cand = cand[hv.loc[cand, "true_debut"].values & hv.loc[cand, "debut_age"].isin([2, 3]).values]
    # 2026 にも走っている馬を優先（延長の影響を受けうる馬）
    ran26 = set(starts_ext.loc[starts_ext["date"].dt.year == 2026, COL_HORSE])
    cand26 = cand[cand.isin(ran26)]
    rng = np.random.default_rng(SEED)
    sample = pd.Series(rng.choice(cand26.values, min(N_SAMPLE, len(cand26)), replace=False))
    assert len(sample) >= 100
    X0 = build_features(starts_v2, horses_v2, K, tables, priors, horse_ids=sample).set_index(COL_HORSE).sort_index()
    X1 = build_features(starts_ext, horses_ext, K, tables, priors, horse_ids=sample).set_index(COL_HORSE).sort_index()
    assert list(X0.index) == list(X1.index)
    feats = spec["features"]
    for c in feats:
        a, b = X0[c], X1[c]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            eq = np.isclose(a.astype(float).values, b.astype(float).values, equal_nan=True)
        else:
            eq = ((a == b) | (a.isna() & b.isna())).values
        assert eq.all(), f"feature {c} differs for {(~eq).sum()} horses after 2026 extension"


@pytest.mark.skipif(not (LIVE / "predictions_latest.parquet").exists(), reason="run analysis.shusse.predict_live first")
def test_predictions_are_probabilities():
    p = pd.read_parquet(LIVE / "predictions_latest.parquet")
    pl = p[[f"p_L{i}" for i in range(5)]]
    assert np.allclose(pl.sum(axis=1).values, 1.0, atol=1e-6)
    cols = ["p_dropout", "p_graded12", "p_graded24", "p_L3plus"] + list(pl.columns)
    assert ((p[cols] >= 0) & (p[cols] <= 1)).all().all()
    assert np.allclose(p["p_graded12"].values, ((1 - p["p_dropout"]) * p["p_L4"]).values)
    assert p["pred_level"].between(0, 4).all()
    assert p["k"].isin([1, 2, 3, 5]).all()
    # 各馬に is_latest はちょうど 1 行、最大 k の行
    lat = p[p["is_latest"]]
    assert lat[COL_HORSE].is_unique and set(lat[COL_HORSE]) == set(p[COL_HORSE])
    kmax = p.groupby(COL_HORSE)["k"].max()
    assert (lat.set_index(COL_HORSE)["k"].reindex(kmax.index).values == kmax.values).all()
    # 未確定行には実績が入っていない
    assert p.loc[~p["confirmed"], "actual_level"].isna().all()
    assert p.loc[~p["confirmed"], "actual_graded12"].isna().all()
    assert p["debut_year"].isin([2023, 2024, 2025, 2026]).all()
