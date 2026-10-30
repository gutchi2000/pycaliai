# -*- coding: utf-8 -*-
"""
tests/test_shusse_asof.py — 出世予測 AI v2 の leak 検査
=======================================================
1. 基準日より後の行を無作為に削除・改変しても k 走目基準の特徴行列が変わらない（200 頭）
2. margin_k は自走行から計算されている（次走の有無と欠損率が独立、勝ち時計差の恒等式）
3. as-of TE に基準日以後の結果が入っていない（手計算と一致）

実行:
    ./venv311/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/test_shusse_asof.py
analysis/shusse/out_v2/starts_v2.parquet が無ければ skip。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "analysis" / "shusse" / "out_v2"
pytestmark = pytest.mark.skipif(not (OUT / "starts_v2.parquet").exists(), reason="run analysis.shusse.build_career_v2 first")

from analysis.shusse.build_career import COL_HORSE  # noqa: E402
from analysis.shusse.features_v2 import (  # noqa: E402
    TE_ENTITIES, TE_M, add_asof_te, build_asof_tables, build_features, build_targets, eligible_mask, feature_cols,
)

K = 3
N_SAMPLE = 200
SEED = 7


@pytest.fixture(scope="module")
def data():
    starts = pd.read_parquet(OUT / "starts_v2.parquet")
    horses = pd.read_parquet(OUT / "horses_v2.parquet")
    targets = build_targets(starts, K)
    elig = eligible_mask(horses, targets, K)
    hid = elig[elig].index.to_series()
    rng = np.random.default_rng(SEED)
    # 後続の走がある馬を優先してサンプル（改変の余地があるように）
    many = starts.groupby(COL_HORSE)["k"].max()
    cand = hid[many.reindex(hid).values >= K + 2]
    sample = pd.Series(rng.choice(cand.values, N_SAMPLE, replace=False))
    tables = build_asof_tables(horses, targets, K)
    priors = {"ran": 0.88, "lvl": 0.8, "l4": 0.03, "l3p": 0.12}
    return starts, horses, targets, tables, priors, sample


def _features(starts, horses, tables, priors, sample):
    X = build_features(starts, horses, K, tables, priors, horse_ids=sample).set_index(COL_HORSE).sort_index()
    return X


def test_future_rows_do_not_change_features(data):
    starts, horses, targets, tables, priors, sample = data
    X0 = _features(starts, horses, tables, priors, sample)
    assert len(X0) == N_SAMPLE
    rng = np.random.default_rng(SEED + 1)
    base = targets.set_index(COL_HORSE).loc[sample, "base_date"]
    s = starts.copy()
    is_sample = s[COL_HORSE].isin(sample)
    after = is_sample & (s["date"].values > base.reindex(s[COL_HORSE]).values)
    assert after.sum() > 0
    # 無作為に半分削除、残りは着順/クラス/着差/馬体重を改変
    drop = after & (rng.random(len(s)) < 0.5)
    s = s[~drop].copy()
    after2 = s[COL_HORSE].isin(sample) & (s["date"].values > base.reindex(s[COL_HORSE]).values)
    s.loc[after2, "pos"] = 1
    s.loc[after2, "win"] = 1
    s.loc[after2, "race_level"] = 8
    s.loc[after2, "margin"] = -5.0
    s.loc[after2, "weight"] = 999.0
    s.loc[after2, "agari_rank"] = 1.0
    # k は基準日以前では不変（後続行だけが消えるので再採番しても一致）
    s = s.sort_values([COL_HORSE, "date"])
    s["k"] = s.groupby(COL_HORSE).cumcount() + 1
    X1 = _features(s, horses, tables, priors, sample)
    feats = feature_cols(X0.reset_index())
    pd.testing.assert_frame_equal(X0[feats], X1[feats], check_dtype=False)


def test_margin_from_own_row_and_independent_of_next_start(data):
    starts, horses, targets, tables, priors, sample = data
    # 恒等式: margin = time_sec − win_time（同レース 1 着馬のタイム）
    s = starts[starts["k"] == K]
    assert s["margin"].notna().all()
    assert np.allclose(s["margin"], s["time_sec"] - s["win_time"])
    assert (s["margin"] >= -1e-9).all()
    winners = s[s["pos"] == 1]
    assert (winners["margin"].abs() < 1e-9).mean() > 0.99
    # 次走の有無と欠損率が独立（v1 の持ち上げ方式なら次走なし = 100% 欠損になる）
    elig = eligible_mask(horses, targets, K)
    t = targets[targets[COL_HORSE].isin(elig[elig].index)]
    has_next = t[COL_HORSE].isin(starts.loc[starts["k"] == K + 1, COL_HORSE])
    sk = s.set_index(COL_HORSE)["margin"].reindex(t[COL_HORSE])
    miss_next = sk[has_next.values].isna().mean()
    miss_nonext = sk[~has_next.values].isna().mean()
    assert abs(miss_next - miss_nonext) < 0.01
    assert (~has_next).sum() > 1000


def test_asof_te_excludes_results_known_on_or_after_base_date(data):
    starts, horses, targets, tables, priors, sample = data
    X = _features(starts, horses, tables, priors, sample)
    h = horses[horses["true_debut"]].set_index(COL_HORSE)
    t = targets[targets["h12_complete"]].set_index(COL_HORSE)
    hist = h.join(t[["base_date", "ran_12m", "y_level"]], how="inner")
    hist["known"] = hist["base_date"] + pd.Timedelta(days=365)
    checked = 0
    for hid, row in X.head(60).iterrows():
        base = pd.Timestamp(row["base_date"])
        for prefix, ent in TE_ENTITIES.items():
            e = row[ent]
            if pd.isna(e):
                continue
            past = hist[(hist[ent].astype(str) == str(e)) & (hist["known"] < base)]
            n = len(past)
            assert row[f"{prefix}_n_known"] == n
            ran = past[past["ran_12m"] == 1]
            exp_ran = (len(ran) + TE_M * priors["ran"]) / (n + TE_M)
            exp_l4 = ((ran["y_level"] >= 4).sum() + TE_M * priors["l4"]) / (len(ran) + TE_M)
            assert abs(row[f"{prefix}_te_ran"] - exp_ran) < 1e-9
            assert abs(row[f"{prefix}_te_l4"] - exp_l4) < 1e-9
            # 基準日以後に確定した馬を含めると値が変わることも確認（情報が実在する）
            later = hist[(hist[ent].astype(str) == str(e)) & (hist["known"] >= base)]
            if len(later) > 0:
                assert len(later) + n == (hist[ent].astype(str) == str(e)).sum()
            checked += 1
    assert checked > 100
