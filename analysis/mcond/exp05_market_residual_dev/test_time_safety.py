# -*- coding: utf-8 -*-
"""
test_time_safety.py — EXP05独自の新規要素(isotonic再fit・offset連鎖)の時点安全性テスト
表特徴自体の時点安全性はEXP04のtest_time_safety.py/test_environment_split.pyで検証済みなので
ここでは重複させない。
実行: python -m pytest analysis/mcond/exp05_market_residual_dev/test_time_safety.py -q
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.exp05_market_residual_dev import models as Mmod  # noqa: E402


def _synth(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    n_race = n // 8
    rows = []
    for r in range(n_race):
        k = rng.integers(6, 12)
        year = 2016 + (r % 10)
        v6 = rng.normal(0, 1, k)
        mkt = v6 * 0.6 + rng.normal(0, 0.5, k)
        x1 = rng.normal(0, 1, k)
        z = v6 + 0.3 * mkt + 0.2 * x1
        p = 1 / (1 + np.exp(-z))
        top3 = (rng.random(k) < p).astype(int)
        for i in range(k):
            rows.append({"rid16": f"r{r:05d}", "ban": i + 1, "year": year,
                        "v6_score": v6[i], "v6_pwin": 1 / (1 + np.exp(-v6[i])),
                        "v6_p3": 1 / (1 + np.exp(-v6[i])),
                        "mkt_p3_pre": 1 / (1 + np.exp(-mkt[i])), "x1": x1[i], "top3": top3[i]})
    df = pd.DataFrame(rows)
    df["train"] = df.year <= 2021
    return df


def test_isotonic_refit_unaffected_by_future_rows():
    df = _synth()
    train = df["train"].to_numpy()

    def fit_iso(frame):
        m = frame["train"].to_numpy()
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(frame.loc[m, "v6_pwin"].to_numpy(), frame.loc[m, "top3"].to_numpy())
        return iso.predict(frame.loc[m, "v6_pwin"].to_numpy())

    full_pred = fit_iso(df)
    future_deleted = df[~((df.year >= 2024))].reset_index(drop=True)
    deleted_pred = fit_iso(future_deleted)
    assert np.allclose(full_pred, deleted_pred), "未来行の削除でtrain期間の較正出力が変わってはいけない"


def test_m3_train_coefficients_unaffected_by_future_rows():
    df = _synth()
    df["f_mkt"] = np.log(df.mkt_p3_pre / (1 - df.mkt_p3_pre))
    df["lp_cal_top3"] = np.log(df.v6_p3 / (1 - df.v6_p3))
    df["rank_v6"] = df.groupby("rid16")["v6_score"].rank(ascending=False)
    df["score_gap_to_top"] = df.v6_score - df.groupby("rid16")["v6_score"].transform("max")
    df["score_gap_to_second"] = 0.0
    df["score_percentile_in_race"] = df.groupby("rid16")["v6_score"].rank(pct=True)
    df["field_score_dispersion"] = df.groupby("rid16")["v6_score"].transform("std").fillna(0.0)
    df["sel"] = False
    y = df["top3"].to_numpy()
    train = df["train"].to_numpy()
    sel = train  # 合成データではtrainをselに流用(本番run.pyでは2022年を使う)

    out_full = Mmod.fit_m0_m3(df, y, train, sel)
    future_deleted = df[~(df.year >= 2024)].reset_index(drop=True)
    y2 = future_deleted["top3"].to_numpy()
    train2 = future_deleted["train"].to_numpy()
    out_deleted = Mmod.fit_m0_m3(future_deleted, y2, train2, train2)

    for name in ("M0", "M1", "M2", "M3"):
        c1 = np.array(list(out_full[name]["coef"].values()))
        c2 = np.array(list(out_deleted[name]["coef"].values()))
        assert np.allclose(c1, c2, atol=1e-6), f"{name}: 未来行の削除で係数が変わった(リーク疑い)"


def test_offset_residual_deterministic_and_no_future_leak():
    df = _synth(seed=1)
    df["f_mkt"] = np.log(df.mkt_p3_pre / (1 - df.mkt_p3_pre))
    y = df["top3"].to_numpy()
    train = df["train"].to_numpy()
    offset = df["f_mkt"].to_numpy()
    r1 = Mmod.fit_offset_residual(df, ["x1"], offset, y, train, train)
    r2 = Mmod.fit_offset_residual(df, ["x1"], offset, y, train, train)
    assert np.allclose(r1["beta"], r2["beta"]), "決定論的であるべき"

    future_deleted = df[~(df.year >= 2024)].reset_index(drop=True)
    y2, train2 = future_deleted["top3"].to_numpy(), future_deleted["train"].to_numpy()
    offset2 = future_deleted["f_mkt"].to_numpy()
    r3 = Mmod.fit_offset_residual(future_deleted, ["x1"], offset2, y2, train2, train2)
    assert np.allclose(r1["beta"], r3["beta"], atol=1e-6), "未来行削除でtrain betaが変わってはいけない"
