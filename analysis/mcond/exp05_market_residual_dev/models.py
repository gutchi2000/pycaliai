# -*- coding: utf-8 -*-
"""
models.py — EXP05 M0〜M6 の学習
==================================
M0-M3: 素のロジスティック回帰 (L2, analysis.mcond.evaluate.fit_predict を流用、C は sel=2022 で選択)
M4/M5: offset(M3) + 表特徴の残差回帰 (analysis.mcond.exp04_invariant_info_dev.methods を流用)
M6:    Benter型対照。w = exp(α·log(v6_pwin) + β·log(market_pi)) を win の race内softmax尤度で
       MLE fit (train期間) し、pl_top3 (Harville) で3着内確率を出す。
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.evaluate import fit_predict, logit  # noqa: E402
from analysis.mcond.v6base import pl_top3  # noqa: E402
from analysis.mcond.exp04_invariant_info_dev import methods as m4methods  # noqa: E402

EPS = 1e-9
C_GRID_RESIDUAL = (0.01, 0.1, 1.0)


def m0_m3_cols():
    return {
        "M0": ["f_mkt"],
        "M1": ["lp_cal_top3", "f_mkt"],
        "M2": ["lp_cal_top3", "f_mkt", "v6_score", "rank_v6"],
        "M3": ["lp_cal_top3", "f_mkt", "v6_score", "rank_v6",
              "score_gap_to_top", "score_gap_to_second", "score_percentile_in_race",
              "field_score_dispersion"],
    }


def fit_m0_m3(df: pd.DataFrame, y: np.ndarray, train: np.ndarray, sel: np.ndarray) -> dict:
    cols = m0_m3_cols()
    out = {}
    for name, cc in cols.items():
        p, C, coef = fit_predict(df, cc, y, train, sel)
        out[name] = {"pred": p, "C": C, "coef": coef, "cols": cc}
    return out


def design_matrix(df: pd.DataFrame, cols: list[str], train_mask: np.ndarray) -> np.ndarray:
    X = df[cols].to_numpy(dtype=float)
    mu = np.nanmean(X[train_mask], axis=0)
    sd = np.nanstd(X[train_mask], axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    return np.where(np.isnan(Z), 0.0, Z)


def fit_offset_residual(df: pd.DataFrame, cols: list[str], offset: np.ndarray, y: np.ndarray,
                        train: np.ndarray, sel: np.ndarray) -> dict:
    """offset付きL2ロジスティック回帰。C は sel=2022 logloss で選ぶ (exp04 methods を流用)。"""
    Z = design_matrix(df, cols, train)
    n_tr = int(train.sum())
    best = None
    for C in C_GRID_RESIDUAL:
        l2 = 1.0 / (C * n_tr)
        beta = m4methods.fit_offset_logit(Z[train], offset[train], y[train], l2)
        p = m4methods.predict_offset_logit(Z[sel], offset[sel], beta)
        ll = -np.mean(y[sel] * np.log(np.clip(p, EPS, 1)) + (1 - y[sel]) * np.log(np.clip(1 - p, EPS, 1)))
        if best is None or ll < best[0]:
            best = (ll, beta, C)
    _, beta, C = best
    pred_all = m4methods.predict_offset_logit(Z, offset, beta)
    return {"pred": pred_all, "beta": beta, "C": C, "cols": cols}


def fit_m6_benter(df: pd.DataFrame, train_mask: np.ndarray) -> dict:
    """w_i = exp(α log v6_pwin_i + β log market_pi_i)。win の race内softmax尤度をtrainでMLE。"""
    logf = np.log(np.clip(df["v6_pwin"].to_numpy(), EPS, 1))
    logpi = np.log(np.clip(df["mkt_pi_pre"].to_numpy(), EPS, 1))
    win = df["win"].to_numpy()
    rid = df["rid16"].to_numpy()

    tr_idx = np.where(train_mask)[0]
    groups = []
    for rid_v, idx in pd.Series(tr_idx).groupby(rid[tr_idx]):
        i = idx.to_numpy()
        if win[i].sum() == 1:
            groups.append(i)

    def nll(theta):
        a, b = theta
        tot = 0.0
        for i in groups:
            z = a * logf[i] + b * logpi[i]
            z = z - z.max()
            w = np.exp(z)
            tot -= (z[win[i] == 1][0] - np.log(w.sum()))
        return tot / len(groups)

    res = minimize(nll, x0=np.array([1.0, 0.3]), method="Nelder-Mead",
                   options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 500})
    alpha, beta = res.x

    z_all = alpha * logf + beta * logpi
    pwin = np.zeros(len(df))
    p3 = np.zeros(len(df))
    for rid_v, idx in pd.Series(np.arange(len(df))).groupby(rid):
        i = idx.to_numpy()
        zz = z_all[i] - z_all[i].max()
        w = np.exp(zz)
        pwin[i] = w / w.sum()
        p3[i] = pl_top3(w) if len(i) >= 3 else np.nan
    return {"pred": p3, "pred_win": pwin, "alpha": float(alpha), "beta": float(beta),
           "n_train_races": len(groups)}
