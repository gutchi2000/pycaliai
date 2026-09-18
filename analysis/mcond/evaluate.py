# -*- coding: utf-8 -*-
"""
evaluate.py — 市場条件付き追加情報検定の共通評価器
==================================================
モデル: ロジスティック回帰 (L2)。全モデル同じクラス・同じ前処理で容量を揃える。
  前処理: train の平均・標準偏差で標準化、欠損は train 平均 (=標準化後0) で補完
  C の選択: selection 期間の logloss 最小を {0.1, 1, 10} から1点だけ選ぶ。最終モデルは train で学習
指標: logloss, Brier, ECE(10等幅), AUC, レース内最上位馬の3着内率
差の推定: 開催日ブロック bootstrap (同じ開催日の全レースを1ブロック)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

C_GRID = (0.1, 1.0, 10.0)
EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_predict(df, cols, y, train, sel, seed=0):
    X = df[cols].to_numpy(dtype=float)
    mu = np.nanmean(X[train], axis=0)
    sd = np.nanstd(X[train], axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    Z = np.where(np.isnan(Z), 0.0, Z)
    best = None
    for C in C_GRID:
        m = LogisticRegression(C=C, max_iter=2000)
        m.fit(Z[train], y[train])
        ll = ll_vec(y[sel], m.predict_proba(Z[sel])[:, 1]).mean()
        if best is None or ll < best[0]:
            best = (ll, C, m)
    _, C, m = best
    coef = dict(zip(cols, m.coef_[0].round(4)))
    return m.predict_proba(Z)[:, 1], C, coef


def ll_vec(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def ece(y, p, bins=10):
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


def metrics(y, p, rid):
    d = pd.DataFrame({"y": y, "p": p, "rid": rid})
    top = d.loc[d.groupby("rid").p.idxmax()]
    return {"n": int(len(y)), "logloss": float(ll_vec(y, p).mean()),
            "brier": float(((p - y) ** 2).mean()), "ece": float(ece(y, p)),
            "auc": float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else np.nan,
            "top_pick_top3": float(top.y.mean())}


def delta_boot(y, pa, pb, day, reps=2000, seed=42, level=0.95):
    """Δlogloss = mean(ll(pa)) - mean(ll(pb))。負なら a が良い。開催日ブロック bootstrap。"""
    la, lb = ll_vec(y, pa), ll_vec(y, pb)
    dd = pd.DataFrame({"d": la - lb, "day": day}).groupby("day")["d"].agg(["sum", "count"])
    s, c = dd["sum"].to_numpy(), dd["count"].to_numpy()
    rng = np.random.default_rng(seed)
    n = len(s)
    bs = np.empty(reps)
    for k in range(reps):
        i = rng.integers(0, n, n)
        bs[k] = s[i].sum() / c[i].sum()
    a = (1 - level) / 2
    return {"delta": float((la - lb).mean()), "ci_lo": float(np.quantile(bs, a)),
            "ci_hi": float(np.quantile(bs, 1 - a)), "p_improve": float((bs < 0).mean()),
            "level": level}
