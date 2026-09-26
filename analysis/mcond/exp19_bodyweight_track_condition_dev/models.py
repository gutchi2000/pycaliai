# -*- coding: utf-8 -*-
"""
models.py — EXP19: arm の結合式と offset conditional-logit (EXP18 fit_cross 系の Newton を再利用)
=================================================================================================
SPEC §5 の固定式:
  N1  log q = a·log(m) + b·s_clean − log Z
  W   log q = a·log(m) + b·s_clean + θᵀW − log Z
  WP  log q = a·log(m) + b·s_clean + θᵀW + φᵀWP − log Z
m は pre (A 系) または terminal close (B 系) の比例 de-vig 単勝市場。全係数は Y−1 以前だけで fit。
fit は EXP18 tomography._clogit_newton (race セグメント上の Newton) をそのまま使い、別の fit 経路を作らない。
共線規則 (EXP18 fit_cross の一般化): race 内中心化した設計行列で、既に採用した列の張る空間に入る列
(race 内定数列を含む) を条件数 > 1e12 または rank 落ちで落とし、係数 0 として同じ経路で返す。
race 内定数の P 列は中心化で 0 になるので必ず落ち、softmax は P 無しの計算と bit 一致する。
"""
from __future__ import annotations

import numpy as np

from ..exp18_cross_pool_market_tomography_dev import tomography as T18

COND_MAX = 1e12


def race_center(X, offsets):
    X = np.asarray(X, float)
    start, cnt = np.asarray(offsets[:-1]), np.diff(offsets)
    mu = np.add.reduceat(X, start, axis=0) / cnt[:, None]
    return X - np.repeat(mu, cnt, axis=0)


def select_columns(X, offsets, names=None):
    """前から順に列を採用し、race 内中心化後に rank が増えない / 条件数 > 1e12 になる列を落とす。
    判定は中心化 Gram 行列 G = CᵀC の部分行列で行う (特異値 = sqrt(G の固有値) なので SVD と同じ判定を安価に得る)"""
    C = race_center(X, offsets)
    G = C.T @ C
    N = C.shape[0]
    keep = []
    for j in range(X.shape[1]):
        cand = keep + [j]
        ev = np.clip(np.linalg.eigvalsh(G[np.ix_(cand, cand)]), 0.0, None)
        s = np.sqrt(ev)
        tol = s.max() * max(N, len(cand)) * np.finfo(float).eps if s.max() > 0 else 0.0
        if np.sum(s > tol) < len(cand):
            continue
        cond = ev.max() / ev.min() if ev.min() > 0 else np.inf
        if not np.isfinite(cond) or cond > COND_MAX:
            continue
        keep.append(j)
    dropped = [j for j in range(X.shape[1]) if j not in keep]
    return keep, ([names[j] for j in dropped] if names else dropped)


def kelly_growth(q, odds, p_true):
    """EXP18 tomography.kelly_cash_growth と同じ現金込み Kelly の閉形式。違いは 0·log 0 = 0 の極限だけ
    (選んだ馬の信念確率の和が 1 に達し保留 R = 0 になる極端な場合に NaN を返さない)"""
    q = np.asarray(q, float)
    o = np.asarray(odds, float)
    p = np.asarray(p_true, float)
    ev = q * o
    order = np.argsort(-ev)
    ps = sig = 0.0
    R = 1.0
    sel = []
    for i in order:
        if ev[i] <= R:
            break
        ps_new, sig_new = ps + q[i], sig + 1.0 / o[i]
        if sig_new >= 1.0:
            break
        R_new = max((1.0 - ps_new) / (1.0 - sig_new), 0.0)
        if ev[i] <= R_new:
            break
        ps, sig, R = ps_new, sig_new, R_new
        sel.append(i)
    if not sel:
        return 0.0
    sel = np.array(sel)
    rest = 1.0 - p[sel].sum()
    reserve = rest * np.log(R) if (R > 0 and rest > 0) else (0.0 if rest <= 0 else -np.inf)
    return float((p[sel] * np.log(ev[sel])).sum() + reserve)


def fit_offset(X, offsets, win_rows, names=None, theta0=None):
    """offset conditional-logit。戻り: 全列長の係数 (落とした列は 0)、採用列、落とした列"""
    X = np.asarray(X, float)
    keep, dropped = select_columns(X, offsets, names)
    th0 = np.zeros(len(keep))
    if theta0 is not None:
        th0 = np.asarray(theta0, float)[keep]
    else:
        th0[0] = 1.0                                   # 第 1 列 = log m の初期値 1
    fallback = False
    try:
        th, ll = T18._clogit_newton(X[:, keep], offsets, win_rows, th0)
        if not (np.all(np.isfinite(th)) and np.isfinite(ll)):
            raise np.linalg.LinAlgError("non-finite Newton result")
    except np.linalg.LinAlgError:
        th, ll = damped_newton(X[:, keep], offsets, win_rows, th0)
        fallback = True
    full = np.zeros(X.shape[1])
    full[keep] = th
    return {"theta": full, "kept": keep, "dropped": dropped, "mean_ll": ll, "fallback_damped": fallback}


def _ll_grad_hess(X, offsets, win_rows, th):
    start, cnt = np.asarray(offsets[:-1]), np.diff(offsets)
    eta = X @ th
    mx = np.maximum.reduceat(eta, start)
    e = np.exp(eta - np.repeat(mx, cnt))
    Zs = np.add.reduceat(e, start)
    p = e / np.repeat(Zs, cnt)
    Ex = np.add.reduceat(p[:, None] * X, start)
    g = X[win_rows].sum(0) - Ex.sum(0)
    Exx = np.add.reduceat(p[:, None, None] * X[:, :, None] * X[:, None, :], start)
    H = -(Exx - Ex[:, :, None] * Ex[:, None, :]).sum(0)
    ll = float((X[win_rows] @ th).sum() - (np.log(Zs) + mx).sum())
    return ll, g, H


def damped_newton(X, offsets, win_rows, theta0, iters=500):
    """EXP18 の Newton が特異 Hessian で停止した場合だけ使う同じ目的関数 (条件付き logit 対数尤度) の最大化。
    方向は擬似逆行列による Newton 方向、歩幅は対数尤度が増えるまで半減 (backtracking)。目的関数は厳密に凹なので
    両経路が収束すれば同じ MLE に一致する"""
    th = np.array(theta0, float)
    R = len(offsets) - 1
    ll, g, H = _ll_grad_hess(X, offsets, win_rows, th)
    for _ in range(iters):
        d = -np.linalg.lstsq(H, g, rcond=None)[0]
        if not np.all(np.isfinite(d)) or g @ d <= 0:
            d = g
        t = 1.0
        while t > 1e-12:
            cand = th + t * d
            llc = _ll_grad_hess(X, offsets, win_rows, cand)[0]
            if np.isfinite(llc) and llc >= ll:
                break
            t *= 0.5
        step = t * d
        th = th + step
        ll_new, g, H = _ll_grad_hess(X, offsets, win_rows, th)
        if np.max(np.abs(step)) < 1e-10 or abs(ll_new - ll) < 1e-12 * max(1.0, abs(ll)):
            ll = ll_new
            break
        ll = ll_new
    return th, ll / R


def log_probs(X, theta, offsets):
    """log q (race 内正規化)。係数 0 の列は計算に入れない (落とした列の寄与は厳密に 0)"""
    X = np.asarray(X, float)
    nz = np.flatnonzero(theta != 0)
    eta = X[:, nz] @ theta[nz] if len(nz) else np.zeros(X.shape[0])
    start, cnt = np.asarray(offsets[:-1]), np.diff(offsets)
    mx = np.maximum.reduceat(eta, start)
    e = eta - np.repeat(mx, cnt)
    lse = np.log(np.add.reduceat(np.exp(e), start))
    return e - np.repeat(lse, cnt)


def arm_columns(arm, w_cols, wp_cols):
    base = ["log_m", "s_clean"]
    return {"N1": base, "W": base + w_cols, "WP": base + w_cols + wp_cols}[arm]


GATES = {
    "A1": {"market": "pre", "null": "N1", "alt": "W", "years": [2019, 2020, 2021, 2022, 2023], "min_years": 4,
           "placebos": ["P1", "P2"], "holm": True},
    "A2": {"market": "pre", "null": "W", "alt": "WP", "years": [2021, 2022, 2023], "min_years": 3,
           "placebos": ["P1", "P2", "P3"], "holm": True},
    "B1": {"market": "terminal", "null": "N1", "alt": "W", "years": [2019, 2020, 2021, 2022, 2023], "min_years": 4,
           "placebos": ["P1", "P2"], "holm": False, "run_if": "A1 in (PASS_SUBFLOOR, PASS_PRACTICAL)"},
    "B2": {"market": "terminal", "null": "W", "alt": "WP", "years": [2021, 2022, 2023], "min_years": 3,
           "placebos": ["P1", "P2", "P3"], "holm": False, "run_if": "A2 in (PASS_SUBFLOOR, PASS_PRACTICAL)"},
}
