# -*- coding: utf-8 -*-
"""
tomography.py — EXP18 の純関数 (結果性能と独立)
================================================
入力はオッズ・確率だけ。着順・払戻は一切受け取らない (fit 関数だけが winner 位置を受け取る)。

  devig_power            m ∝ (1/o)^γ (γ=1 で比例 de-vig)
  harville_top2          T0: 単勝 marginal π → 順不同 top-two (Harville)
  stern_top2             T1: べき割引 Harville (本リポジトリの Stern/LBS 型; P(j 2着|i)∝π_j^λ)
  ordered_prefixes       長さ k の順序付き上位列とその T1 確率
  place_probs            P(i ∈ top-k)
  t2_soft_fukusho        T2: MaxEnt 事前 (=T1) + 単勝 marginal 等式 + 複勝 soft 制約
  temp_probs / cross_probs / fit_temp / fit_cross   温度 null と cross-pool 代替 (共線規則つき)
  oracle_top2_topk       n<=8 の全順列列挙 oracle
"""
from __future__ import annotations

from itertools import permutations

import numpy as np
from scipy.optimize import minimize

COND_MAX = 1e12


# ---------------------------------------------------------------- de-vig
def devig_power(odds, gamma: float = 1.0) -> np.ndarray:
    """m ∝ (1/o)^γ をレース内で正規化する"""
    x = np.power(1.0 / np.asarray(odds, dtype=float), gamma)
    return x / x.sum()


# ---------------------------------------------------------------- T0 / T1
def pair_index(n: int):
    a, b = np.triu_indices(n, k=1)
    return a, b


def stern_top2(pi, lam: float = 1.0) -> np.ndarray:
    """べき割引 Harville の順不同 top-two 確率。順序は pair_index(n)。λ=1 で Harville (T0)。
    P(a 1着, b 2着) = π_a · s_b / (S − s_a),  s = π^λ,  S = Σ s"""
    pi = np.asarray(pi, dtype=float)
    s = np.power(pi, lam)
    S = s.sum()
    a, b = pair_index(len(pi))
    p = pi[a] * s[b] / (S - s[a]) + pi[b] * s[a] / (S - s[b])
    return p / p.sum()


def harville_top2(pi) -> np.ndarray:
    return stern_top2(pi, 1.0)


def ordered_prefixes(n: int, k: int) -> np.ndarray:
    """長さ k の順序付き上位列 (M × k の index 配列)"""
    return np.array(list(permutations(range(n), k)), dtype=np.int64)


def stern_prefix_probs(pi, lam: float, X: np.ndarray) -> np.ndarray:
    """X (M×k) の各順序付き上位列の確率。1 段目は π、2 段目以降は s=π^λ で逐次選択"""
    pi = np.asarray(pi, dtype=float)
    s = np.power(pi, lam)
    S = s.sum()
    p = pi[X[:, 0]].copy()
    used = s[X[:, 0]].copy()
    for t in range(1, X.shape[1]):
        p *= s[X[:, t]] / (S - used)
        used += s[X[:, t]]
    return p


def place_probs(q: np.ndarray, X: np.ndarray, n: int) -> np.ndarray:
    """P(i ∈ top-k) = 順序付き上位列 q の周辺"""
    out = np.zeros(n)
    for t in range(X.shape[1]):
        np.add.at(out, X[:, t], q)
    return out


def top2_from_prefixes(q: np.ndarray, X: np.ndarray, n: int) -> np.ndarray:
    """順序付き上位列 (k>=2) → 順不同 top-two 確率 (pair_index 順)"""
    a, b = pair_index(n)
    idx = np.full((n, n), -1, dtype=np.int64)
    idx[a, b] = np.arange(len(a))
    idx[b, a] = np.arange(len(a))
    out = np.zeros(len(a))
    np.add.at(out, idx[X[:, 0], X[:, 1]], q)
    return out


# ---------------------------------------------------------------- T2
def places(n: int) -> int:
    """JRA 複勝の払戻対象着順数: 8 頭以上 3、5〜7 頭 2"""
    return 3 if n >= 8 else (2 if n >= 5 else 0)


def t2_soft_fukusho(pi, tau, lam: float, w: float, return_detail: bool = False):
    """T2: q(x) ∝ q_T1(x)·exp(μ·z_x) を先頭馬ごとのブロックで再規格化して
    P(x_1=i)=π_i を**等式**で満たし、KL(q||q_T1) + w·Σ_i (P_q(i∈top-k) − τ_i)^2 を最小化する。
    k = places(n)。w=0 は T1 (stern_top2) を同じ計算でそのまま返す (bit 一致)。"""
    pi = np.asarray(pi, dtype=float)
    n = len(pi)
    base_pairs = stern_top2(pi, lam)
    if w == 0:
        return (base_pairs, {"mu": np.zeros(n), "iters": 0}) if return_detail else base_pairs
    k = places(n)
    tau = np.asarray(tau, dtype=float)
    X = ordered_prefixes(n, k)
    q0 = stern_prefix_probs(pi, lam, X)
    q0 = q0 / q0.sum()
    Z = np.zeros((len(X), n))
    for t in range(k):
        Z[np.arange(len(X)), X[:, t]] = 1.0
    first = X[:, 0]
    logq0 = np.log(q0)

    def q_of(mu):
        e = logq0 + Z @ mu
        e = e - e.max()
        u = np.exp(e)
        blk = np.bincount(first, weights=u, minlength=n)
        return u * (pi[first] / blk[first])

    def obj(mu):
        q = q_of(mu)
        lr = np.log(q) - logq0
        P = Z.T @ q
        # 勾配: g_xj = z_xj − E[z_j | x_1]
        blkE = np.zeros((n, n))
        np.add.at(blkE, first, q[:, None] * Z)
        condE = blkE / pi[:, None]
        G = Z - condE[first]
        dKL = G.T @ (q * lr)
        dP = (G * q[:, None]).T @ Z            # dP[j, i] = ∂P_i/∂μ_j
        r = P - tau
        J = float((q * lr).sum() + w * (r ** 2).sum())
        g = dKL + 2 * w * dP @ r
        return J, g

    res = minimize(obj, np.zeros(n), jac=True, method="L-BFGS-B",
                   options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-10})
    q = q_of(res.x)
    pairs = top2_from_prefixes(q, X, n)
    pairs = pairs / pairs.sum()
    if return_detail:
        return pairs, {"mu": res.x, "iters": int(res.nit), "converged": bool(res.success),
                       "place": Z.T @ q, "win": np.bincount(first, weights=q, minlength=n)}
    return pairs


# ---------------------------------------------------------------- oracle
def oracle_top2_topk(pi, lam: float, k: int):
    """n<=8: 全順列を列挙し、逐次べき割引モデルの順不同 top-two と top-k 包含確率を返す"""
    pi = np.asarray(pi, dtype=float)
    n = len(pi)
    assert n <= 8
    s = np.power(pi, lam)
    a, b = pair_index(n)
    idx = {(int(x), int(y)): t for t, (x, y) in enumerate(zip(a, b))}
    top2 = np.zeros(len(a))
    topk = np.zeros(n)
    for perm in permutations(range(n)):
        p = pi[perm[0]]
        rem = s.sum() - s[perm[0]]
        for t in range(1, n):
            p *= s[perm[t]] / rem
            rem -= s[perm[t]]
        x, y = sorted(perm[:2])
        top2[idx[(x, y)]] += p
        for t in range(k):
            topk[perm[t]] += p
    return top2, topk


# ---------------------------------------------------------------- 温度 null / cross-pool 代替
def temp_probs(m, a0: float) -> np.ndarray:
    """q_temp ∝ m^a0"""
    x = a0 * np.log(np.asarray(m, dtype=float))
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def cross_probs(m, qlopo, a: float, beta: float, collinear: bool) -> np.ndarray:
    """q_cross ∝ m^a · q_LOPO^β。共線 (collinear=True) なら同じコードパスで q_temp を返す"""
    if collinear:
        return temp_probs(m, a)
    x = a * np.log(np.asarray(m, dtype=float)) + beta * np.log(np.asarray(qlopo, dtype=float))
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _segments(offsets):
    return np.asarray(offsets[:-1]), np.diff(offsets)


def _clogit_newton(X, offsets, win_rows, theta0, iters=50):
    """race セグメント上の条件付きロジット (切片なし) を Newton で解く"""
    start, cnt = _segments(offsets)
    R = len(cnt)
    th = np.array(theta0, dtype=float)
    for _ in range(iters):
        eta = X @ th
        mx = np.maximum.reduceat(eta, start)
        e = np.exp(eta - np.repeat(mx, cnt))
        Zs = np.add.reduceat(e, start)
        p = e / np.repeat(Zs, cnt)
        Ex = np.add.reduceat(p[:, None] * X, start)
        g = X[win_rows].sum(0) - Ex.sum(0)
        Exx = np.add.reduceat(p[:, None, None] * X[:, :, None] * X[:, None, :], start)
        H = -(Exx - Ex[:, :, None] * Ex[:, None, :]).sum(0)
        step = np.linalg.solve(H, g)
        th = th - step
        if np.max(np.abs(step)) < 1e-12:
            break
    ll = float((X[win_rows] @ th).sum() - (np.log(Zs) + mx).sum())
    return th, ll / R


def fit_temp(logm, offsets, win_rows) -> float:
    """温度 null の a0 (q_temp ∝ m^a0)"""
    th, _ = _clogit_newton(np.asarray(logm, float)[:, None], offsets, win_rows, [1.0])
    return float(th[0])


def centered(x, offsets):
    start, cnt = _segments(offsets)
    mu = np.add.reduceat(x, start) / cnt
    return x - np.repeat(mu, cnt)


def fit_cross(logm, logq, offsets, win_rows) -> dict:
    """cross-pool 代替 (a, β)。race 内中心化した [log m, log q] が rank 落ち / 条件数 > 1e12 なら
    重複列 (log q) を落とし、温度 null と同じ a0 を返して collinear=True とする"""
    logm = np.asarray(logm, float)
    logq = np.asarray(logq, float)
    cm, cq = centered(logm, offsets), centered(logq, offsets)
    D = np.column_stack([cm, cq])
    rank = np.linalg.matrix_rank(D)
    cond = np.linalg.cond(D.T @ D) if rank == 2 else np.inf
    if rank < 2 or not np.isfinite(cond) or cond > COND_MAX:
        a0 = fit_temp(logm, offsets, win_rows)
        return {"a": a0, "beta": 0.0, "collinear": True, "rank": int(rank), "cond": float(cond)}
    th, _ = _clogit_newton(np.column_stack([logm, logq]), offsets, win_rows, [1.0, 0.0])
    return {"a": float(th[0]), "beta": float(th[1]), "collinear": False, "rank": int(rank),
            "cond": float(cond)}


# ---------------------------------------------------------------- 現金込み Kelly (排反事象)
def kelly_cash_growth(q, odds, p_true) -> tuple[float, float, int]:
    """信念 q・倍率 odds で現金込み Kelly (排反事象の閉形式) を組み、真の p_true で期待対数成長を返す。
    戻り: (growth, stake_fraction, n_selected)。賭けない場合は (0, 0, 0)"""
    q = np.asarray(q, float)
    o = np.asarray(odds, float)
    p = np.asarray(p_true, float)
    ev = q * o
    order = np.argsort(-ev)
    ps = 0.0
    sig = 0.0
    R = 1.0
    sel = []
    for i in order:
        if ev[i] <= R:
            break
        ps_new, sig_new = ps + q[i], sig + 1.0 / o[i]
        if sig_new >= 1.0:
            break
        R_new = (1.0 - ps_new) / (1.0 - sig_new)
        if ev[i] <= R_new:
            break
        ps, sig, R = ps_new, sig_new, R_new
        sel.append(i)
    if not sel:
        return 0.0, 0.0, 0
    sel = np.array(sel)
    stake = float(q[sel].sum() - R * (1.0 / o[sel]).sum())
    g = float((p[sel] * np.log(ev[sel])).sum() + (1.0 - p[sel].sum()) * np.log(R))
    return g, stake, len(sel)
