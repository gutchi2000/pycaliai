# -*- coding: utf-8 -*-
"""
methods.py — offset付きロジスティック回帰、I1(環境安定性選別)、I2(環境ロバスト学習)、M5(通常top-k選択)
========================================================================================================
すべて「outcome ~ offset(v6+市場) + Σβ_j x_j」の形で、offset の係数は1に固定し β だけを学習する
(sklearn の LogisticRegression は offset をサポートしないため自前で実装)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

EPS = 1e-9


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


# ------------------------------------------------------------- 多変量 offset-logit
def fit_offset_logit(X: np.ndarray, offset: np.ndarray, y: np.ndarray, l2: float,
                     beta0: np.ndarray | None = None, weight: np.ndarray | None = None) -> np.ndarray:
    """(重み付き) L2正則化ロジスティック回帰 (offsetの係数=1固定)。βを返す。
    weight: サンプル重み (平均1に正規化されていなくてもよい、内部で正規化する)。"""
    n, p = X.shape
    b0 = np.zeros(p) if beta0 is None else beta0
    w = np.ones(n) if weight is None else (weight / weight.mean())

    def nll_grad(beta):
        z = offset + X @ beta
        pr = sigmoid(z)
        nll = -np.mean(w * (y * np.log(np.clip(pr, EPS, 1)) + (1 - y) * np.log(np.clip(1 - pr, EPS, 1))))
        nll += 0.5 * l2 * np.sum(beta ** 2)
        grad = X.T @ (w * (pr - y)) / n + l2 * beta
        return nll, grad

    res = minimize(nll_grad, b0, jac=True, method="L-BFGS-B", options={"maxiter": 300})
    return res.x


def predict_offset_logit(X: np.ndarray, offset: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return sigmoid(offset + X @ beta)


def select_c(X, offset, y, train, sel, grid=(0.01, 0.1, 1.0), weight=None) -> tuple[np.ndarray, float]:
    """C(逆正則化強度)を selection 期間の logloss で選ぶ。l2 = 1/(C*n_train)。戻り値 (beta, C)。"""
    n_tr = int(train.sum())
    best = None
    w = np.ones(len(y)) if weight is None else weight
    for C in grid:
        l2 = 1.0 / (C * n_tr)
        beta = fit_offset_logit(X[train], offset[train], y[train], l2)
        p = predict_offset_logit(X[sel], offset[sel], beta)
        ll = -np.mean(y[sel] * np.log(np.clip(p, EPS, 1)) + (1 - y[sel]) * np.log(np.clip(1 - p, EPS, 1)))
        if best is None or ll < best[0]:
            best = (ll, beta, C)
    return best[1], best[2]


# --------------------------------------------------------- 高速な単変量 offset-logit (Newton法)
def fit_univariate(x: np.ndarray, offset: np.ndarray, y: np.ndarray, weight: np.ndarray | None = None,
                   n_iter: int = 10) -> tuple[float, float]:
    """1特徴だけの offset-logit を Newton法で解く。(beta, se) を返す。se は Wald近似の標準誤差。"""
    w = np.ones(len(y)) if weight is None else weight
    beta = 0.0
    for _ in range(n_iter):
        z = offset + beta * x
        p = sigmoid(z)
        wt = w * p * (1 - p)
        grad = np.sum(w * x * (y - p))
        hess = -np.sum(wt * x * x) - 1e-6
        step = grad / hess
        beta -= step
        if abs(step) < 1e-7:
            break
    z = offset + beta * x
    p = sigmoid(z)
    wt = w * p * (1 - p)
    info = np.sum(wt * x * x) + 1e-9
    se = float(np.sqrt(1.0 / info))
    return float(beta), se


# --------------------------------------------------------------------- I1: 環境安定性選別
def env_stability_select(Xc: dict[str, np.ndarray], offset: np.ndarray, y: np.ndarray,
                         train: np.ndarray, envs: pd.DataFrame, days: np.ndarray,
                         min_level_races: int = 300, sign_frac_thr: float = 0.70,
                         concentration_cap: float = 0.50, boot_reps: int = 200,
                         boot_sign_thr: float = 0.90, fdr_q: float = 0.10,
                         seed: int = 0) -> dict:
    """特徴ごとに year/course/surfdist それぞれの効果方向安定性 + block bootstrap 頑健性を見て選ぶ。
    Xc: {feature_name: 1次元array (標準化前)}。train は学習fold (bool)。
    戻り値: {feature: {...診断値...}}、選ばれたものは "selected": True。
    """
    rng = np.random.default_rng(seed)
    # ★候補特徴は欠損を含む (最大61%)。np.mean/std は NaN を伝播させて全結果を壊すので
    #   nanmean/nanstd で標準化した後、残る NaN は標準化後の 0 (=train平均) で埋める。
    def _std_impute(v):
        m, s = np.nanmean(v[train]), np.nanstd(v[train])
        z = (v - m) / (s + 1e-9)
        return np.where(np.isnan(z), 0.0, z)
    x_std = {k: _std_impute(v) for k, v in Xc.items()}

    uniq_days = np.unique(days[train])
    day_idx = {d: np.where((days == d) & train)[0] for d in uniq_days}

    res = {}
    pvals = []
    names = list(Xc.keys())
    for name in names:
        x = x_std[name]
        beta_full, _ = fit_univariate(x[train], offset[train], y[train])
        axis_stats = {}
        for axis in ["e1_year", "e2_course", "e3_surfdist"]:
            levels = envs.loc[train, axis].unique()
            signs, mags, sizes = [], [], []
            for lv in levels:
                m = train & (envs[axis] == lv).to_numpy()
                if m.sum() < min_level_races:
                    continue
                b, _ = fit_univariate(x[m], offset[m], y[m])
                signs.append(np.sign(b) if b != 0 else 0)
                mags.append(abs(b))
                sizes.append(int(m.sum()))
            if len(signs) == 0:
                axis_stats[axis] = {"n_levels": 0, "same_sign_frac": 0.0, "max_share": 1.0}
                continue
            ref = np.sign(beta_full) if beta_full != 0 else 0
            same = np.mean([s == ref for s in signs]) if ref != 0 else 0.0
            tot_mag = sum(mags) + 1e-12
            max_share = max(mags) / tot_mag if mags else 1.0
            axis_stats[axis] = {"n_levels": len(signs), "same_sign_frac": float(same),
                                "max_share": float(max_share)}
        # block bootstrap (開催日ブロック、全体でのβの符号安定性)
        bsigns = np.empty(boot_reps)
        for b in range(boot_reps):
            pick = rng.choice(uniq_days, size=len(uniq_days), replace=True)
            idx = np.concatenate([day_idx[d] for d in pick])
            bb, _ = fit_univariate(x[idx], offset[idx], y[idx])
            bsigns[b] = np.sign(bb)
        frac_pos = np.mean(bsigns > 0)
        frac_neg = np.mean(bsigns < 0)
        boot_stable = max(frac_pos, frac_neg)
        pval = 2 * min(frac_pos, frac_neg)
        pval = min(pval, 1.0) if pval > 0 else 1.0 / boot_reps
        pvals.append(pval)
        stable_axes = all(axis_stats[a]["n_levels"] < 2 or
                          (axis_stats[a]["same_sign_frac"] >= sign_frac_thr and
                           axis_stats[a]["max_share"] < concentration_cap)
                          for a in axis_stats)
        res[name] = {"beta_full_train": beta_full, "axis_stats": axis_stats,
                    "boot_stable_frac": float(boot_stable), "boot_pval_proxy": float(pval),
                    "passes_axis_stability": bool(stable_axes),
                    "passes_boot_stability": bool(boot_stable >= boot_sign_thr)}

    # BH-FDR (多重比較補正)
    p = np.array(pvals)
    order = np.argsort(p)
    m = len(p)
    thresh = fdr_q * (np.arange(1, m + 1) / m)
    passed = p[order] <= thresh
    keep = np.zeros(m, bool)
    if passed.any():
        kmax = np.max(np.where(passed)[0])
        keep[order[:kmax + 1]] = True
    for i, name in enumerate(names):
        res[name]["passes_fdr"] = bool(keep[i])
        res[name]["selected"] = bool(res[name]["passes_axis_stability"]
                                     and res[name]["passes_boot_stability"]
                                     and res[name]["passes_fdr"])
    return res


# --------------------------------------------------------------------- M5: 通常のtop-k選択
def pooled_topk(Xc: dict[str, np.ndarray], offset: np.ndarray, y: np.ndarray, train: np.ndarray,
                k: int) -> list[str]:
    """環境安定性を使わない通常選択: train全体でのWald |t統計量| が大きい上位k特徴。"""
    stats = []
    for name, x in Xc.items():
        m, s = np.nanmean(x[train]), np.nanstd(x[train])
        xs = (x - m) / (s + 1e-9)
        xs = np.where(np.isnan(xs), 0.0, xs)
        b, se = fit_univariate(xs[train], offset[train], y[train])
        t = abs(b) / (se + 1e-12)
        stats.append((name, t))
    stats.sort(key=lambda t: -t[1])
    return [n for n, _ in stats[:k]]


# --------------------------------------------------------------------- I2: 環境ロバスト学習 (Group-DRO風)
def group_dro_fit(X: np.ndarray, offset: np.ndarray, y: np.ndarray, train: np.ndarray,
                  group_ids: list[np.ndarray], l2: float, eta: float = 1.0,
                  n_outer: int = 15, min_group: int = 200) -> np.ndarray:
    """複数の環境分割 (group_ids: 年度/競馬場/芝ダ距離帯 の3つの長さnラベル配列) の
    どのグループの損失も極端に悪化しないよう、指数化勾配法でグループ重み q を更新する
    Group-DRO (Sagawa et al. 2020) の交互最適化:
      (a) 現在の q から作ったサンプル重みで offset-logit を収束するまで解く (fit_offset_logit)
      (b) 解いた β のもとでの各グループの損失を計算し、q ← q・exp(η・loss) を正規化して更新
    を n_outer 回繰り返す。1回ごとに (a) を毎回収束まで解くので、途中で打ち切って
    非DROの解へ緩和されることはない。train のみで反復し、市場/ROI は見ない。
    """
    tr_idx = np.where(train)[0]
    Xtr, ytr, otr = X[tr_idx], y[tr_idx], offset[tr_idx]
    n = len(ytr)
    groups = []
    for g in group_ids:
        gtr = g[tr_idx]
        for lv in np.unique(gtr):
            idx = np.where(gtr == lv)[0]
            if len(idx) >= min_group:
                groups.append(idx)
    q = np.ones(len(groups)) / len(groups)
    beta = np.zeros(X.shape[1])
    for _ in range(n_outer):
        sw = np.full(n, 1.0 / len(groups))
        for gi, idx in enumerate(groups):
            sw[idx] += q[gi]
        beta = fit_offset_logit(Xtr, otr, ytr, l2, beta0=beta, weight=sw)
        z = otr + Xtr @ beta
        p = sigmoid(z)
        losses = np.array([
            -np.mean(ytr[idx] * np.log(np.clip(p[idx], EPS, 1)) +
                    (1 - ytr[idx]) * np.log(np.clip(1 - p[idx], EPS, 1)))
            for idx in groups])
        q = q * np.exp(eta * (losses - losses.max()))
        q = q / q.sum()
    return beta
