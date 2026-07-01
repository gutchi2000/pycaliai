"""
pl_stern.py
===========
Stern / Lo-Bacon-Shone べき乗補正版 Plackett-Luce joint（後付け・再学習なし）。

Luce 逐次選択チェーンで:
  - 1着選択 (1st pick) は生の重み w_i = exp(s_i)  → **単勝 marginal P(i=1着)=w_i/Σw は λ に依らず不変**
  - 2着目以降 (2nd, 3rd, ...) は v_i = w_i^λ を使う
  λ<1 で下位順位の分布を平坦化し、PL の「上位人気馬が連系上位を独占しすぎる」
  Harville bias を緩和する。

  P(i→j→k) = (w_i/Σw) · (v_j/(Σv−v_i)) · (v_k/(Σv−v_i−v_j))

λ=1 (v=w) で backtest_pl_ev / pl_probs の現行 joint と厳密一致する（__main__ で検証）。

位置別 τ_m 拡張版 (2着用 τ2 / 3着用 τ3) も提供:
  v2=w^τ2 を 2着, v3=w^τ3 を 3着に使う。単一λは τ2=τ3=λ。

本番の pl_probs.py は変更しない（export/backtest/calibrator が依存するため）。
"""
from __future__ import annotations

from itertools import permutations

import numpy as np


def stern_vw(scores, lam):
    """scores → (w, v=w^lam). w は exp(s-max) で (0,1]。"""
    s = np.asarray(scores, dtype=np.float64)
    w = np.exp(s - s.max())
    v = w ** float(lam)
    return w, v


# ============================================================
# 単一 λ 版 (2着以降すべて v=w^λ)
# ============================================================
def all_tansho_stern(w, v):
    """単勝（1着）= 生 w。λ 不変。"""
    return w / w.sum()


def all_umatan_mat_stern(w, v):
    """(N,N) P(i→j) = (w_i/Σw)(v_j/(Σv−v_i))."""
    T = w.sum(); Tv = v.sum()
    P = (w[:, None] / T) * (v[None, :] / (Tv - v[:, None]))
    np.fill_diagonal(P, 0.0)
    return P


def all_umaren_mat_stern(w, v):
    P = all_umatan_mat_stern(w, v)
    return P + P.T


def all_fukusho_vec_stern(w, v):
    """各馬が top3 に入る確率（Stern 補正）。"""
    T = w.sum(); Tv = v.sum(); n = len(w)
    p1 = w / T
    # pos2: P(i=2着) = Σ_{j≠i} (w_j/T)(v_i/(Tv−v_j)) = (v_i/T)·Σ_{j≠i} w_j/(Tv−v_j)
    term = w / (Tv - v)
    S = term.sum()
    p2 = (v / T) * (S - term)
    # pos3: P(i=3着) = Σ_{j≠i}Σ_{k≠i,j} (w_j/T)(v_k/(Tv−v_j))(v_i/(Tv−v_j−v_k))
    p3 = np.zeros(n)
    for j in range(n):
        wj_T = w[j] / T
        remj = Tv - v[j]
        for k in range(n):
            if k == j:
                continue
            coef = wj_T * (v[k] / remj) / (remj - v[k])
            contrib = coef * v
            contrib[j] = 0.0; contrib[k] = 0.0
            p3 += contrib
    return p1 + p2 + p3


def all_sanrenpuku_tensor_stern(w, v):
    """三連単順序付き P3[i,j,k]=P(i→j→k)（Stern 補正）。"""
    n = len(w); T = w.sum(); Tv = v.sum()
    denom_ij = Tv - v[:, None] - v[None, :]
    denom_ij = np.where(denom_ij > 0, denom_ij, 1.0)
    P3 = (w[:, None, None] / T) * (v[None, :, None] / (Tv - v[:, None, None])) * \
         (v[None, None, :] / denom_ij[:, :, None])
    idx = np.arange(n)
    P3[idx, idx, :] = 0; P3[idx, :, idx] = 0; P3[:, idx, idx] = 0
    return P3


def all_wide_mat_stern(w, v):
    n = len(w)
    P3 = all_sanrenpuku_tensor_stern(w, v)
    Psym = np.zeros_like(P3)
    for perm in permutations(range(3)):
        Psym += np.transpose(P3, axes=perm)
    Pw = Psym.sum(axis=2)
    np.fill_diagonal(Pw, 0.0)
    return Pw


# ============================================================
# 位置別 τ 版 (2着=τ2, 3着=τ3)。τ2=τ3=λ で単一λと一致。
# ============================================================
def all_sanrenpuku_tensor_tau(w, tau2, tau3):
    n = len(w); T = w.sum()
    v2 = w ** float(tau2); v3 = w ** float(tau3)
    Tv2 = v2.sum(); Tv3 = v3.sum()
    denom_ij = Tv3 - v3[:, None] - v3[None, :]
    denom_ij = np.where(denom_ij > 0, denom_ij, 1.0)
    P3 = (w[:, None, None] / T) * (v2[None, :, None] / (Tv2 - v2[:, None, None])) * \
         (v3[None, None, :] / denom_ij[:, :, None])
    idx = np.arange(n)
    P3[idx, idx, :] = 0; P3[idx, :, idx] = 0; P3[:, idx, idx] = 0
    return P3


# ============================================================
# 観測着順 (win,plc,sho 局所index) の対数尤度 (λ 推定用)
# ============================================================
def top3_loglik(w, win, plc, sho, lam):
    """1レースの観測 top3 順序の log尤度 (Stern λ)。"""
    v = w ** float(lam)
    T = w.sum(); Tv = v.sum()
    p1 = w[win] / T
    p2 = v[plc] / (Tv - v[win])
    p3 = v[sho] / (Tv - v[win] - v[plc])
    return np.log(max(p1, 1e-300)) + np.log(max(p2, 1e-300)) + np.log(max(p3, 1e-300))


# ============================================================
# self-test: λ=1 で現行 joint と一致するか
# ============================================================
def _test():
    import pl_probs as PL
    from backtest_pl_ev import (
        all_umaren_mat, all_wide_mat, all_umatan_mat,
        all_fukusho_vec_fast, all_sanrenpuku_tensor,
    )
    rng = np.random.default_rng(42)
    for n in (5, 10, 16):
        scores = rng.normal(size=n)
        w, v = stern_vw(scores, lam=1.0)  # v=w
        assert np.allclose(all_tansho_stern(w, v), PL.all_tansho(w)), "tansho"
        assert np.allclose(all_umatan_mat_stern(w, v), all_umatan_mat(w)), "umatan"
        assert np.allclose(all_umaren_mat_stern(w, v), all_umaren_mat(w)), "umaren"
        assert np.allclose(all_fukusho_vec_stern(w, v), all_fukusho_vec_fast(w)), "fukusho"
        assert np.allclose(all_sanrenpuku_tensor_stern(w, v), all_sanrenpuku_tensor(w)), "tensor"
        assert np.allclose(all_wide_mat_stern(w, v), all_wide_mat(w)), "wide"
        # 単勝 marginal 不変チェック: λ≠1 でも P(i=1着)=w/Σw
        for lam in (0.6, 0.8, 1.2):
            w2, v2 = stern_vw(scores, lam)
            P3 = all_sanrenpuku_tensor_stern(w2, v2)
            pwin_from_joint = P3.sum(axis=(1, 2))  # Σ_jk P(i→j→k) = P(i=1着)
            assert np.allclose(pwin_from_joint, w2 / w2.sum(), atol=1e-9), \
                f"win marginal not preserved n={n} lam={lam}"
        print(f"  n={n}: λ=1 一致 OK / 単勝marginal不変 OK")
    print("pl_stern self-test 通過。")


if __name__ == "__main__":
    _test()
