# -*- coding: utf-8 -*-
"""
evt_phase2.py — EVT Phase2: 厳密 N体 不等分散 Gumbel argmax を注入して直接検定
==============================================================================
理論の関数形そのものをモデルフリーで検定する (GBM・in-sample スコア混入を回避):

  各馬 X_i = μ_i + σ_i·ε_i,  ε~Gumbel(0,1)。μ_i = v6 のランクスコア (OOS, test=2024+)。
  σ_i = exp(γ · z_i),  z_i = evt_sigma_resid_gz (detrend残差σの全体z, NaN=0=中立scale1)。
  P(i=argmax) = ∫ (1/σ_i) g((x-μ_i)/σ_i) Π_{j≠i} G((x-μ_j)/σ_j) dx   を数値積分。
    G(z)=exp(-e^{-z}) (Gumbel CDF, max-stable), g(z)=e^{-z-e^{-z}}。

  等分散 (γ=0 / σ=1) のとき P=softmax(μ)=現行PL勝率 に厳密一致 (sanity)。

検定 (test の「勝利=着順1」ラベル, OOS):
  - softmax_p (等分散) vs evt_winprob(γ) (不等分散) の AUC / logloss / Brier。
  - σ注入が勝率の calibration を上げれば evt_winprob が勝つ。
  - evt_delta = evt_winprob - softmax_p が残差(win - softmax_p)を予測するか part_corr。

実行: PYTHONUTF8=1 python evt_phase2.py
出力: reports/evt_phase2.json
"""
from __future__ import annotations
import io, json, re, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

warnings.filterwarnings("ignore")
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

from backtest_pl_ev import score_test, COL_RID, COL_BAN, COL_JYUN

BASE = Path(__file__).parent
EVT = BASE / "data/evt_feats.parquet"
OUT = BASE / "reports/evt_phase2.json"
GAMMAS = [0.0, 0.2, 0.4, 0.6, 0.8]


_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy2.0で trapz→trapezoid 改名


def _rid16(x): return re.sub(r"\D", "", str(x))[:16]


def hetero_winprob(mu: np.ndarray, sigma: np.ndarray, grid_pts: int = 512, pad: float = 9.0):
    """N体 不等分散 Gumbel argmax 確率を数値積分。mu,sigma:(N,) → p:(N,)."""
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-6), sigma, 1.0)
    smax = float(sigma.max())
    lo = float(mu.min()) - pad * smax
    hi = float(mu.max()) + pad * smax
    x = np.linspace(lo, hi, grid_pts)                       # (G,)
    z = (x[None, :] - mu[:, None]) / sigma[:, None]         # (N,G)
    logG = -np.exp(-z)                                      # log Gumbel CDF
    logg = -z - np.exp(-z) - np.log(sigma)[:, None]         # log pdf (1/σ込み)
    sumlogG = logG.sum(axis=0, keepdims=True)               # (1,G)
    integrand = np.exp(logg + (sumlogG - logG))             # pdf_i · Π_{j≠i} G_j
    p = _trapz(integrand, x, axis=1)
    s = p.sum()
    return p / s if s > 0 else np.full_like(mu, 1.0 / len(mu))


def main():
    print("[load] v6 で test を score 化 (OOS)...")
    te = score_test()  # test=2024+ のみ, _score = v6 ランクスコア
    te = te.dropna(subset=[COL_JYUN]).copy()
    te["rid16"] = te[COL_RID].map(_rid16)
    te["ban"] = pd.to_numeric(te[COL_BAN], errors="coerce").astype("Int64")
    evt = pd.read_parquet(EVT)[["rid16", "ban", "evt_sigma_resid_gz"]]
    te = te.merge(evt, on=["rid16", "ban"], how="left")
    te["win"] = (pd.to_numeric(te[COL_JYUN], errors="coerce") == 1).astype(int)
    te["z"] = te["evt_sigma_resid_gz"].fillna(0.0).clip(-4, 4)
    cov = te["evt_sigma_resid_gz"].notna().mean() * 100
    print(f"[rows] test={len(te):,}  σ_resid_gz coverage={cov:.1f}%")

    # --- 各レースごとに softmax_p と evt_winprob(γ) を計算 ---
    out_rows = []
    sanity_max = 0.0
    for rid, g in te.groupby(COL_RID, sort=False):
        if len(g) < 3:
            continue
        mu = g["_score"].to_numpy(dtype=float)
        w = np.exp(mu - mu.max())
        softmax_p = w / w.sum()
        z = g["z"].to_numpy(dtype=float)
        rec = {"idx": g.index.to_numpy(), "win": g["win"].to_numpy(),
               "softmax_p": softmax_p}
        for gamma in GAMMAS:
            sigma = np.exp(gamma * z)
            p = hetero_winprob(mu, sigma)
            rec[f"p_g{gamma}"] = p
            if gamma == 0.0:
                sanity_max = max(sanity_max, float(np.abs(p - softmax_p).max()))
        out_rows.append(rec)

    print(f"[sanity] γ=0 と softmax の最大|差| = {sanity_max:.2e} (≈0 であるべき)")

    # --- フラット化して評価 ---
    win = np.concatenate([r["win"] for r in out_rows])
    soft = np.concatenate([r["softmax_p"] for r in out_rows])

    def metrics(p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return (float(roc_auc_score(win, p)), float(log_loss(win, p)),
                float(brier_score_loss(win, p)))

    print("\n=== 勝利(着順1)予測: 等分散 softmax vs 不等分散 evt_winprob(γ) ===")
    base_auc, base_ll, base_br = metrics(soft)
    print(f"  softmax (γ=0等分散)   AUC={base_auc:.5f}  logloss={base_ll:.5f}  Brier={base_br:.6f}")
    res = {"softmax": dict(auc=base_auc, logloss=base_ll, brier=base_br)}
    best = ("softmax", base_ll)
    for gamma in GAMMAS:
        if gamma == 0.0:
            continue
        p = np.concatenate([r[f"p_g{gamma}"] for r in out_rows])
        a, l, b = metrics(p)
        dll = l - base_ll
        res[f"g{gamma}"] = dict(auc=a, logloss=l, brier=b, d_auc=a - base_auc, d_logloss=dll)
        flag = "◀改善" if dll < 0 else ""
        print(f"  evt_winprob γ={gamma:<3}      AUC={a:.5f} (Δ{a-base_auc:+.5f})  "
              f"logloss={l:.5f} (Δ{dll:+.6f})  Brier={b:.6f} {flag}")
        if l < best[1]:
            best = (f"g{gamma}", l)

    # --- evt_delta が softmax 残差を予測するか (γ=0.6 で代表) ---
    g_rep = 0.6
    p_rep = np.concatenate([r[f"p_g{g_rep}"] for r in out_rows])
    delta = p_rep - soft
    resid = win - soft
    mask = np.isfinite(delta) & np.isfinite(resid)
    pc = float(np.corrcoef(delta[mask], resid[mask])[0, 1])
    print(f"\n  part_corr(evt_delta[γ={g_rep}], win - softmax_p) = {pc:+.4f}")

    print(f"\n=== 判定 ===")
    improved = best[0] != "softmax"
    print(f"  最良 logloss: {best[0]} ({best[1]:.5f})  → "
          f"{'σ注入が改善' if improved else 'σ注入は改善せず (等分散が最良)'}")

    rep = dict(rows=int(len(te)), cov_pct=float(cov), sanity_max_diff=float(sanity_max),
               results=res, best=best[0], delta_partcorr=pc, gammas=GAMMAS)
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(f"[saved] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
