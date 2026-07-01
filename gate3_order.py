# -*- coding: utf-8 -*-
"""
gate3_order.py — 統計力学①の順序依存版: 脚質→着位置の非対称結合を 馬単/三連単 で検定
==============================================================================
M1/M2(gate1/gate2) は対称結合で「どの集合が上位か」しか補正しない(集合内順序は素PL)。
ここでは順序付き joint に脚質の位置非対称性を入れる:

  lead propensity  L_i = 0.5 - style_rank_i   (逃げ>0, 差し<0, style NaN→L=0中立)
  順序特徴 (位置重み 1着+1/中0/3着-1):
     馬単  φ(i→j)   = L_i - L_j
     三連単 φ(i→j→k) = L_i - L_k
  M_order(順列) ∝ P_PL(順列) · exp( α·φ + α_p·φ·pace )

  物理: 遅ペース=逃げ先着(α>0) / 速ペース=差し差し切り(α_p<0 で符号反転)。
        素PLはスコア比のみで順序を決め、この構造を持たない。

α は馬単の順序付き条件付きロジット(offset=log PL馬単)で train(≤2022) fit。
同じ α を馬単・三連単に適用し OOS(2024-25) で ECE と実現ROI を M0 vs M_order 比較。
marginal はオッズ系3 de-vig 固定(gate1/2 と同条件)。

実行: python gate3_order.py   出力: reports/gate3_order.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from gate1_eval1 import (build_records, load_odds_devig, add_style,
                         MASTER, COL_RID, COL_JYUN, COL_BAN)
from backtest_pl_ev import all_umatan_mat, all_sanrenpuku_tensor, load_payouts
from joint_calibration_v6 import new_acc, acc_add, acc_summary, apply_encoders

import io as _io
sys.stdout = _io.TextIOWrapper(open(1, "wb", closefd=False), encoding="utf-8", line_buffering=True)

BASE = Path(__file__).parent
np.random.seed(42)
UNIT = 100.0
KS_UT = [3, 6]
KS_ST = [12, 24]


def lead_prop(rec):
    sr = np.where(np.isnan(rec["style"]), 0.5, rec["style"])
    return 0.5 - sr                                  # 逃げ>0 / 差し<0


def umatan_arrays(rec):
    """馬単 順序付きペア (i≠j) の (offset=logPL, design[φ,φ·pace], win_idx)."""
    m = rec["m"]; n = len(m); L = lead_prop(rec)
    P = all_umatan_mat(m)                            # (n,n) P(i→j)
    I, J = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    mask = I != J
    ii, jj = I[mask], J[mask]
    offset = np.log(P[ii, jj] + 1e-300)
    phi = L[ii] - L[jj]
    design = np.stack([phi, phi * rec["pace"]], axis=1)
    order = np.argsort(rec["jyun"]); win, plc = int(order[0]), int(order[1])
    wi = np.where((ii == win) & (jj == plc))[0]
    return offset, design, (int(wi[0]) if len(wi) else None), (ii, jj)


def sanrentan_arrays(rec):
    """三連単 順序付きトリオ (i,j,k distinct) の (offset, design, win_idx, idxs)."""
    m = rec["m"]; n = len(m); L = lead_prop(rec)
    P3 = all_sanrenpuku_tensor(m)                    # (n,n,n) P(i→j→k)
    idx = np.arange(n)
    I, J, K = np.meshgrid(idx, idx, idx, indexing="ij")
    mask = (I != J) & (I != K) & (J != K)
    ii, jj, kk = I[mask], J[mask], K[mask]
    offset = np.log(P3[ii, jj, kk] + 1e-300)
    phi = L[ii] - L[kk]                              # 位置重み 1着+1 / 3着-1
    design = np.stack([phi, phi * rec["pace"]], axis=1)
    order = np.argsort(rec["jyun"]); w, p, s = int(order[0]), int(order[1]), int(order[2])
    wi = np.where((ii == w) & (jj == p) & (kk == s))[0]
    return offset, design, (int(wi[0]) if len(wi) else None), (ii, jj, kk)


def fit_alpha(train_recs, max_races=12000):
    rng = np.random.default_rng(42)
    recs = train_recs
    if len(recs) > max_races:
        recs = [recs[i] for i in rng.choice(len(recs), max_races, replace=False)]
    data = []
    for r in recs:
        offset, design, wi, _ = umatan_arrays(r)
        if wi is not None:
            data.append((offset, design, wi))

    def nll(a):
        tot = 0.0
        for offset, design, wi in data:
            u = offset + design @ a
            tot += logsumexp(u) - u[wi]
        return tot

    res = minimize(nll, np.zeros(2), method="L-BFGS-B", options={"maxiter": 300})
    return res.x, len(data)


def main():
    bundle = joblib.load(BASE / "models/unified_rank_v6.pkl")
    model, feats, encs = bundle["model"], bundle["feature_cols"], bundle["encoders"]
    print("[load] master + style + odds(系3 de-vig)")
    df = pd.read_csv(MASTER, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]).copy()
    df = add_style(df)
    enc = apply_encoders(df, encs)
    X = enc[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    df["_score"] = model.predict(X)
    recs = build_records(df, load_odds_devig())
    payouts = load_payouts()
    train_recs = [r for r in recs if r["split"] == "train"]
    test_recs = [r for r in recs if r["split"] == "test"]
    print(f"  races: train={len(train_recs):,} test={len(test_recs):,}")

    alpha, n_fit = fit_alpha(train_recs)
    print(f"\n[fit α 馬単順序] n={n_fit:,}")
    print(f"   α (φ=L_i-L_j, 遅ペース逃げ先着なら>0) = {alpha[0]:+.4f}")
    print(f"   α_pace (φ·pace, 速ペース差し差し切りなら<0) = {alpha[1]:+.4f}")

    # --- eval ---
    def newv(KS): return {k: {"stake": 0.0, "ret": 0.0, "bets": 0, "hits": 0} for k in KS}
    ece = {bt: {"M0": new_acc(), "Mord": new_acc()} for bt in ["umatan", "sanrentan"]}
    val = {"umatan": {"M0": newv(KS_UT), "Mord": newv(KS_UT)},
           "sanrentan": {"M0": newv(KS_ST), "Mord": newv(KS_ST)}}

    def add_eval(bt, offset, design, wi, KS, pay):
        p0 = np.exp(offset); p0 = p0 / p0.sum()
        e = np.exp(offset + design @ alpha); mo = e / e.sum()
        y = np.zeros(len(p0)); y[wi] = 1.0
        acc_add(ece[bt]["M0"], p0, y); acc_add(ece[bt]["Mord"], mo, y)
        if pay is None:
            return
        for k in KS:
            for v, p in (("M0", p0), ("Mord", mo)):
                sel = np.argsort(-p)[:k]
                a = val[bt][v][k]
                a["stake"] += UNIT * len(sel); a["bets"] += len(sel)
                if wi in sel:
                    a["ret"] += pay; a["hits"] += 1

    for rec in test_recs:
        po = payouts.get(rec["rid16"])
        o1, d1, w1, _ = umatan_arrays(rec)
        if w1 is not None:
            up = po.get("umatan") if po else None
            up = float(up) if (up and not pd.isna(up)) else None
            add_eval("umatan", o1, d1, w1, KS_UT, up)
        o2, d2, w2, _ = sanrentan_arrays(rec)
        if w2 is not None:
            sp = po.get("sanrentan") if po else None
            sp = float(sp) if (sp and not pd.isna(sp)) else None
            add_eval("sanrentan", o2, d2, w2, KS_ST, sp)

    def roi(a):
        return dict(roi=round(a["ret"] / a["stake"], 4) if a["stake"] else 0.0,
                    hit_rate=round(a["hits"] / a["bets"], 5) if a["bets"] else 0.0, bets=a["bets"])

    out = {"alpha": {"order": float(alpha[0]), "order_x_pace": float(alpha[1])},
           "n_fit": n_fit, "ece": {}, "value": {}}
    print("\n=== ECE (test OOS, 小さいほど良) ===")
    for bt in ["umatan", "sanrentan"]:
        e0 = acc_summary(ece[bt]["M0"]); eo = acc_summary(ece[bt]["Mord"])
        out["ece"][bt] = {"M0": e0["ece"], "Mord": eo["ece"], "delta": e0["ece"] - eo["ece"]}
        print(f"  {bt:10s} M0={e0['ece']:.6f}  Mord={eo['ece']:.6f}  "
              f"Δ(M0-Mord)={e0['ece']-eo['ece']:+.6f} ({'改善' if e0['ece']>eo['ece'] else '悪化'})")

    print("\n=== 実現ROI (prob-first top-k) ===")
    for bt, KS in (("umatan", KS_UT), ("sanrentan", KS_ST)):
        out["value"][bt] = {}
        print(f"  [{bt}]  {'K':>3s} {'M0':>8s} {'Mord':>8s} {'Δ':>7s}")
        for k in KS:
            r0, ro = roi(val[bt]["M0"][k]), roi(val[bt]["Mord"][k])
            out["value"][bt][f"K{k}"] = {"M0": r0, "Mord": ro, "delta_roi": round(ro["roi"]-r0["roi"], 4)}
            print(f"        {k:>3d} {r0['roi']*100:>7.1f}% {ro['roi']*100:>7.1f}% "
                  f"{(ro['roi']-r0['roi'])*100:>+6.1f}pt")

    d_ece = np.mean([out["ece"][bt]["delta"] for bt in ["umatan", "sanrentan"]])
    d_roi = np.mean([out["value"][bt][f"K{k}"]["delta_roi"]
                     for bt, KS in (("umatan", KS_UT), ("sanrentan", KS_ST)) for k in KS])
    win = d_ece > 0 and d_roi > 0
    print(f"\n=== 判定 ===")
    print(f"  ΔECE平均={d_ece:+.6f} / ΔROI平均={d_roi*100:+.2f}pt → "
          f"{'KEEP (順序項に価値)' if win else 'marginal/KILL (順序も素PLに吸収)'}")
    out["verdict"] = "KEEP" if win else "marginal"
    outp = BASE / "reports/gate3_order.json"
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[saved] {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
