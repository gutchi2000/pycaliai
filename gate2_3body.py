# -*- coding: utf-8 -*-
"""
gate2_3body.py — 統計力学②: PL分配関数への「多体(3体)結合項」を三連複で検定
==============================================================================
既存 M1 (gate1, 2体 pairwise 結合) の上に、pairの和に分解できない純3次相互作用を足す:

  M0 : Plackett-Luce 独立 joint (Harville)
  M1 : P_M1(S) ∝ P_M0(S)·exp( Σ_{(a,b)⊂S} β·corr_ab )           (既存, β は gate1 fit)
  M2 : P_M2(S) ∝ P_M0(S)·exp( Σ β·corr_ab + β3·T(S) )           (本実験, 新規)

      T(S) = [ 1[n_front(S)≥3]·pace ,  1[n_closer(S)≥3] ]        (純3体: pair和で書けない)
      物理: 逃げ3頭が高ペースで上位独占は、2体ペナルティの和が予言するより集団的に稀
            → β3_front ≪ 0 を期待。β3≈0 なら「集団効果は2体和に吸収済み」= kill。

検定 (OOS test=2024-25, 三連複):
  - β は gate1 の値に固定し β3 のみ条件付きロジット(offset=log M1)で train(≤2022) fit。
    → 「M1の上に3体を足して効くか」を純粋に分離。
  - ECE (joint_calibration_v6 と同一定義) と 実現ROI を M0/M1/M2 で比較。
  - 多頭逃げ(場に逃げ≥3)・高ペース 部分集合で効果集中を確認。
  marginal はオッズ系3 de-vig 固定 (gate1 と同条件, joint 構造だけを切り分け)。

実行: python gate2_3body.py   出力: reports/gate2_3body.json
"""
from __future__ import annotations
import json, sys
from itertools import combinations, permutations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

import pl_probs as PL
from gate1_eval1 import (build_records, load_odds_devig, pair_feat_tensor,
                         add_style, MASTER, COL_RID, COL_JYUN, COL_BAN)
from backtest_pl_ev import all_sanrenpuku_tensor, load_payouts
from corr_features import PAIR_FEATURE_NAMES
from joint_calibration_v6 import new_acc, acc_add, acc_summary, apply_encoders

# import 済モジュールが sys.stdout を TextIOWrapper で二重wrapし元バッファを閉じる罠を回避:
# fd 1 を開き直して健全な utf-8 stdout を再確立する。
import io as _io
sys.stdout = _io.TextIOWrapper(open(1, "wb", closefd=False), encoding="utf-8", line_buffering=True)

BASE = Path(__file__).parent
np.random.seed(42)
KS = [3, 6, 10]
UNIT = 100.0
FRONT_TH, CLOSER_TH = 0.30, 0.70
T_NAMES = ["3front_x_pace", "3closer"]


def trio_tensors(rec):
    """rec → (tris, p0_trio, Fsum(T,K 2体特徴のトリオ和), T_tri(T,2 純3体), top3_idx, n_front_field)."""
    m = rec["m"]; n = len(m)
    F = pair_feat_tensor(rec["style"], rec["waku"], rec["pace"], n)  # (n,n,K)
    P3 = all_sanrenpuku_tensor(m)
    tris = np.array(list(combinations(range(n), 3)))                # (T,3)
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    p0 = np.zeros(len(tris))
    for prm in permutations(range(3)):
        p0 += P3[tris[:, prm[0]], tris[:, prm[1]], tris[:, prm[2]]]
    Fsum = F[a, b, :] + F[a, c, :] + F[b, c, :]                     # (T,K) 2体特徴のトリオ和

    sr = np.where(np.isnan(rec["style"]), 0.5, rec["style"])
    front = (sr < FRONT_TH).astype(int); closer = (sr >= CLOSER_TH).astype(int)
    nf = front[a] + front[b] + front[c]
    nc = closer[a] + closer[b] + closer[c]
    T = np.stack([(nf >= 3).astype(float) * rec["pace"],
                  (nc >= 3).astype(float)], axis=1)                 # (T,2)

    order = np.argsort(rec["jyun"]); top3 = set(int(x) for x in order[:3])
    top3_idx = next((k for k in range(len(tris))
                     if set(int(x) for x in tris[k]) == top3), None)
    return tris, p0, Fsum, T, top3_idx, int(front.sum())


def _cond_logit(data, dim, offsets):
    """data=[(design(T,dim), offset(T,), win_idx)]。offset 固定で係数を MLE。"""
    def nll(b):
        tot = 0.0
        for design, offset, wi in data:
            u = offset + design @ b
            tot += logsumexp(u) - u[wi]
        return tot
    res = minimize(nll, np.zeros(dim), method="L-BFGS-B", options={"maxiter": 300})
    return res.x


def _sample(recs, max_races):
    rng = np.random.default_rng(42)
    if len(recs) > max_races:
        return [recs[i] for i in rng.choice(len(recs), max_races, replace=False)]
    return recs


def fit_beta3(train_recs, beta, max_races=12000):
    """β3 のみを MLE。offset=log M1(=log p0 + Fsum@beta) 固定 → M1 への純3体上乗せを分離。"""
    data = []
    for r in _sample(train_recs, max_races):
        tris, p0, Fsum, T, wi, _ = trio_tensors(r)
        if wi is None or len(tris) < 2:
            continue
        offset = np.log(p0 + 1e-300) + Fsum @ beta
        data.append((T, offset, wi))
    return _cond_logit(data, len(T_NAMES), None), len(data)


def fit_joint(train_recs, max_races=12000):
    """β(2体K個) と β3(3体2個) を trio 尤度で同時 MLE。offset=log p0 のみ。
    gate1 の β は umaren-fit で三連複に準最適 → trio タスクに最適化し直す。"""
    data = []
    for r in _sample(train_recs, max_races):
        tris, p0, Fsum, T, wi, _ = trio_tensors(r)
        if wi is None or len(tris) < 2:
            continue
        design = np.concatenate([Fsum, T], axis=1)                  # (T, K+2)
        offset = np.log(p0 + 1e-300)
        data.append((design, offset, wi))
    K = len(PAIR_FEATURE_NAMES)
    return _cond_logit(data, K + len(T_NAMES), None), len(data)


def main():
    bundle = joblib.load(BASE / "models/unified_rank_v6.pkl")
    model, feats, encs = bundle["model"], bundle["feature_cols"], bundle["encoders"]
    beta_d = json.load(open(BASE / "reports/gate1_corr_calibration.json", encoding="utf-8"))["beta"]
    beta = np.array([beta_d[n] for n in PAIR_FEATURE_NAMES], dtype=float)
    print("[beta(2体, gate1固定)]", {n: round(beta_d[n], 3) for n in PAIR_FEATURE_NAMES})

    print("[load] master + style + score + odds(系3 de-vig)")
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
    print(f"  races: train={len(train_recs):,} test={len(test_recs):,}  payout={len(payouts):,}")

    KK = len(PAIR_FEATURE_NAMES)
    b3, n_fit = fit_beta3(train_recs, beta)
    bjoint, n_fitj = fit_joint(train_recs)
    bj_pair, bj_3 = bjoint[:KK], bjoint[KK:]
    print(f"\n[fit β3 (M1上乗せ)] n={n_fit:,}: " +
          ", ".join(f"{nm}={v:+.3f}" for nm, v in zip(T_NAMES, b3)))
    print(f"[fit joint (trio尤度で β+β3 同時)] n={n_fitj:,}")
    print("   β_pair:", {n: round(float(v), 3) for n, v in zip(PAIR_FEATURE_NAMES, bj_pair)})
    print("   β3    :", {n: round(float(v), 3) for n, v in zip(T_NAMES, bj_3)})

    MODELS = ["M0", "M1", "M2", "M2j"]
    ece = {v: new_acc() for v in MODELS}
    ece_mf = {v: new_acc() for v in MODELS}   # 場に逃げ≥3
    val = {v: {k: {"stake": 0.0, "ret": 0.0, "bets": 0, "hits": 0} for k in KS}
           for v in MODELS}

    def _norm(x):
        s = x.sum()
        return x / s if s > 0 else np.full_like(x, 1.0 / len(x))

    for rec in test_recs:
        po = payouts.get(rec["rid16"])
        tris, p0, Fsum, T, wi, nfield = trio_tensors(rec)
        if wi is None:
            continue
        corr = Fsum @ beta
        m0 = _norm(p0)
        m1 = _norm(p0 * np.exp(corr))
        m2 = _norm(p0 * np.exp(corr + T @ b3))
        m2j = _norm(p0 * np.exp(Fsum @ bj_pair + T @ bj_3))
        y = np.zeros(len(tris)); y[wi] = 1.0
        for v, p in (("M0", m0), ("M1", m1), ("M2", m2), ("M2j", m2j)):
            acc_add(ece[v], p, y)
            if nfield >= 3:
                acc_add(ece_mf[v], p, y)
        sp = po.get("sanrenpuku") if po else None
        sp = float(sp) if (sp and not pd.isna(sp)) else None
        if sp is None:
            continue
        for k in KS:
            for v, p in (("M0", m0), ("M1", m1), ("M2", m2), ("M2j", m2j)):
                sel = np.argsort(-p)[:k]
                a = val[v][k]
                a["stake"] += UNIT * len(sel); a["bets"] += len(sel)
                if wi in sel:
                    a["ret"] += sp; a["hits"] += 1

    def roi(a):
        return dict(roi=round(a["ret"] / a["stake"], 4) if a["stake"] else 0.0,
                    hit_rate=round(a["hits"] / a["bets"], 5) if a["bets"] else 0.0,
                    bets=a["bets"])

    ec = {v: acc_summary(ece[v]) for v in MODELS}
    ecmf = {v: acc_summary(ece_mf[v]) for v in MODELS}
    print("\n=== 三連複 ECE (test OOS, 小さいほど良) ===")
    print("  " + "  ".join(f"{v}={ec[v]['ece']:.6f}" for v in MODELS) + "   (全体)")
    print("  " + "  ".join(f"{v}={ecmf[v]['ece']:.6f}" if ecmf[v] else f"{v}=n/a" for v in MODELS) + "   (逃げ≥3)")

    print("\n=== 三連複 実現ROI (prob-first top-k フラット) ===")
    print(f"  {'K':>3s} {'M0':>7s} {'M1':>7s} {'M2':>7s} {'M2j':>7s}  {'best-M0':>8s}")
    vout = {}
    for k in KS:
        r = {v: roi(val[v][k]) for v in MODELS}
        best = max(r[v]["roi"] for v in MODELS)
        vout[f"K{k}"] = {v: r[v] for v in MODELS}
        vout[f"K{k}"]["d_best_M0"] = round(best - r["M0"]["roi"], 4)
        print(f"  {k:>3d} " + " ".join(f"{r[v]['roi']*100:>6.1f}%" for v in MODELS) +
              f"  {(best - r['M0']['roi'])*100:>+7.1f}pt")

    # --- 判定 ---
    d_ece_j = ec["M0"]["ece"] - ec["M2j"]["ece"]            # >0 = joint が独立を上回る
    d_roi_j = np.mean([vout[f"K{k}"]["M2j"]["roi"] - vout[f"K{k}"]["M0"]["roi"] for k in KS])
    d_roi_2 = np.mean([vout[f"K{k}"]["M2"]["roi"] - vout[f"K{k}"]["M1"]["roi"] for k in KS])
    print("\n=== 判定 ===")
    print(f"  3体上乗せ ΔROI(M2-M1)平均 = {d_roi_2*100:+.2f}pt")
    print(f"  joint refit: ΔECE(M0-M2j)={d_ece_j:+.6f} ({'M0超え' if d_ece_j>0 else 'M0未満'}) / "
          f"ΔROI(M2j-M0)平均={d_roi_j*100:+.2f}pt")
    win = d_ece_j > 0 and d_roi_j > 0
    print(f"  → joint refit は {'KEEP (独立M0を校正・ROI両面で超えた)' if win else '独立M0を両面では超えず'}")

    out = dict(beta_pair=beta_d, beta3=dict(zip(T_NAMES, b3.tolist())),
               beta_joint=dict(zip(list(PAIR_FEATURE_NAMES) + T_NAMES, bjoint.tolist())),
               n_fit=n_fit,
               ece={"all": {v: ec[v]["ece"] for v in MODELS},
                    "multifront": {v: (ecmf[v]["ece"] if ecmf[v] else None) for v in MODELS}},
               value=vout, d_ece_joint_vs_m0=float(d_ece_j),
               d_roi_joint_vs_m0=float(d_roi_j), d_roi_3body=float(d_roi_2),
               verdict="KEEP" if win else "marginal")
    outp = BASE / "reports/gate2_3body.json"
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[saved] {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
