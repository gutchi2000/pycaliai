# -*- coding: utf-8 -*-
"""
gate4_freeenergy.py — 統計力学①の最終: 結合系の「自由エネルギー」を race特徴化し荒れを予測
==============================================================================
joint確率の補正(M1/M2/順序)は全て薄かった。ここは別ターゲット = レースの荒れ(本命崩壊/
高配当)を、結合の自由エネルギーで予測できるか。joint補正が薄くても race-level の見送り/
配分判断には効く可能性がある。

各レースの量 (marginal はオッズ系3 de-vig):
  H  = -Σ m_i log m_i                       周辺エントロピー(競争度) = ベースライン
  F1 = log Z = log Σ_pair P_M0·exp(corr)     自由エネルギー(結合の平均傾き)
  F2 = KL(M1‖M0)                             結合が分布を歪める度 = フラストレーション
  F3 = E_M0[corr]                            負=対立ペース戦が支配
  (raw: n_front, pace も比較用)

物理: 逃げ多×高ペース=強い負結合=系がフラストレート=本命崩れやすい。

検定 (train≤2022 → test2024-25 OOS):
  T1 本命崩壊: 最有力馬(argmax m)が着外(着>3) を予測する AUC
  T2 配当荒れ: log(三連複配当) を予測する R²/相関
  arm: base[H] / +raw[H,nfront,pace] / +freeE[H,F1,F2,F3]
  事前登録: +freeE が base を AUC↑/R²↑、かつ +raw も上回れば「自由エネルギー形式に価値」。

実行: python gate4_freeenergy.py   出力: reports/gate4_freeenergy.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, r2_score

from gate1_eval1 import (build_records, load_odds_devig, pair_feat_tensor,
                         add_style, MASTER, COL_RID, COL_JYUN)
from backtest_pl_ev import all_umaren_mat, load_payouts
from corr_features import PAIR_FEATURE_NAMES
from joint_calibration_v6 import apply_encoders

import io as _io
sys.stdout = _io.TextIOWrapper(open(1, "wb", closefd=False), encoding="utf-8", line_buffering=True)

BASE = Path(__file__).parent
np.random.seed(42)


def race_features(rec, beta):
    m = rec["m"]; n = len(m)
    H = float(-np.sum(m * np.log(m + 1e-300)))
    F = pair_feat_tensor(rec["style"], rec["waku"], rec["pace"], n)
    corr = F @ beta
    iu, ju = np.triu_indices(n, 1)
    p0 = all_umaren_mat(m)[iu, ju]
    p0 = p0 / p0.sum()
    cpair = corr[iu, ju]
    F1 = float(logsumexp(np.log(p0 + 1e-300) + cpair))      # log Z
    e = p0 * np.exp(cpair); m1 = e / e.sum()
    F2 = float(np.sum(m1 * (np.log(m1 + 1e-300) - np.log(p0 + 1e-300))))  # KL(M1‖M0)
    F3 = float(np.sum(p0 * cpair))                          # E_M0[corr]
    sr = np.where(np.isnan(rec["style"]), 0.5, rec["style"])
    nfront = int((sr < 0.30).sum())
    fav = int(np.argmax(m))
    fav_collapse = int(rec["jyun"][fav] > 3)
    return dict(split=rec["split"], H=H, F1=F1, F2=F2, F3=F3,
                nfront=float(nfront), pace=float(rec["pace"]),
                fav_collapse=fav_collapse, rid16=rec["rid16"])


def main():
    bundle = joblib.load(BASE / "models/unified_rank_v6.pkl")
    model, feats, encs = bundle["model"], bundle["feature_cols"], bundle["encoders"]
    beta_d = json.load(open(BASE / "reports/gate1_corr_calibration.json", encoding="utf-8"))["beta"]
    beta = np.array([beta_d[n] for n in PAIR_FEATURE_NAMES], dtype=float)

    print("[load] master + style + odds")
    df = pd.read_csv(MASTER, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]).copy()
    df = add_style(df)
    enc = apply_encoders(df, encs)
    X = enc[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    df["_score"] = model.predict(X)
    recs = build_records(df, load_odds_devig())
    payouts = load_payouts()

    rows = [race_features(r, beta) for r in recs]
    R = pd.DataFrame(rows)
    R["log_pay"] = R["rid16"].map(
        lambda r: np.log(float(payouts[r]["sanrenpuku"]))
        if (r in payouts and payouts[r]["sanrenpuku"] and not pd.isna(payouts[r]["sanrenpuku"]))
        else np.nan)
    tr = R[R["split"] == "train"]; te = R[R["split"] == "test"]
    print(f"  races: train={len(tr):,} test={len(te):,}  "
          f"本命崩壊率 train={tr['fav_collapse'].mean():.3f}/test={te['fav_collapse'].mean():.3f}")

    arms = {"base": ["H"], "+raw": ["H", "nfront", "pace"], "+freeE": ["H", "F1", "F2", "F3"]}
    out = {"beta": beta_d, "T1_fav_collapse_auc": {}, "T2_logpay_r2": {}}

    print("\n=== T1: 本命崩壊(最有力馬が着外) 予測 AUC (test OOS) ===")
    for name, cols in arms.items():
        sc = StandardScaler().fit(tr[cols])
        clf = LogisticRegression(max_iter=1000).fit(sc.transform(tr[cols]), tr["fav_collapse"])
        p = clf.predict_proba(sc.transform(te[cols]))[:, 1]
        auc = float(roc_auc_score(te["fav_collapse"], p))
        out["T1_fav_collapse_auc"][name] = auc
        print(f"  {name:8s} AUC={auc:.5f}")

    print("\n=== T2: 三連複配当(log) 予測 R² (test OOS) ===")
    trp = tr.dropna(subset=["log_pay"]); tep = te.dropna(subset=["log_pay"])
    for name, cols in arms.items():
        sc = StandardScaler().fit(trp[cols])
        reg = LinearRegression().fit(sc.transform(trp[cols]), trp["log_pay"])
        pred = reg.predict(sc.transform(tep[cols]))
        r2 = float(r2_score(tep["log_pay"], pred))
        out["T2_logpay_r2"][name] = r2
        print(f"  {name:8s} R²={r2:+.5f}")

    # 直交: free-energy 特徴が H 統制後に荒れと相関するか
    print("\n=== 直交: part_corr(F, 本命崩壊残差) on test ===")
    scH = StandardScaler().fit(tr[["H"]])
    clfH = LogisticRegression(max_iter=1000).fit(scH.transform(tr[["H"]]), tr["fav_collapse"])
    resid = te["fav_collapse"].values - clfH.predict_proba(scH.transform(te[["H"]]))[:, 1]
    pcs = {}
    for f in ["F1", "F2", "F3", "nfront", "pace"]:
        pc = float(np.corrcoef(te[f].values, resid)[0, 1])
        pcs[f] = pc
        print(f"  {f:8s} part_corr={pc:+.4f}")
    out["partcorr_resid"] = pcs

    d_auc = out["T1_fav_collapse_auc"]["+freeE"] - out["T1_fav_collapse_auc"]["base"]
    d_auc_vs_raw = out["T1_fav_collapse_auc"]["+freeE"] - out["T1_fav_collapse_auc"]["+raw"]
    d_r2 = out["T2_logpay_r2"]["+freeE"] - out["T2_logpay_r2"]["base"]
    win = d_auc > 0.003 and d_auc_vs_raw > 0
    print(f"\n=== 判定 ===")
    print(f"  T1 ΔAUC(+freeE - base)={d_auc:+.5f} / vs +raw={d_auc_vs_raw:+.5f}")
    print(f"  T2 ΔR²(+freeE - base) ={d_r2:+.5f}")
    print(f"  best|part_corr|={max(abs(v) for v in pcs.values()):.4f}")
    print(f"  → {'KEEP (自由エネルギーが荒れを予測)' if win else 'marginal/KILL (荒れはHに吸収/結合薄)'}")
    out["verdict"] = "KEEP" if win else "marginal"
    outp = BASE / "reports/gate4_freeenergy.json"
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[saved] {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
