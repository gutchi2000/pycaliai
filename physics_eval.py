# -*- coding: utf-8 -*-
"""
physics_eval.py — 物理特徴（Keller / OU）の直交性 + 予測上乗せを厳密評価
========================================================================
問い:
  1. 直交性: 物理特徴は既存 leak-free 特徴（脚質/ペース含む）が取りこぼす成分を持つか？
     → ベース予測を統制した部分相関 corr(feature, y - p_base) で測る。
  2. 上乗せ: fukusho_flag 予測の test(2024+) AUC/logloss を改善するか？
     → 同一 LightGBM・同一分割で baseline vs +物理 を比較。
  3. 市場示唆: 物理特徴の分位で実 top3 率にリフトが出るか（mispricing 探索の素地）。

時系列分割: split 列（train<=2022 / valid=2023 / test=2024+）。
出力: reports/physics_feats_eval.json + コンソール
実行: PYTHONUTF8=1 python physics_eval.py
"""
from __future__ import annotations
import io, json, re, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE = Path(__file__).parent
MASTER = BASE / "data/master_v2_20130105-20251228.csv"
KELLER = BASE / "data/keller_pace_feats.parquet"
OU = BASE / "data/odds_ou_feats.parquet"
OUT = BASE / "reports/physics_feats_eval.json"

from corr_features import add_style, COL_RID
COL_BAN = "馬番"

BASE_FEATS = [
    "jockey_fuku30", "jockey_fuku90", "trainer_fuku30", "trainer_fuku90",
    "horse_fuku10", "horse_fuku30", "prev_pos_rel", "closing_power",
    "kako5_avg_pos", "kako5_std_pos", "kako5_best_pos", "kako5_avg_ninki",
    "kako5_pos_vs_ninki", "kako5_avg_agari3f", "kako5_best_agari3f",
    "kako5_same_td_ratio", "kako5_same_dist_ratio", "kako5_same_place_ratio",
    "kako5_pos_trend", "kako5_race_count", "kako5_expected_good_count",
    "hist_same_cond_top3_rate", "hist_same_cond_count", "prev_hosei", "prev_hosei9",
    "trnH_Time1", "trnH_Lap1", "trnW_5F", "trnW_3F",
    "course_win_rate", "course_top3_rate", "jockey_win_rate", "jockey_top3_rate",
    "距離", "出走頭数",
    "style_rank", "pace_pressure",          # 脚質/場ペースもベースに入れて物理に「超え」を要求
]
KELLER_FEATS = ["kl_reserve", "kl_burn", "kl_draft", "kl_lscap",
                "kl_pace_fit", "kl_surge", "kl_solo_front", "kl_pace_hardness"]
OU_FEATS = ["ou_net", "ou_s2n", "ou_informed", "ou_termvel", "ou_n"]


def _rid16(x):
    return re.sub(r"\D", "", str(x))[:16]


def load() -> pd.DataFrame:
    need = list(dict.fromkeys(
        [COL_RID, COL_BAN, "split", "fukusho_flag", "前走出走頭数",
         "前1角", "前2角", "前3角", "前4角"] + BASE_FEATS))
    need = [c for c in need if c not in ("style_rank", "pace_pressure")]
    df = pd.read_csv(MASTER, encoding="utf-8-sig",
                     usecols=lambda c: c in need, low_memory=False)
    df = add_style(df)                       # style_rank / pace_pressure 付加
    df["rid16"] = df[COL_RID].map(_rid16)
    df["ban"] = pd.to_numeric(df[COL_BAN], errors="coerce").astype("Int64")
    kl = pd.read_parquet(KELLER).drop(columns=["split"], errors="ignore")
    ou = pd.read_parquet(OU)
    df = df.merge(kl, on=["rid16", "ban"], how="left")
    df = df.merge(ou[["rid16", "ban"] + OU_FEATS], on=["rid16", "ban"], how="left")
    df["fukusho_flag"] = pd.to_numeric(df["fukusho_flag"], errors="coerce")
    df = df[df["fukusho_flag"].isin([0, 1])].copy()
    for c in BASE_FEATS + KELLER_FEATS + OU_FEATS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def fit_eval(df, feats, label):
    tr = df[df["split"] == "train"]; va = df[df["split"] == "valid"]; te = df[df["split"] == "test"]
    dtr = lgb.Dataset(tr[feats], tr["fukusho_flag"])
    dva = lgb.Dataset(va[feats], va["fukusho_flag"], reference=dtr)
    params = dict(objective="binary", metric="binary_logloss", learning_rate=0.03,
                  num_leaves=63, min_child_samples=200, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=42)
    m = lgb.train(params, dtr, num_boost_round=1200, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(60, verbose=False)])
    p = m.predict(te[feats], num_iteration=m.best_iteration)
    y = te["fukusho_flag"].values
    auc = roc_auc_score(y, p); ll = log_loss(y, p); ap = average_precision_score(y, p)
    print(f"  [{label:18s}] nfeat={len(feats):2d} best_iter={m.best_iteration:4d}  "
          f"AUC={auc:.5f}  logloss={ll:.5f}  AP={ap:.5f}")
    return dict(label=label, nfeat=len(feats), best_iter=int(m.best_iteration),
                auc=float(auc), logloss=float(ll), ap=float(ap)), p, m, te


def main():
    df = load()
    n = {s: int((df["split"] == s).sum()) for s in ["train", "valid", "test"]}
    print(f"[rows] train={n['train']:,} valid={n['valid']:,} test={n['test']:,}")
    for grp, fs in [("keller", KELLER_FEATS), ("ou", OU_FEATS)]:
        cov = df.loc[df["split"] == "test", fs].notna().mean().mean() * 100
        print(f"  {grp} test coverage(平均非欠損)={cov:.1f}%")

    print("\n=== モデル比較（test=2024+, fukusho_flag）===")
    res = {}
    res["base"], p_base, m_base, te = fit_eval(df, BASE_FEATS, "baseline")
    res["base+keller"], _, _, _ = fit_eval(df, BASE_FEATS + KELLER_FEATS, "base+keller")
    res["base+ou"], _, _, _ = fit_eval(df, BASE_FEATS + OU_FEATS, "base+ou")
    res["base+both"], p_both, m_both, _ = fit_eval(df, BASE_FEATS + KELLER_FEATS + OU_FEATS, "base+keller+ou")

    d_k = res["base+keller"]["auc"] - res["base"]["auc"]
    d_o = res["base+ou"]["auc"] - res["base"]["auc"]
    d_b = res["base+both"]["auc"] - res["base"]["auc"]
    print(f"\n  ΔAUC  +keller={d_k:+.5f}   +ou={d_o:+.5f}   +both={d_b:+.5f}")
    print(f"  Δlogloss +keller={res['base+keller']['logloss']-res['base']['logloss']:+.6f}"
          f"  +ou={res['base+ou']['logloss']-res['base']['logloss']:+.6f}"
          f"  +both={res['base+both']['logloss']-res['base']['logloss']:+.6f}  (負=改善)")

    # --- 直交性: ベース予測統制下の部分相関 corr(feature, y - p_base) ---
    print("\n=== 直交性 part-corr(feature, y - p_base) on test  /  既存との素相関 ===")
    resid = te["fukusho_flag"].values - p_base
    ortho = {}
    for f in KELLER_FEATS + OU_FEATS:
        v = te[f].astype(float)
        mask = v.notna().values
        if mask.sum() < 1000:
            continue
        pc = np.corrcoef(v.values[mask], resid[mask])[0, 1]
        # 既存特徴との最大素相関（重複度）
        ov = {}
        for g in ["style_rank", "pace_pressure", "closing_power"]:
            gg = te[g].astype(float)
            mm = (v.notna() & gg.notna()).values
            if mm.sum() > 1000:
                ov[g] = float(np.corrcoef(v.values[mm], gg.values[mm])[0, 1])
        ortho[f] = dict(part_corr=float(pc), overlap=ov)
        omax = max(abs(x) for x in ov.values()) if ov else 0.0
        print(f"  {f:16s} part_corr={pc:+.4f}   max|overlap|={omax:.3f}  ({ov})")

    # --- リフト表: 物理特徴の分位 vs 実 top3 率 ---
    print("\n=== リフト表（test, 物理特徴の十分位 → 実 top3 率）===")
    lifts = {}
    base_rate = float(te["fukusho_flag"].mean())
    print(f"  test base top3 率 = {base_rate:.4f}")
    for f in ["kl_surge", "kl_pace_fit", "ou_informed"]:
        sub = te[[f, "fukusho_flag"]].dropna()
        if len(sub) < 5000:
            continue
        try:
            sub["q"] = pd.qcut(sub[f], 10, labels=False, duplicates="drop")
        except Exception:
            continue
        tab = sub.groupby("q")["fukusho_flag"].agg(["mean", "size"])
        lifts[f] = {int(k): [float(r["mean"]), int(r["size"])] for k, r in tab.iterrows()}
        lo = tab["mean"].iloc[0]; hi = tab["mean"].iloc[-1]
        print(f"  {f:14s} Q0={lo:.4f}  Q9={hi:.4f}  spread={hi-lo:+.4f}")

    # --- 物理特徴の重要度（gain, base+both）---
    imp = dict(zip(m_both.feature_name(), m_both.feature_importance(importance_type="gain")))
    phys_imp = {k: float(imp.get(k, 0)) for k in KELLER_FEATS + OU_FEATS}
    tot = sum(imp.values())
    phys_share = sum(phys_imp.values()) / tot * 100 if tot else 0
    print(f"\n=== 物理特徴の gain シェア（base+both 内）= {phys_share:.2f}% ===")
    for k, v in sorted(phys_imp.items(), key=lambda x: -x[1])[:6]:
        print(f"  {k:16s} gain={v:,.0f}  ({v/tot*100:.2f}%)")

    out = dict(rows=n, models=res,
               delta_auc=dict(keller=d_k, ou=d_o, both=d_b),
               orthogonality=ortho, lifts=lifts,
               phys_gain_share_pct=phys_share, phys_importance=phys_imp)
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
