# -*- coding: utf-8 -*-
"""
diag_pastform_dominance.py
==========================
仮説: v6は「過去走成績(着順/タイム)」特徴に支配され、実質その焼き直しでは?
測定:
  (1) v6 gain importance を特徴ファミリーに分類 → 過去走系が何%か
  (2) 過去走の単一特徴だけのランカー(素朴baseline)で ◎複勝率は何%出るか (vs v6 62%)
  (3) v6スコア と 過去走特徴 のレース内順位相関 / ◎一致率 (どれだけ collinear か)
入力: models/unified_rank_v6.pkl, master_v2 (score_test)
"""
import sys, io
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
BASE = Path(r"E:\PyCaLiAI"); sys.path.insert(0, str(BASE))
import backtest_pl_ev as be
be.MODEL_PKL = BASE / "models/unified_rank_v6.pkl"
be.CAL_PKL   = BASE / "models/pl_calibrators_v6.pkl"
be.CURVE_PKL = BASE / "data/pl_payout_curve_v6.pkl"
from backtest_pl_ev import COL_RID, COL_BAN

# ---------- (1) feature importance composition ----------
bundle = joblib.load(be.MODEL_PKL)
model, feats = bundle["model"], bundle["feature_cols"]
try:
    booster = model.booster_ if hasattr(model, "booster_") else model
    gain = np.array(booster.feature_importance(importance_type="gain"), dtype=float)
except Exception:
    gain = np.array(model.feature_importances_, dtype=float)
imp = pd.Series(gain, index=feats).sort_values(ascending=False)
imp_pct = imp / imp.sum() * 100

PASTFORM_KEYS = ["kako5", "prev_pos", "closing_power", "前1角","前2角","前3角","前4角",
                 "前走確定着順","前走着差","前走走破","前走上り","前PCI","前走PCI","前走RPCI",
                 "前走Ave","前走平均1F","hist_same","prev_hosei","prev_好走","前好走",
                 "course_win_rate","course_top3_rate","course_n_prev","horse_fuku"]
TRAIN_KEYS  = ["trnH","trnW"]
PED_KEYS    = ["種牡馬","父タイプ","母","母父","毛色"]
JT_KEYS     = ["騎手","調教師","jockey_","trainer_"]
def fam(c):
    if any(k in c for k in PASTFORM_KEYS): return "過去走(自馬成績)"
    if any(k in c for k in TRAIN_KEYS):    return "調教"
    if any(k in c for k in JT_KEYS):       return "騎手厩舎"
    if any(k in c for k in PED_KEYS):      return "血統"
    if "前走馬体重" in c:                  return "過去馬体重"
    return "当日条件/その他"
famser = pd.Series({c: fam(c) for c in feats})
share = imp.groupby(famser).sum().sort_values(ascending=False)
share_pct = (share / share.sum() * 100).round(1)
print("="*64); print("(1) v6 gain importance ファミリー別シェア"); print("="*64)
print(share_pct.to_string())
print("\n  top15 個別特徴 (gain%):")
for c, v in imp_pct.head(15).items():
    print(f"   {v:5.1f}%  [{fam(c)}]  {c}")

# ---------- score + baseline ----------
te = be.score_test(include_valid=True)
te = te[te["year"].isin([2024, 2025])].copy()
te[COL_BAN] = pd.to_numeric(te[COL_BAN], errors="coerce")
te["jyun"] = pd.to_numeric(te["着順"], errors="coerce")

# 過去走 候補特徴 + 自動方向 (着順と正相関なら lower=better → 符号反転)
cand = ["kako5_avg_pos","prev_hosei","prev_pos_rel","closing_power","kako5_best_pos","prev_hosei9"]
cand = [c for c in cand if c in te.columns]
for c in cand:
    te[c] = pd.to_numeric(te[c], errors="coerce")
direction = {}
for c in cand:
    sub = te[[c, "jyun"]].dropna()
    r = np.corrcoef(sub[c], sub["jyun"])[0, 1] if len(sub) > 100 else 0
    direction[c] = -1.0 if r > 0 else 1.0   # 着順と正相関(値大=着順悪)→反転
    print(f"   dir {c}: corr(着順)={r:+.3f} -> sign={direction[c]:+.0f}")

def spearman_within(a, b):
    if len(a) < 3: return np.nan
    ra = pd.Series(a).rank().values; rb = pd.Series(b).rank().values
    if ra.std() == 0 or rb.std() == 0: return np.nan
    return np.corrcoef(ra, rb)[0, 1]

# 過去走 z合成 (importance上位の過去走特徴を レース内z で合成)
zfeats = [c for c in ["kako5_avg_pos","prev_hosei","prev_pos_rel"] if c in te.columns]

rows = []
sp_v6_pos = []   # v6 vs -kako5_avg_pos
sp_v6_comp = []
for rid, g in te.groupby(COL_RID, sort=False):
    if len(g) < 5: continue
    g = g.reset_index(drop=True)
    jy = g["jyun"].values
    if np.isnan(jy).all(): continue
    s = g["_score"].values
    hon_v6 = int(np.argsort(-s)[0])
    rec = {"rid": rid, "v6_top3": int(jy[hon_v6] <= 3)}
    # 単一特徴 baseline ◎
    for c in cand:
        v = g[c].values * direction[c]
        v = np.where(np.isnan(v), -1e18, v)
        h = int(np.argsort(-v)[0])
        rec[f"bl_{c}_top3"] = int(jy[h] <= 3)
        rec[f"bl_{c}_agree"] = int(h == hon_v6)
    # z合成 baseline
    comp = np.zeros(len(g))
    for c in zfeats:
        v = pd.to_numeric(g[c], errors="coerce").values * direction[c]
        m, sd = np.nanmean(v), np.nanstd(v)
        z = (v - m) / sd if sd and not np.isnan(sd) else np.zeros(len(g))
        comp += np.nan_to_num(z, nan=0.0)
    hcomp = int(np.argsort(-comp)[0])
    rec["bl_comp_top3"] = int(jy[hcomp] <= 3)
    rec["bl_comp_agree"] = int(hcomp == hon_v6)
    rows.append(rec)
    # collinearity
    sp_v6_pos.append(spearman_within(s, g["kako5_avg_pos"].values * direction.get("kako5_avg_pos", -1)))
    sp_v6_comp.append(spearman_within(s, comp))

R = pd.DataFrame(rows)
print("\n"+"="*64); print(f"(2) 素朴な過去走ランカーの ◎複勝率 (OOS, n={len(R):,}レース)"); print("="*64)
print(f"  v6(120特徴) ◎複勝率            : {R.v6_top3.mean()*100:5.2f}%   <- 基準")
for c in cand:
    print(f"  単一[{c:14s}] ◎複勝率 : {R[f'bl_{c}_top3'].mean()*100:5.2f}%   (v6一致率 {R[f'bl_{c}_agree'].mean()*100:.1f}%)")
print(f"  過去走z合成(3特徴) ◎複勝率      : {R.bl_comp_top3.mean()*100:5.2f}%   (v6一致率 {R.bl_comp_agree.mean()*100:.1f}%)")
gap = R.v6_top3.mean()*100 - R.bl_comp_top3.mean()*100
print(f"\n  => v6 が過去走合成に上乗せしている分 = {gap:+.2f}pt"
      f"  (119特徴の追加価値)")

print("\n"+"="*64); print("(3) v6スコア と 過去走 のレース内順位相関 (collinearity)"); print("="*64)
print(f"  Spearman(v6, -kako5_avg_pos) 平均 : {np.nanmean(sp_v6_pos):.3f}")
print(f"  Spearman(v6, 過去走z合成)   平均 : {np.nanmean(sp_v6_comp):.3f}")
print(f"  ◎一致(v6◎==過去走合成◎)         : {R.bl_comp_agree.mean()*100:.1f}%")
