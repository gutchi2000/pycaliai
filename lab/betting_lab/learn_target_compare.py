# -*- coding: utf-8 -*-
"""
learn_target_compare.py — 目的変数3本比較 (着順順位 / 複勝確率 / 回収) leak-safe
================================================================================
同じ特徴・同じ分割(train=split'train'≤2022 / eval=2025 OOS)で3モデルを学習し、
「指数1位(各モデルのトップ評価馬)の複勝率・複勝回収」を直接比較する。
  A 着順順位 : label=clip(6-着順,0,5), objective=lambdarank   (= 現v6方式)
  B 複勝確率 : label=fukusho_flag(0/1), objective=binary       (確率。×オッズ=EV)
  C 回収     : label=roi_target(複勝配当/100), objective=tweedie (E[複勝払戻])
評価: roi_target=実現複勝回収 を使うので外部オッズ不要・完全に同一土俵。
実行: PYTHONUTF8=1 python learn_target_compare.py
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import joblib, lightgbm as lgb
from audit_ev_bin_roi import MASTER_CSV, COL_RID, COL_JYUN

b = joblib.load("models/unified_rank_v6.pkl")
feats, encs = b["feature_cols"], b["encoders"]
print("[load] master")
df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]).copy()
df["year"] = df[COL_RID].astype(str).str[:4]
df["fukusho_flag"] = pd.to_numeric(df["fukusho_flag"], errors="coerce")
df["roi_target"] = pd.to_numeric(df["roi_target"], errors="coerce")
df["lab_rank"] = np.clip(6 - df[COL_JYUN].astype(int), 0, 5).astype(int)
for c, le in encs.items():
    if c in df.columns:
        v = df[c].astype(str).fillna("__NaN__")
        df[c] = le.transform(v.where(v.isin(set(le.classes_)), "__NaN__"))

tr = df[df["split"] == "train"].copy()                       # ≤2022
ev = df[(df["split"] == "test") & (df["year"] == "2025")].copy()
print(f"train(≤2022)={len(tr):,}  eval2025={len(ev):,}  races2025={ev[COL_RID].nunique():,}")
Xtr = tr[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999)
Xev = ev[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999)

# lambdarank group: race毎件数 (rid順に並べる)
tr_r = tr.sort_values(COL_RID); Xtr_r = tr_r[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999)
grp = tr_r.groupby(COL_RID, sort=False).size().values

common = dict(learning_rate=0.05, num_leaves=59, min_data_in_leaf=197,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=42)
print("[train] A 着順順位 (lambdarank)")
mA = lgb.train({**common, "objective": "lambdarank", "lambdarank_truncation_level": 5, "metric": "ndcg"},
               lgb.Dataset(Xtr_r, label=tr_r["lab_rank"].values, group=grp), num_boost_round=400)
print("[train] B 複勝確率 (binary)")
mB = lgb.train({**common, "objective": "binary", "metric": "auc"},
               lgb.Dataset(Xtr, label=tr["fukusho_flag"].values), num_boost_round=400)
print("[train] C 回収 (tweedie)")
mC = lgb.train({**common, "objective": "tweedie", "tweedie_variance_power": 1.5, "metric": "rmse"},
               lgb.Dataset(Xtr, label=tr["roi_target"].values), num_boost_round=400)

ev = ev.assign(sA=mA.predict(Xev), sB=mB.predict(Xev), sC=mC.predict(Xev))

def by_rank(scorecol, name):
    e = ev.copy()
    e["rk"] = e.groupby(COL_RID)[scorecol].rank(ascending=False, method="first")
    print(f"\n=== {name} : 指数順位別 (eval 2025 OOS, {e[COL_RID].nunique():,}R) ===")
    print(f"  {'順位':<5}{'n':>6}{'複勝率':>8}{'複勝回収':>9}")
    for r in [1, 2, 3, 4]:
        s = e[e["rk"] == r]
        print(f"  {int(r)}位 {len(s):>6}{s['fukusho_flag'].mean()*100:>7.1f}%{s['roi_target'].mean()*100:>8.0f}")
    print(f"  全体  {len(e):>6}{e['fukusho_flag'].mean()*100:>7.1f}%{e['roi_target'].mean()*100:>8.0f}")
    s1 = e[e["rk"] == 1]
    return s1["fukusho_flag"].mean()*100, s1["roi_target"].mean()*100

print("\n" + "="*60)
rA = by_rank("sA", "A 着順順位(現v6方式)")
rB = by_rank("sB", "B 複勝確率(binary)")
rC = by_rank("sC", "C 回収(tweedie)")
print("\n" + "="*60)
print("【1位馬の比較】複勝率 / 複勝回収率")
print(f"  A 着順順位 : {rA[0]:.1f}% / {rA[1]:.0f}%")
print(f"  B 複勝確率 : {rB[0]:.1f}% / {rB[1]:.0f}%")
print(f"  C 回収     : {rC[0]:.1f}% / {rC[1]:.0f}%")
print("  (複勝回収>100 = 控除率を超える=本物のエッジ)")
