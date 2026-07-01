# -*- coding: utf-8 -*-
"""
learn_deep_test.py — 深い履歴特徴が市場を超える増分を持つか (多年OOS, leak-safe)
学習=対象2018-2024 / 評価=2025。master(v6特徴+fukusho_flag+roi_target) に
deep_hist_feats(補正履歴等)を結合し、市場(起Ｒ単勝オッズ)baselineに対し
 ①市場のみ ②v6特徴+市場 ③v6特徴+市場+deep を OOS で比較。
実行: PYTHONUTF8=1 python learn_deep_test.py
"""
import sys
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import joblib, lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from audit_ev_bin_roi import MASTER_CSV, COL_RID, COL_JYUN

b = joblib.load("models/unified_rank_v6.pkl"); feats, encs = b["feature_cols"], b["encoders"]
RID18 = "レースID(新)"
deep = pd.read_parquet("data/_xai/deep_hist_feats.parquet")
deep["rid18"] = deep["rid18"].astype(str)
DEEPCOLS = [c for c in deep.columns if c not in ("rid18","t_fin","kol_cnt","work_cnt")]  # t_oddsは市場として別扱い
print("deep cols:", DEEPCOLS)

print("[load] master")
df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
df = df.dropna(subset=[COL_JYUN, COL_RID, RID18]).copy()
df["rid18"] = df[RID18].astype(str).str.replace(r"\D","",regex=True)
df["fukusho_flag"] = pd.to_numeric(df["fukusho_flag"], errors="coerce")
df["roi_target"] = pd.to_numeric(df["roi_target"], errors="coerce")
for c, le in encs.items():
    if c in df.columns:
        v = df[c].astype(str).fillna("__NaN__"); df[c] = le.transform(v.where(v.isin(set(le.classes_)), "__NaN__"))
df = df.merge(deep, on="rid18", how="inner")     # deepがある=過去走持つ馬のみ
df["yr"] = df["rid18"].str[:4]
df["q_raw"] = 1.0 / df["t_odds"].clip(lower=1.01)
df["q_mkt"] = df["q_raw"] / df.groupby(COL_RID)["q_raw"].transform("sum")
df = df.dropna(subset=["fukusho_flag","q_mkt"])
tr = df[df["yr"] <= "2024"].copy(); ev = df[df["yr"] == "2025"].copy()
print(f"merged={len(df):,}  train(2018-24)={len(tr):,}  eval2025={len(ev):,} races={ev[COL_RID].nunique():,}")

def lg(p,e=1e-6): p=np.clip(p,e,1-e); return np.log(p/(1-p))
base_ll = log_loss(ev.fukusho_flag, np.clip(ev.q_mkt,1e-6,1-1e-6)); base_auc = roc_auc_score(ev.fukusho_flag, ev.q_mkt)
print(f"\n市場(起Ｒ単勝)単独: logloss={base_ll:.4f} AUC={base_auc:.4f}")

res = {}
def prep(d, cols): return d[cols].apply(pd.to_numeric, errors="coerce").fillna(-9999)
def gbm(cols):
    g = lgb.train(dict(objective="binary",learning_rate=0.03,num_leaves=63,min_data_in_leaf=300,
        feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,verbose=-1,seed=42),
        lgb.Dataset(prep(tr, cols), tr.fukusho_flag), num_boost_round=500)
    return g, g.predict(prep(ev, cols))
gA,res["v6+市場"] = gbm(feats+["q_mkt"])
gB,res["v6+市場+deep"] = gbm(feats+["q_mkt"]+DEEPCOLS)

print(f"\n{'モデル':22s}{'logloss':>9s}{'vs市場':>9s}{'AUC':>8s}{'vs(v6+市場)':>12s}")
print(f"{'市場単独':22s}{base_ll:>9.4f}{'--':>9s}{base_auc:>8.4f}")
llA=log_loss(ev.fukusho_flag,res['v6+市場'])
for k,p in res.items():
    ll=log_loss(ev.fukusho_flag,p)
    print(f"{k:22s}{ll:>9.4f}{base_ll-ll:>+9.4f}{roc_auc_score(ev.fukusho_flag,p):>8.4f}{(llA-ll):>+12.4f}")
print("(vs市場>0=市場超え / vs(v6+市場)>0=deepが増分)")

# deepの寄与
imp = pd.Series(gB.feature_importance("gain"), index=feats+["q_mkt"]+DEEPCOLS).sort_values(ascending=False)
print("\ndeep特徴の寄与(gain上位):")
print(imp[[c for c in imp.index if c in DEEPCOLS]].head(8).round(0).to_string())

# 指数1位の複勝率・複勝回収
print("\n=== 指数1位 複勝率 / 複勝回収 (2025 OOS) ===")
for k,p in res.items():
    e = ev.assign(s=p); e["rk"]=e.groupby(COL_RID)["s"].rank(ascending=False,method="first")
    s1 = e[e["rk"]==1]
    print(f"  {k:18s}: 複勝率{s1.fukusho_flag.mean()*100:.1f}%  複勝回収{s1.roi_target.mean()*100:.0f}")
print("  (複勝回収>100=控除率超え)")
