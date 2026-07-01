# -*- coding: utf-8 -*-
"""
learn_max_2yr.py — 学習2018-2023 / 検証 2024・2025 の2年OOS (leak-safe)
最大火力(v6全特徴+市場+深い履歴+タイム流れ)を 2018-23 で学習し、2024と2025を
別々に検証。妙味ポケット(指数1位×非1番人気の複勝回収)が両年一貫して
100%を超えるか=本物のエッジか偶然かを判定。
"""
import sys
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import joblib, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from audit_ev_bin_roi import MASTER_CSV, COL_RID, COL_JYUN

b=joblib.load("models/unified_rank_v6.pkl"); feats,encs=b["feature_cols"],b["encoders"]
RID18="レースID(新)"
deep=pd.read_parquet("data/_xai/deep_hist_feats.parquet"); deep["rid18"]=deep["rid18"].astype(str)
tf=pd.read_parquet("data/_xai/timeflow_feats.parquet"); tf["rid18"]=tf["rid18"].astype(str)
DEEP=[c for c in deep.columns if c not in("rid18","t_fin","kol_cnt","work_cnt")]
TF=[c for c in tf.columns if c!="rid18"]

df=pd.read_csv(MASTER_CSV,encoding="utf-8-sig",low_memory=False)
df[COL_JYUN]=pd.to_numeric(df[COL_JYUN],errors="coerce")
df=df.dropna(subset=[COL_JYUN,COL_RID,RID18]).copy()
df["rid18"]=df[RID18].astype(str).str.replace(r"\D","",regex=True)
df["fukusho_flag"]=pd.to_numeric(df["fukusho_flag"],errors="coerce")
df["roi_target"]=pd.to_numeric(df["roi_target"],errors="coerce")
for c,le in encs.items():
    if c in df.columns:
        v=df[c].astype(str).fillna("__NaN__"); df[c]=le.transform(v.where(v.isin(set(le.classes_)),"__NaN__"))
df=df.merge(deep,on="rid18",how="inner").merge(tf,on="rid18",how="left")
df["yr"]=df["rid18"].str[:4]
df["q_raw"]=1.0/df["t_odds"].clip(lower=1.01); df["q_mkt"]=df["q_raw"]/df.groupby(COL_RID)["q_raw"].transform("sum")
df=df.dropna(subset=["fukusho_flag","q_mkt"])
df["mkt_rank"]=df.groupby(COL_RID)["t_odds"].rank(method="first")
tr=df[df["yr"]<="2023"].copy()
print(f"train(2018-2023)={len(tr):,}")

def prep(d,cols): return d[cols].apply(pd.to_numeric,errors="coerce").fillna(-9999)
def train(cols):
    return lgb.train(dict(objective="binary",learning_rate=0.03,num_leaves=63,min_data_in_leaf=300,
        feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,verbose=-1,seed=42),
        lgb.Dataset(prep(tr,cols),tr.fukusho_flag),num_boost_round=600)
gBase=train(feats+["q_mkt"]); gMax=train(feats+["q_mkt"]+DEEP+TF)

def report(years,label):
    e=df[df["yr"].isin(years)].copy()
    e["pB"]=gBase.predict(prep(e,feats+["q_mkt"]))
    e["pM"]=gMax.predict(prep(e,feats+["q_mkt"]+DEEP+TF))
    print(f"\n######## 検証 {label}  races={e[COL_RID].nunique():,} ########")
    print(f"  AUC: 市場{roc_auc_score(e.fukusho_flag,e.q_mkt):.4f} / v6+市場{roc_auc_score(e.fukusho_flag,e.pB):.4f} / 最大{roc_auc_score(e.fukusho_flag,e.pM):.4f}")
    for col,nm in [("pB","v6+市場"),("pM","最大+deep+tf")]:
        x=e.copy(); x["rk"]=x.groupby(COL_RID)[col].rank(ascending=False,method="first")
        t1=x[x["rk"]==1]; nf=t1[t1.mkt_rank>1]; fav=t1[t1.mkt_rank==1]; p2=nf[nf.mkt_rank==2]
        print(f"  [{nm}] 指数1位複勝回収 全体{t1.roi_target.mean()*100:.0f}(n{len(t1)}) "
              f"| 1人気一致{fav.roi_target.mean()*100:.0f}(n{len(fav)}) "
              f"| ★非1人気{nf.roi_target.mean()*100:.0f}(n{len(nf)}) "
              f"| ★×2人気{p2.roi_target.mean()*100:.0f}(n{len(p2)})")

report(["2024"],"2024")
report(["2025"],"2025")
report(["2024","2025"],"2024+2025 合算")
print("\n(複勝回収>100=控除率超え / 両年一貫>100なら本物)")
