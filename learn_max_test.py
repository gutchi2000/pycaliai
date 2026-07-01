# -*- coding: utf-8 -*-
"""
learn_max_test.py — 全データ最大火力 + value pocket 狙い撃ち (多年OOS, leak-safe)
v6全特徴 + 市場(起Ｒ単勝) + 深い履歴(走間) + レース流れ(タイム分析) を束ねて
2018-24学習→2025 OOS。AUCの壁でなく「指数1位×非1番人気(市場と食い違う馬)」の
複勝回収が100%を超えるか=唯一エッジが見えたポケットを狙う。
"""
import sys
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import joblib, lightgbm as lgb
from sklearn.metrics import log_loss, roc_auc_score
from audit_ev_bin_roi import MASTER_CSV, COL_RID, COL_JYUN

b=joblib.load("models/unified_rank_v6.pkl"); feats,encs=b["feature_cols"],b["encoders"]
RID18="レースID(新)"
deep=pd.read_parquet("data/_xai/deep_hist_feats.parquet"); deep["rid18"]=deep["rid18"].astype(str)
tf=pd.read_parquet("data/_xai/timeflow_feats.parquet"); tf["rid18"]=tf["rid18"].astype(str)
DEEP=[c for c in deep.columns if c not in("rid18","t_fin","kol_cnt","work_cnt")]
TF=[c for c in tf.columns if c!="rid18"]
print("deep:",len(DEEP)," timeflow:",len(TF))

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
df["mkt_rank"]=df.groupby(COL_RID)["t_odds"].rank(method="first")   # 1=1番人気
tr=df[df["yr"]<="2024"].copy(); ev=df[df["yr"]=="2025"].copy()
print(f"train={len(tr):,} eval2025={len(ev):,} races={ev[COL_RID].nunique():,}")

def prep(d,cols): return d[cols].apply(pd.to_numeric,errors="coerce").fillna(-9999)
def gbm(cols):
    g=lgb.train(dict(objective="binary",learning_rate=0.03,num_leaves=63,min_data_in_leaf=300,
        feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,verbose=-1,seed=42),
        lgb.Dataset(prep(tr,cols),tr.fukusho_flag),num_boost_round=600)
    return g.predict(prep(ev,cols))
pBase=gbm(feats+["q_mkt"])
pMax =gbm(feats+["q_mkt"]+DEEP+TF)
base_auc=roc_auc_score(ev.fukusho_flag,ev.q_mkt)
print(f"\n市場AUC={base_auc:.4f}  v6+市場AUC={roc_auc_score(ev.fukusho_flag,pBase):.4f}  最大AUC={roc_auc_score(ev.fukusho_flag,pMax):.4f}")

def pocket(p,name):
    e=ev.assign(s=p); e["rk"]=e.groupby(COL_RID)["s"].rank(ascending=False,method="first")
    t1=e[e["rk"]==1]
    nonfav=t1[t1["mkt_rank"]>1]; fav=t1[t1["mkt_rank"]==1]
    print(f"\n=== {name} ===")
    print(f"  指数1位 全体     : n={len(t1)} 複勝率{t1.fukusho_flag.mean()*100:.1f}% 複勝回収{t1.roi_target.mean()*100:.0f}")
    print(f"  └ 市場1番人気と一致: n={len(fav)} 複勝率{fav.fukusho_flag.mean()*100:.1f}% 複勝回収{fav.roi_target.mean()*100:.0f}")
    print(f"  └ ★非1番人気(妙味) : n={len(nonfav)} 複勝率{nonfav.fukusho_flag.mean()*100:.1f}% 複勝回収{nonfav.roi_target.mean()*100:.0f}")
    # 非1番人気をオッズ帯で
    for lo,hi in [(2,3),(3,5),(5,99)]:
        s=nonfav[(nonfav.mkt_rank>=lo)&(nonfav.mkt_rank<hi)]
        if len(s): print(f"      市場{lo}-{hi-1}番人気: n={len(s)} 複勝回収{s.roi_target.mean()*100:.0f}")
pocket(pBase,"v6+市場")
pocket(pMax ,"最大(v6+市場+deep+timeflow)")
print("\n(複勝回収>100=控除率超え=本物のエッジ)")
