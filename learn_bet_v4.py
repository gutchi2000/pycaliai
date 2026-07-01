# -*- coding: utf-8 -*-
"""
learn_bet_v4.py — 未使用特徴(Elo/Glicko/Keller/RaceLevel)を全投入し市場と対決 (leak-safe)
================================================================================
v2/v3で「v6の120特徴+NN+money-flow」は市場(9時価格)をOOSで超えなかった。
だが v6 が一切使ってない engineered 特徴が大量に在庫: Elo(16)/Glicko(3)/Keller(8)/
RaceLevel(5)。これらを全部足して、2025 OOS で市場価格を超える増分情報があるか検証。
「足りてない」が正しいなら、ここで vs市場>0 が出るはず。出なければ在庫特徴も冗長。

bet_dyn_rows.parquet (rid,ban,p_win,odds,won,tanpay + odds動き) に merge。master再読込不要。
実行: PYTHONUTF8=1 python learn_bet_v4.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
BASE = Path(__file__).parent

R = pd.read_parquet(BASE/"data/_xai/bet_dyn_rows.parquet")  # rid,ban,p_win,odds,won,tanpay,(+dyn)
print(f"base rows={len(R):,}")

NEWSETS = {
    "elo":   ("data/elo_feats.parquet", None),
    "glicko":("data/glicko_feats.parquet", None),
    "keller":("data/keller_pace_feats.parquet", ["split"]),
    "rlv1":  ("data/race_level_feats.parquet", None),
    "rlv2":  ("data/race_level_feats_v2.parquet", None),
}
newcols = []
for nm,(fp,drop) in NEWSETS.items():
    d = pd.read_parquet(BASE/fp)
    key = "rid16" if "rid16" in d.columns else "rid"
    d = d.rename(columns={key:"rid"})
    cols = [c for c in d.columns if c not in ("rid","ban") and (not drop or c not in drop)]
    d = d[["rid","ban"]+cols].drop_duplicates(["rid","ban"])
    R = R.merge(d, on=["rid","ban"], how="left")
    newcols += cols
    print(f"  +{nm}: {len(cols)}特徴 merged")
print(f"新規特徴 計 {len(newcols)}")

R["q_raw"]=1.0/R["odds"]; R["q_mkt"]=R["q_raw"]/R.groupby("rid")["q_raw"].transform("sum")
def lg(p,e=1e-6): p=np.clip(p,e,1-e); return np.log(p/(1-p))
tr=R[R.year==2024].copy(); ev=R[R.year==2025].copy()

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
import lightgbm as lgb
base_ll=log_loss(ev.won,np.clip(ev.q_mkt,1e-6,1-1e-6)); base_auc=roc_auc_score(ev.won,ev.q_mkt)
print(f"\n市場(9時)単独: logloss={base_ll:.4f} AUC={base_auc:.4f}")
# 新規特徴のカバレッジ
cov={c: R[c].notna().mean() for c in ["elo_T1M1_horse","g2_mu","kl_lscap","race_level_prev_W1"] if c in R}
print("カバレッジ:", {k:f"{v*100:.0f}%" for k,v in cov.items()})

res={}
lr=LogisticRegression(max_iter=2000).fit(np.column_stack([lg(tr.p_win),lg(tr.q_mkt)]),tr.won)
res["Benter(p_win+価格)"]=lr.predict_proba(np.column_stack([lg(ev.p_win),lg(ev.q_mkt)]))[:,1]

def gbm(cols,leaves=63,rounds=600):
    g=lgb.train(dict(objective="binary",learning_rate=0.03,num_leaves=leaves,min_data_in_leaf=300,
        feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,verbose=-1),
        lgb.Dataset(tr[cols].fillna(-9999),tr.won),num_boost_round=rounds)
    return g, g.predict(ev[cols].fillna(-9999))

g_b,res["GBM(p_win+価格)"]=gbm(["p_win","q_mkt"],31,400)
g_n,res["GBM(p_win+価格+新規)"]=gbm(["p_win","q_mkt"]+newcols)
g_o,res["GBM(価格+新規 ※p_win無)"]=gbm(["q_mkt"]+newcols)

print(f"\n{'モデル':28s}{'logloss':>9s}{'vs市場':>9s}{'AUC':>8s}")
print(f"{'市場(9時)単独':28s}{base_ll:>9.4f}{'--':>9s}{base_auc:>8.4f}")
for k,p in res.items():
    print(f"{k:28s}{log_loss(ev.won,p):>9.4f}{base_ll-log_loss(ev.won,p):>+9.4f}{roc_auc_score(ev.won,p):>8.4f}")
print("(vs市場>0 = 価格を超える増分情報あり = 在庫特徴にエッジ)")

imp=pd.Series(g_n.feature_importance("gain"),index=["p_win","q_mkt"]+newcols).sort_values(ascending=False)
print("\n寄与上位15(GBM gain):")
print(imp.head(15).round(0).to_string())

def roi(p):
    E=ev.copy(); E["_pb"]=p/ pd.Series(p,index=E.index).groupby(E["rid"]).transform("sum"); E["_ev"]=E["_pb"]*E["odds"]
    o=[]
    for t in [1.0,1.1,1.2]:
        s=E[E["_ev"]>=t]; n=len(s)
        if n==0: o.append(f"EV≥{t}:n0"); continue
        r=s.tanpay.values; v=r.mean()/100; se=r.std(ddof=1)/np.sqrt(n)/100 if n>1 else 0
        o.append(f"EV≥{t}:n{n} ROI{v*100:.0f}%[{(v-1.96*se)*100:.0f},{(v+1.96*se)*100:.0f}]")
    return " ".join(o)
print("\n単勝EV選抜ROI(2025 OOS):")
print("  GBM(p_win+価格+新規):",roi(res["GBM(p_win+価格+新規)"]))
