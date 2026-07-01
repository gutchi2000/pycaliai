# -*- coding: utf-8 -*-
import sys
from itertools import permutations
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import joblib
from audit_ev_bin_roi import MASTER_CSV, COL_RID, COL_BAN, COL_JYUN, _rid16, load_tanpuk
from backtest_pl_ev import load_payouts

b=joblib.load("models/unified_rank_v6.pkl"); model,feats,encs=b["model"],b["feature_cols"],b["encoders"]
print("[load] master")
df=pd.read_csv(MASTER_CSV,encoding="utf-8-sig",low_memory=False)
df[COL_JYUN]=pd.to_numeric(df[COL_JYUN],errors="coerce")
df=df.dropna(subset=[COL_JYUN,COL_RID]).copy()
df["yr"]=df[COL_RID].astype(str).str[:4]
df=df[df["yr"].isin(["2024","2025"])].copy()
for c,le in encs.items():
    if c in df.columns:
        v=df[c].astype(str).fillna("__NaN__"); df[c]=le.transform(v.where(v.isin(set(le.classes_)),"__NaN__"))
df["_score"]=model.predict(df[feats].apply(pd.to_numeric,errors="coerce").fillna(-9999).values)
tanpuk=load_tanpuk(); payouts=load_payouts()

def fuku_of(po,h):
    for k,bn in [("fuku_win","win"),("fuku_plc","plc"),("fuku_sho","sho")]:
        if po.get(bn)==h and po.get(k) and not pd.isna(po[k]): return float(po[k])
    return 0.0

for K in (4,6):
    acc={y:{} for y in ("2024","2025")}
    def add(y,k,st,rt,hit):
        d=acc[y].setdefault(k,{"st":0,"rt":0,"hit":0,"n":0}); d["st"]+=st; d["rt"]+=rt; d["hit"]+=hit; d["n"]+=1
    favt={"2024":[0,0],"2025":[0,0]}
    for rid,g in df.groupby(COL_RID,sort=False):
        rid16=_rid16(rid); y=rid16[:4]
        po=payouts.get(rid16); sp=tanpuk.get(rid16)
        if not po or not sp or not sp.get("tan9"): continue
        s3=po.get("sanrentan"); s3=float(s3) if (s3 and not pd.isna(s3)) else 0.0
        w,p,sh=po.get("win"),po.get("plc"),po.get("sho")
        if None in (w,p,sh): continue
        trip=(int(w),int(p),int(sh)); top3=set(trip)
        g=g.copy(); g["ban"]=g[COL_BAN].astype(int)
        g=g[g["ban"].map(lambda x: sp["tan9"].get(x) is not None)]
        if len(g)<7: continue
        g["odds"]=g["ban"].map(lambda x: sp["tan9"][x])
        order=list(g.sort_values("_score",ascending=False)["ban"])
        fav=list(g.sort_values("odds")["ban"])[0]
        fav_rank=order.index(fav)+1
        if fav_rank<K: continue
        favt[y][1]+=1; favt[y][0]+= 1 if fav in top3 else 0
        hon=order[0]
        add(y,"危険人気馬の複勝",100,fuku_of(po,fav),1 if fav in top3 else 0)
        add(y,"モデル◎の複勝",100,fuku_of(po,hon),1 if hon in top3 else 0)
        box=[x for x in order if x!=fav][:4]
        cb=set(permutations(box,3)); add(y,"3連単BOX(人気切4頭24点)",len(cb)*100,s3 if trip in cb else 0,1 if trip in cb else 0)
        rel=[x for x in order if x!=fav][1:6]
        cn=set((box[0],x,z) for x in rel for z in rel if x!=z) if box else set()
        add(y,"3連単◎軸流し(人気切)",len(cn)*100,s3 if trip in cn else 0,1 if trip in cn else 0)
    print(f"\n#### 危険人気馬=1番人気がモデル{K}位以下 ####")
    for y in ("2024","2025"):
        ht,n=favt[y]
        print(f"  [{y}] 該当{n}R / その1番人気の複勝率{ht/max(n,1)*100:.1f}%(全体約65%・落ちれば検出成功)")
        for k,d in acc[y].items():
            if d["n"]: print(f"      {k:24s} n{d['n']:>5} 的中{d['hit']/d['n']*100:>5.1f}% 回収{d['rt']/d['st']*100:>5.0f}%")
print("\n(回収>100で控除率超え=本物のエッジ)")
