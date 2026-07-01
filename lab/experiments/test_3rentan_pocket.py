# -*- coding: utf-8 -*-
"""
test_3rentan_pocket.py — 3連単の妙味ポケット実測 (2024-2025 OOS, leak-safe)
理論の核(過剰人気馬を切る=パターンB)だけ抜き出して測る:
  PyCaLiAI◎が市場1番人気でない時(妙味/危険人気疑い)に、その人気馬を外した3連単が
  控除率27.5%を超えて+EV(回収>100)になるか。比較で「市場追従」「モデル追従」も。
モデルrank=v6 _score。市場=9時オッズ(load_tanpuk)。配当/着順=load_payouts(3連単)。
"""
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

def axis_nagashi(axis,partners): return set((axis,x,y) for x in partners for y in partners if x!=y)
def box(hs): return set(permutations(hs,3))

STR=["市場1番人気軸流し","モデル◎軸流し","★妙味:◎軸流し(人気馬除外)","★妙味:モデル上位5BOX(人気馬除外)"]
S={y:{s:{"stake":0,"ret":0,"hit":0,"n":0} for s in STR} for y in ("2024","2025")}
for rid,g in df.groupby(COL_RID,sort=False):
    rid16=_rid16(rid); yr=rid16[:4]
    po=payouts.get(rid16); sp=tanpuk.get(rid16)
    if not po or not sp or not sp.get("tan9"): continue
    s3=po.get("sanrentan"); s3=float(s3) if (s3 and not pd.isna(s3)) else 0.0
    g=g.copy(); g["ban"]=g[COL_BAN].astype(int)
    g=g[g["ban"].map(lambda x: sp["tan9"].get(x) is not None)]
    if len(g)<7: continue
    g["odds"]=g["ban"].map(lambda x: sp["tan9"][x])
    model_order=list(g.sort_values("_score",ascending=False)["ban"])
    pop_order=list(g.sort_values("odds")["ban"])
    hon=model_order[0]; fav=pop_order[0]
    w,p,sh=po.get("win"),po.get("plc"),po.get("sho")
    if None in (w,p,sh): continue
    actual=(int(w),int(p),int(sh))
    def settle(name,combos):
        st=len(combos)*100; rt=s3 if actual in combos else 0.0
        d=S[yr][name]; d["stake"]+=st; d["ret"]+=rt; d["hit"]+= 1 if rt>0 else 0; d["n"]+=1
    settle("市場1番人気軸流し", axis_nagashi(fav, pop_order[1:7]))
    settle("モデル◎軸流し",     axis_nagashi(hon, model_order[1:7]))
    if hon!=fav:  # 妙味ポケット=モデルが市場1番人気を切ってる
        settle("★妙味:◎軸流し(人気馬除外)", axis_nagashi(hon,[x for x in model_order[1:8] if x!=fav][:6]))
        settle("★妙味:モデル上位5BOX(人気馬除外)", box([x for x in model_order if x!=fav][:5]))

for yr in ("2024","2025"):
    print(f"\n######## {yr} (3連単 控除率27.5%=回収72.5%が壁、100超で+EV) ########")
    print(f"  {'戦略':30s}{'R数':>6}{'点/R':>6}{'的中率':>8}{'回収率':>8}")
    for s in STR:
        d=S[yr][s]
        if d["n"]==0: continue
        roi=d["ret"]/d["stake"]*100 if d["stake"] else 0
        print(f"  {s:30s}{d['n']:>6}{d['stake']//d['n']//100:>6}{d['hit']/d['n']*100:>7.1f}%{roi:>7.0f}%")
