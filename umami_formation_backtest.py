# -*- coding: utf-8 -*-
"""
umami_formation_backtest.py — UMAMIフォメ の OOS 検証 (±関係なく正直に)
=====================================================================
circular回避: xROIグレード表を 2024 で fit → 2025 で UMAMIフォメ を組み settle。
比較対象: prob-first 三連複 top4 (oreapro相当)。同じ2025レースで正面比較。
グレード/ゲート/フォメ/25点カスケードは umami.py / umami_formation.py と同一ルール。
"""
import sys, io, glob, re
from collections import defaultdict
from itertools import combinations, permutations, product
from pathlib import Path
import joblib, numpy as np, pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
BASE = Path(r"E:\PyCaLiAI"); sys.path.insert(0, str(BASE))
import pl_probs as PL
from backtest_pl_ev import (all_fukusho_vec_fast, all_sanrenpuku_tensor,
                            load_payouts, COL_RID, COL_BAN, COL_JYUN)

ODIR = BASE / "data/Time _series_odds"
EV_EDGES=[0.8,0.9,1.0,1.1,1.3,1.5,2.0]; EV_LAB=["<0.8","0.8-0.9","0.9-1.0","1.0-1.1","1.1-1.3","1.3-1.5","1.5-2.0","2.0+"]
FAV_EDGES=[3.0,7.0,15.0,50.0]; FAV_LAB=["<3","3-7","7-15","15-50","50+"]
GRADE_EDGES=[(0.85,"S"),(0.80,"A"),(0.72,"B")]
P_WIN_FLOOR,P_SHO_FLOOR,ODDS_CAP,MIN_CELL_N,CAP=0.04,0.12,50.0,200,25
def _lab(v,edges,labs):
    for i,e in enumerate(edges):
        if v<e: return labs[i]
    return labs[-1]
def _rid16(x): return re.sub(r"\D","",str(x))[:16]

# ---- 9時オッズ (tan9, fuku9 mid) ----
def load_tan9():
    f=sorted(glob.glob(str(ODIR/"TANPUK_*.csv")))[-1]
    tp=pd.read_csv(f,encoding="cp932",low_memory=False); cols=list(tp.columns)
    RID,KB,TM=cols[0],cols[1],cols[2]
    tan_c,flo_c,fhi_c={},{},{}
    for c in cols:
        cs=str(c)
        m=re.match(r"^\s*(\d+)\s*単\s*$",cs);  m and tan_c.__setitem__(int(m.group(1)),c)
        m=re.match(r"^\s*(\d+)\s*複\s*Lo\s*$",cs); m and flo_c.__setitem__(int(m.group(1)),c)
        m=re.match(r"^\s*(\d+)\s*複\s*Hi\s*$",cs); m and fhi_c.__setitem__(int(m.group(1)),c)
    tp["rid16"]=tp[RID].map(_rid16)
    tp=tp[(tp["rid16"].str.len()==16)&(tp["rid16"].str[:4].astype(int).isin([2024,2025]))]
    out={}
    for rid,g in tp.groupby("rid16",sort=False):
        mmdd=int(rid[4:8]); kb=pd.to_numeric(g[KB],errors="coerce")
        g1=g[kb==1]
        if len(g1)==0: continue
        tm=pd.to_numeric(g1[TM],errors="coerce"); hh=tm%10000; dd=tm//10000
        same=g1[dd==mmdd]; base=same if len(same)>0 else g1
        row=base.loc[(hh.loc[base.index]-900).abs().idxmin()]
        tan9={}; fuku9={}
        for ban,c in tan_c.items():
            v=pd.to_numeric(pd.Series([row[c]]),errors="coerce").iloc[0]
            if pd.notna(v) and v>1.0: tan9[ban]=float(v)
        for ban,c in flo_c.items():
            lo=pd.to_numeric(pd.Series([row[c]]),errors="coerce").iloc[0]
            hi=pd.to_numeric(pd.Series([row[fhi_c[ban]]]),errors="coerce").iloc[0] if ban in fhi_c else np.nan
            if pd.notna(lo) and lo>1.0: fuku9[ban]=(float(lo),float(hi) if pd.notna(hi) else float(lo))
        out[rid]={"tan9":tan9,"fuku9":fuku9}
    return out

print("[load] model/cal/master/odds/payouts ...")
b=joblib.load(BASE/"models/unified_rank_v6.pkl"); model,feats,encs=b["model"],b["feature_cols"],b["encoders"]
cal=joblib.load(BASE/"models/pl_calibrators_v6.pkl"); cal=cal.get("calibrators",cal)
df=pd.read_csv(BASE/"data/master_v2_20130105-20251228.csv",encoding="utf-8-sig",low_memory=False)
df[COL_JYUN]=pd.to_numeric(df[COL_JYUN],errors="coerce")
df=df.dropna(subset=[COL_JYUN,COL_RID,"split"]); df=df[df["split"]=="test"].copy()
df["year"]=df[COL_RID].astype(str).str[:4].astype(int)
for c,le in encs.items():
    if c in df.columns:
        v=df[c].astype(str).fillna("__NaN__"); df[c]=le.transform(v.where(v.isin(set(le.classes_)),"__NaN__"))
X=df[feats].apply(pd.to_numeric,errors="coerce").fillna(-9999).values
df["_score"]=model.predict(X)
pays=load_payouts(); tan9map=load_tan9()
print(f"  races(test)={df[COL_RID].nunique():,}  9時odds={len(tan9map):,}")

# ---- per-race 記録 (両年) ----
races={2024:[],2025:[]}
for rid,g in df.groupby(COL_RID,sort=False):
    if len(g)<5: continue
    rid16=_rid16(rid); od=tan9map.get(rid16);
    if od is None: continue
    g=g.sort_values(COL_BAN).reset_index(drop=True)
    ban=g[COL_BAN].astype(int).values; w=PL.pl_weights(g["_score"].values); n=len(w)
    p_tan=np.clip(cal["tansho"].predict(PL.all_tansho(w)),0,1)
    p_fuku=np.clip(cal["fukusho"].predict(all_fukusho_vec_fast(w)),0,1)
    order=np.argsort(-g["_score"].values)
    rec={"rid":rid16,"ban":ban,"w":w,"n":n,"p_tan":p_tan,"p_fuku":p_fuku,
         "hon_idx":int(order[0]),"po":pays.get(rid16),
         "o9":[od["tan9"].get(int(ban[i])) for i in range(n)],
         "fo":[od["fuku9"].get(int(ban[i])) for i in range(n)]}
    races[int(g["year"].iloc[0])].append(rec)
print(f"  2024={len(races[2024])}R  2025={len(races[2025])}R")

# ---- 2024 で xROIセル fit ----
cells={"tansho":defaultdict(lambda:[0.,0.]),"fukusho":defaultdict(lambda:[0.,0.])}  # [stake,ret]
evc={"tansho":defaultdict(lambda:[0.,0.]),"fukusho":defaultdict(lambda:[0.,0.])}
for r in races[2024]:
    po=r["po"]
    for i in range(r["n"]):
        o9=r["o9"][i]
        if o9:
            ev=r["p_tan"][i]*o9; won=po and int(r["ban"][i])==po["win"]
            pay=float(po["tansho"]) if (won and po and po["tansho"]==po["tansho"]) else 0.
            k=(_lab(ev,EV_EDGES,EV_LAB),_lab(o9,FAV_EDGES,FAV_LAB))
            cells["tansho"][k][0]+=100; cells["tansho"][k][1]+=pay
            evc["tansho"][k[0]][0]+=100; evc["tansho"][k[0]][1]+=pay
        fo=r["fo"][i]
        if fo and o9:
            mid=(fo[0]+fo[1])/2; ev=r["p_fuku"][i]*mid
            top3={po["win"],po["plc"],po["sho"]} if po else set()
            won=po and int(r["ban"][i]) in top3
            fpay=0.
            if won and po:
                for bn,kk in [(po["win"],"fuku_win"),(po["plc"],"fuku_plc"),(po["sho"],"fuku_sho")]:
                    if int(bn)==int(r["ban"][i]) and po[kk]==po[kk]: fpay=float(po[kk])
            k=(_lab(ev,EV_EDGES,EV_LAB),_lab(o9,FAV_EDGES,FAV_LAB))
            cells["fukusho"][k][0]+=100; cells["fukusho"][k][1]+=fpay
            evc["fukusho"][k[0]][0]+=100; evc["fukusho"][k[0]][1]+=fpay
def roi_of(store,key,n_floor):
    s=store.get(key)
    return (s[1]/s[0]) if (s and s[0]>=n_floor*100) else None

def grade_horse(p_tan,p_fuku,o9,fo):
    """umami.py同等。2024表でgradeを返す。"""
    best=None  # (xroi, grade)
    for kind,p,odds in (("tansho",p_tan,o9),("fukusho",p_fuku,(fo[0]+fo[1])/2 if fo else None)):
        if not odds or not o9: continue
        floor=P_WIN_FLOOR if kind=="tansho" else P_SHO_FLOOR
        if p<floor or o9>ODDS_CAP: continue  # 罠ゲート
        ev=p*odds; k=(_lab(ev,EV_EDGES,EV_LAB),_lab(o9,FAV_EDGES,FAV_LAB))
        xr=roi_of(cells[kind],k,MIN_CELL_N)
        if xr is None: xr=roi_of(evc[kind],k[0],1)
        if xr is None: continue
        g="C"
        for edge,gg in GRADE_EDGES:
            if xr>=edge: g=gg; break
        if best is None or xr>best[0]: best=(xr,g)
    return best[1] if best else "罠"

# ---- フォメ構築 (umami_formation と同一ルール) ----
def build_tris(c1,c2,c3):
    t=set()
    for a in c1:
        for bb in c2:
            for c in c3:
                if len({a,bb,c})==3: t.add(frozenset((a,bb,c)))
    return t
def sbox(box,hon,xr,pr,cap):
    box=sorted(set(box)); tris={frozenset(c) for c in combinations(box,3)}
    weak=sorted([x for x in box if x!=hon],key=lambda b:(xr.get(b,-1),pr.get(b,0)))
    i=0
    while len(tris)>cap and i<len(weak) and len(box)>3:
        box.remove(weak[i]); i+=1; tris={frozenset(c) for c in combinations(box,3)}
    return tris
def formation(r,sbox_th=5,cap=25):
    grades={}; xr={}; pr={}
    for i in range(r["n"]):
        bn=int(r["ban"][i]); grades[bn]=grade_horse(r["p_tan"][i],r["p_fuku"][i],r["o9"][i],r["fo"][i])
        xr[bn]=-1.0; pr[bn]=float(r["p_fuku"][i])
    hon=int(r["ban"][r["hon_idx"]])
    S=[b for b,gd in grades.items() if gd=="S"]; A=[b for b,gd in grades.items() if gd=="A"]
    nontrap=[b for b,gd in grades.items() if gd!="罠"]
    if not S: return None  # 見送り
    if len(S)>=sbox_th: return list(sbox([hon]+S,hon,xr,pr,cap))
    c1=sorted(set([hon]+S)); c2=sorted(set(S+A)); c3=list(nontrap); core=set([hon]+S+A)
    tris=build_tris(c1,c2,c3)
    for b in sorted([x for x in c3 if x not in core],key=lambda b:(xr.get(b,-1),pr.get(b,0))):
        if len(tris)<=cap: break
        c3.remove(b); tris=build_tris(c1,c2,c3)
    for b in sorted([x for x in A if x!=hon],key=lambda b:(xr.get(b,-1),pr.get(b,0))):
        if len(tris)<=cap: break
        c2=[x for x in c2 if x!=b]; c3=[x for x in c3 if x!=b]; tris=build_tris(c1,c2,c3)
    if len(tris)>cap: tris=sbox([hon]+S,hon,xr,pr,cap)
    return list(tris)

def probfirst_tris(r,k=4):
    w=r["w"]; n=r["n"]; P3=all_sanrenpuku_tensor(w); tris=list(combinations(range(n),3))
    pk=np.zeros(len(tris))
    for a,bb,c in permutations(range(3)): pk+=P3[[t[a] for t in tris],[t[bb] for t in tris],[t[c] for t in tris]]
    idx=np.argsort(-pk)[:k]
    return [frozenset(int(r["ban"][t[j]]) for j in range(3)) for t in [tris[m] for m in idx]]

# ---- 2025 settle ----
def settle(make, gated):
    cost=ret=hit=nbet=0
    for r in races[2025]:
        po=r["po"]
        if po is None or not (po["sanrenpuku"]==po["sanrenpuku"]) or not po["sanrenpuku"]: continue
        tris=make(r)
        if tris is None or len(tris)==0:  # 見送り
            continue
        win=frozenset((po["win"],po["plc"],po["sho"]))
        c=len(tris)*100; h=win in set(tris)
        cost+=c; ret+=(float(po["sanrenpuku"]) if h else 0.); hit+=h; nbet+=1
    return dict(roi=round(ret/cost*100,1) if cost else 0, hit=round(hit/nbet*100,1) if nbet else 0,
                nbet=nbet, avg_pts=round(cost/100/nbet,1) if nbet else 0)

pf_all=settle(lambda r:probfirst_tris(r,4), False)
adopted={r["rid"] for r in races[2025] if formation(r,5,25)}  # 採用はcap非依存(S>=1)
pf_same=settle(lambda r:(probfirst_tris(r,4) if r["rid"] in adopted else None), True)

def settle_matched(cap):
    cost=ret=hit=nbet=0
    for r in races[2025]:
        po=r["po"]
        if po is None or not (po["sanrenpuku"]==po["sanrenpuku"]) or not po["sanrenpuku"]: continue
        ut=formation(r,5,cap)
        if not ut: continue
        pt=probfirst_tris(r,len(ut)); win=frozenset((po["win"],po["plc"],po["sho"]))
        cost+=len(pt)*100; ret+=float(po["sanrenpuku"]) if win in set(pt) else 0.; hit+=win in set(pt); nbet+=1
    return round(ret/cost*100,1) if cost else 0

print(f"\n{'='*72}\n点数キャップ・スイープ (2025 OOS 三連複, グレード表=2024fit)\n{'='*72}")
print(f"{'cap':>5}{'UMAMI_ROI':>11}{'的中':>7}{'平均点':>8}{'prob同点ROI':>13}{'選別差':>8}")
for cap in [25,20,15,12,8,6,4]:
    um=settle(lambda r,c=cap:formation(r,5,c), True)
    pm=settle_matched(cap)
    print(f"{cap:>5}{um['roi']:>10}%{um['hit']:>6}%{um['avg_pts']:>8}{pm:>12}%{um['roi']-pm:>+7.1f}")
print(f"\n基準: prob-first 4点 全レース={pf_all['roi']}% / UMAMI採用R限定={pf_same['roi']}% "
      f"(差={round(pf_all['roi']-pf_same['roi'],1)}pt=見送りゲートの害)")
print("[読み] UMAMI_ROIが上がるか / prob同点ROIとの差(選別優位)が絞っても残るか / 81%基準を超える点があるか")
