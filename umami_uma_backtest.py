# -*- coding: utf-8 -*-
"""
umami_uma_backtest.py — 馬単→馬連フォメ振替の OOS検証 (2025)
=============================================================
ユーザー提案: T-10の馬単フォメを「馬連フォメ・10点以内・UMAMI+印の組み合わせ」に。
同一の選別(UMAMI grade S/A + 印rank)で、馬連 / ワイド / 馬単 を正面比較する。
土台は umami_wide_backtest.py (v6→9時前オッズ→2024fit grade→2025評価)。
"""
import sys, io, glob, re
from collections import defaultdict
from pathlib import Path
import joblib, numpy as np, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
BASE = Path(r"E:\PyCaLiAI"); sys.path.insert(0, str(BASE))
import pl_probs as PL
from backtest_pl_ev import all_umaren_mat, all_fukusho_vec_fast, load_payouts, load_wide_payouts, COL_RID, COL_BAN, COL_JYUN

ODIR = BASE / "data/Time _series_odds"
EV_EDGES=[0.8,0.9,1.0,1.1,1.3,1.5,2.0]; EV_LAB=["<0.8","0.8-0.9","0.9-1.0","1.0-1.1","1.1-1.3","1.3-1.5","1.5-2.0","2.0+"]
FAV_EDGES=[3.0,7.0,15.0,50.0]; FAV_LAB=["<3","3-7","7-15","15-50","50+"]
GRADE_EDGES=[(0.85,"S"),(0.80,"A"),(0.72,"B")]
P_WIN_FLOOR,P_SHO_FLOOR,ODDS_CAP,MIN_CELL_N,CAP=0.04,0.12,50.0,200,10
def _lab(v,edges,labs):
    for i,e in enumerate(edges):
        if v<e: return labs[i]
    return labs[-1]
def _rid16(x): return re.sub(r"\D","",str(x))[:16]

def load_tan9():
    f=sorted(glob.glob(str(ODIR/"TANPUK_*.csv")))[-1]
    tp=pd.read_csv(f,encoding="cp932",low_memory=False); cols=list(tp.columns); RID,KB,TM=cols[0],cols[1],cols[2]
    tan_c,flo_c,fhi_c={},{},{}
    for c in cols:
        cs=str(c)
        m=re.match(r"^\s*(\d+)\s*単\s*$",cs); m and tan_c.__setitem__(int(m.group(1)),c)
        m=re.match(r"^\s*(\d+)\s*複\s*Lo\s*$",cs); m and flo_c.__setitem__(int(m.group(1)),c)
        m=re.match(r"^\s*(\d+)\s*複\s*Hi\s*$",cs); m and fhi_c.__setitem__(int(m.group(1)),c)
    tp["rid16"]=tp[RID].map(_rid16)
    tp=tp[(tp["rid16"].str.len()==16)&(tp["rid16"].str[:4].astype(int).isin([2024,2025]))]
    out={}
    for rid,g in tp.groupby("rid16",sort=False):
        mmdd=int(rid[4:8]); kb=pd.to_numeric(g[KB],errors="coerce"); g1=g[kb==1]
        if len(g1)==0: continue
        tm=pd.to_numeric(g1[TM],errors="coerce"); hh=tm%10000; dd=tm//10000
        base=g1[dd==mmdd]; base=base if len(base)>0 else g1
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
df[COL_JYUN]=pd.to_numeric(df[COL_JYUN],errors="coerce"); df=df.dropna(subset=[COL_JYUN,COL_RID,"split"]); df=df[df["split"]=="test"].copy()
df["year"]=df[COL_RID].astype(str).str[:4].astype(int)
for c,le in encs.items():
    if c in df.columns:
        v=df[c].astype(str).fillna("__NaN__"); df[c]=le.transform(v.where(v.isin(set(le.classes_)),"__NaN__"))
X=df[feats].apply(pd.to_numeric,errors="coerce").fillna(-9999).values; df["_score"]=model.predict(X)
pays=load_payouts(); tan9map=load_tan9(); wides=load_wide_payouts()
races={2024:[],2025:[]}
for rid,g in df.groupby(COL_RID,sort=False):
    if len(g)<5: continue
    rid16=_rid16(rid); od=tan9map.get(rid16)
    if od is None: continue
    g=g.sort_values(COL_BAN).reset_index(drop=True); ban=g[COL_BAN].astype(int).values; w=PL.pl_weights(g["_score"].values); n=len(w)
    order=np.argsort(-g["_score"].values)
    races[int(g["year"].iloc[0])].append({"rid":rid16,"ban":ban,"w":w,"n":n,
        "p_tan":np.clip(cal["tansho"].predict(PL.all_tansho(w)),0,1),
        "p_fuku":np.clip(cal["fukusho"].predict(all_fukusho_vec_fast(w)),0,1),
        "hon_idx":int(order[0]),"rank_ban":[int(ban[order[i]]) for i in range(n)],"po":pays.get(rid16),
        "o9":[od["tan9"].get(int(ban[i])) for i in range(n)],
        "fo":[od["fuku9"].get(int(ban[i])) for i in range(n)]})
print(f"  2024={len(races[2024])}R 2025={len(races[2025])}R")

# 2024 で xROIセル fit
cells={"tansho":defaultdict(lambda:[0.,0.]),"fukusho":defaultdict(lambda:[0.,0.])}
evc={"tansho":defaultdict(lambda:[0.,0.]),"fukusho":defaultdict(lambda:[0.,0.])}
for r in races[2024]:
    po=r["po"]
    for i in range(r["n"]):
        o9=r["o9"][i]
        if o9:
            ev=r["p_tan"][i]*o9; won=po and int(r["ban"][i])==po["win"]; pay=float(po["tansho"]) if (won and po and po["tansho"]==po["tansho"]) else 0.
            k=(_lab(ev,EV_EDGES,EV_LAB),_lab(o9,FAV_EDGES,FAV_LAB)); cells["tansho"][k][0]+=100; cells["tansho"][k][1]+=pay; evc["tansho"][k[0]][0]+=100; evc["tansho"][k[0]][1]+=pay
        fo=r["fo"][i]
        if fo and o9:
            ev=r["p_fuku"][i]*((fo[0]+fo[1])/2); top3={po["win"],po["plc"],po["sho"]} if po else set(); won=po and int(r["ban"][i]) in top3; fpay=0.
            if won and po:
                for bn,kk in [(po["win"],"fuku_win"),(po["plc"],"fuku_plc"),(po["sho"],"fuku_sho")]:
                    if int(bn)==int(r["ban"][i]) and po[kk]==po[kk]: fpay=float(po[kk])
            k=(_lab(ev,EV_EDGES,EV_LAB),_lab(o9,FAV_EDGES,FAV_LAB)); cells["fukusho"][k][0]+=100; cells["fukusho"][k][1]+=fpay; evc["fukusho"][k[0]][0]+=100; evc["fukusho"][k[0]][1]+=fpay
def roi_of(store,key,nf):
    s=store.get(key); return (s[1]/s[0]) if (s and s[0]>=nf*100) else None
def grade_horse(p_tan,p_fuku,o9,fo):
    best=None
    for kind,p,odds in (("tansho",p_tan,o9),("fukusho",p_fuku,(fo[0]+fo[1])/2 if fo else None)):
        if not odds or not o9: continue
        floor=P_WIN_FLOOR if kind=="tansho" else P_SHO_FLOOR
        if p<floor or o9>ODDS_CAP: continue
        ev=p*odds; k=(_lab(ev,EV_EDGES,EV_LAB),_lab(o9,FAV_EDGES,FAV_LAB)); xr=roi_of(cells[kind],k,MIN_CELL_N) or roi_of(evc[kind],k[0],1)
        if xr is None: continue
        g="C"
        for edge,gg in GRADE_EDGES:
            if xr>=edge: g=gg; break
        if best is None or xr>best[0]: best=(xr,g)
    return best[1] if best else "罠"

def _grades(r):
    grades={}; pr={}
    for i in range(r["n"]):
        bn=int(r["ban"][i]); grades[bn]=grade_horse(r["p_tan"][i],r["p_fuku"][i],r["o9"][i],r["fo"][i]); pr[bn]=float(r["p_fuku"][i])
    return grades,pr

def _cap_pairs(axis,part,pr,cap):
    def pairs(ax,pt): return {frozenset((a,b)) for a in ax for b in pt if a!=b}
    P=pairs(axis,part); axisset=set(axis)
    drop=sorted([b for b in part if b not in axisset],key=lambda b:pr.get(b,0)); i=0
    while len(P)>cap and i<len(drop):
        part=[b for b in part if b!=drop[i]]; i+=1; P=pairs(axis,part)
    sdrop=sorted([b for b in axis if b!=axis[0]],key=lambda b:pr.get(b,0)); j=0
    while len(P)>cap and j<len(sdrop) and len(axis)>1:
        d=sdrop[j]; j+=1; axis=[b for b in axis if b!=d]; part=[b for b in part if b!=d]; P=pairs(axis,part)
    return list(P)

def umami_form(r,cap=CAP):
    """軸=◎+S / 相手=◎+S+A （UMAMIフォメ。印=◎のみ）"""
    grades,pr=_grades(r); hon=int(r["ban"][r["hon_idx"]])
    S=[b for b,gd in grades.items() if gd=="S"]; A=[b for b,gd in grades.items() if gd=="A"]
    if not S: return None
    axis=[hon]+[b for b in S if b!=hon]; part=sorted(set([hon]+S+A))
    return _cap_pairs(axis,part,pr,cap)

def umami_mark(r,cap=CAP):
    """UMAMI+印: 軸=◎+S / 相手=(◎〇▲ 印)∪(S,A)  ←ユーザー提案"""
    grades,pr=_grades(r); hon=int(r["ban"][r["hon_idx"]])
    S=[b for b,gd in grades.items() if gd=="S"]; A=[b for b,gd in grades.items() if gd=="A"]
    if not S: return None
    marks=r["rank_ban"][:3]  # ◎〇▲
    axis=[hon]+[b for b in S if b!=hon]; part=sorted(set([hon]+S+A+marks))
    return _cap_pairs(axis,part,pr,cap)

def mark_box(r,topn,cap=CAP):
    """純・印ボックス: 上位topn頭(◎〇▲△△)総当たり"""
    bans=r["rank_ban"][:topn]
    P=[frozenset((a,b)) for idx,a in enumerate(bans) for b in bans[idx+1:]]
    return P[:cap]

def pf_pairs(r,k):
    """prob-first: umaren行列の上位k組"""
    M=all_umaren_mat(r["w"]); iu,ju=np.triu_indices(r["n"],1); order=np.argsort(-M[iu,ju])
    return [frozenset((int(r["ban"][iu[o]]),int(r["ban"][ju[o]]))) for o in order[:k]]

def umatan_form(r,cap=8):
    """現行T-10の馬単形: 1着=◎〇(top2) / 2着=◎〇▲△△(top5) の順序付き(参照用)"""
    rk=r["rank_ban"]; firsts=rk[:2]; seconds=rk[:5]
    P=[(a,b) for a in firsts for b in seconds if a!=b]
    return P[:cap]

def settle(make,kind):
    cost=ret=hit=nbet=spts=0
    for r in races[2025]:
        po=r["po"]
        if po is None: continue
        P=make(r)
        if not P: continue
        if kind=="wide":
            wd={frozenset((int(bi),int(bj))):float(pay) for bi,bj,pay in wides.get(r["rid"],[])}
            rr=sum(wd.get(p,0.) for p in P); h=any(p in wd for p in P)
        elif kind=="umaren":
            wp=frozenset((po["win"],po["plc"])); pay=po["umaren"]
            rr=float(pay) if (wp in P and pay==pay) else 0.; h=wp in P
        elif kind=="umatan":
            tgt=(po["win"],po["plc"]); pay=po["umatan"]
            rr=float(pay) if (tgt in P and pay==pay) else 0.; h=tgt in P
        cost+=len(P)*100; ret+=rr; hit+=int(h); nbet+=1; spts+=len(P)
    return dict(roi=round(ret/cost*100,1) if cost else 0,hit=round(hit/nbet*100,1) if nbet else 0,nbet=nbet,avg=round(spts/nbet,1) if nbet else 0)

def show(name,make,kinds=("umaren","wide")):
    for kind in kinds:
        s=settle(make,kind)
        star=" ◎ROI100+" if s['roi']>=100 else (" ★的中40+" if s['hit']>=40 else "")
        print(f"{name:<22}{kind:<8}{s['roi']:>7}%{s['hit']:>7}%{s['avg']:>7}{s['nbet']:>7}{star}")

print(f"\n{'='*70}\n馬単→馬連フォメ振替 (2025 OOS, 上限{CAP}点)\n{'='*70}")
print(f"{'戦略':<22}{'券種':<8}{'ROI':>8}{'的中':>8}{'平均点':>7}{'R数':>7}")
print("-"*70)
print("[参照] 現行 馬単フォメ (◎〇→◎〇▲△△, 8点):")
show("  現行馬単形",lambda r:umatan_form(r,8),kinds=("umatan",))
print("-"*70)
print("[本命] UMAMI+印 馬連フォメ (cap別):")
for cap in [4,6,8,10]:
    show(f"  UMAMI+印 cap{cap}",lambda r,c=cap:umami_mark(r,c),kinds=("umaren",))
print("[比較] UMAMIのみ / 印ボックス / prob-first (cap10 or k):")
show("  UMAMIフォメ",lambda r:umami_form(r,10),kinds=("umaren","wide"))
show("  UMAMI+印",lambda r:umami_mark(r,10),kinds=("umaren","wide"))
show("  印box ◎〇▲(3点)",lambda r:mark_box(r,3,10),kinds=("umaren","wide"))
show("  印box ◎〇▲△△(10)",lambda r:mark_box(r,5,10),kinds=("umaren","wide"))
for k in [4,6,8,10]:
    show(f"  prob-first {k}点",lambda r,kk=k:pf_pairs(r,kk),kinds=("umaren",))
print("\n[読み] 馬連 vs ワイド、同一選別での ROI/的中。控除床~77.5%。見送り(S無し)は除外集計。")
