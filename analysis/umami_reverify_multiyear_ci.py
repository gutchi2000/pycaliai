# -*- coding: utf-8 -*-
"""
umami_reverify_multiyear_ci.py — UMAMI(馬連振替/ワイド) の多年OOS再検証 + CI
==============================================================================
単年(2025)・CI無し・母数不一致 だった umami_*_backtest.py の弱点を全部潰す:
  ① cells(grade xROI表) fit = valid(2023) / eval = test(2024+2025)  ← 循環回避のまま多年化
  ② 同一母数baseline: UMAMIが参加したレースだけで 馬単/prob-first も決済
     (振替効果と参加選別効果を分離。元スクリプトは馬単3438R vs 馬連3137Rで混在)
  ③ race-level bootstrap CI (NBOOT=2000)
  ④ 年別ROI (2024 / 2025 方向一致確認)
  ⑤ paired-diff CI: (UMAMI - baseline) が 0 をまたぐか = 有意性検定
土台: umami_uma_backtest.py / umami_wide_backtest.py のロジックを流用。
実行: PYTHONUTF8=1 ./venv311/Scripts/python.exe analysis/umami_reverify_multiyear_ci.py
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
FIT_YEARS=[2023]; EVAL_YEARS=[2024,2025]; ALLY=FIT_YEARS+EVAL_YEARS; NBOOT=2000

def _lab(v,edges,labs):
    for i,e in enumerate(edges):
        if v<e: return labs[i]
    return labs[-1]
def _rid16(x): return re.sub(r"\D","",str(x))[:16]

def load_tan9(years):
    f=sorted(glob.glob(str(ODIR/"TANPUK_*.csv")))[-1]
    tp=pd.read_csv(f,encoding="cp932",low_memory=False); cols=list(tp.columns); RID,KB,TM=cols[0],cols[1],cols[2]
    tan_c,flo_c,fhi_c={},{},{}
    for c in cols:
        cs=str(c)
        m=re.match(r"^\s*(\d+)\s*単\s*$",cs); m and tan_c.__setitem__(int(m.group(1)),c)
        m=re.match(r"^\s*(\d+)\s*複\s*Lo\s*$",cs); m and flo_c.__setitem__(int(m.group(1)),c)
        m=re.match(r"^\s*(\d+)\s*複\s*Hi\s*$",cs); m and fhi_c.__setitem__(int(m.group(1)),c)
    tp["rid16"]=tp[RID].map(_rid16)
    tp=tp[(tp["rid16"].str.len()==16)&(tp["rid16"].str[:4].astype(int).isin(years))]
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

print("[load] model/cal/master(valid+test)/odds/payouts ...")
b=joblib.load(BASE/"models/unified_rank_v6.pkl"); model,feats,encs=b["model"],b["feature_cols"],b["encoders"]
cal=joblib.load(BASE/"models/pl_calibrators_v6.pkl"); cal=cal.get("calibrators",cal)
df=pd.read_csv(BASE/"data/master_v2_20130105-20251228.csv",encoding="utf-8-sig",low_memory=False)
df[COL_JYUN]=pd.to_numeric(df[COL_JYUN],errors="coerce"); df=df.dropna(subset=[COL_JYUN,COL_RID,"split"])
df=df[df["split"].isin(["valid","test"])].copy()
df["year"]=df[COL_RID].astype(str).str[:4].astype(int)
df=df[df["year"].isin(ALLY)].copy()
for c,le in encs.items():
    if c in df.columns:
        v=df[c].astype(str).fillna("__NaN__"); df[c]=le.transform(v.where(v.isin(set(le.classes_)),"__NaN__"))
X=df[feats].apply(pd.to_numeric,errors="coerce").fillna(-9999).values; df["_score"]=model.predict(X)
pays=load_payouts(); tan9map=load_tan9(ALLY); wides=load_wide_payouts()
races=defaultdict(list); RIDYEAR={}
for rid,g in df.groupby(COL_RID,sort=False):
    if len(g)<5: continue
    rid16=_rid16(rid); od=tan9map.get(rid16)
    if od is None: continue
    g=g.sort_values(COL_BAN).reset_index(drop=True); ban=g[COL_BAN].astype(int).values; w=PL.pl_weights(g["_score"].values); n=len(w)
    order=np.argsort(-g["_score"].values); y=int(g["year"].iloc[0])
    rec={"rid":rid16,"ban":ban,"w":w,"n":n,
        "p_tan":np.clip(cal["tansho"].predict(PL.all_tansho(w)),0,1),
        "p_fuku":np.clip(cal["fukusho"].predict(all_fukusho_vec_fast(w)),0,1),
        "hon_idx":int(order[0]),"rank_ban":[int(ban[order[i]]) for i in range(n)],"po":pays.get(rid16),
        "o9":[od["tan9"].get(int(ban[i])) for i in range(n)],
        "fo":[od["fuku9"].get(int(ban[i])) for i in range(n)]}
    races[y].append(rec); RIDYEAR[rid16]=y
print(f"  fit(2023)={len(races[2023])}R  eval 2024={len(races[2024])}R 2025={len(races[2025])}R")

# ---- cells fit on FIT_YEARS(2023) ----
cells={"tansho":defaultdict(lambda:[0.,0.]),"fukusho":defaultdict(lambda:[0.,0.])}
evc={"tansho":defaultdict(lambda:[0.,0.]),"fukusho":defaultdict(lambda:[0.,0.])}
for y in FIT_YEARS:
    for r in races[y]:
        po=r["po"]
        for i in range(r["n"]):
            o9=r["o9"][i]
            if o9:
                ev=r["p_tan"][i]*o9; won=po and int(r["ban"][i])==po["win"]; pay=float(po["tansho"]) if (won and po and po["tansho"]==po["tansho"]) else 0.
                k=(_lab(ev,EV_EDGES,EV_LAB),_lab(o9,FAV_EDGES,FAV_LAB)); cells["tansho"][k][0]+=100; cells["tansho"][k][1]+=pay; evc["tansho"][k[0]][0]+=100; evc["tansho"][k[0]][1]+=pay
            fo=r["fo"][i]
            if fo and o9:
                top3={po["win"],po["plc"],po["sho"]} if po else set(); won=po and int(r["ban"][i]) in top3; fpay=0.
                if won and po:
                    for bn,kk in [(po["win"],"fuku_win"),(po["plc"],"fuku_plc"),(po["sho"],"fuku_sho")]:
                        if int(bn)==int(r["ban"][i]) and po[kk]==po[kk]: fpay=float(po[kk])
                ev=r["p_fuku"][i]*((fo[0]+fo[1])/2); k=(_lab(ev,EV_EDGES,EV_LAB),_lab(o9,FAV_EDGES,FAV_LAB)); cells["fukusho"][k][0]+=100; cells["fukusho"][k][1]+=fpay; evc["fukusho"][k[0]][0]+=100; evc["fukusho"][k[0]][1]+=fpay

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

def umami_mark(r,cap=CAP):  # 馬連: 軸=◎+S / 相手=(◎〇▲)∪(S,A)
    grades,pr=_grades(r); hon=int(r["ban"][r["hon_idx"]])
    S=[b for b,gd in grades.items() if gd=="S"]; A=[b for b,gd in grades.items() if gd=="A"]
    if not S: return None
    marks=r["rank_ban"][:3]; axis=[hon]+[b for b in S if b!=hon]; part=sorted(set([hon]+S+A+marks))
    return _cap_pairs(axis,part,pr,cap)
def umami_wide_form(r,cap=CAP):  # ワイド: 軸=◎+S / 相手=◎+S+A
    grades,pr=_grades(r); hon=int(r["ban"][r["hon_idx"]])
    S=[b for b,gd in grades.items() if gd=="S"]; A=[b for b,gd in grades.items() if gd=="A"]
    if not S: return None
    axis=sorted(set([hon]+S)); part=sorted(set([hon]+S+A))
    return _cap_pairs(axis,part,pr,cap)
def umatan_form(r,cap=8):  # 現行T-10馬単形(順序付き)
    rk=r["rank_ban"]; firsts=rk[:2]; seconds=rk[:5]
    P=[(a,b) for a in firsts for b in seconds if a!=b]; return P[:cap]
def pf_pairs(r,k):  # prob-first 馬連/ワイド共通(無順序ペア)
    M=all_umaren_mat(r["w"]); iu,ju=np.triu_indices(r["n"],1); order=np.argsort(-M[iu,ju])
    return [frozenset((int(r["ban"][iu[o]]),int(r["ban"][ju[o]]))) for o in order[:k]]

def settle_recs(make,kind,years):
    recs={}
    for y in years:
        for r in races[y]:
            po=r["po"]
            if po is None: continue
            P=make(r)
            if not P: recs[r["rid"]]=None; continue
            if kind=="wide":
                wd={frozenset((int(bi),int(bj))):float(pay) for bi,bj,pay in wides.get(r["rid"],[])}
                rr=sum(wd.get(p,0.) for p in P); h=any(p in wd for p in P)
            elif kind=="umaren":
                wp=frozenset((po["win"],po["plc"])); pay=po["umaren"]; rr=float(pay) if (wp in P and pay==pay) else 0.; h=wp in P
            elif kind=="umatan":
                tgt=(po["win"],po["plc"]); pay=po["umatan"]; rr=float(pay) if (tgt in P and pay==pay) else 0.; h=tgt in P
            recs[r["rid"]]=(len(P)*100,rr,int(h))
    return recs

def roi_ci(pairs,seed=42):
    if not pairs: return (0.,0.,0.,0)
    c=np.array([p[0] for p in pairs],float); rt=np.array([p[1] for p in pairs],float)
    roi=rt.sum()/c.sum()*100 if c.sum() else 0.
    rng=np.random.default_rng(seed); m=len(pairs); o=np.empty(NBOOT)
    for bi in range(NBOOT):
        ix=rng.integers(0,m,m); s=c[ix].sum(); o[bi]=rt[ix].sum()/s*100 if s else 0.
    return roi,float(np.percentile(o,2.5)),float(np.percentile(o,97.5)),m
def paired_diff_ci(pa,pb,seed=42):
    ca=np.array([p[0] for p in pa],float); ra=np.array([p[1] for p in pa],float)
    cb=np.array([p[0] for p in pb],float); rb=np.array([p[1] for p in pb],float)
    diff=(ra.sum()/ca.sum()-rb.sum()/cb.sum())*100 if (ca.sum() and cb.sum()) else 0.
    rng=np.random.default_rng(seed); m=len(pa); o=np.empty(NBOOT)
    for bi in range(NBOOT):
        ix=rng.integers(0,m,m); sa=ca[ix].sum(); sb=cb[ix].sum()
        o[bi]=((ra[ix].sum()/sa if sa else 0)-(rb[ix].sum()/sb if sb else 0))*100
    return diff,float(np.percentile(o,2.5)),float(np.percentile(o,97.5))

def year_roi(recs,year):
    pr=[recs[rid][:2] for rid,v in recs.items() if v is not None and RIDYEAR.get(rid)==year]
    c=sum(p[0] for p in pr); rt=sum(p[1] for p in pr); return rt/c*100 if c else 0.

def block(title,head_name,head_make,head_kind,others):
    print(f"\n{'='*78}\n{title}\n{'='*78}")
    rec_h=settle_recs(head_make,head_kind,EVAL_YEARS)
    part=[rid for rid,v in rec_h.items() if v is not None]
    pairs_h=[rec_h[rid][:2] for rid in part]; hit_h=np.mean([rec_h[rid][2] for rid in part])*100
    roi,lo,hi,m=roi_ci(pairs_h)
    print(f"参加母数(headが参加したレース) n={m}  ※全baselineを同じ母数で決済")
    print(f"{'戦略':<26}{'ROI%[CI95]':<26}{'的中%':>7}{'2024':>8}{'2025':>8}")
    print("-"*78)
    print(f"{('▶ '+head_name):<26}{roi:6.1f} [{lo:5.1f},{hi:5.1f}]      {hit_h:6.1f}{year_roi(rec_h,2024):8.1f}{year_roi(rec_h,2025):8.1f}")
    for nm,mk,kd in others:
        rec=settle_recs(mk,kd,EVAL_YEARS)
        pr=[]; ph=[]
        for rid in part:
            v=rec.get(rid)
            if v is None: continue
            pr.append(v[:2]); ph.append(rec_h[rid][:2])
        if not pr: print(f"{nm:<26}(参加0)"); continue
        roi2,lo2,hi2,m2=roi_ci(pr); hit2=np.mean([rec[rid][2] for rid in part if rec.get(rid) is not None])*100
        d,dlo,dhi=paired_diff_ci(ph,pr)
        sig="有意" if (dlo>0 or dhi<0) else "n.s."
        print(f"{nm:<26}{roi2:6.1f} [{lo2:5.1f},{hi2:5.1f}]      {hit2:6.1f}{year_roi(rec,2024):8.1f}{year_roi(rec,2025):8.1f}")
        print(f"{'   └ head − '+nm.strip():<26}Δ{d:+5.1f}pt [{dlo:+5.1f},{dhi:+5.1f}]  {sig}")

print("\n##### UMAMI 多年OOS再検証 (cells fit=2023 / eval=2024+2025 / bootstrap CI n="+str(NBOOT)+") #####")
print("控除床: 馬連/ワイド~77.5%, 馬単~77.5%。break-even=100%。CI下限>100%なら黒字化、Δ CIが0を跨がねば有意。")

block("[馬連] 馬単→馬連UMAMI振替 (同一母数・head=UMAMI+印cap10)",
      "UMAMI+印 馬連cap10", lambda r:umami_mark(r,10), "umaren",
      [("  現行馬単形 8点", lambda r:umatan_form(r,8), "umatan"),
       ("  prob-first 馬連8点", lambda r:pf_pairs(r,8), "umaren")])

block("[ワイド] UMAMIワイドフォメ (同一母数・head=UMAMIワイドcap3)",
      "UMAMIワイド cap3", lambda r:umami_wide_form(r,3), "wide",
      [("  prob-first ワイド3点", lambda r:pf_pairs(r,3), "wide")])

print("\n[読み] ①CI下限が100%未満なら黒字化せず=負け縮小。②Δが0を跨ぐ(n.s.)なら baseline比の優位は統計的に無し。③2024/2025のROI符号が揃わねば単年ブレ。")
