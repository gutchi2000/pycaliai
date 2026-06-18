# -*- coding: utf-8 -*-
"""
build_wide_substrate.py — ワイド攻略 条件別検証の共通基盤を焼く（重い master ロードを1回に）
================================================================================
2024-2025(test) の全頭 × 全レースを per-horse parquet に。エージェントはこれを読んで
点数/馬場/段差オッズ/人気帯/軸選び 等を高速バックテストする(master再読込なし)。
出力:
  data/_wide/oos_horses.parquet  per-horse: rid,year,場所,馬場,n,ent,ban,mark_rank,
     p_win,p_sho,o9(9時単),fo_lo,fo_hi,grade(UMAMI),imp(市場含意),under,fin(確定着順),top3
  data/_wide/oos_wide_pay.json   per-race: {rid: {"i-j": payout/100}}
  grade は 2024fit→適用(circular回避), umami_wide_backtest と同一定義。
"""
import sys, io, glob, re, json, os
from collections import defaultdict
import joblib, numpy as np, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
sys.path.insert(0, "E:/PyCaLiAI")
import pl_probs as PL
from backtest_pl_ev import all_fukusho_vec_fast, load_payouts, COL_RID, COL_BAN, COL_JYUN
import generate_results as GR

ODIR = "E:/PyCaLiAI/data/Time _series_odds"
OUT = "E:/PyCaLiAI/data/_wide"; os.makedirs(OUT, exist_ok=True)
EV_EDGES=[0.8,0.9,1.0,1.1,1.3,1.5,2.0]; EV_LAB=["<0.8","0.8-0.9","0.9-1.0","1.0-1.1","1.1-1.3","1.3-1.5","1.5-2.0","2.0+"]
FAV_EDGES=[3.0,7.0,15.0,50.0]; FAV_LAB=["<3","3-7","7-15","15-50","50+"]
GRADE_EDGES=[(0.85,"S"),(0.80,"A"),(0.72,"B")]
P_WIN_FLOOR,P_SHO_FLOOR,ODDS_CAP,MIN_CELL_N=0.04,0.12,50.0,200
def _lab(v,e,l):
    for i,x in enumerate(e):
        if v<x: return l[i]
    return l[-1]
def rid16(x): return re.sub(r"\D","",str(x))[:16]

def load_tan9():
    f=sorted(glob.glob(f"{ODIR}/TANPUK_*.csv"))[-1]
    tp=pd.read_csv(f,encoding="cp932",low_memory=False); c=list(tp.columns); RID,KB,TM=c[0],c[1],c[2]
    tan,flo,fhi={},{},{}
    for x in c:
        m=re.match(r"^\s*(\d+)\s*単\s*$",str(x)); m and tan.__setitem__(int(m.group(1)),x)
        m=re.match(r"^\s*(\d+)\s*複\s*Lo\s*$",str(x)); m and flo.__setitem__(int(m.group(1)),x)
        m=re.match(r"^\s*(\d+)\s*複\s*Hi\s*$",str(x)); m and fhi.__setitem__(int(m.group(1)),x)
    tp["rid16"]=tp[RID].map(rid16); tp=tp[(tp["rid16"].str.len()==16)&(tp["rid16"].str[:4].astype(int).isin([2024,2025]))]
    out={}
    for rid,g in tp.groupby("rid16",sort=False):
        mmdd=int(rid[4:8]); kb=pd.to_numeric(g[KB],errors="coerce"); g1=g[kb==1]
        if len(g1)==0: continue
        tm=pd.to_numeric(g1[TM],errors="coerce"); hh=tm%10000; dd=tm//10000
        base=g1[dd==mmdd]; base=base if len(base)>0 else g1
        row=base.loc[(hh.loc[base.index]-900).abs().idxmin()]
        t9={}; f9={}
        for b,col in tan.items():
            v=pd.to_numeric(pd.Series([row[col]]),errors="coerce").iloc[0]
            if pd.notna(v) and v>1.0: t9[b]=float(v)
        for b,col in flo.items():
            lo=pd.to_numeric(pd.Series([row[col]]),errors="coerce").iloc[0]
            hi=pd.to_numeric(pd.Series([row[fhi[b]]]),errors="coerce").iloc[0] if b in fhi else np.nan
            if pd.notna(lo) and lo>1.0: f9[b]=(float(lo),float(hi) if pd.notna(hi) else float(lo))
        out[rid]={"t9":t9,"f9":f9}
    return out

print("[load] model/master/odds/payouts ...")
b=joblib.load("E:/PyCaLiAI/models/unified_rank_v6.pkl"); model,feats,encs=b["model"],b["feature_cols"],b["encoders"]
cal=joblib.load("E:/PyCaLiAI/models/pl_calibrators_v6.pkl"); cal=cal.get("calibrators",cal)
df=pd.read_csv("E:/PyCaLiAI/data/master_v2_20130105-20251228.csv",encoding="utf-8-sig",low_memory=False)
df[COL_JYUN]=pd.to_numeric(df[COL_JYUN],errors="coerce"); df=df.dropna(subset=[COL_JYUN,COL_RID,"split"]); df=df[df["split"]=="test"].copy()
df["year"]=df[COL_RID].astype(str).str[:4].astype(int); df=df[df["year"].isin([2024,2025])]
baba_col=next((c for c in df.columns if c=="馬場状態"),None)
place_col=next((c for c in df.columns if c=="場所"),None)
raw_baba=df[baba_col].astype(str).copy() if baba_col else None
raw_place=df[place_col].astype(str).copy() if place_col else None
for c,le in encs.items():
    if c in df.columns:
        v=df[c].astype(str).fillna("__NaN__"); df[c]=le.transform(v.where(v.isin(set(le.classes_)),"__NaN__"))
if raw_baba is not None: df["_baba"]=raw_baba
if raw_place is not None: df["_place"]=raw_place
X=df[feats].apply(pd.to_numeric,errors="coerce").fillna(-9999).values; df["_score"]=model.predict(X)
pays=load_payouts(); tan9=load_tan9(); wcache=GR.load_wide_payouts()

# ---- 2024 で xROIセル fit (grade用) ----
cells={"t":defaultdict(lambda:[0.,0.]),"f":defaultdict(lambda:[0.,0.])}; evc={"t":defaultdict(lambda:[0.,0.]),"f":defaultdict(lambda:[0.,0.])}
races=[]
for rid,g in df.groupby(COL_RID,sort=False):
    if len(g)<5: continue
    r=rid16(rid); od=tan9.get(r); po=pays.get(r)
    if od is None: continue
    g=g.sort_values(COL_BAN); ban=g[COL_BAN].astype(int).values; w=PL.pl_weights(g["_score"].values); n=len(w)
    pw=np.clip(cal["tansho"].predict(PL.all_tansho(w)),1e-9,1); pf=np.clip(cal["fukusho"].predict(all_fukusho_vec_fast(w)),1e-9,1)
    ent=float(-(pw*np.log(pw)).sum()/np.log(n))
    fin=pd.to_numeric(g[COL_JYUN],errors="coerce").values
    baba=g["_baba"].iloc[0] if "_baba" in g else ""; place=g["_place"].iloc[0] if "_place" in g else ""
    races.append({"rid":r,"year":int(g["year"].iloc[0]),"ban":ban,"pw":pw,"pf":pf,"n":n,"ent":ent,
                  "o9":[od["t9"].get(int(ban[i])) for i in range(n)],"fo":[od["f9"].get(int(ban[i])) for i in range(n)],
                  "fin":fin,"baba":baba,"place":place,"po":po})
    if int(g["year"].iloc[0])==2024 and po:
        for i in range(n):
            o9=od["t9"].get(int(ban[i]))
            if o9:
                ev=pw[i]*o9; won=int(ban[i])==po["win"]; pay=float(po["tansho"]) if (won and po["tansho"]==po["tansho"]) else 0.
                k=(_lab(ev,EV_EDGES,EV_LAB),_lab(o9,FAV_EDGES,FAV_LAB)); cells["t"][k][0]+=100; cells["t"][k][1]+=pay; evc["t"][k[0]][0]+=100; evc["t"][k[0]][1]+=pay
            fo=od["f9"].get(int(ban[i]))
            if fo and o9:
                ev=pf[i]*((fo[0]+fo[1])/2); top3={po["win"],po["plc"],po["sho"]}; won=int(ban[i]) in top3; fp=0.
                if won:
                    for bn,kk in [(po["win"],"fuku_win"),(po["plc"],"fuku_plc"),(po["sho"],"fuku_sho")]:
                        if int(bn)==int(ban[i]) and po[kk]==po[kk]: fp=float(po[kk])
                k=(_lab(ev,EV_EDGES,EV_LAB),_lab(o9,FAV_EDGES,FAV_LAB)); cells["f"][k][0]+=100; cells["f"][k][1]+=fp; evc["f"][k[0]][0]+=100; evc["f"][k[0]][1]+=fp
def roi_of(s,k,nf):
    v=s.get(k); return (v[1]/v[0]) if (v and v[0]>=nf*100) else None
def grade(pw,pf,o9,fo):
    best=None
    for kind,p,odds in (("t",pw,o9),("f",pf,(fo[0]+fo[1])/2 if fo else None)):
        if not odds or not o9: continue
        if p<(P_WIN_FLOOR if kind=="t" else P_SHO_FLOOR) or o9>ODDS_CAP: continue
        ev=p*odds; k=(_lab(ev,EV_EDGES,EV_LAB),_lab(o9,FAV_EDGES,FAV_LAB)); xr=roi_of(cells[kind],k,MIN_CELL_N) or roi_of(evc[kind],k[0],1)
        if xr is None: continue
        gg="C"
        for e,g2 in GRADE_EDGES:
            if xr>=e: gg=g2; break
        if best is None or xr>best[0]: best=(xr,gg)
    return best[1] if best else "罠"

rows=[]; widepay={}
for r in races:
    order=np.argsort(-r["pw"]); rank={int(r["ban"][order[i]]):i+1 for i in range(r["n"])}
    info=(r["rid"][:8], r["place"], None)  # raceno は wcache キーに要る→ban無しでは不明
    for i in range(r["n"]):
        bn=int(r["ban"][i]); o9=r["o9"][i]; fo=r["fo"][i]
        imp=(1.0/o9) if o9 else None
        rows.append(dict(rid=r["rid"],year=r["year"],place=r["place"],baba=r["baba"],n=r["n"],ent=round(r["ent"],4),
            ban=bn,mark_rank=rank[bn],p_win=round(float(r["pw"][i]),5),p_sho=round(float(r["pf"][i]),5),
            o9=o9,fo_lo=(fo[0] if fo else None),fo_hi=(fo[1] if fo else None),
            grade=grade(r["pw"][i],r["pf"][i],o9,fo),imp=(round(imp,5) if imp else None),
            under=(bool(r["pw"][i]>imp) if imp else None),
            fin=(int(r["fin"][i]) if r["fin"][i]==r["fin"][i] else None),
            top3=(1 if (r["fin"][i]==r["fin"][i] and r["fin"][i]<=3) else 0)))
    # wide payouts: rid→(date,place,raceno) を kekka依存せず wcache から place一致で引く必要。
    # wcache キー=(date8,場所名,raceno)。raceno は rid 末尾2桁(レース番号)から。
    raceno=str(int(r["rid"][-2:]))
    wp=wcache.get((r["rid"][:8], r["place"], raceno), {})
    if wp:
        widepay[r["rid"]]={f"{min(p)}-{max(p)}":float(v) for p,v in wp.items()}

H=pd.DataFrame(rows)
H.to_parquet(f"{OUT}/oos_horses.parquet")
json.dump(widepay, open(f"{OUT}/oos_wide_pay.json","w"), ensure_ascii=False)
print(f"[out] oos_horses.parquet rows={len(H)} races={H['rid'].nunique()} "
      f"(2024={ (H.year==2024).sum() }, 2025={ (H.year==2025).sum() })")
print(f"      wide払戻 races={len(widepay)} / {H['rid'].nunique()}")
print(f"      馬場分布={H.groupby('baba')['rid'].nunique().to_dict()}")
print(f"      grade分布={H['grade'].value_counts().to_dict()}")
print(f"      under率={H['under'].mean():.3f}  o9欠損={H['o9'].isna().mean():.3f}")
