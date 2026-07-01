# -*- coding: utf-8 -*-
"""
wide_byrank.py — ◎-(相手N番人気) ワイド1点の ROI を人気順位ごとに出す
================================================================================
「何番人気まで絡める価値があるか」を1番人気刻みで決着。
各レース ◎(=モデル最上位) と、実オッズ人気 N位の馬の ワイド1点を N=2..15+ で別々に集計。
valid2023 / test2024-25, 堅い回(chaos≤0.79) と 全レース。
実行: PYTHONUTF8=1 ./venv311/Scripts/python.exe wide_byrank.py
"""
from __future__ import annotations
import glob, io, sys, time
from pathlib import Path
import joblib, numpy as np, pandas as pd
import backtest_pl_ev as BT
import pl_probs as PL

BASE = Path(__file__).parent
SCORE_CACHE = BASE / "data/_ev_grid_scores.parquet"
V6_CAL = BASE / "models/pl_calibrators_v6.pkl"
OUT_TXT = BASE / "reports/wide_byrank_summary.txt"
COL_RID = "レースID(新/馬番無)"; COL_BAN = "馬番"
BET = 100; CHAOS_MAX = 0.7943; NBOOT = 1200


def log(m): print(m, flush=True)


def load_pop():
    out = {}
    for f in sorted(glob.glob(str(BASE/"data/odds/odds_*.csv"))):
        df = pd.read_csv(f, encoding="cp932", low_memory=False)
        df["rid16"] = df["レースID(新)"].astype(str).str[:16]
        df["ban"] = pd.to_numeric(df["馬番"], errors="coerce")
        df["o"] = pd.to_numeric(df["単勝オッズ"], errors="coerce")
        for rid, g in df.groupby("rid16"):
            g = g.dropna(subset=["ban", "o"]).sort_values("o")
            out[rid] = {int(b): r for r, b in enumerate(g["ban"].values, start=1)}
    return out


def boot(pairs, seed=42):
    if not pairs: return (0., 0., 0.)
    st=np.array([p[0] for p in pairs],float); rt=np.array([p[1] for p in pairs],float)
    rng=np.random.default_rng(seed); m=len(pairs); o=np.empty(NBOOT)
    for b in range(NBOOT):
        i=rng.integers(0,m,m); s=st[i].sum(); o[b]=rt[i].sum()/s if s else 0.
    return (rt.sum()/st.sum() if st.sum() else 0., float(np.percentile(o,2.5)), float(np.percentile(o,97.5)))


def main():
    cal=joblib.load(V6_CAL)["calibrators"]; widepay=BT.load_wide_payouts(); pop=load_pop()
    log(f"[wide]{len(widepay):,} [pop]{len(pop):,}")
    sc=pd.read_parquet(SCORE_CACHE); sc[COL_RID]=sc[COL_RID].astype(str)
    sc=sc[sc["year"].isin([2023,2024,2025])]
    # rank buckets: 2,3,4,5,6,7,8,9,10-11,12-14,15+
    BUCKETS=[("2",lambda r:r==2),("3",lambda r:r==3),("4",lambda r:r==4),("5",lambda r:r==5),
             ("6",lambda r:r==6),("7",lambda r:r==7),("8",lambda r:r==8),("9",lambda r:r==9),
             ("10-11",lambda r:r in(10,11)),("12-14",lambda r:12<=r<=14),("15+",lambda r:r>=15)]
    # acc[(gate,bucket,period)] -> list[(stake,ret)]
    acc={}
    t0=time.time(); n=0
    for rid,g in sc.groupby(COL_RID,sort=False):
        rid_s=str(rid)
        if rid_s not in widepay or rid_s not in pop: continue
        g=g.sort_values(COL_BAN).reset_index(drop=True)
        ban=g[COL_BAN].astype(int).values; w=PL.pl_weights(g["_score"].values); N=len(w)
        if N<5: continue
        p=w/w.sum(); pc=np.clip(p,1e-9,1); ent=float(-(pc*np.log(pc)).sum()/np.log(N))
        hon=int(np.argmax(w)); hon_ban=int(ban[hon])
        Wm=cal["wide"].predict(BT.all_wide_mat(w).ravel()).reshape(N,N)
        winp={frozenset((int(a),int(b))):float(pay) for (a,b,pay) in widepay[rid_s]}
        pops=pop[rid_s]; period="valid" if g["split"].iloc[0]=="valid" else "test"
        gates=[("all",True)]+([("clean",True)] if ent<=CHAOS_MAX else [])
        for j in range(N):
            if j==hon: continue
            bj=int(ban[j]); pr=pops.get(bj)
            if pr is None: continue
            ret=winp.get(frozenset((hon_ban,bj)),0.0)
            for bname,bf in BUCKETS:
                if bf(pr):
                    for gname,_ in gates:
                        acc.setdefault((gname,bname,period),[]).append((BET,ret))
                    break
        n+=1
        if n%3000==0: log(f"  {n} ({time.time()-t0:.0f}s)")
    log(f"  {n} races ({time.time()-t0:.0f}s)")

    L=[]; P=lambda s:(L.append(s),log(s))
    P("="*86)
    P("◎-(相手N番人気) ワイド1点 ROI  [v6 / valid→test / 堅い回=chaos≤0.79]")
    P("◎=モデル最上位。相手の実オッズ人気順位ごと。何番人気まで価値があるか。")
    P("="*86)
    for gate,gl in (("clean","堅い回のみ"),("all","全レース")):
        P(f"\n■ {gl}")
        P(f"{'相手人気':>8}{'valid ROI':>11}{'test ROI':>10}{'  test CI95':>16}{'的中%':>8}{'test件数':>9}")
        for bname,_ in BUCKETS:
            v=acc.get((gate,bname,"valid"),[]); t=acc.get((gate,bname,"test"),[])
            if len(t)<150:
                P(f"{bname:>8}{'(少)':>11}"); continue
            vroi=sum(x[1] for x in v)/sum(x[0] for x in v) if v else 0
            troi,lo,hi=boot(t); hit=sum(1 for x in t if x[1]>0)/len(t)
            P(f"{bname:>8}{vroi*100:>10.1f}%{troi*100:>9.1f}% [{lo*100:5.1f},{hi*100:5.1f}]{hit*100:>7.1f}%{len(t):>9,}")
    P("="*86)
    P("※各行=◎とそのN番人気馬の2頭ワイドを1点だけ買い続けた時の回収率。100%超なら黒字。")
    OUT_TXT.write_text("\n".join(L),encoding="utf-8")
    log(f"[saved]{OUT_TXT}")


if __name__=="__main__":
    main()
