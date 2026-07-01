# -*- coding: utf-8 -*-
"""
m2b — Dr.Z wedge selectivity. 各馬の楔 = fair_p(PL,単勝marginal) - q_place(市場複勝implied de-vig)。
楔の符号 x odds帯 でセルを切り、各セルの test ROI と bootstrap CI / Bonferroni を出す。
複勝implied = (1/mid) を正規化(de-vig)して P(top-k)スケール(Σ=k)に合わせる。
利益化セルがあるか。<=2023でセル選定 -> test評価。
"""
import sys; sys.path.insert(0, "E:/PyCaLiAI/analysis"); import joint_lib as J
import numpy as np, pandas as pd
RNG = np.random.default_rng(11)
H,_ = J.load()

rows = []
for rid, g in H.groupby("rid", sort=False):
    n = int(g.n.iloc[0]); year = int(g.year.iloc[0])
    pi = g.pi.values.astype(float)
    mid = ((g.fuku_lo9.values + g.fuku_hi9.values)/2.0).astype(float)
    fpay = g.fpay.values.astype(float)
    k = 3 if n>=8 else 2
    ok = np.isfinite(mid)&(mid>1.0)&(pi>0)
    if ok.sum() < k+1: continue
    fairp = J.pl_topk_probs(pi, k, K=3000, rng=RNG)
    inv = np.where(ok, 1.0/mid, 0.0)
    s = inv.sum()
    qplace = inv / s * k if s>0 else inv   # market place-implied, scaled to sum=k
    won = np.isfinite(fpay)
    ret = np.where(won, np.nan_to_num(fpay)/100.0, 0.0)
    for idx in np.where(ok)[0]:
        rows.append((year, mid[idx], fairp[idx], qplace[idx], fairp[idx]-qplace[idx],
                     fairp[idx]*mid[idx], int(won[idx]), ret[idx]))
D = pd.DataFrame(rows, columns=["year","mid","fairp","qplace","wedge","ev","won","ret"])
print("legs:", len(D), flush=True)

# odds bands
D["band"] = pd.cut(D.mid, [1,1.5,2,3,5,10,1e9], labels=["1.0-1.5","1.5-2","2-3","3-5","5-10","10+"])
# wedge sign / magnitude
D["wsign"] = np.where(D.wedge>0, "fair>mkt", "fair<mkt")

def roi_ci(sub, n=2000):
    if len(sub)==0: return (np.nan,np.nan,np.nan,0)
    r = sub.ret.values
    roi = r.sum()/len(r)
    rois=[]
    for _ in range(n):
        i=RNG.integers(0,len(r),len(r)); rois.append(r[i].mean())
    lo,hi=np.percentile(rois,[2.5,97.5])
    return (roi,lo,hi,len(r))

le=D[D.year<=2023]; te=D[D.year>=2024]
print("\n=== cells: band x wsign (LE23 select, TEST eval) ===", flush=True)
cells=[]
for band in D.band.cat.categories:
    for ws in ["fair>mkt","fair<mkt"]:
        lsub=le[(le.band==band)&(le.wsign==ws)]
        if len(lsub)<300: continue
        lroi=lsub.ret.mean()
        cells.append((band,ws,lroi,len(lsub)))
nc=len(cells)
print(f"candidate cells (le23 n>=300): {nc}, Bonferroni alpha=0.05/{nc}={0.05/max(nc,1):.4f}", flush=True)
res=[]
for (band,ws,lroi,ln) in cells:
    tsub=te[(te.band==band)&(te.wsign==ws)]
    roi,lo,hi,tn=roi_ci(tsub)
    res.append(dict(band=str(band),wsign=ws,le23_roi=round(lroi,3),
                    test_roi=round(roi,3),test_n=tn,ci_lo=round(lo,3),ci_hi=round(hi,3),
                    beats100=lo>1.0))
rd=pd.DataFrame(res).sort_values("test_roi",ascending=False)
print(rd.to_string(index=False), flush=True)

# strongest positive-wedge with high EV: graded EV within fair>mkt
print("\n=== fair>mkt x EV grade (TEST) ===", flush=True)
for evlo in [1.0,1.1,1.2,1.3]:
    tsub=te[(te.wsign=='fair>mkt')&(te.ev>=evlo)]
    roi,lo,hi,tn=roi_ci(tsub)
    print(f"ev>={evlo}: test ROI={roi:.3f} n={tn} CI[{lo:.3f},{hi:.3f}]", flush=True)
