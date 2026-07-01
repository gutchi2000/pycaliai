# -*- coding: utf-8 -*-
"""
diag_serve_full_impact.py
=========================
真の serve 影響: 実測した dead-26(全部) を master test=2024-25 でマスクし、
baseline / dead12(団が追ってた分) / dead26(実態) で ◎複勝率 + 馬単/三連複的中 を比較。
14個の未追跡死特徴が実際いくら削ってるかを定量。
"""
import sys, io, json
from itertools import combinations, permutations
from pathlib import Path
import joblib, numpy as np, pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
BASE = Path(r"E:\PyCaLiAI"); sys.path.insert(0, str(BASE))
import pl_probs as PL
from backtest_pl_ev import (apply_encoders, load_payouts, all_umatan_mat,
                            all_sanrenpuku_tensor, COL_RID, COL_BAN)

b = joblib.load(BASE/"models/unified_rank_v6.pkl")
model, feats, encs = b["model"], b["feature_cols"], b["encoders"]
catset = set(encs.keys())

bl = json.loads((BASE/"data/serve_feature_baseline.json").read_text(encoding="utf-8"))
DEAD26 = [c for c in bl["serve_dead"] if c in feats]
DEAD12 = ([c for c in feats if c.startswith("hist_same_")] +
          [c for c in ["course_n_prev","course_win_rate","course_top3_rate",
                       "jockey_n_prev","jockey_win_rate","jockey_top3_rate"] if c in feats] +
          [c for c in ["騎手コード","調教師コード"] if c in feats])
print(f"dead12={len(DEAD12)}  dead26={len(DEAD26)}")
print(f"未追跡14 = {sorted(set(DEAD26)-set(DEAD12))}")

df = pd.read_csv(BASE/"data/master_v2_20130105-20251228.csv", encoding="utf-8-sig", low_memory=False)
df = df.dropna(subset=[COL_RID,"split"]).copy()
df = df[df["split"]=="test"].copy()
df["着順"] = pd.to_numeric(df["着順"], errors="coerce")
pays = load_payouts()

def score(dead):
    d = df.copy()
    mc = [c for c in dead if c in catset]
    mn = [c for c in dead if c not in catset]
    for c in mc:
        if c in d.columns: d[c] = "__NaN__"
    d = apply_encoders(d, encs)
    for c in mn:
        if c in d.columns: d[c] = np.nan
    X = d[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    o = df[[COL_RID,COL_BAN,"着順"]].copy()
    o["_score"] = model.predict(X)
    return o

def metrics(sc):
    top3=0; nR=0; um=0; sp=0; nE=0; cost=0; ret=0.0
    for rid,g in sc.groupby(COL_RID, sort=False):
        if len(g)<5: continue
        pr = pays.get(str(rid)[:16])
        g = g.sort_values(COL_BAN)
        ban = pd.to_numeric(g[COL_BAN],errors="coerce").values
        jy = g["着順"].values; s = g["_score"].values
        ok = ~np.isnan(ban)
        if ok.sum()<5: continue
        ban=ban[ok].astype(int); jy=jy[ok]; s=s[ok]; n=len(s)
        order=np.argsort(-s); hon=int(order[0])
        if not np.isnan(jy[hon]):
            top3 += int(jy[hon]<=3); nR+=1
        if pr is None: continue
        b2i={int(x):i for i,x in enumerate(ban)}
        try: lw,lp,ls=b2i[int(pr["win"])],b2i[int(pr["plc"])],b2i[int(pr["sho"])]
        except: continue
        if not(pr["umatan"]==pr["umatan"] and pr["sanrenpuku"]==pr["sanrenpuku"]): continue
        if not pr["umatan"] or not pr["sanrenpuku"]: continue
        w=PL.pl_weights(s); U=all_umatan_mat(w)
        um_rank=int((U[~np.eye(n,dtype=bool)]>U[lw,lp]).sum())
        P3=all_sanrenpuku_tensor(w); tris=np.array(list(combinations(range(n),3)))
        pt=np.zeros(len(tris))
        for a,bb,c in permutations(range(3)): pt+=P3[tris[:,a],tris[:,bb],tris[:,c]]
        wt=tuple(sorted((lw,lp,ls))); wm=(tris[:,0]==wt[0])&(tris[:,1]==wt[1])&(tris[:,2]==wt[2])
        if not wm.any(): continue
        sp_rank=int((pt>pt[wm][0]).sum())
        uh=um_rank<3; sh=sp_rank<4
        cost += 7*100; ret += (pr["umatan"] if uh else 0)+(pr["sanrenpuku"] if sh else 0)
        um+=uh; sp+=sh; nE+=1
    return dict(top3=top3/nR*100, roi=ret/cost*100, um=um/nE*100, sp=sp/nE*100, nR=nR)

print(f"\n{'シナリオ':<16}{'◎複勝率':>9}{'馬単的中':>9}{'三連複的中':>11}{'exotic ROI':>11}")
for name, dead in [("baseline", []), ("dead12(追跡分)", DEAD12), ("dead26(実態)", DEAD26)]:
    m = metrics(score(dead))
    print(f"{name:<16}{m['top3']:>8.1f}%{m['um']:>8.1f}%{m['sp']:>10.1f}%{m['roi']:>10.1f}%")
print("\n[読み] dead12→dead26 の差 = 団が見落としてた14特徴の真のコスト。"
      " dead26 がライブ(馬単9%/三連複14%)に近ければ、ライブ劣化は serve skew で更に説明できる。")
