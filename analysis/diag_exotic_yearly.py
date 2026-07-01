# -*- coding: utf-8 -*-
"""
diag_exotic_yearly.py
=====================
baseline(全特徴) の exotic(馬単3/三連複4) 的中・ROI を年別に出す。
2023→2024→2025 の傾き=陳腐化(drift)の速度。2026への外挿で「再学習が要るか」を判定。
傾きが急なら陳腐化=要再学習、ほぼ平らなら ライブ9%は変動/壊れ期間が主因。
"""
import sys, io
from itertools import combinations, permutations
from pathlib import Path
import joblib, numpy as np, pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
BASE = Path(r"E:\PyCaLiAI"); sys.path.insert(0, str(BASE))
import pl_probs as PL
from backtest_pl_ev import (apply_encoders, load_payouts, all_umatan_mat,
                            all_sanrenpuku_tensor, COL_RID, COL_BAN)

MASTER = BASE / "data/master_v2_20130105-20251228.csv"
MODEL = BASE / "models/unified_rank_v6.pkl"
CONFIGS = [(3, 4)]

b = joblib.load(MODEL); model, feats, encs = b["model"], b["feature_cols"], b["encoders"]
df = pd.read_csv(MASTER, encoding="utf-8-sig", low_memory=False)
df = df.dropna(subset=[COL_RID, "split"]).copy()
df = df[df["split"].isin(["valid", "test"])].copy()
df["year"] = df[COL_RID].astype(str).str[:4].astype(int)
pays = load_payouts()
d = apply_encoders(df.copy(), encs)
X = d[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
df["_score"] = model.predict(X)
print(f"[scored] {len(df):,} rows / years {sorted(df.year.unique())}\n")

def exotic_hit(sub):
    agg = {c: [0, 0.0, 0, 0, 0] for c in CONFIGS}  # cost, ret, um_hit, sp_hit, nR
    for rid, g in sub.groupby(COL_RID, sort=False):
        pr = pays.get(str(rid)[:16])
        if pr is None: continue
        g = g.sort_values(COL_BAN)
        ban = pd.to_numeric(g[COL_BAN], errors="coerce").values
        s = g["_score"].values
        ok = ~np.isnan(ban)
        if ok.sum() < 3: continue
        ban = ban[ok].astype(int); s = s[ok]; n = len(s)
        b2i = {int(x): i for i, x in enumerate(ban)}
        try:
            lwin, lplc, lsho = b2i[int(pr["win"])], b2i[int(pr["plc"])], b2i[int(pr["sho"])]
        except (KeyError, ValueError, TypeError): continue
        if not (pr["umatan"] == pr["umatan"] and pr["sanrenpuku"] == pr["sanrenpuku"]): continue
        if not pr["umatan"] or not pr["sanrenpuku"]: continue
        w = PL.pl_weights(s)
        U = all_umatan_mat(w)
        um_rank = int((U[~np.eye(n, dtype=bool)] > U[lwin, lplc]).sum())
        P3 = all_sanrenpuku_tensor(w)
        tris = np.array(list(combinations(range(n), 3)))
        ptri = np.zeros(len(tris))
        for a, bb, c in permutations(range(3)):
            ptri += P3[tris[:, a], tris[:, bb], tris[:, c]]
        wt = tuple(sorted((lwin, lplc, lsho)))
        wm = (tris[:, 0] == wt[0]) & (tris[:, 1] == wt[1]) & (tris[:, 2] == wt[2])
        if not wm.any(): continue
        sp_rank = int((ptri > ptri[wm][0]).sum())
        for cfg in CONFIGS:
            ku, ks = cfg; a = agg[cfg]
            uh = um_rank < ku; sh = sp_rank < ks
            a[0] += (ku + ks) * 100
            a[1] += (pr["umatan"] if uh else 0) + (pr["sanrenpuku"] if sh else 0)
            a[2] += uh; a[3] += sh; a[4] += 1
    return agg

print(f"{'年':<8}{'R数':>7}{'ROI':>8}{'馬単的中':>9}{'三連複的中':>11}")
for yr in [2023, 2024, 2025]:
    sub = df[df.year == yr]
    if len(sub) == 0: continue
    agg = exotic_hit(sub)
    for cfg in CONFIGS:
        a = agg[cfg]
        if a[4] == 0: continue
        roi = a[1] / a[0] * 100
        print(f"{yr:<8}{a[4]:>7,}{roi:>7.1f}%{a[2]/a[4]*100:>8.1f}%{a[3]/a[4]*100:>10.1f}%")
print("\n[読み] 2023→2025の傾き。緩やか(~1pt/年)なら2026も大崩れせず=ライブ9%は壊れ期間+変動が主因。"
      " 急(数pt/年)なら陳腐化=要再学習。")
