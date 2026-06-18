# -*- coding: utf-8 -*-
"""穴×under(AIが市場より強気な高オッズ馬)の単勝ROIを2025 OOS大標本で検証。罠カットの頑健性確認。"""
import sys, io, glob, re
import joblib, numpy as np, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
sys.path.insert(0, "E:/PyCaLiAI")
import pl_probs as PL
from backtest_pl_ev import all_fukusho_vec_fast, load_payouts, COL_RID, COL_BAN, COL_JYUN

ODIR = "E:/PyCaLiAI/data/Time _series_odds"
def rid16(x): return re.sub(r"\D", "", str(x))[:16]

def load_tan9():
    f = sorted(glob.glob(f"{ODIR}/TANPUK_*.csv"))[-1]
    tp = pd.read_csv(f, encoding="cp932", low_memory=False); cols = list(tp.columns); RID, KB, TM = cols[0], cols[1], cols[2]
    tan_c = {}
    for c in cols:
        m = re.match(r"^\s*(\d+)\s*単\s*$", str(c))
        if m: tan_c[int(m.group(1))] = c
    tp["rid16"] = tp[RID].map(rid16)
    tp = tp[(tp["rid16"].str.len() == 16) & (tp["rid16"].str[:4].astype(int) == 2025)]
    out = {}
    for rid, g in tp.groupby("rid16", sort=False):
        mmdd = int(rid[4:8]); kb = pd.to_numeric(g[KB], errors="coerce"); g1 = g[kb == 1]
        if len(g1) == 0: continue
        tm = pd.to_numeric(g1[TM], errors="coerce"); hh = tm % 10000; dd = tm // 10000
        base = g1[dd == mmdd]; base = base if len(base) > 0 else g1
        row = base.loc[(hh.loc[base.index] - 900).abs().idxmin()]
        tan9 = {}
        for ban, c in tan_c.items():
            v = pd.to_numeric(pd.Series([row[c]]), errors="coerce").iloc[0]
            if pd.notna(v) and v > 1.0: tan9[ban] = float(v)
        out[rid] = tan9
    return out

print("[load] ...")
b = joblib.load("E:/PyCaLiAI/models/unified_rank_v6.pkl"); model, feats, encs = b["model"], b["feature_cols"], b["encoders"]
cal = joblib.load("E:/PyCaLiAI/models/pl_calibrators_v6.pkl"); cal = cal.get("calibrators", cal)
df = pd.read_csv("E:/PyCaLiAI/data/master_v2_20130105-20251228.csv", encoding="utf-8-sig", low_memory=False)
df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce"); df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]); df = df[df["split"] == "test"].copy()
df["year"] = df[COL_RID].astype(str).str[:4].astype(int); df = df[df["year"] == 2025]
for c, le in encs.items():
    if c in df.columns:
        v = df[c].astype(str).fillna("__NaN__"); df[c] = le.transform(v.where(v.isin(set(le.classes_)), "__NaN__"))
X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values; df["_score"] = model.predict(X)
pays = load_payouts(); tan9 = load_tan9()
rows = []
for rid, g in df.groupby(COL_RID, sort=False):
    if len(g) < 5: continue
    r = rid16(rid); od = tan9.get(r); po = pays.get(r)
    if od is None or po is None: continue
    g = g.sort_values(COL_BAN); ban = g[COL_BAN].astype(int).values; w = PL.pl_weights(g["_score"].values)
    p = np.clip(cal["tansho"].predict(PL.all_tansho(w)), 0, 1)
    for i, bn in enumerate(ban):
        o = od.get(int(bn))
        if not o: continue
        under = p[i] > 1.0 / o
        won = int(bn) == po["win"]; pay = float(po["tansho"]) if (won and po["tansho"] == po["tansho"]) else 0.0
        rows.append((o, bool(under), float(p[i]), won, pay))
R = pd.DataFrame(rows, columns=["odds", "under", "p", "won", "pay"])
def roi(d): return f"n{len(d):>6} 単勝ROI{d['pay'].sum()/(len(d)*100)*100:6.1f}% 的中{d['won'].mean()*100:4.1f}%"
print(f"\n2025 OOS 単勝 全頭 {len(R)}")
print("全頭                :", roi(R))
print("odds>=15 全部        :", roi(R[R.odds >= 15]))
print("odds>=15 & under(罠) :", roi(R[(R.odds >= 15) & R.under]), " ←罠仮説")
print("odds>=15 & not under :", roi(R[(R.odds >= 15) & ~R.under]))
print("odds<15  & under     :", roi(R[(R.odds < 15) & R.under]))
print("under 全体           :", roi(R[R.under]))
print("fair/over (not under):", roi(R[~R.under]))
