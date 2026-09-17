# -*- coding: utf-8 -*-
"""2026 as-served: ◎複勝 ROI を「モデル自信度」の分位で切る。
本番 topdown は p 比例で複勝を厚く張る。自信度が高いほど ROI が良いなら傾斜は正しく、
悪いなら傾斜そのものが漏れ。決着をつけるための独立母集団テスト（実績台帳ではなく bundle×kekka）。
実行: python -m analysis.place_sizing_2026
"""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

def ci(pay, n_boot=4000, seed=42):
    n = len(pay)
    if n == 0: return (0, None, None, None, None)
    rng = np.random.default_rng(seed)
    b = np.array([pay[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return (n, 100*pay.mean(), 100*(pay > 0).mean(),
            100*np.percentile(b, 2.5), 100*np.percentile(b, 97.5))

rows = []
for bp in sorted(glob.glob(str(BASE / "reports/cowork_input/*_bundle.json"))):
    date = Path(bp).name[:8]
    if not date.startswith("2026"): continue
    kp = BASE / "data" / "kekka" / f"{date}.csv"
    if not kp.exists(): continue
    k = pd.read_csv(kp, encoding="cp932", low_memory=False)
    k["rid16"] = k["レースID(新)"].astype(str).str[:16]
    k["ban"] = pd.to_numeric(k["馬番"], errors="coerce")
    k["fuku"] = pd.to_numeric(k["複勝配当"], errors="coerce")
    fuku = {(r.rid16, int(r.ban)): r.fuku/100.0 for r in k.itertuples()
            if pd.notna(r.fuku) and r.fuku > 0 and pd.notna(r.ban)}
    raced = {r.rid16 for r in k.itertuples()}
    b = json.loads(Path(bp).read_text(encoding="utf-8"))
    races = b["races"] if isinstance(b["races"], list) else list(b["races"].values())
    for r in races:
        rid = "".join(c for c in str(r.get("race_id", "")) if c.isdigit())[:16]
        if rid not in raced: continue
        hs = r.get("horses", [])
        hon = next((h for h in hs if h.get("mark") == "◎"), None)
        if hon is None: continue
        p = hon.get("p_win") or hon.get("win_prob") or hon.get("p")
        p3 = hon.get("p_top3") or hon.get("p_place") or hon.get("p_fukusho")
        if p is None: continue
        ban = int(hon.get("umaban"))
        rows.append({"date": date, "rid": rid, "p_win": float(p),
                     "p3": float(p3) if p3 is not None else np.nan,
                     "field": len(hs),
                     "pay": fuku.get((rid, ban), 0.0)})

df = pd.DataFrame(rows)
print(f"母集団: {len(df)} レース / {df.date.nunique()} 開催日 / {df.date.min()}-{df.date.max()}")
n, roi, hit, lo, hi = ci(df.pay.values)
print(f"◎複勝 べた買い(均等額): ROI={roi:.1f}%  的中={hit:.1f}%  CI95=[{lo:.1f},{hi:.1f}]\n")

for col, label in [("p_win", "p_win(◎の勝率)"), ("p3", "p_top3(◎の複勝確率)")]:
    d = df.dropna(subset=[col])
    if d.empty: continue
    print(f"--- {label} 五分位別 ◎複勝 ROI（本番はこの値が高いほど厚く張る） ---")
    d = d.assign(q=pd.qcut(d[col], 5, labels=["Q1(低)", "Q2", "Q3", "Q4", "Q5(高)"]))
    for q, g in d.groupby("q", observed=True):
        n, roi, hit, lo, hi = ci(g.pay.values)
        print(f"  {q:<7} n={n:>4}  {col}中央値={g[col].median():.3f}  ROI={roi:>5.1f}%  的中={hit:>5.1f}%  CI95=[{lo:>5.1f},{hi:>5.1f}]")
    # money-weighted (本番の p 比例配分を模擬) vs flat
    w = d[col].values
    mw = 100 * (w * d.pay.values).sum() / w.sum()
    print(f"  → {col} 比例配分ROI={mw:.1f}%   均等配分ROI={100*d.pay.mean():.1f}%   Δ(比例-均等)={mw-100*d.pay.mean():+.2f}pt\n")
