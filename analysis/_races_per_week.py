# -*- coding: utf-8 -*-
"""週あたり何レース打つか: ◎複勝・信頼度選別の各カバレッジでの週次本数を実データで。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
BASE = Path(r"E:\PyCaLiAI"); sys.path.insert(0, str(BASE))
import pl_probs as PL
import backtest_pl_ev as BT
COL_RID = "レースID(新/馬番無)"; COL_BAN = "馬番"
sc = pd.read_parquet(BASE / "data/_ev_grid_scores.parquet")
pays = BT.load_payouts()

rows = []
for rid, g in sc.groupby(COL_RID, sort=False):
    if len(g) < 3:
        continue
    rid_s = str(int(rid)) if isinstance(rid, (int, np.integer)) else str(rid)
    if rid_s not in pays:
        continue
    digits = "".join(ch for ch in rid_s if ch.isdigit())
    date = digits[:8]
    w = PL.pl_weights(g["_score"].values)
    pw = np.asarray(PL.all_tansho(w), float); pw = pw / pw.sum()
    conf = float(pw.max())
    yr = int(g["year"].iloc[0]); spl = str(g["split"].iloc[0])
    period = "valid" if spl == "valid" else ("test" if yr in (2024, 2025) else "other")
    rows.append((period, date, conf))

df = pd.DataFrame(rows, columns=["period", "date", "conf"])
df["dt"] = pd.to_datetime(df["date"], format="%Y%m%d")
df["week"] = df["dt"].dt.strftime("%G-W%V")   # ISO週

for period in ["test"]:
    d = df[df.period == period]
    nweeks = d["week"].nunique(); ndays = d["date"].nunique()
    print(f"=== {period}: {len(d):,}R / {ndays}開催日 / {nweeks}週 ===")
    per_week_total = d.groupby("week").size()
    print(f"  全レース: 週平均 {per_week_total.mean():.1f}R  中央値 {per_week_total.median():.0f}R  "
          f"範囲 {per_week_total.min()}-{per_week_total.max()}R")
    for cov in [100, 50, 40, 30, 20, 10]:
        thr = d["conf"].quantile(1 - cov / 100)
        sel = d[d["conf"] >= thr]
        pw = sel.groupby("week").size()
        # 週0本の週も母数に入れる
        pw = pw.reindex(d["week"].unique(), fill_value=0)
        print(f"  cover{cov:>3}% (信頼度≥{thr:.3f}): 週平均 {pw.mean():4.1f}R  中央値 {pw.median():2.0f}R  "
              f"範囲 {pw.min()}-{pw.max()}R")
