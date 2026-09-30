# -*- coding: utf-8 -*-
"""
top1_by_odds_band.py — 「上位1点」系 (◎単勝 / ◎複勝 / 馬連AI1-2 / 馬単AI1→2) を
◎(AI1位)の9時単勝オッズ帯で層別して比べる。2023-25 OOS 10,299R (bet_substrate)。
実行: python -m analysis.top1_by_odds_band
"""
from __future__ import annotations
import sys
from pathlib import Path
import joblib, numpy as np, pandas as pd
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from analysis.deep_bet_search import block_boot  # noqa

races = joblib.load(BASE / "data/_policy/bet_substrate.pkl")
rows = []
for r in races:
    p, ban, o = r["p"], r["ban"], r["odds9"]
    if len(p) < 5 or not np.isfinite(o).any():
        continue
    order = np.argsort(-p)
    i1, i2, i3 = order[0], order[1], order[2]
    a1, a2, a3 = int(ban[i1]), int(ban[i2]), int(ban[i3])
    w1, w2 = int(r["waku"][i1]), int(r["waku"][i2])
    t = r["third"]
    wide_map = {frozenset((x, y)): v for x, y, v in (r.get("wide") or ())}
    o1 = float(o[i1]) if np.isfinite(o[i1]) and o[i1] > 0 else np.nan
    f, s = r["first"], r["second"]
    rows.append(dict(
        date=r["date"], year=r["date"][:4], ai_odds=o1, p1=float(p[i1]), p2=float(p[i2]),
        tan=(r["tan_pay"] if a1 == f else 0.0),
        fuku=(r["fuku_pay"].get(a1, 0.0) if a1 in r["fuku_pay"] else 0.0),
        umaren=(r["umaren"] if {a1, a2} == {f, s} and np.isfinite(r["umaren"]) else 0.0),
        umatan=(r["umatan"] if (a1 == f and a2 == s) and np.isfinite(r["umatan"]) else 0.0),
        wide=float(wide_map.get(frozenset((a1, a2)), 0.0) or 0.0),
        wakuren=(r["wakuren"] if {w1, w2} == {int(r["waku_first"]), int(r["waku_second"])} and np.isfinite(r["wakuren"]) else 0.0),
        sanpuku=(r["sanpuku"] if {a1, a2, a3} == {f, s, t} and np.isfinite(r["sanpuku"]) else 0.0),
        sanrentan=(r["sanrentan"] if (a1, a2, a3) == (f, s, t) and np.isfinite(r["sanrentan"]) else 0.0),
    ))
df = pd.DataFrame(rows).dropna(subset=["ai_odds"])
bands = [(0, 2, "<2倍"), (2, 3, "2-3倍"), (3, 5, "3-5倍"), (5, 8, "5-8倍"), (8, 15, "8-15倍"), (15, 1e9, "15倍〜")]
df["band"] = pd.cut(df.ai_odds, [b[0] for b in bands] + [1e9], labels=[b[2] for b in bands], right=False)

print(f"母集団 {len(df)}R / {df.year.min()}-{df.year.max()}  (◎=AI1位の9時単勝オッズで層別、100円均等)\n")
COLS = [("tan","◎単勝"),("fuku","◎複勝"),("umaren","馬連1-2"),("umatan","馬単1→2"),("wide","ワイド1-2"),("wakuren","枠連1-2"),("sanpuku","三連複1-2-3"),("sanrentan","三連単1→2→3")]
hdr = f"{'◎オッズ帯':<8}{'n':>6} | " + " | ".join(f"{k:^12}" for _, k in COLS) + "   (ROI% / 的中%)"
print(hdr)
for b in [x[2] for x in bands] + ["全体"]:
    g = df if b == "全体" else df[df.band == b]
    if len(g) < 30:
        print(f"{b:<8}{len(g):>6} | (少なすぎ)"); continue
    cells = []
    for col, _ in COLS:
        ret = g[col].values
        cells.append(f"{ret.mean():6.1f}/{100*np.mean(ret>0):4.1f}")
    print(f"{b:<8}{len(g):>6} | " + " | ".join(cells))

print("\n年別 × 帯 (馬単1点 / 馬連1点 ROI):")
for b in [x[2] for x in bands]:
    g = df[df.band == b]
    ys = []
    for y, gy in g.groupby("year"):
        ys.append(f"{y}: 馬単{100*gy.umatan.mean()/100:5.1f}/馬連{100*gy.umaren.mean()/100:5.1f} (n={len(gy)})")
    print(f"  {b:<8} " + " | ".join(ys))
