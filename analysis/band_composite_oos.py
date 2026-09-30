import sys
sys.path.insert(0, r"E:\PyCaLiAI")
from pathlib import Path
import numpy as np, pandas as pd, joblib
from analysis.deep_bet_search import block_boot
races = joblib.load(r"E:\PyCaLiAI\data\_policy\bet_substrate.pkl")
rows = []
for r in races:
    p, ban, o = r["p"], r["ban"], r["odds9"]
    if len(p) < 5 or not np.isfinite(o).any():
        continue
    order = np.argsort(-p); i1, i2 = order[0], order[1]
    a1, a2 = int(ban[i1]), int(ban[i2])
    o1 = float(o[i1]) if np.isfinite(o[i1]) and o[i1] > 0 else np.nan
    f, s = r["first"], r["second"]
    rows.append(dict(date=r["date"], year=r["date"][:4], ai_odds=o1,
        tan=(r["tan_pay"] if a1 == f else 0.0),
        fuku=(r["fuku_pay"].get(a1, 0.0)),
        umaren=(r["umaren"] if {a1, a2} == {f, s} and np.isfinite(r["umaren"]) else 0.0),
        umatan=(r["umatan"] if (a1 == f and a2 == s) and np.isfinite(r["umatan"]) else 0.0)))
df = pd.DataFrame(rows).dropna(subset=["ai_odds"])
edges = [0, 2, 3, 5, 8, 15, 1e9]; labels = ["<2", "2-3", "3-5", "5-8", "8-15", "15+"]
df["band"] = pd.cut(df.ai_odds, edges, labels=labels, right=False)
cols = ["tan", "fuku", "umaren", "umatan"]

def composite(fit_years, ev_years, tag):
    fit = df[df.year.isin(fit_years)]; ev = df[df.year.isin(ev_years)]
    choice = {}
    for b in labels:
        g = fit[fit.band == b]
        choice[b] = max(cols, key=lambda c: g[c].mean()) if len(g) >= 30 else "fuku"
    ret = np.array([getattr(row, choice[row.band]) for row in ev.itertuples()])
    cost = np.full(len(ev), 100.0)
    lo, hi, p = block_boot(cost, ret, ev.date.values, reps=3000)
    fit_ret = np.array([getattr(row, choice[row.band]) for row in fit.itertuples()])
    print(f"[{tag}] 選択: {choice}")
    print(f"   fit {fit_years} in-sample ROI={fit_ret.mean():.1f}%  →  OOS {ev_years}: n={len(ev)} ROI={100*ret.sum()/cost.sum():.1f}% CI95=[{lo:.1f},{hi:.1f}] P(>100)={p:.3f}")
    for y, gy in ev.groupby("year"):
        r = np.array([getattr(row, choice[row.band]) for row in gy.itertuples()]); print(f"     {y}: {r.mean():.1f}%")

composite(["2023"], ["2024", "2025"], "2023で設計→2024-25")
composite(["2025"], ["2023", "2024"], "2025で設計→2023-24")
composite(["2023", "2024"], ["2025"], "2023-24で設計→2025")
ev = df[df.year != "2023"]; cost = np.full(len(ev), 100.0)
print("対照 (2024-25, 全帯1点固定):")
for c in cols:
    r = ev[c].values; lo, hi, p = block_boot(cost, r, ev.date.values, reps=2000)
    print(f"   {c:<7} {r.mean():.1f}% [{lo:.1f},{hi:.1f}]")

print("\n=== 「fit期で90%(or 95/100)超えのセルを全部採用」→ OOS (各セル100円均等) ===")
def thresh(fit_years, ev_years, thr):
    fit = df[df.year.isin(fit_years)]; ev = df[df.year.isin(ev_years)]
    sel = [(b, c) for b in labels for c in cols if (fit.band == b).sum() >= 30 and fit[fit.band == b][c].mean() >= thr]
    rets, costs, dates = [], [], []
    for row in ev.itertuples():
        for b, c in sel:
            if row.band == b:
                rets.append(getattr(row, c)); costs.append(100.0); dates.append(row.date)
    rets, costs, dates = np.array(rets), np.array(costs), np.array(dates)
    lo, hi, p = block_boot(costs, rets, dates, reps=2000)
    print(f"  thr>={thr:>3} fit={fit_years} 採用{len(sel):>2}セル {[f'{b}:{c}' for b,c in sel]}")
    print(f"      OOS {ev_years}: bets={len(rets)} ROI={100*rets.sum()/costs.sum():.1f}% CI95=[{lo:.1f},{hi:.1f}] P(>100)={p:.3f}")
for thr in (90, 95, 100):
    thresh(["2023"], ["2024", "2025"], thr)
    thresh(["2025"], ["2023", "2024"], thr)
