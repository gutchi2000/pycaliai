# -*- coding: utf-8 -*-
"""
a95_chaos_gate_check.py — 「混戦度の高いレースは見送り」をかけた母集団で、95以上ルールがどうなるか。

混戦度 = 較正 p_win の正規化エントロピー (本番 field_chaos_score と同じ定義)。本番ガードは
参照分布の percentile 0.667 以上を見送るので、ここでは 2023-25 OOS 母集団の上位 33.3% を
見送りとして近似する (+ 7 頭以下・AI1位 p_win<0.05 も見送り)。1R 1万円・均等・100円格子。
実行: python -m analysis.a95_chaos_gate_check
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd, joblib
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from analysis.deep_bet_search import block_boot  # noqa
import shadow_a95 as m

pol = m.load_policy()
COLS = ["tan", "fuku", "umaren", "umatan", "wide", "wakuren", "sanpuku", "sanrentan"]
JP = m.JP
LABELS = [b[2] for b in pol["bands"]]
rows = []
for r in joblib.load(BASE / "data/_policy/bet_substrate.pkl"):
    p, ban, o = np.asarray(r["p"], float), r["ban"], r["odds9"]
    n = len(p)
    if n < 5 or not np.isfinite(o).any():
        continue
    order = np.argsort(-p); i1, i2, i3 = order[:3]
    o1 = float(o[i1]) if np.isfinite(o[i1]) and o[i1] > 0 else np.nan
    if not np.isfinite(o1):
        continue
    pp = p / p.sum(); H = float(-(pp[pp > 0] * np.log(pp[pp > 0])).sum() / np.log(n))
    a1, a2, a3 = int(ban[i1]), int(ban[i2]), int(ban[i3]); f, s, t = r["first"], r["second"], r["third"]
    w1, w2 = int(r["waku"][i1]), int(r["waku"][i2])
    wide = {frozenset((x, y)): v for x, y, v in (r.get("wide") or ())}
    per = dict(tan=(r["tan_pay"] if a1 == f else 0.0), fuku=r["fuku_pay"].get(a1, 0.0),
               umaren=(r["umaren"] if {a1, a2} == {f, s} else 0.0),
               umatan=(r["umatan"] if (a1, a2) == (f, s) else 0.0),
               wide=float(wide.get(frozenset((a1, a2)), 0.0)),
               wakuren=(r["wakuren"] if {w1, w2} == {int(r["waku_first"]), int(r["waku_second"])} else 0.0),
               sanpuku=(r["sanpuku"] if {a1, a2, a3} == {f, s, t} else 0.0),
               sanrentan=(r["sanrentan"] if (a1, a2, a3) == (f, s, t) else 0.0))
    per = {k: (float(v) if np.isfinite(v) else 0.0) for k, v in per.items()}
    rows.append(dict(date=r["date"], year=r["date"][:4], n=n, H=H, p1=float(p[i1]),
                     band=m.band_of(o1, pol), **per))
df = pd.DataFrame(rows)
thr = df.H.quantile(0.667)
df["keep"] = (df.H < thr) & (df.n > 7) & (df.p1 >= 0.05)


def table(d):
    return {b: {c: (d[d.band == b][c].mean() if (d.band == b).sum() >= 30 else np.nan) for c in COLS} for b in LABELS}


def plan_from(d, thr_roi=95.0):
    tb = table(d)
    return {b: [c for c in COLS if np.isfinite(tb[b][c]) and tb[b][c] >= thr_roi] for b in LABELS}


def settle(d, plan):
    out = []
    for r in d.itertuples():
        kinds = plan.get(r.band) or []
        if not kinds:
            continue
        st = m.equal_stakes(kinds, 10000, 100)
        ret = sum(getattr(r, k) / 100 * v for k, v in st.items())
        out.append((r.date, r.year, ret, any(getattr(r, k) > 0 for k in st)))
    return pd.DataFrame(out, columns=["date", "year", "ret", "hit"])


def rep(name, b, n_pop):
    if len(b) == 0:
        print(f"{name:<44} (買い目なし)"); return
    cost = np.full(len(b), 10000.0); lo, hi, p = block_boot(cost, b.ret.values, b.date.values, reps=2000)
    top = np.sort(b.ret.values)[::-1]
    ys = " / ".join(f"{y}:{100*g.ret.sum()/(1e4*len(g)):.0f}" for y, g in b.groupby("year"))
    print(f"{name:<44} R={len(b):>5} ({100*len(b)/n_pop:4.0f}%) ROI {100*b.ret.sum()/cost.sum():6.1f}% [{lo:5.1f},{hi:5.1f}] "
          f"P>100={p:.2f} 上位10除外 {100*(b.ret.sum()-top[:10].sum())/cost.sum():5.1f}% R的中 {100*b.hit.mean():4.1f}% | {ys}")


keep, skip = df[df.keep], df[~df.keep]
print(f"母集団 {len(df)}R / 参戦(低混戦) {len(keep)}R ({100*len(keep)/len(df):.0f}%) / 見送り {len(skip)}R\n")
print("■ 参戦側だけで作った帯別 1 点 ROI 表 (95 以上に *)")
tb = table(keep)
print(f"{'帯':<7}{'n':>6} " + " ".join(f"{JP[c]:>7}" for c in COLS))
for b in LABELS:
    n = int((keep.band == b).sum())
    print(f"{b:<7}{n:>6} " + " ".join((f"{tb[b][c]:6.1f}" + ("*" if np.isfinite(tb[b][c]) and tb[b][c] >= 95 else " ")) if np.isfinite(tb[b][c]) else "     - " for c in COLS))
pk = plan_from(keep)
print("\n参戦側の表で選ぶ 95 以上セル:")
for b in LABELS:
    print(f"   {b:<6} {[JP[c] for c in pk[b]]}")

print("\n■ 同じ表で選んで同じ期間で決済 (in-sample)")
rep("凍結表(全レースで作成) × 全レース", settle(df, pol["plan"]), len(df))
rep("凍結表(全レースで作成) × 参戦側のみ", settle(keep, pol["plan"]), len(df))
rep("参戦側で作り直した表 × 参戦側のみ", settle(keep, pk), len(df))
rep("凍結表 × 見送られる側", settle(skip, pol["plan"]), len(df))

print("\n■ 年で選んで別の年で決済 (参戦側のみで評価)")
for fy, ey in [(["2023"], ["2024", "2025"]), (["2025"], ["2023", "2024"]), (["2023", "2024"], ["2025"]), (["2024", "2025"], ["2023"])]:
    ev = keep[keep.year.isin(ey)]
    rep(f"{'+'.join(fy)}→{'+'.join(ey)} 全レースの表で選択", settle(ev, plan_from(df[df.year.isin(fy)])), len(ev))
    rep(f"{'+'.join(fy)}→{'+'.join(ey)} 参戦側の表で選択", settle(ev, plan_from(keep[keep.year.isin(fy)])), len(ev))
