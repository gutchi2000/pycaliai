# -*- coding: utf-8 -*-
"""
alloc_95_design.py — 「◎オッズ帯×券種で ROI>=95 のセルを買う」設計に、1R 上限 1 万円・100 円単位で
どう配分するのが良いかを比べる。

配分規則はすべて fit 年だけで決め (採用セル・各セルの的中率・ROI)、別の年で決済する。
  均等 / 的中率比例(≒等払戻) / √的中率比例 / 的中率逆比例(穴厚め) / ROI超過比例 / 帯内最良セル全額 / 馬単全額
実行: python -m analysis.alloc_95_design
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
_src = (BASE / "analysis" / "top1_by_odds_band.py").read_text(encoding="utf-8").split('print(f"母集団')[0]
exec(_src)  # df (band, year, date, 8 券種の払戻/100円)
from analysis.deep_bet_search import block_boot  # noqa

COLS = ["tan", "fuku", "umaren", "umatan", "wide", "wakuren", "sanpuku", "sanrentan"]
JP = dict(tan="単勝", fuku="複勝", umaren="馬連", umatan="馬単", wide="ワイド", wakuren="枠連", sanpuku="三連複", sanrentan="三連単")
LABELS = list(df.band.cat.categories)
CAP, UNIT, THR = 10_000, 100, 95.0


def fit_stats(fit):
    st = {}
    for b in LABELS:
        g = fit[fit.band == b]
        if len(g) < 30:
            continue
        cells = [c for c in COLS if g[c].mean() >= THR]
        st[b] = {c: dict(roi=g[c].mean(), hit=max((g[c] > 0).mean(), 1e-4)) for c in cells}
    return st


def weights(cells: dict, rule: str):
    ks = list(cells)
    if not ks:
        return {}
    if rule == "均等":
        w = np.ones(len(ks))
    elif rule == "的中率比例":
        w = np.array([cells[k]["hit"] for k in ks])
    elif rule == "√的中率比例":
        w = np.sqrt([cells[k]["hit"] for k in ks])
    elif rule == "的中率逆比例":
        w = 1.0 / np.array([cells[k]["hit"] for k in ks])
    elif rule == "ROI超過比例":
        w = np.array([max(cells[k]["roi"] - 90.0, 1.0) for k in ks])
    elif rule == "帯内最良セル全額":
        w = np.zeros(len(ks)); w[int(np.argmax([cells[k]["roi"] for k in ks]))] = 1.0
    elif rule == "馬単全額":
        w = np.array([1.0 if k == "umatan" else 0.0 for k in ks])
        if w.sum() == 0:
            w = np.ones(len(ks))
    else:
        raise ValueError(rule)
    w = w / w.sum()
    # 100 円格子: 正の重みのセルには最低 100 円、合計ちょうど CAP
    pos = w > 0
    stake = np.zeros(len(ks))
    stake[pos] = np.maximum(np.floor(w[pos] * CAP / UNIT) * UNIT, UNIT)
    diff = CAP - stake.sum()
    order = np.argsort(-w)
    i = 0
    while diff != 0 and i < 10_000:
        j = order[i % len(order)]
        if pos[j]:
            step = UNIT if diff > 0 else -UNIT
            if stake[j] + step >= UNIT:
                stake[j] += step; diff -= step
        i += 1
    return {k: float(s) for k, s in zip(ks, stake) if s > 0}


def settle(ev, st, rule):
    plan = {b: weights(st.get(b, {}), rule) for b in LABELS}
    rows = []
    for r in ev.itertuples():
        w = plan.get(str(r.band), {})
        if not w:
            continue
        cost = sum(w.values())
        ret = sum(getattr(r, c) / 100.0 * s for c, s in w.items())
        hit = any(getattr(r, c) > 0 for c in w)
        rows.append((r.date, cost, ret, hit))
    return pd.DataFrame(rows, columns=["date", "cost", "ret", "hit"]), plan


def metrics(b):
    cost, ret = b.cost.values, b.ret.values
    lo, hi, p = block_boot(cost, ret, b.date.values, reps=2000)
    pnl = ret - cost
    cum = np.cumsum(pnl); dd = float((np.maximum.accumulate(np.maximum(cum, 0)) - cum).max())
    streak = max(len(s) for s in "".join("W" if x >= 0 else "L" for x in pnl).split("W"))
    top = np.sort(ret)[::-1]
    # 1 年 (104 開催日) リサンプル
    g = b.groupby("date").agg(ret=("ret", "sum"), cost=("cost", "sum"))
    rng = np.random.default_rng(0); idx = rng.integers(0, len(g), size=(3000, 104))
    ypnl = g.ret.values[idx].sum(1) - g.cost.values[idx].sum(1)
    return dict(roi=100 * ret.sum() / cost.sum(), lo=lo, hi=hi, p100=p,
                ex10=100 * (ret.sum() - top[:10].sum()) / cost.sum(),
                hit=100 * b.hit.mean(), plus=100 * (pnl >= 0).mean(), trig=100 * (b.hit & (pnl < 0)).mean(),
                dd=dd, streak=streak, yplus=100 * (ypnl > 0).mean(),
                ymed=float(np.median(ypnl)), y5=float(np.percentile(ypnl, 5)), y95=float(np.percentile(ypnl, 95)))


RULES = ["均等", "的中率比例", "√的中率比例", "的中率逆比例", "ROI超過比例", "帯内最良セル全額", "馬単全額"]
SPLITS = [(["2023"], ["2024", "2025"]), (["2025"], ["2023", "2024"]), (["2023", "2024"], ["2025"]), (["2024", "2025"], ["2023"])]


def main():
    print(f"1R 上限 ¥{CAP:,} / 単位 ¥{UNIT} / 採用セル ROI>={THR:.0f} (fit 年で選択、別の年で決済)\n")
    agg = {r: [] for r in RULES}
    for fy, ey in SPLITS:
        st = fit_stats(df[df.year.isin(fy)]); ev = df[df.year.isin(ey)].sort_values("date")
        print(f"--- {'+'.join(fy)} で選択 → {'+'.join(ey)} で決済")
        for rule in RULES:
            b, _ = settle(ev, st, rule); m = metrics(b); agg[rule].append(m)
            print(f"  {rule:<10} ROI {m['roi']:6.1f}% [{m['lo']:5.1f},{m['hi']:5.1f}] 上位10除外 {m['ex10']:5.1f}% | R的中 {m['hit']:4.1f}% 収支+R {m['plus']:4.1f}% トリガミ {m['trig']:4.1f}% | 最大DD ¥{m['dd']:>10,.0f} 連敗 {m['streak']:>3} | 年+確率 {m['yplus']:4.1f}% 年中央 ¥{m['ymed']:>+11,.0f} (5% ¥{m['y5']:>+11,.0f} / 95% ¥{m['y95']:>+11,.0f})")
    print("\n=== 4 方向の平均 (OOS) ===")
    print(f"  {'配分':<10} {'ROI':>6} {'最悪方向':>7} {'上位10除外':>8} {'R的中':>6} {'収支+R':>6} {'トリガミ':>6} {'最大DD(平均)':>13} {'年+確率':>7} {'年中央(平均)':>13} {'年5%点(平均)':>13}")
    for rule in RULES:
        ms = agg[rule]; f = lambda k: float(np.mean([m[k] for m in ms]))
        print(f"  {rule:<10} {f('roi'):6.1f} {min(m['roi'] for m in ms):7.1f} {f('ex10'):8.1f} {f('hit'):6.1f} {f('plus'):6.1f} {f('trig'):6.1f} {f('dd'):>13,.0f} {f('yplus'):7.1f} {f('ymed'):>+13,.0f} {f('y5'):>+13,.0f}")
    print("\n=== 参考: 全期間表(2023-25)で作った配分表そのもの ===")
    st = fit_stats(df)
    for rule in ("均等", "√的中率比例", "的中率比例"):
        print(f"[{rule}]")
        for b in LABELS:
            w = weights(st.get(b, {}), rule)
            print(f"   {b:<6} " + " / ".join(f"{JP[c]} ¥{int(s):,}" for c, s in w.items()))


if __name__ == "__main__":
    main()
