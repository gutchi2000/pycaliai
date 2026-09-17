# -*- coding: utf-8 -*-
"""
umatan_2026_asserved.py — 「馬単 AI1位→2位 1点」を 2026 実serve で検証する
=========================================================================
deep_bet_search (2023-25 OOS, 10,299R) で唯一 100% 線上にあった構造を、
完全に独立な第4期間 = 2026 の実 bundle 出力 × 実 kekka で確かめる。

2023-25 は「モデルを後から流した OOS」、2026 は「実際に朝出していた印」なので、
serve スケール差・運用差を含んだ最も厳しいテストになる。

実行: python -m analysis.umatan_2026_asserved
"""
from __future__ import annotations
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(BASE))
from analysis.deep_bet_search import block_boot  # noqa: E402


def main() -> None:
    rows = []
    for bp in sorted(glob.glob(str(BASE / "reports/cowork_input/*_bundle.json"))):
        date = Path(bp).name[:8]
        kp = BASE / "data" / "kekka" / f"{date}.csv"
        if not date.startswith("2026") or not kp.exists():
            continue
        k = pd.read_csv(kp, encoding="cp932", low_memory=False)
        k["rid16"] = k["レースID(新)"].astype(str).str[:16]
        for c, n in [("馬番", "ban"), ("確定着順", "fin"), ("馬単", "umatan"),
                     ("馬連", "umaren"), ("単勝配当", "tan")]:
            k[n] = pd.to_numeric(k[c], errors="coerce")
        res = {}
        for rid, g in k.groupby("rid16"):
            g = g[g.fin.notna() & (g.fin >= 1)]
            top = g[g.fin <= 3]
            if len(top) < 2 or sorted(top.fin.tolist()[:2]) != [1, 2]:
                continue
            w1 = g[g.fin == 1]
            w2 = g[g.fin == 2]
            if w1.empty or w2.empty or len(w1) > 1 or len(w2) > 1:
                continue
            res[rid] = (int(w1.ban.iloc[0]), int(w2.ban.iloc[0]),
                        float(w1.umatan.iloc[0]) if pd.notna(w1.umatan.iloc[0]) else np.nan,
                        float(w1.umaren.iloc[0]) if pd.notna(w1.umaren.iloc[0]) else np.nan)

        b = json.loads(Path(bp).read_text(encoding="utf-8"))
        races = b["races"] if isinstance(b["races"], list) else list(b["races"].values())
        for r in races:
            rid = "".join(c for c in str(r.get("race_id", "")) if c.isdigit())[:16]
            if rid not in res:
                continue
            hs = [h for h in r.get("horses", []) if h.get("p_win") is not None
                  and h.get("umaban") is not None]
            if len(hs) < 5:
                continue
            order = sorted(hs, key=lambda h: -float(h["p_win"]))
            a, c2 = int(order[0]["umaban"]), int(order[1]["umaban"])
            f, s, ut, ur = res[rid]
            rows.append(dict(
                date=date, rid=rid, n=len(hs),
                p1=float(order[0]["p_win"]), p2=float(order[1]["p_win"]),
                umatan_ret=(ut if (a == f and c2 == s) and np.isfinite(ut) else 0.0),
                umaren_ret=(ur if ({a, c2} == {f, s}) and np.isfinite(ur) else 0.0),
            ))

    df = pd.DataFrame(rows)
    print(f"母集団: {len(df)}R / {df.date.nunique()}開催日 {df.date.min()}-{df.date.max()}\n")
    days = df.date.values

    def rep(label, ret, cost_per=100.0):
        cost = np.full(len(df), cost_per)
        lo, hi, p100 = block_boot(cost, ret, days, reps=4000)
        print(f"  {label:<28} n={len(df)} 的中={100*np.mean(ret>0):>5.2f}% "
              f"ROI={100*ret.sum()/cost.sum():>7.2f}% CI95=[{lo:.1f},{hi:.1f}] "
              f"P(>100)={p100:.3f}")

    print("=== 2026 as-served (実際に朝出していた印) ===")
    rep("馬単 AI1位→2位 1点 [床77.5]", df.umatan_ret.values)
    rep("馬連 AI1-2位 1点 [床77.5]", df.umaren_ret.values)

    print("\n=== 条件別 (馬単1点) ===")
    df["gap"] = df.p1 - df.p2
    segs = {
        "混戦 gap<.04": df.gap < 0.04,
        "中間 .04-.10": (df.gap >= 0.04) & (df.gap < 0.10),
        "独走 gap>=.10": df.gap >= 0.10,
        "少頭数<=12": df.n <= 12,
        "多頭数>=16": df.n >= 16,
    }
    for name, m in segs.items():
        if m.sum() < 80:
            print(f"  {name:<28} n={int(m.sum())} (少なすぎ)")
            continue
        ret = df.umatan_ret.values[m.values]
        cost = np.full(int(m.sum()), 100.0)
        lo, hi, p100 = block_boot(cost, ret, days[m.values], reps=3000)
        print(f"  {name:<28} n={int(m.sum())} 的中={100*np.mean(ret>0):>5.2f}% "
              f"ROI={100*ret.sum()/cost.sum():>7.2f}% CI95=[{lo:.1f},{hi:.1f}] P(>100)={p100:.3f}")

    print("\n=== 月別 (馬単1点) ===")
    for m, g in df.assign(mm=df.date.str[:6]).groupby("mm"):
        r = g.umatan_ret.values
        print(f"  {m} n={len(g):>4} 的中={100*np.mean(r>0):>5.2f}% ROI={r.sum()/len(g):>7.2f}%")

    print("\n=== 参考: 本番が同期間に実際に出した成績 ===")
    print("  全券種実績 ROI 71.8% / 複勝中心 / 投資 4,141,834円 収支 -1,169,777円")
    tot = len(df) * 100
    print(f"  馬単1点を全{len(df)}Rで100円ずつ: 投資 {tot:,}円 "
          f"収支 {int(df.umatan_ret.sum()-tot):+,}円 ROI {100*df.umatan_ret.sum()/tot:.1f}%")


if __name__ == "__main__":
    main()
