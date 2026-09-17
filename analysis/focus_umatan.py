# -*- coding: utf-8 -*-
"""
focus_umatan.py — deep_bet_search が唯一 100% 線上に置いた構造を精査する
=======================================================================
総当たり 44,928 セルで FDR を通ったセルはゼロだったが、「素の AI 順」ベースラインの
中で 馬単 axis m=2 (= AI 1位→2位 の1点) だけが 3期とも ~100% 付近に居た。
これが本物か、ただの高分散かを、年別/月別/条件別 + ブロック bootstrap で詰める。

実行: python -m analysis.focus_umatan
"""
from __future__ import annotations
import itertools
from pathlib import Path

import joblib
import numpy as np

BASE = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(BASE))
from analysis.deep_bet_search import (orderings, tickets, settle, race_features,  # noqa: E402
                                      block_boot, SEGMENTS, TAKEOUT)


def line(label, cost, ret, days, reps=4000):
    c, v = cost.sum(), ret.sum()
    if c <= 0:
        print(f"  {label:<26} (n=0)")
        return
    lo, hi, p100 = block_boot(cost, ret, days, reps=reps)
    hit = 100 * float(np.mean(ret > 0))
    print(f"  {label:<26} n={len(cost):>5} 的中={hit:>5.2f}% ROI={100*v/c:>7.2f}% "
          f"CI95=[{lo:>6.1f},{hi:>6.1f}] P(>100)={p100:.3f}")


def main() -> None:
    races = joblib.load(BASE / "data/_policy/bet_substrate.pkl")
    feats = [race_features(r) for r in races]
    ords = [orderings(r) for r in races]
    year = np.array([r["date"][:4] for r in races])
    days = np.array([r["date"] for r in races])

    CANDS = [
        ("馬単", "axis", "ai", 2, "AI1位→2位 1点"),
        ("馬単", "axis", "ai", 3, "AI1位→2,3位 2点"),
        ("馬単", "axis", "ai", 4, "AI1位→2-4位 3点"),
        ("馬単", "box", "ai", 2, "AI1-2位 表裏 2点"),
        ("馬連", "box", "ai", 2, "AI1-2位 1点"),
        ("馬連", "axis", "ai", 3, "AI1位軸→2,3位 2点"),
        ("馬単", "axis", "blend", 2, "blend1位→2位 1点"),
        ("馬単", "axis", "mkt", 2, "1人気→2人気 1点(市場)"),
        ("ワイド", "box", "ai", 2, "AI1-2位 ワイド1点"),
        ("単勝", "topk", "ai", 1, "AI1位 単勝"),
        ("複勝", "topk", "ai", 1, "AI1位 複勝"),
    ]

    print("===== 3期通し (2023+2024+2025, 10,299R) =====")
    store = {}
    for kind, struct, on, m, label in CANDS:
        cost = np.zeros(len(races)); ret = np.zeros(len(races))
        for ri, r in enumerate(races):
            c, v = settle(kind, set(tickets(kind, struct, ords[ri][on], r, m)), r)
            cost[ri], ret[ri] = c, v
        store[label] = (cost, ret)
        act = cost > 0
        line(f"{label} [床{TAKEOUT[kind]:.1f}%]", cost[act], ret[act], days[act])

    print("\n===== 馬単 AI1位→2位 1点: 年別 =====")
    cost, ret = store["AI1位→2位 1点"]
    for y in ("2023", "2024", "2025"):
        m = (year == y) & (cost > 0)
        line(y, cost[m], ret[m], days[m])

    print("\n===== 馬単 AI1位→2位 1点: 半期別 (regime 安定性) =====")
    half = np.array([r["date"][:4] + ("H1" if int(r["date"][4:6]) <= 6 else "H2") for r in races])
    for h in sorted(set(half)):
        m = (half == h) & (cost > 0)
        line(h, cost[m], ret[m], days[m])

    print("\n===== 馬単 AI1位→2位 1点: 条件別 =====")
    for sname, fn in SEGMENTS.items():
        m = np.array([fn(f) for f in feats]) & (cost > 0)
        if m.sum() >= 300:
            line(sname, cost[m], ret[m], days[m], reps=2000)

    print("\n===== 払戻分布 (トップ数本に依存していないか) =====")
    act = cost > 0
    pays = np.sort(ret[act])[::-1]
    tot = pays.sum()
    for k in (1, 3, 5, 10, 20):
        print(f"  上位{k:>2}本の払戻が全払戻に占める割合: {100*pays[:k].sum()/tot:>5.2f}%  "
              f"(これを除いた ROI = {100*(tot-pays[:k].sum())/cost[act].sum():>6.2f}%)")


if __name__ == "__main__":
    main()
