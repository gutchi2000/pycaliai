"""
sweep_hit40.py
==============
v4 で (bet_type, K, EV_min, race_thr) のグリッド全探索し
「hit_rate ∈ [0.35, 0.45] かつ ROI ≥ 1.00」を満たす設定を列挙。
OOS = 2024+2025.
"""
from __future__ import annotations
import warnings
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import pl_probs as PL
from backtest_pl_ev import (
    load_payouts, score_test, lookup_pay_vec,
    all_umaren_mat, all_umatan_mat, all_fukusho_vec_fast, all_sanrenpuku_tensor,
    COL_RID, COL_JYUN, COL_BAN, BET,
)
from backtest_pl_formation import race_combos
import backtest_pl_ev as be

warnings.filterwarnings("ignore")
BASE = Path(__file__).parent

# v4 で差し替え
be.MODEL_PKL = BASE / "models/unified_rank_v4.pkl"
be.CAL_PKL   = BASE / "models/pl_calibrators_v4.pkl"
be.CURVE_PKL = BASE / "data/pl_payout_curve_v4.pkl"

RACE_BUDGET = 10000
KELLY_FRAC  = 0.25
UNIT        = 100

GRID = {
    "tansho":     {"K": [1,2,3], "ev": [100,110,120,130,150], "thr": [1.00,1.05,1.10,1.20]},
    "fukusho":    {"K": [1,2,3,4], "ev": [90,95,100,105,110], "thr": [1.00,1.05,1.10]},
    "umaren":     {"K": [1,2,3,5,8], "ev": [80,100,120,150,200], "thr": [1.00,1.05,1.10,1.20,1.30]},
    "umatan":     {"K": [1,2,3,5,8], "ev": [80,100,120,150,200], "thr": [1.00,1.05,1.10,1.20,1.30]},
    "sanrenpuku": {"K": [3,5,8,12], "ev": [100,120,150,200], "thr": [1.00,1.10,1.20,1.30]},
}
HIT_LO, HIT_HI, ROI_MIN = 0.35, 0.45, 1.00


def simulate(te, payouts):
    cal = joblib.load(be.CAL_PKL)["calibrators"]
    curves = joblib.load(be.CURVE_PKL)["curves"]

    # 事前に (bet, K, ev, thr) -> stats
    keys = []
    for bet, g in GRID.items():
        for K, ev, thr in product(g["K"], g["ev"], g["thr"]):
            keys.append((bet, K, ev, thr))
    stats = {k: {"races":0,"bets":0,"hits":0,"stake":0,"ret":0.0} for k in keys}

    n_race = 0
    for rid, g in te.groupby(COL_RID, sort=False):
        if len(g) < 3: continue
        rid_s = str(int(rid)) if isinstance(rid, (int, np.integer)) else str(rid)
        if rid_s not in payouts: continue
        po = payouts[rid_s]
        g = g.sort_values(COL_BAN).reset_index(drop=True)
        ban = g[COL_BAN].astype(int).values
        b2i = {int(b): i for i, b in enumerate(ban)}
        if any(b2i.get(po[k]) is None for k in ("win","plc","sho")): continue
        w = PL.pl_weights(g["_score"].values)
        combos = race_combos(w, po, b2i, cal, curves)
        n_race += 1

        for bet, K, ev_min, thr in keys:
            d = combos[bet]
            ev = d["ev"]; hit = d["hit"]; actual = d["actual"]
            if len(ev) == 0: continue
            cand_idx = np.where(ev >= ev_min)[0]
            if len(cand_idx) == 0: continue
            order_in_cand = np.argsort(-ev[cand_idx])
            top = cand_idx[order_in_cand[:K]]
            ev_top = ev[top]
            edge = np.clip((ev_top - BET) / BET, 0, None)
            if edge.sum() == 0: continue
            ratios = edge / edge.sum()
            stakes = np.floor(ratios * RACE_BUDGET * KELLY_FRAC / UNIT).astype(int) * UNIT
            if stakes.sum() == 0:
                stakes = np.zeros_like(edge, dtype=int); stakes[0] = UNIT
            total_stake = int(stakes.sum())
            race_roi = float((ev_top * stakes / BET).sum()) / total_stake if total_stake > 0 else 0
            if race_roi < thr: continue

            s = stats[(bet, K, ev_min, thr)]
            s["races"] += 1
            s["bets"]  += int((stakes > 0).sum())
            s["stake"] += total_stake
            hit_mask = hit[top] & (stakes > 0)
            s["hits"] += int(hit_mask.sum())
            ret = float(((stakes / BET) * actual[top] * hit[top]).sum())
            s["ret"]  += ret

        if n_race % 1000 == 0:
            print(f"  ..{n_race} races")
    return stats, n_race


def main():
    payouts = load_payouts()
    te = score_test()
    sub = te[te["year"].isin([2024, 2025])]
    stats, n_race = simulate(sub, payouts)

    rows = []
    for (bet, K, ev, thr), s in stats.items():
        if s["bets"] < 10: continue   # 信頼性のため最低 10 bets
        hr = s["hits"] / s["bets"]
        roi = s["ret"] / s["stake"] if s["stake"] > 0 else 0
        rows.append({
            "bet": bet, "K": K, "ev_min": ev, "thr": thr,
            "races": s["races"], "bets": s["bets"], "hits": s["hits"],
            "stake": s["stake"], "ret": int(s["ret"]),
            "roi": roi, "hit_rate": hr,
        })

    # ★ 条件フィルタ: hit 35-45%, ROI ≥ 100%
    matched = [r for r in rows if HIT_LO <= r["hit_rate"] <= HIT_HI and r["roi"] >= ROI_MIN]
    matched.sort(key=lambda r: (-r["races"], -r["roi"]))

    print(f"\n=== hit 35-45% & ROI >= 100% : {len(matched)} configs ===")
    print(f"  {'馬券':10s} {'K':>2s} {'EV':>4s} {'thr':>5s} {'R':>5s} {'bets':>5s} "
          f"{'hit%':>6s} {'stake':>9s} {'ret':>9s} {'ROI%':>6s}")
    for r in matched[:40]:
        print(f"  {r['bet']:10s} {r['K']:>2d} {r['ev_min']:>4d} {r['thr']:>5.2f} "
              f"{r['races']:>5,} {r['bets']:>5,} {r['hit_rate']*100:>5.2f} "
              f"{r['stake']:>9,} {r['ret']:>9,} {r['roi']*100:>6.2f}")

    # 全 rows を races 多い順に上位 20 表示
    rows.sort(key=lambda r: -r["races"])
    print(f"\n--- top 20 by race count (any hit/roi) ---")
    for r in rows[:20]:
        print(f"  {r['bet']:10s} {r['K']:>2d} {r['ev_min']:>4d} {r['thr']:>5.2f} "
              f"{r['races']:>5,} {r['bets']:>5,} {r['hit_rate']*100:>5.2f} "
              f"{r['stake']:>9,} {r['ret']:>9,} {r['roi']*100:>6.2f}")


if __name__ == "__main__":
    main()
