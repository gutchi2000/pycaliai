"""
sweep_target.py
===============
Stage 1 grid sweep: 5 馬券 (単・複・馬連・馬単・ワイド) × (K, EV_min, race_thr)
  + 馬連/馬単の「軸流し」(AI top-1 を固定) モード

Filter: hit_rate ∈ [0.30, 0.40] AND roi ≥ 0.90 AND races ≥ 300

OOS = test 2024+2025. モデルは環境変数 UNIFIED_MODEL / UNIFIED_CAL / UNIFIED_CURVE
で切替 (既定 v4). Kelly 設定は backtest_pl_kelly と同一 (RACE_BUDGET=10000,
KELLY_FRAC=0.25, UNIT=100).

再現性: 完全 deterministic (乱数未使用). 同入力 → 同出力.
"""
from __future__ import annotations
import argparse
import json
import warnings
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import pl_probs as PL
from backtest_pl_ev import (
    load_payouts, load_wide_payouts, score_test,
    COL_RID, COL_BAN, BET,
)
from backtest_pl_formation import race_combos
import backtest_pl_ev as be

warnings.filterwarnings("ignore")
BASE = Path(__file__).parent

RACE_BUDGET = 10000
KELLY_FRAC  = 0.25
UNIT        = 100

# --alloc 選択肢
#   equal : 各 K 本に ¥(RACE_BUDGET*KELLY_FRAC/K) を等分 (切り捨て UNIT 単位)
#   kelly : edge 比率 Kelly (#1 に集中 → 実質 1 点買い化、参考用)
DEFAULT_ALLOC = "equal"

# 合否フィルタ
HIT_LO, HIT_HI, ROI_MIN, MIN_RACES = 0.30, 0.40, 0.90, 300

# グリッド (flow=False = 普通の EV top-K; flow=True = AI top-1 を含む combo の top-K)
GRID = {
    ("tansho",  False):    {"K": [1, 2, 3],          "ev": [80, 100, 120, 150],      "thr": [1.00, 1.05, 1.10]},
    ("fukusho", False):    {"K": [1, 2, 3, 4, 5],    "ev": [60, 80, 95, 110, 130],   "thr": [0.90, 1.00, 1.10]},
    ("umaren",  False):    {"K": [1, 2, 3, 5, 8],    "ev": [60, 80, 100, 120],       "thr": [0.80, 1.00, 1.10, 1.20]},
    ("umatan",  False):    {"K": [1, 2, 3, 5, 8],    "ev": [60, 80, 100, 120],       "thr": [0.80, 1.00, 1.10, 1.20]},
    ("wide",    False):    {"K": [1, 2, 3, 5],       "ev": [60, 80, 100, 120],       "thr": [0.80, 1.00, 1.10]},
    ("umaren",  True):     {"K": [3, 5, 8],          "ev": [60, 80, 100],            "thr": [0.80, 1.00]},
    ("umatan",  True):     {"K": [3, 5, 8],          "ev": [60, 80, 100],            "thr": [0.80, 1.00]},
}


def axis_mask(bet, iu, ju, n):
    """(bet, flow=True) で AI top-1 (idx=axis) を含む combo の bool mask.
    axis は raceごとに変わる — ここでは引数 n と axis_idx を別途渡す."""
    raise NotImplementedError  # 呼び出し側で直接 mask を構築


def allocate(ev_top, alloc):
    """K 本の combo に stake を配分し (stakes, total_stake) を返す.
    alloc="equal" : RACE_BUDGET*KELLY_FRAC を K 等分 → UNIT 単位切り捨て, 全 combo 同額
    alloc="kelly" : edge 比率 Kelly (従来, 1 点集中傾向)
    どちらも hit 判定 / 払戻計算には stakes[k]>0 の combo のみ寄与する.
    """
    K = len(ev_top)
    if alloc == "equal":
        per = int(np.floor(RACE_BUDGET * KELLY_FRAC / K / UNIT)) * UNIT
        if per <= 0:
            # K が大きすぎて 1 本 ¥100 未満になる場合 → UNIT を最小に, 余りは捨てる
            per = UNIT
            # ただし 100*K > budget なら K' に絞る
            K_use = min(K, RACE_BUDGET * KELLY_FRAC // UNIT)
            stakes = np.zeros(K, dtype=int)
            stakes[:int(K_use)] = UNIT
            return stakes, int(stakes.sum())
        stakes = np.full(K, per, dtype=int)
        return stakes, int(stakes.sum())
    # kelly
    edge = np.clip((ev_top - BET) / BET, 0, None)
    if edge.sum() == 0:
        return np.zeros(K, dtype=int), 0
    ratios = edge / edge.sum()
    stakes = np.floor(ratios * RACE_BUDGET * KELLY_FRAC / UNIT).astype(int) * UNIT
    if stakes.sum() == 0:
        stakes = np.zeros(K, dtype=int); stakes[0] = UNIT
    return stakes, int(stakes.sum())


def simulate(te, payouts, wide_map, alloc):
    cal = joblib.load(be.CAL_PKL)["calibrators"]
    curves = joblib.load(be.CURVE_PKL)["curves"]

    # 事前 key 列挙
    keys = []
    for (bet, flow), g in GRID.items():
        for K, ev_min, thr in product(g["K"], g["ev"], g["thr"]):
            keys.append((bet, flow, K, ev_min, thr))
    stats = {k: {"races": 0, "bets": 0, "hits": 0, "stake": 0, "ret": 0.0} for k in keys}

    n_race = 0
    for rid, g in te.groupby(COL_RID, sort=False):
        if len(g) < 3: continue
        rid_s = str(int(rid)) if isinstance(rid, (int, np.integer)) else str(rid)
        if rid_s not in payouts: continue
        po = payouts[rid_s]
        g = g.sort_values(COL_BAN).reset_index(drop=True)
        ban = g[COL_BAN].astype(int).values
        b2i = {int(b): i for i, b in enumerate(ban)}
        if any(b2i.get(po[k]) is None for k in ("win", "plc", "sho")): continue

        w = PL.pl_weights(g["_score"].values)
        n = len(w)
        axis_idx = int(np.argmax(w))  # AI top-1
        combos = race_combos(w, po, b2i, cal, curves, rid_s=rid_s, wide_map=wide_map)
        n_race += 1

        # 軸流し用の bucket mask (bet ごと)
        iu, ju = np.triu_indices(n, k=1)
        # 馬単 (order-aware): 全 i!=j
        idx_all = np.arange(n)
        I, J = np.meshgrid(idx_all, idx_all, indexing="ij")
        m_umatan = (I != J).flatten()
        ut_I = I.flatten()[m_umatan]; ut_J = J.flatten()[m_umatan]

        for (bet, flow, K, ev_min, thr) in keys:
            d = combos[bet]
            ev = d["ev"]; hit = d["hit"]; actual = d["actual"]
            if len(ev) == 0: continue

            # 流し mask: axis を含む combos のみ候補
            if flow:
                if bet == "umaren":
                    flow_mask = (iu == axis_idx) | (ju == axis_idx)
                elif bet == "umatan":
                    # 軸=1着固定のみ (流しの一般的定義)
                    # idx_all 展開: ut_I が axis の行のみ
                    flow_mask = (ut_I == axis_idx)
                else:
                    continue  # 他は flow 定義なし
                cand_idx = np.where(flow_mask & (ev >= ev_min))[0]
            else:
                cand_idx = np.where(ev >= ev_min)[0]

            if len(cand_idx) == 0: continue
            order = np.argsort(-ev[cand_idx])
            top = cand_idx[order[:K]]
            ev_top = ev[top]
            stakes, total_stake = allocate(ev_top, alloc)
            if total_stake <= 0: continue
            race_roi = float((ev_top * stakes / BET).sum()) / total_stake
            if race_roi < thr: continue

            s = stats[(bet, flow, K, ev_min, thr)]
            s["races"] += 1
            s["bets"]  += int((stakes > 0).sum())
            s["stake"] += total_stake
            hit_mask = hit[top] & (stakes > 0)
            s["hits"] += int(hit_mask.sum())
            # 実配当: actual[top] は per-combo 配当 (hit 以外は 0)
            ret = float(((stakes / BET) * actual[top] * hit[top]).sum())
            s["ret"]  += ret

        if n_race % 1000 == 0:
            print(f"  ..{n_race} races")

    return stats, n_race


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v4",
                    help="model tag: v1/v3/v4/v5 (models/unified_rank_{tag}.pkl 等)")
    ap.add_argument("--alloc", default=DEFAULT_ALLOC, choices=["equal", "kelly"],
                    help="stake allocation: equal (K本等分) or kelly (edge比率,1点集中)")
    args = ap.parse_args()

    tag = args.model
    alloc = args.alloc
    print(f"[alloc] {alloc}")
    be.MODEL_PKL = BASE / f"models/unified_rank_{tag}.pkl"
    be.CAL_PKL   = BASE / f"models/pl_calibrators_{tag}.pkl"
    be.CURVE_PKL = BASE / f"data/pl_payout_curve_{tag}.pkl"
    assert be.MODEL_PKL.exists(), f"{be.MODEL_PKL} not found"
    print(f"[model] tag={tag}")
    print(f"  {be.MODEL_PKL}")
    print(f"  {be.CAL_PKL}")
    print(f"  {be.CURVE_PKL}")

    payouts = load_payouts()
    wide_map = load_wide_payouts()
    print(f"[payouts] races={len(payouts):,}  wide races={len(wide_map):,}")

    te = score_test()
    sub = te[te["year"].isin([2024, 2025])]
    print(f"[test 2024+2025] rows={len(sub):,}  races={sub[COL_RID].nunique():,}")

    stats, n_race = simulate(sub, payouts, wide_map, alloc)

    rows = []
    for (bet, flow, K, ev_min, thr), s in stats.items():
        if s["bets"] < 10: continue
        hr  = s["hits"] / s["bets"]
        roi = s["ret"] / s["stake"] if s["stake"] > 0 else 0
        rows.append({
            "bet": bet, "flow": flow, "K": K, "ev_min": ev_min, "thr": thr,
            "races": s["races"], "bets": s["bets"], "hits": s["hits"],
            "stake": s["stake"], "ret": int(s["ret"]),
            "roi": roi, "hit_rate": hr,
        })

    matched = [r for r in rows
               if HIT_LO <= r["hit_rate"] <= HIT_HI
               and r["roi"] >= ROI_MIN
               and r["races"] >= MIN_RACES]
    matched.sort(key=lambda r: (-r["roi"], -r["races"]))

    print(f"\n=== hit {HIT_LO*100:.0f}-{HIT_HI*100:.0f}% & ROI≥{ROI_MIN*100:.0f}% & races≥{MIN_RACES} : {len(matched)} configs ===")
    hdr = f"  {'bet':10s} {'flow':>5s} {'K':>2s} {'EV':>4s} {'thr':>5s} {'R':>5s} {'bets':>6s} {'hit%':>6s} {'stake':>10s} {'ret':>10s} {'ROI%':>6s}"
    print(hdr)
    for r in matched[:60]:
        print(f"  {r['bet']:10s} {str(r['flow']):>5s} {r['K']:>2d} {r['ev_min']:>4d} {r['thr']:>5.2f} "
              f"{r['races']:>5,} {r['bets']:>6,} {r['hit_rate']*100:>5.2f} "
              f"{r['stake']:>10,} {r['ret']:>10,} {r['roi']*100:>6.2f}")

    # 参考: races 多い順 top 30 (any hit/roi)
    rows.sort(key=lambda r: -r["races"])
    print(f"\n--- top 30 by race count (reference) ---")
    print(hdr)
    for r in rows[:30]:
        print(f"  {r['bet']:10s} {str(r['flow']):>5s} {r['K']:>2d} {r['ev_min']:>4d} {r['thr']:>5.2f} "
              f"{r['races']:>5,} {r['bets']:>6,} {r['hit_rate']*100:>5.2f} "
              f"{r['stake']:>10,} {r['ret']:>10,} {r['roi']*100:>6.2f}")

    out_dir = BASE / "reports"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"sweep_target_{tag}_{alloc}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_tag": tag, "alloc": alloc,
            "filter": {"hit_lo": HIT_LO, "hit_hi": HIT_HI, "roi_min": ROI_MIN, "min_races": MIN_RACES},
            "race_budget": RACE_BUDGET, "kelly_frac": KELLY_FRAC, "unit": UNIT,
            "n_race": n_race,
            "matched": matched,
            "all_rows": rows,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {out_path}")
    print(f"\nGATE 1 判定: {'✅ 合格設定あり' if matched else '❌ 全滅 → Stage 2 へ'}")


if __name__ == "__main__":
    main()
