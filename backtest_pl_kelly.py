"""
backtest_pl_kelly.py
====================
Kelly-weighted formation backtest.
各レース:
  1. 全 combo を EV = p_cal × median_pay で計算
  2. EV ≥ EV_MIN_BET の combos のみ候補
  3. 候補のうち top K で formation
  4. 1 レース総予算 RACE_BUDGET を fractional Kelly で配分
  5. race-ROI = Σ EV / RACE_BUDGET ≥ RACE_THR ならベット
"""
from __future__ import annotations

import io
import json
import sys
import warnings
from itertools import combinations, permutations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import pl_probs as PL
from backtest_pl_ev import (
    load_payouts, apply_encoders, lookup_pay_vec, score_test,
    all_umaren_mat, all_umatan_mat, all_fukusho_vec_fast, all_sanrenpuku_tensor,
    BASE, MASTER_CSV, MODEL_PKL, CAL_PKL, CURVE_PKL,
    COL_RID, COL_JYUN, COL_BAN, BET,
)
from backtest_pl_formation import race_combos

warnings.filterwarnings("ignore")

import os
OUT_JSON = BASE / os.environ.get("KELLY_OUT", "reports/backtest_pl_kelly.json")

RACE_BUDGET = 10000   # 1 レース 1 万円 (plan 準拠)
KELLY_FRAC  = 0.25    # fractional Kelly
UNIT        = 100     # ベット最小単位

# 馬券別 K_MAX (上限), EV_MIN_BET (買う EV 閾値), RACE_THR (レース全体 ROI 閾値)
CONFIGS = [
    # (bet_type, K_max, ev_min_bet, race_thresholds)
    ("tansho",     2,   100, [1.00, 1.05, 1.10]),
    ("fukusho",    3,    95, [1.00, 1.05, 1.10]),
    ("umaren",     5,   100, [1.00, 1.05, 1.10, 1.20]),
    ("umatan",     8,   100, [1.00, 1.05, 1.10, 1.20]),
    ("sanrenpuku",12,   100, [1.00, 1.05, 1.10, 1.20]),
    ("sanrentan", 30,   100, [1.00, 1.05, 1.10, 1.20, 1.30]),
]


def kelly_weights(p, pay, stake_total, frac=0.25, unit=100):
    """fractional Kelly → (stake_per_combo, actual_total_stake).
    p: 推定的中確率配列, pay: 推定配当配列, stake_total: 総予算上限.
    Kelly fraction: f_i = max(0, (p*b - (1-p)) / b),  b = pay/100 - 1.
    全 combo 和を正規化して UNIT 単位に丸めて stake_total 以下になるよう調整."""
    b = pay / 100.0 - 1.0
    f_star = np.where(b > 0, (p * (b + 1) - 1) / b, 0.0)
    f_star = np.clip(f_star * frac, 0, None)
    if f_star.sum() <= 0:
        return np.zeros_like(p), 0
    # 相対比率 → stake_total で正規化
    ratios = f_star / f_star.sum()
    raw = ratios * stake_total
    # UNIT 単位に丸め
    stakes = np.floor(raw / unit).astype(int) * unit
    # 最低 1 点に 0 は入れない — 0 の combo は買わない
    actual = int(stakes.sum())
    return stakes, actual


def simulate(te, payouts, cal, curves):
    results = []  # 1行/レース/config/threshold
    stats = {}   # (bet, K, ev_min, race_thr) → summary
    for bet, K, ev_min, thrs in CONFIGS:
        for t in thrs:
            stats[(bet, K, ev_min, t)] = {
                "races_bet": 0, "bets": 0, "hits": 0,
                "stake": 0, "ret": 0.0, "n_race_eval": 0,
            }

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

        for bet, K, ev_min, thrs in CONFIGS:
            d = combos[bet]
            ev = d["ev"]; hit = d["hit"]; actual = d["actual"]
            if len(ev) == 0: continue
            # EV ≥ ev_min な combos だけ候補
            cand_idx = np.where(ev >= ev_min)[0]
            if len(cand_idx) == 0:
                continue
            # top K (EV 降順)
            order_in_cand = np.argsort(-ev[cand_idx])
            top = cand_idx[order_in_cand[:K]]
            ev_top = ev[top]
            p_top  = ev_top / np.maximum(lookup_pay_vec(curves[bet], np.array([1])), 1)  # dummy
            # 実際の p_cal はここで再取得したい → ev = p * pay なので pay 推定
            pay_top = lookup_pay_vec(curves[bet], ev_top / ev_top.sum())  # proxy
            # shortcut: p = ev/pay.  pay = curves[bet] lookup by some p. 再計算は重いので ev,pay 保存必要
            # → race_combos は p,pay を返さない. ここで pay を curves 再 lookup
            # p_cal は hit prob. ev = p * pay_est. → curves bet の median から逆算は不可能 (関数ではない)
            # 簡易化: Kelly は ev/bet を proxy edge として使用
            # b_proxy = (ev_top - BET)/BET  edge
            edge = (ev_top - BET) / BET
            edge = np.clip(edge, 0, None)
            if edge.sum() == 0:
                continue
            ratios = edge / edge.sum()
            stakes = np.floor(ratios * RACE_BUDGET * KELLY_FRAC / UNIT).astype(int) * UNIT
            if stakes.sum() == 0:
                # force 1 point on top combo
                stakes = np.zeros_like(edge, dtype=int)
                stakes[0] = UNIT
            total_stake = int(stakes.sum())

            # race-ROI = Σ EV / total_stake × 100 (単位 100円)
            exp_return = float((ev_top * stakes / BET).sum())
            race_roi = exp_return / total_stake if total_stake > 0 else 0

            for t in thrs:
                key = (bet, K, ev_min, t)
                s = stats[key]
                s["n_race_eval"] += 1
                if race_roi >= t:
                    s["races_bet"] += 1
                    s["bets"] += int((stakes > 0).sum())
                    s["stake"] += total_stake
                    # 実 hit / ret
                    hit_mask = hit[top] & (stakes > 0)
                    s["hits"] += int(hit_mask.sum())
                    # 実配当 = stake / BET * actual_pay_per_100
                    ret = float(((stakes / BET) * actual[top] * hit[top]).sum())
                    s["ret"] += ret

        if n_race % 500 == 0:
            print(f"  ..{n_race} races")

    return stats, n_race


def summarize(stats, n_race, label):
    print(f"\n=== {label} ({n_race} races) ===")
    rows = []
    for (bet, K, ev_min, t), d in stats.items():
        if d["stake"] == 0: continue
        roi = d["ret"] / d["stake"]
        hr  = d["hits"] / max(d["bets"], 1)
        rows.append({
            "bet": bet, "K_max": K, "ev_min": ev_min, "race_thr": t,
            "races_bet": d["races_bet"], "bets": d["bets"], "hits": d["hits"],
            "stake": d["stake"], "ret": int(d["ret"]),
            "roi": round(roi, 4), "hit_rate": round(hr, 4),
            "stake_per_race": round(d["stake"] / max(d["races_bet"], 1)),
        })
    rows.sort(key=lambda r: (-r["roi"], -r["races_bet"]))
    print(f"  {'馬券':10s} {'K':>3s} {'EVmin':>5s} {'thr':>5s} {'races':>6s} "
          f"{'bets':>7s} {'hits':>6s} {'hr%':>6s} {'stake/R':>8s} {'stake':>10s} {'ret':>10s} {'ROI%':>7s}")
    for r in rows[:30]:
        print(f"  {r['bet']:10s} {r['K_max']:>3d} {r['ev_min']:>5d} {r['race_thr']:>5.2f} "
              f"{r['races_bet']:>6,} {r['bets']:>7,} {r['hits']:>6,} {r['hit_rate']*100:>5.2f} "
              f"{r['stake_per_race']:>8,} {r['stake']:>10,} {r['ret']:>10,} {r['roi']*100:>7.2f}")
    return rows


def main():
    print("=" * 75)
    print("backtest_pl_kelly.py  (Kelly-weighted formation)")
    print("=" * 75)
    payouts = load_payouts()
    cal = joblib.load(CAL_PKL)["calibrators"]
    curves = joblib.load(CURVE_PKL)["curves"]
    te = score_test()

    all_results = {}
    for label, mask in [
        ("2024", te["year"] == 2024),
        ("2025", te["year"] == 2025),
        ("2024+2025", te["year"].isin([2024, 2025])),
    ]:
        print(f"\n>>>  {label}  <<<")
        sub = te[mask]
        stats, n_race = simulate(sub, payouts, cal, curves)
        rows = summarize(stats, n_race, label)
        all_results[label] = rows

    OUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "configs":    [{"bet": b, "K_max": K, "ev_min": ev, "race_thr_list": t}
                           for b, K, ev, t in CONFIGS],
            "race_budget": RACE_BUDGET,
            "kelly_frac":  KELLY_FRAC,
            "unit":        UNIT,
            "results":     all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {OUT_JSON}")


if __name__ == "__main__":
    main()
