"""
backtest_pl_wide.py
===================
ワイド専用 backtest. odds_test.csv 由来の wide_payouts_2016-2025.parquet を使う。
EV ゲート スイープ + K フォーメーション + race-ROI 閾値。

目的: ワイドが勝てる馬券種か (ROI ≥ 100%) を OOS 2024-2025 で検証。
"""
from __future__ import annotations
import io, sys, json, warnings
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import pl_probs as PL
from backtest_pl_ev import (
    score_test, lookup_pay_vec, all_umaren_mat,
    BASE, CAL_PKL, CURVE_PKL, COL_RID, COL_BAN, BET,
)

warnings.filterwarnings("ignore")

WIDE_PQ = BASE / "data/wide_payouts_2016-2025.parquet"
OUT_JSON = BASE / "reports/backtest_pl_wide.json"

EV_THRESHOLDS = [80, 90, 100, 105, 110, 120, 130, 150]
K_LIST = [1, 2, 3, 4, 5, 6]
RACE_THR = [1.00, 1.05, 1.10, 1.20]


def load_wide_payouts():
    df = pd.read_parquet(WIDE_PQ)
    df["race_id"] = df["race_id"].astype(str)
    out = {}
    for _, r in df.iterrows():
        combos = {}
        for k in (1, 2, 3):
            i, j, pay = r[f"w{k}_i"], r[f"w{k}_j"], r[f"w{k}_pay"]
            if pd.notna(i) and pd.notna(j) and pd.notna(pay):
                key = tuple(sorted((int(i), int(j))))
                combos[key] = float(pay)
        if combos:
            out[r["race_id"]] = combos
    return out


def main():
    print("=" * 70)
    print("backtest_pl_wide.py  (test=2024-2025)")
    print("=" * 70)
    wide_pay = load_wide_payouts()
    print(f"[wide_payouts] {len(wide_pay):,} races")
    cal = joblib.load(CAL_PKL)["calibrators"]["wide"]
    curve = joblib.load(CURVE_PKL)["curves"]["wide"]
    te = score_test()

    # === simple EV gate ===
    ev_stats = {t: {"bets": 0, "hits": 0, "stake": 0, "ret": 0.0} for t in EV_THRESHOLDS}

    # === formation (top K + race_thr) ===
    fmt_stats = {(K, th): {"races": 0, "bets": 0, "hits": 0, "stake": 0, "ret": 0.0}
                 for K in K_LIST for th in RACE_THR}

    n_race = 0
    for label, mask in [("2024", te["year"] == 2024),
                        ("2025", te["year"] == 2025),
                        ("2024+2025", te["year"].isin([2024, 2025]))]:
        print(f"\n>>> {label} <<<")
        ev_stats = {t: {"bets": 0, "hits": 0, "stake": 0, "ret": 0.0} for t in EV_THRESHOLDS}
        fmt_stats = {(K, th): {"races": 0, "bets": 0, "hits": 0, "stake": 0, "ret": 0.0}
                     for K in K_LIST for th in RACE_THR}
        n_race = 0
        sub = te[mask]
        for rid, g in sub.groupby(COL_RID, sort=False):
            if len(g) < 3: continue
            rid_s = str(int(rid)) if isinstance(rid, (int, np.integer)) else str(rid)
            if rid_s not in wide_pay: continue
            g = g.sort_values(COL_BAN).reset_index(drop=True)
            ban = g[COL_BAN].astype(int).values
            combos_pay = wide_pay[rid_s]  # {(ui,uj): pay}
            # 馬番 → index
            b2i = {int(b): i for i, b in enumerate(ban)}
            # payout を (i,j) index ベースに変換
            hit_pay = {}
            for (ui, uj), p in combos_pay.items():
                ii = b2i.get(ui); jj = b2i.get(uj)
                if ii is None or jj is None: continue
                hit_pay[tuple(sorted((ii, jj)))] = p
            if not hit_pay: continue

            w = PL.pl_weights(g["_score"].values)
            n = len(w)
            # p_wide 全組合わせ (本当は p_wide = Σ(i,j が top3 内) だが all_umaren_mat 行使せず直接)
            # PL.all_wide があれば使う; なければ分解実装
            # pl_probs.py の p_wide は P(both in top3)
            iu, ju = np.triu_indices(n, k=1)
            p_raw = np.array([PL.p_wide(w, int(iu[k]), int(ju[k])) for k in range(len(iu))])
            p_c = cal.predict(p_raw)
            pay = lookup_pay_vec(curve, p_c)
            ev = p_c * pay

            # hit array
            hit_mask = np.zeros(len(iu), dtype=bool)
            actual = np.zeros(len(iu))
            for k in range(len(iu)):
                key = (int(iu[k]), int(ju[k]))
                if key in hit_pay:
                    hit_mask[k] = True
                    actual[k] = hit_pay[key]

            n_race += 1

            # EV gate
            for t in EV_THRESHOLDS:
                gate = ev >= t
                s = ev_stats[t]
                nb = int(gate.sum())
                s["bets"] += nb; s["stake"] += nb * BET
                s["hits"] += int((gate & hit_mask).sum())
                s["ret"]  += float((actual * (gate & hit_mask)).sum())

            # formation
            order = np.argsort(-ev)
            for K in K_LIST:
                top = order[:min(K, len(order))]
                ev_top = ev[top]
                race_roi = ev_top.mean() / BET
                for th in RACE_THR:
                    key = (K, th)
                    s = fmt_stats[key]
                    if race_roi >= th:
                        s["races"] += 1
                        s["bets"] += len(top)
                        s["stake"] += len(top) * BET
                        s["hits"] += int(hit_mask[top].sum())
                        s["ret"]  += float(actual[top].sum())

        print(f"  races evaluated: {n_race:,}")
        print(f"\n  [EV gate]")
        print(f"  {'EVmin':>6s} {'bets':>7s} {'hits':>6s} {'hr%':>6s} {'stake':>10s} {'ret':>10s} {'ROI%':>7s}")
        for t in EV_THRESHOLDS:
            d = ev_stats[t]
            if d["stake"] == 0:
                print(f"  {t:>6d}  (no bets)"); continue
            roi = d["ret"]/d["stake"]*100
            hr = d["hits"]/max(d["bets"],1)*100
            print(f"  {t:>6d} {d['bets']:>7,} {d['hits']:>6,} {hr:>5.2f} "
                  f"{d['stake']:>10,} {int(d['ret']):>10,} {roi:>7.2f}")

        print(f"\n  [formation K × race_thr]")
        print(f"  {'K':>3s} {'thr':>5s} {'races':>6s} {'bets':>7s} {'hits':>6s} "
              f"{'hr%':>6s} {'stake':>10s} {'ret':>10s} {'ROI%':>7s}")
        rows = []
        for (K, th), d in fmt_stats.items():
            if d["stake"] == 0: continue
            roi = d["ret"]/d["stake"]
            hr = d["hits"]/max(d["bets"],1)
            rows.append((K, th, d, roi, hr))
        rows.sort(key=lambda r: -r[3])
        for K, th, d, roi, hr in rows[:20]:
            print(f"  {K:>3d} {th:>5.2f} {d['races']:>6,} {d['bets']:>7,} {d['hits']:>6,} "
                  f"{hr*100:>5.2f} {d['stake']:>10,} {int(d['ret']):>10,} {roi*100:>7.2f}")


if __name__ == "__main__":
    main()
