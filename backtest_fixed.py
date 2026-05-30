"""
backtest_fixed.py
=================
固定フォーメーション backtest (EV/閾値なし).

毎レース: 馬連 ◎〇軸-◎〇▲△₁△₂ 7 点流し × ¥100 = ¥700 / race
  買い目: ◎-〇, ◎-▲, ◎-△₁, ◎-△₂, 〇-▲, 〇-△₁, 〇-△₂

的中条件: (1着, 2着) のペアが 7 点の中にある
  ⇔ {1着,2着} ⊂ top5 かつ {1着,2着} ∩ top2 ≠ ∅
OOS = valid(2023) + test(2024+2025) の 3 年表示. 真 OOS は 2024-2025.
"""
from __future__ import annotations
import argparse
import json
import warnings
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backtest_pl_ev import (
    load_payouts, score_test,
    COL_RID, COL_BAN, BET,
)
import backtest_pl_ev as be

warnings.filterwarnings("ignore")
BASE = Path(__file__).parent

UNIT = 100
TOP_AXIS = 2    # ◎〇
TOP_OPP  = 5    # ◎〇▲△△ (軸含む上位 5 頭)
N_POINTS = 7    # 2 軸 × (5-2) 残り相手 + (◎-〇) = 2*3 + 1 = 7


def simulate(te, payouts):
    stats = {"races": 0, "bets": 0, "hits": 0, "stake": 0, "ret": 0.0}
    n_skip = 0

    for rid, g in te.groupby(COL_RID, sort=False):
        if len(g) < TOP_OPP:
            n_skip += 1; continue
        rid_s = str(int(rid)) if isinstance(rid, (int, np.integer)) else str(rid)
        if rid_s not in payouts:
            n_skip += 1; continue
        po = payouts[rid_s]
        g = g.sort_values(COL_BAN).reset_index(drop=True)
        ban = g[COL_BAN].astype(int).values
        b2i = {int(b): i for i, b in enumerate(ban)}
        wi = b2i.get(po["win"]); pi_ = b2i.get(po["plc"])
        if wi is None or pi_ is None:
            n_skip += 1; continue

        order = np.argsort(-g["_score"].values)
        top5 = set(int(x) for x in order[:TOP_OPP])
        top2 = set(int(x) for x in order[:TOP_AXIS])
        # hit = 1着 and 2着 が top5 に入り、かつ少なくとも片方が top2 (軸) にいる
        actual_pair = {wi, pi_}
        hit = actual_pair.issubset(top5) and bool(actual_pair & top2)
        umaren_pay = float(po["umaren"]) if po["umaren"] and not pd.isna(po["umaren"]) else 0.0

        stats["races"] += 1
        stats["bets"]  += N_POINTS
        stats["stake"] += N_POINTS * UNIT
        if hit:
            stats["hits"] += 1
            stats["ret"]  += umaren_pay  # UNIT=100 → pay は per-100 基準で 1:1

    return stats, n_skip


def summarize(stats, label):
    R = stats["races"]
    hr = stats["hits"] / max(R, 1)
    roi = stats["ret"] / stats["stake"] if stats["stake"] > 0 else 0
    per_race = stats["ret"] / max(R, 1)
    print(f"\n=== {label} ===")
    print(f"  races     : {R:,}")
    print(f"  stake     : {stats['stake']:>12,} 円  ({stats['stake']/R:>6.0f} 円/race)")
    print(f"  return    : {int(stats['ret']):>12,} 円  ({per_race:>6.0f} 円/race)")
    print(f"  races hit : {stats['hits']:,}   hit_rate = {hr*100:.2f}%")
    print(f"  ROI       : {roi*100:.2f}%   (P&L = {int(stats['ret'] - stats['stake']):+,} 円)")
    return {"races": R, "bets": stats["bets"], "hits": stats["hits"],
            "stake": stats["stake"], "ret": int(stats["ret"]),
            "roi": round(roi, 4), "hit_rate": round(hr, 4),
            "return_per_race": round(per_race, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v4", help="model tag: v1/v3/v4/v5")
    args = ap.parse_args()

    tag = args.model
    be.MODEL_PKL = BASE / f"models/unified_rank_{tag}.pkl"
    be.CAL_PKL   = BASE / f"models/pl_calibrators_{tag}.pkl"
    be.CURVE_PKL = BASE / f"data/pl_payout_curve_{tag}.pkl"
    assert be.MODEL_PKL.exists(), f"{be.MODEL_PKL} not found"
    print(f"[model] tag={tag}  {be.MODEL_PKL}")
    print(f"[formation] 馬連 ◎〇軸-◎〇▲△△ 7 点流し × ¥{UNIT} = ¥{N_POINTS*UNIT}/race")

    payouts = load_payouts()
    print(f"[payouts] races={len(payouts):,}")

    te = score_test(include_valid=True)
    print(f"[test+valid] rows={len(te):,}  races={te[COL_RID].nunique():,}")
    print("  ※ 2023 は valid 年 (Optuna ハイパラ選定対象) — 参考値扱い")

    out_all = {}
    for label, mask in [
        ("2023 (valid, 参考)", te["year"] == 2023),
        ("2024", te["year"] == 2024),
        ("2025", te["year"] == 2025),
        ("3 years (2023-2025)", te["year"].isin([2023, 2024, 2025])),
        ("真 OOS (2024-2025)", te["year"].isin([2024, 2025])),
    ]:
        sub = te[mask]
        stats, n_skip = simulate(sub, payouts)
        if stats["races"] == 0:
            print(f"\n=== {label} === (no races)")
            continue
        out_all[label] = summarize(stats, label)

    out_dir = BASE / "reports"; out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"backtest_fixed_{tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_tag": tag, "unit": UNIT,
            "top_axis": TOP_AXIS, "top_opp": TOP_OPP, "n_points": N_POINTS,
            "formation": "馬連 ◎〇軸-◎〇▲△△ 7 点流し",
            "results": out_all,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
