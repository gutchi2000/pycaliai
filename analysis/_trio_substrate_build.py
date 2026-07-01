"""
_trio_substrate_build.py — 三連複 backtest substrate を構築。
mirror of trio_box_width.py の load 経路を使い、1馬1行の long parquet を吐く。

columns (EXACTLY):
  rid, split, year, ban, model_rank, pl_w, score, finish, pop_rank, ou_lastlog, n_horses, tri_pay

sanity:
  - n races valid/test
  - pop_rank coverage %
  - BUY-ALL control: test の全レースで C(n,3) を stake100 ずつ → ROI ~70-80% ばらつき可
"""
from __future__ import annotations
import re
import sys
from math import comb

sys.path.insert(0, "E:/PyCaLiAI")

import joblib
import numpy as np
import pandas as pd

from backtest_pl_ev import load_payouts

MASTER = "E:/PyCaLiAI/data/master_v2_20130105-20251228.csv"
ODDS_OU = "E:/PyCaLiAI/data/odds_ou_feats.parquet"
MODEL_PKL = "E:/PyCaLiAI/models/unified_rank_v6.pkl"
OUT = "E:/PyCaLiAI/analysis/_trio_substrate.parquet"

COL_RID = "レースID(新/馬番無)"
COL_JYUN = "着順"
COL_BAN = "馬番"


def rid16(x):
    return re.sub(r"\D", "", str(x))[:16]


def main():
    print("[load] model + payouts")
    m = joblib.load(MODEL_PKL)
    model, feats, encs = m["model"], m["feature_cols"], m["encoders"]
    payouts = load_payouts()
    print(f"  payout races {len(payouts):,}")

    df = pd.read_csv(MASTER, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"])

    for cc, le in encs.items():
        if cc in df.columns:
            v = df[cc].astype(str).fillna("__NaN__")
            df[cc] = le.transform(v.where(v.isin(set(le.classes_)), "__NaN__"))

    df["_score"] = model.predict(
        df[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    )

    # odds_ou lookup table: (rid16, ban) -> ou_lastlog
    print("[load] odds_ou")
    ou = pd.read_parquet(ODDS_OU)[["rid16", "ban", "ou_lastlog"]].copy()
    ou["rid16"] = ou["rid16"].astype(str)
    ou["ban"] = pd.to_numeric(ou["ban"], errors="coerce")
    ou_map = {}
    for r16, gg in ou.groupby("rid16", sort=False):
        ou_map[r16] = dict(zip(gg["ban"].values, gg["ou_lastlog"].values))

    rows = []
    n_races = {"valid": 0, "test": 0}

    for rid, g in df.groupby(COL_RID, sort=False):
        if len(g) < 6:
            continue
        sp = g["split"].iloc[0]
        if sp not in ("valid", "test"):
            continue
        r16 = rid16(rid)
        po = payouts.get(r16)
        if not po:
            continue
        tri_pay = po.get("sanrenpuku")
        if tri_pay is None or pd.isna(tri_pay) or tri_pay <= 0:
            continue

        g = g.sort_values("_score", ascending=False).reset_index(drop=True)
        n_horses = len(g)
        scores = g["_score"].values.astype(float)
        pl_w = np.exp(scores - scores.max())
        chaku = pd.to_numeric(g[COL_JYUN], errors="coerce").values
        bans = pd.to_numeric(g[COL_BAN], errors="coerce").values
        year = int(r16[:4])
        tri_pay = float(tri_pay)

        # pop_rank: ou_lastlog ascending within race (1 = most popular)
        ou_race = ou_map.get(r16, {})
        ou_vals = np.array([ou_race.get(b, np.nan) for b in bans], dtype=float)
        pop_rank = np.full(n_horses, np.nan)
        valid_mask = ~np.isnan(ou_vals)
        if valid_mask.sum() > 0:
            # rank of ou_lastlog ascending among the valid ones
            order = np.argsort(ou_vals[valid_mask], kind="mergesort")
            ranks = np.empty(valid_mask.sum(), dtype=float)
            ranks[order] = np.arange(1, valid_mask.sum() + 1)
            pop_rank[valid_mask] = ranks

        n_races[sp] += 1
        for i in range(n_horses):
            rows.append({
                "rid": r16,
                "split": sp,
                "year": year,
                "ban": int(bans[i]) if not np.isnan(bans[i]) else -1,
                "model_rank": i + 1,
                "pl_w": float(pl_w[i]),
                "score": float(scores[i]),
                "finish": int(chaku[i]) if not np.isnan(chaku[i]) else -1,
                "pop_rank": float(pop_rank[i]) if not np.isnan(pop_rank[i]) else np.nan,
                "ou_lastlog": float(ou_vals[i]) if not np.isnan(ou_vals[i]) else np.nan,
                "n_horses": n_horses,
                "tri_pay": tri_pay,
            })

    out_df = pd.DataFrame(rows, columns=[
        "rid", "split", "year", "ban", "model_rank", "pl_w", "score",
        "finish", "pop_rank", "ou_lastlog", "n_horses", "tri_pay",
    ])
    out_df.to_parquet(OUT, index=False)
    print(f"[saved] {OUT}  rows={len(out_df):,}")

    # ---- sanity ----
    print(f"\n{'='*60}\nSANITY\n{'='*60}")
    print(f"n races valid={n_races['valid']:,}  test={n_races['test']:,}")
    cov = out_df["pop_rank"].notna().mean() * 100
    print(f"pop_rank coverage: {cov:.1f}%  ({out_df['pop_rank'].notna().sum():,}/{len(out_df):,} rows)")

    # BUY-ALL control on TEST: buy every C(n,3) combo stake 100 each
    # captured if finish 1,2,3 all present (always true) → payout = tri_pay once per race
    test = out_df[out_df["split"] == "test"]
    stake = 0.0
    ret = 0.0
    for rid, g in test.groupby("rid", sort=False):
        n = int(g["n_horses"].iloc[0])
        tp = float(g["tri_pay"].iloc[0])
        n_combos = comb(n, 3)
        stake += n_combos * 100.0
        # buying all combos guarantees the winning trio is covered exactly once
        ret += tp
    roi = ret / stake * 100 if stake else 0.0
    print(f"\nBUY-ALL control (test, all C(n,3) combos @100):")
    print(f"  stake={stake:,.0f}  ret={ret:,.0f}  ROI={roi:.2f}%  (expect ~70-80%)")

    return {
        "n_valid": n_races["valid"],
        "n_test": n_races["test"],
        "cov": cov,
        "buyall_roi": roi,
        "rows": len(out_df),
    }


if __name__ == "__main__":
    main()
