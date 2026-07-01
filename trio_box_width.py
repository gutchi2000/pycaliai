"""
trio_box_width.py — 三連複ボックス top-k の捕捉率/コスト/ROI を valid/test で
問い: 印(◎〇▲=top3)に縛られず網を広げると三連複は捕まるのか? 経済的に得か?
着順1-2-3の馬がモデル top-k に全部入る=捕捉. cost=C(k,3)*100. 払戻=sanrenpuku.
"""
from __future__ import annotations
import json
from collections import defaultdict
from math import comb
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import pl_probs as PL
from backtest_pl_ev import load_payouts

BASE = Path(__file__).parent
MASTER = BASE / "data/master_v2_20130105-20251228.csv"
OUT = BASE / "reports/trio_box_width.json"
COL_RID = "レースID(新/馬番無)"
COL_JYUN = "着順"
COL_BAN = "馬番"
import re
def rid16(x): return re.sub(r"\D", "", str(x))[:16]

KS = [3, 4, 5, 6, 7, 8]


def main():
    print("[load] model + payouts")
    m = joblib.load(BASE / "models/unified_rank_v6.pkl")
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
    df["_score"] = model.predict(df[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values)

    # agg[split][k] = {stake, ret, cap, n}
    agg = {s: {k: {"stake": 0.0, "ret": 0.0, "cap": 0, "n": 0} for k in KS}
           for s in ("valid", "test")}
    depth = {s: defaultdict(int) for s in ("valid", "test")}  # max_placer_rank 分布
    nr = defaultdict(int)

    for rid, g in df.groupby(COL_RID, sort=False):
        if len(g) < 6:
            continue
        sp = g["split"].iloc[0]
        if sp not in ("valid", "test"):
            continue
        po = payouts.get(rid16(rid))
        if not po:
            continue
        tri_pay = po.get("sanrenpuku")
        if tri_pay is None or pd.isna(tri_pay) or tri_pay <= 0:
            continue
        g = g.sort_values("_score", ascending=False).reset_index(drop=True)
        chaku = pd.to_numeric(g[COL_JYUN], errors="coerce").values
        placer_ranks = [i for i in range(len(g)) if chaku[i] in (1, 2, 3)]
        if len(placer_ranks) < 3:
            continue
        mx = max(placer_ranks[:3]) if len(placer_ranks) >= 3 else 99
        # 正確には着順1,2,3の3頭の rank の最大
        ranks123 = sorted([i for i in range(len(g)) if chaku[i] in (1, 2, 3)])[:3]
        mx = max(ranks123)
        nr[sp] += 1
        depth[sp][min(mx, 12)] += 1
        for k in KS:
            cap = mx < k
            cost = comb(k, 3) * 100.0
            a = agg[sp][k]
            a["stake"] += cost
            a["ret"] += float(tri_pay) if cap else 0.0
            a["cap"] += int(cap)
            a["n"] += 1

    def roi(a):
        return round(a["ret"] / a["stake"] * 100, 1) if a["stake"] else 0.0

    res = {"n_races": dict(nr), "by_k": {}, "depth_pct": {}}
    for sp in ("valid", "test"):
        res["by_k"][sp] = {k: {"capture%": round(agg[sp][k]["cap"] / agg[sp][k]["n"] * 100, 1),
                               "combos": comb(k, 3), "roi%": roi(agg[sp][k]),
                               "n": agg[sp][k]["n"]} for k in KS}
        tot = nr[sp]
        res["depth_pct"][sp] = {str(d): round(depth[sp][d] / tot * 100, 1) for d in sorted(depth[sp])}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n{'='*72}\n三連複ボックス top-k  (valid23 / test24-25)\n{'='*72}")
    print(f"races: {dict(nr)}")
    print(f"\n{'k(網)':<6}{'点数':>5}{'捕捉% V/T':>16}{'ROI% V/T':>16}")
    for k in KS:
        v, t = res["by_k"]["valid"][k], res["by_k"]["test"][k]
        tag = "  ◎〇▲" if k == 3 else ""
        cap = f"{v['capture%']}/{t['capture%']}"
        r = f"{v['roi%']}/{t['roi%']}"
        print(f"  top{k:<3}{comb(k,3):>5}{cap:>16}{r:>16}{tag}")
    print("\n着順1-2-3の3頭が収まる最深rank の分布(%):")
    for sp in ("valid", "test"):
        print(f"  [{sp}] " + "  ".join(f"≤rank{d}:{res['depth_pct'][sp].get(str(d),0)}" for d in range(2, 9)))
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
