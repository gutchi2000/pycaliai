# -*- coding: utf-8 -*-
"""
analysis/measure_settle_drift.py — T-10→確定 決済ドリフトの実測 (勝者条件付き)
================================================================================
決済層: 単勝EVの期待払戻 = p_win × E[確定オッズ|勝ち]。T-10オッズで EV を組むと
勝者のオッズ縮み(steam)ぶん系統的に過大計上になる。その実寸をバンド別に測る。

データ: reports/live_odds/*.json (T-10実スナップショット) × data/kekka/{date}.csv (確定配当)
勝者のみで測るのが正しい(外れは払戻0で決済誤差が発生しないため)。

実行: python -m analysis.measure_settle_drift
出力: reports/settle_drift.json (compute_bets の SETTLE_DRIFT_TAN 更新の根拠)
t10ログが増えるたび再実行して係数を更新する。
"""
from __future__ import annotations
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
BANDS = [("1.0-2", 1.0, 2.0), ("2-4", 2.0, 4.0), ("4-8", 4.0, 8.0),
         ("8-20", 8.0, 20.0), ("20+", 20.0, 9e9)]


def main():
    rows, kcache = [], {}
    for f in glob.glob(str(BASE / "reports/live_odds/*.json")):
        j = json.load(open(f, encoding="utf-8"))
        if not j.get("ok"):
            continue
        rid = str(j["race_id"])[:16]
        date = rid[:8]
        if date not in kcache:
            p = BASE / "data" / "kekka" / f"{date}.csv"
            kcache[date] = pd.read_csv(p, encoding="cp932", low_memory=False) if p.exists() else None
        k = kcache[date]
        if k is None:
            continue
        kk = k[k["レースID(新)"].astype(str).str[:16] == rid]
        w = kk[pd.to_numeric(kk["確定着順"], errors="coerce") == 1]
        if len(w) == 0:
            continue
        wban = str(int(w.iloc[0]["馬番"]))
        fin = float(w.iloc[0]["単勝配当"]) / 100
        tan = j.get("tansho") or {}
        t10 = tan.get(wban)
        if not t10 or float(t10) <= 1.0:
            continue
        rows.append({"rid": rid, "date": date, "t10": float(t10), "fin": fin,
                     "drift": float(np.log(fin / float(t10)))})
    d = pd.DataFrame(rows)
    rng = np.random.default_rng(42)
    n = len(d)
    boot = [d["drift"].values[rng.integers(0, n, n)].mean() for _ in range(3000)]
    out = {"n": n, "period": [d["date"].min(), d["date"].max()],
           "mean_mult": round(float(np.exp(d["drift"].mean())), 4),
           "mult_ci95": [round(float(np.exp(np.percentile(boot, 2.5))), 4),
                         round(float(np.exp(np.percentile(boot, 97.5))), 4)],
           "shrink_rate": round(float((d["drift"] < 0).mean()), 3),
           "bands": {}}
    print(f"勝者 T-10→確定 (n={n}, {out['period'][0]}〜{out['period'][1]}): "
          f"×{out['mean_mult']} CI{out['mult_ci95']} 縮む率{out['shrink_rate']*100:.0f}%")
    for lab, lo, hi in BANDS:
        g = d[(d["t10"] >= lo) & (d["t10"] < hi)]
        if len(g) < 3:
            continue
        out["bands"][lab] = {"n": int(len(g)),
                             "mult": round(float(np.exp(g["drift"].mean())), 4)}
        print(f"  T-10 {lab}倍: n={len(g):4d} ×{out['bands'][lab]['mult']}")
    op = BASE / "reports" / "settle_drift.json"
    json.dump(out, open(op, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[saved] {op}")


if __name__ == "__main__":
    main()
