"""
probe_temperature_v5.py
=======================
v5 model + v5 cal の状態で、PL 温度スケール T を変えて ECE を測定。
T=1.0 が現状値。T を [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.8, 2.0] で探索し
ECE_tansho_hon / ECE_fuku_hon / ECE_umaren_hono の改善を確認する。

使い方:
    python probe_temperature_v5.py

出力: stdout に T ごとの ECE 表
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import pl_probs as PL

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = Path(__file__).parent
MASTER_CSV = BASE / "data/master_v2_20130105-20251228.csv"
MODEL_PKL  = BASE / "models/unified_rank_v5.pkl"
CAL_PKL    = BASE / "models/pl_calibrators_v5.pkl"

COL_RID  = "レースID(新/馬番無)"
COL_JYUN = "着順"


def ece(p, y, bins=10):
    """Expected Calibration Error (10 bin equal-width)."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    e = 0.0
    n = len(p)
    if n == 0:
        return 0.0
    for i in range(bins):
        if i == bins - 1:
            m = (p >= edges[i]) & (p <= edges[i + 1])
        else:
            m = (p >= edges[i]) & (p < edges[i + 1])
        if m.sum() == 0:
            continue
        e += (m.sum() / n) * abs(p[m].mean() - y[m].mean())
    return float(e)


def encode(df, encs):
    df = df.copy()
    for c, le in encs.items():
        if c not in df.columns:
            continue
        v = df[c].astype(str).fillna("__NaN__")
        v = v.where(v.isin(set(le.classes_)), "__NaN__")
        df[c] = le.transform(v)
    return df


def main():
    print("=" * 70)
    print("probe_temperature_v5.py  (T sweep on valid=2023)")
    print("=" * 70)

    bundle = joblib.load(MODEL_PKL)
    model, feats, encs = bundle["model"], bundle["feature_cols"], bundle["encoders"]
    cal = joblib.load(CAL_PKL)["calibrators"]

    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"])

    # valid=2023 で T を探索
    vl = df[df["split"] == "valid"].copy()
    vl = encode(vl, encs)
    X = vl[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    vl["_score"] = model.predict(X)
    print(f"  valid races: {vl[COL_RID].nunique():,}  rows={len(vl):,}")

    # 真OOS=2024-2025 で確認
    oos = df[df["split"] == "test"].copy()
    oos = encode(oos, encs)
    X2 = oos[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    oos["_score"] = model.predict(X2)
    print(f"  OOS races:   {oos[COL_RID].nunique():,}  rows={len(oos):,}")

    Ts = [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.8, 2.0]
    print()
    print(f"{'set':<6} {'T':>5}  {'ECE_tan(◎)':>11}  {'ECE_fuku(◎)':>12}  {'ECE_uma(◎-〇)':>14}")
    print("-" * 70)

    for tag, src in [("valid", vl), ("OOS", oos)]:
        for T in Ts:
            tan_p, tan_y = [], []
            fuku_p, fuku_y = [], []
            uma_p, uma_y = [], []
            for rid, g in src.groupby(COL_RID, sort=False):
                if len(g) < 3:
                    continue
                jyun = g[COL_JYUN].astype(int).values
                scores = g["_score"].values / T
                w = PL.pl_weights(scores)
                ai_rank = np.argsort(-w)  # AI 上位
                hon, tai = ai_rank[0], ai_rank[1]
                # 単勝(◎)
                p1 = cal["tansho"].predict([PL.p_tansho(w, hon)])[0]
                tan_p.append(p1); tan_y.append(1 if jyun[hon] == 1 else 0)
                # 複勝(◎)
                p2 = cal["fukusho"].predict([PL.p_fukusho(w, hon)])[0]
                fuku_p.append(p2); fuku_y.append(1 if jyun[hon] <= 3 else 0)
                # 馬連(◎-〇)
                p3 = cal["umaren"].predict([PL.p_umaren(w, hon, tai)])[0]
                uma_p.append(p3); uma_y.append(1 if {jyun[hon], jyun[tai]} == {1, 2} else 0)

            e1 = ece(tan_p, tan_y)
            e2 = ece(fuku_p, fuku_y)
            e3 = ece(uma_p, uma_y)
            mark = "  *" if T == 1.0 else ""
            print(f"{tag:<6} {T:>5.2f}  {e1:>11.4f}  {e2:>12.4f}  {e3:>14.4f}{mark}")
        print()

    # v4 との比較参考
    print("[reference] v4 真OOS:")
    print("  ECE_tan(◎)=0.0396  ECE_fuku(◎)=0.0222  ECE_uma(◎-〇)=0.0255")
    print("[current ] v5 真OOS T=1.00:")
    print("  ECE_tan(◎)=0.0725  ECE_fuku(◎)=0.0427  ECE_uma(◎-〇)=0.0410")


if __name__ == "__main__":
    main()
