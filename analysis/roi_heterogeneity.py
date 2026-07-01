# -*- coding: utf-8 -*-
"""
roi_heterogeneity.py — 「80%は一律か、振れ幅デカくて80%か」を分散分解で答える
==============================================================================
backtest_ev_grid.py の検証済みスコア/シミュレータを再利用し、各券種について
会場×芝ダ×距離帯セルの ROI 分布を:
  観測スプレッド SD_obs   = セル間 ROI の(レース数重み)標準偏差
  ノイズスプレッド SD_noise = 各セルの cluster-bootstrap SE の二乗平均平方根
  真の異質性     tau      = sqrt(max(0, V_obs - V_noise))   ← これが"本物の振れ幅"
に分解する。tau≈0 なら「真値はほぼ一律80%、見かけの散らばりは全部運」。
出力: reports/roi_heterogeneity.json + 標準出力テーブル
実行: PYTHONUTF8=1 ./venv311/Scripts/python.exe analysis/roi_heterogeneity.py
"""
import sys, json
from pathlib import Path
import numpy as np

BASE = Path(r"E:\PyCaLiAI"); sys.path.insert(0, str(BASE))
import joblib
import backtest_ev_grid as G
import backtest_pl_ev as BT


def cluster_roi_se(pairs, nboot=1000, seed=7):
    """race単位 (stake,ret) から ROI と そのcluster-bootstrap SE。"""
    if not pairs:
        return (np.nan, np.nan)
    st = np.array([p[0] for p in pairs], float)
    rt = np.array([p[1] for p in pairs], float)
    roi = rt.sum() / st.sum() if st.sum() > 0 else np.nan
    rng = np.random.default_rng(seed)
    m = len(pairs); arr = np.empty(nboot)
    for b in range(nboot):
        idx = rng.integers(0, m, m)
        s = st[idx].sum()
        arr[b] = rt[idx].sum() / s if s > 0 else np.nan
    return float(roi), float(np.nanstd(arr))


def main():
    print("[setup] payouts / wide / cal / curves / odds / scores(cache) ...")
    payouts = BT.load_payouts()
    try:
        widepays = BT.load_wide_payouts()
    except Exception as e:
        widepays = {}; print("  wide payouts load failed:", e)
    cal = joblib.load(G.V6_CAL)["calibrators"]
    curves = joblib.load(G.V6_CURVE)["curves"]
    oddsmap = G.load_odds()
    sc = G.score_all()                       # cached parquet
    te = sc[sc["year"].isin([2024, 2025])]
    print(f"[simulate test] races={te[G.COL_RID].nunique():,}")
    agg, perrace = G.simulate(te, payouts, widepays, cal, curves, oddsmap)

    out = {}
    print("\n" + "=" * 100)
    print(f"{'券種':12s} {'セル':>4s} {'集計ROI':>8s} {'SD観測':>7s} {'SDノイズ':>8s} "
          f"{'真の振れτ':>9s} {'ノイズ%':>7s} {'セル幅[min-max]':>16s}")
    print("-" * 100)
    for bet in G.ALL_BETS:
        cells = []
        for (cell, b), pairs in perrace.items():
            if b != bet or cell == "ALL":
                continue
            if len(pairs) < G.MIN_RACES:
                continue
            roi, se = cluster_roi_se(pairs)
            if roi == roi:
                cells.append((cell, roi, se, len(pairs)))
        if len(cells) < 3:
            out[bet] = {"n_cells": len(cells)}
            print(f"{bet:12s}  セル不足({len(cells)})")
            continue
        rois = np.array([c[1] for c in cells])
        ses = np.array([c[2] for c in cells])
        nr = np.array([c[3] for c in cells], float)
        w = nr / nr.sum()
        mean_roi = float((w * rois).sum())
        V_obs = float((w * (rois - mean_roi) ** 2).sum())
        V_noise = float((w * ses ** 2).sum())
        tau2 = max(0.0, V_obs - V_noise)
        allpairs = perrace.get(("ALL", bet), [])
        roi_all, se_all = cluster_roi_se(allpairs)
        d = {
            "n_cells": len(cells),
            "agg_roi": roi_all, "agg_se": se_all,
            "cell_roi_mean": mean_roi,
            "SD_observed": V_obs ** 0.5,
            "SD_noise": V_noise ** 0.5,
            "SD_true_tau": tau2 ** 0.5,
            "noise_share": (V_noise / V_obs) if V_obs > 0 else None,
            "roi_min": float(rois.min()), "roi_med": float(np.median(rois)),
            "roi_max": float(rois.max()),
        }
        out[bet] = d
        print(f"{bet:12s} {d['n_cells']:4d} {roi_all*100:7.1f}% {d['SD_observed']*100:6.1f} "
              f"{d['SD_noise']*100:7.1f} {d['SD_true_tau']*100:8.1f}pt "
              f"{(d['noise_share'] or 0)*100:6.0f}% {d['roi_min']*100:6.0f}-{d['roi_max']*100:.0f}%")
    print("=" * 100)
    json.dump(out, open(BASE / "reports/roi_heterogeneity.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=float)
    print("[saved] reports/roi_heterogeneity.json")


if __name__ == "__main__":
    main()
