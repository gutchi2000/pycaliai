"""
backtest_exotic_frontier.py
===========================
高配当系 (馬単 / 三連複 / 三連単) の「prob-first top-k」買いで、
真の OOS (test = 2024-2025) において到達可能な (的中率, 回収率) フロンティアを測る。

- 確率: unified_rank_v6 -> PL weights -> pl_probs の厳密 joint (raw)
        (prob-first の順位は単調 calibration で不変なので raw で十分)
- 選択: prob-first = 確率上位 top-k のみ買う。EV は使わない (= model-only)。
        大会 API が三連系ライブオッズを配らない前提に一致。
- 払戻: kekka_*.csv の確定配当 (実現値, リークなし)
- 出力: 券種 x k の (races, hit%, ROI%) 表
        + トーナメント窓 (96/200R) の実現ROI ブートストラップ分位 + P(ROI>=120%)

実行: python backtest_exotic_frontier.py
"""
from __future__ import annotations

import io
import json
import sys
from itertools import combinations, permutations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import pl_probs as PL
from backtest_pl_ev import all_umatan_mat, all_sanrenpuku_tensor, load_payouts, apply_encoders

try:
    if not isinstance(sys.stdout, io.TextIOWrapper) or (sys.stdout.encoding or "").lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

BASE       = Path(__file__).parent
MASTER_CSV = BASE / "data/master_v2_20130105-20251228.csv"
MODEL_PKL  = BASE / "models/unified_rank_v6.pkl"
OUT_JSON   = BASE / "reports/backtest_exotic_frontier.json"

COL_RID  = "レースID(新/馬番無)"
COL_JYUN = "着順"
COL_BAN  = "馬番"
BET = 100

# JRA 払戻率 (控除後ベースライン = 平均的にここに収束する)
TAKEOUT_BASE = {"umatan": 0.750, "sanrenpuku": 0.750, "sanrentan": 0.725}

# k = 1 レースあたり買う組合せ数
KGRID = {
    "umatan":     [1, 2, 3, 4, 6, 8, 12, 18, 24],
    "sanrenpuku": [1, 2, 4, 6, 10, 15, 20, 30, 40, 60],
    "sanrentan":  [1, 2, 4, 6, 12, 18, 24, 36, 48, 72, 100],
}
# ユーザ想定ライン
TARGET = {
    "umatan":     ("15-20%", "120%+"),
    "sanrenpuku": ("10-20%", "120%+"),
    "sanrentan":  ("5-10%",  "120%+"),
}
LABEL = {"umatan": "馬単", "sanrenpuku": "三連複", "sanrentan": "三連単"}


def score_split(df, model, feats, encs, split_label):
    sub = df[df["split"] == split_label].copy()
    sub = apply_encoders(sub, encs)
    X = sub[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    sub["_score"] = model.predict(X)
    return sub


def main():
    print("=" * 80)
    print("backtest_exotic_frontier.py  —  馬単/三連複/三連単 prob-first フロンティア (true OOS)")
    print("=" * 80)

    bundle = joblib.load(MODEL_PKL)
    model = bundle["model"]; feats = bundle["feature_cols"]; encs = bundle["encoders"]
    print(f"[model] {MODEL_PKL.name}  feats={len(feats)}")

    print("[load]  payouts (kekka)")
    pays = load_payouts()
    print(f"        {len(pays):,} races with confirmed payouts")

    print(f"[load]  {MASTER_CSV.name}")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df = df.dropna(subset=[COL_RID, "split"]).copy()

    sub = score_split(df, model, feats, encs, "test")
    print(f"[score] test rows={len(sub):,}  races={sub[COL_RID].nunique():,}")

    bet_types = ["umatan", "sanrenpuku", "sanrentan"]
    per_race = {bt: [] for bt in bet_types}  # (rank_of_winning_combo, payout_per100)

    n_used = 0
    for rid, g in sub.groupby(COL_RID, sort=False):
        pr = pays.get(str(rid)[:16])
        if pr is None:
            continue
        g = g.sort_values(COL_BAN)
        ban = pd.to_numeric(g[COL_BAN], errors="coerce").values
        sc = g["_score"].values
        ok = ~np.isnan(ban)
        if ok.sum() < 3:
            continue
        ban = ban[ok].astype(int); sc = sc[ok]
        n = len(sc)
        ban2idx = {int(b): i for i, b in enumerate(ban)}
        try:
            lwin = ban2idx[int(pr["win"])]; lplc = ban2idx[int(pr["plc"])]; lsho = ban2idx[int(pr["sho"])]
        except (KeyError, ValueError, TypeError):
            continue
        w = PL.pl_weights(sc)
        n_used += 1

        # ---- 馬単 ----
        pay = pr["umatan"]
        if pay == pay and pay and n >= 2:
            U = all_umatan_mat(w)
            probs = U[~np.eye(n, dtype=bool)]
            win_p = U[lwin, lplc]
            per_race["umatan"].append((int((probs > win_p).sum()), float(pay)))

        # ---- 三連単 / 三連複 共通 tensor (順序付き) ----
        P3 = all_sanrenpuku_tensor(w)

        pay = pr["sanrentan"]
        if pay == pay and pay and n >= 3:
            I, J, K = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
            m3 = (I != J) & (J != K) & (I != K)
            probs = P3[m3]
            win_p = P3[lwin, lplc, lsho]
            per_race["sanrentan"].append((int((probs > win_p).sum()), float(pay)))

        pay = pr["sanrenpuku"]
        if pay == pay and pay and n >= 3:
            tris = np.array(list(combinations(range(n), 3)))
            ptri = np.zeros(len(tris))
            for a, b, c in permutations(range(3)):
                ptri += P3[tris[:, a], tris[:, b], tris[:, c]]
            wt = tuple(sorted((lwin, lplc, lsho)))
            wm = (tris[:, 0] == wt[0]) & (tris[:, 1] == wt[1]) & (tris[:, 2] == wt[2])
            if wm.any():
                win_p = ptri[wm][0]
                per_race["sanrenpuku"].append((int((ptri > win_p).sum()), float(pay)))

    print(f"[done]  races used={n_used:,}")

    # ---- frontier 集計 ----
    report = {}
    for bt in bet_types:
        data = per_race[bt]
        ranks = np.array([d[0] for d in data], dtype=int)
        pys = np.array([d[1] for d in data], dtype=float)
        nR = len(data)
        rows = []
        for k in KGRID[bt]:
            hit = ranks < k
            cost = nR * k * BET
            ret = float(pys[hit].sum())
            rows.append({
                "k": k, "n_races": nR,
                "hit_rate": round(hit.mean() * 100, 2) if nR else 0.0,
                "roi": round(ret / cost * 100, 1) if cost else 0.0,
                "n_hits": int(hit.sum()),
            })
        report[bt] = {"n_races": nR, "rows": rows,
                      "_ranks": ranks, "_pays": pys}

    # ---- print frontier ----
    for bt in bet_types:
        rep = report[bt]
        th, tr = TARGET[bt]
        base = TAKEOUT_BASE[bt] * 100
        print("\n" + "=" * 80)
        print(f"【{LABEL[bt]}】 races={rep['n_races']:,}  払戻率(控除後)={base:.1f}%  "
              f"目標: 的中{th} / 回収{tr}")
        print(f"{'k(点/R)':>8s} {'的中率%':>9s} {'回収率%':>9s} {'当たりR':>9s}")
        for r in rep["rows"]:
            flag = "  <= 100%超" if r["roi"] >= 100 else ""
            print(f"{r['k']:>8d} {r['hit_rate']:>9.2f} {r['roi']:>9.1f} {r['n_hits']:>9d}{flag}")

    # ---- bootstrap tournament window ----
    print("\n" + "=" * 80)
    print("トーナメント窓 実現ROIブートストラップ (各券種を毎R flat top-k, 4000 試行)")
    print("=" * 80)
    rng = np.random.default_rng(0)
    boot = {}
    for bt in bet_types:
        rep = report[bt]
        ranks = rep["_ranks"]; pys = rep["_pays"]
        # ROI 最大の k を採用
        best = max(rep["rows"], key=lambda r: r["roi"])
        k = best["k"]
        ret_series = np.where(ranks < k, pys, 0.0)
        cost = 100 * k
        N = len(ret_series)
        boot[bt] = {"k": k}
        for window in (96, 200):
            if N < window:
                continue
            idx = rng.integers(0, N, size=(4000, window))
            roi_dist = ret_series[idx].sum(axis=1) / (window * cost) * 100
            pcts = np.percentile(roi_dist, [5, 50, 90, 95, 99]).round(0).tolist()
            p120 = round(float((roi_dist >= 120).mean() * 100), 1)
            boot[bt][f"w{window}"] = {"p5_50_90_95_99": pcts, "P_roi_ge_120": p120}
            print(f"{LABEL[bt]:>4s} k={k:<3d} window={window:>3d}R  "
                  f"ROI p5/p50/p90/p95/p99 = "
                  f"{pcts[0]:.0f}/{pcts[1]:.0f}/{pcts[2]:.0f}/{pcts[3]:.0f}/{pcts[4]:.0f}%  "
                  f"P(ROI>=120%)={p120:.1f}%")

    # ---- save ----
    payload = {
        "purpose": "馬単/三連複/三連単 prob-first top-k フロンティア (true OOS test=2024-2025, model-only)",
        "model": MODEL_PKL.name,
        "n_races_used": n_used,
        "takeout_base": TAKEOUT_BASE,
        "target": {k: {"hit": v[0], "roi": v[1]} for k, v in TARGET.items()},
        "frontier": {bt: report[bt]["rows"] for bt in bet_types},
        "bootstrap": boot,
    }
    OUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {OUT_JSON}")


if __name__ == "__main__":
    main()
