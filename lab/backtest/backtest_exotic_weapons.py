"""
backtest_exotic_weapons.py
==========================
馬単(k=3) + 三連複(k=4) を「武器」化する検証。
frontier の次の繰り返し: 参戦ゲート と ポートフォリオ。

1. 全 test(2024-2025) レースを再スコアし、per-race を npz キャッシュ:
   n, p1(最大単勝確率), gap(top1-top2), entropy, um_rank, um_pay, sp_rank, sp_pay
   (2回目以降はキャッシュ即ロード -> ゲート/ポートフォリオの反復が一瞬)

2. レース特徴のデシル別 ROI: どんなレースで馬単/三連複が食えるか可視化
   (= 参戦ゲートの根拠。chalk 過剰人気帯 vs 妙味帯を切り分ける)

3. 馬単+三連複 ポートフォリオ (毎R 2 券種=7点) の 96/192 bet 窓ブートストラップ:
   ungated vs best-gated で 実現ROI 分位 と P(ROI>=120%) を比較

実行: python backtest_exotic_weapons.py          # 初回 (スコア+キャッシュ)
      python backtest_exotic_weapons.py --recache # 強制再計算
"""
from __future__ import annotations

import argparse
import io
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
CACHE_NPZ  = BASE / "reports/_exotic_weapons_cache.npz"

COL_RID = "レースID(新/馬番無)"; COL_BAN = "馬番"
BET = 100
K_UM = 3   # 馬単 operating k
K_SP = 4   # 三連複 operating k
FLOOR = {"um": 75.0, "sp": 75.0}


def score_split(df, model, feats, encs):
    sub = df[df["split"] == "test"].copy()
    sub = apply_encoders(sub, encs)
    X = sub[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    sub["_score"] = model.predict(X)
    return sub


def build_cache():
    print("[cache] re-scoring test split ...")
    bundle = joblib.load(MODEL_PKL)
    model, feats, encs = bundle["model"], bundle["feature_cols"], bundle["encoders"]
    pays = load_payouts()
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df = df.dropna(subset=[COL_RID, "split"]).copy()
    sub = score_split(df, model, feats, encs)

    rec = {k: [] for k in ["n", "p1", "gap", "ent", "um_rank", "um_pay", "sp_rank", "sp_pay"]}
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
        b2i = {int(b): i for i, b in enumerate(ban)}
        try:
            lwin, lplc, lsho = b2i[int(pr["win"])], b2i[int(pr["plc"])], b2i[int(pr["sho"])]
        except (KeyError, ValueError, TypeError):
            continue
        w = PL.pl_weights(sc)
        p = w / w.sum()
        ps = np.sort(p)[::-1]
        p1 = float(ps[0]); p2 = float(ps[1]) if n > 1 else 0.0
        ent = float(-(p * np.log(np.clip(p, 1e-12, 1))).sum() / np.log(n))

        # 馬単
        um_rank, um_pay = -1, np.nan
        pay = pr["umatan"]
        if pay == pay and pay and n >= 2:
            U = all_umatan_mat(w)
            probs = U[~np.eye(n, dtype=bool)]
            um_rank = int((probs > U[lwin, lplc]).sum()); um_pay = float(pay)

        # 三連複
        sp_rank, sp_pay = -1, np.nan
        pay = pr["sanrenpuku"]
        if pay == pay and pay and n >= 3:
            P3 = all_sanrenpuku_tensor(w)
            tris = np.array(list(combinations(range(n), 3)))
            ptri = np.zeros(len(tris))
            for a, b, c in permutations(range(3)):
                ptri += P3[tris[:, a], tris[:, b], tris[:, c]]
            wt = tuple(sorted((lwin, lplc, lsho)))
            wm = (tris[:, 0] == wt[0]) & (tris[:, 1] == wt[1]) & (tris[:, 2] == wt[2])
            if wm.any():
                sp_rank = int((ptri > ptri[wm][0]).sum()); sp_pay = float(pr["sanrenpuku"])

        rec["n"].append(n); rec["p1"].append(p1); rec["gap"].append(p1 - p2); rec["ent"].append(ent)
        rec["um_rank"].append(um_rank); rec["um_pay"].append(um_pay)
        rec["sp_rank"].append(sp_rank); rec["sp_pay"].append(sp_pay)

    arrs = {k: np.array(v, dtype=float) for k, v in rec.items()}
    CACHE_NPZ.parent.mkdir(exist_ok=True)
    np.savez(CACHE_NPZ, **arrs)
    print(f"[cache] saved {CACHE_NPZ.name}  races={len(arrs['n']):,}")
    return arrs


def roi_hit(rank, pay, k, mask):
    """mask レース集合で top-k 買いの (ROI%, hit%, n)."""
    m = mask & (rank >= 0)
    nR = int(m.sum())
    if nR == 0:
        return 0.0, 0.0, 0
    hit = m & (rank < k)
    ret = float(pay[hit].sum())
    cost = nR * k * BET
    return ret / cost * 100, hit.sum() / nR * 100, nR


def decile_table(name, rank, pay, k, feat, feat_name, floor):
    base = rank >= 0
    qs = np.quantile(feat[base], np.linspace(0, 1, 11))
    print(f"\n  {name} (k={k}) を {feat_name} デシル別:  [floor={floor:.0f}%]")
    print(f"    {'decile':>7s} {feat_name+'範囲':>22s} {'n':>6s} {'的中%':>7s} {'ROI%':>7s}")
    for d in range(10):
        lo, hi = qs[d], qs[d + 1]
        if d < 9:
            sel = base & (feat >= lo) & (feat < hi)
        else:
            sel = base & (feat >= lo)
        roi, hr, nR = roi_hit(rank, pay, k, sel)
        flag = " *" if roi >= floor + 5 else ("  <100" if roi >= 100 else "")
        print(f"    {d:>7d} {f'[{lo:.3f},{hi:.3f})':>22s} {nR:>6d} {hr:>7.2f} {roi:>7.1f}{flag}")


def bootstrap_portfolio(arrs, mask, label, rng):
    """馬単(k3)+三連複(k4) を毎R両方買う portfolio。96/192 bet 窓。"""
    m = mask & (arrs["um_rank"] >= 0) & (arrs["sp_rank"] >= 0)
    um_ret = np.where(arrs["um_rank"] < K_UM, np.nan_to_num(arrs["um_pay"]), 0.0)[m]
    sp_ret = np.where(arrs["sp_rank"] < K_SP, np.nan_to_num(arrs["sp_pay"]), 0.0)[m]
    per_race_ret = um_ret + sp_ret
    per_race_cost = (K_UM + K_SP) * BET   # 700/レース
    N = len(per_race_ret)
    mean_roi = per_race_ret.sum() / (N * per_race_cost) * 100 if N else 0
    print(f"\n  [{label}] races={N:,}  平均ROI={mean_roi:.1f}%")
    for races_per_window in (48, 96):  # 48R*2券種=96bet, 96R*2=192bet
        if N < races_per_window:
            continue
        idx = rng.integers(0, N, size=(5000, races_per_window))
        roi = per_race_ret[idx].sum(axis=1) / (races_per_window * per_race_cost) * 100
        p = np.percentile(roi, [5, 50, 90, 95, 99]).round(0)
        p120 = float((roi >= 120).mean() * 100)
        print(f"    {races_per_window:>3d}R({races_per_window*2}bet) "
              f"ROI p5/p50/p90/p95/p99={p[0]:.0f}/{p[1]:.0f}/{p[2]:.0f}/{p[3]:.0f}/{p[4]:.0f}%  "
              f"P(ROI>=120%)={p120:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recache", action="store_true")
    args = ap.parse_args()

    if CACHE_NPZ.exists() and not args.recache:
        arrs = dict(np.load(CACHE_NPZ))
        print(f"[cache] loaded {CACHE_NPZ.name}  races={len(arrs['n']):,}")
    else:
        arrs = build_cache()

    print("\n" + "=" * 78)
    print("STEP1: 参戦ゲート探索 — どのレースで食えるか (デシル別 ROI, true OOS)")
    print("=" * 78)
    for feat_name in ["p1", "gap", "ent"]:
        decile_table("馬単", arrs["um_rank"], arrs["um_pay"], K_UM, arrs[feat_name], feat_name, FLOOR["um"])
        decile_table("三連複", arrs["sp_rank"], arrs["sp_pay"], K_SP, arrs[feat_name], feat_name, FLOOR["sp"])

    print("\n" + "=" * 78)
    print("STEP2: 馬単+三連複 ポートフォリオ ブートストラップ")
    print("=" * 78)
    rng = np.random.default_rng(0)
    all_mask = np.ones(len(arrs["n"]), dtype=bool)
    bootstrap_portfolio(arrs, all_mask, "ungated 全レース", rng)


if __name__ == "__main__":
    main()
