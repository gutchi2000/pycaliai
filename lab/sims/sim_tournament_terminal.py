"""
sim_tournament_terminal.py
==========================
複利を入れたトーナメント終端ポイント分布を、去年の実バーに対して評価。

方策 (ハイブリッド):
  ベース = 複勝◎ (モデル top1 を複勝。高的中で中央値を守る)
  アッパー = 馬単 top3 + 三連複 top4 (前回検証した変動エンジン)
  毎レース: stake = f * bankroll を base:exotic = (1-mix):mix で配分し、複利。

サンプリング:
  OOS(2024-2025) のレースを idx で 100R 抽出 (base と exotic は同一レースで連動)。
  N=5000 トーナメント試行 -> 終端 bankroll 分布。

評価 (去年の実バー, 初期100万):
  P(>=1,024,000) = 10位/入賞ライン (≈トントン)
  P(>=1,291,000) = 3位/表彰台(実弾)
  P(>=1,723,000) = 1位/優勝

実行: python sim_tournament_terminal.py            # 初回 (スコア+キャッシュ)
      python sim_tournament_terminal.py --recache  # 再計算
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
CACHE_NPZ  = BASE / "reports/_tournament_cache.npz"
COL_RID = "レースID(新/馬番無)"; COL_BAN = "馬番"

K_UM, K_SP = 3, 4
START = 1_000_000
# 去年の実績: 9位(賞金最下)=106万, 5位=121万, 表彰台=129万, 優勝=172万
BARS = {"トントン100万": 1_000_000, "入賞9位106万": 1_060_000,
        "5位121万": 1_212_000, "表彰台129万": 1_291_000, "優勝172万": 1_723_000}
N_RACES = 100      # トーナメントの投票数
N_SIMS  = 5000


def build_cache():
    print("[cache] re-scoring test split for tournament sim ...")
    bundle = joblib.load(MODEL_PKL)
    model, feats, encs = bundle["model"], bundle["feature_cols"], bundle["encoders"]
    pays = load_payouts()
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df = df.dropna(subset=[COL_RID, "split"]).copy()
    sub = df[df["split"] == "test"].copy()
    sub = apply_encoders(sub, encs)
    X = sub[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    sub["_score"] = model.predict(X)

    r_base, r_exotic = [], []          # 1円あたりリターン (複勝◎ / 馬単+三連複)
    base_hit, um_hit, sp_hit = [], [], []
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
        um_pay, sp_pay = pr["umatan"], pr["sanrenpuku"]
        f_win, f_plc, f_sho = pr["fuku_win"], pr["fuku_plc"], pr["fuku_sho"]
        if not (um_pay == um_pay and sp_pay == sp_pay and f_win == f_win):
            continue
        if n < 3 or not um_pay or not sp_pay:
            continue

        w = PL.pl_weights(sc)

        # ベース: 複勝◎ (モデル top1)
        maru = int(np.argmax(sc))
        if maru == lwin:   rb = float(f_win) / 100.0; bh = 1
        elif maru == lplc: rb = float(f_plc) / 100.0; bh = 1
        elif maru == lsho: rb = float(f_sho) / 100.0; bh = 1
        else:              rb = 0.0; bh = 0

        # アッパー: 馬単 top3 + 三連複 top4 (stake を 7 等分)
        U = all_umatan_mat(w)
        probs = U[~np.eye(n, dtype=bool)]
        um_r = int((probs > U[lwin, lplc]).sum())
        uh = 1 if um_r < K_UM else 0

        P3 = all_sanrenpuku_tensor(w)
        tris = np.array(list(combinations(range(n), 3)))
        ptri = np.zeros(len(tris))
        for a, b, c in permutations(range(3)):
            ptri += P3[tris[:, a], tris[:, b], tris[:, c]]
        wt = tuple(sorted((lwin, lplc, lsho)))
        wm = (tris[:, 0] == wt[0]) & (tris[:, 1] == wt[1]) & (tris[:, 2] == wt[2])
        if not wm.any():
            continue
        sp_r = int((ptri > ptri[wm][0]).sum())
        sh = 1 if sp_r < K_SP else 0

        re = (1.0 / 7.0) * (uh * float(um_pay) / 100.0) + (1.0 / 7.0) * (sh * float(sp_pay) / 100.0)

        r_base.append(rb); r_exotic.append(re)
        base_hit.append(bh); um_hit.append(uh); sp_hit.append(sh)

    arrs = {
        "r_base": np.array(r_base), "r_exotic": np.array(r_exotic),
        "base_hit": np.array(base_hit), "um_hit": np.array(um_hit), "sp_hit": np.array(sp_hit),
    }
    CACHE_NPZ.parent.mkdir(exist_ok=True)
    np.savez(CACHE_NPZ, **arrs)
    print(f"[cache] saved races={len(r_base):,}")
    return arrs


def simulate(r_base, r_exotic, f, mix, rng):
    """N_SIMS トーナメント。各 100R 複利。終端 bankroll ベクトルを返す."""
    R = len(r_base)
    idx = rng.integers(0, R, size=(N_SIMS, N_RACES))
    rb = r_base[idx]; re = r_exotic[idx]          # (N_SIMS, N_RACES)
    bank = np.full(N_SIMS, float(START))
    for t in range(N_RACES):
        stake = f * bank
        ret = stake * ((1 - mix) * rb[:, t] + mix * re[:, t])
        bank = bank - stake + ret
        np.maximum(bank, 0.0, out=bank)
    return bank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recache", action="store_true")
    args = ap.parse_args()

    if CACHE_NPZ.exists() and not args.recache:
        arrs = dict(np.load(CACHE_NPZ)); print(f"[cache] loaded races={len(arrs['r_base']):,}")
    else:
        arrs = build_cache()
    r_base, r_exotic = arrs["r_base"], arrs["r_exotic"]
    fin = np.isfinite(r_base) & np.isfinite(r_exotic)
    r_base, r_exotic = r_base[fin], r_exotic[fin]
    print(f"[clean] finite races={fin.sum():,} (dropped {(~fin).sum()})")

    print("\n[sanity] 単独平均ROI (フラット):")
    print(f"   ベース複勝◎ : ROI={r_base.mean()*100:5.1f}%  的中={arrs['base_hit'][fin].mean()*100:5.1f}%")
    print(f"   アッパー馬単+三連複: ROI={r_exotic.mean()*100:5.1f}%  "
          f"(馬単的中={arrs['um_hit'][fin].mean()*100:.1f}% 三連複的中={arrs['sp_hit'][fin].mean()*100:.1f}%)")

    print("\n" + "=" * 92)
    print(f"複利トーナメント終端分布  ({N_RACES}R, {N_SIMS}試行, 初期{START:,})")
    print("バー: 10位≥1,024,000 / 3位≥1,291,000 / 1位≥1,723,000")
    print("=" * 92)
    rng = np.random.default_rng(0)
    F_GRID = [0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025, 0.03]
    MIX_GRID = [1.0]   # pure exotic 確定
    bar_keys = list(BARS.keys())
    hdr = " ".join(f"{k:>11s}" for k in bar_keys)
    print(f"{'f(賭率)':>7s} {'中央値':>10s} {'p90':>10s} {'p99':>10s}  {hdr}")
    best = {k: None for k in bar_keys}
    for f in F_GRID:
        for mix in MIX_GRID:
            term = simulate(r_base, r_exotic, f, mix, rng)
            med = np.median(term); p90 = np.percentile(term, 90); p99 = np.percentile(term, 99)
            P = {k: float((term >= v).mean() * 100) for k, v in BARS.items()}
            for k in best:
                if best[k] is None or P[k] > best[k][1]:
                    best[k] = (f, P[k])
            pstr = " ".join(f"{P[k]:>10.1f}%" for k in bar_keys)
            print(f"{f:>7.3f} {med:>10,.0f} {p90:>10,.0f} {p99:>10,.0f}  {pstr}")

    print("\n[各バー最適 f] (pure exotic)")
    for k, (f, p) in best.items():
        print(f"   {k:<14s}: f={f}  ->  P={p:.1f}%")


if __name__ == "__main__":
    main()
