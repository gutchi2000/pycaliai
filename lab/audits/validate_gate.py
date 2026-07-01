"""
validate_gate.py
================
参戦ゲートの年分割 OOS 検証 + 終端分布リフト。

1. test を再スコアし per-race キャッシュ (year, p1, gap, ent, r_exotic[馬単3+三連複4]).
2. ゲート閾値は 2024 のみから算出 -> 2024(fit) と 2025(validate) の exotic ROI を並べる。
   汎化する(2025でも ungated 超え)ゲートだけ採用に値する。
3. 採用候補で f=2.5% 複利トーナメント終端をシミュ -> 入賞/表彰台/優勝 が上がるか。

実行: python validate_gate.py [--recache]
"""
from __future__ import annotations
import argparse, io, sys
from itertools import combinations, permutations
from pathlib import Path
import joblib, numpy as np, pandas as pd
import pl_probs as PL
from backtest_pl_ev import all_umatan_mat, all_sanrenpuku_tensor, load_payouts, apply_encoders

try:
    if not isinstance(sys.stdout, io.TextIOWrapper) or (sys.stdout.encoding or "").lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

BASE = Path(__file__).parent
MASTER_CSV = BASE / "data/master_v2_20130105-20251228.csv"
MODEL_PKL = BASE / "models/unified_rank_v6.pkl"
CACHE = BASE / "reports/_gate_cache.npz"
COL_RID = "レースID(新/馬番無)"; COL_BAN = "馬番"
K_UM, K_SP = 3, 4
START = 1_000_000
BARS = {"入賞106万": 1_060_000, "表彰台129万": 1_291_000, "優勝172万": 1_723_000}
N_RACES, N_SIMS, F = 100, 5000, 0.025


def build_cache():
    print("[cache] re-scoring ...")
    b = joblib.load(MODEL_PKL); model, feats, encs = b["model"], b["feature_cols"], b["encoders"]
    pays = load_payouts()
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df = df.dropna(subset=[COL_RID, "split"]).copy()
    sub = df[df["split"] == "test"].copy()
    sub = apply_encoders(sub, encs)
    X = sub[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    sub["_score"] = model.predict(X)

    rec = {k: [] for k in ["year", "p1", "gap", "ent", "r_exotic"]}
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
        ban = ban[ok].astype(int); sc = sc[ok]; n = len(sc)
        b2i = {int(x): i for i, x in enumerate(ban)}
        try:
            lwin, lplc, lsho = b2i[int(pr["win"])], b2i[int(pr["plc"])], b2i[int(pr["sho"])]
        except (KeyError, ValueError, TypeError):
            continue
        um_pay, sp_pay = pr["umatan"], pr["sanrenpuku"]
        if not (um_pay == um_pay and sp_pay == sp_pay) or n < 3 or not um_pay or not sp_pay:
            continue
        w = PL.pl_weights(sc); p = w / w.sum()
        ps = np.sort(p)[::-1]
        ent = float(-(p * np.log(np.clip(p, 1e-12, 1))).sum() / np.log(n))

        U = all_umatan_mat(w)
        um_r = int((U[~np.eye(n, dtype=bool)] > U[lwin, lplc]).sum())
        P3 = all_sanrenpuku_tensor(w)
        tris = np.array(list(combinations(range(n), 3)))
        ptri = np.zeros(len(tris))
        for a, bb, c in permutations(range(3)):
            ptri += P3[tris[:, a], tris[:, bb], tris[:, c]]
        wt = tuple(sorted((lwin, lplc, lsho)))
        wm = (tris[:, 0] == wt[0]) & (tris[:, 1] == wt[1]) & (tris[:, 2] == wt[2])
        if not wm.any():
            continue
        sp_r = int((ptri > ptri[wm][0]).sum())
        re = (1 / 7) * ((um_r < K_UM) * float(um_pay) / 100) + (1 / 7) * ((sp_r < K_SP) * float(sp_pay) / 100)

        rec["year"].append(int(str(rid)[:4])); rec["p1"].append(float(ps[0]))
        rec["gap"].append(float(ps[0] - ps[1])); rec["ent"].append(ent); rec["r_exotic"].append(re)
    arrs = {k: np.array(v, dtype=float) for k, v in rec.items()}
    CACHE.parent.mkdir(exist_ok=True); np.savez(CACHE, **arrs)
    print(f"[cache] saved races={len(arrs['year']):,}")
    return arrs


def simulate(pool, f, rng):
    idx = rng.integers(0, len(pool), size=(N_SIMS, N_RACES))
    re = pool[idx]
    bank = np.full(N_SIMS, float(START))
    for t in range(N_RACES):
        stake = f * bank
        bank = bank - stake + stake * re[:, t]
        np.maximum(bank, 0.0, out=bank)
    return bank


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--recache", action="store_true")
    args = ap.parse_args()
    arrs = dict(np.load(CACHE)) if (CACHE.exists() and not args.recache) else build_cache()
    year, p1, gap, ent, rex = arrs["year"], arrs["p1"], arrs["gap"], arrs["ent"], arrs["r_exotic"]
    fin = np.isfinite(rex)
    year, p1, gap, ent, rex = year[fin], p1[fin], gap[fin], ent[fin], rex[fin]
    y24, y25 = year == 2024, year == 2025
    print(f"[data] 2024={y24.sum():,}R  2025={y25.sum():,}R")

    # ---- ゲート候補 (閾値は 2024 のみから) ----
    q = lambda a, qq: float(np.quantile(a[y24], qq))
    gates = {
        "ungated":                np.ones(len(rex), bool),
        "gap<=q90(脱・本命堅)":     gap <= q(gap, .90),
        "ent<=q60(構造明快)":       ent <= q(ent, .60),
        "ent in[q05,q85]":         (ent >= q(ent, .05)) & (ent <= q(ent, .85)),
        "gap<=q90 & ent<=q70":     (gap <= q(gap, .90)) & (ent <= q(ent, .70)),
    }
    print("\n" + "=" * 76)
    print("ゲート年分割: exotic ROI (馬単3+三連複4, flat). 閾値=2024算出")
    print("=" * 76)
    print(f"{'gate':<24s} {'ROI24':>7s} {'n24':>7s} {'ROI25':>7s} {'n25':>7s} {'汎化':>6s}")
    base25 = rex[y25].mean() * 100
    keep = []
    for name, gmask in gates.items():
        r24 = rex[gmask & y24].mean() * 100; n24 = int((gmask & y24).sum())
        r25 = rex[gmask & y25].mean() * 100; n25 = int((gmask & y25).sum())
        gen = "○" if (name != "ungated" and r25 > base25 + 0.5) else ("-" if name == "ungated" else "×")
        if gen == "○":
            keep.append((name, gmask, r25))
        print(f"{name:<24s} {r24:>7.1f} {n24:>7d} {r25:>7.1f} {n25:>7d} {gen:>6s}")

    # ---- 終端リフト: 汎化ゲートで sim ----
    print("\n" + "=" * 76)
    print(f"終端リフト (f={F}, pure exotic, {N_SIMS}試行). pool=全期間ゲート通過")
    print("=" * 76)
    rng = np.random.default_rng(0)

    def report(label, pool):
        term = simulate(pool, F, rng)
        P = {k: float((term >= v).mean() * 100) for k, v in BARS.items()}
        print(f"  {label:<26s} n={len(pool):>5d} 中央値={np.median(term):>9,.0f}  "
              + "  ".join(f"{k}={P[k]:.1f}%" for k in BARS))

    report("ungated(全レース)", rex)
    for name, gmask, _ in sorted(keep, key=lambda x: -x[2]):
        report(name, rex[gmask])
    if not keep:
        print("  汎化したゲート無し -> ゲートは過学習。ungated のまま運用が正しい。")


if __name__ == "__main__":
    main()
