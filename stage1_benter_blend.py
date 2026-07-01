"""
stage1_benter_blend.py
======================
Stage 1: Benter 式二段階合成（単勝）。p_i = softmax(α·log f_i + β·log π_i) をレース内で
実際の勝ち馬に MLE フィット。f=v6較正PL単勝確率, π=de-vig市場marginal。

未実施の最大の欠落＝「モデル確率を市場と合成した後に α が有意に残るか」を測る。

変種A: π=π9（当日9時, 実戦運用）★主役 / 変種B: π=πc（確定, 理論上限）
対照: fundamental-only(β=0), market-only(α=0)
fit=≤2023, eval=test(2024-25)。

指標: (1)α,β+有意性(bootstrap CI) (2)ΔR²(McFadden, vs market-only) (3)賭けROI(EVゲート,
決済=確定odds, vs控除率80%) (4)CLV(log odds9 − log oddsc 平均符号) +変動層別。

leak規約: A の EV ゲート/意思決定は 当日9時 odds のみ。確定は決済とCLVのみ。
  B は確定odds（理論上限/oracle）。閾値は≤2023選定、test評価のみ。

出力: reports/stage1_benter_blend.json
実行: python stage1_benter_blend.py
"""
from __future__ import annotations
import glob, json, re, sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

import pl_probs as PL
from joint_calibration_v6 import apply_encoders, COL_RID, COL_JYUN, COL_BAN, MASTER_CSV

BASE = Path(__file__).parent
np.random.seed(42)
ODIR = BASE / "data" / "Time _series_odds"
EV_GRID = [0.9, 1.0, 1.05, 1.1, 1.2, 1.3, 1.5]
N_BOOT = 120
MIN_TUNE_BETS = 300


def load_tanpuk_snapshots():
    files = sorted(glob.glob(str(ODIR / "TANPUK_*.csv")))
    tp = pd.concat([pd.read_csv(f, encoding="cp932", low_memory=False) for f in files],
                   ignore_index=True)
    cols = list(tp.columns)
    RID, KB, TM, TOU = cols[0], cols[1], cols[2], cols[3]
    tan_cols = {}
    for c in cols:
        m = re.match(r"^\s*(\d+)\s*単\s*$", str(c))
        if m:
            tan_cols[int(m.group(1))] = c
    tp["rid16"] = tp[RID].astype(str).str.replace(r"\D", "", regex=True).str[:16]
    tp[KB] = pd.to_numeric(tp[KB], errors="coerce")
    tp[TM] = pd.to_numeric(tp[TM], errors="coerce")
    tp = tp[(tp["rid16"].str.len() == 16) & (tp["rid16"].str[:4].astype(int) >= 2013)]
    tan_items = sorted(tan_cols.items())

    def extract(row, tou):
        out = {}
        for k, c in tan_items:
            if k <= tou:
                v = pd.to_numeric(pd.Series([row[c]]), errors="coerce").iloc[0]
                if pd.notna(v) and v > 1.0:
                    out[k] = float(v)
        return out

    snaps = {}
    for rid, g in tp.groupby("rid16", sort=False):
        try:
            tou = int(pd.to_numeric(g[TOU], errors="coerce").dropna().iloc[0])
        except Exception:
            continue
        race_mmdd = int(rid[4:8])
        g1 = g[g[KB] == 1]; g4 = g[g[KB] == 4]
        if len(g1) == 0 or len(g4) == 0:
            continue
        hh = (g1[TM] % 10000); mmdd = (g1[TM] // 10000)
        same = g1[mmdd == race_mmdd]; base = same if len(same) > 0 else g1
        am9 = base.loc[(hh.loc[base.index] - 900).abs().idxmin()]
        fin = g4.loc[g4[TM].idxmax()]
        o9 = extract(am9, tou); oc = extract(fin, tou)
        if len(o9) >= 5 and len(oc) >= 5:
            snaps[rid] = {"o9": o9, "oc": oc, "t9": int(am9[TM]), "tou": tou}
    return snaps


def devig(odds_dict, keys):
    inv = np.array([1.0 / odds_dict[k] for k in keys])
    s = inv.sum()
    return inv / s if s > 0 else inv


def build_races():
    print("[load] TANPUK 単勝 時系列 (3結合, 当日9時/確定 snapshot)...")
    snaps = load_tanpuk_snapshots()
    print(f"  races with 9時+確定: {len(snaps):,}")
    # de-vig sanity (確認(b) 再表示)
    cnt = 0
    for rid, sp in snaps.items():
        ks = list(sp["o9"].keys())
        sinv = float(np.sum([1.0 / sp["o9"][k] for k in ks]))
        print(f"  [devig sanity] rid={rid} 頭数={sp['tou']} Σ(1/単9時)={sinv:.4f} (~1.25=控除20%)")
        cnt += 1
        if cnt >= 3:
            break

    b = joblib.load(BASE / "models/unified_rank_v6.pkl")
    model, feats, encs = b["model"], b["feature_cols"], b["encoders"]
    cal = joblib.load(BASE / "models/pl_calibrators_v6.pkl"); cal = cal.get("calibrators", cal)
    print("[score] v6 PL 単勝 f on master...")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]).copy()
    enc = apply_encoders(df, encs)
    X = enc[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    df["_score"] = model.predict(X)

    races = []
    for rid, g in df.groupby(COL_RID, sort=False):
        if len(g) < 5:
            continue
        rid16 = str(rid)[:16]
        if rid16 not in snaps:
            continue
        g = g.sort_values(COL_BAN).reset_index(drop=True)
        ban = g[COL_BAN].astype(int).values
        w = PL.pl_weights(g["_score"].values)
        f_all = cal["tansho"].predict(PL.all_tansho(w)) if "tansho" in cal else PL.all_tansho(w)
        f_by_ban = {int(ban[i]): max(float(f_all[i]), 1e-9) for i in range(len(ban))}
        jyun = g[COL_JYUN].astype(int).values
        win_ban = int(ban[int(np.argmin(jyun))])
        sp = snaps[rid16]
        common = [k for k in sp["o9"].keys() if k in sp["oc"] and k in f_by_ban]
        if win_ban not in common or len(common) < 5:
            continue
        keys = sorted(common)
        f = np.array([f_by_ban[k] for k in keys])
        p9 = devig(sp["o9"], keys); pc = devig(sp["oc"], keys)
        o9 = np.array([sp["o9"][k] for k in keys]); oc = np.array([sp["oc"][k] for k in keys])
        wi = keys.index(win_ban)
        races.append({"logf": np.log(f), "logp9": np.log(p9), "logpc": np.log(pc),
                      "o9": o9, "oc": oc, "win": wi, "year": int(rid16[:4]),
                      "split": g["split"].iloc[0]})
    print(f"  blend対象レース: {len(races):,}")
    return races


def pack(races, pcol):
    """レース群を flat 配列 + segment offset 化（vectorized NLL 用）。"""
    sizes = np.array([len(r["logf"]) for r in races])
    starts = np.zeros(len(races), dtype=int)
    if len(races) > 1:
        starts[1:] = np.cumsum(sizes)[:-1]
    logf = np.concatenate([r["logf"] for r in races])
    logp = np.concatenate([r[pcol] for r in races])
    win_g = np.array([starts[i] + races[i]["win"] for i in range(len(races))])
    return {"logf": logf, "logp": logp, "starts": starts, "win_g": win_g, "sizes": sizes}


def nll_packed(a, b, pk):
    u = a * pk["logf"] + b * pk["logp"]
    m = np.maximum.reduceat(u, pk["starts"])
    m_flat = np.repeat(m, pk["sizes"])
    seg = np.add.reduceat(np.exp(u - m_flat), pk["starts"])
    lse = m + np.log(seg)
    return float(lse.sum() - u[pk["win_g"]].sum())


def fit_pk(pk, fix=None):
    """fix=None→(α,β); fix='b0'→β=0(fund-only); fix='a0'→α=0(market-only)."""
    if fix == "b0":
        r = minimize(lambda x: nll_packed(x[0], 0.0, pk), [1.0], method="Nelder-Mead")
        return r.x[0], 0.0
    if fix == "a0":
        r = minimize(lambda x: nll_packed(0.0, x[0], pk), [1.0], method="Nelder-Mead")
        return 0.0, r.x[0]
    r = minimize(lambda x: nll_packed(x[0], x[1], pk), [1.0, 1.0], method="Nelder-Mead")
    return r.x[0], r.x[1]


def mcfadden_r2_pk(pk, a, b):
    u = a * pk["logf"] + b * pk["logp"]
    m = np.maximum.reduceat(u, pk["starts"])
    m_flat = np.repeat(m, pk["sizes"])
    seg = np.add.reduceat(np.exp(u - m_flat), pk["starts"])
    lse = m + np.log(seg)
    ll_m = float(u[pk["win_g"]].sum() - lse.sum())
    ll_0 = float(-np.log(pk["sizes"]).sum())
    return 1.0 - ll_m / ll_0


def blend_p(r, a, b, pcol):
    u = a * r["logf"] + b * r[pcol]
    u = u - u.max()
    e = np.exp(u)
    return e / e.sum()


def bet_roi(races, a, b, pcol, ocol, thr):
    """EV=p_blend×odds(ocol) >= thr の馬を買い、確定oddsで決済。"""
    stake = ret = 0.0; bets = hits = 0
    clv = []; strat = {"nomove": [0.0, 0.0, 0], "small": [0.0, 0.0, 0], "big": [0.0, 0.0, 0]}
    for r in races:
        p = blend_p(r, a, b, pcol)
        od = r[ocol]
        ev = p * od
        sel = np.where(ev >= thr)[0]
        for i in sel:
            stake += 100.0; bets += 1
            won = (i == r["win"])
            payoff = r["oc"][i] * 100.0 if won else 0.0
            if won:
                ret += payoff; hits += 1
            mv = np.log(r["o9"][i]) - np.log(r["oc"][i])  # >0=短縮(steam)
            clv.append(mv)
            lab = "nomove" if abs(mv) < 0.05 else ("small" if abs(mv) < 0.15 else "big")
            strat[lab][0] += 100.0; strat[lab][1] += payoff; strat[lab][2] += 1
    roi = ret / stake if stake else 0.0
    strat_out = {k: {"roi": round(v[1] / v[0], 4) if v[0] else 0.0, "n": v[2]}
                 for k, v in strat.items()}
    return {"roi": round(roi, 4), "bets": bets,
            "hit_rate": round(hits / bets, 4) if bets else 0.0,
            "mean_clv": round(float(np.mean(clv)), 4) if clv else None,
            "by_move": strat_out}


def main():
    races = build_races()
    tr = [r for r in races if r["year"] <= 2023]
    te = [r for r in races if r["year"] >= 2024]
    te24 = [r for r in te if r["year"] == 2024]; te25 = [r for r in te if r["year"] == 2025]
    print(f"  fit(≤2023)={len(tr):,}  test={len(te):,} (2024={len(te24):,} 2025={len(te25):,})")

    out = {"n_races": len(races), "n_fit": len(tr), "n_test": len(te),
           "ev_grid": EV_GRID, "takeout_floor": 0.80, "variants": {}}
    rng = np.random.default_rng(42)

    for vname, pcol, ocol in [("A_9am", "logp9", "o9"), ("B_final", "logpc", "oc")]:
        print(f"\n===== 変種 {vname} (π={pcol}) =====")
        pk_tr = pack(tr, pcol); pk_te = pack(te, pcol)
        a, b = fit_pk(pk_tr)
        af, _ = fit_pk(pk_tr, fix="b0")     # fundamental-only
        _, bm = fit_pk(pk_tr, fix="a0")     # market-only
        print(f"  α={a:.4f}  β={b:.4f}  (fund-only α={af:.3f}, market-only β={bm:.3f})")

        # bootstrap CI for α (≤2023, race-level)
        boot_a = []; boot_b = []
        idx = np.arange(len(tr))
        for _ in range(N_BOOT):
            samp = [tr[i] for i in rng.choice(idx, len(idx), replace=True)]
            ba, bb = fit_pk(pack(samp, pcol))
            boot_a.append(ba); boot_b.append(bb)
        a_ci = [round(float(np.percentile(boot_a, 2.5)), 4), round(float(np.percentile(boot_a, 97.5)), 4)]
        b_ci = [round(float(np.percentile(boot_b, 2.5)), 4), round(float(np.percentile(boot_b, 97.5)), 4)]
        a_sig = (a_ci[0] > 0) or (a_ci[1] < 0)
        print(f"  α 95%CI={a_ci} 有意={a_sig}   β 95%CI={b_ci}")

        # ΔR² (McFadden) on test
        r2_blend = mcfadden_r2_pk(pk_te, a, b)
        r2_mkt = mcfadden_r2_pk(pk_te, 0.0, bm)
        r2_fund = mcfadden_r2_pk(pk_te, af, 0.0)
        dr2 = r2_blend - r2_mkt
        print(f"  R²(test): blend={r2_blend:.4f} market-only={r2_mkt:.4f} fund-only={r2_fund:.4f}  ΔR²(vs市場)={dr2:.4f}")

        # ROI: EV sweep tune on ≤2023, eval test
        sweep23 = {str(t): bet_roi(tr, a, b, pcol, ocol, t) for t in EV_GRID}
        cand = [(t, sweep23[str(t)]) for t in EV_GRID if sweep23[str(t)]["bets"] >= MIN_TUNE_BETS]
        tstar = max(cand, key=lambda x: x[1]["roi"])[0] if cand else EV_GRID[0]
        roi_test = bet_roi(te, a, b, pcol, ocol, tstar)
        roi_24 = bet_roi(te24, a, b, pcol, ocol, tstar)
        roi_25 = bet_roi(te25, a, b, pcol, ocol, tstar)
        print(f"  τ*={tstar} (≤2023 ROI={sweep23[str(tstar)]['roi']*100:.1f}%) → "
              f"test ROI={roi_test['roi']*100:.1f}% (2024={roi_24['roi']*100:.1f}% 2025={roi_25['roi']*100:.1f}%) "
              f"n={roi_test['bets']:,} CLV={(roi_test['mean_clv'] or 0)*100:+.1f}%")
        print(f"  変動層別 ROI(test, τ*): " +
              "  ".join(f"{k}={v['roi']*100:.0f}%(n{v['n']})" for k, v in roi_test["by_move"].items()))

        out["variants"][vname] = {
            "alpha": round(a, 4), "beta": round(b, 4),
            "alpha_ci95": a_ci, "alpha_sig": bool(a_sig), "beta_ci95": b_ci,
            "fund_only_alpha": round(af, 4), "market_only_beta": round(bm, 4),
            "r2_test": {"blend": round(r2_blend, 5), "market_only": round(r2_mkt, 5),
                        "fund_only": round(r2_fund, 5), "delta_r2_vs_market": round(dr2, 5)},
            "tau_star": tstar,
            "roi": {"le2023": sweep23[str(tstar)], "test": roi_test,
                    "test_2024": roi_24, "test_2025": roi_25},
            "ev_sweep_le2023": sweep23,
        }

    outp = BASE / "reports/stage1_benter_blend.json"
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {outp}")


if __name__ == "__main__":
    main()
