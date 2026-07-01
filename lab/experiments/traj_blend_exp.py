"""
traj_blend_exp.py — (b) オッズ軌跡の市場増分 (9時点オッズ二段合成への第3成分)
================================================================================
9時点オッズ π9 で二段合成 softmax(α logf + β logπ9) した後、さらに 9時以前軌跡 traj を
第3成分 γ·traj_z として足すと ΔR² が +0.007 から動くか / ROI・CLV が動くかを測る。
軌跡が点オッズに対し増分を持つなら ΔR² 上昇。持たないなら点オッズに吸収済。

★リーク: traj は build_odds_traj_feats(9時以前のみ, 検査済) を流用。π9=9時de-vig, 決済=確定。
  fit は ≤2023, test=2024-25 評価 (stage1b と同規律)。
判定(事前固定): ΔR²(3成分) が ΔR²(2成分=+0.007) から有意上昇(γ CI が0除外) かつ CLV/ROI 改善 → 軌跡は市場増分。
出力: reports/traj_blend_exp.json  実行: PYTHONUTF8=1 python traj_blend_exp.py
"""
from __future__ import annotations
import json, warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import pl_probs as PL
import stage1_benter_blend as S1
from joint_calibration_v6 import apply_encoders, COL_RID, COL_JYUN, COL_BAN, MASTER_CSV

warnings.filterwarnings("ignore")
BASE = Path(__file__).parent
TRAJ_PQ = BASE / "data/odds_traj_feats.parquet"
OUT_JSON = BASE / "reports/traj_blend_exp.json"
np.random.seed(42)
ALPHA_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
EV_GRID = [0.9, 1.0, 1.05, 1.1, 1.2, 1.3, 1.5]
N_BOOT = 120
MIN_TUNE = 200


def build_races_traj():
    snaps = S1.load_tanpuk_snapshots()
    b = joblib.load(BASE / "models/unified_rank_v6.pkl")
    model, feats, encs = b["model"], b["feature_cols"], b["encoders"]
    cal = joblib.load(BASE / "models/pl_calibrators_v6.pkl"); cal = cal.get("calibrators", cal)
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]).copy()
    enc = apply_encoders(df, encs)
    X = enc[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    df["_score"] = model.predict(X)
    # traj lookup
    tj = pd.read_parquet(TRAJ_PQ)
    traj = {(str(r["rid16"]), int(r["ban"])): r["traj_logchg"] for _, r in tj.iterrows()
            if pd.notna(r.get("traj_logchg"))}
    races = []
    for rid, g in df.groupby(COL_RID, sort=False):
        if len(g) < 5: continue
        rid16 = str(rid)[:16]
        if rid16 not in snaps: continue
        g = g.sort_values(COL_BAN).reset_index(drop=True)
        ban = g[COL_BAN].astype(int).values
        w = PL.pl_weights(g["_score"].values)
        f_all = cal["tansho"].predict(PL.all_tansho(w)) if "tansho" in cal else PL.all_tansho(w)
        f_by = {int(ban[i]): max(float(f_all[i]), 1e-9) for i in range(len(ban))}
        jy = g[COL_JYUN].astype(int).values; win_ban = int(ban[int(np.argmin(jy))])
        sp = snaps[rid16]
        common = [k for k in sp["o9"] if k in sp["oc"] and k in f_by]
        if win_ban not in common or len(common) < 5: continue
        keys = sorted(common)
        f = np.array([f_by[k] for k in keys]); p9 = S1.devig(sp["o9"], keys)
        o9 = np.array([sp["o9"][k] for k in keys]); oc = np.array([sp["oc"][k] for k in keys])
        tr = np.array([traj.get((rid16, k), np.nan) for k in keys])
        races.append({"logf": np.log(f), "logp9": np.log(p9), "o9": o9, "oc": oc,
                      "win": keys.index(win_ban), "year": int(rid16[:4]), "traj_raw": tr})
    # train(≤2022) で traj 標準化、NaN→0 (中立)
    tr_vals = np.concatenate([r["traj_raw"][~np.isnan(r["traj_raw"])] for r in races if r["year"] <= 2022])
    mu, sd = float(np.nanmean(tr_vals)), float(np.nanstd(tr_vals) or 1.0)
    for r in races:
        z = (r["traj_raw"] - mu) / sd
        r["traj"] = np.nan_to_num(z, nan=0.0)
    cov = float(np.mean([np.mean(~np.isnan(r["traj_raw"])) for r in races]))
    print(f"  races={len(races):,}  traj標準化(mu={mu:.3f},sd={sd:.3f})  traj非欠損率={cov*100:.1f}%")
    return races


def pack(races):
    sizes = np.array([len(r["logf"]) for r in races]); starts = np.zeros(len(races), int)
    if len(races) > 1: starts[1:] = np.cumsum(sizes)[:-1]
    return {"logf": np.concatenate([r["logf"] for r in races]),
            "logp": np.concatenate([r["logp9"] for r in races]),
            "traj": np.concatenate([r["traj"] for r in races]),
            "starts": starts, "sizes": sizes,
            "win_g": np.array([starts[i] + races[i]["win"] for i in range(len(races))])}


def _lse(u, pk):
    m = np.maximum.reduceat(u, pk["starts"]); mf = np.repeat(m, pk["sizes"])
    seg = np.add.reduceat(np.exp(u - mf), pk["starts"]); return m + np.log(seg)


def nll(params, pk, mode):
    a, b, c = params if mode == "abc" else (params[0], params[1], 0.0) if mode == "ab" else (0.0, params[0], 0.0)
    u = a * pk["logf"] + b * pk["logp"] + c * pk["traj"]
    return float(_lse(u, pk).sum() - u[pk["win_g"]].sum())


def fit(pk, mode):
    x0 = {"abc": [1.0, 1.0, 0.0], "ab": [1.0, 1.0], "m": [1.0]}[mode]
    r = minimize(lambda x: nll(x, pk, mode), x0, method="Nelder-Mead")
    if mode == "abc": return float(r.x[0]), float(r.x[1]), float(r.x[2])
    if mode == "ab": return float(r.x[0]), float(r.x[1]), 0.0
    return 0.0, float(r.x[0]), 0.0


def r2(pk, a, b, c):
    u = a * pk["logf"] + b * pk["logp"] + c * pk["traj"]
    ll_m = float(u[pk["win_g"]].sum() - _lse(u, pk).sum()); ll_0 = float(-np.log(pk["sizes"]).sum())
    return 1.0 - ll_m / ll_0


def blend_p(r, a, b, c):
    u = a * r["logf"] + b * r["logp9"] + c * r["traj"]; u -= u.max(); e = np.exp(u); return e / e.sum()


def bet_roi(races, a, b, c, thr):
    stake = ret = 0.0; bets = hits = 0; clv = []
    for r in races:
        p = blend_p(r, a, b, c); ev = p * r["o9"]; sel = np.where(ev >= thr)[0]
        for i in sel:
            stake += 100; bets += 1; won = (i == r["win"])
            if won: ret += r["oc"][i] * 100; hits += 1
            clv.append(np.log(r["o9"][i]) - np.log(r["oc"][i]))
    return {"roi": round(ret / stake, 4) if stake else 0.0, "bets": bets,
            "hit": round(hits / bets, 4) if bets else 0.0,
            "clv": round(float(np.mean(clv)), 4) if clv else None}


def main():
    print("[build] races + traj ...")
    races = build_races_traj()
    tr = [r for r in races if r["year"] <= 2023]; te = [r for r in races if r["year"] >= 2024]
    pk_tr, pk_te = pack(tr), pack(te)
    print(f"  fit(≤2023)={len(tr):,} test={len(te):,}")

    _, bm, _ = fit(pk_tr, "m"); r2_mkt = r2(pk_te, 0.0, bm, 0.0)
    a2, b2, _ = fit(pk_tr, "ab"); r2_2 = r2(pk_te, a2, b2, 0.0)
    a3, b3, c3 = fit(pk_tr, "abc"); r2_3 = r2(pk_te, a3, b3, c3)
    dr2_2 = r2_2 - r2_mkt; dr2_3 = r2_3 - r2_mkt; incr = r2_3 - r2_2

    # γ bootstrap CI (≤2023 race resample)
    rng = np.random.default_rng(42); idx = np.arange(len(tr)); gs = []
    for _ in range(N_BOOT):
        samp = [tr[i] for i in rng.choice(idx, len(idx), replace=True)]
        _, _, gg = fit(pack(samp), "abc"); gs.append(gg)
    g_ci = [round(float(np.percentile(gs, 2.5)), 4), round(float(np.percentile(gs, 97.5)), 4)]
    g_sig = bool(g_ci[0] > 0 or g_ci[1] < 0)

    # ROI: τ を ≤2023 で選定 (2成分と3成分)
    def sel_tau(a, b, c):
        best = (EV_GRID[0], -9)
        for t in EV_GRID:
            rr = bet_roi(tr, a, b, c, t)
            if rr["bets"] >= MIN_TUNE and rr["roi"] > best[1]: best = (t, rr["roi"])
        return best[0]
    t2 = sel_tau(a2, b2, 0.0); t3 = sel_tau(a3, b3, c3)
    roi2 = bet_roi(te, a2, b2, 0.0, t2); roi3 = bet_roi(te, a3, b3, c3, t3)

    if g_sig and incr >= 0.002 and (roi3["clv"] or 0) > (roi2["clv"] or 0):
        verdict = (f"MARKET_INCREMENT: γ有意(CI={g_ci}) & ΔR²増分={incr:+.5f} & CLV改善。"
                   "軌跡は点オッズに対し市場増分を持つ。最優先で深掘り。")
    elif g_sig and incr >= 0.002:
        verdict = f"WEAK_INCREMENT: γ有意(CI={g_ci}) & ΔR²増分={incr:+.5f} だがCLV非改善。要精査。"
    else:
        verdict = (f"REDUNDANT_(b): γ CI={g_ci}(有意={g_sig}) ΔR²増分={incr:+.5f}(≈0)。"
                   "軌跡は9時点オッズに吸収済=市場増分なし。この弾も冗長。")

    out = {"protocol": "f=v6較正単勝, π9=9時de-vig, traj=9時以前軌跡(logchg,train標準化). fit≤2023/test2024-25",
           "r2_market_only_test": round(r2_mkt, 5),
           "2成分_f_pi9": {"alpha": round(a2, 4), "beta": round(b2, 4), "dR2_test": round(dr2_2, 5),
                          "tau": t2, "roi_test": roi2["roi"], "clv": roi2["clv"], "bets": roi2["bets"]},
           "3成分_f_pi9_traj": {"alpha": round(a3, 4), "beta": round(b3, 4), "gamma": round(c3, 4),
                               "gamma_ci95": g_ci, "gamma_sig": g_sig, "dR2_test": round(dr2_3, 5),
                               "tau": t3, "roi_test": roi3["roi"], "clv": roi3["clv"], "bets": roi3["bets"]},
           "traj_increment_dR2": round(incr, 5), "verdict": verdict}
    OUT_JSON.parent.mkdir(exist_ok=True)
    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\n" + "=" * 76)
    print(f"market-only R²(test)={r2_mkt:.5f}")
    print(f"2成分(f,π9):      α={a2:.3f} β={b2:.3f}  ΔR²={dr2_2:+.5f}  ROI={roi2['roi']*100:.1f}% CLV={(roi2['clv'] or 0)*100:+.1f}%")
    print(f"3成分(+traj):     α={a3:.3f} β={b3:.3f} γ={c3:+.4f}  ΔR²={dr2_3:+.5f}  ROI={roi3['roi']*100:.1f}% CLV={(roi3['clv'] or 0)*100:+.1f}%")
    print(f"γ 95%CI={g_ci} 有意={g_sig}   ★traj増分 ΔR²={incr:+.5f}")
    print(f"判定: {verdict}")
    print(f"[saved] {OUT_JSON}")


if __name__ == "__main__":
    main()
