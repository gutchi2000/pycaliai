"""
gate1_eval1.py
==============
ゲート1 評価1: 相関補正 M1 が オッズ素朴PL M0 を連系キャリブレーションで上回るか。

marginal はオッズ(系3 de-vig)に固定。joint 構造だけ M0(素朴PL/Harville) vs
M1(=normalize(M0 × exp(β·corr))) で比較。参考に v6 PL+Isotonic joint も測る。

β は umaren ペアの条件付きロジット(offset=log M0)を train(≤2022)でMLEフィット。
同じ β を pair 単位で umaren / wide / sanrenpuku(トリオは3ペアの和) に適用。

出力: reports/gate1_corr_calibration.json
実行: python gate1_eval1.py
"""
from __future__ import annotations
import io, json, sys, warnings
from datetime import datetime
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

import pl_probs as PL
from backtest_pl_ev import all_umaren_mat, all_wide_mat, all_sanrenpuku_tensor
from corr_features import add_style, PAIR_FEATURE_NAMES
from joint_calibration_v6 import new_acc, acc_add, acc_summary, BINS, apply_encoders

warnings.filterwarnings("ignore")
# stdout は import 済 joint_calibration_v6 側で一度だけ utf-8 wrap 済（二重wrap回避）。
BASE = Path(__file__).parent
np.random.seed(42)

MASTER = BASE / "data/master_v2_20130105-20251228.csv"
KEKKA_V2 = Path(r"E:/競馬過去走データ/kekka_20130105-20251228_v2.csv")
COL_RID = "レースID(新/馬番無)"; COL_JYUN = "着順"; COL_BAN = "馬番"
K = len(PAIR_FEATURE_NAMES)
TPCOL = 19  # 系3 発走30分前 単勝（前売り, leak-safe）


def pair_feat_tensor(style, waku, pace, n):
    """(n,n,K) 対称ペア特徴テンソル。style NaN→0.5。"""
    s = np.where(np.isnan(style), 0.5, style)
    si = s[:, None]; sj = s[None, :]
    both_front = ((si < 0.30) & (sj < 0.30)).astype(float)
    both_closer = ((si >= 0.70) & (sj >= 0.70)).astype(float)
    lo = np.minimum(si, sj); hi = np.maximum(si, sj)
    front_closer = ((lo < 0.30) & (hi >= 0.55)).astype(float)
    style_gap = np.abs(si - sj)
    bf_pace = both_front * float(pace)
    w = waku.astype(float)
    draw_gap = np.abs(w[:, None] - w[None, :]) / 8.0
    F = np.stack([both_front, both_closer, front_closer, style_gap, bf_pace, draw_gap], axis=-1)
    return F  # (n,n,K)


def load_odds_devig():
    """外部 kekka_v2 の系3単勝から (rid16,umaban)->odds。"""
    k = pd.read_csv(KEKKA_V2, encoding="cp932", low_memory=False)
    kc = list(k.columns)
    rid = k[kc[7]].astype(str).str.replace(r"\D", "", regex=True).str[:16]
    ban = pd.to_numeric(k[kc[4]], errors="coerce")
    od = pd.to_numeric(k[kc[TPCOL]], errors="coerce")
    out = {}
    for r, b, o in zip(rid.values, ban.values, od.values):
        if pd.isna(b) or pd.isna(o) or o <= 1.0:
            continue
        out[(r, int(b))] = float(o)
    return out


def build_records(df, odds_map):
    """レースごとに dict(m, style, waku, jyun, score, pace, split) を作る。全頭odds必須。"""
    recs = []
    for rid, g in df.groupby(COL_RID, sort=False):
        if len(g) < 5:
            continue
        g = g.sort_values(COL_BAN).reset_index(drop=True)
        rid16 = str(rid)[:16]
        ban = g[COL_BAN].astype(int).values
        odds = np.array([odds_map.get((rid16, int(b)), np.nan) for b in ban])
        if np.any(~np.isfinite(odds)) or np.any(odds <= 1.0):
            continue
        inv = 1.0 / odds
        m = inv / inv.sum()  # de-vig market marginal
        recs.append({
            "m": m,
            "style": g["style_rank"].values.astype(float),
            "waku": pd.to_numeric(g["枠番"], errors="coerce").fillna(0).values,
            "jyun": g[COL_JYUN].astype(int).values,
            "score": g["_score"].values.astype(float),
            "pace": float(g["pace_pressure"].iloc[0]),
            "split": g["split"].iloc[0],
            "ban": ban,
            "rid16": rid16,
        })
    return recs


def fit_beta_umaren(train_recs, max_races=12000):
    """umaren ペア条件付きロジット(offset=log M0)で β を MLE。"""
    rng = np.random.default_rng(42)
    if len(train_recs) > max_races:
        idx = rng.choice(len(train_recs), max_races, replace=False)
        recs = [train_recs[i] for i in idx]
    else:
        recs = train_recs
    data = []
    for r in recs:
        m = r["m"]; n = len(m)
        M0 = all_umaren_mat(m)
        iu, ju = np.triu_indices(n, 1)
        lm0 = np.log(M0[iu, ju] + 1e-300)
        F = pair_feat_tensor(r["style"], r["waku"], r["pace"], n)[iu, ju]  # (P,K)
        order = np.argsort(r["jyun"]); win, plc = int(order[0]), int(order[1])
        wp = {min(win, plc), max(win, plc)}
        win_idx = next((k for k in range(len(iu)) if {int(iu[k]), int(ju[k])} == wp), None)
        if win_idx is None:
            continue
        data.append((lm0, F, win_idx))

    def nll(beta):
        tot = 0.0
        for lm0, F, wi in data:
            u = lm0 + F @ beta
            tot += logsumexp(u) - u[wi]
        return tot

    res = minimize(nll, np.zeros(K), method="L-BFGS-B",
                   options={"maxiter": 200})
    return res.x, len(data)


def joint_probs(rec, beta, kind):
    """rec から (combo_probs M0, M1, labels, comp, style_pair_flags) を返す。
    kind in {umaren, wide, sanrenpuku}."""
    m = rec["m"]; n = len(m)
    F = pair_feat_tensor(rec["style"], rec["waku"], rec["pace"], n)  # (n,n,K)
    corr = F @ beta  # (n,n) log-correction per pair
    order = np.argsort(rec["jyun"]); win, plc, sho = int(order[0]), int(order[1]), int(order[2])
    top2 = {win, plc}; top3 = {win, plc, sho}
    sr = np.where(np.isnan(rec["style"]), 0.5, rec["style"])

    if kind == "umaren":
        M0mat = all_umaren_mat(m); iu, ju = np.triu_indices(n, 1)
        p0 = M0mat[iu, ju]
        c = np.exp(corr[iu, ju])
        p1 = p0 * c; p1 = p1 / p1.sum() * p0.sum()
        y = np.array([1.0 if {int(iu[k]), int(ju[k])} == top2 else 0.0 for k in range(len(iu))])
        bf = ((sr[iu] < 0.30) & (sr[ju] < 0.30)).astype(int)  # 両前フラグ（層別用）
        return p0, p1, y, bf
    if kind == "wide":
        Wmat = all_wide_mat(m); iu, ju = np.triu_indices(n, 1)
        p0 = Wmat[iu, ju]
        c = np.exp(corr[iu, ju])
        p1 = p0 * c; p1 = p1 / p1.sum() * p0.sum()
        y = np.array([1.0 if {int(iu[k]), int(ju[k])}.issubset(top3) else 0.0 for k in range(len(iu))])
        bf = ((sr[iu] < 0.30) & (sr[ju] < 0.30)).astype(int)
        return p0, p1, y, bf
    # sanrenpuku: トリオ = 3ペアの corr 和
    P3 = all_sanrenpuku_tensor(m)
    tris = np.array(list(combinations(range(n), 3)))
    from itertools import permutations
    p0 = np.zeros(len(tris))
    for prm in permutations(range(3)):
        p0 += P3[tris[:, prm[0]], tris[:, prm[1]], tris[:, prm[2]]]
    tri_corr = corr[tris[:, 0], tris[:, 1]] + corr[tris[:, 0], tris[:, 2]] + corr[tris[:, 1], tris[:, 2]]
    c = np.exp(tri_corr)
    p1 = p0 * c; p1 = p1 / p1.sum() * p0.sum()
    y = np.array([1.0 if set(int(x) for x in t) == top3 else 0.0 for t in tris])
    nfront = ((sr[tris[:, 0]] < 0.30).astype(int) + (sr[tris[:, 1]] < 0.30).astype(int) + (sr[tris[:, 2]] < 0.30).astype(int))
    return p0, p1, y, nfront


def eval_test(test_recs, beta, v6cal):
    bets = ["umaren", "wide", "sanrenpuku"]
    acc = {bt: {"M0": new_acc(), "M1": new_acc(), "v6": new_acc()} for bt in bets}
    # 層別 (umaren): pace 層 × 両前フラグ
    strat = {f"pace_{lab}": {"M0": new_acc(), "M1": new_acc()} for lab in ["lo", "mid", "hi"]}
    strat_bf = {"both_front_pairs": {"M0": new_acc(), "M1": new_acc()},
                "other_pairs": {"M0": new_acc(), "M1": new_acc()}}
    for rec in test_recs:
        # v6 joint (参考): scores -> PL -> Isotonic
        w6 = PL.pl_weights(rec["score"]); n = len(w6)
        iu, ju = np.triu_indices(n, 1)
        pace_lab = "lo" if rec["pace"] < 0.25 else ("hi" if rec["pace"] > 0.40 else "mid")
        for bt in bets:
            p0, p1, y, flag = joint_probs(rec, beta, bt)
            acc_add(acc[bt]["M0"], p0, y); acc_add(acc[bt]["M1"], p1, y)
            # v6 reference
            if bt == "umaren":
                v = v6cal.get("umaren")
                pv = all_umaren_mat(w6)[iu, ju]
            elif bt == "wide":
                v = v6cal.get("wide"); pv = all_wide_mat(w6)[iu, ju]
            else:
                v = v6cal.get("sanrenpuku")
                P3 = all_sanrenpuku_tensor(w6); tris = np.array(list(combinations(range(n), 3)))
                from itertools import permutations
                pv = np.zeros(len(tris))
                for prm in permutations(range(3)):
                    pv += P3[tris[:, prm[0]], tris[:, prm[1]], tris[:, prm[2]]]
            pv_c = v.predict(pv) if v is not None else pv
            acc_add(acc[bt]["v6"], pv_c, y)
            # umaren 層別
            if bt == "umaren":
                acc_add(strat[f"pace_{pace_lab}"]["M0"], p0, y)
                acc_add(strat[f"pace_{pace_lab}"]["M1"], p1, y)
                bfm = flag.astype(bool)
                if bfm.any():
                    acc_add(strat_bf["both_front_pairs"]["M0"], p0[bfm], y[bfm])
                    acc_add(strat_bf["both_front_pairs"]["M1"], p1[bfm], y[bfm])
                if (~bfm).any():
                    acc_add(strat_bf["other_pairs"]["M0"], p0[~bfm], y[~bfm])
                    acc_add(strat_bf["other_pairs"]["M1"], p1[~bfm], y[~bfm])
    return acc, strat, strat_bf


def main():
    bundle = joblib.load(BASE / "models/unified_rank_v6.pkl")
    model, feats, encs = bundle["model"], bundle["feature_cols"], bundle["encoders"]
    cb = joblib.load(BASE / "models/pl_calibrators_v6.pkl")
    v6cal = cb.get("calibrators", cb)
    print("[load] master + style + odds(系3 de-vig)")
    df = pd.read_csv(MASTER, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]).copy()
    df = add_style(df)
    enc = apply_encoders(df, encs)
    X = enc[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    df["_score"] = model.predict(X)
    odds_map = load_odds_devig()
    print(f"  odds entries={len(odds_map):,}")

    recs = build_records(df, odds_map)
    tr = [r for r in recs if r["split"] == "train"]
    te = [r for r in recs if r["split"] == "test"]
    print(f"  races: train={len(tr):,}  test={len(te):,}  (全頭系3odds有)")

    print("[fit] β (umaren conditional logit, offset=log M0) on train ...")
    beta, nfit = fit_beta_umaren(tr)
    print(f"  fit races={nfit:,}")
    for nm, b in zip(PAIR_FEATURE_NAMES, beta):
        print(f"    β[{nm:18s}] = {b:+.4f}")

    print("[eval] test 2024-25: M0 vs M1 vs v6 ...")
    acc, strat, strat_bf = eval_test(te, beta, v6cal)

    def ece(a):
        s = acc_summary(a); return None if s is None else s

    out = {"generated_at": datetime.now().isoformat(timespec="seconds"),
           "odds_timepoint": "系3 発走30分前 (前売り, de-vig)",
           "beta": {nm: round(float(b), 4) for nm, b in zip(PAIR_FEATURE_NAMES, beta)},
           "n_fit_races": nfit, "n_test_races": len(te),
           "ece": {}, "strat_pace_umaren": {}, "strat_bothfront_umaren": {}}
    print("\n券種別 ECE (test 2024-25):")
    print(f"{'bet':<12s} {'M0':>9s} {'M1':>9s} {'v6(Iso)':>9s}  {'M1<M0?':>7s}")
    for bt in ["umaren", "wide", "sanrenpuku"]:
        e0 = ece(acc[bt]["M0"]); e1 = ece(acc[bt]["M1"]); ev = ece(acc[bt]["v6"])
        out["ece"][bt] = {"M0": e0["ece"], "M1": e1["ece"], "v6_iso": ev["ece"],
                          "agg_actual": e0["agg_actual"]}
        flag = "yes" if e1["ece"] < e0["ece"] else "no"
        print(f"{bt:<12s} {e0['ece']:>9.5f} {e1['ece']:>9.5f} {ev['ece']:>9.5f}  {flag:>7s}")
    print("\numaren ECE by pace 層:")
    for lab in ["lo", "mid", "hi"]:
        a = strat[f"pace_{lab}"]; e0 = ece(a["M0"]); e1 = ece(a["M1"])
        out["strat_pace_umaren"][lab] = {"M0": e0["ece"], "M1": e1["ece"], "n": e0["n"]}
        print(f"  pace={lab:3s}: M0={e0['ece']:.5f} M1={e1['ece']:.5f} (n={e0['n']:,})")
    print("\numaren ECE by 両前ペア:")
    for key in ["both_front_pairs", "other_pairs"]:
        a = strat_bf[key]; e0 = ece(a["M0"]); e1 = ece(a["M1"])
        out["strat_bothfront_umaren"][key] = {"M0": e0["ece"], "M1": e1["ece"], "n": e0["n"]}
        print(f"  {key:18s}: M0={e0['ece']:.5f} M1={e1['ece']:.5f} (n={e0['n']:,})")

    outp = BASE / "reports/gate1_corr_calibration.json"
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {outp}")


if __name__ == "__main__":
    main()
