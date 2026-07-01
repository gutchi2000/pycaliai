"""
task_a_stern_fit.py
===================
Task A: Stern べき乗補正 (pl_stern) の λ を valid 2023 で推定し、
test 2024-25 で点2の構成別 Harville bias が 1.0 へ平坦化するか確認する。

- λ 推定: valid 2023 の観測 top3 順序の NLL 最小化 (scipy minimize_scalar bounded)。
  単勝 marginal は λ 不変なので、実質 2着/3着の条件付き尤度を fit する。
- test 評価: 馬連/三連複の「構成馬のうちモデルtop3頭数」別 pred/actual を raw(λ=1) と
  corrected(λ*) で比較 + 連系の相対ECE 変化。
- 単勝の no-edge 結論は不変 (補正は連系 joint 限定、単勝確率は変えない)。
- 出力: reports/stern_correction_v6.json

実行: python task_a_stern_fit.py --model v6
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import warnings
from datetime import datetime
from itertools import combinations, permutations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

import pl_stern as ST
from joint_calibration_v6 import (
    new_acc, acc_add, acc_summary, new_harv, harv_add, harv_summary,
    apply_encoders, COL_RID, COL_JYUN, COL_BAN, MASTER_CSV,
)

warnings.filterwarnings("ignore")
# NOTE: stdout は import した joint_calibration_v6 側で一度だけ utf-8 wrap される。
# ここで再 wrap すると二重 wrap で buffer が閉じられるため行わない。

BASE = Path(__file__).parent
np.random.seed(42)

BETS = ["fukusho", "umaren", "wide", "umatan", "sanrenpuku"]


def cache_races(sub):
    """1レースごとに (scores, (win,plc,sho), model_top3_set) を作る。len>=5。"""
    out = []
    for rid, g in sub.groupby(COL_RID, sort=False):
        if len(g) < 5:
            continue
        g = g.sort_values(COL_BAN).reset_index(drop=True)
        jyun = g[COL_JYUN].astype(int).values
        sc = g["_score"].values.astype(np.float64)
        order = np.argsort(jyun)
        win, plc, sho = int(order[0]), int(order[1]), int(order[2])
        mt3 = set(int(x) for x in np.argsort(-sc)[:3])
        out.append((sc, (win, plc, sho), mt3))
    return out


def fit_lambda(valid_races):
    """valid の観測 top3 順序 NLL を最小化して λ を求める。"""
    ws = [(np.exp(sc - sc.max()), wps) for sc, wps, _ in valid_races]

    def nll(lam):
        tot = 0.0
        for w, (win, plc, sho) in ws:
            tot -= ST.top3_loglik(w, win, plc, sho, lam)
        return tot

    res = minimize_scalar(nll, bounds=(0.3, 1.6), method="bounded",
                          options={"xatol": 1e-3})
    lam = float(res.x)
    # 参考グリッド
    grid = {round(L, 2): round(nll(L), 2) for L in np.arange(0.5, 1.51, 0.1)}
    return lam, float(nll(1.0)), float(nll(lam)), grid


def measure(races, lam):
    acc = {bt: new_acc() for bt in BETS}
    harv = {"umaren": new_harv(2), "sanrenpuku": new_harv(3)}
    for sc, (win, plc, sho), mt3 in races:
        w = np.exp(sc - sc.max()); v = w ** lam
        n = len(w)
        top3 = {win, plc, sho}; top2 = {win, plc}
        iu, ju = np.triu_indices(n, k=1)

        # fukusho
        p = ST.all_fukusho_vec_stern(w, v)
        y = np.array([1.0 if i in top3 else 0.0 for i in range(n)])
        acc_add(acc["fukusho"], p, y)
        # umaren
        M = ST.all_umaren_mat_stern(w, v); p = M[iu, ju]
        y = np.array([1.0 if {int(iu[k]), int(ju[k])} == top2 else 0.0 for k in range(len(iu))])
        acc_add(acc["umaren"], p, y)
        cls = np.array([(int(iu[k]) in mt3) + (int(ju[k]) in mt3) for k in range(len(iu))])
        harv_add(harv["umaren"], cls, p, y)
        # wide
        Wd = ST.all_wide_mat_stern(w, v); p = Wd[iu, ju]
        y = np.array([1.0 if {int(iu[k]), int(ju[k])}.issubset(top3) else 0.0 for k in range(len(iu))])
        acc_add(acc["wide"], p, y)
        # umatan
        U = ST.all_umatan_mat_stern(w, v)
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        m = ii != jj; iif, jjf = ii[m], jj[m]; p = U[m]
        y = ((iif == win) & (jjf == plc)).astype(float)
        acc_add(acc["umatan"], p, y)
        # sanrenpuku
        P3 = ST.all_sanrenpuku_tensor_stern(w, v)
        tris = np.array(list(combinations(range(n), 3)))
        p_pk = np.zeros(len(tris))
        for perm in permutations(range(3)):
            p_pk += P3[tris[:, perm[0]], tris[:, perm[1]], tris[:, perm[2]]]
        y = np.array([1.0 if set(int(x) for x in t) == top3 else 0.0 for t in tris])
        acc_add(acc["sanrenpuku"], p_pk, y)
        cls = np.array([sum(int(x) in mt3 for x in t) for t in tris])
        harv_add(harv["sanrenpuku"], cls, p_pk, y)

    rel = {}
    for bt in BETS:
        s = acc_summary(acc[bt])
        rel[bt] = {"ece": s["ece"], "agg_actual": s["agg_actual"],
                   "rel_ece": round(s["ece"] / s["agg_actual"], 4) if s["agg_actual"] > 0 else None}
    harvs = {"umaren": harv_summary(harv["umaren"]),
             "sanrenpuku": harv_summary(harv["sanrenpuku"])}
    return rel, harvs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v6")
    args = ap.parse_args()
    tag = args.model

    bundle = joblib.load(BASE / f"models/unified_rank_{tag}.pkl")
    model, feats, encs = bundle["model"], bundle["feature_cols"], bundle["encoders"]
    print(f"[model] unified_rank_{tag}.pkl feats={len(feats)}")

    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]).copy()

    def score(split):
        s = df[df["split"] == split].copy()
        s = apply_encoders(s, encs)
        X = s[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
        s["_score"] = model.predict(X)
        return s

    print("[score] valid 2023")
    valid_races = cache_races(score("valid"))
    print(f"  races={len(valid_races):,}")
    print("[fit] λ on valid (top3 order NLL)...")
    lam, nll1, nllL, grid = fit_lambda(valid_races)
    print(f"  λ* = {lam:.4f}   NLL(λ=1)={nll1:.1f}  NLL(λ*)={nllL:.1f}  ΔNLL={nll1-nllL:.1f}")

    print("[score] test 2024-25")
    test_races = cache_races(score("test"))
    print(f"  races={len(test_races):,}")
    print("[measure] raw (λ=1) ...")
    rel_raw, harv_raw = measure(test_races, 1.0)
    print(f"[measure] corrected (λ={lam:.3f}) ...")
    rel_cor, harv_cor = measure(test_races, lam)

    payload = {
        "model_tag": tag,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "method": "Stern/Lo-Bacon-Shone power correction: 1st pick raw w, 2nd+ uses v=w^λ. Win marginal invariant.",
        "lambda": round(lam, 4),
        "lambda_fit": {"objective": "top3-order NLL on valid 2023",
                       "nll_lambda1": round(nll1, 2), "nll_lambdastar": round(nllL, 2),
                       "delta_nll": round(nll1 - nllL, 2), "grid_nll": grid,
                       "n_valid_races": len(valid_races)},
        "test_2024_2025": {
            "n_races": len(test_races),
            "harville_umaren_by_modeltop3_members": {"raw": harv_raw["umaren"], "corrected": harv_cor["umaren"]},
            "harville_sanrenpuku_by_modeltop3_members": {"raw": harv_raw["sanrenpuku"], "corrected": harv_cor["sanrenpuku"]},
            "relative_ece": {bt: {"raw": rel_raw[bt], "corrected": rel_cor[bt]} for bt in BETS},
        },
    }
    out = BASE / f"reports/stern_correction_{tag}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[saved] {out}")

    # 読みやすいサマリ
    print("\n" + "=" * 64)
    print(f"λ* = {lam:.4f}  (λ<1 ほど下位順位を平坦化)")
    print("\n馬連 Harville (構成馬のモデルtop3頭数別 pred/actual):")
    print(f"{'members':>8s} {'raw':>10s} {'corrected':>10s}")
    for k in sorted(harv_raw["umaren"].keys()):
        r = harv_raw["umaren"][k]["ratio_pred_over_actual"]
        c = harv_cor["umaren"][k]["ratio_pred_over_actual"]
        print(f"{k:>8s} {r:>10.4f} {c:>10.4f}")
    print("\n三連複 Harville (構成馬のモデルtop3頭数別 pred/actual):")
    print(f"{'members':>8s} {'raw':>10s} {'corrected':>10s}")
    for k in sorted(harv_raw["sanrenpuku"].keys()):
        r = harv_raw["sanrenpuku"][k]["ratio_pred_over_actual"]
        c = harv_cor["sanrenpuku"][k]["ratio_pred_over_actual"]
        print(f"{k:>8s} {r:>10.4f} {c:>10.4f}")
    print("\n相対ECE (ECE/平均的中率) raw → corrected:")
    for bt in BETS:
        print(f"  {bt:12s} {rel_raw[bt]['rel_ece']} → {rel_cor[bt]['rel_ece']}")


if __name__ == "__main__":
    main()
