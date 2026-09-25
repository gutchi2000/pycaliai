# -*- coding: utf-8 -*-
"""
anchor_le2018.py — EXP18 S0-A: 2018 年以前だけで crux_joint の T0 頭対頭を再現する anchor と、
                   Stage 0 の dry-run パラメータ fit (γ・λ)
======================================================================================
結果 loader は **max(year) <= 2018 を hard assert** する。2019 年以降の着順・払戻・realized top2 は読まない。
T0 頭対頭は既実施 (crux_joint.py: 9 時、λ fit<=2023、2024/25 評価、市場 3.343 < Harville 3.380) であり、
ここでは**再現 anchor** としてだけ扱い、新規成果とは主張しない。

事前に固定した anchor 規則 (実行前に固定):
  snapshot   9 時 (区分1・レース当日・09:00 に最も近い 1 本)。TANPUK と UMAREN は同一 月日時分 で結合
  race set   base race set ∩ 9 時 snapshot が両プールにある ∩ 9 時 UMAREN 格子が完全
             ∩ DNF なし ∩ 1着・2着に同着なし ∩ realized top2 が 9 時 starter 内
  starter    9 時の単勝オッズ > 1.0 かつ terminal でも starter (締切まで残った馬)
  市場       9 時 UMAREN の比例 de-vig (crux と同じ)
  Harville   9 時 TANSHO の比例 de-vig π から λ 割引 Harville。λ は crux と同じ格子 {0.5..1.2}
  主判定     λ を 2013-2016 で fit し 2017-2018 で評価 (≤2018 内の OOS)
             Δ = LL(Harville) − LL(市場)。要求: Δ > 0 (市場優位) かつ Δ ∈ [0.0185, 0.074]
  参考       ≤2018 全体 in-sample、年別
dry-run fit (≤2018, terminal, 同じ除外規則):
  γ          terminal UMAREN の power-law de-vig m ∝ (1/o)^γ を realized top2 logloss で fit
  λ_T1       terminal TANSHO の λ 割引 Harville を realized top2 logloss で fit (連続値)
出力: out/anchor_le2018.json
実行: python -m analysis.mcond.exp18_cross_pool_market_tomography_dev.anchor_le2018
"""
from __future__ import annotations

import json
import time

import numpy as np
from scipy.optimize import minimize_scalar

from . import tomography as T
from .loaders import OUT, RESULT_MAX_YEAR, dnf_horses, load_outcomes, load_structure, realized_top2
from .market_build import (PAIR_POS, base_race_status, pool_matrices, race_arrays, snapshot_index,
                           umaren_grid_complete)

YEARS = list(range(2013, RESULT_MAX_YEAR + 1))
LAMBDA_GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2)
FIT_YEARS = (2013, 2016)
EVAL_YEARS = (2017, 2018)
DELTA_RANGE = (0.0185, 0.074)


def build_anchor_races(st, outc_top):
    tan, um, info = st["tan"], st["um"], st["info"]
    W, LO, HI, U = pool_matrices(tan, um)
    idx = snapshot_index(tan, um, info)
    top = outc_top.set_index("rid16")
    am9, term, funnel = [], [], {}

    def bump(k):
        funnel[k] = funnel.get(k, 0) + 1

    for rid, row in idx.iterrows():
        if not (row.get("term_tan") == row.get("term_tan")):
            bump("no_terminal"); continue
        at = race_arrays(W, LO, HI, U, int(row["term_tan"]), int(row["term_um"]))
        inf = info.loc[rid] if rid in info.index else None
        stt = base_race_status(inf, at)
        if stt != "base":
            bump(stt); continue
        if rid not in top.index:
            bump("no_outcome"); continue
        o = top.loc[rid]
        if dnf_horses(at["bans"], o["finishers"]):
            bump("excl_dnf"); continue
        if o["dead_heat_top2"] or o["top2"] is None:
            bump("excl_dead_heat_top2"); continue
        y = int(rid[:4])
        # terminal (dry-run fit 用)
        if umaren_grid_complete(at) and set(o["top2"]) <= set(at["bans"].tolist()):
            term.append({"rid": rid, "year": y, "bans": at["bans"], "win": at["win"],
                         "umaren": at["umaren"], "top2": o["top2"]})
        # 9 時 anchor
        a9t, a9u = row.get("am9_tan"), row.get("am9_um")
        if not (a9t == a9t) or int(a9u) < 0:
            bump("anchor_no_am9_both_pools"); continue
        a = race_arrays(W, LO, HI, U, int(a9t), int(a9u))
        keep = np.isin(a["bans"], at["bans"])
        bans = a["bans"][keep]
        if len(bans) < 5 or not set(o["top2"]) <= set(bans.tolist()):
            bump("anchor_top2_outside_am9_starters"); continue
        # 9 時 starter ∩ terminal starter 上で組み直す
        win9 = a["win"][keep]
        n = len(bans)
        ia, ib = np.triu_indices(n, k=1)
        uo = U[int(a9u), [PAIR_POS[(int(bans[x]), int(bans[z]))] for x, z in zip(ia, ib)]]
        if not (np.all(np.isfinite(uo)) and np.all(uo >= 1.0)):
            bump("anchor_am9_grid_incomplete"); continue
        am9.append({"rid": rid, "year": y, "bans": bans, "win": win9, "umaren": uo, "top2": o["top2"]})
        bump("anchor_eligible")
    return am9, term, funnel


def pair_pos(bans, top2):
    ia, ib = np.triu_indices(len(bans), k=1)
    a, b = np.searchsorted(bans, top2[0]), np.searchsorted(bans, top2[1])
    return int(np.flatnonzero((ia == a) & (ib == b))[0])


def ll_harville(races, lam):
    out = []
    for r in races:
        pi = (1 / r["win"]) / (1 / r["win"]).sum()
        q = T.stern_top2(pi, lam)
        out.append(-np.log(q[pair_pos(r["bans"], r["top2"])]))
    return np.array(out)


def ll_market(races, gamma=1.0):
    return np.array([-np.log(T.devig_power(r["umaren"], gamma)[pair_pos(r["bans"], r["top2"])])
                     for r in races])


def main():
    import sys
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    t0 = time.time()
    st = load_structure(YEARS)
    outc = load_outcomes(RESULT_MAX_YEAR)
    assert int(outc["year"].max()) <= RESULT_MAX_YEAR
    top = realized_top2(outc)
    am9, term, funnel = build_anchor_races(st, top)
    fit = [r for r in am9 if FIT_YEARS[0] <= r["year"] <= FIT_YEARS[1]]
    ev = [r for r in am9 if EVAL_YEARS[0] <= r["year"] <= EVAL_YEARS[1]]
    curve = {lam: float(ll_harville(fit, lam).mean()) for lam in LAMBDA_GRID}
    lam_star = min(curve, key=curve.get)
    llh = ll_harville(ev, lam_star)
    llm = ll_market(ev, 1.0)
    d = float(llh.mean() - llm.mean())
    anchor_pass = bool(d > 0 and DELTA_RANGE[0] <= abs(d) <= DELTA_RANGE[1])
    per_year = {}
    for y in YEARS:
        ry = [r for r in am9 if r["year"] == y]
        if ry:
            per_year[str(y)] = {"races": len(ry),
                                "ll_harville": float(ll_harville(ry, lam_star).mean()),
                                "ll_market": float(ll_market(ry).mean()),
                                "delta_H_minus_M": float(ll_harville(ry, lam_star).mean() - ll_market(ry).mean())}
    all_curve = {lam: float(ll_harville(am9, lam).mean()) for lam in LAMBDA_GRID}
    lam_all = min(all_curve, key=all_curve.get)
    d_all = float(ll_harville(am9, lam_all).mean() - ll_market(am9).mean())

    # ---- dry-run fits (terminal, ≤2018)
    def nll_gamma(g):
        return float(ll_market(term, g).mean())
    rg = minimize_scalar(nll_gamma, bounds=(0.5, 2.0), method="bounded", options={"xatol": 1e-6})

    def nll_lam(lam):
        return float(ll_harville(term, lam).mean())
    rl = minimize_scalar(nll_lam, bounds=(0.3, 1.5), method="bounded", options={"xatol": 1e-6})

    res = {
        "role": "≤2018 だけで crux_joint の T0 頭対頭の方向と桁を再現する anchor + dry-run fit。新規成果ではない",
        "known_prior_result": "crux_joint.py: 9h, λ fit<=2023, eval 2024-25, market LL 3.343 < Harville 3.380 (|Δ|=0.037)",
        "result_loader_max_year": int(outc["year"].max()),
        "funnel": funnel,
        "anchor": {
            "rule": "λ を 2013-2016 で格子 fit、2017-2018 で評価。Δ = LL(Harville) − LL(市場)",
            "lambda_grid_curve_fit": curve, "lambda_star": lam_star,
            "n_fit_races": len(fit), "n_eval_races": len(ev),
            "ll_harville_eval": float(llh.mean()), "ll_market_eval": float(llm.mean()),
            "delta_H_minus_M": d, "required_direction": "Δ > 0 (市場優位)",
            "required_abs_delta_range": list(DELTA_RANGE),
            "direction_ok": bool(d > 0),
            "magnitude_ok": bool(DELTA_RANGE[0] <= abs(d) <= DELTA_RANGE[1]),
            "anchor_pass": anchor_pass,
        },
        "reference_in_sample_le2018": {"lambda_star": lam_all, "delta_H_minus_M": d_all,
                                        "n_races": len(am9)},
        "per_year_eval_lambda_star": per_year,
        "dryrun_fits_terminal_le2018": {
            "n_races": len(term),
            "gamma_powerlaw_devig": float(rg.x), "ll_at_gamma": float(rg.fun),
            "ll_at_gamma_1_proportional": float(ll_market(term, 1.0).mean()),
            "lambda_T1": float(rl.x), "ll_at_lambda_T1": float(rl.fun),
            "ll_T0_harville": float(ll_harville(term, 1.0).mean()),
            "note": "Stage 1 では年 Y ごとに ≤Y−1 で refit する。ここでの値は Stage 0 の検出力監査・計算量見積りにだけ使う",
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "anchor_le2018.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float),
                                            encoding="utf-8")
    print(json.dumps(res["anchor"], ensure_ascii=False))
    print(json.dumps(res["dryrun_fits_terminal_le2018"], ensure_ascii=False))
    print(json.dumps(funnel, ensure_ascii=False))
    print(f"[saved] out/anchor_le2018.json ({res['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
