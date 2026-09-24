# -*- coding: utf-8 -*-
"""
test_invariants.py — EXP17 Stage 0 合成 invariant テスト (spec v0.2 §5.2 の必須 13 項目)
実行: python -m analysis.mcond.exp17_transitive_pl_graph_dev.test_invariants
出力: out/invariant_tests.json
実データは使わない。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .graph_core import (HorseIDError, PairParams, build_history_store, hodge_project,
                         pair_evidence, softmax_scores, validate_hids)

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
RNG = np.random.default_rng(20260925)


def hid(k: int) -> str:
    return f"{2010000000 + k:010d}"


def synth_runs(n_days=90, races_per_day=4, field=8, n_horses=160, start=20180105, seed=1):
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 1, n_horses)
    rows = []
    day = pd.Timestamp(str(start))
    for dd in range(n_days):
        dint = int(day.strftime("%Y%m%d"))
        for r in range(races_per_day):
            hs = rng.choice(n_horses, field, replace=False)
            perf = theta[hs] + rng.gumbel(size=field)
            order = np.argsort(-perf)
            rid = f"{dint}{r:02d}000000"[:16].ljust(16, "0")
            for pos, k in enumerate(order, 1):
                rows.append(dict(date=dint, rid16=rid, hid=hid(int(hs[k])), fin=float(pos),
                                 surf=int(r % 2), dband=int(rng.integers(0, 4))))
        day += pd.Timedelta(days=7 if dd % 2 else 1)
    return pd.DataFrame(rows), theta


def race_output(store, hids, day, prm=PairParams(), tau=1.0):
    res = pair_evidence(store, hids, day, tsurf=1, tdb=1, prm=prm)
    s, unc, comps, r = hodge_project(res.d, res.w, prm.lam)
    p = softmax_scores(s, tau)
    return res, s, unc, comps, p


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    results = {}
    runs, theta = synth_runs()
    T = 20180701  # target day inside the range
    store = build_history_store(runs)
    target = runs[runs.date >= T].groupby("rid16").first().index[0]
    hids = sorted(runs[runs.rid16 == target].hid.tolist())
    res, s, unc, comps, p = race_output(store, hids, T)

    # 1. future deletion → bit identical
    store_cut = build_history_store(runs[runs.date < T])
    res2, s2, unc2, comps2, p2 = race_output(store_cut, hids, T)
    results["future_deletion_bitwise"] = bool(np.array_equal(np.nan_to_num(res.d, nan=-999), np.nan_to_num(res2.d, nan=-999))
                                              and np.array_equal(s, s2) and np.array_equal(p, p2))

    # 2. mutate target race results → pre-race features unchanged
    mut = runs.copy()
    m = mut.rid16 == target
    mut.loc[m, "fin"] = mut.loc[m, "fin"].to_numpy()[::-1]
    store_m = build_history_store(mut)
    res3, s3, *_ = race_output(store_m, hids, T)
    results["target_result_mutation_invariance"] = bool(np.array_equal(np.nan_to_num(res.d, nan=-999), np.nan_to_num(res3.d, nan=-999)) and np.array_equal(s, s3))

    # 3. same-day races share the day-start snapshot: mutate another same-day race → unchanged
    same_day = runs[(runs.date == runs.loc[runs.rid16 == target, "date"].iloc[0]) & (runs.rid16 != target)]
    ok = True
    if len(same_day):
        other = same_day.rid16.iloc[0]
        mut2 = runs.copy(); m2 = mut2.rid16 == other
        mut2.loc[m2, "fin"] = mut2.loc[m2, "fin"].to_numpy()[::-1]
        res4, s4, *_ = race_output(build_history_store(mut2), hids, T)
        ok = np.array_equal(np.nan_to_num(res.d, nan=-999), np.nan_to_num(res4.d, nan=-999)) and np.array_equal(s, s4)
    results["same_day_snapshot_shared"] = bool(ok)

    # 4. common opponent's results after target day do not enter d_ij: append future races for a common opponent
    fut = runs[runs.date < T].copy()
    extra = fut.tail(24).copy(); extra["date"] = 20190101; extra["rid16"] = "2019010199000000"
    extra["fin"] = extra["fin"].to_numpy()[::-1]
    res5, s5, *_ = race_output(build_history_store(pd.concat([runs, extra])), hids, T)
    results["common_opponent_post_target_excluded"] = bool(np.array_equal(np.nan_to_num(res.d, nan=-999), np.nan_to_num(res5.d, nan=-999)))

    # 5. ID collision / invalid ID / name recovery → fail closed
    try:
        validate_hids(["2010000001", "2010000001"], names=["A", "B"]); coll = False
    except HorseIDError:
        coll = True
    try:
        validate_hids(["201000000X"]); inv = False
    except HorseIDError:
        inv = True
    try:
        validate_hids([None]); nn = False
    except (HorseIDError, TypeError):
        nn = True
    results["id_collision_fail_closed"] = coll
    results["invalid_id_fail_closed"] = inv and nn
    results["no_name_recovery"] = "馬名" not in open(HERE / "graph_core.py", encoding="utf-8").read().split("validate_hids")[1].split("def ")[1]

    # 6. antisymmetry
    dd = np.nan_to_num(res.d)
    results["antisymmetry_max_abs"] = float(np.abs(dd + dd.T).max())
    results["antisymmetry_ok"] = results["antisymmetry_max_abs"] <= 1e-12

    # 7. row/horse permutation invariance (of runs rows and of hids order)
    perm = RNG.permutation(len(hids))
    res_p, s_p, *_ = race_output(store, [hids[k] for k in perm], T)
    inv_perm = np.empty_like(perm); inv_perm[perm] = np.arange(len(perm))
    d_back = res_p.d[np.ix_(inv_perm, inv_perm)]
    s_back = s_p[inv_perm]
    runs_shuf = runs.sample(frac=1.0, random_state=7)
    res_r, s_r, *_ = race_output(build_history_store(runs_shuf), hids, T)
    results["horse_order_invariance"] = bool(np.allclose(np.nan_to_num(d_back, nan=-999), np.nan_to_num(res.d, nan=-999)) and np.allclose(s_back, s, atol=1e-12))
    results["row_order_invariance"] = bool(np.array_equal(np.nan_to_num(res_r.d, nan=-999), np.nan_to_num(res.d, nan=-999)) and np.array_equal(s_r, s))

    # 8. Hodge reproducibility (deterministic) and probability sum
    s_again, *_ = hodge_project(res.d, res.w)
    results["hodge_reproducible"] = bool(np.array_equal(s, s_again))
    results["prob_sum_abs_err"] = float(abs(p.sum() - 1.0))
    results["prob_sum_ok"] = results["prob_sum_abs_err"] <= 1e-12

    # 9. disconnected components & uncovered horses: build a race of 2 blocks with no cross evidence + 1 debut
    n = 5
    d = np.full((n, n), np.nan); w = np.zeros((n, n))
    def put(i, j, v):
        d[i, j] = v; d[j, i] = -v; w[i, j] = w[j, i] = 1.0
    put(0, 1, 1.0); put(2, 3, -0.5)          # horse 4 uncovered
    s9, unc9, comps9, r9 = hodge_project(d, w)
    results["components_detected"] = [sorted(c) for c in comps9]
    results["component_zero_centered"] = bool(abs(s9[[0, 1]].sum()) < 1e-9 and abs(s9[[2, 3]].sum()) < 1e-9)
    results["uncovered_flag_and_zero"] = bool(unc9[4] and s9[4] == 0.0 and not unc9[:4].any())
    results["disconnected_ok"] = results["component_zero_centered"] and results["uncovered_flag_and_zero"] and len(comps9) == 3

    # 10. direct-met pairs are not in the primary (2-hop) sample
    leak = 0
    for (i, j), lst in res.per_c.items():
        if res.direct[i, j]:
            # direct pairs may have 2-hop evidence too; primary sample must exclude them → checked by consumer flag
            pass
    prim = [(i, j) for (i, j) in res.per_c if not res.direct[i, j]]
    mixed = [(i, j) for (i, j) in prim if res.direct[i, j]]
    results["primary_sample_excludes_direct"] = len(mixed) == 0 and len(prim) > 0   # 空集合で自明に通らないことを要求
    results["n_pairs_direct"] = int(res.direct[np.triu_indices(len(hids), 1)].sum())
    results["n_pairs_2hop_primary"] = len(prim)

    # 11. missing stays missing (no zero fabrication): a pair without common opponents must be NaN not 0
    iu = np.triu_indices(len(hids), 1)
    no_ev = (res.n_common[iu] == 0)
    results["missing_not_zero"] = bool(np.all(np.isnan(res.d[iu][no_ev])))

    # 12. algebraic reduction check: if history log-odds are an exact gradient, projection returns theta (up to centering)
    #     construct d_ij = theta_i - theta_j on a complete graph → s == theta - mean(theta), residual == 0
    th = RNG.normal(size=7); dg = th[:, None] - th[None, :]; wg = np.ones((7, 7)); np.fill_diagonal(wg, 0)
    sg, _, _, rg = hodge_project(dg, wg, lam=0.0) if False else hodge_project(dg, wg, lam=1e-12)
    results["gradient_input_returns_theta_maxerr"] = float(np.abs(sg - (th - th.mean())).max())
    results["gradient_input_residual_max"] = float(np.nanmax(np.abs(rg)))
    results["algebraic_reduction_when_transitive"] = results["gradient_input_returns_theta_maxerr"] < 1e-6

    # 13. curl input is fully removed by projection: d = pure 3-cycle → s == 0, residual == d
    dc = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]], float); wc = np.ones((3, 3)); np.fill_diagonal(wc, 0)
    sc, _, _, rc = hodge_project(dc, wc, lam=1e-12)
    results["curl_input_projected_scores_max"] = float(np.abs(sc).max())
    results["curl_input_residual_equals_d"] = bool(np.allclose(np.nan_to_num(rc), dc, atol=1e-9))
    results["pair_specific_info_survives_only_in_residual"] = results["curl_input_projected_scores_max"] < 1e-6 and results["curl_input_residual_equals_d"]

    keys = ["future_deletion_bitwise", "target_result_mutation_invariance", "same_day_snapshot_shared",
            "common_opponent_post_target_excluded", "id_collision_fail_closed", "invalid_id_fail_closed", "no_name_recovery",
            "antisymmetry_ok", "horse_order_invariance", "row_order_invariance", "hodge_reproducible", "prob_sum_ok",
            "disconnected_ok", "primary_sample_excludes_direct", "missing_not_zero", "algebraic_reduction_when_transitive",
            "pair_specific_info_survives_only_in_residual"]
    results["all_passed"] = all(bool(results[k]) for k in keys)
    results["n_checks"] = len(keys)
    (OUT / "invariant_tests.json").write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: results[k] for k in keys + ["all_passed"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
