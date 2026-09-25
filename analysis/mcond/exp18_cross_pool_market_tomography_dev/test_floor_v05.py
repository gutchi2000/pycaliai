# -*- coding: utf-8 -*-
"""
test_floor_v05.py — v0.5 floor 探索**前**に通す合成 oracle / invariant テスト
===========================================================================
結果ラベルは使わない (cache の ≤2018 formal 件数を除く)。out/floor_v05_tests.json へ保存。
実行: python -m analysis.mcond.exp18_cross_pool_market_tomography_dev.test_floor_v05
"""
from __future__ import annotations

import inspect
import json
import sys
import time

import numpy as np

from . import floor_v05 as F
from . import tomography as T
from .gate_grade import run_boundary_tests
from .loaders import OUT

RES = []


def rec(name, ok, detail=""):
    RES.append({"test": name, "pass": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


def se_clogit(X, mk, win, th):
    eta = X @ th
    lp = mk.norm_log(eta)
    p = np.exp(lp)
    Ex = mk.sum(p[:, None] * X)
    Exx = mk.sum(p[:, None, None] * X[:, :, None] * X[:, None, :])
    H = (Exx - Ex[:, :, None] * Ex[:, None, :]).sum(0)
    return np.sqrt(np.diag(np.linalg.inv(H)))


def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    t0 = time.time()
    W = F.load_world()
    mk = W["mk"]
    meta = json.loads((OUT / "floor_v05_cache_meta.json").read_text(encoding="utf-8"))

    # 1 真値は q_cross 族の中 (主方式): log p = norm((1−ε) log m + ε log q)
    eps = 0.37
    lp = F.truth_logp(eps, F.direction(1.0, W), W)
    alt = mk.norm_log((1 - eps) * W["logm"] + eps * W["logq"])
    rec("truth_primary_in_q_cross_family", np.max(np.abs(lp - alt)) < 1e-10,
        f"max|Δlog|={np.max(np.abs(lp - alt)):.1e}")

    # 2 ε=0 は市場そのもの、真の Δ=0
    lp0 = F.truth_logp(0.0, F.direction(1.0, W), W)
    d0, a00 = F.delta_true(0.0, F.direction(1.0, W), W)
    rec("eps0_truth_is_market_and_delta_zero",
        np.max(np.abs(lp0 - W["logm"])) < 1e-12 and abs(d0) < 1e-12 and abs(a00 - 1) < 1e-9,
        f"Δ={d0:.1e} a0*={a00:.12f}")

    # 3 ε 較正
    d = F.direction(1.0, W)
    e1 = F.calibrate_eps(0.01, d, W)
    dt, _ = F.delta_true(e1, d, W)
    rec("eps_calibration_hits_target", abs(dt - 0.01) < 1e-9, f"ε={e1:.6f} Δ_true={dt:.12f}")

    # 4 a0* は期待 logloss の最良温度 (独立な導関数なし最適化と一致)
    from scipy.optimize import minimize_scalar
    r0, r1 = W["ev_r"]
    smk, h0, h1 = F.sub_mk(W["off"], r0, r1)
    lm = W["logm"][h0:h1]
    p = np.exp(smk.norm_log(lm + e1 * d[h0:h1]))
    a_n = F.fit_temp_expected(lm, p, smk)
    r = minimize_scalar(lambda a: -float((smk.sum(p * smk.norm_log(a * lm))).sum()), bounds=(0.5, 1.5),
                        method="bounded", options={"xatol": 1e-10})
    rec("a0_star_matches_derivative_free", abs(a_n - r.x) < 1e-6, f"Newton {a_n:.9f} vs bounded {r.x:.9f}")

    # 5 ρ<1 の直交方向
    v, c = W["v"], W["c"]
    X = np.column_stack([np.ones_like(W["logm"]), W["logm"], W["logq"]])
    dots = np.abs(mk.sum(X * v[:, None]))
    nc, nv = np.sqrt(mk.sum(c * c)), np.sqrt(mk.sum(v * v))
    rec("rho_direction_orthogonal_and_scaled",
        dots.max() < 1e-8 and np.max(np.abs(nv - nc) / np.maximum(nc, 1e-300)) < 1e-9,
        f"max|<v,x>|={dots.max():.1e} max rel ‖v‖−‖c‖={np.max(np.abs(nv - nc) / nc):.1e}")
    rec("rho1_direction_is_primary_bitwise", F.direction(1.0, W) is W["c"])
    d5 = F.direction(0.5, W)
    rec("rho05_direction_mix", np.max(np.abs(d5 - (0.5 * c + np.sqrt(0.75) * v))) == 0.0)

    # 6 大標本で正しく指定された推定器が真値を回収する (2013-2022 の合成 winner 1 draw)
    e5 = F.calibrate_eps(0.05, d, W)
    lp5 = F.truth_logp(e5, d, W)
    rng = np.random.default_rng(99)
    win = F.draw_winners(lp5, W, rng)
    fc, _, nfit = F.fit_year(2023, win, W)
    rr0, rr1 = W["yr_rng"][2013][0], W["yr_rng"][2022][1]
    fmk, g0, g1 = F.sub_mk(W["off"], rr0, rr1)
    Xf = np.column_stack([W["logm"][g0:g1], W["logq"][g0:g1]])
    se = se_clogit(Xf, fmk, win[rr0:rr1] - g0, np.array([fc["a"], fc["beta"]]))
    za, zb = (fc["a"] - (1 - e5)) / se[0], (fc["beta"] - e5) / se[1]
    rec("well_specified_recovery_large_n", abs(za) < 4 and abs(zb) < 4,
        f"ε={e5:.4f} â={fc['a']:.4f} β̂={fc['beta']:.4f} z=({za:.2f},{zb:.2f}) n={nfit}")

    # 7 ρ=0.5 では推定器が真の構造の一部しか回収しない
    d5c = F.calibrate_eps(0.05, d5, W)
    lp55 = F.truth_logp(d5c, d5, W)
    win5 = F.draw_winners(lp55, W, np.random.default_rng(98))
    fc5, _, _ = F.fit_year(2023, win5, W)
    ratio = fc5["beta"] / d5c
    rec("rho05_partial_recovery", 0.25 < ratio < 0.75,
        f"ε={d5c:.4f} β̂={fc5['beta']:.4f} β̂/ε={ratio:.3f} (期待 ≈ 0.5)")

    # 8 真値 = 市場なら実オッズで正の期待値の組が無い → どの推定器でも成長 <= 0
    pm = np.exp(W["logm"][h0:h1]) * W["odds"][h0:h1]
    g0r = F.growth_rep(1.0, 0.0, [1, 2, 3], W)
    rec("null_market_no_positive_ev_pair_and_growth_le_0", pm.max() <= 1.0 and g0r["growth_per_race"] <= 1e-15,
        f"max p·o={pm.max():.4f} growth={g0r['growth_per_race']:.2e}")

    # 9 oracle (q=p) の Kelly 成長は各 race で非負
    Y = 2019
    a_, b_ = W["yr_rng"][Y]
    ymk, y0, y1 = F.sub_mk(W["off"], a_, b_)
    pp = np.exp(lp5[y0:y1])
    gs = [T.kelly_cash_growth(pp[ymk.off[i]:ymk.off[i + 1]], W["odds"][y0 + ymk.off[i]:y0 + ymk.off[i + 1]],
                              pp[ymk.off[i]:ymk.off[i + 1]])[0] for i in range(ymk.R)]
    rec("oracle_kelly_growth_nonnegative_each_race", min(gs) >= -1e-12, f"min={min(gs):.2e} mean={np.mean(gs):.5f}")

    # 10 fit 値が真値なら q_cross = p (独立ノイズ注入経路が無い)
    qc = F.cross_logq({"a": 1 - e5, "beta": e5, "collinear": False}, W["logm"], W["logq"], mk)
    rec("true_params_reproduce_truth", np.max(np.abs(qc - lp5)) < 1e-10, f"{np.max(np.abs(qc - lp5)):.1e}")
    src = inspect.getsource(F.growth_rep) + inspect.getsource(F.power_rep) + inspect.getsource(F.fit_year)
    rec("no_declared_independent_noise_in_estimator_path", "normal(" not in src and "sqrt(2" not in src)

    # 11 決定性
    ga = F.growth_rep(1.0, e1, [5, 6, 7], W)
    gb = F.growth_rep(1.0, e1, [5, 6, 7], W)
    rec("growth_rep_deterministic_given_seed", json.dumps(ga, sort_keys=True) == json.dumps(gb, sort_keys=True))

    # 12 決済オッズは実 terminal 馬連オッズ (構造 loader から再構築して bit 一致)
    from .loaders import load_structure
    from .market_build import pool_matrices, race_arrays, snapshot_index
    st = load_structure([2019])
    Wm, LO, HI, U = pool_matrices(st["tan"], st["um"])
    idx = snapshot_index(st["tan"], st["um"], st["info"])
    ok = True
    ra, rb = W["yr_rng"][2019]
    for i in range(ra, ra + 50):
        rid = str(W["rid"][i])
        row = idx.loc[rid]
        at = race_arrays(Wm, LO, HI, U, int(row["term_tan"]), int(row["term_um"]))
        ok &= bool(np.array_equal(at["umaren"], W["odds"][W["off"][i]:W["off"][i + 1]]))
    rec("settlement_odds_are_terminal_umaren_bitwise", ok, "2019 先頭 50 race")

    # 13 n_fit 表
    fb = {int(k): v for k, v in meta["formal_by_year_sim"].items()}
    nf = {int(k): v for k, v in meta["n_fit_by_eval_year"].items()}
    okn = all(nf[Y] == sum(fb[y] for y in range(F.FIT_START, Y)) for Y in F.EVAL_YEARS)
    okn &= all(int((W["year"] == y).sum()) == fb[y] for y in fb)
    okn &= meta["outcome_loader_max_year"] <= 2018 and sum(fb[y] for y in range(2013, 2019)) == meta["formal_le2018_exact"]
    rec("n_fit_matches_real_formal_counts", okn, json.dumps(nf))
    fcY = F.fit_year(2019, win, W)[2]
    rec("fit_year_uses_exactly_n_fit_races", fcY == nf[2019], f"{fcY} vs {nf[2019]}")
    rec("no_sealed_years_in_cache", int(W["year"].max()) <= 2023)

    # 14 floor の補間規則
    cases = [({0.0: 0.0, 0.01: 0.00005, 0.02: 0.00015}, 0.015),
             ({0.0: 0.0, 0.01: -0.001, 0.02: -0.0005}, None),
             ({0.0: 0.0, 0.01: 0.0002, 0.02: -0.001, 0.03: 0.0003}, 0.005)]
    okf = all((F.floor_from_curve(cv) is None and want is None) or
              (want is not None and abs(F.floor_from_curve(cv) - want) < 1e-12) for cv, want in cases)
    rec("floor_interpolation_rule", okf)

    # 15 Gate 関数の境界
    bad = run_boundary_tests()
    rec("gate_grade_boundary_tests", not bad, "; ".join(bad))

    n_ok = sum(r["pass"] for r in RES)
    out = {"n_tests": len(RES), "n_pass": n_ok, "all_passed": n_ok == len(RES), "tests": RES,
           "elapsed_sec": round(time.time() - t0, 1),
           "note": "floor 探索 (floor_v05.py floor) の前に実行。結果ラベル不使用"}
    (OUT / "floor_v05_tests.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{n_ok}/{len(RES)} passed ({out['elapsed_sec']}s)")
    sys.exit(0 if out["all_passed"] else 1)


if __name__ == "__main__":
    main()
