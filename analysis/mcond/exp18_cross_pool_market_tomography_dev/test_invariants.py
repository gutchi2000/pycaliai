# -*- coding: utf-8 -*-
"""
test_invariants.py — EXP18 Stage 0: synthetic oracle と invariant の検査
=======================================================================
合成データの検査は結果性能と独立。実データを使う検査 (未来年削除・outcome 改変・target 列不存在) は
市場構造 loader の 2018 年分と結果 loader (<=2018) だけを使い、2019 年以降の結果は読まない。
出力: out/invariant_tests.json
実行: python -m analysis.mcond.exp18_cross_pool_market_tomography_dev.test_invariants
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import tomography as T
from . import payout_formula as PF
from .loaders import (OUT, RESULT_MAX_YEAR, SEALED_FROM_YEAR, _drop_sealed,
                      assert_no_outcome_columns, load_outcomes, load_structure, realized_top2)
from .market_build import pool_matrices, race_arrays, snapshot_index

RNG = np.random.default_rng(20260925)
results = []


def check(name, ok, detail=""):
    results.append({"test": name, "pass": bool(ok), "detail": detail})


def rand_pi(n, conc=0.8):
    a = RNG.dirichlet(np.full(n, conc))
    return a / a.sum()


def pair_key_map(bans, probs):
    a, b = T.pair_index(len(bans))
    return {(int(min(bans[x], bans[y])), int(max(bans[x], bans[y]))): float(p)
            for x, y, p in zip(a, b, probs)}


def synth_races(n_races, lam_true=0.8, umaren_lam=None, tan_noise=None, seed=1):
    """合成 race 群。真の着順は λ 割引 Harville (λ=lam_true)。
    umaren_lam: UMAREN を別の λ で値付けする (温度では吸収できない構造的な歪み)。
    tan_noise:  TANSHO marginal に乗法ノイズを入れる (source 側の歪み)"""
    rng = np.random.default_rng(seed)
    logm, logq, off, win = [], [], [0], []
    for _ in range(n_races):
        n = int(rng.integers(8, 17))
        pi = rng.dirichlet(np.full(n, 0.8))
        p_true = T.stern_top2(pi, lam_true)
        m = p_true.copy() if umaren_lam is None else T.stern_top2(pi, umaren_lam)
        pi_src = pi.copy()
        if tan_noise is not None:
            pi_src = pi * np.exp(rng.normal(0, tan_noise, n))
            pi_src = pi_src / pi_src.sum()
        q = T.stern_top2(pi_src, lam_true)
        k = rng.choice(len(p_true), p=p_true)
        logm.append(np.log(m)); logq.append(np.log(q))
        win.append(off[-1] + k)
        off.append(off[-1] + len(m))
    return np.concatenate(logm), np.concatenate(logq), np.array(off), np.array(win)


def delta_ll(logm, logq, off, win, fit_c, a0, with_se=False):
    """同じ race 上での LL(q_cross) − LL(q_temp) (平均、with_se なら race 単位 SE も返す)"""
    d = []
    for r in range(len(off) - 1):
        s, e = off[r], off[r + 1]
        m = np.exp(logm[s:e]); q = np.exp(logq[s:e])
        pt = T.temp_probs(m, a0)
        pc = T.cross_probs(m, q, fit_c["a"], fit_c["beta"], fit_c["collinear"])
        w = win[r] - s
        d.append(-np.log(pc[w]) + np.log(pt[w]))
    d = np.array(d)
    if with_se:
        return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
    return float(d.mean())


def main():
    t0 = time.time()
    # ---- 1-3: n<=8 全順列 oracle
    maxerr_t0 = maxerr_t1 = maxerr_pk = 0.0
    for n in range(5, 9):
        for _ in range(3):
            pi = rand_pi(n)
            for lam in (1.0, 0.8):
                o2, ok3 = T.oracle_top2_topk(pi, lam, 3)
                f2 = T.stern_top2(pi, lam)
                X = T.ordered_prefixes(n, 3)
                pk = T.place_probs(T.stern_prefix_probs(pi, lam, X), X, n)
                if lam == 1.0:
                    maxerr_t0 = max(maxerr_t0, float(np.abs(T.harville_top2(pi) - o2).max()))
                maxerr_t1 = max(maxerr_t1, float(np.abs(f2 - o2).max()))
                maxerr_pk = max(maxerr_pk, float(np.abs(pk - ok3).max()))
    check("oracle_harville_unordered_top2_n<=8", maxerr_t0 < 1e-12, f"max err {maxerr_t0:.2e}")
    check("oracle_stern_unordered_top2_n<=8", maxerr_t1 < 1e-12, f"max err {maxerr_t1:.2e}")
    check("oracle_place_topk_n<=8", maxerr_pk < 1e-12, f"max err {maxerr_pk:.2e}")

    # ---- 4: race 内確率和
    sums = []
    for n in (5, 9, 14, 18):
        pi = rand_pi(n)
        sums += [T.stern_top2(pi, 0.85).sum(), T.devig_power(1 / pi * 1.25, 1.1).sum(),
                 T.temp_probs(T.stern_top2(pi, 1.0), 0.9).sum()]
        X = T.ordered_prefixes(n, T.places(n))
        sums.append(T.place_probs(T.stern_prefix_probs(pi, 0.85, X), X, n).sum() / T.places(n))
    check("race_sum_one", max(abs(s - 1) for s in sums) < 1e-12,
          f"max |Σ−1| {max(abs(s - 1) for s in sums):.2e} (place は Σ/k)")

    # ---- 5: 馬番順序不変
    pi = rand_pi(12)
    bans = np.arange(1, 13)
    perm = RNG.permutation(12)
    m1 = pair_key_map(bans, T.stern_top2(pi, 0.8))
    m2 = pair_key_map(bans[perm], T.stern_top2(pi[perm], 0.8))
    check("horse_order_invariance", max(abs(m1[k] - m2[k]) for k in m1) < 1e-14)

    # ---- 6: 行順 (race 順) 不変の fit
    logm, logq, off, win = synth_races(400, umaren_lam=1.0, seed=3)
    f1 = T.fit_cross(logm, logq, off, win)
    order = RNG.permutation(len(off) - 1)
    lm2, lq2, off2, w2 = [], [], [0], []
    for r in order:
        s, e = off[r], off[r + 1]
        lm2.append(logm[s:e]); lq2.append(logq[s:e]); w2.append(off2[-1] + win[r] - s)
        off2.append(off2[-1] + e - s)
    f2 = T.fit_cross(np.concatenate(lm2), np.concatenate(lq2), np.array(off2), np.array(w2))
    check("row_order_invariance_fit", abs(f1["a"] - f2["a"]) < 1e-9 and abs(f1["beta"] - f2["beta"]) < 1e-9,
          f"Δa={abs(f1['a']-f2['a']):.1e} Δβ={abs(f1['beta']-f2['beta']):.1e}")

    # ---- 7: power-law de-vig
    o = np.array([2.1, 3.5, 7.0, 15.0, 40.0])
    prop = (1 / o) / (1 / o).sum()
    check("devig_power_gamma1_equals_proportional", np.allclose(T.devig_power(o, 1.0), prop, atol=1e-15))
    g2 = T.devig_power(o, 1.3)
    check("devig_power_monotone_and_sharpens", np.all(np.diff(g2) < 0) and g2[0] > prop[0])

    # ---- 8-9: 温度 null・cross 代替の回復
    rng = np.random.default_rng(11)
    lm, lq, of, wn = [], [], [0], []
    for _ in range(3000):
        n = int(rng.integers(8, 17))
        pi = rng.dirichlet(np.full(n, 0.8))
        m = T.stern_top2(pi, 1.0)
        q = T.stern_top2(rng.dirichlet(np.full(n, 0.8)) * 0.3 + pi * 0.7, 1.0)
        truth = T.cross_probs(m, q, 0.9, 0.35, False)
        k = rng.choice(len(truth), p=truth)
        lm.append(np.log(m)); lq.append(np.log(q)); wn.append(of[-1] + k); of.append(of[-1] + len(m))
    lm, lq, of, wn = np.concatenate(lm), np.concatenate(lq), np.array(of), np.array(wn)
    fc = T.fit_cross(lm, lq, of, wn)
    check("cross_alt_recovers_generating_params", abs(fc["a"] - 0.9) < 0.12 and abs(fc["beta"] - 0.35) < 0.12,
          f"a={fc['a']:.3f} (0.9) β={fc['beta']:.3f} (0.35)")
    m = T.stern_top2(rand_pi(10), 1.0)
    check("temp_null_a0_1_is_identity", np.allclose(T.temp_probs(m, 1.0), m, atol=1e-15))

    # ---- 10: coherent synthetic market (UMAREN = Harville(π) with 22.5% takeout)
    pi = rand_pi(14)
    h = T.harville_top2(pi)
    um_odds = 0.775 / h
    m_um = T.devig_power(um_odds, 1.0)
    check("coherent_market_tansho_equals_umaren", float(np.abs(m_um - h).max()) < 1e-14,
          f"max diff {float(np.abs(m_um - h).max()):.1e}")
    logm, logq, off, win = synth_races(300, lam_true=1.0, seed=5)
    fcoh = T.fit_cross(logm, logq, off, win)
    check("coherent_market_collinear_q_cross_equals_q_temp", fcoh["collinear"],
          f"rank={fcoh['rank']} cond={fcoh['cond']}")

    # ---- 11: 1 プールだけ歪めた合成市場
    #   UMAREN を IIA (λ=1) で誤って値付け (真は λ=0.45) → cross-pool が検出する (12,000 race)
    #   TANSHO だけにノイズ (UMAREN は整合) → target の改善として検出しない
    #   (純粋な冪の歪みは温度 null が吸収する設計なので、ここでは構造的な歪みを使う)
    logm, logq, off, win = synth_races(12000, lam_true=0.45, umaren_lam=1.0, seed=7)
    fu = T.fit_cross(logm, logq, off, win)
    a0u = T.fit_temp(logm, off, win)
    du, seu = delta_ll(logm, logq, off, win, fu, a0u, with_se=True)
    logm, logq, off, win = synth_races(12000, lam_true=0.45, tan_noise=0.3, seed=7)
    ft = T.fit_cross(logm, logq, off, win)
    a0t = T.fit_temp(logm, off, win)
    dt, set_ = delta_ll(logm, logq, off, win, ft, a0t, with_se=True)
    # 検出 = race 単位 z < −3 かつ β > 0 / 非検出 = |z| < 3 (合成データ上の感度検査で、Gate ではない)
    check("distorted_target_pool_detected", du / seu < -3 and fu["beta"] > 0,
          f"UMAREN 歪み: Δ={du:+.5f} z={du/seu:.1f} β={fu['beta']:.3f}")
    check("distorted_source_pool_not_detected_as_target_gain", abs(dt / set_) < 3,
          f"TANSHO 歪み: Δ={dt:+.5f} z={dt/set_:.1f} β={ft['beta']:.3f}")

    # ---- 12: U_SELF (UMAREN 自身を検算入力にすると同じ de-vig を bit 一致で再現)
    o = 0.775 / T.stern_top2(rand_pi(11), 0.9) * np.exp(RNG.normal(0, 0.05, 55))
    m_prop = T.devig_power(o, 1.0)
    q_self = T.devig_power(o, 1.0)
    check("U_SELF_devig_bitwise", np.array_equal(m_prop, q_self))
    logm, logq, off, win = synth_races(200, seed=9)
    fself = T.fit_cross(logm, logm.copy(), off, win)
    same = all(np.array_equal(
        T.cross_probs(np.exp(logm[off[r]:off[r + 1]]), np.exp(logm[off[r]:off[r + 1]]),
                      fself["a"], fself["beta"], fself["collinear"]),
        T.temp_probs(np.exp(logm[off[r]:off[r + 1]]), T.fit_temp(logm, off, win)))
        for r in range(len(off) - 1))
    check("U_SELF_offset_collinear_q_cross_equals_q_temp_bitwise", fself["collinear"] and same)

    # ---- 13: 共線 (条件数 > 1e12) で重複列を落とす
    near = logm + 1e-9 * RNG.normal(size=len(logm))
    fnear = T.fit_cross(logm, near, off, win)
    check("collinearity_cond_gt_1e12_drops_duplicate", fnear["collinear"],
          f"cond={fnear['cond']:.2e}")

    # ---- 14: 一様 q_LOPO → |Δ_P5| <= 1e-12
    uni = np.concatenate([np.full(off[r + 1] - off[r], -np.log(off[r + 1] - off[r]))
                          for r in range(len(off) - 1)])
    funi = T.fit_cross(logm, uni, off, win)
    dp5 = delta_ll(logm, uni, off, win, funi, T.fit_temp(logm, off, win))
    check("P5_uniform_qlopo_delta_le_1e-12", funi["collinear"] and abs(dp5) <= 1e-12,
          f"|Δ_P5|={abs(dp5):.1e}")

    # ---- 15-16: T2
    pi = rand_pi(12)
    k = T.places(12)
    tau = k * rand_pi(12)
    t1 = T.stern_top2(pi, 0.85)
    t2w0 = T.t2_soft_fukusho(pi, tau, 0.85, 0.0)
    check("T2_w0_equals_T1_bitwise", np.array_equal(t1, t2w0))
    pairs, det = T.t2_soft_fukusho(pi, tau, 0.85, 1.0, return_detail=True)
    X = T.ordered_prefixes(12, k)
    base_place = T.place_probs(T.stern_prefix_probs(pi, 0.85, X) / T.stern_prefix_probs(pi, 0.85, X).sum(), X, 12)
    moved = np.abs(det["place"] - tau).sum() < np.abs(base_place - tau).sum()
    check("T2_win_marginal_equality", float(np.abs(det["win"] - pi).max()) < 1e-9,
          f"max |win−π|={float(np.abs(det['win'] - pi).max()):.1e}")
    check("T2_place_moves_toward_tau_and_sum_is_k",
          moved and abs(det["place"].sum() - k) < 1e-9 and abs(pairs.sum() - 1) < 1e-12)
    # 勾配の数値検算 (有限差分)
    n = 7
    pi7 = rand_pi(n); tau7 = T.places(n) * rand_pi(n)
    X7 = T.ordered_prefixes(n, T.places(n))
    q07 = T.stern_prefix_probs(pi7, 0.9, X7); q07 /= q07.sum()
    Z7 = np.zeros((len(X7), n)); [Z7.__setitem__((np.arange(len(X7)), X7[:, t]), 1.0) for t in range(X7.shape[1])]

    def J(mu, w=0.7):
        e = np.log(q07) + Z7 @ mu; u = np.exp(e - e.max())
        blk = np.bincount(X7[:, 0], weights=u, minlength=n); q = u * pi7[X7[:, 0]] / blk[X7[:, 0]]
        return float((q * (np.log(q) - np.log(q07))).sum() + w * ((Z7.T @ q - tau7) ** 2).sum())
    mu0 = RNG.normal(0, 0.3, n)
    num = np.array([(J(mu0 + 1e-6 * np.eye(n)[j]) - J(mu0 - 1e-6 * np.eye(n)[j])) / 2e-6 for j in range(n)])
    # 解析勾配は t2 内部と同式。最適化結果が数値最適と一致するかで検算
    r_num = minimize(J, np.zeros(n), method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-14})
    _, d7 = T.t2_soft_fukusho(pi7, tau7, 0.9, 0.7, return_detail=True)
    check("T2_analytic_optimum_matches_derivative_free", abs(J(d7["mu"]) - r_num.fun) < 1e-7,
          f"J_analytic={J(d7['mu']):.10f} J_nm={r_num.fun:.10f}")

    # ---- 20: 現金込み Kelly 閉形式 = 数値最適
    worst = 0.0
    for _ in range(20):
        n = int(RNG.integers(4, 9))
        pi = rand_pi(n); qb = rand_pi(n); o = 0.8 / pi
        g, st, ns = T.kelly_cash_growth(qb, o, qb)

        def negf(b):
            wv = b[0] + b[1:] * o
            return -float((qb * np.log(np.clip(wv, 1e-12, None))).sum())
        r = minimize(negf, np.full(n + 1, 1 / (n + 1)), bounds=[(0, 1)] * (n + 1),
                     constraints=[{"type": "eq", "fun": lambda b: b.sum() - 1}], method="SLSQP",
                     options={"ftol": 1e-14, "maxiter": 500})
        worst = max(worst, abs(g - (-r.fun)))
    check("kelly_closed_form_equals_slsqp", worst < 1e-6, f"max |Δgrowth|={worst:.1e}")

    # ---- 21-22: 複勝式の合成往復と τ の和
    ok_rate = []
    for _ in range(200):
        n = int(RNG.integers(5, 19)); k = PF.places(n)
        v = rand_pi(n, 1.2)
        lo, hi = PF.display_lo_hi(v, k)
        vi = PF.invert_display(lo, hi, k)
        lo2, hi2 = PF.display_lo_hi(vi, k)
        ok_rate.append(np.mean(PF.roundtrip_ok(lo, lo2) & PF.roundtrip_ok(hi, hi2)))
    check("place_formula_synthetic_roundtrip", np.mean(ok_rate) >= 0.99,
          f"合成往復 一致率 {np.mean(ok_rate):.4f}")
    v = rand_pi(12)
    check("place_tau_sum_is_places_n", abs(PF.tau_from_shares(v, 3).sum() - 3) < 1e-6)

    # ---- 23-24: 封印と構造 loader の禁止列
    fake = pd.DataFrame({"x": [1, 2, 3]})
    kept = _drop_sealed(fake, pd.Series([2023, 2024, 2025]), "test")
    check("sealed_2024_2025_dropped_on_read", len(kept) == 1)
    try:
        load_outcomes(2019)
        check("result_loader_rejects_2019", False)
    except AssertionError:
        check("result_loader_rejects_2019", True)
    try:
        assert_no_outcome_columns(["rid16", "1単", "着順"], "test")
        check("structure_loader_forbidden_column_assert", False)
    except AssertionError:
        check("structure_loader_forbidden_column_assert", True)

    # ---- 17-19: 実データ (2018 構造 + <=2018 結果)
    st = load_structure(range(2017, 2020))           # 2019 は削除不変性の検査用 (結果は読まない)
    tan, um, info = st["tan"], st["um"], st["info"]

    def digest(tan_, um_, info_, year):
        W, LO, HI, U = pool_matrices(tan_, um_)
        idx = snapshot_index(tan_, um_, info_)
        h = hashlib.sha256()
        for rid in sorted(r for r in idx.index if r.startswith(str(year))):
            row = idx.loc[rid]
            if not (row.get("term_tan") == row.get("term_tan")):
                continue
            arr = race_arrays(W, LO, HI, U, int(row["term_tan"]), int(row.get("term_um", -1)))
            q = T.stern_top2((1 / arr["win"]) / (1 / arr["win"]).sum(), 0.85) if arr["n"] >= 2 else np.zeros(0)
            h.update(rid.encode()); h.update(arr["bans"].tobytes()); h.update(np.nan_to_num(arr["umaren"]).tobytes())
            h.update(q.tobytes())
        return h.hexdigest()

    d_full = digest(tan, um, info, 2018)
    t_cut = tan[tan["year"] <= 2018].reset_index(drop=True)
    u_cut = um[um["year"] <= 2018].reset_index(drop=True)
    d_cut = digest(t_cut, u_cut, info[info["year"] <= 2018], 2018)
    check("future_year_deletion_invariance_2018", d_full == d_cut)
    outc = load_outcomes(2018)
    outc2 = outc.copy(); outc2["jyun"] = outc2["jyun"].sample(frac=1.0, random_state=1).to_numpy()
    d_after_outcome_change = digest(tan, um, info, 2018)
    top_a = realized_top2(outc[outc["year"] == 2018]); top_b = realized_top2(outc2[outc2["year"] == 2018])
    check("outcome_modification_invariance", d_after_outcome_change == d_full and
          not top_a["top2"].equals(top_b["top2"]),
          "結果を改変しても市場確率・race set・T1 出力の hash は不変 (改変は実際に top2 を変えている)")
    src_cols = [c for c in tan.columns]
    check("target_umaren_columns_absent_from_source_input",
          not any(str(c).startswith("馬") and "-" in str(c) for c in src_cols) and
          "votes_umaren_total" not in src_cols,
          "q_LOPO の入力 (TANPUK 行列) に UMAREN の組合せ列・票数列が無い")
    check("result_loader_max_year_le_2018", int(outc["year"].max()) <= RESULT_MAX_YEAR)
    check("structure_loader_no_outcome_columns",
          all(not any(p in str(c) for p in ["着順", "払戻", "top2"]) for c in list(tan.columns) + list(um.columns) + list(info.columns)))

    res = {"n_tests": len(results), "n_pass": sum(r["pass"] for r in results),
           "all_passed": all(r["pass"] for r in results), "tests": results,
           "rng_seed": 20260925, "elapsed_sec": round(time.time() - t0, 1),
           "note": "合成検査は結果性能と独立。実データ検査は 2017-2019 の市場構造と <=2018 の結果だけを使う"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "invariant_tests.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for r in results:
        print(("PASS " if r["pass"] else "FAIL ") + r["test"] + (f"  [{r['detail']}]" if r["detail"] else ""))
    print(f"{res['n_pass']}/{res['n_tests']} passed ({res['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
