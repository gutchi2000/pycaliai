# -*- coding: utf-8 -*-
"""
test_stage1_machinery.py — evaluate_stage1.py の合成テスト (実 outcome を読まない。v0.5-final 凍結前に実行)
======================================================================================================
合成 race 配列を evaluate_stage1 の配列 cache (_A) へ直接入れ、vectorized 確率の oracle 一致・placebo の性質・
決定性・P5・cross-check の純関数・結果 loader の凍結ガードを確かめる。out/stage1_machinery_tests.json へ保存。
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from . import evaluate_stage1 as E
from . import tomography as T
from .floor_v05 import Mk
from .loaders import OUT, load_outcomes_stage1

RES = []


def rec(name, ok, detail=""):
    RES.append({"test": name, "pass": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


def synth(per_year=1500, seed=0, gamma_true=None, lam_true=None):
    rng = np.random.default_rng(seed)
    races = []
    for y in range(2013, 2024):
        for k in range(per_year):
            n = int(rng.integers(5, 15))
            pi = rng.dirichlet(np.full(n, 1.3))
            win = 0.8 / pi
            q0 = T.stern_top2(pi, 1.0)
            um = 0.775 / (q0 * np.exp(rng.normal(0, 0.15, len(q0))))
            um = np.maximum(um, 1.0)
            venue = int(rng.integers(0, 10))
            day = y * 10000 + int(rng.integers(1, 13)) * 100 + int(rng.integers(1, 29))
            races.append((y, day, venue, n, pi, win, um))
    hcnt = np.array([r[3] for r in races])
    hoff = np.concatenate([[0], np.cumsum(hcnt)])
    off = np.concatenate([[0], np.cumsum(hcnt * (hcnt - 1) // 2)])
    pa, pb = [], []
    for r, x in enumerate(races):
        a, b = T.pair_index(x[3])
        pa.append(a + hoff[r])
        pb.append(b + hoff[r])
    pi = np.concatenate([(1 / x[5]) / (1 / x[5]).sum() for x in races])
    X = np.concatenate([-np.log(x[6]) for x in races])
    ent = np.array([-(x[4] * np.log(x[4])).sum() for x in races])
    cuts = np.quantile(ent, [1 / 3, 2 / 3])
    cell_s = np.array([f"{x[2]}|{x[3]}|{int(np.searchsorted(cuts, e))}" for x, e in zip(races, ent)])
    _, cell = np.unique(cell_s, return_inverse=True)
    # winners: γ_true なら市場から、λ_true なら T1 から
    win_row = []
    for r, x in enumerate(races):
        if gamma_true is not None:
            p = T.devig_power(x[6], gamma_true)
        else:
            p = T.stern_top2((1 / x[5]) / (1 / x[5]).sum(), lam_true if lam_true is not None else 0.85)
        win_row.append(off[r] + int(rng.choice(len(p), p=p)))
    A = {"rid": np.array([f"{x[0]}{i:012d}" for i, x in enumerate(races)]), "year": np.array([x[0] for x in races]),
         "day": np.array([x[1] for x in races]), "venue": np.array([x[2] for x in races]), "n": hcnt,
         "hoff": hoff, "off": off, "pa": np.concatenate(pa), "pb": np.concatenate(pb),
         "win_row": np.array(win_row), "pi": pi, "x": X, "cell": cell,
         "venues": np.array([f"V{i}" for i in range(10)]), "ent_cuts": cuts}
    A["mk"] = Mk(off)
    A["hmk"] = Mk(hoff)
    A["race_of_horse"] = np.repeat(np.arange(len(hcnt)), hcnt)
    A["yr"] = {y: (int(np.flatnonzero(A["year"] == y)[0]), int(np.flatnonzero(A["year"] == y)[-1] + 1))
               for y in range(2013, 2024)}
    return A, races


def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    t0 = time.time()
    A, races = synth(per_year=400, seed=1)
    E._A.clear()
    E._A.update(A)

    # 1 vectorized T1 / de-vig = tomography の単レース関数
    lam, gam = 0.83, 1.07
    lq, mk, h0, h1 = E.logq_rows(A["pi"], lam, 0, len(races), A)
    lm = E.logm_rows(gam, 0, len(races), A)[0]
    e1 = max(np.max(np.abs(lq[A["off"][r]:A["off"][r + 1]] -
                           np.log(T.stern_top2(A["pi"][A["hoff"][r]:A["hoff"][r + 1]], lam)))) for r in range(300))
    e2 = max(np.max(np.abs(lm[A["off"][r]:A["off"][r + 1]] - np.log(T.devig_power(races[r][6], gam))))
             for r in range(300))
    rec("vectorized_T1_equals_stern_top2", e1 < 1e-12, f"{e1:.1e}")
    rec("vectorized_devig_equals_devig_power", e2 < 1e-12, f"{e2:.1e}")

    # 2 P1: race 内 multiset 保存・実際に置換
    rng = np.random.default_rng(3)
    p1 = E.p1_pi(A, rng)
    ok = all(np.array_equal(np.sort(p1[A["hoff"][r]:A["hoff"][r + 1]]), np.sort(A["pi"][A["hoff"][r]:A["hoff"][r + 1]]))
             for r in range(len(races)))
    rec("P1_within_race_permutation_keeps_multiset", ok and not np.array_equal(p1, A["pi"]))

    # 3 P2: donor は同 cell・自分以外 (singleton は自分)
    p2, singles = E.p2_pi(A, np.random.default_rng(4))
    members = {}
    for r, c in enumerate(A["cell"]):
        members.setdefault(int(c), []).append(r)
    ok2, n_self = True, 0
    for r in range(len(races)):
        seg = p2[A["hoff"][r]:A["hoff"][r + 1]]
        cand = [d for d in members[int(A["cell"][r])]
                if np.array_equal(seg, A["pi"][A["hoff"][d]:A["hoff"][d + 1]])]
        if len(members[int(A["cell"][r])]) == 1:
            ok2 &= cand == [r]
            n_self += 1
        else:
            ok2 &= any(d != r for d in cand)
    rec("P2_donor_same_cell_not_self", ok2 and n_self == singles, f"singletons={singles}")

    # 4 year_delta の決定性と P5
    Y = 2019
    f0, f1 = E.fit_window(Y, A)
    lmf, mkf, g0, _ = E.logm_rows(1.0, f0, f1, A)
    wf = A["win_row"][f0:f1] - g0
    a0 = T.fit_temp(lmf, mkf.off, wf)
    d1, fc1 = E.year_delta(Y, A["pi"], 0.85, 1.0, a0, A)
    d2, fc2 = E.year_delta(Y, A["pi"], 0.85, 1.0, a0, A)
    rec("year_delta_deterministic_bitwise", np.array_equal(d1, d2) and fc1 == fc2)
    uni = -np.log(np.repeat(mkf.cnt, mkf.cnt).astype(float))
    fc5 = T.fit_cross(lmf, uni, mkf.off, wf)
    rec("P5_uniform_source_collinear", fc5["collinear"] and abs(fc5["a"] - a0) == 0.0)
    rec("synthetic_T1_source_detected_direction", float(d1.mean()) < 0,
        f"mean Δ={d1.mean():.5f} β={fc1['beta']:.3f} (winners drawn from T1)")

    # 5 γ・λ fit が合成真値を回収
    Ag, _ = synth(per_year=1500, seed=5, gamma_true=1.12)
    E._A.clear()
    E._A.update(Ag)
    gf = E.fit_gamma(*E.fit_window(2023, Ag), Ag)
    Al, _ = synth(per_year=1500, seed=6, lam_true=0.8)
    E._A.clear()
    E._A.update(Al)
    lf = E.fit_lambda(Al["pi"], *E.fit_window(2023, Al), Al)
    rec("fit_gamma_recovers_truth", abs(gf - 1.12) < 0.05, f"γ̂={gf:.4f} (真 1.12)")
    rec("fit_lambda_recovers_truth", abs(lf - 0.8) < 0.05, f"λ̂={lf:.4f} (真 0.8)")
    E._A.clear()

    # 6 cross-check 純関数
    ref = {y: {f"{y}{i:04d}" for i in range(10)} for y in E.EVAL_YEARS}
    dh = {"20190003", "20210007"}
    s18 = {y: set(ref[y]) - dh for y in E.EVAL_YEARS}
    r_ok = E.crosscheck_sets(s18, ref, ref, dh)
    s_bad = {y: set(v) for y, v in s18.items()}
    s_bad[2020].discard("20200005")
    r_bad = E.crosscheck_sets(s_bad, ref, ref, dh)
    ref17_bad = {y: set(v) for y, v in ref.items()}
    ref17_bad[2023].add("20239999")
    r_bad17 = E.crosscheck_sets(s18, ref, ref17_bad, dh)
    rec("crosscheck_match_with_declared_D", r_ok["match_all"] and r_ok["total_D"] == 2)
    rec("crosscheck_detects_missing_race",
        (not r_bad["match_all"]) and r_bad["per_year"]["2020"]["only_in_reference_minus_D"] == ["20200005"])
    rec("crosscheck_detects_exp16a_exp17_disagreement", not r_bad17["match_all"])

    # 7 凍結前は Stage 1 結果 loader / 評価が拒否する
    for fn, name in ((load_outcomes_stage1, "stage1_outcome_loader_refuses_before_final"),
                     (E.build_and_crosscheck, "stage1_build_refuses_before_final")):
        try:
            fn()
            rec(name, False, "実行されてしまった")
        except AssertionError as e:
            rec(name, "0.5-final" in str(e) or "凍結" in str(e), str(e)[:60])

    n_ok = sum(r["pass"] for r in RES)
    out = {"n_tests": len(RES), "n_pass": n_ok, "all_passed": n_ok == len(RES), "tests": RES,
           "elapsed_sec": round(time.time() - t0, 1), "note": "合成データのみ。実 outcome は読まない"}
    (OUT / "stage1_machinery_tests.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                     encoding="utf-8")
    print(f"{n_ok}/{len(RES)} passed ({out['elapsed_sec']}s)")
    sys.exit(0 if out["all_passed"] else 1)


if __name__ == "__main__":
    main()
