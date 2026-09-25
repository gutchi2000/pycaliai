# -*- coding: utf-8 -*-
"""
evaluate_stage1.py — EXP18 Stage 1: UB2 (T1 offset) vs 較正温度 null の rolling 評価 (spec v0.5-final)
=====================================================================================================
手順 (spec.json stage1_protocol_v05 / stage1_race_set_crosscheck / gates.M1 のとおり。v0.5-final 凍結後だけ動く):
  1. 構造 loader で 2013-2023 の eligible race (平地・starter>=5・複勝 Lo/Hi・馬連格子完全) を作る
  2. Stage 1 結果 loader (2013-2023) で DNF (starter − finisher)・top2 同着・top2 が starter 外を除外 → 正式 set
  3. **hard cross-check**: 2019-2023 の正式 set が EXP16A = EXP17 の正式 set − D (2 着同着 race) と年別に完全一致。
     不一致なら Δ を一切計算せずに停止し、差分 race_id を out/stage1_race_set_diff.json に書く
  4. 年 Y ごとに 2013..Y−1 で γ・λ・a0・(a, β) を fit (UB2 = T1、UB1 = T0 参考)
  5. Δ_r = LL(q_cross) − LL(q_temp) (改善が負)、年層化暦日 bootstrap B=10000、年方向、LOO、決定性検査
  6. P1 / P2 各 200 draw (rolling refit)、P4 sanity 200 draw、P5 構成 assert
  7. gate_grade.grade_m1 で等級 (手計算で上書きしない)
2024/2025 は読まない。ROI・候補生成・資金配分・production 変更はしない。
実行: python -m analysis.mcond.exp18_cross_pool_market_tomography_dev.evaluate_stage1
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import minimize_scalar

from . import tomography as T
from .floor_v05 import Mk, day_boot_ci, sub_mk
from .gate_grade import grade_m1, placebo_exceeded
from .loaders import (BASE, HERE, OUT, RESEARCH, dnf_horses, load_outcomes_stage1, load_structure,
                      realized_top2)

SEED_BOOT = 20260930
SEED_PLACEBO = 20260931
BOOT = 10000
BOOT_P4 = 1000
N_PLACEBO = 200
FIT_START = 2013
EVAL_YEARS = [2019, 2020, 2021, 2022, 2023]
WORKERS = 8
ARR = RESEARCH / "stage1_arrays.npz"
REF16 = BASE / "data" / "_research" / "mcond" / "exp16a" / "official_rids_by_year.json"
REF17 = HERE.parent / "exp17_transitive_pl_graph_dev" / "out" / "races_2019_2023.parquet"


# ---------------------------------------------------------------- 1-3 正式 set と cross-check
def crosscheck_sets(s18: dict, ref16: dict, ref17: dict, second_dh: set) -> dict:
    """純関数: EXP18 正式 set が EXP16A (= EXP17) − D (2 着同着 race) と年別に完全一致するか"""
    report = {"per_year": {}, "exp16a_equals_exp17": all(ref16[y] == ref17[y] for y in EVAL_YEARS)}
    ok = report["exp16a_equals_exp17"]
    for y in EVAL_YEARS:
        D = ref16[y] & second_dh
        expect = ref16[y] - D
        only18, onlyref = sorted(s18[y] - expect), sorted(expect - s18[y])
        report["per_year"][str(y)] = {"exp18": len(s18[y]), "exp16a": len(ref16[y]), "exp17": len(ref17[y]),
                                      "D_second_place_dead_heat": len(D), "expected": len(expect),
                                      "only_in_exp18": only18, "only_in_reference_minus_D": onlyref,
                                      "match": not only18 and not onlyref}
        ok &= not only18 and not onlyref
    report["total_exp18"] = sum(len(s18[y]) for y in EVAL_YEARS)
    report["total_reference"] = sum(len(ref16[y]) for y in EVAL_YEARS)
    report["total_D"] = sum(v["D_second_place_dead_heat"] for v in report["per_year"].values())
    report["match_all"] = bool(ok)
    return report


def build_and_crosscheck():
    from .market_build import (base_race_status, pool_matrices, race_arrays, snapshot_index, tan_entropy,
                               umaren_grid_complete)
    spec = json.loads((HERE / "spec.json").read_text(encoding="utf-8"))
    assert spec["version"] == "0.5-final", "spec v0.5-final 凍結前は Stage 1 を実行しない"
    st = load_structure(range(FIT_START, 2024))
    tan, um, info = st["tan"], st["um"], st["info"]
    W, LO, HI, U = pool_matrices(tan, um)
    idx = snapshot_index(tan, um, info)
    elig = []
    for rid, r in idx.iterrows():
        if not (r.get("term_tan") == r.get("term_tan")) or int(r["term_um"]) < 0:
            continue
        at = race_arrays(W, LO, HI, U, int(r["term_tan"]), int(r["term_um"]))
        inf = info.loc[rid] if rid in info.index else None
        if base_race_status(inf, at) != "base" or not umaren_grid_complete(at):
            continue
        elig.append({"rid": rid, "year": int(rid[:4]), "day": int(rid[:8]), "venue": str(inf["venue"]),
                     "n": int(at["n"]), "bans": at["bans"], "win": at["win"], "umaren": at["umaren"],
                     "ent": tan_entropy(at["win"])})
    cuts = np.quantile([e["ent"] for e in elig], [1 / 3, 2 / 3])      # 構造のみ・全 eligible 2013-2023

    outc = load_outcomes_stage1()
    top = realized_top2(outc, max_year=2023).set_index("rid16")
    funnel, formal = {}, []
    for e in elig:
        y = str(e["year"])
        f = funnel.setdefault(y, {"eligible": 0, "no_outcome": 0, "excl_dnf": 0, "excl_top2_dead_heat": 0,
                                  "excl_top2_outside_starters": 0, "formal": 0})
        f["eligible"] += 1
        if e["rid"] not in top.index:
            f["no_outcome"] += 1
            continue
        o = top.loc[e["rid"]]
        if dnf_horses(e["bans"], o["finishers"]):
            f["excl_dnf"] += 1
            continue
        if o["dead_heat_top2"] or o["top2"] is None:
            f["excl_top2_dead_heat"] += 1
            continue
        if not set(o["top2"]) <= set(int(b) for b in e["bans"]):
            f["excl_top2_outside_starters"] += 1
            continue
        f["formal"] += 1
        e["top2"] = tuple(int(x) for x in o["top2"])
        formal.append(e)

    # ---- hard cross-check (2019-2023)
    import pandas as pd
    ref16_raw = json.loads(REF16.read_text(encoding="utf-8"))
    ref16 = {int(y): set(v) for y, v in ref16_raw.items() if int(y) in EVAL_YEARS}
    r17 = pd.read_parquet(REF17, columns=["rid16", "year"])
    ref17 = {y: set(r17.loc[r17["year"] == y, "rid16"].astype(str)) for y in EVAL_YEARS}
    s18 = {y: {e["rid"] for e in formal if e["year"] == y} for y in EVAL_YEARS}
    second_dh = {rid for y in EVAL_YEARS for rid in ref16[y]
                 if rid in top.index and int(top.loc[rid, "n_first"]) == 1 and int(top.loc[rid, "n_second"]) >= 2}
    report = crosscheck_sets(s18, ref16, ref17, second_dh)
    ok = report["match_all"]
    report["funnel"] = funnel
    (OUT / "stage1_race_set_crosscheck.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                                         encoding="utf-8")
    if not ok:
        (OUT / "stage1_race_set_diff.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                                       encoding="utf-8")
        print("[STOP] race set cross-check FAILED — Δ は計算していない。out/stage1_race_set_diff.json")
        sys.exit(2)
    print(f"[crosscheck] PASS exp18={report['total_exp18']} reference={report['total_reference']} "
          f"D={report['total_D']}", flush=True)

    # ---- 配列化 (年・race_id 順)
    formal.sort(key=lambda e: (e["year"], e["rid"]))
    venues = sorted({e["venue"] for e in formal})
    hcnt = np.array([e["n"] for e in formal])
    hoff = np.concatenate([[0], np.cumsum(hcnt)])
    pcnt = hcnt * (hcnt - 1) // 2
    off = np.concatenate([[0], np.cumsum(pcnt)])
    pa, pb, win_row = [], [], []
    for r, e in enumerate(formal):
        a, b = T.pair_index(e["n"])
        pa.append(a + hoff[r])
        pb.append(b + hoff[r])
        i, j = np.searchsorted(e["bans"], e["top2"][0]), np.searchsorted(e["bans"], e["top2"][1])
        k = int(np.flatnonzero((a == min(i, j)) & (b == max(i, j)))[0])
        win_row.append(off[r] + k)
    pi = np.concatenate([(1 / e["win"]) / (1 / e["win"]).sum() for e in formal])
    cell = np.array([f'{e["venue"]}|{e["n"]}|{int(np.searchsorted(cuts, e["ent"]))}' for e in formal])
    _, cell_id = np.unique(cell, return_inverse=True)
    np.savez_compressed(
        ARR, rid=np.array([e["rid"] for e in formal]), year=np.array([e["year"] for e in formal]),
        day=np.array([e["day"] for e in formal]), venue=np.array([venues.index(e["venue"]) for e in formal]),
        n=hcnt, hoff=hoff, off=off, pa=np.concatenate(pa), pb=np.concatenate(pb), win_row=np.array(win_row),
        pi=pi, x=np.concatenate([-np.log(e["umaren"]) for e in formal]), cell=cell_id,
        venues=np.array(venues), ent_cuts=cuts)
    return report


# ---------------------------------------------------------------- 配列上の vectorized 確率
_A = {}


def arrays():
    if "off" in _A:
        return _A
    z = np.load(ARR)
    for k in z.files:
        _A[k] = z[k]
    _A["mk"] = Mk(_A["off"])
    _A["hmk"] = Mk(_A["hoff"])
    _A["race_of_horse"] = np.repeat(np.arange(len(_A["n"])), _A["n"])
    y = _A["year"]
    _A["yr"] = {int(v): (int(np.flatnonzero(y == v)[0]), int(np.flatnonzero(y == v)[-1] + 1)) for v in np.unique(y)}
    return _A


def logm_rows(gamma, r0, r1, A):
    mk, h0, h1 = sub_mk(A["off"], r0, r1)
    return mk.norm_log(gamma * A["x"][h0:h1]), mk, h0, h1


def logq_rows(pi, lam, r0, r1, A):
    """λ 割引 Harville の順不同 top2 (T.stern_top2 と同式、race 内で正規化) を行に対して計算"""
    mk, h0, h1 = sub_mk(A["off"], r0, r1)
    s = np.power(pi, lam)
    S = A["hmk"].sum(s)
    Sr = S[A["race_of_horse"]]
    a, b = A["pa"][h0:h1], A["pb"][h0:h1]
    p = pi[a] * s[b] / (Sr[a] - s[a]) + pi[b] * s[a] / (Sr[b] - s[b])
    return mk.norm_log(np.log(p)), mk, h0, h1


def fit_gamma(r0, r1, A):
    wr = A["win_row"][r0:r1]

    def nll(g):
        lm, mk, h0, _ = logm_rows(g, r0, r1, A)
        return -float(lm[wr - h0].mean())
    return float(minimize_scalar(nll, bounds=(0.5, 2.0), method="bounded", options={"xatol": 1e-7}).x)


def fit_lambda(pi, r0, r1, A):
    wr = A["win_row"][r0:r1]

    def nll(lam):
        lq, mk, h0, _ = logq_rows(pi, lam, r0, r1, A)
        return -float(lq[wr - h0].mean())
    return float(minimize_scalar(nll, bounds=(0.3, 1.5), method="bounded", options={"xatol": 1e-7}).x)


def fit_window(Y, A):
    return A["yr"][FIT_START][0], A["yr"][Y - 1][1]


def year_delta(Y, pi, lam, gam, a0, A):
    """年 Y: ≤Y−1 で (a, β) を fit し、評価年の Δ_r = LL(q_cross) − LL(q_temp) を返す"""
    f0, f1 = fit_window(Y, A)
    lmf, mkf, g0, _ = logm_rows(gam, f0, f1, A)
    lqf = logq_rows(pi, lam, f0, f1, A)[0]
    wf = A["win_row"][f0:f1] - g0
    fc = T.fit_cross(lmf, lqf, mkf.off, wf)
    e0, e1 = A["yr"][Y]
    lme, mke, h0, _ = logm_rows(gam, e0, e1, A)
    lqe = logq_rows(pi, lam, e0, e1, A)[0]
    we = A["win_row"][e0:e1] - h0
    lt = mke.norm_log(a0 * lme)[we]
    lc = (mke.norm_log(fc["a"] * lme) if fc["collinear"] else mke.norm_log(fc["a"] * lme + fc["beta"] * lqe))[we]
    return -lc + lt, fc


def stats(vals, yrs, days, rng, boot=BOOT):
    point = float(vals.mean())
    lo, hi = day_boot_ci(vals, yrs, days, rng, boot)
    ym = {str(y): float(vals[yrs == y].mean()) for y in EVAL_YEARS}
    loo = {str(y): day_boot_ci(vals, yrs, days, rng, boot, drop_year=y)[1] for y in EVAL_YEARS}
    return {"point": point, "ci95": [lo, hi], "year_means": ym,
            "years_improved": int(sum(v < 0 for v in ym.values())), "loo_ci_uppers": loo}


# ---------------------------------------------------------------- placebo worker
_P = {}


def _pinit():
    A = arrays()
    _P.update(json.loads((OUT / "stage1_fits.json").read_text(encoding="utf-8")))


def p1_pi(A, rng):
    keys = rng.random(len(A["pi"]))
    order = np.lexsort((keys, A["race_of_horse"]))
    return A["pi"][order]


def p2_pi(A, rng):
    cell = A["cell"]
    members = {}
    for r, c in enumerate(cell):
        members.setdefault(int(c), []).append(r)
    out = np.empty_like(A["pi"])
    singles = 0
    hoff = A["hoff"]
    for r in range(len(cell)):
        m = members[int(cell[r])]
        if len(m) == 1:
            d = r
            singles += 1
        else:
            d = r
            while d == r:
                d = m[int(rng.integers(len(m)))]
        out[hoff[r]:hoff[r + 1]] = A["pi"][hoff[d]:hoff[d + 1]]
    return out, singles


def task_placebo(args):
    kind, k = args
    A = arrays()
    if not _P:
        _pinit()
    rng = np.random.default_rng([SEED_PLACEBO, kind, k])
    singles = None
    if kind == 1:
        pi = p1_pi(A, rng)
    else:
        pi, singles = p2_pi(A, rng)
    vals = []
    for Y in EVAL_YEARS:
        f = _P["fits"][str(Y)]
        d, _ = year_delta(Y, pi, f["lambda"], f["gamma"], f["a0"], A)
        vals.append(d)
    return kind, k, float(np.concatenate(vals).mean()), singles


# ---------------------------------------------------------------- main
def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    t0 = time.time()
    spec = json.loads((HERE / "spec.json").read_text(encoding="utf-8"))
    assert spec["version"] == "0.5-final"
    floor = spec["stage0"]["power"]["practical_floor_nats"]
    assert isinstance(floor, (int, float)) and floor > 0
    assert spec["stage0"]["power"]["signal_power_gate_pass_v05"] is True, "SIGNAL 検出力 Gate 未通過"
    cross = build_and_crosscheck()                      # 不一致ならここで exit(2)
    A = arrays()
    rng = np.random.default_rng(SEED_BOOT)

    # ---- 4 rolling fits
    fits, d_ub2, d_ub1, h2h, yrs, days, venues, rids = {}, [], [], [], [], [], [], []
    for Y in EVAL_YEARS:
        f0, f1 = fit_window(Y, A)
        gam = fit_gamma(f0, f1, A)
        lam = fit_lambda(A["pi"], f0, f1, A)
        lmf, mkf, g0, _ = logm_rows(gam, f0, f1, A)
        wf = A["win_row"][f0:f1] - g0
        a0 = T.fit_temp(lmf, mkf.off, wf)
        d2, fc2 = year_delta(Y, A["pi"], lam, gam, a0, A)
        d1, fc1 = year_delta(Y, A["pi"], 1.0, gam, a0, A)
        # 決定性: 同じ fit をもう一度 (bit 一致)
        d2b, fc2b = year_delta(Y, A["pi"], lam, gam, a0, A)
        e0, e1 = A["yr"][Y]
        lq1, mke, h0, _ = logq_rows(A["pi"], lam, e0, e1, A)
        lq0 = logq_rows(A["pi"], 1.0, e0, e1, A)[0]
        we = A["win_row"][e0:e1] - h0
        h2h.append(-lq1[we] + lq0[we])
        # P5: 一様 q_LOPO → 共線経路 → Δ = 0
        lmf2 = logm_rows(gam, f0, f1, A)[0]
        uni = -np.log(np.repeat(mkf.cnt, mkf.cnt).astype(float))
        fc5 = T.fit_cross(lmf2, uni, mkf.off, wf)
        lme = logm_rows(gam, e0, e1, A)[0]
        lt = mke.norm_log(a0 * lme)[we]
        lc5 = mke.norm_log(fc5["a"] * lme)[we] if fc5["collinear"] else None
        p5 = float(np.max(np.abs(lc5 - lt))) if lc5 is not None else float("inf")
        fits[str(Y)] = {"gamma": gam, "lambda": lam, "a0": a0, "n_fit_races": int(f1 - f0),
                        "UB2": fc2, "UB1": fc1, "deterministic_refit_bitwise": bool(
                            np.array_equal(d2, d2b) and fc2 == fc2b),
                        "P5_collinear": bool(fc5["collinear"]), "P5_max_abs_delta": p5}
        d_ub2.append(d2)
        d_ub1.append(d1)
        yrs.append(A["year"][e0:e1])
        days.append(A["day"][e0:e1])
        venues.append(A["venue"][e0:e1])
        rids.append(A["rid"][e0:e1])
        print(f"[{Y}] γ={gam:.4f} λ={lam:.4f} a0={a0:.4f} UB2 a={fc2['a']:.4f} β={fc2['beta']:.4f} "
              f"UB1 β={fc1['beta']:.4f} n_fit={f1 - f0}", flush=True)
    d_ub2, d_ub1, h2h = np.concatenate(d_ub2), np.concatenate(d_ub1), np.concatenate(h2h)
    yrs, days, venues = np.concatenate(yrs), np.concatenate(days), np.concatenate(venues)
    assert all(f["P5_collinear"] and f["P5_max_abs_delta"] <= 1e-12 for f in fits.values()), "P5 assert 失敗"
    assert all(f["deterministic_refit_bitwise"] for f in fits.values()), "決定性検査失敗"
    (OUT / "stage1_fits.json").write_text(json.dumps({"fits": fits}, ensure_ascii=False, indent=1,
                                                     default=float), encoding="utf-8")

    # ---- 5 stats
    s2 = stats(d_ub2, yrs, days, rng)
    print(f"[UB2] point={s2['point']:.6f} CI95={s2['ci95']} years={s2['years_improved']}/5", flush=True)

    # ---- 6 placebo
    with ProcessPoolExecutor(WORKERS, initializer=_pinit) as ex:
        pres = list(ex.map(task_placebo, [(k, i) for k in (1, 2) for i in range(N_PLACEBO)], chunksize=2))
    p1 = np.array([r[2] for r in pres if r[0] == 1])
    p2 = np.array([r[2] for r in pres if r[0] == 2])
    p2_singles = [r[3] for r in pres if r[0] == 2][0]
    p1_ok, p2_ok = placebo_exceeded(s2["point"], p1), placebo_exceeded(s2["point"], p2)
    # P4 sanity: 実 fit 値のまま、評価年 outcome を m_cal から再標本化
    p4 = []
    rng4 = np.random.default_rng([SEED_PLACEBO, 4])
    for k in range(N_PLACEBO):
        vals = []
        for Y in EVAL_YEARS:
            f = fits[str(Y)]
            e0, e1 = A["yr"][Y]
            lme, mke, h0, _ = logm_rows(f["gamma"], e0, e1, A)
            lqe = logq_rows(A["pi"], f["lambda"], e0, e1, A)[0]
            key = lme - np.log(-np.log(rng4.random(len(lme))))
            mx = mke.maxv(key)
            hit = np.flatnonzero(key == mke.spread(mx))
            ror = np.repeat(np.arange(mke.R), mke.cnt)
            w = hit[np.unique(ror[hit], return_index=True)[1]]
            fc = f["UB2"]
            lt = mke.norm_log(f["a0"] * lme)[w]
            lc = (mke.norm_log(fc["a"] * lme) if fc["collinear"] else
                  mke.norm_log(fc["a"] * lme + fc["beta"] * lqe))[w]
            vals.append(-lc + lt)
        v = np.concatenate(vals)
        lo, hi = day_boot_ci(v, yrs, days, rng4, BOOT_P4)
        p4.append((float(v.mean()), hi < 0))

    # ---- 7 grade (凍結済み関数)
    seeds_improved = 5 if s2["point"] < 0 else 0
    grade = grade_m1(point=s2["point"], ci_lower=s2["ci95"][0], ci_upper=s2["ci95"][1],
                     years_improved=s2["years_improved"], n_years=5, seeds_improved=seeds_improved, n_seeds=5,
                     loo_ci_uppers=list(s2["loo_ci_uppers"].values()), p1_exceeded=p1_ok, p2_exceeded=p2_ok,
                     floor=floor)

    # ---- secondary (報告のみ)
    s1 = stats(d_ub1, yrs, days, rng)
    s21 = stats(d_ub2 - d_ub1, yrs, days, rng)
    sh = stats(h2h, yrs, days, rng)
    vnames = [str(v) for v in A["venues"]]
    per_venue = {vnames[v]: {"races": int((venues == v).sum()), "mean_delta": float(d_ub2[venues == v].mean())}
                 for v in np.unique(venues)}
    lovo = {}
    for v in np.unique(venues):
        m = venues != v
        lovo[vnames[v]] = day_boot_ci(d_ub2[m], yrs[m], days[m], rng, BOOT)[1]
    stop8 = bool(grade["grade"] == "PASS-PRACTICAL" and any(u >= 0 for u in lovo.values()))
    res = {
        "role": "EXP18 Stage 1 (spec v0.5-final)。UB2 vs 較正温度 null、2019-2023 rolling。2024/2025 未開封",
        "spec_version": spec["version"], "practical_floor_nats": floor,
        "race_set_crosscheck": {k: v for k, v in cross.items() if k != "per_year"} |
                               {"per_year": {y: {k2: v2 for k2, v2 in d.items() if not k2.startswith("only_")}
                                             for y, d in cross["per_year"].items()}},
        "n_eval_races": int(len(d_ub2)), "n_meeting_days": int(len(np.unique(days))),
        "fits": fits,
        "UB2_gate": {"delta_definition": "LL(q_cross) - LL(q_temp), improvement negative", **s2},
        "placebo": {"P1": {"draws": int(len(p1)), "q025": float(np.quantile(p1, 0.025)),
                           "median": float(np.median(p1)), "exceeded": p1_ok},
                    "P2": {"draws": int(len(p2)), "q025": float(np.quantile(p2, 0.025)),
                           "median": float(np.median(p2)), "exceeded": p2_ok, "singleton_cells_races": p2_singles},
                    "P4_sanity": {"draws": len(p4), "mean_delta": float(np.mean([x[0] for x in p4])),
                                  "ci_upper_lt_0_rate": float(np.mean([x[1] for x in p4])),
                                  "note": "sanity only, not gate evidence"},
                    "P5_assert": "all years collinear path and |Δ| <= 1e-12"},
        "seeds": {"deterministic": True, "seeds_improved": seeds_improved},
        "grade": grade,
        "secondary": {"UB1_reference": s1, "UB1_beta_by_year": {y: f["UB1"]["beta"] for y, f in fits.items()},
                      "UB2_minus_UB1": s21, "T1_minus_T0_head_to_head": sh,
                      "U0P_proportional_devig": "gate models identical by construction (power of power absorbed by a, a0)",
                      "per_venue": per_venue, "leave_one_venue_out_ci_upper": lovo,
                      "stop_rule_8_material_sensitivity": stop8},
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (OUT / "stage1_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float),
                                          encoding="utf-8")
    np.savez_compressed(RESEARCH / "stage1_delta_by_race.npz", rid=np.concatenate(rids), delta_ub2=d_ub2,
                        delta_ub1=d_ub1, year=yrs, day=days)
    print(f"[grade] {grade['grade']} — {grade['statement']}")
    print(f"[saved] out/stage1_eval.json ({res['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
