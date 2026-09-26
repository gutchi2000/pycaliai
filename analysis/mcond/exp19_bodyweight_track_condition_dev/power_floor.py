# -*- coding: utf-8 -*-
"""
power_floor.py — EXP19 Stage 0 S0-D: EXP18 v0.5 方式の実務 floor と検出力 (結果ラベル不使用)
==========================================================================================
SPEC §6 S0-D を実装する。seed・rep・格子・補間は実行前に本ファイルで固定。
  生成   log p = log m_base + ε·c − log Z。c = block (A1/B1: W 設計列、A2/B2: WP 列) の race 内中心化・標準化後の
         第 1 主成分方向 (構造データだけで凍結、結果不使用)。真値は alt arm の族に入る (a=1, b=0, θ=ε·v)
  真の Δ 評価 race 上で、期待 logloss 最良の null arm (A1/B1: [log m, s_clean]、A2/B2: + W) に対する KL(p‖q_null)
  推定器 実際に使う offset conditional-logit (models.fit_offset) を、年 Y の n_fit(Y) = 実 fit 窓 race 数と同数の
         合成 outcome で fit。独立な宣言ノイズは足さない
  決済   実 terminal 単勝オッズ上の現金込み Kelly (控除 20% は実オッズに内在)
  floor  rep 平均の期待成長 >= 1e-4/race となる最小の真の Δ (初回交差点と直前点の線形補間)
  検出力 floor の真の効果で Stage 1 と同じ rolling fit・年層化暦日 bootstrap (B=10,000)・年方向・LOO を合成 outcome で
         回し gate_grade.grade で等級。placebo は通過と仮定 (宣言)。PASS_PRACTICAL ∪ PASS_SUBFLOOR の検出力 >= 0.80 が進行条件
**Δ=0 健全性**: 真値 = base 市場のとき期待成長が閾値未満でなければ floor は定義できない。A 系 (base = pre、決済 = terminal)
は pre→terminal のオッズ差だけで正の成長が出るため、凍結仕様のままでは floor を定められない (run 時に記録して A の floor は
算出しない)。A 系は決済に依存しない SIGNAL 検出力 (MDE) だけを参考として出す。
実行: python -m analysis.mcond.exp19_bodyweight_track_condition_dev.power_floor
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

from ..exp18_cross_pool_market_tomography_dev import tomography as T18
from ..exp18_cross_pool_market_tomography_dev.floor_v05 import Mk, sub_mk
from . import models as M
from .gate_grade import grade, grade_a
from .loaders import OUT, RESEARCH

SEED = 20260926
GROWTH_THRESHOLD = 1e-4
# 上限 0.1: 初回実行 (commit 前・結果未開封) で Δ=0.3/0.5 の合成真値が完全分離に近づき Newton が特異行列で停止した。
# 単勝市場で真の Δ=0.1 nats は既に巨大な効果量なので、格子を 0.1 までに固定する (0.1 で未到達なら floor なしと報告)
DELTA_GRID = [0.0, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05,
              0.075, 0.1]
REPS_GROWTH = 24
REPS_POWER_FLOOR = 400
REPS_POWER_OTHER = 200
A_MDE_GRID = [0.0005, 0.001, 0.002, 0.004, 0.008]
REPS_A_MDE = 100
BOOT = 10000
FIT_START = 2016
ZERO_GROWTH_TOL = GROWTH_THRESHOLD           # Δ=0 の期待成長がこれ以上なら floor は定義不能
WORKERS = 8
ARR = RESEARCH / "power_arrays.npz"

_G = {}


def world():
    if "off" in _G:
        return _G
    z = np.load(ARR, allow_pickle=True)
    for k in z.files:
        _G[k] = z[k]
    _G["mk"] = Mk(_G["off"])
    _G["ror"] = np.repeat(np.arange(len(_G["year"])), np.diff(_G["off"]))
    return _G


def gate_races(g):
    """gate ごとの (fit 窓の race 集合を返す関数, 評価 race index)"""
    W = world()
    spec = M.GATES[g]
    yr = W["year"]
    need_wp = spec["alt"] == "WP"
    ok = W["wp_ok"].astype(bool) if need_wp else np.ones(len(yr), bool)
    ev = np.flatnonzero(np.isin(yr, spec["years"]) & ok)
    fits = {Y: np.flatnonzero((yr >= FIT_START) & (yr <= Y - 1) & ok) for Y in spec["years"]}
    return fits, ev


def cols(g):
    W = world()
    spec = M.GATES[g]
    base = W["log_m_pre"] if spec["market"] == "pre" else W["log_m_term"]
    null = np.column_stack([base, W["s_clean"]] + ([W["W"]] if spec["null"] == "W" else []))
    blk = W["W"] if spec["alt"] == "W" else W["WP"]
    alt = np.column_stack([null, blk])
    return base, null, alt, blk


def direction(g):
    """block の race 内中心化・列標準化後の第 1 主成分 (構造データのみ、2016-2023 の全 race 行を 1/7 間引き)"""
    W = world()
    _, _, _, blk = cols(g)
    C = M.race_center(blk, W["off"])
    sd = C.std(0)
    keep = sd > 1e-12
    Z = C[:, keep] / sd[keep]
    _, _, vt = np.linalg.svd(Z[::7], full_matrices=False)             # 行間引きで主成分方向を得る (構造のみ)
    v = np.zeros(blk.shape[1])
    v[np.flatnonzero(keep)] = vt[0] / sd[keep]
    c = C @ v
    c = c / np.sqrt(np.mean(c * c))
    return c, v


def rows_of(races):
    W = world()
    off = W["off"]
    return np.concatenate([np.arange(off[r], off[r + 1]) for r in races])


def sub_off(races):
    W = world()
    cnt = np.diff(W["off"])[races]
    return np.concatenate([[0], np.cumsum(cnt)])


def expected_fit(X, p, off, iters=60):
    """期待対数尤度 Σ p log q を最大化する null 係数 (race セグメント上の Newton)"""
    k = X.shape[1]
    th = np.zeros(k)
    th[0] = 1.0
    start, cnt = off[:-1], np.diff(off)
    for _ in range(iters):
        eta = X @ th
        mx = np.maximum.reduceat(eta, start)
        e = np.exp(eta - np.repeat(mx, cnt))
        q = e / np.repeat(np.add.reduceat(e, start), cnt)
        Ex = np.add.reduceat(q[:, None] * X, start)
        g = (np.add.reduceat(p[:, None] * X, start) - Ex).sum(0)
        Exx = np.add.reduceat(q[:, None, None] * X[:, :, None] * X[:, None, :], start)
        H = -(Exx - Ex[:, :, None] * Ex[:, None, :]).sum(0)
        step = np.linalg.lstsq(H, g, rcond=None)[0]
        th = th - step
        if np.max(np.abs(step)) < 1e-12:
            break
    return th


def truth(g, eps):
    W = world()
    base, _, _, _ = cols(g)
    c, _ = direction(g)
    return W["mk"].norm_log(base + eps * c)


def delta_true(g, eps):
    _, ev = gate_races(g)
    _, null, _, _ = cols(g)
    lp_all = truth(g, eps)
    rows = rows_of(ev)
    off = sub_off(ev)
    lp, p = lp_all[rows], np.exp(lp_all[rows])
    th = expected_fit(null[rows], p, off)
    lq = M.log_probs(null[rows], th, off)
    kl = np.add.reduceat(p * (lp - lq), off[:-1])
    return float(kl.mean())


def calibrate(g, target):
    if target <= 0:
        return 0.0
    lo, hi = 0.0, 0.05
    while delta_true(g, hi) < target:
        hi *= 2
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if delta_true(g, mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10 * hi:
            break
    return 0.5 * (lo + hi)


def draw(lp, rng):
    W = world()
    key = lp - np.log(-np.log(rng.random(len(lp))))
    mx = W["mk"].maxv(key)
    hit = np.flatnonzero(key == W["mk"].spread(mx))
    return hit[np.unique(W["ror"][hit], return_index=True)[1]]


def fit_arm(X, races, win):
    rows = rows_of(races)
    off = sub_off(races)
    pos = np.full(int(world()["off"][-1]), -1)
    pos[rows] = np.arange(len(rows))
    return M.fit_offset(X[rows], off, pos[win[races]])


def growth_rep(g, eps, seed):
    W = world()
    rng = np.random.default_rng(seed)
    lp = truth(g, eps)
    p = np.exp(lp)
    win = draw(lp, rng)
    fits, ev = gate_races(g)
    _, _, alt, _ = cols(g)
    yr = W["year"]
    tot, n, nfb = 0.0, 0, 0
    for Y, fr in fits.items():
        f = fit_arm(alt, fr, win)
        nfb += int(f["fallback_damped"])
        er = ev[yr[ev] == Y]
        rows = rows_of(er)
        off = sub_off(er)
        lq = M.log_probs(alt[rows], f["theta"], off)
        q, pp, oo = np.exp(lq), p[rows], W["odds_term"][rows]
        for i in range(len(er)):
            a, b = off[i], off[i + 1]
            tot += M.kelly_growth(q[a:b], oo[a:b], pp[a:b])
        n += len(er)
    return (tot / n, nfb)


def oracle_growth(g, eps):
    W = world()
    lp = truth(g, eps)
    _, ev = gate_races(g)
    tot = 0.0
    for r in ev:
        a, b = W["off"][r], W["off"][r + 1]
        pp = np.exp(lp[a:b])
        tot += M.kelly_growth(pp, W["odds_term"][a:b], pp)
    return tot / len(ev)


def day_boot(vals, yrs, days, years, rng, drop=None):
    sums, cnts = [], []
    for y in years:
        if y == drop:
            continue
        s = yrs == y
        dd, inv = np.unique(days[s], return_inverse=True)
        ds = np.bincount(inv, weights=vals[s], minlength=len(dd))
        dc = np.bincount(inv, minlength=len(dd)).astype(float)
        pick = rng.integers(0, len(dd), size=(BOOT, len(dd)))
        sums.append(ds[pick].sum(1))
        cnts.append(dc[pick].sum(1))
    return np.sum(sums, 0) / np.sum(cnts, 0)


def power_rep(g, eps, floor, seed, holm_alpha=None):
    """Stage 1 と同じ集計を合成 outcome で。A 系は Holm の保守側 (α/2 = 0.0125) で近似"""
    W = world()
    rng = np.random.default_rng(seed)
    spec = M.GATES[g]
    lp = truth(g, eps)
    win = draw(lp, rng)
    fits, ev = gate_races(g)
    _, null, alt, _ = cols(g)
    yr = W["year"]
    dv, nfb = [], 0
    for Y, fr in fits.items():
        fn, fa = fit_arm(null, fr, win), fit_arm(alt, fr, win)
        nfb += int(fn["fallback_damped"]) + int(fa["fallback_damped"])
        er = ev[yr[ev] == Y]
        rows = rows_of(er)
        off = sub_off(er)
        pos = np.full(int(W["off"][-1]), -1)
        pos[rows] = np.arange(len(rows))
        w = pos[win[er]]
        ln = M.log_probs(null[rows], fn["theta"], off)[w]
        la = M.log_probs(alt[rows], fa["theta"], off)[w]
        dv.append(-la + ln)
    dv = np.concatenate(dv)
    yrs, days = yr[ev], W["day"][ev]
    boot = day_boot(dv, yrs, days, spec["years"], rng)
    alpha = 0.0125 if spec["holm"] else 0.025
    up0 = float(np.quantile(boot, 1 - alpha))
    ym = [float(dv[yrs == y].mean()) for y in spec["years"]]
    loo = [float(np.quantile(day_boot(dv, yrs, days, spec["years"], rng, drop=y), 1 - alpha)) for y in spec["years"]]
    if g.startswith("A"):          # v0.3: A は SIGNAL / FAIL だけ (経済 floor なし)
        gr = grade_a(gate=g, point=float(dv.mean()), signal_ok=up0 < 0,
                     years_improved=int(sum(v < 0 for v in ym)), n_years=len(spec["years"]),
                     min_years=spec["min_years"], loo_ci_uppers=loo, placebo={k: True for k in spec["placebos"]})
    else:
        fl = floor if floor else 1e9
        gr = grade(gate=g, point=float(dv.mean()), signal_ok=up0 < 0, practical_ok=up0 < -fl,
                   years_improved=int(sum(v < 0 for v in ym)), n_years=len(spec["years"]), min_years=spec["min_years"],
                   loo_ci_uppers=loo, placebo={k: True for k in spec["placebos"]}, floor=fl)
    return {"grade": gr["grade"], "point": float(dv.mean()), "upper": up0, "fallbacks": nfb}


def task(args):
    kind, g, eps, extra, seed = args
    if kind == "growth":
        return (kind, g, extra, growth_rep(g, eps, seed))
    return (kind, g, extra, power_rep(g, eps, extra[1], seed))


def task_cal(args):
    g, d = args
    eps = calibrate(g, d)
    return g, d, eps, delta_true(g, eps), oracle_growth(g, eps)


def floor_from_curve(curve):
    xs = sorted(curve)
    for i, x in enumerate(xs):
        if curve[x] >= GROWTH_THRESHOLD:
            if i == 0:
                return float(x)
            x0 = xs[i - 1]
            return float(x0 + (GROWTH_THRESHOLD - curve[x0]) * (x - x0) / (curve[x] - curve[x0]))
    return None


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def run_map(ex, fn, items, label, chunksize=2):
    """進捗を 5% ごとに表示しながら map する (結果順は入力順)"""
    out, n = [], len(items)
    step = max(1, n // 20)
    t0 = time.time()
    for k, r in enumerate(ex.map(fn, items, chunksize=chunksize), 1):
        out.append(r)
        if k % step == 0 or k == n:
            el = time.time() - t0
            print(f"[{label}] {k}/{n} ({el:.0f}s, remaining ~{el / k * (n - k):.0f}s)", flush=True)
    return out


def main(run_gates=tuple(M.GATES), out_name="power_floor.json", v03=False):
    """v03=True: A は経済 floor を持たず MDE だけを出す。seed は gate の M.GATES 内の位置 i から作るので、
    部分集合で回しても v0.2 と同じ seed になる"""
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    t0 = time.time()
    world()
    run = [g for g in M.GATES if g in run_gates]
    res = {"seed": SEED, "seed_fixed_before_run": True, "delta_grid": DELTA_GRID, "reps_growth": REPS_GROWTH,
           "boot": BOOT, "fit_start": FIT_START, "gates_run": run, "v03": v03,
           "wp_columns": [str(c) for c in world()["wp_cols"]], "gates": {}}
    with ProcessPoolExecutor(WORKERS) as ex:
        cal = run_map(ex, task_cal, [(g, d) for g in run for d in DELTA_GRID], "calibrate", chunksize=1)
        calm = {(g, d): (e, dt, og) for g, d, e, dt, og in cal}
        sane = {g: calm[(g, 0.0)][2] < ZERO_GROWTH_TOL for g in run}
        econ = {g: sane[g] and not (v03 and g.startswith("A")) for g in run}
        print("[Delta=0 oracle growth] " + json.dumps({g: calm[(g, 0.0)][2] for g in run}), flush=True)
        tasks = [("growth", g, (d,), calm[(g, d)][0], SEED + 1000 * i + 17 * j + r)
                 for i, g in enumerate(M.GATES) if g in run and econ[g]
                 for j, d in enumerate(DELTA_GRID) for r in range(REPS_GROWTH)]
        tasks = [(k, g, e, x, s) for (k, g, x, e, s) in tasks]
        gro = run_map(ex, task, tasks, "growth") if tasks else []
        floors = {}
        for g in run:
            n_fit = {Y: int(len(fr)) for Y, fr in gate_races(g)[0].items()}
            info = {"market": M.GATES[g]["market"], "block": M.GATES[g]["alt"], "n_fit": n_fit,
                    "n_eval_races": int(len(gate_races(g)[1])),
                    "delta0_oracle_growth": calm[(g, 0.0)][2], "delta0_sane": bool(sane[g])}
            if econ[g]:
                curve = {d: float(np.mean([r[3][0] for r in gro if r[1] == g and r[2][0] == d])) for d in DELTA_GRID}
                se = {d: float(np.std([r[3][0] for r in gro if r[1] == g and r[2][0] == d], ddof=1) / np.sqrt(REPS_GROWTH))
                      for d in DELTA_GRID}
                fbk = {d: int(sum(r[3][1] for r in gro if r[1] == g and r[2][0] == d)) for d in DELTA_GRID}
                floors[g] = floor_from_curve(curve)
                info.update({"growth_curve": {str(d): {"eps": calm[(g, d)][0], "delta_true": calm[(g, d)][1],
                                                       "growth": curve[d], "mc_se": se[d],
                                                       "oracle_growth": calm[(g, d)][2],
                                                       "damped_fallback_fits": fbk[d]} for d in DELTA_GRID},
                             "practical_floor_nats": floors[g]})
            else:
                floors[g] = None
                info["practical_floor_nats"] = None
                info["floor_status"] = ("v0.3: A is signal-only; no economic floor (MDE reported only)"
                                        if (v03 and g.startswith("A")) else
                                        "undefined_under_frozen_spec: truth anchored at pre market but settled at "
                                        "terminal odds gives positive expected growth at Delta=0")
            res["gates"][g] = info
            print(f"[{g}] floor={floors[g]} sane={sane[g]}", flush=True)
        LAB = {"floor": 1, "0.5x": 2, "2x": 3, "zero": 4}
        plan = []
        for i, g in enumerate(M.GATES):
            if g not in run:
                continue
            if floors[g] is not None:
                for lab, mult, reps in (("floor", 1.0, REPS_POWER_FLOOR), ("0.5x", 0.5, REPS_POWER_OTHER),
                                        ("2x", 2.0, REPS_POWER_OTHER), ("zero", 0.0, REPS_POWER_OTHER)):
                    plan.append((g, floors[g] * mult, lab, floors[g], reps, SEED + 500000 + 100000 * i + 10000 * LAB[lab]))
            elif g.startswith("A"):
                for k, d in enumerate(A_MDE_GRID):
                    plan.append((g, d, f"mde_{d}", None, REPS_A_MDE, SEED + 900000 + 100000 * i + 1000 * k))
        pcal = run_map(ex, task_cal, [(g, d) for g, d, *_ in plan], "calibrate_power", chunksize=1)
        ptasks = []
        for (g, d, lab, fl, reps, s0), (_, _, eps, dt, _) in zip(plan, pcal):
            ptasks += [("power", g, eps, (lab, fl), s0 + r) for r in range(reps)]
        pw = run_map(ex, task, ptasks, "power")
    for g in run:
        out = {}
        labs = sorted({r[2][0] for r in pw if r[1] == g})
        for lab in labs:
            rr = [r[3] for r in pw if r[1] == g and r[2][0] == lab]
            n = len(rr)
            sig = sum(x["grade"] in ("PASS_SUBFLOOR", "PASS_PRACTICAL", "SIGNAL") for x in rr)
            pra = sum(x["grade"] == "PASS_PRACTICAL" for x in rr)
            out[lab] = {"reps": n, "signal_power": sig / n, "signal_wilson95": wilson(sig, n),
                        "practical_power": pra / n, "practical_wilson95": wilson(pra, n),
                        "median_point": float(np.median([x["point"] for x in rr])),
                        "damped_fallback_fits": int(sum(x["fallbacks"] for x in rr))}
        res["gates"][g]["power"] = out
        if floors[g] is not None:
            res["gates"][g]["progression_power_pass"] = bool(out["floor"]["signal_power"] >= 0.80)
    res["elapsed_sec"] = round(time.time() - t0, 1)
    (OUT / out_name).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps({g: {"floor": res["gates"][g]["practical_floor_nats"],
                          "power": {k: v["signal_power"] for k, v in res["gates"][g]["power"].items()}}
                      for g in run}, ensure_ascii=False))
    print(f"[saved] out/{out_name} ({res['elapsed_sec']}s)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "v03":
        # v0.3 一回限りの縮約 WP 再監査: A2 (MDE のみ) と B2 (floor・検出力)。seed・fit 窓・bootstrap・補間は v0.2 と同一
        main(run_gates=("A2", "B2"), out_name="power_floor_v03.json", v03=True)
    else:
        main()
