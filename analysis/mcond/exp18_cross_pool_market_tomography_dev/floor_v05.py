# -*- coding: utf-8 -*-
"""
floor_v05.py — EXP18 v0.5 S0-E: 実務床の再構築 (推定器の有限標本 fit による主方式)
====================================================================================
v0.4 の宣言慣習「per-pair log 確率へ SD=sqrt(2Δ) の独立ノイズ」は廃止した
(per-race logloss 差の SD を別の量へ転用しており、真の利得を構成上ほぼ打ち消していた)。
v0.5 の主方式は**宣言的な独立ノイズを加えず**、凍結済みの実際の推定器 q_cross を有限標本から
fit したときの誤差だけで推定誤差を表す。**結果ラベルは使わない** (≤2018 の formal race 件数だけは
≤2018 結果 loader から数える。2019-2023 の着順・払戻・realized top2 は読まない)。

合成世界 (市場構造は 2013-2023 の実 terminal 市場):
  m_cal   = power-law de-vig(実 terminal 馬連オッズ, γ)、γ・λ は ≤2018 dry-run fit 値 (anchor_le2018.json)
  q_T1    = λ 割引 Harville (実 terminal 単勝 π)
  方向 c  = race 内中心化した log q_T1 − log m_cal
  真値    主方式 (ρ=1):  log p = log m_cal + ε·c   (= q_cross 族 a=1−ε, β=ε の中。正しく指定)
          感度 (ρ=0.5): log p = log m_cal + ε·(ρ·c + sqrt(1−ρ²)·v)
                        v は race 内で [1, log m_cal, log q_T1] に直交化し ‖v_r‖=‖c_r‖ に揃えた固定乱数方向。
                        推定器が回収できるのは ρ·c 成分だけ (真の log 傾きの振幅の 50%)
  真の Δ  = 評価 race (2019-2023) 上で、期待 logloss 最良の温度 null q_temp(a0*) に対する KL(p‖q_temp) の平均。
          ε は真の Δ が格子値になるよう二分法で較正
  fit     各年 Y について ≤Y−1 の合成 formal race (件数 n_fit(Y) = 実 formal race 数) の合成 winner から
          凍結済み fit_cross で (a, β) を実際に fit (λ・γ は固定。λ 推定誤差は含めない = 床は楽観側)
  決済    年 Y の合成評価 race で、実 terminal 馬連オッズに対し fit 済み q_cross で現金込み Kelly を組み、
          真の p の下での期待対数成長 (race 平均) を解析的に計算
  床      rep 平均の期待成長が 1e-4 / race 以上となる最小の真の Δ (格子の線形補間。初回交差の前の点と結ぶ)

n_fit(Y): ≤2018 は実 formal race 数 (障害・DNF (starter − finisher)・top2 同着を除外した実数)。
          2019..Y−1 は結果を読めないので、構造 eligible 件数 × (≤2018 の formal / eligible 保持率) を
          seed 固定の一様抽出で選ぶ (件数を合わせるだけ。どの race かは結果に依存しない)

実行:
  python -m analysis.mcond.exp18_cross_pool_market_tomography_dev.floor_v05 cache
  python -m analysis.mcond.exp18_cross_pool_market_tomography_dev.floor_v05 floor
  python -m analysis.mcond.exp18_cross_pool_market_tomography_dev.floor_v05 power
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

from . import tomography as T
from .loaders import OUT, RESEARCH, dnf_horses, load_outcomes, load_structure, realized_top2

# ---------------------------------------------------------------- 実行前に固定した設定
SEED = 20260927                      # 主 seed (floor・power 共通の基底)
SEED_SUBSET = 20260928               # 2019+ の formal 件数合わせの抽出
SEED_V = 20260929                    # ρ<1 の直交方向 v
RHO_PRIMARY = 1.0
RHO_SENS = 0.5
GROWTH_THRESHOLD = 1e-4
DELTA_GRID = [0.0, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03,
              0.05, 0.075, 0.1, 0.2, 0.3, 0.5]
REPS_GROWTH = 24
POWER_REPS_FLOOR = 400
POWER_REPS_OTHER = 200
POWER_MULTIPLIERS = (0.5, 2.0)
BOOT_POWER = 1000
FIT_START = 2013
EVAL_YEARS = [2019, 2020, 2021, 2022, 2023]
WORKERS = 8
CACHE = RESEARCH / "floor_v05_markets.npz"
INTERPOLATION = "linear between the first grid point with mean growth >= threshold and the previous grid point"


# ---------------------------------------------------------------- segment utilities
class Mk:
    def __init__(self, off):
        self.off = np.asarray(off, dtype=np.int64)
        self.start = self.off[:-1]
        self.cnt = np.diff(self.off)
        self.R = len(self.cnt)

    def sum(self, x):
        return np.add.reduceat(x, self.start, axis=0)

    def spread(self, v):
        return np.repeat(v, self.cnt, axis=0)

    def maxv(self, x):
        return np.maximum.reduceat(x, self.start)

    def lse(self, x):
        mx = self.maxv(x)
        return np.log(self.sum(np.exp(x - self.spread(mx)))) + mx

    def norm_log(self, x):
        return x - self.spread(self.lse(x))

    def center(self, x):
        return x - self.spread(self.sum(x) / self.cnt)


def sub_mk(off, r0, r1):
    """race 区間 [r0, r1) の offsets と行区間"""
    h0, h1 = int(off[r0]), int(off[r1])
    return Mk(off[r0:r1 + 1] - h0), h0, h1


# ---------------------------------------------------------------- cache (結果ラベル不使用・≤2018 件数のみ)
def build_cache():
    from .market_build import base_race_status, pool_matrices, race_arrays, snapshot_index, umaren_grid_complete
    t0 = time.time()
    anc = json.loads((OUT / "anchor_le2018.json").read_text(encoding="utf-8"))
    gamma = anc["dryrun_fits_terminal_le2018"]["gamma_powerlaw_devig"]
    lam = anc["dryrun_fits_terminal_le2018"]["lambda_T1"]
    st = load_structure(range(FIT_START, 2024))
    tan, um, info = st["tan"], st["um"], st["info"]
    W, LO, HI, U = pool_matrices(tan, um)
    idx = snapshot_index(tan, um, info)
    races = []
    for rid, r in idx.iterrows():
        if not (r.get("term_tan") == r.get("term_tan")) or int(r["term_um"]) < 0:
            continue
        at = race_arrays(W, LO, HI, U, int(r["term_tan"]), int(r["term_um"]))
        inf = info.loc[rid] if rid in info.index else None
        if base_race_status(inf, at) != "base" or not umaren_grid_complete(at):
            continue
        pi = (1 / at["win"]) / (1 / at["win"]).sum()
        races.append((int(rid[:4]), rid, at["bans"], at["umaren"],
                      T.devig_power(at["umaren"], gamma), T.stern_top2(pi, lam)))
    races.sort(key=lambda x: (x[0], x[1]))
    eligible_by_year = {}
    for x in races:
        eligible_by_year[x[0]] = eligible_by_year.get(x[0], 0) + 1

    # ≤2018 の実 formal set (障害は構造で除外済み。DNF = starter − finisher、top2 同着を除外)
    outc = load_outcomes(2018)
    assert int(outc["year"].max()) <= 2018
    top = realized_top2(outc).set_index("rid16")
    formal_le2018 = set()
    for y, rid, bans, *_ in races:
        if y > 2018 or rid not in top.index:
            continue
        o = top.loc[rid]
        if dnf_horses(bans, o["finishers"]) or o["dead_heat_top2"] or o["top2"] is None:
            continue
        if not set(o["top2"]) <= set(int(b) for b in bans):
            continue
        formal_le2018.add(rid)
    elig_le2018 = sum(v for k, v in eligible_by_year.items() if k <= 2018)
    retention = len(formal_le2018) / elig_le2018

    rng = np.random.default_rng(SEED_SUBSET)
    keep = []
    formal_by_year = {}
    for y in sorted(eligible_by_year):
        ids = [i for i, x in enumerate(races) if x[0] == y]
        if y <= 2018:
            sel = [i for i in ids if races[i][1] in formal_le2018]
        else:
            k = int(round(retention * len(ids)))
            sel = sorted(rng.choice(ids, size=k, replace=False).tolist())
        keep.extend(sel)
        formal_by_year[y] = len(sel)
    keep = sorted(keep)
    sel_races = [races[i] for i in keep]
    cnt = np.array([len(x[3]) for x in sel_races])
    off = np.concatenate([[0], np.cumsum(cnt)])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE, rid=np.array([x[1] for x in sel_races]), year=np.array([x[0] for x in sel_races]), day=np.array([int(x[1][:8]) for x in sel_races]),
        off=off, logm=np.log(np.concatenate([x[4] for x in sel_races])),
        logq=np.log(np.concatenate([x[5] for x in sel_races])),
        odds=np.concatenate([x[3] for x in sel_races]).astype(float))
    n_fit = {Y: int(sum(v for k, v in formal_by_year.items() if FIT_START <= k <= Y - 1)) for Y in EVAL_YEARS}
    meta = {"gamma": gamma, "lambda": lam, "eligible_by_year": {str(k): v for k, v in eligible_by_year.items()},
            "formal_by_year_sim": {str(k): v for k, v in formal_by_year.items()},
            "formal_le2018_exact": len(formal_le2018), "eligible_le2018": elig_le2018,
            "retention_le2018": retention, "n_fit_by_eval_year": {str(k): v for k, v in n_fit.items()},
            "n_eval_by_year": {str(y): formal_by_year[y] for y in EVAL_YEARS},
            "pairs": int(off[-1]), "races": len(sel_races),
            "outcome_loader_max_year": int(outc["year"].max()), "elapsed_sec": round(time.time() - t0, 1)}
    (OUT / "floor_v05_cache_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                                   encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))
    return meta


# ---------------------------------------------------------------- 合成世界
_G = {}


def load_world():
    if "logm" in _G:
        return _G
    z = np.load(CACHE)
    for k in z.files:
        _G[k] = z[k]
    mk = Mk(_G["off"])
    _G["mk"] = mk
    _G["race_of_row"] = np.repeat(np.arange(mk.R), mk.cnt)
    _G["c"] = mk.center(_G["logq"] - _G["logm"])
    _G["v"] = orth_direction(_G["logm"], _G["logq"], _G["c"], mk, SEED_V)
    y = _G["year"]
    _G["yr_rng"] = {int(yy): (int(np.flatnonzero(y == yy)[0]), int(np.flatnonzero(y == yy)[-1] + 1))
                    for yy in np.unique(y)}
    ev0 = _G["yr_rng"][EVAL_YEARS[0]][0]
    ev1 = _G["yr_rng"][EVAL_YEARS[-1]][1]
    _G["ev_r"] = (ev0, ev1)
    return _G


def orth_direction(logm, logq, c, mk, seed):
    """race 内で [1, log m, log q] に直交化し、‖v_r‖ = ‖c_r‖ に揃えた固定乱数方向"""
    rng = np.random.default_rng(seed)
    v = rng.normal(0.0, 1.0, len(logm))
    X = np.column_stack([np.ones_like(logm), logm, logq])
    G = mk.sum(X[:, :, None] * X[:, None, :])
    h = mk.sum(X * v[:, None])
    beta = np.einsum("rij,rj->ri", np.linalg.pinv(G), h)
    v = v - (X * mk.spread(beta)).sum(1)
    nc = np.sqrt(mk.sum(c * c))
    nv = np.sqrt(mk.sum(v * v))
    return v * mk.spread(nc / np.maximum(nv, 1e-300))


def direction(rho, W=None):
    W = W or load_world()
    if rho == 1.0:
        return W["c"]
    return rho * W["c"] + np.sqrt(1.0 - rho * rho) * W["v"]


def truth_logp(eps, d, W=None):
    W = W or load_world()
    return W["mk"].norm_log(W["logm"] + eps * d)


def fit_temp_expected(logm, p, mk, a=1.0, iters=60):
    """期待 logloss 最良の温度 a0* (Σ_r Σ_i p_i log q_a,i を最大化する Newton)"""
    for _ in range(iters):
        lq = mk.norm_log(a * logm)
        q = np.exp(lq)
        eq = mk.sum(q * logm)
        g = float((mk.sum(p * logm) - eq).sum())
        h = float(-(mk.sum(q * logm * logm) - eq * eq).sum())
        step = g / h
        a -= step
        if abs(step) < 1e-13:
            break
    return a


def delta_true(eps, d, W=None):
    """評価 race (2019-2023) 上の KL(p ‖ q_temp(a0*)) の平均と a0*"""
    W = W or load_world()
    r0, r1 = W["ev_r"]
    mk, h0, h1 = sub_mk(W["off"], r0, r1)
    lm = W["logm"][h0:h1]
    lp = mk.norm_log(lm + eps * d[h0:h1])
    p = np.exp(lp)
    a0 = fit_temp_expected(lm, p, mk)
    lq = mk.norm_log(a0 * lm)
    return float(mk.sum(p * (lp - lq)).mean()), a0


def calibrate_eps(target, d, W=None, iters=60):
    if target <= 0:
        return 0.0
    W = W or load_world()
    lo, hi = 0.0, 0.1
    while delta_true(hi, d, W)[0] < target:
        hi *= 2.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if delta_true(mid, d, W)[0] < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12 * max(hi, 1e-12):
            break
    return 0.5 * (lo + hi)


def draw_winners(lp, W, rng):
    mk = W["mk"]
    key = lp - np.log(-np.log(rng.random(len(lp))))
    mx = mk.maxv(key)
    hit = np.flatnonzero(key == mk.spread(mx))
    return hit[np.unique(W["race_of_row"][hit], return_index=True)[1]]


def fit_year(Y, win, W, with_temp=False):
    """≤Y−1 (FIT_START..Y−1) の合成 formal race で凍結済み fit_cross (と fit_temp) を実行"""
    r0 = W["yr_rng"][FIT_START][0]
    r1 = W["yr_rng"][Y - 1][1]
    mk, h0, h1 = sub_mk(W["off"], r0, r1)
    wf = win[r0:r1] - h0
    fc = T.fit_cross(W["logm"][h0:h1], W["logq"][h0:h1], mk.off, wf)
    a0 = T.fit_temp(W["logm"][h0:h1], mk.off, wf) if with_temp else None
    return fc, a0, r1 - r0


def cross_logq(fc, lm, lq, mk):
    if fc["collinear"]:
        return mk.norm_log(fc["a"] * lm)
    return mk.norm_log(fc["a"] * lm + fc["beta"] * lq)


def kelly_growth_year(Y, lqc, p, W):
    """年 Y の評価 race で現金込み Kelly (実 terminal 馬連オッズ) を組み、真の p の下の期待成長を合計"""
    r0, r1 = W["yr_rng"][Y]
    off = W["off"]
    h0 = int(off[r0])
    qc = np.exp(lqc)
    g_sum, entered, stake = 0.0, 0, 0.0
    for r in range(r0, r1):
        a, b = int(off[r]) - h0, int(off[r + 1]) - h0
        g, st, ns = T.kelly_cash_growth(qc[a:b], W["odds"][off[r]:off[r + 1]], p[a:b])
        g_sum += g
        if ns:
            entered += 1
            stake += st
    return g_sum, entered, stake, r1 - r0


# ---------------------------------------------------------------- tasks
def task_calibrate(args):
    rho, target = args
    W = load_world()
    d = direction(rho, W)
    eps = calibrate_eps(target, d, W)
    dt, a0 = delta_true(eps, d, W)
    # oracle 成長 (q = p、fit しない) = 上限の参考
    lp = truth_logp(eps, d, W)
    p = np.exp(lp)
    g_or, n_or = 0.0, 0
    for Y in EVAL_YEARS:
        r0, r1 = W["yr_rng"][Y]
        h0, h1 = int(W["off"][r0]), int(W["off"][r1])
        g, *_ , n = kelly_growth_year(Y, lp[h0:h1], p[h0:h1], W)
        g_or += g
        n_or += n
    return {"rho": rho, "target": target, "eps": eps, "delta_true": dt, "a0_star": a0,
            "oracle_growth_per_race": g_or / n_or}


def growth_rep(rho, eps, rep_seed, W=None):
    W = W or load_world()
    rng = np.random.default_rng(rep_seed)
    d = direction(rho, W)
    lp = truth_logp(eps, d, W)
    p = np.exp(lp)
    win = draw_winners(lp, W, rng)
    g_tot, n_tot, ent, stk = 0.0, 0, 0, 0.0
    fits = {}
    for Y in EVAL_YEARS:
        fc, _, nfit = fit_year(Y, win, W)
        r0, r1 = W["yr_rng"][Y]
        mk, h0, h1 = sub_mk(W["off"], r0, r1)
        lqc = cross_logq(fc, W["logm"][h0:h1], W["logq"][h0:h1], mk)
        g, e, s, n = kelly_growth_year(Y, lqc, p[h0:h1], W)
        g_tot += g
        n_tot += n
        ent += e
        stk += s
        fits[str(Y)] = {"a": fc["a"], "beta": fc["beta"], "collinear": fc["collinear"], "n_fit": nfit}
    return {"growth_per_race": g_tot / n_tot, "share_entered": ent / n_tot,
            "mean_stake_when_entered": (stk / ent) if ent else 0.0, "fits": fits}


def task_growth(args):
    rho_i, d_i, rep, rho, eps = args
    return (rho_i, d_i, rep, growth_rep(rho, eps, [SEED, rho_i, d_i, rep]))


def floor_from_curve(curve: dict, thr: float = GROWTH_THRESHOLD):
    xs = sorted(curve)
    ys = [curve[x] for x in xs]
    for i, (x, y) in enumerate(zip(xs, ys)):
        if y >= thr:
            if i == 0:
                return float(x)
            x0, y0 = xs[i - 1], ys[i - 1]
            return float(x0 + (thr - y0) * (x - x0) / (y - y0))
    return None


# ---------------------------------------------------------------- Stage 1 と同じ判定を合成 outcome で回す (検出力)
def day_boot_ci(vals, yrs, days, rng, boot, drop_year=None, level=0.95):
    sums, cnts = [], []
    for y in EVAL_YEARS:
        if y == drop_year:
            continue
        sel = yrs == y
        dd, inv = np.unique(days[sel], return_inverse=True)
        ds = np.bincount(inv, weights=vals[sel], minlength=len(dd))
        dc = np.bincount(inv, minlength=len(dd)).astype(float)
        pick = rng.integers(0, len(dd), size=(boot, len(dd)))
        sums.append(ds[pick].sum(1))
        cnts.append(dc[pick].sum(1))
    tot = np.sum(sums, axis=0) / np.sum(cnts, axis=0)
    a = (1 - level) / 2
    return float(np.quantile(tot, a)), float(np.quantile(tot, 1 - a))


def stage1_stats(dvals, yrs, days, rng, boot):
    """Stage 1 と同じ集計 (pooled・年層化暦日 bootstrap・年方向・LOO)"""
    point = float(dvals.mean())
    lo, hi = day_boot_ci(dvals, yrs, days, rng, boot)
    ym = {int(y): float(dvals[yrs == y].mean()) for y in EVAL_YEARS}
    loo = [day_boot_ci(dvals, yrs, days, rng, boot, drop_year=y)[1] for y in EVAL_YEARS]
    return {"point": point, "ci": [lo, hi], "year_means": ym,
            "years_improved": int(sum(v < 0 for v in ym.values())), "loo_ci_uppers": loo}


def power_rep(eps, floor, rep_seed, W=None):
    from .gate_grade import grade_m1
    W = W or load_world()
    rng = np.random.default_rng(rep_seed)
    lp = truth_logp(eps, direction(RHO_PRIMARY, W), W)
    win = draw_winners(lp, W, rng)
    dv, yv, dayv = [], [], []
    for Y in EVAL_YEARS:
        fc, a0, _ = fit_year(Y, win, W, with_temp=True)
        r0, r1 = W["yr_rng"][Y]
        mk, h0, h1 = sub_mk(W["off"], r0, r1)
        lm, lq = W["logm"][h0:h1], W["logq"][h0:h1]
        wy = win[r0:r1] - h0
        lt = mk.norm_log(a0 * lm)[wy]
        lc = cross_logq(fc, lm, lq, mk)[wy]
        dv.append(-lc + lt)
        yv.append(W["year"][r0:r1])
        dayv.append(W["day"][r0:r1])
    dv, yv, dayv = np.concatenate(dv), np.concatenate(yv), np.concatenate(dayv)
    s = stage1_stats(dv, yv, dayv, rng, BOOT_POWER)
    # 推定器は決定論的 (seed を持つ学習要素なし): 5 seed は同一推定 → 方向は pooled 点推定と同じ
    seeds_improved = 5 if s["point"] < 0 else 0
    g = grade_m1(point=s["point"], ci_lower=s["ci"][0], ci_upper=s["ci"][1],
                 years_improved=s["years_improved"], n_years=5, seeds_improved=seeds_improved, n_seeds=5,
                 loo_ci_uppers=s["loo_ci_uppers"], p1_exceeded=True, p2_exceeded=True, floor=floor)
    return {"grade": g["grade"], "point": s["point"], "ci": s["ci"],
            "ci_upper_lt_0": s["ci"][1] < 0, "ci_upper_lt_minus_floor": s["ci"][1] < -floor}


def task_power(args):
    label, k, eps, floor = args
    return (label, k, power_rep(eps, floor, [SEED, 7, hash_label(label), k]))


def hash_label(label):
    return {"floor": 1, "0.5x": 2, "2.0x": 3, "zero": 4}[label]


# ---------------------------------------------------------------- phases
def run_floor():
    t0 = time.time()
    load_world()
    targets = [(rho, d) for rho in (RHO_PRIMARY, RHO_SENS) for d in DELTA_GRID]
    with ProcessPoolExecutor(WORKERS) as ex:
        cal = list(ex.map(task_calibrate, targets))
        print(f"[calibrated] {len(cal)} ({time.time()-t0:.0f}s)", flush=True)
        cal_map = {(c["rho"], c["target"]): c for c in cal}
        tasks = []
        for ri, rho in enumerate((RHO_PRIMARY, RHO_SENS)):
            for di, d in enumerate(DELTA_GRID):
                for rep in range(REPS_GROWTH):
                    tasks.append((ri, di, rep, rho, cal_map[(rho, d)]["eps"]))
        res = list(ex.map(task_growth, tasks, chunksize=2))
    out = {}
    for ri, rho in enumerate((RHO_PRIMARY, RHO_SENS)):
        curve, detail = {}, {}
        for di, d in enumerate(DELTA_GRID):
            reps = [r[3] for r in res if r[0] == ri and r[1] == di]
            gs = np.array([x["growth_per_race"] for x in reps])
            curve[d] = float(gs.mean())
            betas = {str(Y): float(np.mean([x["fits"][str(Y)]["beta"] for x in reps])) for Y in EVAL_YEARS}
            c = cal_map[(rho, d)]
            detail[str(d)] = {"eps": c["eps"], "delta_true": c["delta_true"], "a0_star": c["a0_star"],
                              "mean_growth_per_race": float(gs.mean()),
                              "mc_se": float(gs.std(ddof=1) / np.sqrt(len(gs))),
                              "min_rep": float(gs.min()), "max_rep": float(gs.max()),
                              "share_entered": float(np.mean([x["share_entered"] for x in reps])),
                              "mean_stake_when_entered": float(np.mean([x["mean_stake_when_entered"]
                                                                        for x in reps])),
                              "oracle_growth_per_race": c["oracle_growth_per_race"],
                              "mean_beta_by_eval_year": betas, "reps": len(gs)}
        out[str(rho)] = {"curve": detail, "floor": floor_from_curve(curve)}
        print(f"[rho={rho}] floor={out[str(rho)]['floor']} " +
              json.dumps({k: round(v, 7) for k, v in curve.items()}), flush=True)
    meta = json.loads((OUT / "floor_v05_cache_meta.json").read_text(encoding="utf-8"))
    res_json = {
        "role": "v0.5 実務床。凍結済み q_cross を n_fit 件の合成 outcome から実際に fit し、実 terminal 馬連オッズで"
                "決済したときの期待 Kelly 成長から床を求める。宣言的な独立ノイズなし。結果ラベル不使用",
        "seed": SEED, "seed_subset": SEED_SUBSET, "seed_v": SEED_V, "seed_fixed_before_run": True,
        "delta_grid": DELTA_GRID, "reps_growth": REPS_GROWTH, "interpolation": INTERPOLATION,
        "growth_threshold_per_race": GROWTH_THRESHOLD,
        "inputs": {k: meta[k] for k in ("gamma", "lambda", "n_fit_by_eval_year", "n_eval_by_year",
                                        "retention_le2018", "formal_le2018_exact")},
        "primary_rho_1": out[str(RHO_PRIMARY)], "sensitivity_rho_0_5": out[str(RHO_SENS)],
        "practical_floor_nats": out[str(RHO_PRIMARY)]["floor"],
        "sensitivity_floor_rho_0_5_nats": out[str(RHO_SENS)]["floor"],
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (OUT / "floor_v05.json").write_text(json.dumps(res_json, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[saved] out/floor_v05.json primary={res_json['practical_floor_nats']} "
          f"rho0.5={res_json['sensitivity_floor_rho_0_5_nats']} ({res_json['elapsed_sec']}s)")


def run_power():
    from .gate_grade import wilson
    t0 = time.time()
    fl = json.loads((OUT / "floor_v05.json").read_text(encoding="utf-8"))
    floor = fl["practical_floor_nats"]
    assert floor is not None, "主 floor が数値でないので検出力を測れない"
    load_world()
    specs = [("floor", floor, POWER_REPS_FLOOR), ("0.5x", 0.5 * floor, POWER_REPS_OTHER),
             ("2.0x", 2.0 * floor, POWER_REPS_OTHER), ("zero", 0.0, POWER_REPS_OTHER)]
    with ProcessPoolExecutor(WORKERS) as ex:
        cal = list(ex.map(task_calibrate, [(RHO_PRIMARY, d) for _, d, _ in specs]))
        tasks = [(lab, k, c["eps"], floor) for (lab, d, n), c in zip(specs, cal) for k in range(n)]
        res = list(ex.map(task_power, tasks, chunksize=2))
    out = {}
    for (lab, d, n), c in zip(specs, cal):
        rr = [r[2] for r in res if r[0] == lab]
        sig = sum(x["grade"] in ("PASS-SIGNAL", "PASS-PRACTICAL") for x in rr)
        pra = sum(x["grade"] == "PASS-PRACTICAL" for x in rr)
        ci0 = sum(x["ci_upper_lt_0"] for x in rr)
        pts = np.array([x["point"] for x in rr])
        out[lab] = {"delta_true": c["delta_true"], "eps": c["eps"], "reps": n,
                    "signal_power": sig / n, "signal_wilson95": list(wilson(sig, n)),
                    "practical_power": pra / n, "practical_wilson95": list(wilson(pra, n)),
                    "ci_upper_lt_0_rate": ci0 / n,
                    "median_point": float(np.median(pts)),
                    "recovery_median_point_over_minus_delta": (float(np.median(pts) / -c["delta_true"])
                                                               if c["delta_true"] > 0 else None)}
        print(f"[power {lab}] " + json.dumps(out[lab]), flush=True)
    res_json = {
        "role": "v0.5 検出力。主 floor の真の効果で Stage 1 と同じ rolling fit・年層化暦日 bootstrap・年方向・LOO・"
                "grade_m1 を合成 outcome で回す。placebo 条件は通過と仮定 (計算量のため、宣言)。結果ラベル不使用",
        "seed": SEED, "boot": BOOT_POWER, "floor": floor, "results": out,
        "required_signal_power": 0.8,
        "signal_power_gate_pass": bool(out["floor"]["signal_power"] >= 0.8),
        "practical_power_note": "CI95上限 < −floor は床ちょうどの効果に対して高確率で通る検出条件ではなく、実務床を95%"
                                "信頼水準で上回ったと主張するための厳格な等級。floor ちょうどでの PASS-PRACTICAL power は低い",
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (OUT / "power_v05.json").write_text(json.dumps(res_json, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[saved] out/power_v05.json ({res_json['elapsed_sec']}s)")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    phase = sys.argv[1] if len(sys.argv) > 1 else ""
    {"cache": build_cache, "floor": run_floor, "power": run_power}[phase]()
