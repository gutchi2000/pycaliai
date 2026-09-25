# -*- coding: utf-8 -*-
"""
power_audit.py — EXP18 S0-E: 実務床 (PRACTICAL floor) の算出と検出力監査
=======================================================================
**結果ラベルを一切使わない**。2019-2023 の市場構造 (terminal UMAREN / TANSHO) と、≤2018 の dry-run fit 値
(γ・λ、out/anchor_le2018.json) だけを使い、合成 winner で検出力を測る。

固定した生成構造 (spec S0-E。seed は実行前に固定):
  真の分布      p(i,j) ∝ m_cal(i,j) · exp(ε · s_true(i,j))
                s_true = race 内で中心化・標準化した log q_T1 − log m_cal (UB2 が使う方向)
                ε は真の期待 logloss 改善 (= 平均 KL(p‖m_cal)) が Δ になるよう較正
  推定          q_est ∝ m_cal · exp(ε · s_true + noise)、noise ~ N(0, 2Δ) を組ごとに独立
                (SD(noise) = sqrt(2Δ) は label-free の**宣言慣習**であり、導出ではない)
  決済          馬連倍率 o = (1 − 0.225) / m_cal (控除率 22.5%)
  参加          max(q_est / m_cal) > 1/0.775 のレースだけ、現金込み Kelly (排反事象の閉形式) で賭ける
  成長          真の p の下での期待対数成長 (賭けないレースは 0) の race 平均
  PRACTICAL floor = 期待対数成長 >= 1e-4 / race となる最小の Δ (格子の線形補間)
検出力 (宣言効果 = PRACTICAL floor):
  合成 winner を p から引き、年 Y (2019-2023) ごとに 2016..Y−1 の合成データで
  温度 null q_temp ∝ m_cal^a0 と cross 代替 q_cross ∝ m_cal^a · q_T1^β を fit、
  Δ_r = LL(q_cross) − LL(q_temp) を年層化・暦日 cluster bootstrap で集計し、SIGNAL = CI95 上限 < 0
  進行条件: 宣言効果での SIGNAL 検出力 >= 80% (Wilson 95% CI も記録)
出力: out/power_audit.json
実行: python -m analysis.mcond.exp18_cross_pool_market_tomography_dev.power_audit
"""
from __future__ import annotations

import json
import time

import numpy as np

from . import tomography as T
from .loaders import OUT, load_structure
from .market_build import base_race_status, pool_matrices, race_arrays, snapshot_index, umaren_grid_complete

SEED = 20260925                    # 実行前に固定
TAKEOUT = 0.225
ENTRY = 1.0 / (1.0 - TAKEOUT)
GROWTH_THRESHOLD = 1e-4
DELTA_GRID = [0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075,
              0.1, 0.15, 0.2]
NOISE_DRAWS = 3
REPS_POWER = 400
REPS_CURVE = 200
BOOT = 1000
FIT_YEAR_MIN = 2016
EVAL_YEARS = [2019, 2020, 2021, 2022, 2023]
REQUIRED_POWER = 0.80


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def load_markets(gamma, lam):
    st = load_structure(range(FIT_YEAR_MIN, 2024))
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
        m = T.devig_power(at["umaren"], gamma)
        pi = (1 / at["win"]) / (1 / at["win"]).sum()
        q = T.stern_top2(pi, lam)
        races.append((int(rid[:4]), rid[:8], rid, m, q))
    races.sort(key=lambda x: (x[0], x[2]))
    year = np.array([x[0] for x in races])
    day = np.array([x[1] for x in races])
    cnt = np.array([len(x[3]) for x in races])
    off = np.concatenate([[0], np.cumsum(cnt)])
    m = np.concatenate([x[3] for x in races])
    q = np.concatenate([x[4] for x in races])
    return year, day, off, m, q


class Seg:
    def __init__(self, off):
        self.off, self.start, self.cnt = off, off[:-1], np.diff(off)
        self.R = len(self.cnt)

    def sum(self, x):
        return np.add.reduceat(x, self.start)

    def spread(self, v):
        return np.repeat(v, self.cnt)

    def maxv(self, x):
        return np.maximum.reduceat(x, self.start)


def zscore(x, sg):
    mu = sg.sum(x) / sg.cnt
    c = x - sg.spread(mu)
    sd = np.sqrt(sg.sum(c * c) / sg.cnt)
    return c / np.maximum(sg.spread(sd), 1e-12)


def tilt(logm, s, eps, sg):
    e = logm + eps * s
    e = e - sg.spread(sg.maxv(e))
    u = np.exp(e)
    return u / sg.spread(sg.sum(u))


def mean_kl(logm, s, eps, sg):
    p = tilt(logm, s, eps, sg)
    return float((sg.sum(p * (np.log(p) - logm))).mean())


def calibrate_eps(logm, s, sg, target):
    lo, hi = 0.0, 0.05
    while mean_kl(logm, s, hi, sg) < target:
        hi *= 2
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if mean_kl(logm, s, mid, sg) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def growth_curve(logm, s, sg, rng):
    out = {}
    m = np.exp(logm)
    odds = (1.0 - TAKEOUT) / m
    for d in DELTA_GRID:
        eps = calibrate_eps(logm, s, sg, d)
        p = tilt(logm, s, eps, sg)
        sd = np.sqrt(2 * d)
        gs, entered, stake = [], 0, []
        for _ in range(NOISE_DRAWS):
            qe = tilt(logm, s * eps + rng.normal(0, sd, len(logm)), 1.0, sg)
            ratio_max = sg.maxv(qe / m)
            g_draw = 0.0
            for r in np.flatnonzero(ratio_max > ENTRY):
                a, b = sg.off[r], sg.off[r + 1]
                g, st, _ = T.kelly_cash_growth(qe[a:b], odds[a:b], p[a:b])
                g_draw += g
                stake.append(st)
            entered += int((ratio_max > ENTRY).sum())
            gs.append(g_draw / sg.R)
        out[d] = {"eps": eps, "mean_growth_per_race": float(np.mean(gs)),
                  "growth_sd_over_noise_draws": float(np.std(gs, ddof=1)),
                  "share_races_entered": entered / (sg.R * NOISE_DRAWS),
                  "mean_stake_fraction_when_entered": float(np.mean(stake)) if stake else 0.0}
    return out


def floor_from_curve(curve):
    xs = sorted(curve)
    ys = [curve[x]["mean_growth_per_race"] for x in xs]
    for i, (x, y) in enumerate(zip(xs, ys)):
        if y >= GROWTH_THRESHOLD:
            if i == 0:
                return x
            x0, y0 = xs[i - 1], ys[i - 1]
            return float(x0 + (GROWTH_THRESHOLD - y0) * (x - x0) / (y - y0))
    return None


def day_boot(vals, yrs, days, rng, drop_year=None, boot=BOOT):
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
    return float(np.quantile(tot, 0.025)), float(np.quantile(tot, 0.975))


def power_at(delta, logm, logq, s, year, day, off, rng, reps):
    sg = Seg(off)
    eps = calibrate_eps(logm, s, sg, delta)
    logp = np.log(tilt(logm, s, eps, sg))
    ysl = {}
    for y in sorted(set(year.tolist())):
        r = np.flatnonzero(year == y)
        ysl[y] = (int(r[0]), int(r[-1] + 1))
    ev_r = np.concatenate([np.arange(*ysl[y]) for y in EVAL_YEARS])
    race_of_row = np.repeat(np.arange(sg.R), sg.cnt)
    signal = full = 0
    ests = []
    for _ in range(reps):
        g = -np.log(-np.log(rng.random(len(logm))))
        key = logp + g
        mx = np.maximum.reduceat(key, sg.start)
        hit = np.flatnonzero(key == sg.spread(mx))
        win = hit[np.unique(race_of_row[hit], return_index=True)[1]]
        dvals = np.full(sg.R, np.nan)
        for y in EVAL_YEARS:
            r0, r1 = ysl[FIT_YEAR_MIN][0], ysl[y - 1][1]
            h0, h1 = off[r0], off[r1]
            sub_off = off[r0:r1 + 1] - h0
            wf = win[r0:r1] - h0
            a0 = T.fit_temp(logm[h0:h1], sub_off, wf)
            fc = T.fit_cross(logm[h0:h1], logq[h0:h1], sub_off, wf)
            a, b = ysl[y]
            e0, e1 = off[a], off[b]
            so = off[a:b + 1] - e0
            sgy = Seg(so)
            wy = win[a:b] - e0
            lm, lq = logm[e0:e1], logq[e0:e1]
            et = a0 * lm
            lt = et[wy] - (np.log(sgy.sum(np.exp(et - sgy.spread(sgy.maxv(et))))) + sgy.maxv(et))
            if fc["collinear"]:
                lc = lt
            else:
                ec = fc["a"] * lm + fc["beta"] * lq
                lc = ec[wy] - (np.log(sgy.sum(np.exp(ec - sgy.spread(sgy.maxv(ec))))) + sgy.maxv(ec))
            dvals[a:b] = -lc + lt
        vals = dvals[ev_r]
        yrs = year[ev_r]
        days = day[ev_r]
        lo, hi = day_boot(vals, yrs, days, rng)
        sig = hi < 0
        signal += int(sig)
        ym = [float(np.mean(vals[yrs == y])) for y in EVAL_YEARS]
        loo = [day_boot(vals, yrs, days, rng, drop_year=y)[1] for y in EVAL_YEARS]
        full += int(sig and sum(v < 0 for v in ym) >= 4 and all(u < 0 for u in loo))
        ests.append([float(vals.mean()), lo, hi])
    ests = np.array(ests)
    return {"delta": delta, "eps": eps, "reps": reps, "signal_successes": signal,
            "signal_power": signal / reps, "signal_wilson95": wilson(signal, reps),
            "full_statistical_conditions_successes": full,
            "full_statistical_conditions_power": full / reps,
            "median_estimate": float(np.median(ests[:, 0])),
            "median_ci95": [float(np.median(ests[:, 1])), float(np.median(ests[:, 2]))],
            "attenuation": float(np.median(ests[:, 0]) / -delta)}


def main():
    t0 = time.time()
    anc = json.loads((OUT / "anchor_le2018.json").read_text(encoding="utf-8"))
    gamma = anc["dryrun_fits_terminal_le2018"]["gamma_powerlaw_devig"]
    lam = anc["dryrun_fits_terminal_le2018"]["lambda_T1"]
    year, day, off, m, q = load_markets(gamma, lam)
    sg = Seg(off)
    logm, logq = np.log(m), np.log(q)
    s = zscore(logq - logm, sg)
    ev = np.isin(year, EVAL_YEARS)
    ev_races = np.flatnonzero(ev)
    e0, e1 = off[ev_races[0]], off[ev_races[-1] + 1]
    sg_ev = Seg(off[ev_races[0]:ev_races[-1] + 2] - e0)
    rng = np.random.default_rng(SEED)
    print(f"[data] races {sg.R:,} (eval {len(ev_races):,}) pairs {len(m):,} γ={gamma:.4f} λ={lam:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    curve = growth_curve(logm[e0:e1], s[e0:e1], sg_ev, rng)
    floor = floor_from_curve(curve)
    print("[growth] " + json.dumps({k: round(v["mean_growth_per_race"], 7) for k, v in curve.items()}),
          flush=True)
    print(f"[floor] PRACTICAL floor = {floor}", flush=True)

    power = {}
    if floor is not None:
        power["at_floor"] = power_at(floor, logm, logq, s, year, day, off, rng, REPS_POWER)
        print("[power@floor] " + json.dumps(power["at_floor"]), flush=True)
        for f in (0.5, 2.0):
            power[f"at_{f}x_floor"] = power_at(floor * f, logm, logq, s, year, day, off, rng, REPS_CURVE)
            print(f"[power@{f}x] " + json.dumps(power[f"at_{f}x_floor"]), flush=True)
    ok = bool(floor is not None and power["at_floor"]["signal_power"] >= REQUIRED_POWER)
    res = {
        "role": "実務床の算出と検出力監査。結果ラベルを使わず、2019-2023 の市場構造と合成 winner だけを使う",
        "seed": SEED, "seed_fixed_before_run": True,
        "generation": {
            "true": "p ∝ m_cal · exp(ε s_true)、s_true = race 内標準化した log q_T1 − log m_cal、ε は平均 KL = Δ に較正",
            "estimate": "q_est ∝ m_cal · exp(ε s_true + noise)、noise ~ N(0, 2Δ) 組ごと独立",
            "noise_sd_note": "SD(noise)=sqrt(2Δ) は label-free の宣言慣習であり導出ではない",
            "odds": "o = 0.775 / m_cal (控除率 22.5%)",
            "entry": "max(q_est/m_cal) > 1/0.775", "staking": "現金込み Kelly (排反事象の閉形式)",
            "growth_threshold_per_race": GROWTH_THRESHOLD, "noise_draws": NOISE_DRAWS,
        },
        "inputs": {"gamma_devig_le2018": gamma, "lambda_T1_le2018": lam,
                   "races_total": int(sg.R), "races_eval_2019_2023": int(len(ev_races)),
                   "pairs_total": int(len(m)), "fit_years_start": FIT_YEAR_MIN,
                   "inference_unit": "暦日 (YYYYMMDD) cluster、年層化 bootstrap",
                   "bootstrap_reps": BOOT},
        "growth_curve": {str(k): v for k, v in curve.items()},
        "practical_floor_nats": floor,
        "power": power,
        "required_power": REQUIRED_POWER,
        "power_gate_pass": ok,
        "reference_full_investment_bound": "-log(0.775) = 0.2549 nats は全額比例投入の参考条件で hard gate にしない",
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (OUT / "power_audit.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float),
                                          encoding="utf-8")
    print(f"[saved] out/power_audit.json floor={floor} power_gate_pass={ok} ({res['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
