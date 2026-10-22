# -*- coding: utf-8 -*-
"""L-4 calendar bias — shared machinery (frozen settings = docs/research/L4_CALENDAR_BIAS_PREREG_20261022.md).

Conditional logit / bootstrap / Holm are minimal copies of EXP25 common25.py (same rules).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
OUT = HERE / "out"
PREREG = BASE / "docs" / "research" / "L4_CALENDAR_BIAS_PREREG_20261022.md"
NPZ = BASE / "data" / "_research" / "mcond" / "exp16a" / "official_races_all.npz"
MARKET = BASE / "data" / "_research" / "mcond" / "market.parquet"
OOF = BASE / "data" / "_research" / "mcond" / "oof_scores_c1master.parquet"
MASTER = BASE / "data" / "master_v2_20130105-20251228.csv"
INPUTS = OUT / "inputs.npz"

LAYERS = ["L4a_R12", "L4b_WEEKDAY", "L4c_KAI_LAST", "L4d_G1DAY_OTHER"]
YEAR_MIN, YEAR_MAX = 2016, 2023
EVAL_YEARS = list(range(2017, 2024))
FIT_START = 2016
FLOOR = 0.005
ALPHA = 0.025
MIN_YEARS_NEG = 5
BOOT_B = 10_000
SEED = 20261022
MIN_POWER = 0.80
NULL_WILSON_UPPER_MAX = 0.05
PLACEBO_DRAWS = 200
DIRECTIONS = ["logpi", "z"]


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Seg:
    def __init__(self, off):
        self.off = np.asarray(off, np.int64)
        self.start, self.cnt = self.off[:-1], np.diff(self.off)
        self.R = len(self.cnt)
        self.race_of_row = np.repeat(np.arange(self.R), self.cnt)

    def sum(self, x):
        return np.add.reduceat(x, self.start, axis=0)

    def spread(self, v):
        return np.repeat(v, self.cnt, axis=0)

    def lse(self, x):
        mx = np.maximum.reduceat(x, self.start)
        return np.log(self.sum(np.exp(x - self.spread(mx)))) + mx

    def norm_log(self, x):
        return x - self.spread(self.lse(x))


def sub_seg(seg: Seg, races: np.ndarray):
    if len(races) == 0:
        return np.zeros(0, np.int64), Seg([0])
    starts, cnts = seg.off[races], seg.cnt[races]
    rows = np.repeat(starts - np.concatenate([[0], np.cumsum(cnts)[:-1]]), cnts) + np.arange(cnts.sum())
    return rows, Seg(np.concatenate([[0], np.cumsum(cnts)]))


def clogit(X, seg: Seg, win_rows, theta0, iters=50, p_target=None):
    """Conditional logit by Newton. If p_target is given, maximises expected LL under p_target instead."""
    th = np.array(theta0, float)
    d = X.shape[1]
    for _ in range(iters):
        p = np.exp(seg.norm_log(X @ th))
        Ex = seg.sum(p[:, None] * X)
        if p_target is None:
            g = X[win_rows].sum(0) - Ex.sum(0)
        else:
            g = (seg.sum(p_target[:, None] * X) - Ex).sum(0)
        Exx = seg.sum(p[:, None, None] * X[:, :, None] * X[:, None, :])
        H = -(Exx - Ex[:, :, None] * Ex[:, None, :]).sum(0) - 1e-12 * np.eye(d)
        step = np.linalg.solve(H, g)
        th = th - step
        if np.max(np.abs(step)) < 1e-10:
            break
    return th


def draw_winners(logp, seg: Seg, rng):
    key = logp - np.log(-np.log(rng.random(len(logp))))
    mx = np.maximum.reduceat(key, seg.start)
    hit = np.flatnonzero(key == seg.spread(mx))
    return hit[np.unique(seg.race_of_row[hit], return_index=True)[1]]


class DayBoot:
    """Year-stratified calendar-day cluster bootstrap weights (fixed seed, reused for every statistic)."""

    def __init__(self, days, seed=SEED, b=BOOT_B):
        self.udays = np.unique(days)
        years = self.udays // 10000
        rng = np.random.default_rng(seed)
        self.W = np.zeros((b, len(self.udays)), np.float32)
        for y in np.unique(years):
            idx = np.flatnonzero(years == y)
            pick = rng.integers(0, len(idx), size=(b, len(idx)))
            np.add.at(self.W, (np.repeat(np.arange(b), len(idx)), idx[pick.ravel()]), 1.0)

    def means(self, days, vals):
        di = np.searchsorted(self.udays, days)
        assert np.array_equal(self.udays[np.minimum(di, len(self.udays) - 1)], days), "day outside bootstrap frame"
        sn = np.bincount(di, weights=vals, minlength=len(self.udays))
        sd = np.bincount(di, minlength=len(self.udays)).astype(float)
        return (self.W @ sn) / np.maximum(self.W @ sd, 1e-300)


def ci95(t):
    return [float(np.quantile(t, 0.025)), float(np.quantile(t, 0.975))]


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(max(0.0, c - h)), float(min(1.0, c + h))]


def holm_adjusted(p: dict) -> dict:
    order = sorted(p, key=lambda k: p[k])
    m, run, out = len(order), 0.0, {}
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * p[k]))
        out[k] = run
    return out


def load_inputs():
    z = np.load(INPUTS, allow_pickle=False)
    A = {k: z[k] for k in z.files}
    assert A["year"].max() <= YEAR_MAX and A["year"].min() >= YEAR_MIN, "2024/25 sealed"
    return A


# ---------------------------------------------------------------- the frozen test procedure (§4-§6)
class Design:
    """Holds per-year fit/eval subsets for the null and every layer; label-free."""

    def __init__(self, A, layer_masks: dict):
        self.seg = Seg(A["offsets"])
        self.logpi, self.z = A["logpi"], A["z"]
        self.X = np.column_stack([self.logpi, self.z])
        self.year, self.day = A["year"], A["day"]
        self.layers = list(layer_masks)
        self.N = {}
        for Y in EVAL_YEARS:
            races = np.flatnonzero((self.year >= FIT_START) & (self.year <= Y - 1))
            rows, sg = sub_seg(self.seg, races)
            self.N[Y] = (races, rows, sg)
        self.L = {}
        for h, mask in layer_masks.items():
            H = {}
            for Y in EVAL_YEARS:
                fr = np.flatnonzero(mask & (self.year >= FIT_START) & (self.year <= Y - 1))
                er = np.flatnonzero(mask & (self.year == Y))
                H[(Y, "fit")] = (fr,) + sub_seg(self.seg, fr)
                H[(Y, "ev")] = (er,) + sub_seg(self.seg, er)
            ev = np.flatnonzero(mask & np.isin(self.year, EVAL_YEARS))
            H["ev_races"], H["ev_days"], H["ev_years"] = ev, self.day[ev], self.year[ev]
            self.L[h] = H

    def win_rows(self, win_global, races, sg):
        return sg.off[:-1] + (win_global[races] - self.seg.off[races])

    def deltas(self, win_global, keep_coefs=False):
        """Per-race LL(A_h) - LL(N) on eval layer races, for every layer (ordered as L[h]['ev_races'])."""
        out = {h: [] for h in self.layers}
        coefs = {h: {} for h in self.layers}
        for Y in EVAL_YEARS:
            races, rows, sg = self.N[Y]
            thN = clogit(self.X[rows], sg, self.win_rows(win_global, races, sg), [1.0, 0.0])
            for h in self.layers:
                fr, frows, fsg = self.L[h][(Y, "fit")]
                er, erows, esg = self.L[h][(Y, "ev")]
                thA = clogit(self.X[frows], fsg, self.win_rows(win_global, fr, fsg), thN)
                we = self.win_rows(win_global, er, esg)
                Xe = self.X[erows]
                lN = esg.norm_log(Xe @ thN)
                lA = esg.norm_log(Xe @ thA)
                out[h].append(-lA[we] + lN[we])
                if keep_coefs:
                    coefs[h][Y] = {"null_a_c": thN.tolist(), "layer_a_c": thA.tolist(),
                                   "fit_races_layer": int(len(fr)), "eval_races_layer": int(len(er))}
        d = {h: np.concatenate(v) for h, v in out.items()}
        return (d, coefs) if keep_coefs else d

    def stats(self, h, d, boot: DayBoot):
        H = self.L[h]
        bm = boot.means(H["ev_days"], d)
        yp = {int(Y): float(d[H["ev_years"] == Y].mean()) for Y in EVAL_YEARS}
        loo = {int(Y): float(d[H["ev_years"] != Y].mean()) for Y in EVAL_YEARS}
        return {"point": float(d.mean()), "ci95": ci95(bm),
                "p": float((1 + (bm >= 0).sum()) / (len(bm) + 1)), "year_points": yp, "loo_points": loo}


def grade(st: dict, p_holm: float, placebo_q025=None, placebo_declared_passed=False) -> tuple[str, dict]:
    cond = {"holm_p_lt_alpha": p_holm < ALPHA,
            "ci95_upper_lt_0": st["ci95"][1] < 0,
            "ci95_upper_lt_minus_floor": st["ci95"][1] < -FLOOR,
            "years_negative_ge_5_of_7": sum(v < 0 for v in st["year_points"].values()) >= MIN_YEARS_NEG,
            "loo_all_negative": all(v < 0 for v in st["loo_points"].values()),
            "placebo_exceeded": True if placebo_declared_passed else bool(st["point"] < placebo_q025)}
    base = all(cond[k] for k in ("holm_p_lt_alpha", "ci95_upper_lt_0", "years_negative_ge_5_of_7",
                                 "loo_all_negative", "placebo_exceeded"))
    g = "ABOVE_SCREEN" if base and cond["ci95_upper_lt_minus_floor"] else ("SIGNAL_SUBSCREEN" if base else "FAIL")
    return g, cond
