# -*- coding: utf-8 -*-
"""L-4 power audit (prereg §8). Label-free: winners are synthetic only; winner_pos is never read.

truth      log p = log pi_pre + b * I_h * w,  w in {logpi, z}  (only layer-h races tilted)
Delta_true mean over 2017-2023 layer-h races of KL(p || N*), N* = expected-LL-best (a, c) on all 2017-2023 races
per rep    draw winners for every race -> frozen procedure (rolling fit, day bootstrap, Holm over 4 layers,
           >=5/7 years, all LOO) with placebo declared passed
POWERED    P(SIGNAL_SUBSCREEN or ABOVE_SCREEN) at 0.005 >= 0.80 in BOTH directions

python -m analysis.calendar_bias.power
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import common as C

NULL_REPS = int(os.environ.get("L4_NULL_REPS", 1000))
FLOOR_REPS = int(os.environ.get("L4_FLOOR_REPS", 400))
GRID_REPS = int(os.environ.get("L4_GRID_REPS", 100))
GRID = [0.0025, 0.0075, 0.01, 0.015, 0.02, 0.03]
WORKERS = int(os.environ.get("L4_WORKERS", 14))
_G: dict = {}


def build():
    A = C.load_inputs()
    masks = {h: A[f"I_{h}"].astype(bool) for h in C.LAYERS}
    D = C.Design(A, masks)
    ev_all = np.flatnonzero(np.isin(A["year"], C.EVAL_YEARS))
    boot = C.DayBoot(A["day"][ev_all])
    rows_ev, seg_ev = C.sub_seg(D.seg, ev_all)
    return {"A": A, "D": D, "boot": boot, "masks": masks, "ev_all": ev_all, "rows_ev": rows_ev, "seg_ev": seg_ev}


def init_worker():
    _G.clear()
    _G.update(build())


def direction(G, w):
    return G["D"].logpi if w == "logpi" else G["D"].z


def truth_logp(G, h, w, b):
    D = G["D"]
    x = D.logpi.copy()
    if h is not None and b != 0:
        x = x + b * D.seg.spread(G["masks"][h].astype(float)) * direction(G, w)
    return D.seg.norm_log(x)


def delta_true(G, h, w, b):
    D = G["D"]
    lp_all = truth_logp(G, h, w, b)
    rows, sg = G["rows_ev"], G["seg_ev"]
    lp = lp_all[rows]
    p = np.exp(lp)
    X = D.X[rows]
    th = C.clogit(X, sg, None, [1.0, 0.0], p_target=p)
    lq = sg.norm_log(X @ th)
    kl = sg.sum(p * (lp - lq))
    lay = G["masks"][h][G["ev_all"]]
    return float(kl[lay].mean())


def calibrate(G, h, w, target):
    lo, hi = 0.0, 0.25
    while delta_true(G, h, w, hi) < target:
        hi *= 2.0
        assert hi < 1e3
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if delta_true(G, h, w, mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-7 * hi:
            break
    b = 0.5 * (lo + hi)
    return b, delta_true(G, h, w, b)


def task_cal(args):
    h, w, t = args
    b, dt = calibrate(_G, h, w, t)
    return h, w, t, b, dt


def grade_all(G, win):
    D = G["D"]
    d = D.deltas(win)
    st = {h: D.stats(h, d[h], G["boot"]) for h in C.LAYERS}
    ph = C.holm_adjusted({h: st[h]["p"] for h in C.LAYERS})
    out = {}
    for h in C.LAYERS:
        g, _ = C.grade(st[h], ph[h], placebo_declared_passed=True)
        out[h] = {"grade": g, "point": st[h]["point"], "ci_upper": st[h]["ci95"][1]}
    return out


def task_rep(args):
    label, h, w, t, b, rep = args
    key = [C.SEED, {"null": 0, "floor": 1, "grid": 2}[label], C.LAYERS.index(h) + 1 if h else 0,
           C.DIRECTIONS.index(w) + 1 if w else 0, int(round(t * 1e5)), rep]
    rng = np.random.default_rng(key)
    win = C.draw_winners(truth_logp(_G, h, w, b), _G["D"].seg, rng)
    return label, h, w, t, grade_all(_G, win)


def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    t0 = time.time()
    prereg_sha = C.sha256(C.PREREG)
    with ProcessPoolExecutor(WORKERS, initializer=init_worker) as ex:
        cal_tasks = [(h, w, C.FLOOR) for h in C.LAYERS for w in C.DIRECTIONS] + \
                    [(h, "logpi", t) for h in C.LAYERS for t in GRID]
        cal = list(ex.map(task_cal, cal_tasks))
        bmap = {(h, w, t): (b, dt) for h, w, t, b, dt in cal}
        print(f"[calibrated] {len(cal)} ({time.time()-t0:.0f}s)", flush=True)
        tasks = [("null", None, None, 0.0, 0.0, r) for r in range(NULL_REPS)]
        for h in C.LAYERS:
            for w in C.DIRECTIONS:
                tasks += [("floor", h, w, C.FLOOR, bmap[(h, w, C.FLOOR)][0], r) for r in range(FLOOR_REPS)]
            for t in GRID:
                tasks += [("grid", h, "logpi", t, bmap[(h, "logpi", t)][0], r) for r in range(GRID_REPS)]
        res = []
        for i, r in enumerate(ex.map(task_rep, tasks, chunksize=4)):
            res.append(r)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)
    det = ("SIGNAL_SUBSCREEN", "ABOVE_SCREEN")
    null = [r[4] for r in res if r[0] == "null"]
    fw = sum(any(g[h]["grade"] in det for h in C.LAYERS) for g in null)
    per_null = {h: sum(g[h]["grade"] in det for g in null) for h in C.LAYERS}
    wil = C.wilson(fw, len(null))
    out_h = {}
    for h in C.LAYERS:
        floor = {}
        for w in C.DIRECTIONS:
            rr = [r[4][h] for r in res if r[0] == "floor" and r[1] == h and r[2] == w]
            k = sum(g["grade"] in det for g in rr)
            floor[w] = {"effect_b": bmap[(h, w, C.FLOOR)][0], "delta_true": bmap[(h, w, C.FLOOR)][1],
                        "reps": len(rr), "detect_power": k / len(rr), "detect_wilson95": C.wilson(k, len(rr)),
                        "above_screen_power": sum(g["grade"] == "ABOVE_SCREEN" for g in rr) / len(rr),
                        "median_point": float(np.median([g["point"] for g in rr]))}
        curve = {str(C.FLOOR): floor["logpi"]["detect_power"]}
        for t in GRID:
            rr = [r[4][h] for r in res if r[0] == "grid" and r[1] == h and r[3] == t]
            curve[str(t)] = sum(g["grade"] in det for g in rr) / len(rr)
        xs = sorted(float(k) for k in curve)
        ys = [curve[str(x)] for x in xs]
        mde = None
        for i, (x, y) in enumerate(zip(xs, ys)):
            if y >= C.MIN_POWER:
                mde = x if i == 0 else xs[i - 1] + (C.MIN_POWER - ys[i - 1]) * (x - xs[i - 1]) / (y - ys[i - 1])
                break
        pmin = min(floor[w]["detect_power"] for w in C.DIRECTIONS)
        out_h[h] = {"floor": floor, "curve_logpi_detect_power": dict(zip([str(x) for x in xs], ys)),
                    "mde_detect_0_80_logpi": mde, "power_at_floor_min_over_directions": pmin,
                    "status": "POWERED" if pmin >= C.MIN_POWER else "UNPOWERED"}
    stop = wil[1] > C.NULL_WILSON_UPPER_MAX
    rep = json.loads((C.OUT / "inputs_report.json").read_text(encoding="utf-8"))
    out = {"role": "L-4 calendar layers — synthetic-outcome power audit (label-free)",
           "prereg": str(C.PREREG.relative_to(C.BASE)), "prereg_sha256": prereg_sha,
           "seed": C.SEED, "bootstrap_B": C.BOOT_B, "floor": C.FLOOR, "eval_years": C.EVAL_YEARS,
           "reps": {"null": NULL_REPS, "floor_per_layer_direction": FLOOR_REPS, "grid": GRID_REPS},
           "layer_counts": rep["layer_counts"], "layers": out_h,
           "null_familywise": {"reps": len(null), "any_detect": fw, "rate": fw / len(null), "wilson95": wil,
                               "per_layer": per_null, "stop": bool(stop)},
           "powered_layers": [h for h in C.LAYERS if out_h[h]["status"] == "POWERED"] if not stop else [],
           "elapsed_sec": round(time.time() - t0, 1), "outcomes_opened": False}
    (C.OUT / "power.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({h: {k: out_h[h][k] for k in ("power_at_floor_min_over_directions", "mde_detect_0_80_logpi",
                                                    "status")} for h in C.LAYERS}, ensure_ascii=False, indent=1))
    print(json.dumps(out["null_familywise"]), out["powered_layers"], f"{out['elapsed_sec']}s")


if __name__ == "__main__":
    main()
