# -*- coding: utf-8 -*-
"""L-4 main test (prereg §4-§7) — POWERED layers only. Reads winner_pos (the only outcome) once.

Refuses to run unless out/power.json exists, was produced under the frozen prereg (sha256 match) and lists
at least one POWERED layer with the null family-wise check passed.
python -m analysis.calendar_bias.stage2
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import common as C

WORKERS = int(os.environ.get("L4_WORKERS", 14))
_G: dict = {}


def read_winners(A):
    with zipfile.ZipFile(C.NPZ) as zf, zf.open("winner_pos.npy") as f:
        wpos = np.lib.format.read_array(f, allow_pickle=False)
    wp = wpos[A["npz_race_index"]].astype(np.int64)
    off = A["offsets"]
    assert (wp >= 0).all() and (wp < np.diff(off)).all()
    return off[:-1] + wp


def null_coefs(D: C.Design, win):
    th = {}
    for Y in C.EVAL_YEARS:
        races, rows, sg = D.N[Y]
        th[Y] = C.clogit(D.X[rows], sg, D.win_rows(win, races, sg), [1.0, 0.0])
    return th


def layer_delta(D: C.Design, mask, win, thN):
    """Pooled per-race Δ on eval races of an arbitrary layer mask (same procedure as Design.deltas)."""
    out = []
    for Y in C.EVAL_YEARS:
        fr = np.flatnonzero(mask & (D.year >= C.FIT_START) & (D.year <= Y - 1))
        er = np.flatnonzero(mask & (D.year == Y))
        frows, fsg = C.sub_seg(D.seg, fr)
        erows, esg = C.sub_seg(D.seg, er)
        thA = C.clogit(D.X[frows], fsg, D.win_rows(win, fr, fsg), thN[Y])
        we = D.win_rows(win, er, esg)
        Xe = D.X[erows]
        out.append(-esg.norm_log(Xe @ thA)[we] + esg.norm_log(Xe @ thN[Y])[we])
    return np.concatenate(out)


def placebo_mask(A, h, rng):
    year, day, dv = A["year"], A["day"], A["dv"]
    real = A[f"I_{h}"].astype(bool)
    m = np.zeros(len(year), bool)
    if h == "L4a_R12":
        order = rng.permutation(len(dv))
        _, first = np.unique(dv[order], return_index=True)
        m[order[first]] = True
        return m
    unit = day if h == "L4b_WEEKDAY" else dv
    for y in range(C.YEAR_MIN, C.YEAR_MAX + 1):
        sel = year == y
        units = np.unique(unit[sel])
        k = len(np.unique(unit[sel & real]))
        pick = rng.choice(units, size=k, replace=False)
        m |= sel & np.isin(unit, pick)
    return m


def init_worker(win):
    A = C.load_inputs()
    D = C.Design(A, {})
    _G.update({"A": A, "D": D, "win": win, "thN": null_coefs(D, win)})


def task_placebo(args):
    h, i = args
    rng = np.random.default_rng([C.SEED, 7, C.LAYERS.index(h) + 1, i])
    m = placebo_mask(_G["A"], h, rng)
    d = layer_delta(_G["D"], m, _G["win"], _G["thN"])
    return h, i, float(d.mean()), int((m & np.isin(_G["A"]["year"], C.EVAL_YEARS)).sum())


def main():
    t0 = time.time()
    pw = json.loads((C.OUT / "power.json").read_text(encoding="utf-8"))
    sha = C.sha256(C.PREREG)
    assert pw["prereg_sha256"] == sha, "prereg changed after power audit"
    assert not pw["null_familywise"]["stop"], "null family-wise check failed: stop"
    powered = pw["powered_layers"]
    if not powered:
        print("no POWERED layer: outcomes not opened")
        return
    A = C.load_inputs()
    masks = {h: A[f"I_{h}"].astype(bool) for h in powered}
    D = C.Design(A, masks)
    win = read_winners(A)
    ev_all = np.flatnonzero(np.isin(A["year"], C.EVAL_YEARS))
    boot = C.DayBoot(A["day"][ev_all])
    d, coefs = D.deltas(win, keep_coefs=True)
    st = {h: D.stats(h, d[h], boot) for h in powered}
    ph = C.holm_adjusted({h: st[h]["p"] for h in powered})
    print(f"[real] done ({time.time()-t0:.0f}s)", flush=True)

    with ProcessPoolExecutor(WORKERS, initializer=init_worker, initargs=(win,)) as ex:
        pl = list(ex.map(task_placebo, [(h, i) for h in powered for i in range(C.PLACEBO_DRAWS)], chunksize=2))
    res = {}
    for h in powered:
        vals = np.array([v for hh, _, v, _ in pl if hh == h])
        q = {"q025": float(np.quantile(vals, 0.025)), "median": float(np.median(vals)),
             "q975": float(np.quantile(vals, 0.975)), "draws": len(vals),
             "eval_races_median": float(np.median([n for hh, _, _, n in pl if hh == h])),
             "real_rank_share_below": float((vals <= st[h]["point"]).mean())}
        g, cond = C.grade(st[h], ph[h], placebo_q025=q["q025"])
        dslope = {Y: c["layer_a_c"][0] - c["null_a_c"][0] for Y, c in coefs[h].items()}
        res[h] = {"grade": g, "conditions": cond, "eval_races": int(len(d[h])),
                  "eval_calendar_days": int(len(np.unique(D.L[h]["ev_days"]))),
                  "pooled_delta": st[h]["point"], "ci95": st[h]["ci95"], "p_one_sided": st[h]["p"],
                  "p_holm": ph[h], "year_points": st[h]["year_points"], "loo_points": st[h]["loo_points"],
                  "placebo": q, "coefs_by_eval_year": coefs[h],
                  "descriptive_layer_minus_null_logpi_slope": dslope}
    survive = [h for h in powered if res[h]["grade"] == "ABOVE_SCREEN"]
    out = {"role": "L-4 calendar layers — main test (POWERED layers only)", "prereg_sha256": sha,
           "power_json_sha256": C.sha256(C.OUT / "power.json"), "powered_layers": powered,
           "unpowered_layers_not_opened": [h for h in C.LAYERS if h not in powered],
           "holm_family": powered, "results": res,
           "verdict": "SURVIVE" if survive else "CLOSED",
           "surviving_layers": survive, "outcome_member_read": "official_races_all.npz:winner_pos (once)",
           "years_used": "2016-2023 (2024/25 sealed)", "elapsed_sec": round(time.time() - t0, 1)}
    (C.OUT / "stage2.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for h in powered:
        r = res[h]
        print(h, r["grade"], f"Δ={r['pooled_delta']:+.6f}", "CI", [round(x, 6) for x in r["ci95"]],
              f"p_holm={r['p_holm']:.3f}", "placebo q025", round(r["placebo"]["q025"], 6))
    print(out["verdict"], f"{out['elapsed_sec']}s")


if __name__ == "__main__":
    main()
