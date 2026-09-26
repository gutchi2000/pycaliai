# -*- coding: utf-8 -*-
"""
compute_dry_run.py — EXP19 Stage 0 S0-E: 計算量 (結果ラベル不使用、合成 winner だけ)
====================================================================================
1 年分の特徴構築、fit、10,000 bootstrap、placebo 1 draw (×200 の見積り) の時間と最大 RSS を測る。
他の重い計算と同時に走らせない (時間が汚れるため)。出力: out/compute_dry_run.json
"""
from __future__ import annotations

import ctypes
import json
import time
from ctypes import wintypes

import numpy as np

from . import features as F
from . import models as M
from . import placebos as PL
from . import power_floor as PF
from .loaders import OUT, load_torch_struct


def peak_rss_mb() -> float:
    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    c = PMC()
    c.cb = ctypes.sizeof(PMC)
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    fn = ctypes.windll.psapi.GetProcessMemoryInfo
    fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    fn.restype = wintypes.BOOL
    assert fn(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.PeakWorkingSetSize / 1e6


def main():
    res = {}
    t = time.time()
    tor = load_torch_struct()
    res["torch_load_sec"] = round(time.time() - t, 1)
    t = time.time()
    F.build_w(tor)
    res["W_build_all_2013_2023_sec"] = round(time.time() - t, 1)
    res["W_build_one_year_estimate_sec"] = round(res["W_build_all_2013_2023_sec"] / 11, 1)
    PF.world()
    rng = np.random.default_rng(1)
    lp = PF.truth("A1", 0.3)
    win = PF.draw(lp, rng)
    fits, ev = PF.gate_races("A1")
    _, null, alt, _ = PF.cols("A1")
    t = time.time()
    PF.fit_arm(alt, fits[2020], win)
    res["fit_W_arm_2016_2019_window_sec"] = round(time.time() - t, 2)
    t = time.time()
    for Y, fr in fits.items():
        PF.fit_arm(null, fr, win)
        PF.fit_arm(alt, fr, win)
    res["rolling_fits_null_and_alt_5_years_sec"] = round(time.time() - t, 1)
    vals = rng.normal(0, 0.3, len(ev))
    t = time.time()
    PF.day_boot(vals, PF._G["year"][ev], PF._G["day"][ev], [2019], rng)
    res["bootstrap_10000_one_year_sec"] = round(time.time() - t, 2)
    t = time.time()
    PF.day_boot(vals, PF._G["year"][ev], PF._G["day"][ev], M.GATES["A1"]["years"], rng)
    res["bootstrap_10000_five_years_sec"] = round(time.time() - t, 2)
    t = time.time()
    Wp = PL.p1_permute(PF._G["W"], PF._G["off"], rng)
    alt_p = np.column_stack([null, Wp])
    for Y, fr in fits.items():
        PF.fit_arm(alt_p, fr, win)
    one = time.time() - t
    res["placebo_P1_one_draw_sec"] = round(one, 1)
    res["placebo_200_draws_estimate_sec_per_placebo"] = round(one * 200, 0)
    res["peak_rss_mb"] = round(peak_rss_mb(), 1)
    res["note"] = "A1 (最大の fit 窓) で計測。A2/B2 は WP 利用可能 race だけなので小さい。単一 process"
    (OUT / "compute_dry_run.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
