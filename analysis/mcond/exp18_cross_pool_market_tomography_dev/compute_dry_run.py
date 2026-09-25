# -*- coding: utf-8 -*-
"""
compute_dry_run.py — EXP18 S0-F: 計算量の dry run (結果ラベルを使わない)
=======================================================================
測るもの:
  * 1 年分 (2022) の市場構築: 構造 loader → snapshot → race 配列 → m_cal・q_T1 の時間
  * 最大 RSS (Windows: PeakWorkingSetSize)
  * 全期間 (fit 2016-2018 + development 2019-2023 = 8 年) の推定時間
  * 年層化・暦日 cluster bootstrap (3,000 reps) の時間
  * P1 (race 内 source identity 置換) / P2 (セル内 source snapshot 時間ずらし) 各 200 draw の推定時間
    (1 draw = source 置換 → q_T1 再計算 → 年別の温度 null / cross 代替の fit → Δ)
  * T2 MaxEnt: 往復 Gate の結果で実装可否が決まる。未実装なら計測しない
  * 近似法: 使わない (T1・T2 とも厳密計算。n<=8 oracle 誤差は invariant_tests.json)
合成 winner を使うので結果ラベルは読まない。
出力: COMPUTE_DRY_RUN.json
実行: python -m analysis.mcond.exp18_cross_pool_market_tomography_dev.compute_dry_run
"""
from __future__ import annotations

import ctypes
import json
import time
from ctypes import wintypes

import numpy as np

from . import tomography as T
from .loaders import HERE, OUT, load_structure
from .market_build import (base_race_status, pool_matrices, race_arrays, snapshot_index, tan_entropy,
                           umaren_grid_complete)

SEED = 20260925


def peak_rss_mb() -> float:
    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    c = PMC()
    c.cb = ctypes.sizeof(PMC)
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = wintypes.HANDLE          # 64bit の擬似 handle を切り詰めない
    fn = ctypes.windll.psapi.GetProcessMemoryInfo
    fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    fn.restype = wintypes.BOOL
    ok = fn(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    assert ok, "GetProcessMemoryInfo failed"
    return c.PeakWorkingSetSize / 1e6


def build_year(years, gamma, lam):
    st = load_structure(years)
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
        races.append({"rid": rid, "year": int(rid[:4]), "day": rid[:8], "n": at["n"],
                      "venue": inf["venue"], "ent": tan_entropy(at["win"]), "pi": pi,
                      "m": T.devig_power(at["umaren"], gamma), "q": T.stern_top2(pi, lam)})
    return races


def main():
    import sys
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    t00 = time.time()
    anc = json.loads((OUT / "anchor_le2018.json").read_text(encoding="utf-8"))
    gamma = anc["dryrun_fits_terminal_le2018"]["gamma_powerlaw_devig"]
    lam = anc["dryrun_fits_terminal_le2018"]["lambda_T1"]
    rt = json.loads((OUT / "fukusho_roundtrip.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(SEED)

    t0 = time.time()
    races = build_year([2022], gamma, lam)
    t_year = time.time() - t0
    rss_after_year = peak_rss_mb()

    # 合成 winner (m_cal から) で 1 年分の fit・Δ・bootstrap を計測
    off = np.concatenate([[0], np.cumsum([len(r["m"]) for r in races])])
    logm = np.log(np.concatenate([r["m"] for r in races]))
    logq = np.log(np.concatenate([r["q"] for r in races]))
    win = np.array([off[i] + rng.choice(len(r["m"]), p=r["m"]) for i, r in enumerate(races)])
    t0 = time.time()
    T.fit_temp(logm, off, win)
    T.fit_cross(logm, logq, off, win)
    t_fit_year = time.time() - t0

    days = np.array([r["day"] for r in races])
    vals = rng.normal(0, 0.3, len(races))
    t0 = time.time()
    dd, inv = np.unique(days, return_inverse=True)
    ds = np.bincount(inv, weights=vals, minlength=len(dd))
    dc = np.bincount(inv, minlength=len(dd)).astype(float)
    pick = rng.integers(0, len(dd), size=(3000, len(dd)))
    _ = (ds[pick].sum(1) / dc[pick].sum(1))
    t_boot_1y = time.time() - t0

    # P1: race 内で source (単勝 π) の馬 identity を置換 → q_T1 再計算 → fit
    t0 = time.time()
    qp = np.concatenate([T.stern_top2(r["pi"][rng.permutation(r["n"])], lam) for r in races])
    T.fit_cross(logm, np.log(qp), off, win)
    t_p1_draw_1y = time.time() - t0
    # P2: 同じ場 × 頭数 × entropy 三分位セル内で source snapshot を別 race へ (同頭数のみ)
    t0 = time.time()
    cut = np.quantile([r["ent"] for r in races], [1 / 3, 2 / 3])
    cells = {}
    for i, r in enumerate(races):
        key = (r["venue"], r["n"], int(np.searchsorted(cut, r["ent"])))
        cells.setdefault(key, []).append(i)
    qs = []
    for i, r in enumerate(races):
        key = (r["venue"], r["n"], int(np.searchsorted(cut, r["ent"])))
        j = rng.choice(cells[key])
        qs.append(T.stern_top2(races[j]["pi"], lam))
    T.fit_cross(logm, np.log(np.concatenate(qs)), off, win)
    t_p2_draw_1y = time.time() - t0

    n_years_total = 8
    eval_years = 5
    # Stage 1 の 1 draw = 5 評価年 × (fit 窓は最大 7 年) の fit。fit 時間は行数にほぼ比例
    fit_cost_units = sum(y - 2016 for y in range(2019, 2024))          # 3+4+5+6+7 年分
    res = {
        "role": "EXP18 Stage 0 の計算量 dry run。合成 winner を使い、結果ラベルは読まない",
        "one_year_build": {"year": 2022, "races_eligible": len(races), "pairs": int(off[-1]),
                           "seconds": round(t_year, 1)},
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "peak_rss_mb_after_one_year": round(rss_after_year, 1),
        "full_period_build_estimate_seconds": round(t_year * n_years_total, 1),
        "fit_one_year_seconds": round(t_fit_year, 2),
        "rolling_fit_per_draw_estimate_seconds": round(t_fit_year * fit_cost_units, 1),
        "bootstrap_3000_one_year_seconds": round(t_boot_1y, 3),
        "bootstrap_3000_five_years_estimate_seconds": round(t_boot_1y * eval_years, 2),
        "P1_one_draw_one_year_seconds": round(t_p1_draw_1y, 2),
        "P1_200_draws_estimate_seconds": round(t_p1_draw_1y * fit_cost_units * 200, 0),
        "P2_one_draw_one_year_seconds": round(t_p2_draw_1y, 2),
        "P2_200_draws_estimate_seconds": round(t_p2_draw_1y * fit_cost_units * 200, 0),
        "T2_maxent": ("計測しない (往復 Gate 未達で T2 未実装)" if not rt.get("gate_pass")
                      else "fukusho_roundtrip が通過した場合は別途計測"),
        "approximation": "近似法は使わない。T1 は閉形式、T2 は順序付き top-k の厳密計算。n<=8 oracle 誤差は "
                         "out/invariant_tests.json (<= 5.6e-14)",
        "power_audit_elapsed_reference": "out/power_audit.json の elapsed_sec",
        "elapsed_sec": round(time.time() - t00, 1),
    }
    (HERE / "COMPUTE_DRY_RUN.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
