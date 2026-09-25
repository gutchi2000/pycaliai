# -*- coding: utf-8 -*-
"""
fukusho_roundtrip.py — EXP18 S0-C: 複勝 Lo/Hi の実データ往復 Gate
=================================================================
TOMOGRAPHY_IDENTIFIABILITY_AUDIT.md §2-§4 で**実行前に固定した**式・解法・Gate をそのまま使う。
使うのは市場構造 loader の表示オッズだけ (着順・払戻は読まない)。
  Gate: terminal snapshot・base race set・2013-2023 の**各年**で、
        Lo と Hi の両方が |再生成 − 表示| <= max(0.05, 0.02×表示) の馬行が 99% 以上
  未達: T2 は未実装として停止 (別の関数形で救済しない)
出力: out/fukusho_roundtrip.json
実行: python -m analysis.mcond.exp18_cross_pool_market_tomography_dev.fukusho_roundtrip
"""
from __future__ import annotations

import json
import time

import numpy as np

from . import payout_formula as PF
from .loaders import OUT, load_structure
from .market_build import base_race_status, pool_matrices, race_arrays, snapshot_index

YEARS = list(range(2013, 2024))
GATE_RATE = 0.99


def run_snapshot(which, idx, W, LO, HI, U, info):
    per_year = {}
    band = {}
    diag = {"lo_regen_lt_display": 0, "lo_regen_gt_display": 0, "hi_regen_lt_display": 0,
            "hi_regen_gt_display": 0, "rows": 0, "spread_ratio_display": [], "spread_ratio_regen": [],
            "rel_err_lo": [], "rel_err_hi": []}
    k_split = {2: [0, 0], 3: [0, 0]}
    tie_races = 0
    for rid, row in idx.iterrows():
        ti = row.get(f"{which}_tan")
        if not (ti == ti):
            continue
        y = int(rid[:4])
        arr = race_arrays(W, LO, HI, U, int(ti), -1)
        inf = info.loc[rid] if rid in info.index else None
        if which == "term":
            if base_race_status(inf, arr) != "base":
                continue
        else:
            if inf is None or bool(inf["is_jump"]) or arr["n"] < 5 or not (
                    np.all(np.isfinite(arr["place_lo"])) and np.all(arr["place_lo"] > 0)
                    and np.all(np.isfinite(arr["place_hi"])) and np.all(arr["place_hi"] > 0)):
                continue
        n = arr["n"]
        k = PF.places(n)
        lo, hi = arr["place_lo"], arr["place_hi"]
        if len(set(zip(lo, hi))) < n:
            tie_races += 1
        v = PF.invert_display(lo, hi, k)
        lo2, hi2 = PF.display_lo_hi(v, k)
        ok = PF.roundtrip_ok(lo, lo2) & PF.roundtrip_ok(hi, hi2)
        tau = PF.tau_from_shares(v, k)
        assert abs(tau.sum() - k) <= 1e-6, "place_sum 構成 assert"
        d = per_year.setdefault(y, {"races": 0, "horse_rows": 0, "horse_rows_ok": 0,
                                    "races_all_ok": 0, "max_abs_err_lo": 0.0, "max_abs_err_hi": 0.0})
        d["races"] += 1
        d["horse_rows"] += n
        d["horse_rows_ok"] += int(ok.sum())
        d["races_all_ok"] += int(ok.all())
        d["max_abs_err_lo"] = max(d["max_abs_err_lo"], float(np.abs(lo2 - lo).max()))
        d["max_abs_err_hi"] = max(d["max_abs_err_hi"], float(np.abs(hi2 - hi).max()))
        diag["rows"] += n
        diag["lo_regen_lt_display"] += int((lo2 < lo).sum())
        diag["lo_regen_gt_display"] += int((lo2 > lo).sum())
        diag["hi_regen_lt_display"] += int((hi2 < hi).sum())
        diag["hi_regen_gt_display"] += int((hi2 > hi).sum())
        if len(diag["rel_err_lo"]) < 400000:
            diag["spread_ratio_display"].extend((hi / lo).tolist())
            diag["spread_ratio_regen"].extend((hi2 / lo2).tolist())
            diag["rel_err_lo"].extend(((lo2 - lo) / lo).tolist())
            diag["rel_err_hi"].extend(((hi2 - hi) / hi).tolist())
        k_split[k][0] += n
        k_split[k][1] += int(ok.sum())
        for x, o in zip(lo, ok):
            b = "<2" if x < 2 else "2-5" if x < 5 else "5-20" if x < 20 else ">=20"
            e = band.setdefault(b, [0, 0])
            e[0] += 1
            e[1] += int(o)
    for d in per_year.values():
        d["horse_row_pass_rate"] = d["horse_rows_ok"] / d["horse_rows"]
        d["race_all_ok_rate"] = d["races_all_ok"] / d["races"]
    return {"per_year": {str(k): v for k, v in sorted(per_year.items())},
            "by_place_lo_band": {b: {"rows": v[0], "pass_rate": v[1] / v[0]} for b, v in band.items()},
            "by_k": {str(k): {"rows": v[0], "pass_rate": (v[1] / v[0] if v[0] else None)}
                     for k, v in k_split.items()},
            "races_with_display_ties": tie_races,
            "mismatch_diagnostics": {
                "note": "記述統計のみ。固定した式・Gate は変えず、別の関数形は試していない",
                "rows": diag["rows"],
                "share_lo_regen_below_display": diag["lo_regen_lt_display"] / max(diag["rows"], 1),
                "share_lo_regen_above_display": diag["lo_regen_gt_display"] / max(diag["rows"], 1),
                "share_hi_regen_below_display": diag["hi_regen_lt_display"] / max(diag["rows"], 1),
                "share_hi_regen_above_display": diag["hi_regen_gt_display"] / max(diag["rows"], 1),
                "median_spread_ratio_hi_over_lo_display": float(np.median(diag["spread_ratio_display"])) if diag["rows"] else None,
                "median_spread_ratio_hi_over_lo_regen": float(np.median(diag["spread_ratio_regen"])) if diag["rows"] else None,
                "median_rel_err_lo": float(np.median(diag["rel_err_lo"])) if diag["rows"] else None,
                "median_rel_err_hi": float(np.median(diag["rel_err_hi"])) if diag["rows"] else None}}


def main():
    import sys
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    t0 = time.time()
    st = load_structure(YEARS)
    tan, um, info = st["tan"], st["um"], st["info"]
    W, LO, HI, U = pool_matrices(tan, um)
    idx = snapshot_index(tan, um, info)
    term = run_snapshot("term", idx, W, LO, HI, U, info)
    pre = run_snapshot("pre", idx, W, LO, HI, U, info)
    rates = {y: v["horse_row_pass_rate"] for y, v in term["per_year"].items()}
    gate = all(r >= GATE_RATE for r in rates.values()) and len(rates) == len(YEARS)
    res = {
        "role": "複勝 Lo/Hi 表示の往復 Gate (T2 の前提)。表示オッズだけを使い、着順・払戻は読まない",
        "formula": "TOMOGRAPHY_IDENTIFIABILITY_AUDIT.md §2 (実行前に固定)",
        "tolerance": "Lo と Hi の両方が |再生成 − 表示| <= max(0.05, 0.02×表示)",
        "gate_rule": f"terminal・base race set・{YEARS[0]}-{YEARS[-1]} の各年で馬行合格率 >= {GATE_RATE}",
        "terminal": term,
        "historical_pre_snapshot_reference": pre,
        "terminal_yearly_pass_rates": rates,
        "min_yearly_pass_rate": min(rates.values()) if rates else None,
        "gate_pass": bool(gate),
        "decision": ("T2_implementable" if gate else "T2_unimplemented_stop"),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fukusho_roundtrip.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                                encoding="utf-8")
    print(json.dumps({"rates": rates, "gate_pass": res["gate_pass"],
                      "by_k": term["by_k"], "by_band": term["by_place_lo_band"]}, ensure_ascii=False))
    print(f"[saved] out/fukusho_roundtrip.json ({res['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
