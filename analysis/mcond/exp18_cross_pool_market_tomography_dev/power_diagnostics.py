# -*- coding: utf-8 -*-
"""
power_diagnostics.py — EXP18 S0-E 補助診断 (記述用。Gate・floor の定義は変えない)
==================================================================================
power_audit.py の宣言生成構造 (SD(noise)=sqrt(2Δ)) では期待対数成長が格子全域で負になり、
PRACTICAL floor が定まらなかった。それが実装の誤りでなく宣言慣習の帰結であることを確かめる。
結果ラベルは使わない (2019-2023 の市場構造と合成 p だけ)。

  D1 noise-free 対照: q_est = p (noise 0)。現金込み Kelly は真の p の下で期待成長 >= 0 でなければならない
     (負なら Kelly・決済・参加条件の実装誤り)。
  D2 格子の延長: 宣言慣習のまま Δ = 0.3, 0.5, 1.0 でも成長が負のままか (単調性の確認)。
  D3 解析的な分解: 同じ p・q_est で E_p[log q_est − log m] = KL(p‖m) − KL(p‖q_est) を測り、
     宣言慣習では推定の情報利得が 0 付近に潰れることを示す。
出力: out/power_diagnostics.json
"""
from __future__ import annotations

import json
import time

import numpy as np

from . import tomography as T
from .loaders import OUT
from .power_audit import (ENTRY, EVAL_YEARS, SEED, TAKEOUT, Seg, calibrate_eps, load_markets, tilt,
                          zscore)

DIAG_SEED = SEED + 1               # power_audit と別系列 (実行前に固定)
D1_GRID = [0.001, 0.005, 0.02, 0.05, 0.1, 0.2]
D2_GRID = [0.3, 0.5, 1.0]
D3_GRID = [0.001, 0.005, 0.02, 0.05, 0.1, 0.2]
DRAWS = 3


def growth(logm, s, sg, delta, noise_sd, rng, draws):
    m = np.exp(logm)
    odds = (1.0 - TAKEOUT) / m
    eps = calibrate_eps(logm, s, sg, delta)
    p = tilt(logm, s, eps, sg)
    gs, ent = [], 0
    for _ in range(draws):
        noise = rng.normal(0, noise_sd, len(logm)) if noise_sd > 0 else 0.0
        qe = tilt(logm, s * eps + noise, 1.0, sg)
        rm = sg.maxv(qe / m)
        g = 0.0
        for r in np.flatnonzero(rm > ENTRY):
            a, b = sg.off[r], sg.off[r + 1]
            g += T.kelly_cash_growth(qe[a:b], odds[a:b], p[a:b])[0]
        ent += int((rm > ENTRY).sum())
        gs.append(g / sg.R)
    return {"eps": eps, "noise_sd": noise_sd, "mean_growth_per_race": float(np.mean(gs)),
            "min_over_draws": float(np.min(gs)), "share_races_entered": ent / (sg.R * draws)}


def info_gain(logm, s, sg, delta, rng):
    eps = calibrate_eps(logm, s, sg, delta)
    p = tilt(logm, s, eps, sg)
    qe = tilt(logm, s * eps + rng.normal(0, np.sqrt(2 * delta), len(logm)), 1.0, sg)
    kl_pm = float(sg.sum(p * (np.log(p) - logm)).mean())
    kl_pq = float(sg.sum(p * (np.log(p) - np.log(qe))).mean())
    return {"KL_p_m": kl_pm, "KL_p_qest": kl_pq, "expected_LL_gain_of_qest_over_m": kl_pm - kl_pq}


def main():
    import sys
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    t0 = time.time()
    anc = json.loads((OUT / "anchor_le2018.json").read_text(encoding="utf-8"))
    fx = anc["dryrun_fits_terminal_le2018"]
    year, day, off, m, q = load_markets(fx["gamma_powerlaw_devig"], fx["lambda_T1"])
    sg = Seg(off)
    logm, logq = np.log(m), np.log(q)
    s = zscore(logq - logm, sg)
    ev = np.flatnonzero(np.isin(year, EVAL_YEARS))
    e0, e1 = off[ev[0]], off[ev[-1] + 1]
    sge = Seg(off[ev[0]:ev[-1] + 2] - e0)
    lm, ss = logm[e0:e1], s[e0:e1]
    rng = np.random.default_rng(DIAG_SEED)

    d1 = {str(d): growth(lm, ss, sge, d, 0.0, rng, 1) for d in D1_GRID}
    print("[D1 noise-free] " + json.dumps({k: round(v["mean_growth_per_race"], 6) for k, v in d1.items()}),
          flush=True)
    d2 = {str(d): growth(lm, ss, sge, d, float(np.sqrt(2 * d)), rng, DRAWS) for d in D2_GRID}
    print("[D2 declared, extended] " + json.dumps({k: round(v["mean_growth_per_race"], 6)
                                                    for k, v in d2.items()}), flush=True)
    d3 = {str(d): info_gain(lm, ss, sge, d, rng) for d in D3_GRID}
    print("[D3 info gain] " + json.dumps({k: round(v["expected_LL_gain_of_qest_over_m"], 6)
                                          for k, v in d3.items()}), flush=True)
    res = {
        "role": "power_audit の補助診断 (記述用)。floor・Gate・生成構造の定義は変えない。結果ラベル不使用",
        "seed": DIAG_SEED, "seed_fixed_before_run": True,
        "D1_noise_free_kelly_control": d1,
        "D1_all_nonnegative": bool(all(v["min_over_draws"] >= -1e-12 for v in d1.values())),
        "D2_declared_convention_extended_grid": d2,
        "D2_all_negative": bool(all(v["mean_growth_per_race"] < 0 for v in d2.values())),
        "D3_expected_logloss_gain_of_estimate": d3,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (OUT / "power_diagnostics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                                encoding="utf-8")
    print(f"[saved] out/power_diagnostics.json ({res['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
