# -*- coding: utf-8 -*-
"""
power_audit.py — EXP21 Stage 0: G1 形状再現の MDE / 検出力と insufficient 閾値 (結果ラベル不使用)
================================================================================================
結果前に固定する設定:
  SEED 20260927 (Stage 1 の bootstrap seed は 20260928、B = 10,000、暦日 cluster)
  対象: 単勝・複勝・馬連の D0 primary 10 帯 (発見 2013-2018 / 評価 2019-2023 の実 ticket 構造)
  真値 (合成): p_j ∝ q_j · exp(δ · z_b(j))、z_b は帯 1 (本命) の +1 から帯 10 (大穴) の −1 まで線形。race 内で再正規化
              (複勝は的中数 places(n) へ再正規化し独立 Bernoulli 近似、確率は 1 で打ち切り)
  帯 ROI の分布: race 単位の解析的平均・分散による正規近似 (帯間は独立近似、宣言)。払戻 = terminal 価格
  G1 判定 (spec): 控除率基準との差の符号一致 >= 8/10 帯 かつ 帯 ROI の Spearman >= 0.70
  検出力 = G1 PASS 率 (2,000 rep)。MDE = 検出力 >= 0.80 となる最小 δ (格子の線形補間)。δ = 0 の PASS 率 = 誤 PASS 率
  δ の解釈用に、帯 1 と帯 10 の期待 ROI 差を併記する
insufficient 閾値 (bands.py と同じ規則、結果前に固定): 較正 null 下の CI95 半幅 > 0.10 / 暦日 < 30 / 期待的中 mass < 30
出力: out/power_audit.json
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
from scipy.stats import spearmanr

from . import bands as B
from . import loaders as L

SEED = 20260927
STAGE1_BOOT_SEED = 20260928
STAGE1_BOOT_B = 10000
REPS = 2000
DELTA_GRID = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]
BASELINE = {"tansho": 0.80, "fukusho": 0.80, "umaren": 0.775}
REQUIRED = 0.80


def load(t, period):
    z = np.load(L.RESEARCH / f"tickets_{t}_D0_{period}.npz")
    return {k: z[k] for k in z.files}


def band_moments(d, t, delta):
    key, q, race, pay = d["key"], d["q"], d["race"], d["pay"]
    band = B.mass_bands(key, q, 10)
    z = 1.0 - 2.0 * band / 9.0
    w = q * np.exp(delta * z)
    _, inv = np.unique(race, return_inverse=True)
    if t == "fukusho":
        k = np.bincount(inv, weights=q)                      # places(n) (q は和 = places に正規化済み)
        p = np.minimum(w * (k / np.bincount(inv, weights=w))[inv], 1.0)
        # 複勝の quote は払戻の範囲で決済額ではない。δ=0 の較正 null で全帯 ROI = 控除率基準になるよう
        # 合成払戻を baseline / q_null に置く (quote キーを払戻に使うと δ=0 でも全帯 > 基準となり誤 PASS 38%)
        q0 = np.minimum(q, 1.0)
        pay = BASELINE["fukusho"] / q0
        var_ticket = p * (1 - p) * pay * pay
        mean = np.bincount(band, weights=p * pay, minlength=10)
        var = np.bincount(band, weights=var_ticket, minlength=10)
    else:
        p = w / np.bincount(inv, weights=w)[inv]
        mean = np.bincount(band, weights=p * pay, minlength=10)
        # race 内の排反: Var(Σ hit o) = Σ p o² − (Σ p o)²  (race×band ごと)
        rb = inv * 10 + band
        _, rinv = np.unique(rb, return_inverse=True)
        s1 = np.bincount(rinv, weights=p * pay)
        s2 = np.bincount(rinv, weights=p * pay * pay)
        bb = np.bincount(rinv, weights=band.astype(float)) / np.bincount(rinv)
        var = np.bincount(bb.astype(int), weights=s2 - s1 * s1, minlength=10)
    n = np.bincount(band, minlength=10).astype(float)
    return mean / n, np.sqrt(var) / n


def g1_pass(roi_d, roi_e, base):
    up_d, up_e = roi_d - base, roi_e - base
    signs = int(np.sum(np.sign(up_d) == np.sign(up_e)))
    rho = spearmanr(roi_d, roi_e).correlation
    return signs >= 8 and rho >= 0.70


def mde(curve):
    xs = sorted(curve)
    for i, x in enumerate(xs):
        if curve[x] >= REQUIRED:
            if i == 0:
                return float(x)
            x0 = xs[i - 1]
            return float(x0 + (REQUIRED - curve[x0]) * (x - x0) / (curve[x] - curve[x0]))
    return None


def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    res = {"seed": SEED, "reps": REPS, "delta_grid": DELTA_GRID, "required_power": REQUIRED,
           "stage1_bootstrap": {"B": STAGE1_BOOT_B, "seed": STAGE1_BOOT_SEED, "cluster": "calendar day (YYYYMMDD)"},
           "g1_rule": "sign agreement of uplift vs takeout baseline >= 8/10 primary bands AND Spearman(band ROI) >= 0.70",
           "insufficient_rule": {"null_ci95_half_width_gt": B.HALF_WIDTH_MAX, "days_lt": B.MIN_DAYS,
                                 "expected_hit_mass_lt": B.MIN_MASS},
           "approximations": ["band ROI ~ Normal(analytic race-level mean, variance); bands independent",
                              "fukusho hits independent Bernoulli with prob capped at 1",
                              "payout = terminal odds for tansho/umaren; fukusho synthetic payout = baseline / q_null so that the "
                              "delta=0 null has ROI = takeout baseline in every band (quote sqrt(Lo*Hi) as payout gave a spurious "
                              "38% false pass at delta=0 in the first, uncommitted run)"],
           "types": {}}
    for t in ("tansho", "fukusho", "umaren"):
        dd, de = load(t, "discovery_2013_2018"), load(t, "evaluation_2019_2023")
        curve, detail = {}, {}
        for delta in DELTA_GRID:
            md, sd = band_moments(dd, t, delta)
            me, se = band_moments(de, t, delta)
            ok = 0
            for _ in range(REPS):
                rd = md + sd * rng.standard_normal(10)
                re = me + se * rng.standard_normal(10)
                ok += g1_pass(rd, re, BASELINE[t])
            curve[delta] = ok / REPS
            detail[str(delta)] = {"g1_pass_rate": ok / REPS, "discovery_band_roi_expected": md.round(4).tolist(),
                                  "evaluation_band_roi_expected": me.round(4).tolist(),
                                  "fav_minus_longshot_roi_spread_discovery": float(md[0] - md[9]),
                                  "band_roi_sd_discovery": sd.round(4).tolist()}
        m = mde(curve)
        res["types"][t] = {"baseline": BASELINE[t], "curve": detail, "false_pass_rate_delta0": curve[0.0],
                           "mde_delta": m,
                           "mde_roi_spread_approx": (float(np.interp(m, DELTA_GRID, [detail[str(x)]["fav_minus_longshot_roi_spread_discovery"]
                                                                                         for x in DELTA_GRID])) if m is not None else None)}
        print(f"[{t}] MDE δ={m} false pass={curve[0.0]:.3f} curve=" +
              json.dumps({str(k): round(v, 3) for k, v in curve.items()}), flush=True)
    res["elapsed_sec"] = round(time.time() - t0, 1)
    (L.OUT / "power_audit.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[saved] power_audit.json ({res['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
