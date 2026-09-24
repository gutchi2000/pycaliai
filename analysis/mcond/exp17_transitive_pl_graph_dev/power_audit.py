# -*- coding: utf-8 -*-
"""
power_audit.py — EXP17 Stage 0: G1 機構 Gate (pairwise logloss, PATH − DYNPL) の検出力監査 (label-free)
=====================================================================================================
2019-2023 の実 race 構造 (未対戦かつ共通対戦馬あり pair の配置、meeting-day、EXP02 as-of μ) を使い、
結果 (着順) は一切使わず合成する。

生成モデル (各 rep):
  perf_h = b·μ_h + η_h + Gumbel,  η_h ~ N(0, σ_η²)  (σ_η = 0.5 × レース内 SD(b·μ))
  pair label y_ij = 1[perf_i > perf_j]
  DYNPL arm: logit = a·Δμ  (a を rep 内 1-param logistic で較正; scalar rating が知り得る全て)
  PATH  arm: logit = a'·Δμ + β·ẑ_ij,  ẑ_ij = Δη_ij + ε_ij,  ε ~ N(0, s²)  (2-param logistic で較正)
  真の pair 情報量 s を二分探索で調整し、E[pairwise logloss 改善] を目標 Δ に合わせる。
推論: race 内平均 → meeting-day 平均 → day を推論単位 (pair を独立標本にしない)。
判定 (G1 の統計条件): CI95 上限 < 0 (day cluster 正規近似)、5 年中 4 年改善、LOO 5 通り全て CI95 上限 < 0。
機構 floor (事前固定): 0.001 nats/pair。floor 以上を要求する PASS 確率も併記。
出力: out/power_audit.json
実行: python -m analysis.mcond.exp17_transitive_pl_graph_dev.power_audit
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
OUT = HERE / "out"
EXP02 = BASE / "data" / "_research" / "mcond" / "exp02_features.parquet"
SEED = 20260925
REPS = 400
TARGETS = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]
MECHANISM_FLOOR = 0.001
ETA_SCALE = 0.5


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit_fit(X, y, iters=25):
    """small Newton logistic regression without intercept (antisymmetric problem)."""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = sigmoid(X @ w)
        g = X.T @ (y - p)
        Hm = -(X * (p * (1 - p))[:, None]).T @ X - 1e-9 * np.eye(X.shape[1])
        step = np.linalg.solve(Hm, g)
        w -= step
        if np.abs(step).max() < 1e-9:
            break
    return w


def ll(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    ph = k / n; den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den; h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return [float(c - h), float(c + h)]


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    eq = json.loads((OUT / "equivalence_audit.json").read_text(encoding="utf-8"))
    b = float(eq["bt_pair_scale"]["b"])
    H = pd.read_parquet(OUT / "horses_2019_2023.parquet")
    P = pd.read_parquet(OUT / "pairs_2019_2023.parquet")
    mu = pd.read_parquet(EXP02, columns=["rid16", "hid", "dyn_skill_mu"])
    mu["rid16"] = mu["rid16"].astype(str); mu["hid"] = mu["hid"].astype(str)
    H = H.merge(mu, on=["rid16", "hid"], how="left")
    H["row"] = np.arange(len(H))
    H["bmu"] = b * H["dyn_skill_mu"]
    H["bmu"] = H["bmu"].fillna(H.groupby("rid16")["bmu"].transform("mean")).fillna(0.0)
    key = dict(zip(zip(H["rid16"], H["hid"]), H["row"]))
    P["ri"] = [key.get((r, h), -1) for r, h in zip(P["rid16"], P["hid_i"])]
    P["rj"] = [key.get((r, h), -1) for r, h in zip(P["rid16"], P["hid_j"])]
    P = P[(P.ri >= 0) & (P.rj >= 0)].reset_index(drop=True)
    ri = P["ri"].to_numpy(); rj = P["rj"].to_numpy()
    bmu = H["bmu"].to_numpy()
    dmu = bmu[ri] - bmu[rj]
    race_id = pd.factorize(P["rid16"])[0]; n_races = race_id.max() + 1
    race_day = P.groupby("rid16")["date"].first()
    race_year = (race_day // 10000).to_numpy(); race_day = race_day.to_numpy()
    race_of = pd.factorize(P["rid16"])[1]
    day_of_race = pd.Series(race_day, index=race_of)
    uday, day_idx = np.unique(race_day, return_inverse=True)
    year_of_day = np.array([d // 10000 for d in uday])
    within_sd = H.groupby("rid16")["bmu"].std().mean()
    sigma_eta = ETA_SCALE * within_sd
    struct = {"races": int(n_races), "meeting_days": int(len(uday)), "pairs": int(len(P)),
              "pairs_per_race_mean": float(len(P) / n_races), "within_race_sd_bmu": float(within_sd), "sigma_eta": float(sigma_eta),
              "b": b, "years": {int(y): int((year_of_day == y).sum()) for y in np.unique(year_of_day)}}
    print(struct, flush=True)

    n_h = len(H)

    coef_cache = {}

    def coefs(s_noise, rng):
        """目標ごとに較正 rep で (a, w) を fit して凍結 (rep 毎の再 fit は行わない; 計算量のため)。"""
        k = round(float(s_noise), 6)
        if k not in coef_cache:
            r = np.random.default_rng(SEED + 1000 + int(k * 1e4) % 100000)
            eta = r.normal(0, sigma_eta, n_h); perf = bmu + eta + r.gumbel(size=n_h)
            y = (perf[ri] > perf[rj]).astype(float)
            z = (eta[ri] - eta[rj]) + r.normal(0, s_noise, len(P))
            coef_cache[k] = (logit_fit(dmu[:, None], y)[0], logit_fit(np.c_[dmu, z], y))
        return coef_cache[k]

    def one_rep(s_noise, rng, return_stats=True):
        a, w = coefs(s_noise, rng)
        eta = rng.normal(0, sigma_eta, n_h)
        perf = bmu + eta + rng.gumbel(size=n_h)
        y = (perf[ri] > perf[rj]).astype(float)
        p0 = sigmoid(a * dmu)
        z = (eta[ri] - eta[rj]) + rng.normal(0, s_noise, len(P))
        p1 = sigmoid(np.c_[dmu, z] @ w)
        diff = ll(p1, y) - ll(p0, y)                    # negative = PATH better
        race_mean = np.bincount(race_id, weights=diff, minlength=n_races) / np.bincount(race_id, minlength=n_races)
        day_mean = np.bincount(day_idx, weights=race_mean, minlength=len(uday)) / np.bincount(day_idx, minlength=len(uday))
        if not return_stats:
            return float(diff.mean())
        return day_mean

    def decide(day_mean):
        m = day_mean.mean(); se = day_mean.std(ddof=1) / np.sqrt(len(day_mean))
        ci_up = m + 1.959964 * se
        yrs = np.unique(year_of_day)
        ymeans = np.array([day_mean[year_of_day == yv].mean() for yv in yrs])
        loo_ok = True
        for yv in yrs:
            dm = day_mean[year_of_day != yv]
            if dm.mean() + 1.959964 * dm.std(ddof=1) / np.sqrt(len(dm)) >= 0:
                loo_ok = False
        detect = (ci_up < 0) and ((ymeans < 0).sum() >= 4) and loo_ok
        return dict(mean=float(m), se=float(se), ci_upper=float(ci_up), detect=bool(detect),
                    detect_ci_only=bool(ci_up < 0), years_improved=int((ymeans < 0).sum()), loo_ok=bool(loo_ok),
                    floor_ok=bool(m <= -MECHANISM_FLOOR))

    # ---- calibrate noise s for each target Δ (label-free; uses 3 reps average)
    calib = {}
    def expected_gain(s_noise):
        r = np.random.default_rng(SEED + 7)
        return -np.mean([one_rep(s_noise, r, return_stats=False) for _ in range(2)])
    g_max = expected_gain(0.0)
    struct["max_gain_with_perfect_eta_knowledge"] = float(g_max)
    for tgt in TARGETS:
        if tgt >= g_max:
            calib[tgt] = None; continue
        lo, hi = 0.0, 50.0
        for _ in range(18):
            mid = 0.5 * (lo + hi)
            if expected_gain(mid) > tgt:
                lo = mid
            else:
                hi = mid
        calib[tgt] = 0.5 * (lo + hi)
    print("calibrated noise", calib, flush=True)

    # ---- power
    results = {}
    for tgt in TARGETS:
        s_noise = calib[tgt]
        if s_noise is None:
            results[str(tgt)] = {"skipped": "target exceeds max achievable gain"}; continue
        dets = []; dets_ci = []; floors = []; means = []
        for r in range(REPS):
            dm = one_rep(s_noise, rng)
            d = decide(dm)
            dets.append(d["detect"]); dets_ci.append(d["detect_ci_only"]); floors.append(d["detect"] and d["floor_ok"]); means.append(d["mean"])
        k = int(np.sum(dets)); kc = int(np.sum(dets_ci)); kf = int(np.sum(floors))
        results[str(tgt)] = {"true_effect_nats_per_pair": tgt, "noise_sd": s_noise, "reps": REPS,
                             "success_detect": k, "power_detect": k / REPS, "wilson95_detect": wilson(k, REPS),
                             "power_ci_only": kc / REPS, "success_detect_and_floor": kf, "power_detect_and_floor": kf / REPS,
                             "wilson95_detect_and_floor": wilson(kf, REPS),
                             "realized_mean_effect": float(-np.mean(means)), "realized_effect_sd_across_reps": float(np.std(means))}
        print(tgt, results[str(tgt)], flush=True)

    # one bootstrap validation of the normal cluster approximation (single rep at floor)
    dm = one_rep(calib.get(MECHANISM_FLOOR) or calib[TARGETS[1]], np.random.default_rng(SEED + 99))
    boots = np.array([rng.choice(dm, len(dm), replace=True).mean() for _ in range(3000)])
    validation = {"rep_mean": float(dm.mean()), "normal_se": float(dm.std(ddof=1) / np.sqrt(len(dm))),
                  "bootstrap_se": float(boots.std(ddof=1)), "bootstrap_ci95_upper": float(np.percentile(boots, 97.5)),
                  "normal_ci95_upper": float(dm.mean() + 1.959964 * dm.std(ddof=1) / np.sqrt(len(dm)))}

    # MDE (80%) by linear interpolation on power_detect
    xs = [t for t in TARGETS if str(t) in results and "power_detect" in results[str(t)]]
    ps = [results[str(t)]["power_detect"] for t in xs]
    mde = None
    for k in range(1, len(xs)):
        if ps[k - 1] < 0.8 <= ps[k]:
            mde = xs[k - 1] + (0.8 - ps[k - 1]) * (xs[k] - xs[k - 1]) / (ps[k] - ps[k - 1]); break
    if mde is None and ps and ps[0] >= 0.8:
        mde = f"<= {xs[0]} (smallest tested level already >= 80%)"
    out = {"seed": SEED, "reps": REPS, "structure": struct, "calibrated_noise_by_target": {str(k): v for k, v in calib.items()},
           "mechanism_floor_nats_per_pair": MECHANISM_FLOOR,
           "power_by_true_effect": results, "mde_80pct_detect": mde, "cluster_se_validation": validation,
           "decision_rule": ("機構 floor 0.001 nats/pair を注入したとき detect (CI95上限<0 ∧ 4/5年 ∧ LOO) の power ≥ 0.80 なら G1 は検出可能。"
                             "floor 到達 (点推定 ≤ -floor) を同時に要求した PASS 確率は別に報告し進行可否には使わない (EXP16A と同じ分離)"),
           "caveats": ["placebo 条件は合成では満たされたものとして扱う (実 power はこれより低い)",
                       "生成モデルは per-horse 潜在 η (scalar rating が知らない race-day 変動) を PATH が部分観測する形。真の pair 固有 (非推移的) 効果ではなく、"
                       "『scalar rating が持たない情報を pair 表現が持つ』最良ケースの代理",
                       "cluster SE は正規近似。1 rep の bootstrap で検証 (cluster_se_validation)"],
           "elapsed_sec": time.time() - t0}
    (OUT / "power_audit.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("mde_80pct_detect", "cluster_se_validation")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
