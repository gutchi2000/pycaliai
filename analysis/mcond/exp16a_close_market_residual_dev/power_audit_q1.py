# -*- coding: utf-8 -*-
"""
power_audit_q1.py — EXP16A Stage 1-2: **実方向** log(Q1/π) での検出力監査
=========================================================================
spec v0.4 (凍結, commit e7603678) の power_audit.tier2 を、Stage 0 で測れなかった
`q1_log_ratio` 方向で再実行する。2019-2023 の実結果指標は開封しない
(使うのは 2022 以前の正式 race set と、rolling OOF から得た Q1 確率だけ)。

Stage 0 との違い:
  * 方向 s を合成ノイズ (seed_jitter) ではなく **実測の 5 seed の log(Q1/π)** にする
    - 真の傾斜方向: 5 seed の標準化方向の平均を再標準化したもの (consensus)
    - 各 seed の fit: その seed 自身の方向 (= 実測の seed 間ばらつきがそのまま入る)
  * 判定規則・実務床・等級 (gate_grade) は**一切変更しない**
出力: out/power_audit_q1.json
実行: python -m analysis.mcond.exp16a_close_market_residual_dev.power_audit_q1
"""
from __future__ import annotations

import json
import time

import numpy as np

from .provenance import OUT
from .gate_grade import FLOOR, grade_gate, wilson
from .power_audit import (BOOT, EVAL_YEARS, FIT_YEAR_MIN, RNG_SEED, SEEDS, DayBoot, Seg,
                          calibrate_delta, delta_per_race, draw_winners, fit_beta,
                          interp_threshold, load_races, zscore)

RESEARCH_Q1 = None      # 下で build
REPS = 400
TARGETS = [0.005, 0.008]


def load_q1_dirs(d: dict, sg: Seg, pi: np.ndarray):
    """q1_official.npz (全年) を ≤2022 の並びへ key で対応付け、seed 別の標準化方向を作る"""
    from .build_oof import RESEARCH
    z = np.load(RESEARCH / "q1_official.npz", allow_pickle=True)
    key_src, prob, seeds = z["key"].astype(str), z["prob"], z["seeds"].tolist()
    idx = {k: i for i, k in enumerate(key_src)}
    keys = np.array([f"{d['rid16'][i]}_{d['ban'][j]}"
                     for i in range(len(d["rid16"]))
                     for j in range(d["offsets"][i], d["offsets"][i + 1])])
    take = np.array([idx[k] for k in keys])
    assert len(take) == len(pi)
    dirs = []
    for si in range(len(seeds)):
        raw = np.log(np.clip(prob[si][take], 1e-12, None)) - np.log(np.clip(pi, 1e-12, None))
        dirs.append(zscore(raw, sg))
    dirs = np.array(dirs)
    cons = zscore(dirs.mean(axis=0), sg)
    spread = dirs.std(axis=0, ddof=1)
    stats = {
        "seeds": seeds,
        "mean_per_horse_sd_across_seeds": float(spread.mean()),
        "median_per_horse_sd_across_seeds": float(np.median(spread)),
        "mean_pairwise_correlation": float(np.mean([np.corrcoef(dirs[i], dirs[j])[0, 1]
                                                   for i in range(len(seeds))
                                                   for j in range(i + 1, len(seeds))])),
        "mean_corr_seed_vs_consensus": float(np.mean([np.corrcoef(dirs[i], cons)[0, 1]
                                                      for i in range(len(seeds))])),
        "note": "seed_jitter の合成値ではなく実測の seed 間ばらつき",
    }
    return cons, dirs, stats


def empirical_power_real(pi_all, s_truth, s_by_seed, sg_all, meta, base_name, gate: str,
                         reps: int, target: float, floor: float = FLOOR,
                         rng_seed: int = RNG_SEED):
    """真の効果 = target を s_truth 方向に注入し、各 seed は自分の実測方向で fit する"""
    year, day = meta["year"], meta["day"]
    delta, real_impr = calibrate_delta(pi_all, s_truth, sg_all, target)
    n_race = sg_all.R
    yslice = {}
    for y in sorted(set(year.tolist())):
        r = np.flatnonzero(year == y)
        yslice[y] = (int(r[0]), int(r[-1] + 1))
    hslice = {y: (int(sg_all.off[a]), int(sg_all.off[b])) for y, (a, b) in yslice.items()}

    def sub_seg(a, b):
        return Seg(sg_all.off[a:b + 1] - sg_all.off[a])

    segs = {y: sub_seg(*yslice[y]) for y in yslice}
    fitsets = {}
    for y in EVAL_YEARS:
        r0, r1 = yslice[FIT_YEAR_MIN][0], yslice[y - 1][1]
        fitsets[y] = (r0, r1, sub_seg(r0, r1), int(sg_all.off[r0]), int(sg_all.off[r1]))
    boot = DayBoot(year, day, EVAL_YEARS)
    rng = np.random.default_rng(rng_seed)
    counts = {"PASS-PRACTICAL": 0, "PASS-SIGNAL": 0, "FAIL": 0}
    fail_reasons, est, warm = {}, [], {y: delta for y in EVAL_YEARS}

    for _ in range(reps):
        win_rows = draw_winners(pi_all, s_truth, delta, sg_all, rng)
        pooled, per_year_vals, dvals = [], [], []
        for j in range(SEEDS):
            sj = s_by_seed[j]
            val = np.full(n_race, np.nan)
            for y in EVAL_YEARS:
                r0, r1, sgf, h0, h1 = fitsets[y]
                beta = fit_beta(pi_all[h0:h1], sj[h0:h1], sgf, win_rows[r0:r1] - h0, b0=warm[y])
                warm[y] = beta
                a, b = yslice[y]
                hy0, hy1 = hslice[y]
                val[a:b] = delta_per_race(pi_all[hy0:hy1], sj[hy0:hy1], segs[y], beta,
                                          win_rows[a:b] - hy0)
            m = np.array([np.nanmean(val[slice(*yslice[y])]) for y in EVAL_YEARS])
            ev = np.concatenate([val[slice(*yslice[y])] for y in EVAL_YEARS])
            pooled.append(float(ev.mean()))
            per_year_vals.append(m)
            dvals.append(val)
        pooled = np.array(pooled)
        k = int(np.argsort(pooled)[SEEDS // 2])
        med_val, med_year = dvals[k], per_year_vals[k]
        lo, hi = boot.ci(med_val, rng)
        loo_uppers = [boot.ci(med_val, rng, drop_year=y)[1] for y in EVAL_YEARS]
        g = grade_gate(gate, point=float(pooled[k]), ci_lower=lo, ci_upper=hi,
                       years_improved=int((med_year < 0).sum()), n_years=len(EVAL_YEARS),
                       seeds_improved=int((pooled < 0).sum()), n_seeds=SEEDS,
                       loo_ci_uppers=loo_uppers,
                       placebo_exceeded=(True if gate == "A" else None), floor=floor)
        counts[g["grade"]] += 1
        for r in g["fail_reasons"]:
            fail_reasons[r.split(" ")[0]] = fail_reasons.get(r.split(" ")[0], 0) + 1
        est.append([pooled[k], lo, hi, max(loo_uppers)])
    est = np.array(est)
    npr = counts["PASS-PRACTICAL"]
    ndet = npr + counts["PASS-SIGNAL"]
    wp, wd = wilson(npr, reps), wilson(ndet, reps)
    return {
        "base_market": base_name, "direction": "q1_log_ratio (real 5 seeds)", "gate": gate,
        "target_nats": target, "practical_floor_used": floor, "injected_delta": delta,
        "realized_true_improvement_nats": real_impr,
        "reps": reps, "rng_seed": rng_seed, "seeds": SEEDS, "n_races": int(n_race),
        "eval_years": EVAL_YEARS, "counts": counts,
        "successes_pass_practical": npr, "successes_pass_practical_or_signal": ndet,
        "power_pass_practical": npr / reps, "power_pass_practical_or_signal": ndet / reps,
        "wilson95_pass_practical": [wp[0], wp[1]],
        "wilson95_pass_practical_or_signal": [wd[0], wd[1]],
        "fail_reason_counts": fail_reasons,
        "median_point_estimate": float(np.median(est[:, 0])),
        "median_ci95": [float(np.median(est[:, 1])), float(np.median(est[:, 2]))],
        "median_worst_loo_ci_upper": float(np.median(est[:, 3])),
        "share_point_estimate_beyond_floor": float((est[:, 0] < -floor).mean()),
        "attenuation_vs_injected": float(np.median(est[:, 0]) / -target),
        "placebo_in_simulation": ("満たされたものとして扱う (placebo は合成できない)"
                                  if gate == "A" else "Gate B の等級条件に placebo は含めない"),
    }


def main():
    t0 = time.time()
    d = load_races()
    sg = Seg(d["offsets"])
    cons_close, dirs_close, stats = load_q1_dirs(d, sg, d["pi_close"])
    cons_pre, dirs_pre, stats_pre = load_q1_dirs(d, sg, d["pi_pre"])
    res = {
        "role": "実方向 log(Q1/π) での検出力監査 (Stage 1-2)。判定規則・実務床・等級は spec v0.4 のまま",
        "created": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "data": {"races": int(sg.R), "horses": int(len(d["ban"])),
                 "years": sorted(set(d["year"].tolist())),
                 "note": "2022 以前の正式 race set のみ。2023 は読まない"},
        "direction_source": {"artifact": "data/_research/mcond/exp16a/q1_official.npz",
                             "definition": "log(Q1_seed / π) をレース内標準化",
                             "truth_direction": "5 seed の標準化方向の平均を再標準化した consensus",
                             "seed_fits": "各 seed は自分の実測方向で fit する (合成 jitter を使わない)"},
        "measured_seed_spread_close": stats,
        "measured_seed_spread_pre": stats_pre,
        "grading_implementation": "gate_grade.grade_gate (spec v0.4 と同一)",
        "runs": {},
    }
    for tg in TARGETS:
        k = f"terminal_close_market__q1_log_ratio__true_{tg}"
        print(f"[q1 power] {k} ({round(time.time()-t0)}s)", flush=True)
        res["runs"][k] = empirical_power_real(d["pi_close"], cons_close, dirs_close, sg, d,
                                              "terminal_close_market", "A", REPS, tg)
        print("   ", json.dumps({x: res["runs"][k][x] for x in
                                 ["power_pass_practical", "power_pass_practical_or_signal"]}),
              flush=True)
    k = "historical_pre_snapshot__q1_log_ratio__true_0.005"
    print(f"[q1 power] {k} ({round(time.time()-t0)}s)", flush=True)
    res["runs"][k] = empirical_power_real(d["pi_pre"], cons_pre, dirs_pre, sg, d,
                                          "historical_pre_snapshot", "B", REPS, 0.005)

    floor_run = res["runs"]["terminal_close_market__q1_log_ratio__true_0.005"]
    res["verdict"] = {
        "progression_rule": "真の効果 0.005 で PASS-PRACTICAL ∪ PASS-SIGNAL の検出確率 >= 80% "
                            "(PASS-PRACTICAL 単独の 80% は要求しない)",
        "power_pass_practical_or_signal_at_floor": floor_run["power_pass_practical_or_signal"],
        "wilson95_lower_at_floor": floor_run["wilson95_pass_practical_or_signal"][0],
        "power_pass_practical_at_floor": floor_run["power_pass_practical"],
        "reps": REPS, "rng_seed": RNG_SEED,
        "result": ("proceed_to_2019_2023_evaluation"
                   if (floor_run["power_pass_practical_or_signal"] >= 0.80 and
                       floor_run["wilson95_pass_practical_or_signal"][0] >= 0.75)
                   else "stop_do_not_open_2019_2023"),
        "note": "停止時も閾値・期間・Gate は変更しない",
    }
    res["elapsed_sec"] = round(time.time() - t0, 1)
    (OUT / "power_audit_q1.json").write_text(json.dumps(res, ensure_ascii=False, indent=1,
                                                        default=float), encoding="utf-8")
    print(json.dumps(res["verdict"], ensure_ascii=False))
    print(json.dumps(res["measured_seed_spread_close"], ensure_ascii=False))
    print(f"[saved] out/power_audit_q1.json ({res['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
