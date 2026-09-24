# -*- coding: utf-8 -*-
"""
power_audit.py — EXP16A Stage 0 (改訂3): 検出力監査を 3 段に分け、等級別に測る
==============================================================================
学習しない (LightGBM は使わない)。2023 は一切読まない。ROI 評価も候補生成もしない。

  tier1 `ideal_local_approximation`
        SD(Δ_r) ≈ sqrt(2Δ̄) と MDE ≈ 2·2.8²/R。局所指数傾斜・最適方向・正しいモデル
        仕様・独立レースを仮定した **理論的な楽観側の基準**。本実験の保証値ではない。
  tier2 `empirical_cluster_power`  ← 正式な検出力判定
        2022 以前の正式 race set の実構造を保ったまま、真の期待 logloss 改善が指定値になる
        局所傾斜 p_δ(i) ∝ π(i)·exp(δ·s_i) を注入し、合成 winner を反復生成して
        **実際と同じ 年別 fit → 年層化 meeting-day bootstrap → seed 判定 → leave-one-year-out**
        を通し、gate_grade.grade_gate で **PASS-PRACTICAL / PASS-SIGNAL / FAIL** を判定する。
        進行条件は `真の効果 0.005 で PASS-PRACTICAL ∪ PASS-SIGNAL の確率 >= 80%`。
        PASS-PRACTICAL 単独の 80% は要求しない (真の効果が境界値と等しいとき、CI 全体で
        境界を超える証明力が低いのは統計的に自然)。
  tier3 `secondary_reference_check`
        2022 の pre→close 差について理論 SD と実測 cluster SE を比較する。一致しても
        普遍式の証明とは扱わない。

方向 (s):
  * `q1_log_ratio`              = log(Q1/π)。rolling OOF Q1 が必要なため Stage 0 では測れない
                                  (Stage 1 で OOF 作成直後・結果開封前に必ず測る)
  * `prefixed_residual_linear`  = 事前固定した残差特徴の線形予測子 (符号のみ事前固定・等重み)
  * `standardized_random`       = 標準化した乱数方向 (per-race 分散の形だけを変える対照)

出力: out/power_audit.json
実行: python -m analysis.mcond.exp16a_close_market_residual_dev.power_audit
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .provenance import BASE, OUT, MASTER
from .gate_grade import FLOOR, grade_gate, wilson

RESEARCH = BASE / "data" / "_research" / "mcond" / "exp16a"
NPZ = RESEARCH / "official_races_le2022.npz"
Z80 = 1.959964 + 0.841621        # 両側 5% / 検出力 80% ≒ 2.80
EVAL_YEARS = [2018, 2019, 2020, 2021, 2022]      # 5 年判定の ≤2022 版アナログ
FIT_YEAR_MIN = 2016
SEEDS = 5
JITTERS = [0.0, 0.1, 0.25]       # seed 間ばらつきの事前固定水準 (Stage 1 で実測値へ差し替え)
# 真の効果の水準。0.005 = 実務床、0.0069 = 実務床超えを 80% 程度で証明できる効果量として記録する値
DECISION_TARGETS = [0.005, 0.0069]
CURVE_TARGETS = [0.005, 0.0069, 0.0075, 0.010, 0.015, 0.020, 0.030]
REPS_DECISION = 400
REPS_CURVE = 200
BOOT = 1000
RNG_SEED = 20260924

# 事前固定した残差特徴と符号 (0 = 方向の事前知識が無いので合成方向に使わない)
RESIDUAL_SIGNS = {
    "前走確定着順": -1.0, "前走上り3F順": -1.0, "kako5_avg_pos": -1.0, "kako5_best_pos": -1.0,
    "horse_fuku30": +1.0, "jockey_fuku90": +1.0, "prev_hosei": -1.0, "間隔": 0.0,
}


# ---------------------------------------------------------------- segment utils
class Seg:
    """レース = 連続セグメントとして扱うための道具"""

    def __init__(self, offsets: np.ndarray):
        self.off = offsets
        self.start = offsets[:-1]
        self.cnt = np.diff(offsets)
        self.R = len(self.cnt)
        self.idx = np.repeat(np.arange(self.R), self.cnt)

    def sum(self, x):
        return np.add.reduceat(x, self.start)

    def spread(self, v):
        return np.repeat(v, self.cnt)


def zscore(x: np.ndarray, sg: Seg) -> np.ndarray:
    """レース内 z 化。NaN はレース平均 (= 0) に置く"""
    x = x.astype(float)
    ok = np.isfinite(x)
    xs = np.where(ok, x, 0.0)
    n = sg.sum(ok.astype(float))
    s = sg.sum(xs)
    mu = np.where(n > 0, s / np.maximum(n, 1), 0.0)
    c = np.where(ok, xs - sg.spread(mu), 0.0)
    sd = np.sqrt(np.maximum(sg.sum(c * c) / np.maximum(n, 1), 0.0))
    return c / np.maximum(sg.spread(sd), 1e-9)


# ---------------------------------------------------------------- data loading
def load_races():
    d = np.load(NPZ, allow_pickle=True)
    rid, year, day = d["rid16"].astype(str), d["year"], d["day"].astype(str)
    wpos, ban = d["winner_pos"], d["ban"]
    pic, pip, off = d["pi_close"], d["pi_pre"], d["offsets"]
    order = np.lexsort((rid, year))                     # 年 → rid16 で並べ替え (年が連続になる)
    cnt = np.diff(off)
    new_off = np.concatenate([[0], np.cumsum(cnt[order])])
    hidx = np.concatenate([np.arange(off[i], off[i + 1]) for i in order])
    return {"rid16": rid[order], "year": year[order], "day": day[order],
            "winner_pos": wpos[order], "ban": ban[hidx],
            "pi_close": pic[hidx], "pi_pre": pip[hidx], "offsets": new_off}


def load_features(rid16: np.ndarray, ban: np.ndarray, sg: Seg) -> tuple[np.ndarray, dict]:
    """事前固定した残差特徴 → レース内 z → 符号付き等重み和 → 再標準化"""
    cols = ["日付", "レースID(新/馬番無)", "馬番"] + list(RESIDUAL_SIGNS)
    want = set(rid16)
    parts = []
    for ch in pd.read_csv(MASTER, encoding="utf-8-sig", dtype=str, usecols=cols, chunksize=200_000):
        dd = pd.to_numeric(ch["日付"], errors="coerce")
        ch = ch[(dd >= 20160101) & (dd <= 20221231)]
        if not len(ch):
            continue
        ch = ch.assign(rid16=ch["レースID(新/馬番無)"].astype(str)
                       .str.replace(r"\.0$", "", regex=True).str[:16])
        ch = ch[ch["rid16"].isin(want)]
        if len(ch):
            parts.append(ch)
    f = pd.concat(parts, ignore_index=True)
    f["ban"] = pd.to_numeric(f["馬番"], errors="coerce")
    key = pd.DataFrame({"rid16": sg.spread(rid16), "ban": ban.astype(float)})
    f = key.merge(f.drop_duplicates(["rid16", "ban"]), on=["rid16", "ban"], how="left")
    assert len(f) == len(ban)
    cover, s = {}, np.zeros(len(ban))
    for c, sign in RESIDUAL_SIGNS.items():
        v = pd.to_numeric(f[c], errors="coerce").to_numpy(dtype=float)
        cover[c] = {"non_null_share": float(np.isfinite(v).mean()), "sign": sign}
        if sign:
            s = s + sign * zscore(v, sg)
    return zscore(s, sg), cover


# ---------------------------------------------------------------- tilt / fit
def calibrate_delta(pi: np.ndarray, s: np.ndarray, sg: Seg, target: float) -> tuple[float, float]:
    """真の期待 logloss 改善 (= mean_r KL(p_δ||π)) が target になる δ を二分探索で求める"""
    def improvement(delta):
        e = pi * np.exp(delta * s)
        Z = sg.sum(e)
        p = e / sg.spread(Z)
        contrib = p * (delta * s - sg.spread(np.log(Z)))
        return float(sg.sum(contrib).mean())

    lo, hi = 0.0, 0.05
    while improvement(hi) < target:
        hi *= 2
        if hi > 50:
            raise RuntimeError("delta not found")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if improvement(mid) < target:
            lo = mid
        else:
            hi = mid
    mid = 0.5 * (lo + hi)
    return mid, improvement(mid)


def draw_winners(pi: np.ndarray, s: np.ndarray, delta: float, sg: Seg, rng) -> np.ndarray:
    """p_δ ∝ π exp(δ s) から Gumbel-max で合成 winner の **グローバル行 index** を引く"""
    logp = np.log(pi) + delta * s
    g = -np.log(-np.log(rng.random(len(pi))))
    key = logp + g
    mx = np.maximum.reduceat(key, sg.start)
    hit = np.flatnonzero(key == sg.spread(mx))
    first = np.unique(sg.idx[hit], return_index=True)[1]
    return hit[first]


def fit_beta(pi, s, sg: Seg, win_rows: np.ndarray, b0: float = 0.0, iters: int = 30) -> float:
    """offset log π + β s のレース内条件付きロジットを 1 次元 Newton で解く (warm start 可)"""
    b = float(b0)
    sw = s[win_rows].sum()
    for _ in range(iters):
        e = pi * np.exp(b * s)
        Z = sg.sum(e)
        w = e / sg.spread(Z)
        Es = sg.sum(w * s)
        Es2 = sg.sum(w * s * s)
        grad = sw - Es.sum()
        hess = -(Es2 - Es ** 2).sum()
        if hess == 0 or not np.isfinite(hess):
            break
        step = grad / hess
        b -= step
        if abs(step) < 1e-9:
            break
    return float(b)


def delta_per_race(pi, s, sg: Seg, beta: float, win_rows: np.ndarray) -> np.ndarray:
    """Gate 規約の Δ_r = LL(q) − LL(π) (改善は負)"""
    e = pi * np.exp(beta * s)
    Z = sg.sum(e)
    return -beta * s[win_rows] + np.log(Z)


# ---------------------------------------------------------------- bootstrap
class DayBoot:
    """年層化 meeting-day bootstrap。日ごとの (和, 件数) から平均を再構成する"""

    def __init__(self, years: np.ndarray, days: np.ndarray, eval_years):
        self.groups = []
        for y in eval_years:
            sel = years == y
            dd, inv = np.unique(days[sel], return_inverse=True)
            self.groups.append((y, np.flatnonzero(sel), inv, len(dd)))

    def ci(self, val: np.ndarray, rng, drop_year=None, boot=BOOT) -> tuple[float, float]:
        sums, cnts = [], []
        for y, rows, inv, nd in self.groups:
            if drop_year is not None and y == drop_year:
                continue
            ds = np.bincount(inv, weights=val[rows], minlength=nd)
            dc = np.bincount(inv, minlength=nd).astype(float)
            pick = rng.integers(0, nd, size=(boot, nd))
            sums.append(ds[pick].sum(1))
            cnts.append(dc[pick].sum(1))
        tot = np.sum(sums, axis=0) / np.sum(cnts, axis=0)
        return float(np.quantile(tot, 0.025)), float(np.quantile(tot, 0.975))


# ---------------------------------------------------------------- tier 2
def empirical_power(pi_all, s_all, sg_all, meta, direction_name, base_name, gate: str,
                    reps: int, jitters, target: float, floor: float = FLOOR,
                    rng_seed: int = RNG_SEED):
    """真の効果 = target を注入したときの等級別 pass 確率 (gate_grade を通す)"""
    year, day = meta["year"], meta["day"]
    delta, real_impr = calibrate_delta(pi_all, s_all, sg_all, target)
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

    out = {}
    for eta in jitters:
        rng = np.random.default_rng(rng_seed)          # 水準ごとに同じ乱数列から始める
        counts = {"PASS-PRACTICAL": 0, "PASS-SIGNAL": 0, "FAIL": 0}
        fail_reasons = {}
        est, warm = [], {y: delta for y in EVAL_YEARS}
        for _ in range(reps):
            win_rows = draw_winners(pi_all, s_all, delta, sg_all, rng)
            pooled, per_year_vals, dvals = [], [], []
            for j in range(SEEDS):
                sj = s_all if (eta == 0 and j > 0) else zscore(
                    s_all + (eta * rng.normal(size=len(s_all)) if eta else 0.0), sg_all)
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
            k = int(np.argsort(pooled)[SEEDS // 2])          # median seed
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
        n_practical = counts["PASS-PRACTICAL"]
        n_detect = counts["PASS-PRACTICAL"] + counts["PASS-SIGNAL"]
        wp, wd = wilson(n_practical, reps), wilson(n_detect, reps)
        out[f"seed_jitter_{eta}"] = {
            "reps": reps, "rng_seed": rng_seed,
            "counts": counts,
            "successes_pass_practical": n_practical,
            "successes_pass_practical_or_signal": n_detect,
            "power_pass_practical": n_practical / reps,
            "power_pass_practical_or_signal": n_detect / reps,
            "wilson95_pass_practical": [wp[0], wp[1]],
            "wilson95_pass_practical_or_signal": [wd[0], wd[1]],
            "fail_reason_counts": fail_reasons,
            "median_point_estimate": float(np.median(est[:, 0])),
            "median_ci95": [float(np.median(est[:, 1])), float(np.median(est[:, 2]))],
            "median_worst_loo_ci_upper": float(np.median(est[:, 3])),
            "share_point_estimate_beyond_floor": float((est[:, 0] < -floor).mean()),
            "attenuation_vs_injected": float(np.median(est[:, 0]) / -target),
        }
    return {"base_market": base_name, "direction": direction_name, "gate": gate,
            "injected_delta": delta, "realized_true_improvement_nats": real_impr,
            "target_nats": target, "practical_floor_used": floor, "n_races": int(n_race),
            "eval_years": EVAL_YEARS, "seeds": SEEDS,
            "placebo_in_simulation": ("満たされたものとして扱う (placebo は合成できないため)"
                                       if gate == "A" else "Gate B の等級条件に placebo は含めない"),
            "by_seed_jitter": out}


def interp_threshold(xs, ys, level=0.80):
    """pass 確率が level に達する最小の真の効果を線形補間で求める。
    最小の試験水準で既に level を超えている場合は『xs[0] 以下』を意味する xs[0] を返す。"""
    if not xs:
        return None
    if ys[0] >= level:
        return xs[0]
    for i in range(1, len(xs)):
        if ys[i] >= level:
            if ys[i] == ys[i - 1]:
                return xs[i]
            w = (level - ys[i - 1]) / (ys[i] - ys[i - 1])
            return xs[i - 1] + w * (xs[i] - xs[i - 1])
    return None


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_races()
    sg = Seg(d["offsets"])
    year, day = d["year"], d["day"]
    n_by_year = {int(y): int((year == y).sum()) for y in sorted(set(year.tolist()))}
    res = {
        "role": "検出力監査。実務床 0.005 nats/race は事前固定で、監査結果によって変更しない",
        "data": {"npz": str(NPZ), "races": int(sg.R), "horses": int(len(d["ban"])),
                 "races_by_year": n_by_year,
                 "note": "2022 以前の正式 race set (障害除外・DNF なし・平地) のみ。2023 は読まない"},
        "grading_implementation": "gate_grade.grade_gate (spec.json の gates と同一実装。境界テストは stage0_checks.py)",
    }

    # ---------------- tier 1: 理想化した局所近似 (楽観側の基準)
    pooled5 = sum(n_by_year[y] for y in EVAL_YEARS)
    res["tier1_ideal_local_approximation"] = {
        "formulas": {"SD(Delta_r)": "sqrt(2*Delta_bar)", "MDE": "2*2.8^2/R = 15.68/R"},
        "assumptions": ["局所指数傾斜 (δ→0)", "方向が最適 (正しい 1 次元スコア)",
                        "モデル仕様が正しい (係数推定の誤差を無視)",
                        "レースが独立 (meeting-day クラスタなし)",
                        "等級条件 (CI と床の比較・seed 判定・年方向・LOO・placebo) を課さない"],
        "status": "理論的な楽観側の基準。本実験に対する厳密式・保証値ではない",
        "SD_at_floor": float(np.sqrt(2 * FLOOR)),
        "races_required_for_floor": float(2 * 2.8 ** 2 / FLOOR),
        "MDE_by_R": {**{str(y): float(2 * 2.8 ** 2 / n_by_year[y]) for y in EVAL_YEARS},
                     "pooled_5y_le2022": float(2 * 2.8 ** 2 / pooled5)},
    }

    # ---------------- 方向を作る
    s_pref, cover = load_features(d["rid16"], d["ban"], sg)
    rng0 = np.random.default_rng(RNG_SEED + 7)
    s_rand = zscore(rng0.normal(size=len(d["ban"])), sg)
    res["directions"] = {
        "q1_log_ratio": {"status": "pending_stage1",
                         "definition": "log(Q1/π)。rolling OOF Q1 が必要なため Stage 0 では測れない",
                         "requirement": "Stage 1 で rolling OOF を作った直後、**結果を開封する前に** "
                                        "この方向で tier2 を再実行する"},
        "prefixed_residual_linear": {"features_and_signs": cover,
                                     "construction": "各特徴をレース内 z 化 → 事前固定符号で等重み和 → 再標準化。"
                                                     "符号 0 の列は合成方向に使わない"},
        "standardized_random": {"construction": "N(0,1) をレース内 z 化。効果量は同じ値へ較正するので、"
                                                "per-race 分散の形だけを変える対照"},
    }

    # ---------------- tier 2: 経験的 cluster power (正式判定)
    runs = {}

    def run(base_name, pi, dname, s, gate, jitters, target, reps):
        key = f"{base_name}__{dname}__true_{target}"
        print(f"[tier2] {key} gate={gate} jitters={jitters} reps={reps} "
              f"({round(time.time()-t0)}s)", flush=True)
        runs[key] = empirical_power(pi, s, sg, d, dname, base_name, gate,
                                    reps=reps, jitters=jitters, target=target)
        print("   ", json.dumps({k: [v["power_pass_practical"],
                                     v["power_pass_practical_or_signal"]]
                                 for k, v in runs[key]["by_seed_jitter"].items()}), flush=True)

    # 判定に使う 2 水準は 400 反復・3 jitter
    for tg in DECISION_TARGETS:
        run("terminal_close_market", d["pi_close"], "prefixed_residual_linear", s_pref, "A",
            JITTERS, tg, REPS_DECISION)
    # power curve (jitter 0.1)
    for tg in CURVE_TARGETS:
        if tg in DECISION_TARGETS:
            continue
        run("terminal_close_market", d["pi_close"], "prefixed_residual_linear", s_pref, "A",
            [0.1], tg, REPS_CURVE)
    # 対照方向と Gate B 基準 (pre)
    run("terminal_close_market", d["pi_close"], "standardized_random", s_rand, "A",
        [0.1], 0.005, REPS_DECISION)
    for tg in (0.005, 0.0069):
        run("historical_pre_snapshot", d["pi_pre"], "prefixed_residual_linear", s_pref, "B",
            [0.1], tg, REPS_DECISION)

    def curve_for(metric, eta="seed_jitter_0.1"):
        xs, ys = [], []
        for tg in CURVE_TARGETS:
            k = f"terminal_close_market__prefixed_residual_linear__true_{tg}"
            if k in runs and eta in runs[k]["by_seed_jitter"]:
                xs.append(tg)
                ys.append(runs[k]["by_seed_jitter"][eta][metric])
        return xs, ys

    xs_p, ys_p = curve_for("power_pass_practical")
    xs_d, ys_d = curve_for("power_pass_practical_or_signal")
    floor_key = "terminal_close_market__prefixed_residual_linear__true_0.005"
    at_floor = runs[floor_key]["by_seed_jitter"]
    worst_detect = min(v["power_pass_practical_or_signal"] for v in at_floor.values())
    worst_detect_wilson_lo = min(v["wilson95_pass_practical_or_signal"][0] for v in at_floor.values())
    proceed = bool(worst_detect >= 0.80 and worst_detect_wilson_lo >= 0.75)
    res["tier2_empirical_cluster_power"] = {
        "runs": runs,
        "progression_rule": {
            "condition": "真の効果 0.005 で PASS-PRACTICAL ∪ PASS-SIGNAL の検出確率 >= 80%",
            "not_required": "PASS-PRACTICAL 単独の 80% は要求しない",
            "wilson_requirement": "推定 power だけでなく Wilson 95% CI 下限が大きく 80% を割っていないことを確認する "
                                   "(本実装では下限 >= 0.75 を確認条件にした)",
            "reps_note": "反復数が少なく 100% になっている場合は反復数を増やして確認する "
                          f"(判定水準は {REPS_DECISION} 反復)",
            "forbidden": "理論 MDE (tier1) だけで進行可否を決めない",
        },
        "verdict": {
            "min_power_detect_at_floor": worst_detect,
            "min_wilson95_lower_at_floor": worst_detect_wilson_lo,
            "power_pass_practical_at_floor": {k: v["power_pass_practical"]
                                              for k, v in at_floor.items()},
            "power_pass_practical_at_0_0069": {
                k: v["power_pass_practical"] for k, v in
                runs["terminal_close_market__prefixed_residual_linear__true_0.0069"]
                ["by_seed_jitter"].items()},
            "result": ("proceed_to_stage1_pending_review" if proceed else "do_not_open_2019_2023"),
        },
        "power_curves": {
            "base": "terminal_close_market", "direction": "prefixed_residual_linear",
            "seed_jitter": 0.1,
            "true_effect_nats": xs_p,
            "power_pass_practical": ys_p,
            "power_pass_practical_or_signal": ys_d,
            "min_true_effect_for_80pct_pass_practical_nats": interp_threshold(xs_p, ys_p),
            "min_true_effect_for_80pct_detect_nats": interp_threshold(xs_d, ys_d),
        },
        "known_optimism": ["方向 s は正しく与えられている (Q1 の推定誤差は seed_jitter でのみ近似)",
                           "真のモデルが厳密に局所指数傾斜である",
                           "Gate A の placebo 条件は満たされたものとして扱っている",
                           "評価年は 2018-2022 で、実判定の 2019-2023 と 1 年ずれる"],
    }

    # ---------------- tier 3: 2022 pre→close の理論 SD と実測 cluster SE
    sel = np.flatnonzero(year == 2022)
    h0, h1 = int(sg.off[sel[0]]), int(sg.off[sel[-1] + 1])
    win_rows = sg.off[sel] + d["winner_pos"][sel] - h0
    lc = -np.log(d["pi_close"][h0:h1][win_rows])
    lp = -np.log(d["pi_pre"][h0:h1][win_rows])
    dd = lp - lc
    rng = np.random.default_rng(RNG_SEED)
    dboot = DayBoot(np.full(len(sel), 2022), day[sel], [2022])
    lo, hi = dboot.ci(dd, rng, boot=4000)
    se_cluster = (hi - lo) / (2 * 1.959964)
    res["tier3_secondary_reference_check"] = {
        "target": "2022 の per-race (LL(pre) − LL(close)) で理論 SD と実測 cluster SE を比較する",
        "n_races": int(len(dd)), "n_meeting_days": int(len(set(day[sel].tolist()))),
        "mean_gap_nats": float(dd.mean()),
        "measured_sd_per_race": float(dd.std(ddof=1)),
        "theoretical_sd_sqrt_2mean": float(np.sqrt(2 * dd.mean())),
        "sd_ratio_measured_over_theoretical": float(dd.std(ddof=1) / np.sqrt(2 * dd.mean())),
        "iid_se": float(dd.std(ddof=1) / np.sqrt(len(dd))),
        "meeting_day_cluster_se": float(se_cluster),
        "cluster_over_iid_se": float(se_cluster / (dd.std(ddof=1) / np.sqrt(len(dd)))),
        "theoretical_iid_se_at_this_mean": float(np.sqrt(2 * dd.mean()) / np.sqrt(len(dd))),
        "interpretation": "pre→close 差は局所傾斜ではなく大きな差 (平均 0.047 nats) なので、"
                          "理論 SD 式と一致する保証はない。一致しても普遍式の証明とは扱わない",
    }

    res["elapsed_sec"] = round(time.time() - t0, 1)
    (OUT / "power_audit.json").write_text(json.dumps(res, ensure_ascii=False, indent=1,
                                                     default=float), encoding="utf-8")
    print(json.dumps(res["tier2_empirical_cluster_power"]["verdict"], ensure_ascii=False))
    print(json.dumps(res["tier2_empirical_cluster_power"]["power_curves"], ensure_ascii=False))
    print(f"[saved] out/power_audit.json  ({res['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
