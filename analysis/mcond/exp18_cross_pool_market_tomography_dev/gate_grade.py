# -*- coding: utf-8 -*-
"""
gate_grade.py — EXP18 Gate M1 を 3 等級で判定する唯一の実装 (v0.5)
====================================================================
検出力 simulation (floor_v05.py)・Stage 1 本判定 (evaluate_stage1.py)・境界テストが**同じ関数**を通る。

符号: Δ = LL(q_cross) − LL(q_temp)。改善は負。Gate arm は UB2 の 1 本 (Bonferroni 係数 1 → CI95)。

  PASS-SIGNAL    : CI95 上限 < 0、かつ 5 年中 4 年以上で改善方向、5 seed 中 4 以上で改善方向、
                   leave-one-year-out の全結果で CI95 上限 < 0、P1 と P2 の両 placebo を改善側 97.5 percentile で超過
  PASS-PRACTICAL : PASS-SIGNAL に加えて CI95 上限 < −practical_floor_nats
  FAIL           : SIGNAL 条件のいずれかが不成立

`CI95 上限 < −floor` は「真の効果が床なら高確率で通る検出条件」ではない。実務床を 95% 信頼水準で
上回ったと主張するための厳格な等級である (floor ちょうどの効果に対する PASS-PRACTICAL power は低く、
floor_v05.py が別途計算して報告する)。Stage 1 の進行可否は SIGNAL 検出力で決める。
点推定は等級条件に使わない (報告値)。
"""
from __future__ import annotations

from math import sqrt

MIN_YEARS = 4
MIN_SEEDS = 4
N_GATE_ARMS = 1

STATEMENTS = {
    "PASS-PRACTICAL": "較正済み terminal 馬連市場に対し、単勝由来 T1 の offset 残差情報が検出され、かつ凍結した"
                      "実務床を 95% 信頼水準で上回った",
    "PASS-SIGNAL": "残差情報は検出されたが、凍結した長期成長基準を超える証拠なし",
    "FAIL": "較正済み terminal 馬連市場に対する単勝由来 T1 の offset 残差情報を、2019-2023 rolling 評価で"
            "検出できなかった。全券種・全プール・JRA 全市場へ一般化しない",
}
CONSEQUENCES = {
    "PASS-PRACTICAL": ["仕様で許可済みの economic checks へ進んでよい",
                       "ROI 主張・候補生成・production 変更はしない", "2024/2025 は開かない"],
    "PASS-SIGNAL": ["EXP18 を終了する", "economic checks・ROI・候補生成へ進まない", "2024/2025 は開かない"],
    "FAIL": ["EXP18 を終了する", "結果後の救済探索 (券種・人気帯・de-vig・T2 復活) をしない"],
}


def wilson(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def ci_level(n_arms: int = N_GATE_ARMS) -> float:
    """Bonferroni 補正後の両側 CI 水準"""
    return 1.0 - 0.05 / n_arms


def grade_m1(*, point: float, ci_lower: float, ci_upper: float, years_improved: int, n_years: int,
             seeds_improved: int, n_seeds: int, loo_ci_uppers, p1_exceeded: bool, p2_exceeded: bool,
             floor: float, min_years: int = MIN_YEARS, min_seeds: int = MIN_SEEDS) -> dict:
    if floor is None or not (floor > 0):
        raise ValueError("practical_floor_nats は正の数値でなければならない")
    if p1_exceeded is None or p2_exceeded is None:
        raise ValueError("P1・P2 の判定は必須")
    fails = []
    if not ci_upper < 0:
        fails.append("CI95 上限 >= 0")
    if not years_improved >= min_years:
        fails.append(f"year_direction {years_improved}/{n_years} < {min_years}")
    if not seeds_improved >= min_seeds:
        fails.append(f"seed_direction {seeds_improved}/{n_seeds} < {min_seeds}")
    loo = list(loo_ci_uppers)
    if not loo or not all(u < 0 for u in loo):
        fails.append("leave_one_year_out CI95 上限 < 0 を満たさない結果がある")
    if not p1_exceeded:
        fails.append("P1 placebo を改善側 97.5 percentile で超えない")
    if not p2_exceeded:
        fails.append("P2 placebo を改善側 97.5 percentile で超えない")
    if fails:
        g = "FAIL"
    elif ci_upper < -floor:
        g = "PASS-PRACTICAL"
    else:
        g = "PASS-SIGNAL"
    return {"grade": g, "statement": STATEMENTS[g], "consequences": CONSEQUENCES[g], "fail_reasons": fails,
            "reported": {"pooled_point_estimate": point, "ci95": [ci_lower, ci_upper],
                         "ci_upper_below_zero": bool(ci_upper < 0),
                         "ci_upper_below_minus_floor": bool(ci_upper < -floor),
                         "point_estimate_beyond_floor": bool(point < -floor),
                         "years_improved": f"{years_improved}/{n_years}",
                         "seeds_improved": f"{seeds_improved}/{n_seeds}",
                         "loo_ci_uppers": loo, "p1_exceeded": p1_exceeded, "p2_exceeded": p2_exceeded,
                         "floor": floor, "n_gate_arms": N_GATE_ARMS},
            "note": "点推定は報告値であり等級条件ではない"}


def placebo_exceeded(real_delta: float, placebo_deltas) -> bool:
    """real Δ が placebo 分布の改善側 97.5 percentile (= 2.5% 分位) より改善側にあるか"""
    import numpy as np
    x = np.asarray(list(placebo_deltas), float)
    if len(x) < 200:
        raise ValueError("placebo は最低 200 draw")
    return bool(real_delta < np.quantile(x, 0.025))


def boundary_cases(floor: float = 0.01) -> list[dict]:
    ok = dict(years_improved=5, n_years=5, seeds_improved=5, n_seeds=5,
              loo_ci_uppers=[-0.004] * 5, p1_exceeded=True, p2_exceeded=True, floor=floor)
    f = floor
    return [
        dict(name="ci_crosses_zero", point=-0.02, ci_lower=-0.03, ci_upper=0.001, expect="FAIL", **ok),
        dict(name="ci_upper_exactly_zero", point=-0.02, ci_lower=-0.03, ci_upper=0.0, expect="FAIL", **ok),
        dict(name="signal_ci_crosses_minus_floor", point=-0.9 * f, ci_lower=-1.5 * f, ci_upper=-0.2 * f,
             expect="PASS-SIGNAL", **ok),
        dict(name="point_beyond_floor_ci_not", point=-1.3 * f, ci_lower=-2.0 * f, ci_upper=-0.5 * f,
             expect="PASS-SIGNAL", **ok),
        dict(name="ci_upper_exactly_minus_floor", point=-2 * f, ci_lower=-3 * f, ci_upper=-f,
             expect="PASS-SIGNAL", **ok),
        dict(name="ci_entirely_beyond_floor", point=-2 * f, ci_lower=-3 * f, ci_upper=-1.01 * f,
             expect="PASS-PRACTICAL", **ok),
        dict(name="year_direction_3_of_5", point=-2 * f, ci_lower=-3 * f, ci_upper=-1.5 * f, expect="FAIL",
             **{**ok, "years_improved": 3}),
        dict(name="year_direction_4_of_5", point=-2 * f, ci_lower=-3 * f, ci_upper=-1.5 * f,
             expect="PASS-PRACTICAL", **{**ok, "years_improved": 4}),
        dict(name="seed_direction_fail", point=-2 * f, ci_lower=-3 * f, ci_upper=-1.5 * f, expect="FAIL",
             **{**ok, "seeds_improved": 3}),
        dict(name="loo_one_year_fails", point=-2 * f, ci_lower=-3 * f, ci_upper=-1.5 * f, expect="FAIL",
             **{**ok, "loo_ci_uppers": [-0.004, -0.004, 0.0001, -0.004, -0.004]}),
        dict(name="p1_fail", point=-2 * f, ci_lower=-3 * f, ci_upper=-1.5 * f, expect="FAIL",
             **{**ok, "p1_exceeded": False}),
        dict(name="p2_fail", point=-2 * f, ci_lower=-3 * f, ci_upper=-1.5 * f, expect="FAIL",
             **{**ok, "p2_exceeded": False}),
    ]


def run_boundary_tests(floor: float = 0.01) -> list[str]:
    bad = []
    for c in boundary_cases(floor):
        c = dict(c)
        name, expect = c.pop("name"), c.pop("expect")
        got = grade_m1(**c)["grade"]
        if got != expect:
            bad.append(f"{name}: expected {expect} got {got}")
    for kw in ({"floor": None}, {"floor": 0.0}, {"p1_exceeded": None}):
        try:
            base = dict(point=-1, ci_lower=-1, ci_upper=-1, years_improved=5, n_years=5, seeds_improved=5,
                        n_seeds=5, loo_ci_uppers=[-1] * 5, p1_exceeded=True, p2_exceeded=True, floor=0.01)
            base.update(kw)
            grade_m1(**base)
            bad.append(f"invalid input accepted: {kw}")
        except ValueError:
            pass
    import numpy as np
    pl = np.linspace(-0.001, 0.001, 200)
    if placebo_exceeded(-0.0009, pl) or not placebo_exceeded(-0.0011, pl):
        bad.append("placebo_exceeded の境界")
    try:
        placebo_exceeded(-1.0, pl[:199])
        bad.append("placebo < 200 draw accepted")
    except ValueError:
        pass
    if abs(ci_level(1) - 0.95) > 1e-15:
        bad.append("ci_level(1) != 0.95")
    return bad


if __name__ == "__main__":
    b = run_boundary_tests()
    print("boundary tests:", "ALL PASSED" if not b else b)
