# -*- coding: utf-8 -*-
"""
gate_grade.py — EXP16A の Gate A / Gate B を 3 等級で判定する唯一の実装
=======================================================================
spec.json の `gates` と同じ規則をコードで一本化する。検出力 simulation
(power_audit.py) と本番判定 (Stage 1) と境界テスト (stage0_checks.py) が
**同じ関数**を通るようにするため、判定ロジックをここだけに置く。

等級 (2019-2023 pooled / median seed / year-stratified meeting-day bootstrap):

  PASS-PRACTICAL : CI95 上限 < -floor かつ 安定性条件すべて
  PASS-SIGNAL    : CI95 上限 < 0 (ただし < -floor ではない) かつ 安定性条件すべて
  FAIL           : CI95 上限 >= 0、または安定性条件のいずれかが不成立

安定性条件:
  * 5 年中 4 年以上で改善方向
  * 5 seed 中 4 以上で改善方向
  * leave-one-year-out の **全結果** で CI95 上限 < 0
  * (Gate A のみ) placebo 97.5 パーセンタイルを超える

**点推定は等級条件に使わない**。点推定が -0.005 を下回っても CI95 上限が
-0.005 を下回らなければ実務床超えは未証明なので PASS-SIGNAL である。
点推定は報告値として残す。
"""
from __future__ import annotations

from math import sqrt

FLOOR = 0.005          # 事前固定した実務床 (nats/race)。検出力監査の結果で変更しない
MIN_YEARS = 4          # 5 年中
MIN_SEEDS = 4          # 5 seed 中

STATEMENTS = {
    "A": {
        "PASS-PRACTICAL": "締切市場に対する残差情報が存在し、かつ 0.005 nats/race の実務床を超える証拠が得られた",
        "PASS-SIGNAL": "締切市場に対する残差情報は検出されたが、0.005 nats/race の実務床を超える証拠は得られなかった",
        "FAIL": "単勝close市場に対するR0-cleanの残差情報を2019-2023 crossfitで検出できなかった。全券種・JRA全研究・馬券依存構造へ一般化しない",
    },
    "B": {
        "PASS-PRACTICAL": "historical_pre_snapshot 時点でも残差情報が再現され、かつ 0.005 nats/race の実務床を超える証拠が得られた",
        "PASS-SIGNAL": "historical_pre_snapshot 時点で残差情報は再現されたが、0.005 nats/race の実務床を超える証拠は得られなかった",
        "FAIL": "締切市場に対する残差情報は検出されたが、historical_pre_snapshot時点で同等の実用可能な改善を再現できなかった",
    },
}

CONSEQUENCES = {
    "A": {
        "PASS-PRACTICAL": ["Gate B へ進む", "auxiliary economic checks を実施してよい",
                            "候補生成・配分最適化・production 接続にはまだ進まない"],
        "PASS-SIGNAL": ["Gate B は情報の時点再現性を確認する科学的診断として実施してよい",
                         "auxiliary economic checks は実施しない",
                         "候補生成・ROI 評価・配分最適化へ進まない"],
        "FAIL": ["scoped fail statement を使用して終了する", "結果後の救済探索をしない"],
    },
    "B": {
        "PASS-PRACTICAL": ["Gate A も PASS-PRACTICAL の場合にかぎり economic checks を実施してよい",
                            "Gate A が PASS-SIGNAL なら economic checks へ自動進行しない"],
        "PASS-SIGNAL": ["economic checks は実施しない", "候補生成・ROI 評価へ進まない"],
        "FAIL": ["scoped fail statement を使用する", "『市場に吸収された』と断定しない"],
    },
}


def wilson(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """二項比率の Wilson 95% 信頼区間"""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def grade_gate(gate: str, *, point: float, ci_lower: float, ci_upper: float,
               years_improved: int, n_years: int, seeds_improved: int, n_seeds: int,
               loo_ci_uppers, placebo_exceeded=None, floor: float = FLOOR,
               min_years: int = MIN_YEARS, min_seeds: int = MIN_SEEDS) -> dict:
    """Gate A / B の等級を返す。Δ は LL(arm) − LL(baseline) で改善は負。"""
    assert gate in ("A", "B"), gate
    fails = []
    if not (years_improved >= min_years):
        fails.append(f"year_direction {years_improved}/{n_years} < {min_years}")
    if not (seeds_improved >= min_seeds):
        fails.append(f"seed_direction {seeds_improved}/{n_seeds} < {min_seeds}")
    loo = list(loo_ci_uppers)
    if not loo or not all(u < 0 for u in loo):
        fails.append("leave_one_year_out CI95 上限 < 0 を満たさない結果がある")
    if gate == "A":
        if placebo_exceeded is None:
            raise ValueError("Gate A は placebo 判定が必須")
        if not placebo_exceeded:
            fails.append("placebo 97.5 パーセンタイルを超えない")
    detected = ci_upper < 0
    if not detected:
        fails.append("CI95 上限 >= 0")
    if fails:
        g = "FAIL"
    elif ci_upper < -floor:
        g = "PASS-PRACTICAL"
    else:
        g = "PASS-SIGNAL"
    return {
        "gate": gate, "grade": g, "statement": STATEMENTS[gate][g],
        "consequences": CONSEQUENCES[gate][g],
        "fail_reasons": fails,
        "reported": {
            "pooled_point_estimate": point, "ci95": [ci_lower, ci_upper],
            "point_estimate_beyond_floor": bool(point < -floor),
            "ci_entirely_beyond_floor": bool(ci_upper < -floor),
            "ci_upper_below_zero": bool(detected),
            "years_improved": f"{years_improved}/{n_years}",
            "seeds_improved": f"{seeds_improved}/{n_seeds}",
            "loo_ci_uppers": loo,
            "placebo_exceeded": placebo_exceeded,
            "floor": floor,
        },
        "note": "点推定が床を超えたかは報告値であり、等級条件ではない",
    }


def economic_checks_allowed(grade_a: str, grade_b: str) -> bool:
    """economic checks は Gate A と Gate B がともに PASS-PRACTICAL のときだけ"""
    return grade_a == "PASS-PRACTICAL" and grade_b == "PASS-PRACTICAL"


def boundary_cases() -> list[dict]:
    """合成境界テスト。stage0_checks.py から実行する"""
    ok = dict(years_improved=5, n_years=5, seeds_improved=5, n_seeds=5,
              loo_ci_uppers=[-0.004, -0.004, -0.004, -0.004, -0.004], placebo_exceeded=True)
    cases = [
        # CI が 0 を跨ぐ
        dict(name="ci_crosses_zero", gate="A", point=-0.006, ci_lower=-0.012, ci_upper=0.001,
             expect="FAIL", **ok),
        dict(name="ci_upper_exactly_zero", gate="A", point=-0.006, ci_lower=-0.012, ci_upper=0.0,
             expect="FAIL", **ok),
        # CI は 0 未満だが -0.005 を跨ぐ
        dict(name="ci_below_zero_crosses_floor", gate="A", point=-0.0049, ci_lower=-0.0090,
             ci_upper=-0.0008, expect="PASS-SIGNAL", **ok),
        dict(name="point_beyond_floor_but_ci_not", gate="A", point=-0.0061, ci_lower=-0.0105,
             ci_upper=-0.0018, expect="PASS-SIGNAL", **ok),
        dict(name="ci_upper_exactly_minus_floor", gate="A", point=-0.0080, ci_lower=-0.0110,
             ci_upper=-0.005, expect="PASS-SIGNAL", **ok),
        # CI 全体が -0.005 未満
        dict(name="ci_entirely_beyond_floor", gate="A", point=-0.0090, ci_lower=-0.0130,
             ci_upper=-0.0051, expect="PASS-PRACTICAL", **ok),
        # 安定性条件の個別失敗
        dict(name="year_direction_fail", gate="A", point=-0.0090, ci_lower=-0.0130, ci_upper=-0.0051,
             expect="FAIL", **{**ok, "years_improved": 3}),
        dict(name="seed_direction_fail", gate="A", point=-0.0090, ci_lower=-0.0130, ci_upper=-0.0051,
             expect="FAIL", **{**ok, "seeds_improved": 3}),
        dict(name="loo_fail_one_year", gate="A", point=-0.0090, ci_lower=-0.0130, ci_upper=-0.0051,
             expect="FAIL",
             **{**ok, "loo_ci_uppers": [-0.004, -0.004, 0.0005, -0.004, -0.004]}),
        dict(name="placebo_fail", gate="A", point=-0.0090, ci_lower=-0.0130, ci_upper=-0.0051,
             expect="FAIL", **{**ok, "placebo_exceeded": False}),
        # Gate B は placebo を等級条件にしない
        dict(name="gateB_no_placebo_needed", gate="B", point=-0.0090, ci_lower=-0.0130,
             ci_upper=-0.0051, expect="PASS-PRACTICAL",
             **{**ok, "placebo_exceeded": None}),
        dict(name="gateB_signal", gate="B", point=-0.0049, ci_lower=-0.0090, ci_upper=-0.0008,
             expect="PASS-SIGNAL", **{**ok, "placebo_exceeded": None}),
        dict(name="gateB_fail_ci", gate="B", point=-0.0049, ci_lower=-0.0090, ci_upper=0.0002,
             expect="FAIL", **{**ok, "placebo_exceeded": None}),
    ]
    return cases


def run_boundary_tests() -> list[str]:
    """失敗した境界テストの説明を返す (空なら全通過)"""
    bad = []
    for c in boundary_cases():
        c = dict(c)
        name, expect, gate = c.pop("name"), c.pop("expect"), c.pop("gate")
        got = grade_gate(gate, **c)["grade"]
        if got != expect:
            bad.append(f"{name}: expected {expect} got {got}")
    # economic checks の組合せ
    combos = {("PASS-PRACTICAL", "PASS-PRACTICAL"): True,
              ("PASS-PRACTICAL", "PASS-SIGNAL"): False,
              ("PASS-SIGNAL", "PASS-PRACTICAL"): False,
              ("PASS-SIGNAL", "PASS-SIGNAL"): False,
              ("PASS-PRACTICAL", "FAIL"): False, ("FAIL", "PASS-PRACTICAL"): False}
    for (a, b), want in combos.items():
        if economic_checks_allowed(a, b) != want:
            bad.append(f"economic_checks_allowed({a},{b}) != {want}")
    return bad


if __name__ == "__main__":
    bad = run_boundary_tests()
    print("boundary tests:", "ALL PASSED" if not bad else bad)
