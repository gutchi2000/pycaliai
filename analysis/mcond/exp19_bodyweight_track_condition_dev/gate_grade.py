# -*- coding: utf-8 -*-
"""
gate_grade.py — EXP19 Gate A1/A2/B1/B2 を 3 等級で判定する唯一の実装 (SPEC §7)
==============================================================================
符号: Δ = LL(alt) − LL(null)、改善は負。等級:
  PASS_PRACTICAL : SIGNAL 条件すべて + 上限 < −floor
  PASS_SUBFLOOR  : SIGNAL 条件すべてだが floor 未達
  FAIL           : SIGNAL 条件のいずれかが不成立
SIGNAL 条件: (多重補正後) 上限 < 0、年方向 (A1/B1 4/5、A2/B2 3/3)、LOO 全通り上限 < 0、対応 placebo すべて超過
Holm (A1/A2): 片側 bootstrap p 値 p0 = P*(Δ* >= 0)、pf = P*(Δ* >= −floor) を A1/A2 で Holm 補正し、
  補正後 p < 0.025 を「上限 < 0」「上限 < −floor」とみなす (1 本だけの検定なら percentile CI95 上限と同値)。
  B1/B2 は Holm の対象外で、percentile CI95 上限を直接使う。
placebo: real Δ < quantile(placebo Δ, 0.025)、最低 200 draw。
"""
from __future__ import annotations

import numpy as np

ALPHA_ONE_SIDED = 0.025


def boot_p(boot: np.ndarray, threshold: float) -> float:
    """改善方向 (負) の片側 p 値: bootstrap 平均が threshold 以上になる割合"""
    return float(np.mean(np.asarray(boot) >= threshold))


def holm(pvals: dict, alpha: float = ALPHA_ONE_SIDED) -> dict:
    """Holm step-down。戻り: 各検定の reject (bool)"""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, still = {}, True
    for k, (name, p) in enumerate(items):
        ok = still and p < alpha / (m - k)
        out[name] = bool(ok)
        still = ok
    return out


def placebo_exceeded(real: float, draws) -> bool:
    x = np.asarray(list(draws), float)
    if len(x) < 200:
        raise ValueError("placebo は最低 200 draw")
    return bool(real < np.quantile(x, 0.025))


def grade(*, gate: str, point: float, signal_ok: bool, practical_ok: bool, years_improved: int, n_years: int,
          min_years: int, loo_ci_uppers, placebo: dict, floor: float) -> dict:
    """signal_ok / practical_ok は Holm 補正後 (A) または CI95 上限 (B) の判定済み bool"""
    if floor is None or not floor > 0:
        raise ValueError("floor は正の数値")
    if any(v is None for v in placebo.values()):
        raise ValueError("placebo 判定が欠けている")
    fails = []
    if not signal_ok:
        fails.append("上限 < 0 (多重補正後) を満たさない")
    if years_improved < min_years:
        fails.append(f"year_direction {years_improved}/{n_years} < {min_years}")
    loo = list(loo_ci_uppers)
    if len(loo) != n_years or not all(u < 0 for u in loo):
        fails.append("LOO 全通りで上限 < 0 を満たさない")
    for k, v in placebo.items():
        if not v:
            fails.append(f"{k} placebo 未超過")
    g = "FAIL" if fails else ("PASS_PRACTICAL" if practical_ok else "PASS_SUBFLOOR")
    return {"gate": gate, "grade": g, "fail_reasons": fails, "point": point, "floor": floor,
            "years_improved": f"{years_improved}/{n_years}", "loo_ci_uppers": loo, "placebo": placebo}


def signal_practical_from_boot(boots: dict, floor: dict, holm_gates: list) -> dict:
    """boots: gate → bootstrap 平均の配列。Holm 対象 gate は Holm、それ以外は CI95 上限"""
    out = {}
    hp0 = {g: boot_p(boots[g], 0.0) for g in holm_gates if g in boots}
    hpf = {g: boot_p(boots[g], -floor[g]) for g in holm_gates if g in boots}
    r0, rf = (holm(hp0) if hp0 else {}), (holm(hpf) if hpf else {})
    for g, b in boots.items():
        if g in holm_gates:
            out[g] = {"signal_ok": r0[g], "practical_ok": rf[g], "p0": hp0[g], "pf": hpf[g], "holm": True}
        else:
            up = float(np.quantile(b, 0.975))
            out[g] = {"signal_ok": up < 0, "practical_ok": up < -floor[g], "ci95_upper": up, "holm": False}
    return out


def boundary_tests() -> list[str]:
    bad = []
    ok = dict(years_improved=5, n_years=5, min_years=4, loo_ci_uppers=[-1e-3] * 5,
              placebo={"P1": True, "P2": True}, floor=0.01, point=-0.02)
    cases = [
        (dict(signal_ok=True, practical_ok=True), "PASS_PRACTICAL"),
        (dict(signal_ok=True, practical_ok=False), "PASS_SUBFLOOR"),
        (dict(signal_ok=False, practical_ok=False), "FAIL"),
        (dict(signal_ok=True, practical_ok=True, years_improved=3), "FAIL"),
        (dict(signal_ok=True, practical_ok=True, years_improved=4), "PASS_PRACTICAL"),
        (dict(signal_ok=True, practical_ok=True, loo_ci_uppers=[-1e-3, -1e-3, 0.0, -1e-3, -1e-3]), "FAIL"),
        (dict(signal_ok=True, practical_ok=True, placebo={"P1": True, "P2": False}), "FAIL"),
    ]
    for kw, want in cases:
        got = grade(gate="A1", **{**ok, **kw})["grade"]
        if got != want:
            bad.append(f"{kw}: {got} != {want}")
    a2 = dict(years_improved=2, n_years=3, min_years=3, loo_ci_uppers=[-1e-3] * 3,
              placebo={"P1": True, "P2": True, "P3": True}, floor=0.01, point=-0.02, signal_ok=True, practical_ok=True)
    if grade(gate="A2", **a2)["grade"] != "FAIL":
        bad.append("A2 は 3/3 年必須")
    # Holm: 2 本とも p=0.02 → 小さい方も 0.0125 を超えるので両方不採択。0.01 と 0.02 なら両方採択
    if holm({"A1": 0.02, "A2": 0.02}) != {"A1": False, "A2": False}:
        bad.append("holm 0.02/0.02")
    if holm({"A1": 0.01, "A2": 0.02}) != {"A1": True, "A2": True}:
        bad.append("holm 0.01/0.02")
    if holm({"A1": 0.01, "A2": 0.03}) != {"A1": True, "A2": False}:
        bad.append("holm 0.01/0.03")
    pl = np.linspace(-1, 1, 200)
    q = np.quantile(pl, 0.025)
    if placebo_exceeded(q, pl) or not placebo_exceeded(q - 1e-9, pl):
        bad.append("placebo 境界 (等号は未超過)")
    try:
        placebo_exceeded(-5, pl[:199])
        bad.append("placebo < 200 を受理")
    except ValueError:
        pass
    # Holm なし (B): CI95 上限
    b = np.linspace(-0.03, -0.001, 10001)
    r = signal_practical_from_boot({"B1": b}, {"B1": 0.01}, holm_gates=[])
    if not (r["B1"]["signal_ok"] and not r["B1"]["practical_ok"]):
        bad.append("B1 CI 判定")
    return bad
