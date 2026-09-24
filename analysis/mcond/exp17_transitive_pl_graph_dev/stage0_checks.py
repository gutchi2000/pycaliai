# -*- coding: utf-8 -*-
"""
stage0_checks.py — EXP17 Stage 0 成果物間の整合検査 (母集団・期間・Gate・停止条件・数値)
実行: python -m analysis.mcond.exp17_transitive_pl_graph_dev.stage0_checks
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    checks = []

    def ok(name, cond, detail=""):
        checks.append((name, bool(cond), detail))

    req = ["PRIOR_ART_EQUIVALENCE_AUDIT.md", "GRAPH_DATA_AUDIT.md", "POWER_AUDIT.md", "COMPUTE_DRY_RUN.json", "SPEC.md",
           "spec.json", "README.md", "graph_core.py", "test_invariants.py", "coverage_audit.py", "equivalence_audit.py", "power_audit.py"]
    for f in req:
        ok(f"exists:{f}", (HERE / f).exists())
    for f in ["GRAPH_COVERAGE.json", "race_population.json", "invariant_tests.json", "equivalence_audit.json", "power_audit.json", "compute_timings.json"]:
        ok(f"exists:out/{f}", (OUT / f).exists())

    spec = json.loads((HERE / "spec.json").read_text(encoding="utf-8"))
    SPEC = (HERE / "SPEC.md").read_text(encoding="utf-8")
    README = (HERE / "README.md").read_text(encoding="utf-8")
    PA = (HERE / "PRIOR_ART_EQUIVALENCE_AUDIT.md").read_text(encoding="utf-8")
    GD = (HERE / "GRAPH_DATA_AUDIT.md").read_text(encoding="utf-8")
    PW = (HERE / "POWER_AUDIT.md").read_text(encoding="utf-8")
    cov = json.loads((OUT / "GRAPH_COVERAGE.json").read_text(encoding="utf-8"))
    pop = json.loads((OUT / "race_population.json").read_text(encoding="utf-8"))
    inv = json.loads((OUT / "invariant_tests.json").read_text(encoding="utf-8"))
    eq = json.loads((OUT / "equivalence_audit.json").read_text(encoding="utf-8"))
    pw = json.loads((OUT / "power_audit.json").read_text(encoding="utf-8"))
    cd = json.loads((HERE / "COMPUTE_DRY_RUN.json").read_text(encoding="utf-8"))

    # version / status
    ok("spec.json version 0.2", spec["version"].startswith("0.2"))
    ok("SPEC.md version 0.2", "**版:** 0.2" in SPEC)
    ok("spec.json status terminated", spec["status"] == "terminated_at_stage0")
    for doc, name in ((SPEC, "SPEC"), (README, "README")):
        ok(f"{name} says Stage 0 終了", "Stage 0" in doc and "終了" in doc)

    # population rules consistent
    for token in ["51", "59", "5頭以上", "同着", "DNF", "血統登録番号"]:
        ok(f"SPEC population token:{token}", token in SPEC)
    ok("spec.json horse_key", spec["population"]["horse_key"].startswith("血統登録番号"))
    ok("spec.json horse_name_join forbidden", spec["population"]["horse_name_join"] == "forbidden")
    ok("official races 15951 in race_population", pop["official_races"] == 15951)
    ok("official races 15951 in spec.json", spec["stage0_results"]["official_race_set_2019_2023"] == 15951)
    ok("official races 15951 in GRAPH_DATA_AUDIT", "15,951" in GD)
    ok("official races 15951 in SPEC", "15,951" in SPEC)
    ok("by_year matches EXP16A", pop["official_by_year"] == {"2019": 3185, "2020": 3169, "2021": 3200, "2022": 3206, "2023": 3191})

    # periods
    ok("development 2019-2023", spec["rolling_protocol"]["development"] == [2019, 2020, 2021, 2022, 2023])
    ok("sealed 2024/2025", spec["rolling_protocol"]["sealed"] == [2024, 2025])
    ok("SPEC sealed", "2024/2025" in SPEC and "封印" in SPEC)

    # floors consistency
    pf = spec["stage0"]["provisional_coverage_floors_for_review"]
    cf = cov["provisional_floors_from_spec"]
    ok("coverage floors spec==json", all(abs(pf[k] - cf[k]) < 1e-12 for k in pf))
    ok("coverage floor result recorded", spec["stage0_results"]["coverage_floor_check"]["all_pass"] is False and cov["floor_check"]["all_pass"] is False)
    ok("mechanism floor 0.001 spec==power", abs(spec["gates"]["G1_mechanism"]["mechanism_floor_nats_per_pair"] - pw["mechanism_floor_nats_per_pair"]) < 1e-12)
    ok("practical floor 0.005", abs(spec["gates"]["G2_terminal_close_residual"]["practical_floor_nats"] - 0.005) < 1e-12 and "0.005 nats/race" in SPEC)
    ok("POWER_AUDIT floor text", "0.001 nats/pair" in PW)

    # equivalence verdict consistency
    ok("equivalence fail flag", eq["E0_B_excess_variance_vs_bt_null"]["fail"] is True)
    ok("rho recorded in spec", abs(spec["stage0_results"]["E0_B_rho"] - eq["E0_B_excess_variance_vs_bt_null"]["rho_real_over_null"]) < 1e-9)
    ok("PRIOR_ART says E0 FAIL", "E0 FAIL" in PA)
    ok("SPEC says E0 FAIL", "E0 FAIL" in SPEC)
    ok("fail statement identical spec/SPEC/README", spec["fail_statement"] in SPEC and spec["fail_statement"] in README and spec["fail_statement"] in PA)

    # invariants / power
    ok("invariants all_passed", inv["all_passed"] is True and inv["n_checks"] == 17)
    ok("power at floor >= 0.8", pw["power_by_true_effect"]["0.001"]["power_detect"] >= 0.8)
    ok("power reps 400 & seed", pw["reps"] == 400 and pw["seed"] == 20260925)
    ok("placebo draws >= 200", spec["placebos"]["draws_minimum"] >= 200)
    ok("inference unit calendar day", "開催日（暦日）" in SPEC and spec["stage0"]["power"]["inference_unit"] == "calendar_meeting_day")

    # stop rules
    ok("stop rules triggered recorded", spec["stage0_results"]["stop_rules_triggered"] == [1, 2])
    ok("SPEC stop rules 1,2", "該当したもの: 1, 2" in SPEC)

    # prohibitions unchanged
    for p in ["2024/2025 outcome evaluation", "ticket candidate generation", "stake optimization", "production integration"]:
        ok(f"prohibition:{p}", p in spec["prohibitions"])
    # compute dry run present
    ok("compute dry run has extrapolation", "stage1_extrapolation_if_it_had_proceeded" in cd)

    n_ok = sum(1 for _, c, _ in checks if c)
    for name, c, detail in checks:
        if not c:
            print("FAIL", name, detail)
    print(f"OK   {n_ok}/{len(checks)}")
    print("ALL CHECKS PASSED" if n_ok == len(checks) else "SOME CHECKS FAILED")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
