# -*- coding: utf-8 -*-
"""
stage0_checks.py — EXP19 Stage 0 (v0.3 追補を含む) 成果物の整合検査 (読み取りだけ)
==================================================================================
v0.3-frozen (commit 23e417ec) から変えてよいのは: status、stage0_results (v0.2 実測、superseded 表示の追加)、
stage0_v03_results (新規)、v03_override.B_gates の B2_floor_nats / B2_status (一回限り再監査の結果記入)、
およびFable承認後の非規範注記 gates.superseded_by / feature_contract.WP_superseded_by だけ。
実行: python -m analysis.mcond.exp19_bodyweight_track_condition_dev.stage0_checks
"""
from __future__ import annotations

import json
import subprocess
import sys

from .gate_grade import boundary_tests
from .loaders import BASE, FORBIDDEN_EXACT, FORBIDDEN_PREFIX, HERE, OUT, loader_sha256

FROZEN_V03 = "23e417ec"
REL = "analysis/mcond/exp19_bodyweight_track_condition_dev"
OUTS = ["oof_nobw_manifest.json", "oof_nobw_checks.json", "w_param_fixing.json", "population_coverage.json",
        "stage0_manifest.json", "forward_parity.json", "invariant_tests.json", "power_floor.json",
        "power_floor_v03.json", "compute_dry_run.json", "power_data_meta.json"]
B1_FLOOR = 0.006474307618072295
oks, fails = [], []


def check(n, c, d=""):
    (oks if c else fails).append(n + (f" — {d}" if d else ""))


def load(n):
    return json.loads((OUT / n).read_text(encoding="utf-8"))


def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    for n in OUTS:
        check(f"exists:{n}", (OUT / n).exists())
    check("exists:STAGE0_REPORT.md", (HERE / "STAGE0_REPORT.md").exists())
    if fails:
        return report()
    spec = json.loads((HERE / "spec.json").read_text(encoding="utf-8"))
    frozen = json.loads(subprocess.run(["git", "show", f"{FROZEN_V03}:{REL}/spec.json"], capture_output=True, text=True,
                                       encoding="utf-8", cwd=BASE).stdout)
    allowed = {"status", "stage0_results", "stage0_v03_results", "v03_override"}
    for k in frozen:
        if k not in allowed:
            fv, sv = json.loads(json.dumps(frozen[k])), json.loads(json.dumps(spec.get(k)))
            if k == "gates" and isinstance(sv, dict):
                sv.pop("superseded_by", None)
            if k == "feature_contract" and isinstance(sv, dict):
                sv.pop("WP_superseded_by", None)
            check(f"spec_v03_frozen_unchanged:{k}", fv == sv)
    fo, so = json.loads(json.dumps(frozen["v03_override"])), json.loads(json.dumps(spec["v03_override"]))
    for d in (fo, so):
        d["B_gates"].pop("B2_floor_nats", None)
        d["B_gates"].pop("B2_status", None)
    check("v03_override_unchanged_except_B2_result", fo == so)
    check("spec_no_unexpected_keys", set(spec) - set(frozen) <= allowed, str(set(spec) - set(frozen)))
    check("spec_version_0.3-frozen", spec["version"] == "0.3-frozen")
    check("B1_floor_fixed_value", spec["v03_override"]["B_gates"]["B1_floor_nats"] == B1_FLOOR)
    check("A_economic_floor_null", spec["v03_override"]["A_gates"]["economic_floor"] is None)
    check("loader_forbidden_equals_spec", set(FORBIDDEN_EXACT) == set(spec["loader_contract"]["forbidden_exact"])
          and set(FORBIDDEN_PREFIX) == set(spec["loader_contract"]["forbidden_prefix"]))

    oof, oc = load("oof_nobw_manifest.json"), load("oof_nobw_checks.json")
    check("oof_40_fits_110_features", oc["fits"] == 40 and oof["baseline_contract"]["n_features"] == 110
          and "斤量体重比" not in oof["features"])
    check("oof_reproducible_and_no_future_rows", oc["reproducibility"]["predictions_bitwise_equal"] is True
          and all(v["ok"] for v in oc["future_row_checks"].values()))

    pc, man = load("population_coverage.json"), load("stage0_manifest.json")
    check("known_structure_reproduced", pc["known_structure_reproduction"]["reproduced"] is True)
    check("main_population_15951_diff0", pc["main_population"]["total"] == pc["main_population"]["base_exp16a_total"] == 15951)
    check("wp_joint_floor_each_year", pc["wp_joint_floor_pass"] is True)
    check("loader_sha_current", man["loader_sha256"] == loader_sha256())
    check("manifest_has_v03_wp", man["feature_manifest"].get("WP_columns_v03") == ["wp1_z5_x_cushion_turf", "wp3_z5_x_moistgp_shared"])

    inv = load("invariant_tests.json")
    check("invariant_tests_all_passed", inv["all_passed"], f"{inv['n_pass']}/{inv['n_tests']}")
    check("gate_boundary_tests", not boundary_tests())

    fp = load("forward_parity.json")
    h, s = fp["historical_stage1_entry_v03"], fp["forward_serve_entry_v03"]
    check("forward_gates_split", h["pass"] is False and h["remaining_days"] == 4 - h["meeting_days_observed"]
          and s["blocks_historical_stage1"] is False)
    check("measurement_age_forward_only", "measurement_age_forward_only" in fp)

    v2, v3 = load("power_floor.json"), load("power_floor_v03.json")
    check("v02_power_kept_superseded", abs(v2["gates"]["B1"]["practical_floor_nats"] - B1_FLOOR) < 1e-15
          and v2["gates"]["B2"]["practical_floor_nats"] is not None)
    check("v03_same_seed_grid_boot", v3["seed"] == v2["seed"] and v3["delta_grid"] == v2["delta_grid"]
          and v3["boot"] == v2["boot"] and v3["reps_growth"] == v2["reps_growth"] and v3["fit_start"] == v2["fit_start"])
    check("v03_same_fit_windows", all(v3["gates"][g]["n_fit"] == v2["gates"][g]["n_fit"] for g in ("A2", "B2")))
    check("v03_wp_two_columns", v3["wp_columns"] == ["wp1_z5_x_cushion_turf", "wp3_z5_x_moistgp_shared"])
    check("v03_A2_no_economic_floor", v3["gates"]["A2"]["practical_floor_nats"] is None)
    b2 = v3["gates"]["B2"]
    passed = bool(b2["practical_floor_nats"] is not None and b2.get("progression_power_pass"))
    res = spec.get("stage0_v03_results", {})
    check("v03_decision_recorded", res.get("wp_route") == ("open" if passed else "closed"))
    check("v03_spec_B2_floor_consistent", spec["v03_override"]["B_gates"]["B2_floor_nats"] ==
          (b2["practical_floor_nats"] if passed else None))
    check("stage1_not_started", res.get("stage1_started") is False and spec["v03_override"]["outcomes_opened"] is False)
    check("compute_measured", load("compute_dry_run.json").get("peak_rss_mb", 0) > 0)
    return report()


def report():
    print(f"OK   {len(oks)}")
    for f in fails:
        print("FAIL " + f)
    print("ALL CHECKS PASSED" if not fails else f"{len(fails)} CHECK(S) FAILED")
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
