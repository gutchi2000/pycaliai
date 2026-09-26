# -*- coding: utf-8 -*-
"""
stage0_checks.py — EXP19 Stage 0 成果物の整合検査 (読み取りだけ)
==============================================================
実行: python -m analysis.mcond.exp19_bodyweight_track_condition_dev.stage0_checks
"""
from __future__ import annotations

import json
import subprocess
import sys

from .gate_grade import boundary_tests
from .loaders import BASE, HERE, OUT, loader_sha256

FROZEN = "4a4c4903"
REL = "analysis/mcond/exp19_bodyweight_track_condition_dev"
OUTS = ["oof_nobw_manifest.json", "oof_nobw_checks.json", "w_param_fixing.json", "population_coverage.json",
        "stage0_manifest.json", "forward_parity.json", "invariant_tests.json", "power_floor.json",
        "compute_dry_run.json", "power_data_meta.json"]
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
    frozen = json.loads(subprocess.run(["git", "show", f"{FROZEN}:{REL}/spec.json"], capture_output=True, text=True,
                                       encoding="utf-8", cwd=BASE).stdout)
    allowed = {"status", "stage0_results"}
    for k in frozen:
        if k not in allowed:
            check(f"spec_frozen_unchanged:{k}", frozen[k] == spec.get(k))
    check("spec_no_unexpected_keys", set(spec) - set(frozen) <= allowed, str(set(spec) - set(frozen)))
    check("spec_version_still_0.2-frozen", spec["version"] == "0.2-frozen")

    oof, oc = load("oof_nobw_manifest.json"), load("oof_nobw_checks.json")
    check("oof_40_fits", oc["fits"] == 40 and len(oof["fits"]) == 40)
    check("oof_110_features_without_kinryo_bw_ratio", oof["baseline_contract"]["n_features"] == 110
          and "斤量体重比" not in oof["features"] and all(c in oof["features"] for c in ["前走馬体重", "前走馬体重増減", "斤量", "馬齢斤量差"]))
    check("oof_reproducible_bitwise", oc["reproducibility"]["predictions_bitwise_equal"] is True)
    check("oof_no_future_rows", all(v["ok"] for v in oc["future_row_checks"].values()))
    check("oof_manifest_hashes", all(k in oof for k in ("feature_list_sha256", "input_sha256", "loader_sha256", "row_hashes"))
          and all("model_sha256" in f for f in oof["fits"].values()))

    pc, man = load("population_coverage.json"), load("stage0_manifest.json")
    check("known_structure_reproduced", pc["known_structure_reproduction"]["reproduced"] is True)
    check("main_population_matches_exp16a_base", pc["main_population"]["total"] == pc["main_population"]["base_exp16a_total"] == 15951)
    check("wp_joint_floor_each_year", pc["wp_joint_floor_pass"] is True)
    check("loader_sha_current", man["loader_sha256"] == loader_sha256())
    check("race_set_and_raw_input_hashes", len(man["race_set"]["main_2019_2023_sha256"]) == 64 and "torch" in man["raw_input_sha256"])
    check("features_exclude_change_pct_from_W", "bw_change_pct" not in man["feature_manifest"]["W_main"])

    inv = load("invariant_tests.json")
    check("invariant_tests_all_passed", inv["all_passed"], f"{inv['n_pass']}/{inv['n_tests']}")
    check("gate_boundary_tests", not boundary_tests())

    fp = load("forward_parity.json")
    check("forward_status_reported_honestly", fp["status"] == "収集中" and fp["gate_S0A_pass"] is False
          and fp["parity_with_historical_target"]["pairs_compared"] == 0)

    pf = load("power_floor.json")
    for g in ("B1", "B2"):
        gi = pf["gates"][g]
        check(f"{g}_floor_numeric", isinstance(gi.get("practical_floor_nats"), (int, float)), repr(gi.get("practical_floor_nats")))
        check(f"{g}_delta0_sane", gi["delta0_sane"] is True)
    for g in ("A1", "A2"):
        gi = pf["gates"][g]
        check(f"{g}_floor_not_fabricated", gi["practical_floor_nats"] is None and gi["delta0_sane"] is False)
    res = spec.get("stage0_results", {})
    check("spec_floors_match_power_json", all(res.get("floors_nats", {}).get(g) == pf["gates"][g]["practical_floor_nats"]
                                              for g in ("A1", "A2", "B1", "B2")))
    check("compute_measured", load("compute_dry_run.json").get("peak_rss_mb", 0) > 0)
    txt = json.dumps(pf) + json.dumps(pc)
    check("no_real_outcome_performance_in_outputs", "real_delta" not in txt and "auc" not in txt.lower())
    return report()


def report():
    print(f"OK   {len(oks)}")
    for f in fails:
        print("FAIL " + f)
    print("ALL CHECKS PASSED" if not fails else f"{len(fails)} CHECK(S) FAILED")
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
