# -*- coding: utf-8 -*-
"""
stage0_checks.py — EXP18 Stage 0 成果物の整合検査 (読み取りだけ)
================================================================
  * 必須成果物の存在と JSON 妥当性
  * spec.json の凍結部分が commit 67b874ec から変わっていないこと
    (許可: status・stage0.power.practical_floor_nats (null→数値)・stage0_results の追加)
  * practical_floor_nats が数値で、out/power_audit.json と一致
  * 合成 oracle / invariant が全通過
  * 結果 loader が 2018 年以下、2019-2023 の結果指標が成果物に無いこと
  * T2 往復 Gate・anchor・coverage floor・power の判定が spec.json / 文書と一致
  * 禁止表現 (券種全体票数を組合せ別実票数と呼ぶ 等) が無いこと
実行: python -m analysis.mcond.exp18_cross_pool_market_tomography_dev.stage0_checks
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
OUT = HERE / "out"
FROZEN_COMMIT = "67b874ec"
DELIVERABLES = ["PRIOR_ART_EQUIVALENCE_AUDIT.md", "MARKET_POOL_PROVENANCE.md", "POOL_SCHEMA_MANIFEST.json",
                "RACE_AND_TICKET_COVERAGE.json", "TOMOGRAPHY_IDENTIFIABILITY_AUDIT.md", "POWER_AUDIT.md",
                "out/power_audit.json", "COMPUTE_DRY_RUN.json", "out/invariant_tests.json",
                "out/anchor_le2018.json", "out/fukusho_roundtrip.json", "SPEC.md", "spec.json", "README.md"]
oks, fails = [], []


def check(name, cond, detail=""):
    (oks if cond else fails).append(name + (f" — {detail}" if detail else ""))


def load(p):
    return json.loads((HERE / p).read_text(encoding="utf-8"))


def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    for f in DELIVERABLES:
        check(f"exists:{f}", (HERE / f).exists())
    data = {}
    for f in [x for x in DELIVERABLES if x.endswith(".json")]:
        try:
            data[f] = load(f)
            check(f"json:{f}", True)
        except Exception as e:
            check(f"json:{f}", False, str(e))
    if fails:
        return report()
    spec = data["spec.json"]
    frozen = json.loads(subprocess.run(
        ["git", "show", f"{FROZEN_COMMIT}:analysis/mcond/exp18_cross_pool_market_tomography_dev/spec.json"],
        capture_output=True, text=True, encoding="utf-8", cwd=BASE).stdout)

    # ---- 凍結部分の不変
    allowed_top = {"status", "stage0_results"}
    for k in frozen:
        if k in allowed_top:
            continue
        if k == "stage0":
            a = json.loads(json.dumps(frozen[k]))
            b = json.loads(json.dumps(spec[k]))
            a["power"].pop("practical_floor_nats", None)
            b["power"].pop("practical_floor_nats", None)
            check("frozen_unchanged:stage0 (floor 以外)", a == b)
        else:
            check(f"frozen_unchanged:{k}", frozen[k] == spec.get(k))
    extra = set(spec) - set(frozen) - allowed_top
    check("no_unexpected_new_keys", not extra, str(extra))

    # ---- 実務床
    pw = data["out/power_audit.json"]
    fl = spec["stage0"]["power"]["practical_floor_nats"]
    check("practical_floor_numeric", isinstance(fl, (int, float)), repr(fl))
    check("practical_floor_matches_power_audit",
          isinstance(fl, (int, float)) and pw["practical_floor_nats"] is not None
          and abs(fl - pw["practical_floor_nats"]) < 1e-12)
    check("power_seed_recorded", pw.get("seed") == 20260925 and pw.get("seed_fixed_before_run") is True)
    af = pw["power"].get("at_floor", {})
    check("power_wilson_and_reps_recorded", "signal_wilson95" in af and af.get("reps", 0) >= 200)
    check("power_gate_consistent_with_spec",
          spec["stage0_results"]["power"]["power_gate_pass"] == pw["power_gate_pass"])

    # ---- invariant
    inv = data["out/invariant_tests.json"]
    check("invariant_tests_all_passed", inv["all_passed"] and inv["n_tests"] >= 30,
          f"{inv['n_pass']}/{inv['n_tests']}")
    names = {t["test"] for t in inv["tests"]}
    for need in ["oracle_harville_unordered_top2_n<=8", "oracle_stern_unordered_top2_n<=8",
                 "U_SELF_devig_bitwise", "P5_uniform_qlopo_delta_le_1e-12", "T2_w0_equals_T1_bitwise",
                 "collinearity_cond_gt_1e12_drops_duplicate", "outcome_modification_invariance",
                 "future_year_deletion_invariance_2018", "target_umaren_columns_absent_from_source_input",
                 "coherent_market_tansho_equals_umaren", "distorted_target_pool_detected"]:
        check(f"invariant_present:{need}", need in names)

    # ---- 結果 loader と 2019+ の結果指標
    anc = data["out/anchor_le2018.json"]
    check("result_loader_max_year_le_2018", anc["result_loader_max_year"] <= 2018)
    check("anchor_years_le_2018", all(int(y) <= 2018 for y in anc["per_year_eval_lambda_star"]))
    check("anchor_consistent_with_spec",
          spec["stage0_results"]["anchor"]["anchor_pass"] == anc["anchor"]["anchor_pass"])
    cov = data["RACE_AND_TICKET_COVERAGE.json"]
    check("no_2019_2023_outcome_counts",
          all(int(y) <= 2018 for y in cov["outcome_conditioned_exclusions"]["le2018_measured"]))

    # ---- T2 往復 Gate
    rt = data["out/fukusho_roundtrip.json"]
    check("t2_decision_consistent", spec["stage0_results"]["t2_roundtrip"]["decision"] == rt["decision"])
    ta = (HERE / "TOMOGRAPHY_IDENTIFIABILITY_AUDIT.md").read_text(encoding="utf-8")
    check("t2_doc_reports_decision", rt["decision"] in ta)
    if not rt["gate_pass"]:
        check("ub3_not_a_gate_arm_when_t2_fails",
              "UB3" in json.dumps(spec["stage0_results"], ensure_ascii=False)
              and "除外" in json.dumps(spec["stage0_results"]["t2_roundtrip"], ensure_ascii=False))

    # ---- coverage floor
    fc = cov["floor_check_pooled_2019_2023"]
    check("coverage_floors_consistent",
          spec["stage0_results"]["coverage"]["all_floors_pass"] == cov["all_floors_pass_pooled"])
    for k, v in fc.items():
        check(f"coverage_value_recorded:{k}", v["value"] is not None)

    # ---- 禁止表現
    docs = "".join((HERE / f).read_text(encoding="utf-8") for f in DELIVERABLES if f.endswith(".md"))
    check("votes_not_called_combination_level",
          not re.search(r"組合せ別(の)?実票数(を|として)(使|扱)", docs) and "券種全体" in docs)
    check("pre_not_called_T10", not re.search(r"historical_pre_snapshot[^\n]{0,20}(= ?T-10|をT-10)", docs))
    check("noise_sd_declared_not_derived", "導出ではない" in (HERE / "POWER_AUDIT.md").read_text(encoding="utf-8"))
    check("T0_T1_not_claimed_novel",
          "既実施" in (HERE / "PRIOR_ART_EQUIVALENCE_AUDIT.md").read_text(encoding="utf-8"))
    return report()


def report():
    print(f"OK   {len(oks)}")
    for f in fails:
        print("FAIL " + f)
    print("ALL CHECKS PASSED" if not fails else f"{len(fails)} CHECK(S) FAILED")
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
