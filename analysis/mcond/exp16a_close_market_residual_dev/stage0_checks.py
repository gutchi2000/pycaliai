# -*- coding: utf-8 -*-
"""
stage0_checks.py — EXP16A Stage 0 の JSON 検証・文書間整合性チェック
====================================================================
読み取りだけ。学習も評価もしない。失敗した検査を列挙して非 0 で終了する。
実行: python -m analysis.mcond.exp16a_close_market_residual_dev.stage0_checks
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

from .gate_grade import (FLOOR, STATEMENTS, economic_checks_allowed,
                         run_boundary_tests)

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
OUT = HERE / "out"
DOCS = ["PRIOR_ART_AUDIT.md", "INFORMATION_TO_GROWTH_DERIVATION.md", "MARKET_DATA_PROVENANCE.md",
        "RACE_POPULATION_AUDIT.md", "OOF_STACKING_PLAN.md", "POWER_AUDIT.md"]
JSONS = ["spec.json", "STAGE0_DRY_RUN.json"]
OUT_JSONS = ["race_population.json", "power_audit.json", "market_provenance.json",
             "derivation_checks.json"]
CODE = ["provenance.py", "race_population.py", "power_audit.py", "verify_growth.py",
        "gate_grade.py", "stage0_dry_run.py", "stage0_checks.py"]

fails: list[str] = []
oks: list[str] = []


def check(name: str, cond: bool, detail: str = ""):
    (oks if cond else fails).append(f"{name}{(' — ' + detail) if detail else ''}")


def num(x) -> float:
    return float(x)


def main():
    # ---- 1. ファイルの存在と JSON の妥当性
    for f in DOCS + JSONS + CODE:
        check(f"exists:{f}", (HERE / f).exists())
    for f in OUT_JSONS:
        check(f"exists:out/{f}", (OUT / f).exists())
    data = {}
    for f in JSONS:
        try:
            data[f] = json.loads((HERE / f).read_text(encoding="utf-8"))
            check(f"json_parse:{f}", True)
        except Exception as e:
            check(f"json_parse:{f}", False, str(e))
    for f in OUT_JSONS:
        try:
            data[f] = json.loads((OUT / f).read_text(encoding="utf-8"))
            check(f"json_parse:out/{f}", True)
        except Exception as e:
            check(f"json_parse:out/{f}", False, str(e))
    if fails:
        report()
        return
    spec, dry = data["spec.json"], data["STAGE0_DRY_RUN.json"]
    pop, pw = data["race_population.json"], data["power_audit.json"]
    texts = {f: (HERE / f).read_text(encoding="utf-8") for f in DOCS}

    # ---- 2. spec の必須キー
    for k in ["experiment", "spec_version", "purpose", "periods", "official_race_set", "arms",
              "metrics", "thresholds_fixed_before_power_audit", "power_audit", "gates",
              "five_year_judgement", "artifact_contract", "stage0_deliverables",
              "stage1_entry_condition", "c2c_economic_scale_fixed_interpretation",
              "auxiliary_economic_checks"]:
        check(f"spec_key:{k}", k in spec)

    # ---- 3. race 母集団の一致 (spec / dry-run / race_population.json / 監査文書)
    tot = pop["totals"]
    check("official_total:spec==pop",
          spec["official_race_set"]["measured_2016_2023"]["official_eligible"] == tot["official_eligible"],
          f'{spec["official_race_set"]["measured_2016_2023"]["official_eligible"]} vs {tot["official_eligible"]}')
    check("official_total:dry==pop",
          dry["official_race_set"]["totals"]["official_eligible"] == tot["official_eligible"])
    md = texts["RACE_POPULATION_AUDIT.md"]
    m = re.search(r"\|\s*\*\*合計\*\*\s*\|([^\n]+)\|", md)
    check("audit_md_total_row_present", m is not None)
    if m:
        cells = [c.strip().replace("**", "").replace(",", "") for c in m.group(1).split("|")]
        nums = [int(c) for c in cells if re.fullmatch(r"\d+", c)]
        for key in ["races_joined", "excl_jump", "official_eligible"]:
            check(f"audit_md_contains:{key}", tot[key] in nums, f"{tot[key]} not in {nums}")

    # ---- 4. 2022 基準値の一致 (4 箇所)
    ref22 = pop["reference_values_by_year_le2022"]["2022"]
    close, pre = ref22["terminal_close_market_logloss"], ref22["historical_pre_snapshot_logloss"]
    gap = ref22["pre_minus_close_gap_nats"]
    check("ref2022:spec close",
          abs(num(spec["reference_values_2022_official_set"]["terminal_close_market_race_logloss"]) - close) < 1e-9)
    check("ref2022:spec pre",
          abs(num(spec["reference_values_2022_official_set"]["historical_pre_snapshot_race_logloss"]) - pre) < 1e-9)
    check("ref2022:spec n_races",
          spec["reference_values_2022_official_set"]["n_races"] == ref22["n_races"])
    check("ref2022:dry close",
          abs(num(dry["reference_values_2022_official_set"]["terminal_close_market"]["race_categorical_logloss"]) - close) < 1e-9)
    check("ref2022:spec gap",
          abs(num(spec["metrics"]["recovery_ratio"]["measured_gap_2022_official"]) - gap) < 1e-9)
    for doc, pat in [("MARKET_DATA_PROVENANCE.md", f"{close:.5f}"),
                     ("RACE_POPULATION_AUDIT.md", f"{close:.6f}")]:
        check(f"ref2022_in_doc:{doc}", pat in texts[doc], pat)
    check("ref2022:n_races_in_market_doc", f'{ref22["n_races"]:,}' in texts["MARKET_DATA_PROVENANCE.md"]
          or str(ref22["n_races"]) in texts["MARKET_DATA_PROVENANCE.md"])

    # ---- 5. 2023 の結果値が記録されていないこと
    check("no_2023_reference_values",
          "2023" not in pop["reference_values_by_year_le2022"] and
          "2023" not in dry["reference_values_by_year_le2022"])
    check("power_audit_le2022_only",
          all(int(y) <= 2022 for y in pw["data"]["races_by_year"]))

    # ---- 6. 障害除外の定義が production と一致
    el = (BASE / "race_eligibility.py").read_text(encoding="utf-8")
    check("jump_range_production", "JUMP_TRACK_MIN, JUMP_TRACK_MAX = 51, 59" in el)
    for doc in ["RACE_POPULATION_AUDIT.md"]:
        check(f"jump_range_doc:{doc}", "51..59" in texts[doc])
    check("jump_range_spec", "51..59" in json.dumps(spec, ensure_ascii=False))
    check("jump_count:spec==pop",
          spec["official_race_set"]["measured_2016_2023"]["excl_jump"] == tot["excl_jump"])

    # ---- 7. 閾値の一致 (床 0.005 / recovery 0.10)
    th = spec["thresholds_fixed_before_power_audit"]
    check("floor:spec", num(th["absolute_practical_floor_nats_per_race"]) == 0.005)
    check("floor:dry", num(dry["thresholds_fixed_before_power_audit"]
                           ["absolute_practical_floor_nats_per_race"]) == 0.005)
    check("floor:power_audit_runs",
          all(num(v["practical_floor_used"]) == 0.005
              for v in pw["tier2_empirical_cluster_power"]["runs"].values()))
    check("recovery_ratio:spec", num(th["recovery_ratio_floor"]) == 0.10)
    check("recovery_ratio:metrics", num(spec["metrics"]["recovery_ratio"]["floor"]) == 0.10)

    # ---- 8. 検出力の結論と進行規則が全文書で一致
    t2 = pw["tier2_empirical_cluster_power"]
    ver, sres = t2["verdict"], spec["power_audit"]["tier2_result"]
    check("power_verdict:spec", sres["result"] == ver["result"])
    check("power_verdict:doc", ver["result"] in texts["POWER_AUDIT.md"])
    check("power_progression_rule:spec",
          "PASS-PRACTICAL ∪ PASS-SIGNAL" in json.dumps(spec["power_audit"], ensure_ascii=False) and
          "80%" in json.dumps(spec["power_audit"], ensure_ascii=False))
    check("power_progression_rule:not_practical_only",
          "PASS-PRACTICAL 単独の 80% は要求しない" in
          json.dumps(spec["power_audit"], ensure_ascii=False))
    check("power_detect_at_floor:spec==json",
          abs(num(min(sres["power_pass_practical_or_signal_at_floor"].values())) -
              num(ver["min_power_detect_at_floor"])) < 1e-12)
    check("power_practical_at_0069:spec==json",
          sorted(num(x) for x in sres["power_pass_practical_at_0_0069"].values()) ==
          sorted(num(x) for x in ver["power_pass_practical_at_0_0069"].values()),
          "spec=%s json=%s" % (sres["power_pass_practical_at_0_0069"],
                               ver["power_pass_practical_at_0_0069"]))
    check("power_0069_not_a_floor_change",
          "床の変更ではな" in json.dumps(sres, ensure_ascii=False))
    check("power_tier1_not_a_gate",
          "理論 MDE" in json.dumps(spec["power_audit"], ensure_ascii=False))
    # 各 run が反復数・seed・成功回数・power・Wilson CI を記録している
    need = ["reps", "rng_seed", "successes_pass_practical", "successes_pass_practical_or_signal",
            "power_pass_practical", "power_pass_practical_or_signal",
            "wilson95_pass_practical", "wilson95_pass_practical_or_signal"]
    ok_rec = all(all(k in v for k in need)
                 for r in t2["runs"].values() for v in r["by_seed_jitter"].values())
    check("power_runs_record_required_fields", ok_rec)
    dec = t2["runs"]["terminal_close_market__prefixed_residual_linear__true_0.005"]["by_seed_jitter"]
    check("power_decision_reps>=400", all(v["reps"] >= 400 for v in dec.values()),
          str({k: v["reps"] for k, v in dec.items()}))
    check("power_wilson_lower_recorded",
          all(isinstance(v["wilson95_pass_practical_or_signal"], list) for v in dec.values()))
    check("power_wilson_lower_not_far_below_80",
          num(ver["min_wilson95_lower_at_floor"]) >= 0.75 or ver["result"] != "proceed_to_stage1_pending_review",
          str(ver["min_wilson95_lower_at_floor"]))

    # ---- 8b. Gate の 3 等級 (境界テストと文言一致)
    bad = run_boundary_tests()
    check("gate_grade_boundary_tests", not bad, "; ".join(bad))
    for g in ["A", "B"]:
        gk = "A_close_market_residual" if g == "A" else "B_decision_time_reproducibility"
        grades = spec["gates"][gk]["grades"]
        check(f"gate_{g}_has_three_grades",
              sorted(grades) == ["FAIL", "PASS-PRACTICAL", "PASS-SIGNAL"], str(sorted(grades)))
        for grade, stmt in STATEMENTS[g].items():
            check(f"gate_{g}_statement_matches_code:{grade}",
                  grades[grade]["statement"] == stmt)
    check("gate_A_practical_requires_ci_below_floor",
          any("CI95 上限 < -0.005" in c
              for c in spec["gates"]["A_close_market_residual"]["grades"]["PASS-PRACTICAL"]["conditions"]))
    check("gate_A_signal_excludes_floor",
          any("CI95 上限 < -0.005 は満たさない" in c
              for c in spec["gates"]["A_close_market_residual"]["grades"]["PASS-SIGNAL"]["conditions"]))
    check("point_estimate_not_a_condition",
          "点推定 <= -0.005 を PASS-PRACTICAL の条件にしない" in spec["gates"]["point_estimate_policy"])
    for v in ["pooled point estimate", "CI95 (下限・上限)"]:
        check(f"reported_values_contains:{v}", v in spec["gates"]["reported_values"])
    check("economic_checks_both_practical",
          "ともに PASS-PRACTICAL" in spec["gates"]["economic_checks_condition"]["rule"] and
          "ともに PASS-PRACTICAL" in spec["auxiliary_economic_checks"]["run_if"])
    check("economic_checks_helper", economic_checks_allowed("PASS-PRACTICAL", "PASS-PRACTICAL") and
          not economic_checks_allowed("PASS-SIGNAL", "PASS-PRACTICAL"))
    check("grade_names_in_power_doc",
          all(x in texts["POWER_AUDIT.md"] for x in ["PASS-PRACTICAL", "PASS-SIGNAL", "FAIL"]))
    check("gate_grade_module_is_single_source",
          "gate_grade" in spec["gates"]["grading_implementation"] and
          "gate_grade" in pw.get("grading_implementation", ""))

    # ---- 8c. C2c の固定解釈
    c2c = spec["c2c_economic_scale_fixed_interpretation"]
    dc = data["derivation_checks.json"]["C2c_calibrated_tilt_selection"]["by_target"]
    check("c2c:participation_0.005",
          abs(num(c2c["measured"]["delta_0.005"]["participation_share_of_races"]) -
              num(dc["true_improvement_0.005"]["share_races_participation_condition"])) < 1e-12)
    check("c2c:growth_0.005",
          abs(num(c2c["measured"]["delta_0.005"]["expected_log_growth"]) -
              num(dc["true_improvement_0.005"]["mean_growth_true_belief_cash_kelly"])) < 1e-15)
    check("c2c:flip_0.005",
          abs(num(c2c["measured"]["delta_0.005"]["sign_flip_noise_sd"]) -
              num(dc["true_improvement_0.005"]["sign_flip_noise_sd"])) < 1e-12)
    check("c2c:growth_0.02",
          abs(num(c2c["measured"]["delta_0.02"]["expected_log_growth"]) -
              num(dc["true_improvement_0.02"]["mean_growth_true_belief_cash_kelly"])) < 1e-15)
    check("c2c:growth_0.05",
          abs(num(c2c["measured"]["delta_0.05"]["expected_log_growth"]) -
              num(dc["true_improvement_0.05"]["mean_growth_true_belief_cash_kelly"])) < 1e-15)
    check("c2c:fixed_interpretation_wording",
          "候補生成へ進む根拠にはならない" in c2c["fixed_interpretation"])
    check("c2c:simulation_caveat", "合成シミュレーションの前提" in c2c["caveat"])

    # ---- 8d. DNF 再正規化の prior-art 注記
    md_pa = texts["PRIOR_ART_AUDIT.md"]
    sens = pop["dnf_sensitivity"]["flat_dnf_races_le2022"]
    flat_le2022 = sum(pop["per_year"][str(y)]["flat_races"] for y in range(2016, 2023))
    overall = num(sens["diff_starter_minus_finisher"]) * sens["n_races_total"] / flat_le2022
    check("prior_art_dnf_note_present", "DNF の市場再正規化" in md_pa)
    check("prior_art_dnf_0.061", "0.061366" in md_pa or "0.061" in md_pa)
    check("prior_art_dnf_overall_matches", f"{overall:.5f}"[:6] in md_pa or "0.00247" in md_pa,
          f"{overall:.6f}")
    check("prior_art_dnf_not_overturning",
          "それだけで過去実験の結論を覆すものではない" in md_pa)
    check("prior_art_dnf_exp16a_avoids", "平地・DNF なし" in md_pa)

    # ---- 9. crossfit 年の一致
    yrs = [2019, 2020, 2021, 2022, 2023]
    check("crossfit_years:spec", spec["retrospective_rolling_crossfit_development"]["years"] == yrs)
    check("crossfit_years:dry",
          sorted(int(k) for k in dry["retrospective_rolling_crossfit_development"]) == yrs)
    check("crossfit_years:pop_boundaries",
          sorted(int(k) for k in pop["learned_subset_boundaries_by_eval_year"]) == yrs)
    check("crossfit_name_in_docs",
          all("retrospective_rolling_crossfit_development" in texts[d]
              for d in ["OOF_STACKING_PLAN.md"]))
    check("no_auto_open_2024_2025",
          "自動開封" in json.dumps(spec["periods"], ensure_ascii=False) or
          "別途レビュー" in json.dumps(spec["periods"], ensure_ascii=False))

    # ---- 10. 禁止語・誤り表現
    blob = json.dumps(spec, ensure_ascii=False) + json.dumps(dry, ensure_ascii=False) + \
        "".join(texts.values())
    def only_with_disclaimer(term: str, words: tuple) -> bool:
        """term の出現がすべて『削除した / 呼ばない』という否定文脈の中にあること"""
        for m in re.finditer(re.escape(term), blob):
            w = blob[max(0, m.start() - 120): m.end() + 120]
            if not any(x in w for x in words):
                return False
        return True

    for bad in ["offline 限定", "offline限定"]:
        check(f"forbidden_absent:{bad}", bad not in blob)
    check("Q3_blend_only_as_removed_term", only_with_disclaimer("Q3 blend", ("削除",)))
    check("oracle_only_as_rejected_naming",
          only_with_disclaimer("oracle", ("呼ばない", "呼ばず")) and "terminal_close_market" in blob)
    check("kl_lower_bound_claim_removed",
          "下界だから" not in texts["INFORMATION_TO_GROWTH_DERIVATION.md"])
    check("C2c_present", "C2c" in texts["INFORMATION_TO_GROWTH_DERIVATION.md"] and
          "C2c_calibrated_tilt_selection" in data["derivation_checks.json"])
    check("dnf_not_treated_as_scratch",
          "DNF を取消として扱わない" in texts["RACE_POPULATION_AUDIT.md"])
    check("provisional_population_wording",
          "暫定母集団" in texts["RACE_POPULATION_AUDIT.md"] and
          "暫定母集団" in json.dumps(spec, ensure_ascii=False))

    # ---- 11. DNF 感度が実際に差を出していること
    s = pop["dnf_sensitivity"]
    check("dnf_sens:official_zero", abs(num(s["official_set_le2022"]["diff_starter_minus_finisher"])) < 1e-12)
    check("dnf_sens:dnf_races_nonzero",
          num(s["flat_dnf_races_le2022"]["diff_starter_minus_finisher"]) > 1e-3,
          str(s["flat_dnf_races_le2022"]["diff_starter_minus_finisher"]))
    check("dnf_sens:starter_defined_by_odds",
          "確定" in pop["horse_set_definitions"]["starter"] and "1.0" in pop["horse_set_definitions"]["starter"])

    # ---- 12. artifact 契約
    ac = dry["artifact_contract"]
    mv = ac["master_v2"]
    st = os.stat(mv["absolute_path"])
    check("artifact:master_size", int(mv["size_bytes"]) == st.st_size)
    check("artifact:master_mtime",
          mv["mtime"] == time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)))
    check("artifact:sha256_format", bool(re.fullmatch(r"[0-9a-f]{64}", mv["sha256"])))
    check("artifact:row_hash_verified_vs_exp15", bool(mv["row_hash_matches_exp15"]),
          f'now={mv["row_hash_2016_2023"][:12]} exp15={str(mv["exp15_recorded_row_hash"])[:12]}')
    check("artifact:feature_schema_verified", bool(ac["feature_schema"]["verified"]))
    check("artifact:p0_5_measured", isinstance(ac["p0_5_status"]["is_post_p0_5_c1_artifact"], bool))
    for k in ["jockey_fuku90", "prev_hosei"]:
        check(f"artifact:provenance:{k}", k in ac["feature_provenance"])
    check("artifact:spec_mirrors_contract",
          spec["artifact_contract"]["required_fields"] and
          "master_v2" in json.dumps(spec["artifact_contract"], ensure_ascii=False))

    # ---- 13. 5 年判定の条件が spec と OOF plan で一致
    fy = spec["five_year_judgement"]
    check("five_year:loo_hard_condition",
          any("leave-one-year-out" in c for c in fy["primary_conditions"]))
    check("five_year:grades_referenced",
          "PASS-PRACTICAL" in json.dumps(fy, ensure_ascii=False) or
          "gates" in json.dumps(fy, ensure_ascii=False))
    check("five_year:doc",
          "leave-one-year-out" in texts["OOF_STACKING_PLAN.md"])
    check("five_year:years", fy["years"] == yrs)

    report()


def report():
    print(f"OK   {len(oks)}")
    for f in fails:
        print("FAIL " + f)
    print(("ALL CHECKS PASSED" if not fails else f"{len(fails)} CHECK(S) FAILED"))
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
