# -*- coding: utf-8 -*-
"""
v05_checks.py — EXP18 v0.5-final 凍結後・2019-2023 結果開封前の整合検査 (読み取りだけ)
======================================================================================
全 PASS でなければ evaluate_stage1.py を実行しない。
  * spec が v0.5-final、実務床・ρ=0.5 参考値が数値で out/floor_v05.json と一致、SIGNAL 検出力 Gate 通過
  * 変更禁止項目が v0.4 最終 (commit 3c1deb28) から不変
  * 旧 floor 0.8843 が superseded_invalid_economic_floor として残っている
  * 合成テスト (Stage 0 invariant / floor_v05 / Stage 1 machinery / Gate 境界) がすべて PASS
  * Stage 1 の出力がまだ存在しない (結果未開封)
  * EXP18 ディレクトリに未 commit の変更が無い (凍結コード = commit 済みコード)
実行: python -m analysis.mcond.exp18_cross_pool_market_tomography_dev.v05_checks
"""
from __future__ import annotations

import json
import subprocess
import sys

from .gate_grade import run_boundary_tests
from .loaders import BASE, HERE, OUT, RESEARCH

V04_FINAL = "3c1deb28"
REL = "analysis/mcond/exp18_cross_pool_market_tomography_dev"
FORBIDDEN_TOP = ["primary_comparison", "arms", "hierarchy", "placebos", "placebo_draws_min", "placebo_threshold",
                 "population", "development", "t2", "sealed_caveat", "hard_prohibitions", "stop_rules",
                 "sample_space", "primary_loss", "inference_unit", "null_market", "candidate_families", "data",
                 "primary_target_pool", "purpose", "experiment", "stage0_deliverables"]
FORBIDDEN_M1 = ["comparison", "PASS_SIGNAL", "FAIL", "progression"]
FORBIDDEN_POWER = ["signal", "growth_threshold_per_race", "required_power", "takeout", "freeze_rule"]
oks, fails = [], []


def check(name, cond, detail=""):
    (oks if cond else fails).append(name + (f" — {detail}" if detail else ""))


def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    spec = json.loads((HERE / "spec.json").read_text(encoding="utf-8"))
    v04 = json.loads(subprocess.run(["git", "show", f"{V04_FINAL}:{REL}/spec.json"], capture_output=True, text=True,
                                    encoding="utf-8", cwd=BASE).stdout)
    check("version_0.5_final", spec.get("version") == "0.5-final", spec.get("version"))
    for k in FORBIDDEN_TOP:
        check(f"unchanged:{k}", spec.get(k) == v04.get(k))
    for k in FORBIDDEN_M1:
        check(f"unchanged:gates.M1.{k}", spec["gates"]["M1"].get(k) == v04["gates"]["M1"].get(k))
    check("unchanged:gates.M2", spec["gates"]["M2"] == v04["gates"]["M2"])
    for k in FORBIDDEN_POWER:
        check(f"unchanged:stage0.power.{k}", spec["stage0"]["power"].get(k) == v04["stage0"]["power"].get(k))
    for k in [k for k in v04["stage0"] if k != "power"]:
        check(f"unchanged:stage0.{k}", spec["stage0"].get(k) == v04["stage0"].get(k))
    check("growth_threshold_1e-4", spec["stage0"]["power"]["growth_threshold_per_race"] == 0.0001)

    pw = spec["stage0"]["power"]
    fl = json.loads((OUT / "floor_v05.json").read_text(encoding="utf-8"))
    po = json.loads((OUT / "power_v05.json").read_text(encoding="utf-8"))
    check("floor_numeric", isinstance(pw["practical_floor_nats"], (int, float)) and pw["practical_floor_nats"] > 0,
          repr(pw["practical_floor_nats"]))
    check("floor_matches_floor_v05_json", pw["practical_floor_nats"] == fl["practical_floor_nats"])
    check("rho05_reference_recorded_and_matches", pw.get("sensitivity_floor_rho_0_5_nats") ==
          fl["sensitivity_floor_rho_0_5_nats"])
    check("power_uses_the_committed_floor", po["floor"] == pw["practical_floor_nats"])
    check("signal_power_gate_pass", po["signal_power_gate_pass"] is True and
          pw.get("signal_power_gate_pass_v05") is True,
          f"power={po['results']['floor']['signal_power']}")
    sup = pw.get("superseded_invalid_economic_floor", {})
    check("old_floor_kept_as_superseded", sup.get("value_nats") == 0.8843125150706383 and
          "superseded_invalid_economic_floor" in sup.get("status", ""))
    check("pass_practical_form_v05", "CI95 upper < -practical_floor_nats" in spec["gates"]["M1"]["PASS_PRACTICAL"])
    check("gate_arm_ub2_only", "UB2" in spec["stage1_protocol_v05"]["arm_for_gate"])
    check("t2_ub3_stopped", spec["stage0_results"]["t2_roundtrip"]["decision"] == "T2_unimplemented_stop")

    for f in ("invariant_tests.json", "floor_v05_tests.json", "stage1_machinery_tests.json"):
        d = json.loads((OUT / f).read_text(encoding="utf-8"))
        check(f"tests_all_passed:{f}", d["all_passed"], f"{d['n_pass']}/{d['n_tests']}")
    bad = run_boundary_tests(pw["practical_floor_nats"])
    check("gate_boundary_tests_at_committed_floor", not bad, "; ".join(bad))

    for f in (OUT / "stage1_eval.json", OUT / "stage1_race_set_crosscheck.json", OUT / "stage1_race_set_diff.json",
              RESEARCH / "stage1_arrays.npz", RESEARCH / "stage1_delta_by_race.npz", OUT / "stage1_fits.json"):
        check(f"not_yet_opened:{f.name}", not f.exists())

    st = subprocess.run(["git", "status", "--porcelain", "--", REL], capture_output=True, text=True,
                        encoding="utf-8", cwd=BASE).stdout.strip()
    check("exp18_dir_committed_clean", st == "", st[:200])

    print(f"OK   {len(oks)}")
    for f in fails:
        print("FAIL " + f)
    print("ALL CHECKS PASSED" if not fails else f"{len(fails)} CHECK(S) FAILED")
    (OUT / "v05_checks.json").write_text(json.dumps({"n_ok": len(oks), "fails": fails, "all_passed": not fails},
                                                    ensure_ascii=False, indent=1), encoding="utf-8")
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
