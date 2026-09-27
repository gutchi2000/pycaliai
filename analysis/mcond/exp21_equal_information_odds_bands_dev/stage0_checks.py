# -*- coding: utf-8 -*-
"""
stage0_checks.py — EXP21 Stage 0 の整合検査 (読み取りだけ)
実行: python -m analysis.mcond.exp21_equal_information_odds_bands_dev.stage0_checks
"""
from __future__ import annotations

import json
import subprocess
import sys

from .loaders import BASE, HERE, OUT

FROZEN = "8cf169ec"
REL = "analysis/mcond/exp21_equal_information_odds_bands_dev"
OUTS = ["data_manifest.json", "g0_audit.json", "label_free_band_coverage.json", "power_audit.json",
        "winner_claim_uncertainty.json", "invariant_tests.json"]
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
    for n in ("STAGE0_DATA_AUDIT.md", "WINNER_CLAIM_UNCERTAINTY.md"):
        check(f"exists:{n}", (HERE / n).exists())
    if fails:
        return report()
    spec = json.loads((HERE / "spec.json").read_text(encoding="utf-8"))
    frozen = json.loads(subprocess.run(["git", "-c", f"safe.directory={BASE.as_posix()}", "show", f"{FROZEN}:{REL}/spec.json"], capture_output=True, text=True,
                                       encoding="utf-8", cwd=BASE).stdout)
    allowed = {"status", "stage0_started", "stage0_results", "version", "v03", "stage1_results"}      # v0.3-frozen で追加・変更が許される項目
    closed = "final_status" in spec   # Stage 1 後の最終文書化 (Fable 最終レビュー承認)
    # 最終文書化で許される変更は、v0.2 の撤回表現に superseded 注記を付けることと、開封記録だけ
    amended = {"price_layers", "scope_matrix", "outcomes_opened_by_exp21"} if closed else set()
    if closed:
        allowed |= {"final_status", "outcomes_opening_record"}
    for k in frozen:
        if k not in allowed and k not in amended:
            check(f"spec_frozen_unchanged:{k}", frozen[k] == spec.get(k))
    check("spec_no_unexpected_keys", set(spec) - set(frozen) <= allowed, str(set(spec) - set(frozen)))
    if not closed:
        check("outcomes_not_opened_for_roi", spec["outcomes_opened_by_exp21"] is False)
    else:
        rec_ = spec.get("outcomes_opening_record", {})
        check("closed_outcomes_opening_recorded", spec["outcomes_opened_by_exp21"] is True and "7cd7d452" in rec_.get("stage1", "")
              and "2019-2023" in rec_.get("stage1", "") and "layout" in rec_.get("stage0", "") and rec_.get("sealed_untouched") == ["2024", "2025"])
        fp, sp = frozen["price_layers"], spec["price_layers"]
        check("closed_price_layers_only_D1_forward_superseded",
              {k: v for k, v in sp.items() if k not in ("D1_forward", "D1_forward_superseded")} == {k: v for k, v in fp.items() if k != "D1_forward"}
              and sp["D1_forward"].startswith("SUPERSEDED by v0.3") and fp["D1_forward"] in sp["D1_forward"] and sp.get("D1_forward_superseded") is True)
        fs, ss = frozen["scope_matrix"], spec["scope_matrix"]
        check("closed_scope_matrix_only_other_five_superseded",
              {k: v for k, v in ss.items() if k != "other_five"} == {k: v for k, v in fs.items() if k != "other_five"}
              and ss["other_five"]["history_superseded_v02"] == fs["other_five"]["history"]
              and ss["other_five"]["history"].startswith("SUPERSEDED by v0.3") and ss["other_five"]["g1_g2"] == fs["other_five"]["g1_g2"])

    g = load("g0_audit.json")
    man = load("data_manifest.json")
    check("manifest_has_sha256_for_all_sources", all(len(s["sha256"]) == 64 for s in man["sources"]) and len(man["sources"]) >= 50)
    check("raw2023_identified_terminal", g["raw2023_timing"]["identified"] == "terminal"
          and g["raw2023_timing"]["match"]["tan"]["terminal"]["rate"] >= 0.99
          and g["raw2023_timing"]["match"]["umaren"]["terminal"]["rate"] >= 0.99)
    for t in ("wakuren", "umaren", "wide", "umatan"):
        check(f"layout_2023_pass:{t}", g["raw2023_layout"][t]["layout_pass"])
    for t in ("wakuren", "umaren", "wide", "umatan", "sanrenpuku"):
        check(f"layout_2026_terminal_pass:{t}", g["od2026_terminal_layout"][t]["layout_pass"])
    check("key_checks_pass", g["raw2023_keys"]["pass"] and g["od2026_terminal_keys"]["pass"])
    check("sanrentan_g0_fail", g["g0"]["sanrentan"]["pass"] is False)
    check("no_od_day_called_T10", all(v["identified"] != "T-10" for v in g["od2026_files"].values())
          and spec["stage0_results"]["od2026_timing"]["identified_T10_days"] == 0)
    check("spec_g0_matches_audit", spec["stage0_results"]["g0"] == {t: g["g0"][t]["pass"] for t in g["g0"]})

    c = load("label_free_band_coverage.json")
    check("band_coverage_label_free", c["rules"]["no_outcome_columns_read"] is True)
    hist = c["history"]
    check("history_primary_10_bands_all_sufficient",
          all(not b["insufficient"] for v in hist.values() for b in v["bands"]["primary_mass10"] if b["tickets"]))
    band_keys = {k for grp in list(hist.values()) + list(c["other_types"].values())
                 for bl in grp["bands"].values() for row in bl for k in row}
    check("no_realized_roi_in_band_outputs", {k for k in band_keys if "roi" in k.lower() or "hit" in k.lower()
                                              and k != "expected_hit_mass"} <= {"null_roi_expectation"},
          str(sorted(band_keys)))
    check("no_sanrentan_bands", not any(k.startswith("sanrentan") for k in c["other_types"]))

    pw = load("power_audit.json")
    check("power_seeds_frozen", pw["seed"] == 20260927 and pw["stage1_bootstrap"]["B"] == 10000
          and pw["stage1_bootstrap"]["seed"] == 20260928)
    check("power_false_pass_low", all(v["false_pass_rate_delta0"] <= 0.025 for v in pw["types"].values()))
    check("power_mde_defined", all(v["mde_delta"] is not None for v in pw["types"].values()))

    inv = load("invariant_tests.json")
    check("invariant_tests_all_passed", inv["all_passed"], f"{inv['n_pass']}/{inv['n_tests']}")
    txt = (HERE / "STAGE0_DATA_AUDIT.md").read_text(encoding="utf-8")
    check("report_does_not_call_pre_T10", "約 T−28" in txt and "T−28 を T−10 と呼ばない" in txt)
    return report()


def report():
    print(f"OK   {len(oks)}")
    for f in fails:
        print("FAIL " + f)
    print("ALL CHECKS PASSED" if not fails else f"{len(fails)} CHECK(S) FAILED")
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
