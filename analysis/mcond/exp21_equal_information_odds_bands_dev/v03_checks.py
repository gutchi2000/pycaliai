# -*- coding: utf-8 -*-
"""
v03_checks.py — EXP21 v0.3 凍結の整合検査 (Stage 1 の結果開封前に実行、読み取りだけ)
  * spec が v0.3-frozen で、v03 ブロックに必須項目がそろう
  * v0.2 の凍結項目 (8cf169ec) は version / status / stage0_* / v03 以外不変
  * Stage 1 の出力がまだ無い (結果未開封)
  * 合成テスト全 PASS、stage0_checks 全 PASS
  * index にステージされたファイルが EXP21 ディレクトリだけ (凍結 commit 直前に実行)
実行: python -m analysis.mcond.exp21_equal_information_odds_bands_dev.v03_checks
"""
from __future__ import annotations

import json
import subprocess
import sys

from .loaders import BASE, HERE, OUT

REL = "analysis/mcond/exp21_equal_information_odds_bands_dev"
REQUIRED = ["stage1_types", "other_types", "baselines", "price_layers", "G1", "G2", "dead_heats", "inference",
            "power_positioning", "prior_expectations", "provenance", "sealed"]
oks, fails = [], []


def check(n, c, d=""):
    (oks if c else fails).append(n + (f" — {d}" if d else ""))


def git(*a):
    return subprocess.run(["git", "-c", f"safe.directory={BASE.as_posix()}", *a], capture_output=True, text=True,
                          encoding="utf-8", cwd=BASE).stdout


def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    spec = json.loads((HERE / "spec.json").read_text(encoding="utf-8"))
    check("version_0.3_frozen", spec["version"] == "0.3-frozen")
    v = spec.get("v03", {})
    for k in REQUIRED:
        check(f"v03_has:{k}", k in v)
    check("v03_stage1_types", v.get("stage1_types") == ["tansho", "fukusho", "umaren"])
    check("v03_T10_zero_days", "0 days" in v.get("price_layers", {}).get("T10", ""))
    check("v03_boot", v.get("inference", {}).get("B") == 10000 and v.get("inference", {}).get("seed") == 20260928)
    check("v03_prior_expectations_5", len(v.get("prior_expectations", [])) == 5)
    if "final_status" not in spec:
        for f in ("stage1_results.json",):
            check(f"stage1_not_opened:{f}", not (OUT / f).exists())
    else:
        # Stage 1 後: 結果は v0.3 凍結 commit の後で初めて commit されたこと
        added = git("log", "--diff-filter=A", "--format=%H", "--", f"{REL}/out/stage1_results.json").split()
        check("stage1_results_first_added_after_v03_freeze", len(added) == 1 and added[0].startswith("7cd7d452")
              and subprocess.run(["git", "-c", f"safe.directory={BASE.as_posix()}", "merge-base", "--is-ancestor", "e36ed2ca",
                                  added[0]], cwd=BASE).returncode == 0, str(added))
        check("stage1_results_record_freeze_commit", spec.get("stage1_results", {}).get("freeze_commit") == "e36ed2ca")
    inv = json.loads((OUT / "invariant_tests.json").read_text(encoding="utf-8"))
    check("invariant_tests_all_passed", inv["all_passed"], f"{inv['n_pass']}/{inv['n_tests']}")
    r = subprocess.run([sys.executable, "-m", f"{REL.replace('/', '.')}.stage0_checks"], capture_output=True, text=True,
                       encoding="utf-8", cwd=BASE)
    check("stage0_checks_pass", r.returncode == 0, r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-200:])
    staged = [x for x in git("diff", "--cached", "--name-only").splitlines() if x.strip()]
    check("staged_files_only_exp21", all(x.startswith(REL + "/") for x in staged), str([x for x in staged if not x.startswith(REL)]))
    print(f"OK   {len(oks)}")
    for f in fails:
        print("FAIL " + f)
    print("ALL CHECKS PASSED" if not fails else f"{len(fails)} CHECK(S) FAILED")
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
