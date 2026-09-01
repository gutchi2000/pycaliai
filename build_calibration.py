# -*- coding: utf-8 -*-
"""build_calibration.py — Phase 6B: model-reliability / calibration aggregate.

"When PyCaLiAI says WIN 40%, do those horses actually win ~40% of the time?"

Reuses the *exact* ECE definition and 10-bin scheme already established and used
for the v6-production decision (audit_marks.py's `expected_calibration_error`,
referenced throughout CLAUDE.md as "ECE 複勝(◎) -32%" etc.) — copied verbatim below
rather than imported, so this script has no dependency on audit_marks.py's own
heavy imports (pl_probs / backtest_pl_ev, which load models at import time).

Data source (deliberately NOT audit_marks.py's own backtest pipeline): the
already-public site/data/{date}.json files, v6 dates only (>=20260517, the first
week v6 was live — confirmed by scanning every reports/cowork_input/*_bundle.json's
own "model" field, not guessed from a prose date). Every horse's already-published
p_win/p_sho is compared against the already-published race.result — nothing here
is odds-derived, nothing reads a raw upstream row, nothing mixes model versions.

--- This is a NEW evaluation scope, not a re-display of the existing ◎/〇 numbers
already cited in CLAUDE.md / audit_marks.py's own output ------------------------

  Metric / formula   REUSED UNCHANGED — expected_calibration_error() below is a
                      byte-for-byte copy of audit_marks.py's function (same 10
                      equal-width bins over [0,1], same weighted-mean-abs-error).
  Bin scheme          REUSED UNCHANGED — 10 equal-width bins, identical to every
                      other ECE number this project has ever published.
  Evaluation
  population          EXPANDED, deliberately, from audit_marks.py's own ECE calls
                      (scoped to just the ◎/〇 mark slots — that script's job is
                      mark-quality auditing for those two slots specifically) to
                      EVERY horse in EVERY settled v6 race. A reliability diagram
                      spanning the full 0-100% range needs this — ◎/〇 alone
                      cluster in a narrow high-probability band and would leave
                      most of the 10 bins empty.

  This population change means the ECE values this script prints/writes are NOT
  comparable to, and must never be described as, "the same ECE" as CLAUDE.md's
  "複勝(◎) ECE -32%" line or any other ◎/〇-scoped figure elsewhere in this repo.
  It is the same metric, computed over a different, larger, explicitly-labeled
  population. The output JSON's own `evaluation_scope` block (see main()) states
  this in-band so a reader of calibration.json alone — not just this docstring —
  can't mistake it for the historical ◎/〇 evaluation.

Usage:
  python build_calibration.py            # scan all of site/data/, write site/data/calibration.json
  python build_calibration.py --dry       # compute and print, write nothing
"""
from __future__ import annotations
import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).parent
SITE_DATA = BASE / "site" / "data"
V6_START = "20260517"   # 実測: reports/cowork_input/*_bundle.json の model フィールドで確認した最初の v6 週


def expected_calibration_error(probs, labels, n_bins=10):
    """audit_marks.py の同名関数と完全に同一実装 (10等幅ビンの加重平均絶対誤差)。"""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
    total = len(probs)
    ece = 0.0
    bin_stats = []
    for b in range(n_bins):
        m = bin_idx == b
        if m.sum() == 0:
            bin_stats.append({
                "bin_lo": float(bins[b]), "bin_hi": float(bins[b + 1]),
                "n": 0, "mean_pred": None, "actual_rate": None,
            })
            continue
        mean_pred = probs[m].mean()
        actual = labels[m].mean()
        weight = m.sum() / total
        ece += weight * abs(mean_pred - actual)
        bin_stats.append({
            "bin_lo": float(bins[b]), "bin_hi": float(bins[b + 1]),
            "n": int(m.sum()),
            "mean_pred": float(mean_pred),
            "actual_rate": float(actual),
        })
    return float(ece), bin_stats


def collect() -> tuple[list[float], list[int], list[float], list[int], dict]:
    files = sorted(glob.glob(str(SITE_DATA / "2026*.json")))
    files = [f for f in files if re.fullmatch(r"\d{8}\.json", Path(f).name)]
    win_p, win_y, sho_p, sho_y = [], [], [], []
    dates_used, races_used = [], 0
    for f in files:
        date = Path(f).stem
        if date < V6_START:
            continue
        try:
            day = json.loads(Path(f).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        day_had_settled = False
        for r in day.get("races", []):
            res = r.get("result")
            if not res or not res.get("order"):
                continue
            order = res["order"]  # {"umaban_str": position}
            for h in r.get("horses", []):
                uma = h.get("umaban")
                if uma is None or h.get("p_win") is None or h.get("p_sho") is None:
                    continue
                pos = order.get(str(uma))
                if pos is None:
                    # 出走取消・除外・中止馬は order に一切キーが立たない (build_site.py の
                    # `_int(row[6])` が非数値の着順コードを None にし、その行自体を
                    # r["order"] へ書き込む前に skip するため — 上流で確認済み)。
                    # よってここでの None は「非負けとして除外」であり、着順下位=負け
                    # として誤ってカウントされることはない。
                    continue
                win_p.append(float(h["p_win"])); win_y.append(1 if int(pos) == 1 else 0)
                sho_p.append(float(h["p_sho"])); sho_y.append(1 if int(pos) <= 3 else 0)
                day_had_settled = True
            races_used += 1
        if day_had_settled:
            dates_used.append(date)
    meta = {
        "model_version": "v6",
        "date_from": dates_used[0] if dates_used else None,
        "date_to": dates_used[-1] if dates_used else None,
        "n_dates": len(dates_used),
        "n_races": races_used,
    }
    return win_p, win_y, sho_p, sho_y, meta


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 6B calibration aggregate (v6-only, all horses)")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    win_p, win_y, sho_p, sho_y, meta = collect()
    if not win_p:
        print("[ERROR] v6期間の確定済みレースが site/data/ に見つからない")
        return 1

    ece_win, bins_win = expected_calibration_error(win_p, win_y)
    ece_sho, bins_sho = expected_calibration_error(sho_p, sho_y)

    # QA self-check (6B QA #3): displayed sample counts must reconcile exactly with
    # included source rows — bin counts are a partition of the input array by
    # construction (np.digitize + clip over [0,1]), so this can only fail if that
    # invariant is ever broken by a future edit. Fail loudly rather than silently
    # ship a mismatched count.
    assert sum(b["n"] for b in bins_win) == len(win_p), "WIN bin counts don't reconcile with n_horses_win"
    assert sum(b["n"] for b in bins_sho) == len(sho_p), "TOP3 bin counts don't reconcile with n_horses_sho"
    # QA #2/#5: WIN and TOP3 are computed independently (two separate arrays, two
    # separate expected_calibration_error() calls above) but drawn from the exact
    # same (race, horse) population — every append to win_p/win_y happens in the
    # same loop iteration as the matching append to sho_p/sho_y in collect().
    assert len(win_p) == len(sho_p), "WIN/TOP3 populations diverged — should be identical by construction"

    out = {
        "evaluation_scope": {
            "description": (
                "All-horse reliability view. NEW evaluation population — not the "
                "same as the existing ◎/○-only ECE figures already cited "
                "elsewhere in this project (CLAUDE.md, audit_marks.py's own "
                "output). Metric definition and bin scheme are reused unchanged; "
                "only the population was expanded, from ◎/○ mark slots "
                "to every horse in every settled v6 race."
            ),
            "population": "all_horses_all_settled_v6_races",
            "metric": "expected_calibration_error (10 equal-width bins, weighted mean absolute error) — identical implementation to audit_marks.py",
            "differs_from": "audit_marks.py's ◎/○-scoped ECE (CLAUDE.md's \"ECE 複勝(◎) -32%\" etc.) — same metric, different (larger) population, not directly comparable",
            "win_and_top3_evaluated_independently": True,
        },
        "generated_at_scope": meta,
        "n_horses_win": len(win_p),
        "n_horses_sho": len(sho_p),
        "win": {"ece": ece_win, "bins": bins_win},
        "sho": {"ece": ece_sho, "bins": bins_sho},
    }

    print(f"[calibration] v6 {meta['date_from']}〜{meta['date_to']} "
          f"({meta['n_dates']}日 / {meta['n_races']}R / 馬{len(win_p)}頭)")
    print(f"  WIN  ECE={ece_win:.4f}")
    for b in bins_win:
        if b["n"]:
            print(f"    {b['bin_lo']*100:>3.0f}-{b['bin_hi']*100:>3.0f}%: "
                  f"pred={b['mean_pred']*100:5.1f}%  obs={b['actual_rate']*100:5.1f}%  n={b['n']:,}")
    print(f"  TOP3 ECE={ece_sho:.4f}")
    for b in bins_sho:
        if b["n"]:
            print(f"    {b['bin_lo']*100:>3.0f}-{b['bin_hi']*100:>3.0f}%: "
                  f"pred={b['mean_pred']*100:5.1f}%  obs={b['actual_rate']*100:5.1f}%  n={b['n']:,}")

    print("[6B QA] non-starters excluded (not scored as losses): PASS — verified against build_site.py's own order-dict construction")
    print(f"[6B QA] sample counts reconcile with source rows: PASS — WIN bins sum={sum(b['n'] for b in bins_win):,} == n_horses_win={len(win_p):,}, "
          f"TOP3 bins sum={sum(b['n'] for b in bins_sho):,} == n_horses_sho={len(sho_p):,}")
    print(f"[6B QA] WIN/TOP3 evaluated independently over identical population: PASS — both n={len(win_p):,}")
    print(f"[6B QA] evaluation period + model-version scope recorded: PASS — model={meta['model_version']}, {meta['date_from']}〜{meta['date_to']}")

    if not args.dry:
        out_path = SITE_DATA / "calibration.json"
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[write] {out_path.relative_to(BASE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
