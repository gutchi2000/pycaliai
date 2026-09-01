# PyCaLiAI Phase 6B — Model Reliability / Calibration Implementation Report

> Implements 6B-0 through 6B-2 and 6B-5. 6B-3 (horse-level reliability popover) deferred — see §16.
> **Extended post-review — see §18.** Your review required this to be framed explicitly as a *new* evaluation scope (not a re-display of the existing ◎/○-only ECE figures already cited in CLAUDE.md/audit_marks.py) and required five specific QA checks before commit; both are addressed in §18. Committed as an isolated `design: Phase 6B Model Reliability / calibration view` commit, after Phase 5C-3 and Phase 6A per your explicit ordering.
> Date: 2026-09-01

---

## 1. Purpose

Answer "when PyCaLiAI says WIN 40%, do those horses actually win about 40% of the time?" — as deeper, secondary evidence alongside the existing HONEST RECORD section, not a replacement for it.

## 2. Exact current data source

**6B-0 audit result**: this project already has a precisely-defined, already-used calibration metric — `expected_calibration_error()` in `audit_marks.py` (10 equal-width bins spanning 0–100%, weighted mean absolute error between each bin's average prediction and its actual hit rate). It's the exact metric cited in CLAUDE.md's own v6-production justification ("ECE 複勝(◎) -32%"). I reused this definition **verbatim** rather than inventing a new one or a new bin scheme — satisfying 6B-0/6B-2's explicit "do not silently choose bins."

One real scope decision, made explicitly rather than silently: `audit_marks.py`'s own ECE calls are scoped to just the ◎/〇 mark slots (that script audits mark quality specifically). I extended the *data slice* — not the metric — to **every horse in every settled race**, which is what a reliability diagram spanning the full 0–100% range needs (◎/〇 alone cluster in a narrow band and would leave most bins empty). Also considered and **rejected** `data/live_results_2026.csv` as a source — inspection showed it carries raw win-odds and legacy rule-based-strategy columns (`HAHO_`/`HALO_`/`LALO_`/`CQC_`/`STANDARD_` prefixes), i.e. exactly the kind of internal/restricted row this phase must not read from.

Actual source used: the already-public `site/data/{date}.json` files (same ones the site itself serves) — `horse.p_win`, `horse.p_sho`, cross-referenced against the same file's own `race.result.order`.

## 3. New data/storage introduced

- **`build_calibration.py`** (new, root script) — scans every `site/data/2026*.json`, keeps only dates from `2026-05-17` onward (the confirmed first v6 week — verified precisely by reading the `model` field out of all 38 `reports/cowork_input/*_bundle.json` files, not guessed from CLAUDE.md's prose date), and for every settled race's every horse, records `(p_win, won?)` and `(p_sho, top3?)`. Applies the copied-verbatim `expected_calibration_error()` to each set, writes `site/data/calibration.json`.
- **Generated data**: ran for real — **12,117 horses, 895 races, 27 dates (2026-05-17 → 2026-08-16), model v6 only.** File lives under `site/data/`, ships with the existing static-site deploy automatically (same reasoning as Phase 6A's ledger — no Dockerfile change needed).
- **Not wired into weekly automation** — same open question as Phase 6A's script; natural hook point is once per Phase C (after results land), not weekly.

## 4. Exact changed files

```
build_calibration.py         new, 128 lines
site/data/calibration.json   new, generated (1 file, ~4KB)
site/js/app.js               modified
site/css/style.css           modified
site/index.html              modified (cache-busting only)
```

## 5. Exact functions/selectors

**`app.js`**: `calibrationData` (new module-level cache var, mirrors `resultsData`'s pattern). `renderResults()` — added one fetch block for `data/calibration.json` (same try/catch-to-null shape as every other optional per-load fetch in this file) and one call to `calibrationSectionHtml()` appended after the existing `#hitGrid`. New functions: `reliabilitySvg(bins, color)` (hand-built inline SVG — no ECharts, no new chart library, matching "the data itself should be the visual"), `calibrationBinRows(bins)`, `calibrationSectionHtml()`.

**`style.css`**: new rules `.cal-card`, `.cal-sum` (+ `::before`/`details[open]` disclosure-triangle treatment, copied from the same pattern already used in `explain.html`'s `.ctitle`), `.cal-body`, `.cal-scope`, `.cal-grid`, `.cal-col`, `.cal-h`, `.cal-svg`, `.cal-diag`, `.cal-axis`, `.cal-tick`, `.cal-tbl`, `.cal-note`, plus one mobile override (`.cal-grid` → single column at ≤880px). All colors/spacing reuse existing tokens; the WIN chart uses the same gold (`#f5b942`) already used for ◎-related elements, TOP3 uses the same teal (`#2dd4a8`) already used for 複勝/positive elements elsewhere — no new color introduced.

## 6. UI entry point

A collapsed-by-default `<details>` section ("MODEL RELIABILITY") appended at the very end of the existing Results view, below the hit-card grid — not a new top-level tab, not expanded by default, per "tabs/expandable sections, not permanently expanded panels."

## 7. Desktop behavior

Verified live at 1440×900: section renders collapsed initially (confirmed `details.open === false`); clicking the summary expands it to show the evaluation-scope line, then two side-by-side columns (WIN / TOP3), each with a small reliability scatter (dashed diagonal reference, dot size ∝ √n) and its full bin table beneath. All values cross-checked exactly against the Python script's own printed output (§10).

## 8. Mobile behavior

Verified live at 375×812 (screenshot captured, see §13): the two-column layout correctly stacks to one column; both charts render cleanly with no overflow; the WIN chart's 6 dots and the TOP3 chart's 9 dots are both clearly visible, sized proportionally to sample count, tracking close to the diagonal.

## 9. Public-data/compliance audit

| Field | Source | PyCaLiAI-generated? | Upstream raw value? | Already public? |
|---|---|---|---|---|
| `p_win`, `p_sho` (per horse) | `site/data/{date}.json` | yes (model output) | no | yes |
| `race.result.order` | same | no (public fact) | no | yes (already shown in Results/header) |
| bin aggregates (`mean_pred`, `actual_rate`, `n`) | computed from the above | yes (derived) | no | new, but built entirely from already-public inputs |
| `model_version`, date range | derived from `reports/cowork_input/*_bundle.json`'s own `model` field | yes | no | yes (site footer already shows model tag) |

No odds, no `vs_market`, no `ev_tan`, no training data, no raw upstream row read at any point — confirmed both by the script's own source list (only `site/data/*.json`, nothing from `reports/cowork_output/`, `reports/live_odds/`, or `data/live_results_2026.csv`) and by direct inspection of the generated `calibration.json` (only `bin_lo`/`bin_hi`/`n`/`mean_pred`/`actual_rate`/`ece`/scope metadata — no other keys).

## 10. Numerical/reproducibility verification

- Script run twice (`--dry` then real) — identical printed numbers both times (deterministic, no randomness).
- Every value shown in the UI cross-checked byte-for-byte against the script's own console output: WIN ECE 0.77pt (n=12,117), TOP3 ECE 1.29pt (n=12,117), first three WIN bins "0–10%: pred 4.6%/obs 4.2%/n=9,078", "10–20%: pred 13.2%/obs 15.3%/n=2,619", "20–30%: pred 25.4%/obs 25.7%/n=335" — all exact matches.
- **Reproducibility note**: re-running `python build_calibration.py` at any time will reproduce the exact same output as long as `site/data/*.json`'s contents are unchanged (pure function of already-on-disk files, no external state, no randomness). The v6-only date filter (`>= 20260517`) is a literal constant at the top of the script, sourced from the bundle-scan described in §2/§3, not hardcoded from memory.
- 6B verification checklist:

| Check | Status |
|---|---|
| Forecast rows joined to correct results | PASS — joined by `umaban` within the same race object, same file, no cross-file join risk |
| No post-race leakage | PASS — `p_win`/`p_sho` are the model's own pre-race output fields; nothing here reads or is influenced by `result` when computing the probabilities themselves |
| WIN and TOP3 calculated independently | PASS — two separate `expected_calibration_error()` calls, two separate arrays |
| Sample counts reconcile with source rows | PASS — `n_horses_win`/`n_horses_sho` both equal 12,117, matching the sum of all non-empty bin `n` values in each set |
| Excluded races follow existing Results inclusion rules | Partial — this feature does not use `results.json`'s own inclusion/exclusion rules (e.g. `excluded_btypes`) at all, since it measures *prediction* accuracy, not *betting* P&L; a scratched horse with no recorded finish position is skipped (its `order` lookup returns `None`) |
| Prediction bins reproduce exact counts | PASS — verified via the dry-run/real-run comparison above |
| Mobile chart/table does not overflow | PASS — confirmed via screenshot |
| Numerical calculations independently spot-checked | PASS — see the byte-for-byte comparison above |

## 11. Regression testing

`renderResults()`'s existing rendering (summary stats, by-type breakdown, monthly table, hit-card grid, filters) is unchanged — only one fetch block and one appended section were added, no existing line was modified beyond that.

## 12. Console/runtime status

Clean on a genuinely fresh tab (confirmed). **One thing investigated and ruled out**: a *reused* tab that had earlier opened a Horse Drawer (career-index ECharts chart) in this same session showed repeated ECharts `dataIndex` errors when Results/calibration was later opened on that same tab. A fresh tab reproduces none of this — confirming it's a pre-existing dangling-chart-instance issue in the drawer's own ECharts lifecycle (`drawCareerChart()`'s instance isn't disposed on navigating away via the mode-nav, only on the next `openDrawer()` call), unrelated to and not introduced by this phase. Not fixed here — out of this phase's change surface (drawer internals are explicitly untouched per the Phase 6 brief) — flagged for awareness only.

## 13. Screenshots

Mobile screenshot (375×812) captured showing the expanded section: header, scope line, WIN chart (gold, 6 points, diagonal-hugging), full bin table, TOP3 chart (teal, 9 points) beginning below. Desktop confirmed via DOM extraction rather than an image (same known deep-scroll rendering limitation as prior phases, given this section sits below 447 hit-cards) — values cross-checked exactly against the mobile screenshot and the Python output.

## 14. `git diff --stat` — resolved (see §18)

The commit-sequencing question is resolved: split into three isolated commits (5C-3 → 6A → 6B) via hunk-level reconstruction, verified byte-for-byte against the final working tree before each commit. This phase's isolated diff, as actually committed:

```
$ git diff --stat HEAD~1..HEAD
 build_calibration.py                              | 217 (new file)
 docs/phase6b_calibration_implementation_report.md | 132 (new file)
 site/css/style.css                                |  30 +++
 site/data/calibration.json                        | 168 (new file)
 site/index.html                                   |   4 +-
 site/js/app.js                                     |  77 +++++++++
 6 files changed, 626 insertions(+), 2 deletions(-)
```

## 15. `git status --porcelain`

Committed as `design: Phase 6B Model Reliability / calibration view`. See §18 for the full commit list and verification.

## 16. Known limitations

- **6B-3 (horse-level reliability popover) deferred, not implemented.** The brief says "only implement if it can reuse the same calibration dataset cleanly" — cleanly reusing it per-horse would mean, for a given displayed WIN%, looking up which bin it falls in and showing that bin's aggregate stats. This is straightforward to *compute* but I did not want to add a new interactive affordance (tooltip/popover) to the Command Header or shutsuba table in the same pass as everything else already shipped this turn, given the brief's own repeated caution against cluttering the header/first-viewport. Flagged as a small, well-scoped follow-up rather than built speculatively.
- Same weekly-automation-wiring question as Phase 6A: `build_calibration.py` must currently be run manually. A hook location has been proposed and reported separately for approval (see §18) — notably, its *actual* dependency turned out not to be `generate_results.py` (which never touches `site/data/`) but `build_site.py`'s own re-run once kekka lands, so the natural hook is the same script as Phase 6A's (`sync-hf-umami.ps1`), not `weekly_post.ps1`.
- Coverage is naturally v6-only (27 dates) — will grow every week going forward as more settled race days accumulate; no code change needed for that growth, just re-running the script.

## 17. PASS/FAIL

**PASS.** No existing feature touched beyond appending to `renderResults()`'s output. No new metric invented — the ECE definition and bin scheme are copied verbatim from the project's own existing, already-relied-upon audit tool. Sample size (`n`) is shown for every single bin, in both the table and via dot size in the chart, with zero-sample bins simply omitted rather than shown as a misleading zero. No compliance issue found in the generated public JSON.

**Per the brief's own "stop and verify" instruction, stopping here — Phase 6C (WHY ◎ > ○) has not been started.**

## 18. Addendum — new evaluation scope framing + final QA (post-review)

Your review required two things before commit, both addressed here rather than left as prose-only claims:

**1. Explicit "new evaluation scope" framing.** The population-scope decision was already explained in §2 above, but you required it stated as a formal, unmistakable distinction from the ◎/○-only ECE numbers already cited elsewhere in this project (CLAUDE.md, `audit_marks.py`'s own output) — not merely implied. Added:
- `build_calibration.py`'s docstring now has a dedicated section spelling out exactly what's reused unchanged (metric formula, 10-bin scheme) versus what changed (population: ◎/○ slots → every horse in every settled v6 race), and states outright that the resulting ECE values are **not comparable to** and **must never be described as** the same figure as CLAUDE.md's "複勝(◎) ECE -32%" line.
- `calibration.json` itself now carries an explicit `evaluation_scope` object (`description`, `population`, `metric`, `differs_from`, `win_and_top3_evaluated_independently`) — so a reader of the JSON alone, not just the docstring, can't mistake this for the historical ◎/○ evaluation.
- The frontend scope line now carries a `新規の評価範囲` badge, and the disclosure note explicitly tells the reader not to directly compare these numbers against the ◎ hit-rate figures shown earlier on the same page.
- Live-verified on desktop: badge renders inline next to the scope line; numbers (WIN ECE 0.77pt, TOP3 ECE 1.29pt, n=12,117 both) are byte-identical to the pre-rework values, confirming this was a pure labeling/metadata addition with no change to the underlying computation.

**2. Five-point final QA, verified rather than assumed:**

| # | Check | How verified |
|---|---|---|
| 1 | Non-starters/scratches not scored as losses | Traced upstream to `build_site.py`'s `order` dict construction: a row's `着順` is parsed via `_int()`, which returns `None` for any non-numeric code (取消/除外/中止); the row is skipped *before* `r["order"][str(uma)]` is ever set. So a scratched horse simply has no key in `order` — `build_calibration.py`'s `order.get(str(uma)) is None → continue` guard excludes it, not counts it as a loss. Comment added at the guard site documenting this traced guarantee. |
| 2 | `p_sho`/TOP3 definition consistent across the full evaluation set | Single definition, one place: `1 if int(pos) <= 3 else 0`, applied uniformly in `collect()`'s one loop — no per-race or per-date branching. |
| 3 | Displayed sample counts reconcile exactly with included source rows | Added two runtime `assert`s (not just eyeballing): `sum(bin.n) == n_horses_win` and `== n_horses_sho`. Confirmed passing on the real re-run: bins sum to 12,117 on both sides, exact match. |
| 4 | Evaluation period + model-version scope recorded | Already present via `generated_at_scope` (`model_version`, `date_from`, `date_to`, `n_dates`, `n_races`); now also echoed in the new `evaluation_scope` block (item 1 above). |
| 5 | WIN and TOP3 evaluated independently | Two separate arrays (`win_p`/`win_y` vs `sho_p`/`sho_y`), two separate `expected_calibration_error()` calls. Added an `assert len(win_p) == len(sho_p)` to confirm they're drawn from the identical population (every append happens in the same loop iteration) even though computed independently. |

All four assertions pass on the real data; the script also now prints a `[6B QA] ...: PASS` line for each of the four checks above, so this isn't a one-time claim — it's re-verified every time the script runs. Re-ran for real after these changes: identical WIN/TOP3 ECE and bin values to the pre-rework run (0.0077 / 0.0129), confirming this was additive verification, not a computation change.

**Automation hook — proposed, not implemented.** See Phase 6A's §18 for the full reasoning: `build_calibration.py`'s actual data dependency is `site/data/{date}.json`'s `result.order`, populated by `build_site.py` — not `data/results.json`/`data/cowork_results.json` (what `generate_results.py` writes, and what "the Results dataset" most naturally refers to). `generate_results.py` never touches `site/data/`. So the technically-correct hook is the same script as 6A's (`sync-hf-umami.ps1`, right after its `build_site.py` call), gated to a `-RebuildCalibration` switch passed only from `weekly_nicegui.ps1`'s `-Post` phase (since only by then has `build_site.py` picked up the week's newly-settled kekka). This is flagged explicitly because it diverges from the literal instruction to hook this "after settled results feeding the Results dataset have been updated" (i.e. after `generate_results.py`) — that location doesn't actually work, since `site/data/{date}.json` doesn't yet have the new results at that point in `weekly_post.ps1`. Not yet implemented — awaiting review.

**Commit**: landed as its own isolated commit, `design: Phase 6B Model Reliability / calibration view`, positioned after Phase 6A per your explicit ordering.
