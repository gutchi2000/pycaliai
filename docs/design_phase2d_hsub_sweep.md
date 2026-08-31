# PyCaLiAI Design System — Mobile `.hsub` Font-Size Sweep (measurement only)

> Measurement only. `site/css/style.css` was temporarily edited 5 times during this sweep and has been **restored to the accepted pre-Phase-2D state** (`10px`) afterward.
> **Status: accepted as final.** 10px is the confirmed, kept value — no further `.hsub` change pending. The §3 DESIGN.md wording proposed below was accepted (in generalized form, not naming `.hsub`/`10px`) and is now live in DESIGN.md §6.
> Date: 2026-08-31

---

## 1. Measurement table

Same race (札幌1R, 14 horses), same 14 rows, mobile viewport (`innerWidth: 375` confirmed at every step), fresh cache-busted stylesheet + forced row re-render for each candidate (methodology validated by re-measuring 12px and getting numbers identical to the independent measurement from the prior turn — 9/14 clipped, same rows, same overflow amounts).

| `.hsub` size | Clipped rows | Clipped % | `.hsub` height | Row height | `.hsub` line-height | Worst-case overflow |
|---|---|---|---|---|---|---|
| **10px** | **0 / 14** | **0%** | 18.67px | 53.05px | 15.5px | none |
| 10.5px | 8 / 14 | 57.1% | 18.67px | 53.05px | 16.275px | 7px (row 0, "横山和生(替)") |
| 11px | 8 / 14 | 57.1% | 19.67px | 54.05px | 17.05px | 7px (row 0) |
| 11.5px | 9 / 14 | 64.3% | 19.17px | 53.55px | 17.825px | 17px (row 0) |
| 12px | 9 / 14 | 64.3% | 20.17px | 54.55px | 18.6px | 17px (row 0) |

### Longest-string behavior (row 0, `タガノシルフィー`, text `"S81逃げ牝3 55.0k 横山和生(替)"` — the "(替)" jockey-change suffix is what pushes this line over)

| Size | Rendered |
|---|---|
| 10px | Full text visible, no truncation |
| 10.5px–12px | Truncates to `"S81逃げ牝3 55.0…"` — the jockey name and the "(替)" change-marker are both cut off |

### Clipped-row set is stable, not gradually growing

The same 7–8 rows clip at every tested size above 10px (all the ones with a "(替)" suffix or an unusually long jockey name); one additional row (`i:13`, `コイバナ`) joins the clipped set at 11.5px and stays clipped at 12px. There is no gentle, gradually-increasing-overflow zone between 10px and 10.5px — it's a hard threshold: 0% at 10px, 57.1% at the very next tested increment.

---

## 2. Recommended value

**10px — retain the original value.** Per the decision rule ("choose the largest size that produces zero clipped rows... if no value above 10px produces zero clipping, retain 10px"): none of the four tested values above 10px (10.5, 11, 11.5, 12) produce zero clipping, so the rule resolves unambiguously to keeping 10px. `site/css/style.css` has already been restored to this state — see §4/§5.

## Why the result looks like this

The jump from 0% to 57.1% clipping between 10px and 10.5px is steep because the binding constraint isn't the general legibility of the text — it's one specific, short substring: the "(替)" jockey-change marker that `app.js` appends to a jockey's name when app.js's `subLine()` detects `h.kawari`. At 10px, the combined string (level chip + running-style tag + sex/age + weight + jockey name + "(替)") fits inside the existing `167px` mobile hsub column *exactly*, with no slack. Any size increase, even half a pixel, removes the last bit of headroom for every row that happens to include that suffix — which in this particular race is more than half the field (jockey changes are common in JRA on the day a card is finalized). This isn't really a "10px vs 12px" legibility question so much as a "does this specific column width have any spare capacity" question, and measured, it does not.

---

## 3. DESIGN.md clarification for dense secondary race metadata (accepted, now live in DESIGN.md §6)

The existing intermediate-size clause (§6, added in Phase 0.6) already licenses sizes *above* the table/data-dense scale for hierarchy reasons (horse name, primary numbers). It doesn't address the opposite case — a value *below* the 12px floor that's necessary because the alternative is silently cutting off data. Proposed addition, immediately after that existing clause in §6:

> Secondary metadata lines that combine several dense fields on one row (e.g. running-style tag, sex/age, weight, and jockey name, including a jockey-change marker) may render below the table/data-dense floor on narrow viewports, when the alternative is silently truncating that data. Prefer the largest size — verified against real content, not assumed — that produces no clipping for the field in question. Do not default to the nominal floor if it demonstrably cuts off information; a legible smaller size beats a truncated larger one.

This is deliberately worded to require *measurement* (as this sweep did) rather than hard-coding "10px" into the design system doc — if the underlying content changes (e.g., the "(替)" suffix logic changes, or the column gets more room from an unrelated future change), the right floor could change too, and the doc shouldn't need editing just because a number moved. **Accepted and applied to DESIGN.md §6** in generalized form (no mention of `.hsub` or `10px` in the doc itself, per instructions) as a follow-up turn to this sweep.

---

## 4. `git diff -- DESIGN.md`

```diff
diff --git a/DESIGN.md b/DESIGN.md
index 3db5083460..25f6080fdd 100644
--- a/DESIGN.md
+++ b/DESIGN.md
@@ -371,6 +371,8 @@ sticky header
 subtle row separators (navy, §9)
 40–44px row height
 
+40–44px is the preferred row height for single-line dense data rows. Compound race rows containing a primary horse-name line plus secondary metadata may exceed this range when required for readability and hierarchy. Do not compress multi-line race rows solely to satisfy the nominal 40–44px target.
+
 Horse name is primary text.
 
 Secondary metadata:
```
Only the prior turn's row-height clarification — kept intact, untouched this turn, and the proposed §3 wording above was **not** added.

## 5. `git diff -- site/css/style.css`

```
(no output — file is identical to HEAD)
```
Confirms the 5 temporary sweep edits (10 → 10.5 → 11 → 11.5 → 12 → back to 10px) leave no net diff. The accepted pre-Phase-2D state is restored exactly.

## 6. `git status --porcelain -- site/`

```
(empty)
```

Nothing pending under `site/`. `DESIGN.md` now carries both the row-height clarification from the prior turn and the dense-secondary-metadata exception from §3 above — see `docs/design_phase2d_implementation_report.md`'s "Final Outcome" section for the complete record, and the commit that includes both DESIGN.md and this document.
