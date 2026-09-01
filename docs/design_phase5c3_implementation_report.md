# PyCaLiAI Design System — Phase 5C-3: Compact Command Header Recomposition

> Implementation. Not committed — left for review, per instructions.
> Uses the pre-Phase-5B-2 header's spatial structure (two-column: identity+prediction on the left, gauges upper-right) as the baseline, with the Phase 5B-2/5C-1 ◎/WIN/TOP3 headline and state line folded into the left column instead of stacked as separate full-width rows.
> Date: 2026-09-01

---

## 1. Exact changed files

```
site/css/style.css   39 lines changed
site/index.html       4 lines changed (cache-busting only)
site/js/app.js        16 lines changed
```

No other file touched. Drawer, cache/service-worker work, race table, prediction values, model/data/API, homepage, results, and explain.html are all untouched — confirmed via `git diff --stat` scope (§13).

## 2. Exact changed functions/selectors

**`site/js/app.js`**, `renderHeader()`: `cmdPickHtml(honmei)` and `cmdStateHtml(...)` calls, and the conditional `.judge` block, moved from being direct children of `<div class="card rh">` to being children of `<div class="rh-main">`, inserted immediately after `.rh-sub` and before `.rh-main`'s closing tag. `cmdPickHtml()`/`cmdStateHtml()` themselves are unchanged — same fields, same logic, only their call *site* in the template moved.

**`site/css/style.css`**:
- `.cmd-pick`: `flex-basis:100%` removed (no longer a full-width row); `gap` 14px→10px; `margin-top` 14px→9px.
- `.cmd-pick .mark.m1`: 30px→22px.
- `.cmd-name`: 23px→17px.
- `.cmd-stats`: `margin-left:auto` **removed** (this was the cause of the WIN/TOP3 group being pushed to the far right edge); `gap` 22px→14px.
- `.cmd-stat`: layout changed from `flex-direction:column;align-items:flex-end` (number stacked over label) to `display:flex;align-items:baseline;gap:3px` (number and label side-by-side on one line) — shorter vertically, and reads as "21.1% WIN" inline rather than a two-line block.
- `.cmd-stat b`: 28px→20px. `.cmd-stat b small`: 15px→12px. `.cmd-lab`: 11px→10px, no longer needs its own `margin-top` (same line now).
- `.cmd-state`: `flex-basis:100%` removed; the `border-bottom`/`padding-bottom` divider added in Phase 5C-1 **removed** (no longer needed — see §3); `margin` 8px→5px top, `font-size` 13px→12.5px.
- `.judge`: `flex-basis:100%` removed; `padding-top`/`border-top` (added in 5B-2) already gone since 5C-1; `margin-top` unchanged at 9px (matches the pre-5B-2 original exactly).
- `.gauges`: `margin-top:14px` (5B-2) → **removed**, `margin-left:auto` **restored** — this is the exact pre-5B-2 rule, byte-for-byte.
- `@media (max-width: 880px)`: the four `cmd-*` mobile-size overrides (added in 5B-2 for the then-larger desktop sizes) rebalanced to be smaller than the new, already-smaller desktop values (mark 25→19px, name 19→15px, stats-gap 16→10px, stat-b 24→18px, stat-b-small 13→11px). No new mobile rule was needed for gauges — the pre-existing `.gauges{margin-left:0;flex-basis:100%}` override at this same breakpoint (untouched since before Phase 5B-2) automatically applies again now that gauges sit in the same flex row as `.rh-main` on desktop.

## 3. Structural before → after

**Before (Phase 5B-2/5C-1)**: `<div class="card rh">` contained, as **five separate full-width flex children**: `.rh-main` (title+sub only) → `.cmd-pick` → `.cmd-state` → `.judge` (conditional) → `.gauges`, each stacking vertically, each claiming the full card width even though most of their content didn't need it.

**After (this phase)**: `<div class="card rh">` contains **two side-by-side children**, exactly like the original pre-5B-2 header: `.rh-main` (now: title, sub, cmd-pick, cmd-state, judge — all stacked *within* this one column) and `.gauges` (pushed to the upper-right via `margin-left:auto`, sharing the same row as `.rh-main`). The divider introduced in Phase 5C-1 (to separate the "new" section from the "old" section) was removed — with everything unified into one column, a horizontal rule between sub-elements read as clutter rather than a meaningful section break.

## 4. Header height before/after

Measured live, same race (`2026083001020401`, 札幌1R, pre-result), same 1440×900 viewport, `.card.rh`'s full `getBoundingClientRect().height` (includes member-level + prior-strip, not just the top row):

| Version | Height |
|---|---|
| Old/pre-5B-2 baseline | 507.28px |
| Phase 5B-2/5C-1 (tall) | 737.91px |
| **This phase (hybrid)** | **529.41px** |

The hybrid is **208.5px shorter than the tall version** and only **22.1px taller than the original old header** — for a header that now also states the ◎ pick's WIN/TOP3 probabilities and race-state line, which the old header never showed at all.

## 5. Table start Y before/after

Same race, same measurement point (`.sh-head`'s `top`):

| Version | Desktop (1440×900) | Mobile (375×812) |
|---|---|---|
| Old/pre-5B-2 baseline | 762.56px | 1171.58px |
| Phase 5B-2/5C-1 (tall) | 993.19px | 1329.27px |
| **This phase (hybrid)** | **784.69px** | **1190.41px** |

Desktop: 208.5px earlier than the tall version, 22.1px later than the original (same delta as the header-height difference, as expected). Mobile: 138.9px earlier than the tall version, 18.8px later than the original.

## 6. Desktop visual comparison

Three references captured at the same 1440×900 viewport, same race, same live data (screenshots taken this session, not reproduced here as images but described precisely):

- **Old/current production reference** (extracted from commit `f43a609dc7`, the state immediately before Phase 5B-2): race identity → bare "—"/荒れ row (the placeholder bug this project fixed in 5C-1, present here because this is the historical baseline) → gauges upper-right (`x=949.5, y=216.3`) → member-level card → prior-strip → tabs → table.
- **Phase 5B-2/5C-1 (tall)**: race identity → ◎ headline with WIN/TOP3 spread across the *entire* card width (`.cmd-pick` measured `width:1130px`, right edge at `x=1277.5` — i.e., the numbers sit at the far right of the card, ~700+px from the horse name) → state line → judge (when present) → gauges now moved to the **left**, below all of that (`x=147.5, y=482.6`) → member-level → tabs → table.
- **This phase (hybrid)**: race identity → ◎ headline with WIN/TOP3 immediately beside the horse name (`.cmd-pick` measured `width:782px`, natural content width, gap from name to stats = **10px**) → state line → judge (when present) → gauges **restored to the exact original position** (`x=949.5, y=216.3` — pixel-identical to the old reference) → member-level → tabs → table.

The hybrid reads as a natural evolution of the old header (same two-column skeleton, same gauge position) with one new, clearly-grouped line of information added to the left column — not as a different kind of component bolted on top.

## 7. Mobile visual comparison

375×812, same race. All three versions already stack gauges to their own full-width row below the identity column (this was true even in the old/pre-5B-2 header — mobile never had the two-column layout, only desktop did). The visible difference is purely vertical distance to the table:

- Old: table at 1171.58px.
- Tall: table at 1329.27px (+157.7px vs old).
- **Hybrid: table at 1190.41px (+18.8px vs old, -138.9px vs tall).**

Screenshots confirm the hybrid's ◎/WIN/TOP3 line renders on one compact row ("◎ エリカビアリッツ 21.1%WIN 46%TOP3"), directly below the metadata chips, with the full 4-gauge row and the member-level card's heading ("メンバーレベル ハイレベル") both visible in the same first viewport — more information visible above the fold than either prior version, not less.

## 8. WIN/TOP3 grouping assessment

**Fixed.** In the tall version, `.cmd-stats` had `margin-left:auto` inside a `flex-basis:100%` (full card width) container — this pushed WIN/TOP3 to the card's far right edge, ~700+px from the horse name it belonged to. In this phase, `.cmd-pick` is a natural-width flex row inside the narrower `.rh-main` column with no `margin-left:auto` on its stats group — measured gap from the end of the horse name to the start of the WIN/TOP3 group is **10px**. They now read unambiguously as one cluster: mark, name, WIN%, TOP3%, left-to-right, no visual separation.

## 9. Gauge-placement assessment

**Restored exactly.** Measured bounding box (`x:949.5, y:216.28, width:328, height:88.375`) is **pixel-identical** to the original old-header reference measured this same session. Gauges are readable (unchanged size/color/label treatment) and sit naturally secondary to the left column's bolder identity+headline content — they were never the focal point in the old header and remain not the focal point here, just no longer wastefully relegated to their own full-width row below everything else.

## 10. Numerical equality

Cross-checked ◎ WIN/TOP3 in the header against the shutsuba table's own ◎ row, same load: header `21.1%` / `46%`, table row `21.1%` / `46%` — exact match. Verified on both the pre-result race (`2026083001020401`) and a settled race (`2026081601010801`, テンレッドサン ◎ 23.8%/50%). No prediction value, formula, or data source was touched by this phase — only presentation/layout.

## 11. Console/runtime errors

Clean, confirmed via a fresh single-click tab isolated from this session's own scripted-testing noise (consistent with the pattern established since Phase 2D). Two things seen in the same-tab accumulated history and ruled out as unrelated to this change: the known `document.startViewTransition()` rapid-click artifact, and a `data/changes_20260816.json` 404 (pre-existing behavior when browsing to a historical date, already investigated and ruled out in the Phase 5C-1 report — not reproduced again here, same known cause).

## 12. PASS/FAIL against the 7 acceptance criteria

| # | Criterion | Result |
|---|---|---|
| 1 | Is ◎/WIN/TOP3 easier to find than in the old header? | **PASS** — the old header never showed WIN/TOP3 at all; the hybrid states it immediately, in gold, directly beside the horse name. |
| 2 | Is the hybrid at least as visually clean as the old/current production header? | **PASS** — same two-column skeleton, same gauge position/size, no dividers, no KPI-card grid; the only addition is one tightly-grouped line of real information. |
| 3 | Is the header materially shorter than the current Phase 5B-2 implementation? | **PASS** — 529.4px vs 737.9px, a 208.5px (28%) reduction. |
| 4 | Does the race table begin earlier in the viewport than Phase 5B-2? | **PASS** — 208.5px earlier on desktop, 138.9px earlier on mobile. |
| 5 | Are gauges still readable without becoming a primary focal point? | **PASS** — unchanged size/styling, restored to their original, already-proven-successful position. |
| 6 | Are WIN/TOP3 visually grouped with the ◎ horse rather than stranded on the far right? | **PASS** — 10px gap, confirmed via both measurement and screenshot. |
| 7 | Does desktop use horizontal space efficiently? | **PASS** — the right side of the card is occupied by gauges again, not empty space beside a stranded stat cluster. |

All seven criteria pass, including the two (#2, #3) that the brief flagged as fail-the-whole-implementation if unmet.

## 13. `git diff --stat`

```
$ git diff --stat -- site/
 site/css/style.css | 39 ++++++++++++++++++++-------------------
 site/index.html    |  4 ++--
 site/js/app.js      | 16 ++++++++--------
 3 files changed, 30 insertions(+), 29 deletions(-)
```

## 14. `git status --porcelain -- site/`

```
$ git status --porcelain -- site/
 M site/css/style.css
 M site/index.html
 M site/js/app.js
```

**Not committed**, per instructions. Scope confirmed: no changes to `#drawer`/bottom-sheet CSS, `sw.js`, `deploy/pycaliai-umami/server.py`, race-table rendering (`renderTable()`/`umamiTableHtml()`), any data/model file, `index.html`'s marketing sections, or `explain.html`.
