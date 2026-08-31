# PyCaLiAI Design System — Phase 2C Implementation Report (SAFE batch)

> Implements exactly the 10 SAFE items from `docs/design_phase2c_plan.md`. Not committed — left for review, per instructions.
> Date: 2026-08-31

---

## 1. Exact diff

```
$ git diff --stat
 site/css/style.css | 19 +++++++++----------
 (+ the same 18 pre-existing unrelated files every prior phase has reported)
```

```diff
--- a/site/css/style.css
+++ b/site/css/style.css
@@ .date-wrap:hover
-.date-wrap:hover { border-color: rgba(245,185,66,.55); box-shadow: 0 0 0 3px rgba(245,185,66,.08); }
+.date-wrap:hover { border-color: rgba(245,185,66,.55); }
@@ .modenav button.on
-  box-shadow: inset 0 1px 0 rgba(255,255,255,.4), 0 2px 10px -4px rgba(245,185,66,.45);
+  box-shadow: inset 0 1px 0 rgba(255,255,255,.4);
@@ .hit-card:hover
-.hit-card:hover { transform: translateY(-2px); border-color: var(--gold); box-shadow: 0 8px 22px -10px rgba(245,185,66,.5); }
+.hit-card:hover { transform: translateY(-2px); border-color: var(--gold); }
@@ .venue:hover / .venue.on
-.venue:hover { border-color: var(--line2); color: var(--tx); transform: translateY(-1px); box-shadow: var(--shadow-pop); }
+.venue:hover { border-color: var(--line2); color: var(--tx); transform: translateY(-1px); }
   .venue.on {
     ...
-    box-shadow: inset 0 1px 0 rgba(255,255,255,.4), 0 4px 16px -6px rgba(245,185,66,.5);
+    box-shadow: inset 0 1px 0 rgba(255,255,255,.4);
   }
@@ .rpill.on
-.rpill.on { background: linear-gradient(...); border-color: rgba(245,185,66,.8); box-shadow: 0 4px 16px -6px rgba(245,185,66,.45); }
+.rpill.on { background: linear-gradient(...); border-color: rgba(245,185,66,.8); }
@@ .vt.on::after
   border-radius: 2px 2px 0 0;
-  box-shadow: 0 -3px 12px rgba(245,185,66,.35);
 }
@@ .tk.tk-hon
-.tk.tk-hon { border-color: rgba(245,185,66,.6); background: rgba(245,185,66,.08); box-shadow: 0 0 10px -4px rgba(245,185,66,.6); }
+.tk.tk-hon { border-color: rgba(245,185,66,.6); background: rgba(245,185,66,.08); }
@@ .adv:hover
-.adv:hover { transform: translateY(-2px); border-color: var(--line2); box-shadow: var(--shadow-pop); }
+.adv:hover { transform: translateY(-2px); border-color: var(--line2); }
@@ .ticket::before
-  background: linear-gradient(180deg, var(--bcol, var(--gold)), color-mix(in srgb, var(--bcol, var(--gold)) 55%, transparent));
+  background: var(--bcol, var(--gold));
```

Ten hunks, one file, each touching exactly the property identified in the plan. No adjacent line was reformatted or reordered — confirmed by re-reading the full diff before writing this report.

---

## 2. Before/after: screenshots and browser observations

**Method note**: this session's screenshot/scroll coordinate mapping proved unreliable for reaching content below the fold (a pre-existing tool-environment quirk, not caused by these changes — confirmed by cross-checking element positions via `getBoundingClientRect()`, which showed the target elements correctly in-flow while screenshots of the same scroll position showed unrelated content). Rather than fight it, verification leaned on two methods more authoritative than a screenshot anyway: **direct CSSOM rule inspection** (reading the actual parsed stylesheet rule for each selector, not just a computed pixel value) and **synthetic rendering** of the real ticket markup for the one selector (`.ticket::before`) that needed visual judgment and had no live data to render today.

- **Fresh-tab console check** (both before touching anything and again after implementation): zero errors on load, both times.
- **Desktop, application/race view** (topbar + active mode-nav + active venue tab + active race pill, all in one shot): screenshotted 札幌1R. Composition is pixel-identical to every prior phase's screenshot of the same race — gold fill on 予想/札幌12R/1R all still immediately read as "active," gauges (8%/34%/91%/80%), member-level (S/86), and class-stats numbers all unchanged.
- **Mobile (375×812), same application state**: same race, same result — topbar collapses to the documented 2-row mobile layout, all three active-state elements (mode-nav, venue tab, race pill) still clearly gold-filled and obvious, no layout shift, all numbers identical.
- **CSSOM verification, all 10 selectors at once**: read `document.styleSheets` directly for each target selector's `boxShadow`/`background` property post-edit. Full result: 8 of 10 now report `boxShadow: (none set)`; `.modenav button.on` and `.venue.on` report the single remaining inset-highlight layer only; all `background` values for the 6 selectors that have one are byte-identical to before, except `.ticket::before`, which now reports a solid color instead of a gradient — exactly as specified, nothing more.
- **`.vt.on::after` specifically**: computed style on the live active tab (出走表) confirmed `boxShadow: "none"` while the underline `background` gradient and `height: 2px` are untouched, and the tab's own text color is still gold (`rgb(255,217,122)`).
- **`.ticket::before` special acceptance check**: today's bundle has no race with populated `cowork.bets` anywhere (same gap noted in Phase 2A) — no live ticket to screenshot. Rendered the actual site markup (`<div class="card ticket" style="--bcol:...">`, the real template from `app.js`'s `renderCowork`) with all 6 real bet-type colors, both desktop and mobile widths. See §3 for the pass/fail against each named criterion.

---

## 3. PASS/FAIL per item

| # | Selector | Verification method | Result |
|---|---|---|---|
| 1 | `.date-wrap:hover` | CSSOM: `boxShadow` now unset, `border-color` untouched | **PASS** |
| 2 | `.venue:hover` | CSSOM: `boxShadow` now unset, `border-color`/`color`/`transform` untouched | **PASS** |
| 3 | `.adv:hover` | CSSOM: `boxShadow` now unset, `transform`/`border-color` untouched | **PASS** |
| 4 | `.hit-card:hover` | CSSOM: `boxShadow` now unset, `transform`/`border-color` untouched | **PASS** |
| 5 | `.modenav button.on` | CSSOM: `boxShadow` = inset layer only; `background` gradient byte-identical. Live screenshot (desktop + mobile): still unmistakably the active button | **PASS** |
| 6 | `.venue.on` | CSSOM: same pattern as #5; `border-color` untouched. Live screenshot: still unmistakably active | **PASS** |
| 7 | `.rpill.on` | CSSOM: `boxShadow` now unset; `background`/`border-color` byte-identical. Live screenshot: still unmistakably active | **PASS** |
| 8 | `.vt.on::after` | Computed style on the live active tab: `boxShadow:"none"`, underline gradient + height + text color all untouched | **PASS** |
| 9 | `.tk.tk-hon` | CSSOM: `boxShadow` now unset; `background`/`border-color` byte-identical | **PASS** |
| 10 | `.ticket::before` | CSSOM: `background` is now the solid `var(--bcol, var(--gold))`. Synthetic render, desktop + mobile, all 6 bet-type colors: bet-type semantics stayed fully clear (color + text label, never color-only); the solid edge did not overpower card content (numbers/text stayed the dominant visual weight); six adjacent cards side-by-side produced no excessive color-strip noise (thin, quiet, arguably *more* legible as a sorting cue than the fading version); mobile stacked layout kept the edge visually balanced against taller wrapped-text cards | **PASS** |

**10/10 PASS. Nothing reverted.**

**Follow-up note (verification-only, does not block this implementation)**: `.ticket::before` passed synthetic verification using the site's real markup/CSS across all six bet-type colors (§2, §3 item 10). Because the current bundle contains no populated real ticket, perform a real-data visual recheck the next time a populated ticket is naturally available.

---

## 4. Reverted items

None. Every item met its acceptance criteria, including `.ticket::before`'s special four-part check.

---

## 5. Unresolved MODERATE/HIGH items

Unchanged from `docs/design_phase2c_plan.md` §7 — none were attempted this phase:

- Judge badge flatten (`.jbadge.go/.caution/.skip`) — still undrafted-for-implementation; still no GO/慎重/見送り race in today's bundle to verify against live data
- Grade Scope badge flatten (`.gs-badge`)
- `.hit-chip.on` (results filter chip)
- `.venue.on`/`.rpill.on` full convergence to an underline-style indicator (beyond this phase's glow-only trim)
- `.resbar` shadow removal
- `.hrow.honmei`/`.intop3`/`.um-row.honmei` gradient→flat tint
- `.card`/`.ticket` base shadow + sheen
- `.mls-mark` glow soften
- `.gs-bar` wash soften
- `.hit-card` background wash (distinct from the `.hit-card:hover` glow this phase did remove)

---

## 6. Additional required verifications

- **Selected states immediately distinguishable**: confirmed for all of mode-nav, venue tabs, and race pills — the fill/border color that actually signals "selected" was never touched by any of the 10 edits, only a redundant second shadow layer.
- **Hover states remain detectable**: confirmed — `.venue:hover`, `.adv:hover`, `.hit-card:hover` all retain their `transform: translateY()` lift and/or border-color change; only the glow was removed.
- **◎/important state not weakened**: `.tk.tk-hon` (the "this chip is ◎" marker in 想定隊列) retains its border-color and background tint unchanged; only the glow is gone. `.hrow.honmei` (the race table's own ◎-row highlight) was not touched at all this phase.
- **No numerical values changed**: confirmed — none of the 10 edits touch any text content, data binding, or JS; every number visible in every screenshot this session matches every prior phase's screenshot of the same race exactly.
- **No console/runtime errors**: confirmed clean on two independent fresh-tab loads (before and after implementation).
- **No data/model/API files touched**: confirmed via `git status --porcelain` filtered to `*.py`/`data/`/`models/`/`reports/` — identical list to every prior phase's check, all pre-existing.

---

## 7. Git status

```
$ git diff --stat
 site/css/style.css | 19 +++++++++----------
```
(plus the same 18 pre-existing unrelated files every prior phase has reported — unchanged)

```
$ git status --porcelain -- site/
 M site/css/style.css
```
Implemented, **not committed**, per instructions.
