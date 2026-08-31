# PyCaLiAI Design System — Phase 2B Report (Mechanical Token Groundwork)

> Implements items 1–3 only from `docs/design_phase2a_audit.md`'s Phase 2B table. Hard rule honored: **zero intentional visual change** — every migrated selector was verified to resolve to the exact same computed pixel value before and after.
> Date: 2026-08-31

---

## Files/selectors changed

One file: `site/css/style.css`. 13 edits — 1 token-block addition, 11 selectors re-pointed to `--r-card`, 1 selector re-pointed to `--r-pill-true`.

```
$ git diff --stat
 site/css/style.css | 27 +++++++++++++++++----------
 1 file changed, 17 insertions(+), 10 deletions(-)
```
(the other 18 files listed by a bare `git diff --stat` at the repo root are pre-existing, unrelated modifications that predate this session — confirmed unchanged again this phase, see the git status check at the end)

---

## Before / resolved / after token table

Per-selector: current literal → replacement token → resolved value before → resolved value after → identity assertion. "Resolved value" was checked the rigorous way — not by eye, but by creating a live DOM element of each class and reading `getComputedStyle(el).borderRadius` directly from the browser's CSS engine (script and full output in the verification section below).

| # | File:line | Selector | Current literal | Replacement token | Resolved before | Resolved after | Identical? |
|---|---|---|---|---|---|---|---|
| 1 | style.css:44–46 (`:root`) | — | *(tokens didn't exist)* | adds `--r-input:6px; --r-card:8px; --r-pill-true:999px;` | n/a | n/a | n/a — purely additive, no selector reads these yet except items 2–3 below |
| 2 | style.css:329 | `.rs-bt` | `8px` | `var(--r-card)` | `8px` | `8px` | ✅ |
| 3 | style.css:363 | `.hit-month` | `8px` | `var(--r-card)` | `8px` | `8px` | ✅ |
| 4 | style.css:403 | `.rs-back` | `8px` | `var(--r-card)` | `8px` | `8px` | ✅ |
| 5 | style.css:848 | `.tk` | `8px` | `var(--r-card)` | `8px` | `8px` | ✅ |
| 6 | style.css:900 | `.rb-fit` | `8px` | `var(--r-card)` | `8px` | `8px` | ✅ |
| 7 | style.css:1017 | `.dw-stat` | `8px` | `var(--r-card)` | `8px` | `8px` | ✅ |
| 8 | style.css:1071 | `.skel-bar` | `8px` | `var(--r-card)` | `8px` | `8px` | ✅ |
| 9 | style.css:1096 | `.tk-lineup` | `8px` | `var(--r-card)` | `8px` | `8px` | ✅ |
| 10 | style.css:1102 | `.tk-chip` | `8px` | `var(--r-card)` | `8px` | `8px` | ✅ |
| 11 | style.css:1127 | `.gs-badge` | `8px` | `var(--r-card)` | `8px` | `8px` | ✅ |
| 12 | style.css:1191 | `.tr-top-c` | `8px` | `var(--r-card)` | `8px` | `8px` | ✅ |
| 13 | style.css:890 | `.rb-waku` | `999px` | `var(--r-pill-true)` | `999px` | `999px` | ✅ |

**One deliberate deviation from the Phase 2A proposal, disclosed:** the Phase 2A report's Step A text proposed `--r-pill-true: 9999px` (matching DESIGN.md §8's literal "pill: 9999px"). This phase defines it as `999px` instead — matching `.rb-waku`'s *current* literal exactly, rather than DESIGN.md's spec value. `999px` vs `9999px` render identically for any real element (border-radius clamps to half the element's height regardless), but the hard rule here is "zero intentional visual change" with an explicit before/after identity assertion — so the current literal, not the eventual spec target, is what this phase's token had to match. Aligning `--r-pill-true` to DESIGN.md's `9999px` is a one-line, still-zero-visual-effect follow-up whenever that's wanted; not done here to keep this phase's "identical" claim exact rather than "identical in practice."

---

## Verification performed

**1. `git diff --stat`** — shown above.

**2. Exact Phase 2B diff** — shown in full to the user in this session (13 hunks in `site/css/style.css`); every hunk is a `border-radius: <literal>` → `border-radius: var(<token>)` substitution or the one token-block addition. No other property touched on any of the 12 lines.

**3. Computed styles for every migrated selector** — verified programmatically, not by inspection. Script: for each of the 12 re-pointed selectors, created a detached `<div class="X">`, appended it to `document.body`, read `getComputedStyle(el).borderRadius`, removed it. This reads the browser's actual cascade resolution for the class regardless of whether a real instance is currently on screen (several of these — `.dw-stat`, `.gs-badge`, `.skel-bar`, `.rb-waku`/`.rb-fit`, `.tr-top-c` — only render under specific, hard-to-force states: drawer open, a graded race, mid-load, cross-day realized-bias data, or not at all since the 調教 tab that used `.tr-top-c` was retired 2026-07-31 though its CSS rule remains). Result: **all 12 resolved to exactly the expected literal (`8px` ×11, `999px` ×1). Zero mismatches.**

**4. Desktop and mobile rendering** — screenshotted the results tab (`.rs-bt` bet-type cards, visually unchanged from the pre-Phase-2B baseline) and the race view at both desktop and 375×812 mobile widths. No layout shift, no visual anomaly, all displayed numbers identical to prior-phase screenshots of the same race (8%/34%/91%/80% gauges, レベル86, etc.).

**5. No intended visual delta** — confirmed twofold: the computed-style check (item 3) is the authoritative, pixel-exact confirmation; the screenshots (item 4) confirm nothing else moved as a side effect.

**One console item investigated and ruled out**: a stale browser tab (left open since Phase 2A's automated race-pill-clicking loop) showed 24 accumulated `AbortError: Transition was skipped` console entries — these are `document.startViewTransition()` promise rejections from clicking through race pills faster than the View Transitions API could complete each transition, a known artifact of rapid *automated* clicking, not something a human user triggers at normal pace. Confirmed unrelated to this phase's changes by opening a **fresh tab**: clean load shows **zero console errors**, on both `予想` and `成績` modes, desktop and mobile.

**6. No prediction/data/API files touched** — confirmed via `git status --porcelain` filtered to `*.py`/`data/`/`models/`/`reports/`: every modified file listed predates this session, identical list to every prior phase's check.

---

## Unresolved radius migrations

Everything **not** covered by items 1–3 remains exactly as documented in the Phase 2A report's Phase 2B table (items 4–7) — none of it was in scope for this phase and none of it was touched:

- `.btn-prime`/`.btn-ghost`: still `var(--r-sm)` (2px), not moved to `var(--r-input)` (6px) — would be a real, visible change (2px→6px corners on both hero buttons).
- `.card`/`.mlvl`: still `var(--r-md)` (4px), not moved to `var(--r-card)` (8px) — would be the largest single visual change in the whole radius system, sitewide and simultaneous.
- `.hh`/`.resbar` mixed-corner declarations: still reference `var(--r-md)`, not `var(--r-card)`.
- The 15 `--r-pill` (4px) call sites (chips/badges/`.venue`): untouched — whether these should become true pills (`var(--r-pill-true)`) or stay compact under a clearer name is still an open product decision, not attempted here.
- The ~35 remaining hardcoded radius values *other than* the 11 `8px` and 1 `999px` sites fixed this phase (the `6px`, `10px`, `4px`, `2px`, `50%` clusters catalogued in the Phase 2A report §1.3) — none were in the item 1–3 scope and none were touched.

Also untouched, exactly as instructed: any glow/gradient/shadow, race-table density, judge/advisor presentation, gauges, layout, and every displayed numerical value.

---

## Decision Log

| Item | Outcome |
|---|---|
| Step 1 — Phase 1 commit | `96672b3636` — exactly `site/css/style.css`, `site/index.html`, `site/explain.html`, `docs/design_phase1_report.md`. Nothing unrelated included. |
| Step 2 — Phase 2A commit | `0e0bc15dff` — exactly `docs/design_phase2a_audit.md`. Nothing unrelated included. |
| `git status --porcelain -- site/` after Phase 1 commit | Empty, as expected. |
| Phase 2B items 1–3 | Implemented exactly as scoped: 3 new tokens added, 11 `8px` sites + 1 `999px` site re-pointed. |
| `--r-pill-true` value | Set to `999px` (matching the current literal) rather than the Phase 2A proposal's `9999px` (DESIGN.md's spec value), to satisfy this phase's stricter "identical, not just visually equivalent" requirement. Disclosed above, not silently substituted. |
| Phase 2B items 4–7 | **Not implemented** — each is a real visual change (see Phase 2A report), explicitly out of this phase's "zero intentional visual change" scope. |
| Glow/gradient/shadow, race-table density, judge/advisor, gauges, layout, numeric values | **Untouched**, as instructed. |
| Verification | All 6 requested checks performed; one console-error false alarm investigated and ruled out (stale tab, not a regression); zero actual regressions found. |
| Prediction/data/API files | **Untouched** this phase, confirmed by filtered `git status`. |

---

## Recommended Phase 2C scope

Phase 2C wasn't started this turn per instructions. When it's time: the Phase 2A report's Phase 2C table (§6) already has 8 grouped items in a recommended shipping order — hover-glow removal first (lowest risk, most isolated), then badge flattening (`.jbadge`, `.gs-badge`), then the gradient-softening group (row-highlight tints, ticket edge, hit-card background), then the list-nav convergence (`.venue.on`/`.rpill.on` → the lighter `.vt.on`-style indicator), ending with the `.card`/`.ticket` base-shadow reduction — that last one specifically *after* list-nav convergence lands and is confirmed good, since it's the single biggest visual-character change in that phase and benefits from not being compounded with another large change in the same review pass.

One adjustment worth flagging before 2C starts: Phase 2A's item 5 in the *2B* table (`.card`/`.mlvl` from `var(--r-md)` to `var(--r-card)`, i.e. 4px→8px corners) is a **radius** change, not a shadow/glow change — it was correctly left out of this phase's zero-diff scope, but it touches the exact same selector (`.card`) that Phase 2C's shadow-reduction item also touches. Recommend deciding radius-4px-vs-8px *before* touching `.card`'s shadow in 2C, so `.card` doesn't get revisited twice for two unrelated visual decisions.

---

## Git status

```
$ git diff --stat
```
(shown above — only `site/css/style.css`, plus the same 18 pre-existing unrelated files every prior phase has reported)

```
$ git status --porcelain -- site/
 M site/css/style.css
```
Expected — this phase's changes are implemented but **not yet committed**, left for review alongside this report, matching the pattern from every prior phase.
