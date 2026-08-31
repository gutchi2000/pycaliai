# PyCaLiAI Design System — Phase 1 Report (Foundation)

> Implements the Phase 1 scope from DESIGN.md v1.0. Files listed here are **uncommitted working-tree changes** — see Status at the end.
> Date: 2026-08-31

---

## Files changed

```
 site/css/style.css |  9 +++++----
 site/explain.html  | 16 ++++++++--------
 site/index.html    |  6 +++---
 3 files changed, 16 insertions(+), 15 deletions(-)
```

No other file changed. Confirmed via `git status --porcelain` that every `.py`/`.json`/`.csv`/`.parquet` modification present in the working tree predates this session (the same 18 files listed in this conversation's opening `git status` snapshot) — none were opened or touched while implementing Phase 1.

---

## Design rules addressed, with before/after

| File | Selector | Before | After | DESIGN.md rule |
|---|---|---|---|---|
| `style.css` `:root` | `--ring` | `0 0 0 3px rgba(245,185,66,.22)` | `0 0 0 2px rgba(245,185,66,.4)` | §23 Accessibility — 2px brand-gold focus ring |
| `style.css` `:root` | `--border-subtle` | did not exist | `rgba(30,43,74,.5)` (added) | §3 Brand Colors — completes the border-token set |
| `style.css` | `.btn-prime` | `font-weight: 700` | `600` | §5 Typography — 600 ceiling, general UI |
| `style.css` | `.btn-ghost` | `font-weight: 700` | `600` | §5 Typography |
| `style.css` | `.modenav button` | `font-weight: 700` | `600` | §5 Typography |
| `index.html` | `.modenav` buttons | `🏇 予想` / `📊 成績` / `🔬 分析カード` | `予想` / `成績` / `分析カード` | §10 Navigation — remove decorative emoji (the exact named example) |
| `explain.html` | `<h1>` | `🏇 レース分析カード...` | `レース分析カード...` | §10 Navigation |
| `explain.html` | `<meta name="theme-color">` | `#010E27` (didn't match the real canvas) | `#070b14` (matches `--bg-0`, same as `index.html`) | Adjacent fix found while touching this file — not a Phase 0 finding, disclosed rather than silently bundled |
| `explain.html` | Google Fonts `<link>` | `Noto+Sans+JP` + `Oswald` (Oswald loaded but never referenced by any `font-family` on this page — dead weight, not actually rendered) | `Noto+Sans+JP` + `Zen+Kaku+Gothic+New` + `IBM+Plex+Mono`, matching `index.html`'s stack | §10/architecture consolidation. **Correction to my own Phase 0 write-up**: I originally described this as a "noticeable font-letterform shift." On implementation it turned out Oswald was never actually applied anywhere in `explain.html`'s CSS (nothing names it) — the real, visible effect of this change is that `.num`/`var(--disp)` text (the numeric columns) now renders in the intended monospace face instead of silently falling back to Noto Sans JP. Confirmed by screenshot: numeric columns visibly tightened/aligned after the fix. |
| `explain.html` | `select`, `.toolbar button`, `.badge`, `.backlink` | hardcoded `border-radius: 8px` | `var(--r-pill)` (4px) | Architecture consolidation — matches chips/badges/inputs everywhere else in the app |
| `explain.html` | `.card` | hardcoded `border-radius: 14px` | `var(--r-md)` (4px) | Architecture consolidation — matches every other `.card` in the app |

---

## Verification performed

1. **`git diff --stat`** — 3 files, 16 insertions / 15 deletions. Matches the planned scope exactly (see chat transcript for the full pre-implementation plan table).
2. **`git diff`** for all three files — reviewed in full, every hunk traces to a specific planned change; no incidental edits.
3. **Console/runtime errors** — local preview served via the existing `.claude/launch.json` "site" config (`python -m http.server 8765 --directory site`). `read_console_messages(onlyErrors: true)` returned "No console logs" on both `index.html` and `explain.html`.
4. **Navigation functionality** — clicked into the app from the landing CTA (works); verified the `予想`/`成績` mode switch both visually (screenshot, active-state gold fill intact) and programmatically (`document.querySelector('#modeNav [data-mode="results"]').click()` → `body` loses `on-landing`, `#main` hides, `#resultsMain` shows, active button text reads `成績` — the emoji-free label is still read/matched correctly by nothing, since the switch logic keys off `data-mode`, not text).
5. **Desktop layout** — screenshotted hero, app race view (race header, gauges, member-level card, class-stats strip), results tab (KPI grid, monthly table), and `explain.html`'s card/table view. All render correctly; active-state gold fills, focus styling, and card borders unaffected in appearance beyond the intended weight/radius/ring changes.
6. **Mobile/responsive layout** — resized to the 375×812 mobile preset. Topbar collapses to the documented 2-row layout, mode-nav becomes a full-width 3-up row, hero and shutsuba table both render correctly; mark/umaban/name stay visible in the mobile-collapsed race table per §21.
7. **No displayed numerical values changed** — spot-checked the same race (エリカビアリッツ, 札幌1R) across three views: hero-spot widget top-5 (42.2%/16.6%/10.5%/10.5%/7.2%), `explain.html`'s table (21.1%/45.8%/4.7/5.2/1509/+5.3), and the mobile shutsuba table (100 AI指数/21.1%勝率) — all consistent with each other and with pre-change values (nothing in this phase touches any `.py`, `data/`, or JS computation, so no value could have changed; this check confirms the CSS/markup edits didn't accidentally clip or reformat a displayed number).
8. **No prediction/data/API files changed** — confirmed via `git status --porcelain` filtered to `*.py data/ models/ reports/`: every modified file listed predates this session (matches the conversation's opening `git status` snapshot verbatim). Nothing under those paths was opened this session.

No regressions found.

---

## Unresolved DESIGN.md deviations (known, deliberately deferred)

These remain non-compliant after Phase 1 — none were in scope, all are flagged in DESIGN.md's own §4 Surface System / §24 as Application-surface targets for later work:

- Judge badges (`.jbadge`), advisor medals (`.adv-medal`), ticket accent borders (`.ticket::before`), Grade Scope badge (`.gs-badge`), and the career-chart area-fill still use gradient + glow on Application/Data surfaces.
- Active-state treatment on venue tabs (`.venue.on`) and race pills (`.rpill.on`) is still the filled-gradient-glow pattern, not the lighter list-nav indicator DESIGN.md v1.0 §10 now specifies for list-style (as opposed to mode-switch) navigation.
- `--r-sm`/`--r-md` values (2px/4px) are still short of DESIGN.md §8's 6px/8px targets globally — Phase 1 only pointed `explain.html`'s own hardcoded radii at the *existing* tokens, it didn't move the tokens themselves.
- `--r-pill: 4px` still doesn't behave like a true pill (9999px).
- Race-table/table-cell typography still ranges up to ~17.5px in places (`.hname`, `.pwin`), above §6's 12–15px table/data-dense ceiling — covered by the intermediate-size clarification added to DESIGN.md, not a violation, but noted for completeness.
- The four items DESIGN.md itself still lists as DEFER (HONEST RECORD KPI-card/gauge density, wins-only results gallery, the missing literal 単勝/odds column, and the general Application-surface shadow/gradient flattening) are all untouched, as instructed.

---

## Deferred Phase 2 candidates

In rough priority order, based on what Phase 1 explicitly could not touch:

1. **Application/Data surface glow & gradient flattening** — judge badges, advisor medals, ticket borders, Grade Scope badge, active-pill treatment on venue tabs/race pills. Largest visual-impact item; needs its own before/after screenshot review per component, not a blanket sweep.
2. **List-nav vs. segmented-control convergence** — `.venue.on`/`.rpill.on` toward the lighter indicator `.vt.on` already uses correctly.
3. **Radius token realignment** (`--r-sm`/`--r-md` toward 6px/8px, and/or splitting a separate icon-radius token) — sitewide, simultaneous, visible on every card/button; deserves its own scoped pass with full-page screenshots rather than being folded into "foundation."
4. **`--r-pill` naming/value fix** — low-priority hygiene, bundle with #3.
5. The four still-DEFERRED items from DESIGN.md itself (KPI density, wins-only gallery, odds column, dual max-width already resolved in v1.0 so this one's closed) remain open product/IA questions, not CSS work.

---

## Decision Log

| Item | Outcome |
|---|---|
| Git baseline (Step 0) | Committed — 2 commits, exactly the 6 design docs + the type-scale clarification, verified nothing else was swept in |
| Type-scale clarification clause | Added to DESIGN.md §6 verbatim as requested, committed |
| Phase 1 scope items 1–5 (tokens, typography, borders/radius-where-shared, focus ring, nav emoji) | Implemented — see table above |
| Phase 1 scope item 6 (`explain.html` consolidation) | Implemented — font stack, 5 hardcoded radii, plus one adjacent theme-color fix disclosed separately |
| Phase 1 scope item 7 (small prerequisite refactors) | None needed beyond what's listed above |
| Global `--r-sm`/`--r-md` value changes | **Not done** — explicitly out of "very small" scope; sitewide simultaneous visual change, deferred to Phase 2 |
| CSS custom-property renames to match DESIGN.md's token names verbatim | **Not done** — pure churn, no visual benefit, real break risk; values already match, names don't need to |
| Judge-badge/medal/ticket/gauge gradient or glow | **Not touched** — explicitly excluded this phase |
| Race table, KPI density, wins-only gallery, odds column | **Not touched** — explicitly excluded this phase |
| Verification | All 8 requested checks performed; zero regressions found |

---

## Status

`site/css/style.css`, `site/index.html`, `site/explain.html` are **modified but not committed** — left for review alongside this report and the diff already shown in chat, matching how the Phase 0/0.5 documents sat before their own "establish baseline" instruction. Say the word to commit as its own Phase 1 commit.

Phase 2 not started.
