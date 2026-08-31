# PyCaLiAI Design System — Phase 2A: Application/Data Surface Audit & Migration Plan

> Planning only. No production files were edited to produce this document — every finding below was gathered by reading `site/css/style.css` in full and grepping it for every `--r-sm`/`--r-md`/`--r-pill`, `box-shadow`, `linear-gradient`, and `backdrop-filter` occurrence, plus live browser observation of the running site.
> Date: 2026-08-31

---

## 1. Radius/token system audit

### 1.1 Token definitions (unchanged from Phase 1)

```
--r-sm: 2px;
--r-md: 4px;
--r-pill: 4px;
```

### 1.2 Every tokenized call site (29 total), classified

| Token | Selector | File:line | Semantic component | Surface |
|---|---|---|---|---|
| `--r-sm` | `:focus-visible` | style.css:89 | utility (focus outline) | both |
| `--r-sm` | `.btn-prime` | style.css:111 | button | Marketing |
| `--r-sm` | `.btn-ghost` | style.css:118 | button | Marketing |
| `--r-sm` | `.lp-race-name .lp-grade` | style.css:196 | badge | Marketing |
| `--r-sm` | `.lpf-row .vm` | style.css:214 | chip | Marketing |
| `--r-sm` | `.lpi-row .lpi-grade` | style.css:232 | badge | Marketing |
| `--r-sm` | `.brand-logo` | style.css:273 | icon | App (topbar, both surfaces) |
| `--r-sm` | `.brand-ver` | style.css:283 | chip | App (topbar) |
| `--r-sm` | `.tk-unknown` | style.css:1109 | other (notice box) | App |
| `--r-md` | `.hero-spot` | style.css:126 | card | Marketing |
| `--r-md` | `.card` | style.css:475 | **card** (universal, ~40+ instances via class reuse) | App (almost exclusively) |
| `--r-md` | `.mlvl` | style.css:556 | card | App |
| `--r-md` | `.resbar` (mixed: `4px var(--r-md) var(--r-md) 4px`) | style.css:649 | other (banner) | App |
| `--r-md` | `.hh` (mixed: `var(--r-md) var(--r-md) 0 0`) | style.css:718 | other (sticky table-header cap) | App |
| `--r-pill` | `.date-wrap` | style.css:293 | other (input wrapper) | App |
| `--r-pill` | `.modenav` (container) | style.css:307 | other (segmented-control shell) | App |
| `--r-pill` | `.modenav button` | style.css:308 | button (segmented) | App |
| `--r-pill` | `.hit-chip` | style.css:355 | chip | App |
| `--r-pill` | `.hit-type` | style.css:392 | badge | App |
| `--r-pill` | `.venue` | style.css:432 | true pill (list-nav tab) | App |
| `--r-pill` | `.tdchip` | style.css:488 | badge | App |
| `--r-pill` | `.mchip` | style.css:491 | chip | App |
| `--r-pill` | `.jbadge` | style.css:493 | badge | App |
| `--r-pill` | `.jtag` | style.css:499 | chip | App |
| `--r-pill` | `.res-pays i` | style.css:659 | chip | App |
| `--r-pill` | `.hitchip` | style.css:663 | badge (status stamp) | App |
| `--r-pill` | `.vschip` | style.css:827 | chip | App |
| `--r-pill` | `.adv-tag` | style.css:952 | chip | App |
| `--r-pill` | `.tk-pace-chip` | style.css:1094 | badge | App |

### 1.3 Beyond the 3 named tokens: the hardcoded majority (essential context)

The 3 tokens above account for **29 of roughly 75+ total `border-radius` declarations** in the stylesheet. The remaining ~46 are hardcoded literals, never touching `--r-sm`/`--r-md`/`--r-pill` at all. This wasn't explicitly asked for, but a "safe migration strategy" for the 3 tokens is incomplete without knowing what they're competing against. Grouped by value:

| Hardcoded value | Count (approx.) | Representative selectors | What they are |
|---|---|---|---|
| `50%` | 8 | `.brand-ver::before`, `.rs-dot`, `.rpill .rdot`, `.resb`, `.adv-medal`, `.tk-badge`, `.legdot`, `.mlvl-badge`(10px, see below) | **Circles** (status dots, medal/avatar circles) — a distinct semantic category from rectangular radius entirely, shouldn't be folded into the pill/card/button system |
| `2px`–`3px` | 6 | `.hs-bar`, `.mls-mark::before`, `.vt.on::after`, `.sh-title::before`, `.dw-sec::before`, `.bar`/`.bar i` | Thin accent bars / progress-bar fills — structural, not chips |
| `4px`–`5px` | 8 | `.rb-fit-sty`, `.ir .kw`, `.scrbadge`, `.rh-time .tchg`, `.vt.soon i`, `.mls-mark b`, `.rt-set > b` | Small inline tags — functionally identical to what `--r-sm` should be if it moves to DESIGN.md's 4px "small/icon" target |
| `6px` | ~10 | `.kyaku`, `.um-g`, `.rkb`, `.tact-row.won`, `.pair.win`, `.rh-time`, `.pace` | Medium chips — currently split between this hardcoded 6px and the tokenized `--r-pill` (4px) for visually-equivalent components |
| `8px` | ~15 | `.rs-bt`, `.hit-month`, `.rs-back`, `.tk-chip`, `.tk-lineup`, `.skel-bar`, `.dw-stat`, `.tr-top-c`, `.gs-badge` | **This is already the de facto dominant "medium card/control" radius** — closer to DESIGN.md's "application card: 8px" target than the tokenized `--r-md` (4px) is |
| `10px`–`12px` | 5 | `.rs-stat`, `.mlvl-badge`, `.hit-card` | Larger cards, ad hoc |
| `999px` | 1 | `.rb-waku` | **A genuine true pill**, hardcoded because `--r-pill` (4px) doesn't deliver one |

### 1.4 Where token names and actual semantics disagree

1. **`--r-pill` isn't a pill.** 15 call sites use it for chips/badges/a true list-nav tab (`.venue`) that visually read as "compact rounded control," yet the token renders a 4px corner on elements 20–36px tall — roughly 15–20% of the visual softness a real pill (9999px) would give. Meanwhile a genuine pill (`.rb-waku`, 999px) exists in the codebase entirely *outside* the token system, because the token couldn't deliver what was needed. This is the clearest name/semantic mismatch in the file.
2. **`--r-sm` conflates three different jobs.** DESIGN.md §8 wants "small/icon: 4px" separate from "button/input: 6px." Production's single 2px `--r-sm` is used for a 26px-tall primary CTA button *and* a 30px icon *and* a focus outline — three components DESIGN.md treats as different radius tiers, collapsed into one value.
3. **`--r-md` undershoots its own most common real-world value.** The tokenized `--r-md` is 4px, but the *actual* most frequently used "medium card" radius in the file (hardcoded, ~15 occurrences) is 8px — exactly DESIGN.md's "application card: 8px" target. The token exists but isn't the value people reach for when they want that look.
4. **Circles were never part of this system and shouldn't be pulled into it.** `50%` (8 occurrences) is a completely different concept (status dots, medals) that happens to also use `border-radius`. Any migration plan should explicitly *not* touch these.

### 1.5 Proposed migration strategy (no values changed in this phase)

A three-step plan, each step independently revertible:

**Step A — add, don't change.** Introduce three new, correctly-scoped tokens alongside the existing ones (existing tokens keep their current values and every current call site untouched — zero visual regression risk from this step alone):
```
--r-input: 6px;   /* DESIGN.md §8 "button/input" */
--r-card: 8px;    /* DESIGN.md §8 "application card" — matches the already-dominant hardcoded 8px */
--r-pill-true: 9999px;  /* an actual pill, for .rb-waku and any future true-pill need */
```

**Step B — tokenize matches first (zero visual change).** Re-point every hardcoded `8px` call site to `var(--r-card)`, and `.rb-waku`'s hardcoded `999px` to `var(--r-pill-true)`. Since the values are identical, this produces **no rendered difference whatsoever** — pure code-quality/consistency work, fully safe to batch in one commit.

**Step C — the real visual decisions (each flagged individually, not batched).** Everything else — moving `.btn-prime`/`.btn-ghost` from `--r-sm`(2px) to `--r-input`(6px), moving `.card`/`.mlvl` from `--r-md`(4px) to `--r-card`(8px), deciding whether `.venue`/chips/badges should move from `--r-pill`(4px) to `--r-pill-true`(9999px) or stay compact — each of these **does** change how the page looks, sitewide, simultaneously across every instance of that component. These are listed individually in the Phase 2B table below with their own visual-impact/risk/verification, not assumed to ship together.

---

## 2. Gradient/glow/shadow inventory — Application/Data surfaces only

Marketing-surface rules (`.hero-*`, `.hs-*`, `#landing`, `.lp-*`) are excluded per instructions — those are already governed by DESIGN.md v1.0 §4's separate Marketing allowance and out of scope here.

| Selector | File:line | Effect | Classification | Recommendation | Why |
|---|---|---|---|---|---|
| `:focus-visible` | 89 | box-shadow ring | accessibility | KEEP | Required, already narrow (§23) |
| `.topbar` | 264 | drop shadow | structural depth | KEEP | Functional sticky-nav separation from scrolling content |
| `.brand-ver::before` | 289 | glow on 6px dot | status | REDUCE | Blur is unnecessary on a status dot this small; a plain filled dot reads identically |
| `.date-wrap:hover` | 297 | gold glow | decorative | REMOVE | Hover feedback via border-color change alone is sufficient and matches §4 |
| `.modenav button.on` | 313 | inset highlight + outer glow | selected state | REDUCE | §10's segmented-control exception covers the *fill*, not an added outer glow layer — drop the glow, keep the embossed inset highlight |
| `.hit-card:hover` | 378, 381 | lift + colored glow | decorative | REDUCE | Results tab is a grid of many repeated cards — keep the lift (functional "clickable" cue), drop the glow |
| `.ticket.won` | 406 | inset ring | status | KEEP | Informational (marks a winning ticket), inset-only, no blur — already the right pattern |
| `.venue:hover` | 438 | shadow-pop | decorative | REMOVE | Same repeated-list-item reasoning as elsewhere in §10's convergence |
| `.venue.on` | 443, 466(rpill equiv) | gradient fill + glow | selected state | REPLACE | Converge to the lighter indicator `.vt.on` already uses correctly (per DESIGN.md v1.0 §10) |
| `.rpill.on` | 466 | gradient + glow | selected state | REPLACE | Same as `.venue.on` — same component family |
| `.card` | 473, 476 | subtle white sheen + drop shadow | structural depth / decorative | REMOVE (gradient), REDUCE (shadow) | `.card` already has a border; sheen is decorative on the single most-repeated component in the app (40+ instances/page); shadow is redundant elevation-signaling once the border exists |
| `.jbadge.go/.caution/.skip` | 494–496 | 2-stop gradient + glow + inset highlight | status | REDUCE→REPLACE | The GO/慎重/見送り signal is genuinely meaningful — keep the color coding and label, move from gradient-fill+glow to the low-alpha-background pattern `.tdchip`/`.vschip` already use successfully |
| `.mls-mark` (pointer badge) | 613, 620 | glow on line + badge | structural/wayfinding | REDUCE | Some visual weight is functional (pointer must stand out against the colored zone bar it overlaps) — soften, don't remove |
| `.resbar` | 647, 651 | gradient wash + shadow-card | status/structural | REDUCE | Wash can stay (subtle, informational — "this race has a result"); the extra shadow atop an already-bordered, already-tinted strip is redundant |
| `.hitchip.miss` | 667 | none (already flat) | status | KEEP | Already the compliant pattern — good existing example |
| `.resb.r1/r2/r3` | 674–676 | medal gradient (+glow on r1) | status/celebratory | KEEP | Matches DESIGN.md v1.0's explicit win-stamp exception — a single, infrequent podium marker, not a repeated element |
| `.vt.on::after` | 688–690 | gradient underline + glow | selected state | REDUCE | The underline itself is the *correct* minimal pattern — drop only the extra glow beneath it |
| `.hrow` hover/`.honmei` accent bar | 732, 736, 739, 741 | inset solid-color box-shadow (border trick) | selected/highlight state | KEEP | Already exactly what §4 asks for — a thin accent bar, not a glow. Positive existing-compliance example |
| `.hrow.honmei`/`.intop3` background | 738, 742–743 | gradient-fade background wash | selected/highlight state | REDUCE | A flat low-alpha tint communicates "this row is ◎ / finished top-3" with less visual noise than a fading gradient, repeated down every table |
| `.tk.tk-hon` | 856 | border + bg + glow | brand/highlight | REDUCE | Border-color + background tint already signals "this is the ◎ chip" — glow is additive decoration on a sufficient signal |
| `.adv:hover` | 939 | lift + shadow-pop | decorative | REMOVE | Same repeated-card-grid reasoning as `.hit-card:hover` |
| `.adv-medal.gA` | 945 | gradient + glow | status | REDUCE | Grade is real, meaningful data (confirmed by browser observation — reads clearly as a "medal"); keep the small gradient fill, soften/drop the surrounding glow |
| `.ticket::before` | 975 | 2-stop color→transparent gradient | status/categorization | REDUCE | This is a bet-type color key (単勝=gold, 複勝=teal…) — a solid color edge communicates the same category with one fewer visual layer than a gradient fade |
| `.ticket` | 967–968 | shadow-card + hover shadow-pop | structural / decorative | REDUCE (base), REMOVE (hover) | Same reasoning as `.card` and `.hit-card` |
| `#drawer` | 996, 998 | radial/linear background wash + entrance shadow | decorative (wash) / structural (shadow) | KEEP both | Entrance shadow is DESIGN.md's explicit true-overlay exception; the background wash is very low-opacity ambient texture, same family as the site-wide `body::before` treatment, doesn't compete with the drawer's content |
| `.gs-bar` | 1121 | diagonal gold wash | brand/decorative | REDUCE | Once-per-graded-race header, low repetition — could stay as a restrained "featured" treatment, but current 16%-opacity wash is more prominent than a flat app surface calls for |
| `.um-row.honmei` | 1165 | gradient-fade background | selected/highlight state | REDUCE | Same as `.hrow.honmei` — flatten to solid low-alpha tint |
| `.bar i` / `.rb-bar i` / `.why.neg .bar i` / `.lapmini i.last` | 793, 800, 805, 887, 1035, 1211 | value-encoding gradient fills | **data visualization**, not decoration | KEEP | These are literal bar charts (AI-index score, front-rate %, feature-contribution magnitude, lap split) — a gradient fill on a value bar is standard, accepted data-viz convention, categorically different from decorative gradients |
| `::before` section-marker bars (`.sh-title`, `.cw-title`, `.dw-sec`) | 698, 921, 1028 | 3px-wide gradient accent bar | structural/wayfinding | KEEP | Tiny (3×14px), functions as a heading marker/rule, not competing decoration |
| `.won-badge` / `.hit-stamp` / `.hitchip` (win family) | 384–385, 400–401, 408, 662–663 | gradient stamp | celebratory | KEEP | Explicit DESIGN.md v1.0 win-stamp exception |
| `.hit-card` background wash (not the stamp) | 376, 400 | gradient tint across the whole card | brand/decorative | REDUCE | The stamp alone can carry the "celebratory" signal (already blessed) — tinting the entire surrounding card is decoration beyond what the exception covers |
| `body::before`/`::after` | 66–79 | ambient radial + grid texture | decorative | KEEP | Whole-page background texture behind all content (both surfaces), extremely low opacity (≤.4), doesn't compete with any component |
| `.track-stripe` | 253 (& app.js usage) | thin gradient line, top of every page | decorative | KEEP | 2px, 50% opacity, present once per page load — minimal footprint, arguably a legitimate brand motif (literally a track/rail line) |
| `.mlvl` background wash | 557 | tier-colored gradient wash | status | KEEP | Low-opacity (~9%), tier-coded, informational — colorizes by S/A/B/C/D grade |
| `backdrop-filter` (topbar, `.hh`, `#overlay`) | 261–262, 715–716, 987 | blur | functional | KEEP | Already resolved in DESIGN.md v1.0 §4 — sticky-header/scrim legibility, not decorative glassmorphism |

**Tally: 42 occurrences classified. KEEP 18 · REDUCE 16 · REMOVE 5 · REPLACE 3.**

---

## 3. Race/data UI audit

| Aspect | DESIGN.md v1.0 rule | Current implementation | Verdict |
|---|---|---|---|
| Table typography | §6 table/data-dense: 12–15px | Mostly 12–17.5px; `.hname` (18px) and `.pwin` (17.5px) exceed the ceiling | **Justified exception** under the intermediate-size clarification added in Phase 0.6 — horse name is deliberately the largest text in each row (hierarchy), not a violation |
| Numeric font | §5: JetBrains Mono / IBM Plex Mono | `.num{font-family:var(--disp)}` = IBM Plex Mono first choice, applied broadly | **Compliant** |
| Tabular numbers | §5: `font-variant-numeric: tabular-nums` | Applied via shared `.num` utility class across every numeric display | **Compliant** |
| Column alignment | §13: right alignment for numeric | `.ta-r`/`.ta-c` utilities used consistently on numeric columns | **Compliant** |
| Row height | §13: 40–44px | `.hrow{padding:7px 14px}` around 1–2 lines of content (single-line rows land near the target; two-line name+sub-line rows likely render taller, ~50–60px) | **Borderline** — needs a live pixel measurement in Phase 2D, not resolvable from CSS alone |
| Row separators | §9: navy hairline | `border-top:1px solid var(--line)` | **Compliant** |
| Hover state | §4: one consistent indicator, not glow | `background:rgba(255,255,255,.04)` + inset accent bar, no blur | **Compliant** — a positive existing example |
| Selected state | §4 | No persistent "selected row" concept exists (clicking opens the drawer; no lingering selected style) — only ◎/top-3 status highlighting | **Not applicable** / observation only |
| Mark hierarchy | §14: gold ◎ accent, marks readable without color | Size+color ladder (◎21px gold → 〇/▲/△19px → unmarked 12px muted); marks are always the ◎〇▲△ glyphs, never color-only | **Compliant** |
| Card-vs-table usage | §13: tables preferred for comparison | Shutsuba (main race table) is a proper CSS-grid table — compliant. Results-tab hit-gallery, advisor grid, ticket grid, pedigree grid all use card grids instead | **Mixed** — the peripheral card-grid usage is the already-DEFERRED KPI/card-density question (Phase 0/0.5), not re-opened here |
| Spacing | §7: 4px grid | Same off-grid half-values found in Phase 0 (7px, 9px, 13px, 14px) still present — Phase 1 didn't touch table internals | **Unresolved**, unchanged since Phase 0 |
| Borders | §9: navy hairline | Navy throughout | **Compliant** (DESIGN.md itself was aligned to this in Phase 0.6) |
| Responsive horizontal behavior | §21: column-collapse or scroll, keep mark/number/name | Column-collapse strategy at 7 breakpoints; mark/umaban/name never hidden at any width | **Compliant** |
| Sticky behavior | §4: functional blur permitted | `.hh{position:sticky;top:56px}` (offset below the sticky topbar), `backdrop-filter:blur(10px)` for legibility | **Compliant** |

---

## 4. Judge/advisor/status UI: meaningful information vs. decorative excess

| Component | Verdict | Detail |
|---|---|---|
| Judge badges (GO/慎重/見送り) | **Information is meaningful; presentation is excessive.** | Color + text jointly encode a real participation recommendation, and the same color also drives the race-strip status dots (`.rdot`) — the coding itself is load-bearing for fast scanning. The gradient-fill+glow+inset-highlight execution, though, is heavier than a status badge needs to be (see §2). |
| Advisor medals (A/B/C) | **Meaningful, and already reasonably well-executed.** | Grade is real Cowork-assigned classification. Direct browser observation (screenshot, this session) showed it reading clearly and restrained — small circle, 2-stop gradient, legible at a glance. Only the outer glow is worth softening; the core treatment can stay. |
| Status chips (`.tdchip`, `.vschip`, `.mchip`, `.jtag`) | **Meaningful and already compliant.** | Low-alpha background + border + text label — this is the model pattern other components (judge badges, Grade Scope badge) should converge toward, not something needing its own fix. |
| Active pills (`.venue.on`, `.rpill.on`, `.hit-chip.on`) | **The selection state is meaningful; the visual weight is not.** | A user needs to know what's currently selected — that's real information. Gradient fill + glow is more weight than the information requires; DESIGN.md v1.0 §10 already resolved the target (converge to `.vt.on`'s lighter pattern). |
| Gauges (4 donut rings: 本命優位/上位集中/混戦度/市場一致) | **The 4 numbers are meaningful; the radial-gauge chart type is in tension with DESIGN.md's own product identity.** | These are 4 real confidence metrics computed from the model. But 4 side-by-side colored circular progress rings is a recognizable "generic AI SaaS dashboard" pattern — exactly what DESIGN.md §1 says the product must *not* feel like. This doesn't mean removing the data; it means the chart type is worth reconsidering (e.g., 4 numbers with small inline bars, or a compact stat row) as a Phase 2D candidate. Directly related to the already-DEFERRED KPI-density question — not a new decision, a related one. |
| Tickets (bet-type cards) | **Meaningful; simplifiable.** | Shows actual recommended bets — real, must-keep information. The bet-type color-coded left edge is a legitimate legend and should stay, just as a solid color rather than a gradient fade (see §2). The entrance "stamp" animation and drop shadow are decorative layers on top of meaningful content and are reducible without losing any information. |

---

## 5. Browser observations (this session)

Live-rendered via the existing `.claude/launch.json` "site" config (localhost:8765). Screenshots appear inline earlier in this conversation; summarized here for the record:

- **Desktop race/application view**: race header with the 4-gauge row, member-level spectrum card (gold pointer badge, tier-colored wash), class-stats strip. Confirmed no console errors.
- **Desktop advisor cards**: A/B/C medal circles + 軸/妙味/罠 tags rendering cleanly and legibly — the strongest existing example of "meaningful color coding without excess," worth using as the reference pattern for judge-badge/Grade-Scope-badge simplification in 2C.
- **Mobile (375×812) race/application view**: shutsuba table collapses correctly, mark/frame-color/umaban/name/level-chip/running-style all stay visible and legible at the narrowest tested width; AI指数/勝率/市場 columns preserved per §21.
- **Mobile advisor cards**: single-column, full-width, no overflow or clipping.
- **Explainability view** (`explain.html`, post-Phase-1): numeric columns now render properly monospaced (confirms the Phase 1 font-link fix); card radius and chip radius now visually match the rest of the app.
- One incidental finding, not a real bug: advisor cards briefly render at `opacity:0` when scrolled into view via a fast programmatic scroll — their mount-animation (`animation-delay: calc(var(--i)*60ms + .1s)`) needs ~1–2s to fully resolve under automated interaction. Under normal human scrolling this isn't noticeable; not flagged as a Phase 2 item.

---

## 6. Migration plan

Every item below is a **plan**, not an executed change. Phases are separated so each can ship (and be reviewed/reverted) independently.

### Phase 2B — Radius/token semantics

| # | File | Selector | Current | Proposed | DESIGN.md rule | Visual impact | Regression risk | Verification |
|---|---|---|---|---|---|---|---|---|
| 1 | `style.css` `:root` | — | No `--r-input`/`--r-card`/`--r-pill-true` tokens | Add all three (§1.5 Step A) | §8 Radius | **None** — purely additive | None | n/a |
| 2 | `style.css` | ~15 selectors currently hardcoding `8px` (`.rs-bt`, `.hit-month`, `.rs-back`, `.tk-chip`, `.tk-lineup`, `.skel-bar`, `.dw-stat`, `.tr-top-c`, `.gs-badge`, …) | `border-radius: 8px` | `border-radius: var(--r-card)` | §8 | **None** — identical value, now tokenized | None | Diff-only review (rendered output is byte-identical) |
| 3 | `style.css` | `.rb-waku` | `border-radius: 999px` | `border-radius: var(--r-pill-true)` | §8 | **None** — identical value | None | Diff-only review |
| 4 | `style.css` | `.btn-prime`, `.btn-ghost` | `var(--r-sm)` (2px) | `var(--r-input)` (6px) | §8 "button/input: 6px" | Small — 2 hero buttons get visibly rounder corners | Low — Marketing surface, 2 elements, easy to screenshot-verify | Screenshot hero before/after |
| 5 | `style.css` | `.card`, `.mlvl` | `var(--r-md)` (4px) | `var(--r-card)` (8px) | §8 "application card: 8px" | **Sitewide, simultaneous** — every card everywhere gets visibly rounder corners at once | Medium — highest-visibility single change in this phase; recommend its own dedicated screenshot pass across every view (race, results, drawer, all analytics tabs) before shipping | Full-page screenshots per view, both breakpoints |
| 6 | `style.css` | `.hh`, `.resbar` (mixed-corner radii) | `var(--r-md)` component | `var(--r-card)` component | §8 | Small — top-corner rounding on sticky header / left-corner on result banner | Low | Screenshot table header + result banner |
| 7 | `style.css` | 15 `--r-pill` call sites (chips/badges/`.venue`) | `var(--r-pill)` (4px) | **Decision needed, not proposed here**: either stay compact (rename token only, e.g. `--r-chip`) or move to `var(--r-pill-true)` (9999px, true capsule) | §8 "pill: 9999px" | If moved to true-pill: **large** — every chip/badge/tab sitewide becomes a rounded capsule, a real character change | High if executed as a value change; zero if executed as a rename only | Requires its own side-by-side mockup before a value decision — flagged as an unresolved product decision below, not scheduled |

Items 1–3 are safe to ship as a single commit (zero visual diff). Items 4–6 are visual changes and should each get their own before/after screenshot pair even though individually low-risk. Item 7 is explicitly **not** scheduled — it's a product decision (see §7).

### Phase 2C — Application surface flattening

Drawn from the REDUCE/REMOVE/REPLACE rows in §2. Grouped by mechanism so related selectors ship together:

| # | File | Selector(s) | Current | Proposed | DESIGN.md rule | Visual impact | Regression risk | Verification |
|---|---|---|---|---|---|---|---|---|
| 1 | `style.css` | `.date-wrap:hover`, `.venue:hover`, `.hit-card:hover`, `.adv:hover` | Colored glow shadow on hover | Drop the glow; keep any existing `transform: translateY()` lift | §4 Application: functional-only shadow | Small per-element, but touches 4 different hover states across the app | Low | Hover each in browser, confirm the lift/feedback still reads as "clickable" without the glow |
| 2 | `style.css` | `.venue.on`, `.rpill.on`, `.hit-chip.on` | Gradient fill + glow | Converge to `.vt.on`'s pattern: text-color change + underline/left-accent-bar, no fill | §10 Navigation (list-nav vs. segmented-control distinction) | **Moderate–large** — changes how "currently selected" reads across venue tabs, race pills, and results filter chips everywhere they appear | Medium — this is a real UX-legibility question (does the lighter indicator stay obviously "selected" enough at a glance?), worth a dedicated before/after review, not a blind swap | Screenshot each component with an item selected, both themes of "obvious enough" judged against the current gradient version |
| 3 | `style.css` | `.jbadge.go/.caution/.skip`, `.gs-badge` | Gradient fill + glow + inset highlight | Low-alpha background + colored text + colored border (the `.tdchip`/`.vschip` pattern) | §12 Badges "low-alpha background" | Moderate — judge badges appear on every race header; Grade Scope badge appears once per graded race | Low-medium — color coding and text label are unchanged, so the *information* can't regress, only the weight of its presentation | Screenshot race header for a GO, 慎重, and 見送り race side by side; confirm all three remain distinguishable at a glance |
| 4 | `style.css` | `.card`, `.ticket` (base state) | White sheen gradient + `--shadow-card` | Drop the sheen; lighten or drop the drop-shadow, rely on the existing 1px navy border + surface-level tonal step for elevation | §3/§4 (borders instead of shadow on Application surfaces) | **Sitewide, simultaneous** — every card in the app loses its current "lit from above" polish | Medium-high — this is the single biggest visual-character change in Phase 2C; do this *after* item 2's list-nav convergence lands and is confirmed good, not simultaneously | Full-page screenshots per view; specifically check that cards still read as visually separated from the page background using border + tonal step alone |
| 5 | `style.css` | `.hrow.honmei`/`.intop3`, `.um-row.honmei` | Gradient-fade background wash | Flat, low-alpha background tint (same color, no fade) | §4 (flatten repeated elements) | Small per-row, but repeats down the whole table (up to ~18 rows) | Low | Screenshot a race with a confirmed result showing ◎/top-3 highlighting |
| 6 | `style.css` | `.ticket::before` | 2-stop color→transparent gradient | Solid bet-type color | §23 (no decorative gradients) | Small | Low | Screenshot the ticket grid across a few bet types |
| 7 | `style.css` | `.hit-card` (background only, not `.hit-stamp`) | Gradient tint across the whole card | Flat card background; keep the stamp gradient as-is (explicit exception) | §23, with the win-stamp carve-out preserved | Small–moderate — results tab hit-gallery is a repeated grid | Low | Screenshot results tab hit-gallery before/after |
| 8 | `style.css` | `.mls-mark`, `.tk.tk-hon`, `.adv-medal.gA`, `.vt.on::after`, `.gs-bar` | Various glows/washes flagged REDUCE in §2 | Soften blur radius / opacity roughly by half, don't remove entirely | §4 | Small, subtle | Low | Visual spot-check, not expected to be noticeable enough to need formal before/after |

Recommended commit grouping within 2C: **(1) hover-glow removal → (3) badge flattening → (5)+(6)+(7) gradient softening → (2) list-nav convergence → (4) card/ticket base shadow**, in roughly that order — starting with the lowest-risk/most-isolated items and ending with the two that touch the whole app simultaneously, so each commit can be independently verified before the next raises the stakes.

### Phase 2D — Race table/data-density refinement

| # | File | Selector | Current | Proposed | DESIGN.md rule | Visual impact | Regression risk | Verification |
|---|---|---|---|---|---|---|---|---|
| 1 | `style.css` | `.hrow` | `padding: 7px 14px`, row height not directly measured | Measure actual rendered height live (two-line vs. one-line rows); adjust padding only if meaningfully over §13's 40–44px target | §13 | Unknown until measured — plan is "measure first," not "change first" | n/a until measured | `javascript_tool`: `getBoundingClientRect().height` on a sample of rows, both content densities |
| 2 | `style.css` | `.hh`/`.hrow` grid gaps, various table-internal padding | Off-grid values (7px, 9px, 13px) | Snap to the 4px grid (4/8/12/16) where it doesn't require a column-width renegotiation | §7 Spacing | Small, cumulative | Low-medium — table column widths are already tightly budgeted across 7 breakpoints; any padding change needs a full responsive re-check | Screenshot all 7 breakpoints |
| 3 | `style.css` + `app.js` (`donut()`) | `.gauges`/`.gauge`/`.g-*` (4-donut row) | 4 side-by-side radial progress gauges | **Not proposed here — flagged for a product decision.** Candidate alternative: 4 numbers in a compact stat row with small inline bars, consistent with §19's "data over decorative visualization" | §1 (must not feel like generic AI SaaS), §19 | **Large** — this is the race header's most visually prominent element | High — changes a core, always-visible piece of the race view; needs a mockup and explicit sign-off before any code, not a Phase-2D default | Requires a dedicated before/after comparison, likely its own mini-review before scheduling |
| 4 | — | Card-vs-table usage in results/advisor/pedigree tabs | Card grids | **Not proposed here** — same DEFER as Phase 0/0.5's KPI-density question | §18/§13 | — | — | — |

---

## 7. Decision Log

| Item | Outcome |
|---|---|
| Radius audit (§1) | 29 tokenized call sites classified; ~46 additional hardcoded values inventoried for context; 4 explicit name/semantic mismatches identified (`--r-pill` isn't a pill; `--r-sm` conflates 3 roles; `--r-md` undershoots its own dominant real-world value; circles are a separate category) |
| Shadow/glow/gradient inventory (§2) | 42 occurrences classified — 18 KEEP / 16 REDUCE / 5 REMOVE / 3 REPLACE — with the data-visualization bars (AI-index, front-rate, lap-split) explicitly separated out as *not* decorative |
| Race/data UI audit (§3) | 14 sub-aspects checked; 11 compliant, 1 justified exception (hname/pwin sizing), 1 borderline needing live measurement (row height), 1 unresolved (spacing grid, unchanged since Phase 0) |
| Judge/advisor/status UI (§4) | Judge badges and Grade Scope badge: information meaningful, presentation excessive → flatten. Advisor medals and status chips: already good, minor/no change. Active pills: selection state meaningful, weight excessive → converge to existing light pattern. Gauges: data meaningful, chart type in tension with product identity → flagged, not resolved. Tickets: meaningful, simplifiable. |
| Migration plan | Split into 2B (radius/tokens, mostly zero-risk plus one flagged product decision), 2C (surface flattening, 8 grouped items with a recommended commit order), 2D (table density — 2 measurement/cleanup items plus 2 explicitly-flagged-not-proposed product decisions) |
| Production code | **Untouched.** No `site/` file was edited this phase. |

## Unresolved product decisions (not Phase 2A's to make)

1. **`--r-pill` future value** (Phase 2B item 7): rename-only (stays compact) vs. true 9999px pill sitewide across every chip/badge/tab. A real character decision, needs a mockup.
2. **Gauge row redesign** (Phase 2D item 3): keep the 4-donut radial-gauge pattern, or move to a flatter stat-row presentation. Directly tied to DESIGN.md §1's "must not feel like generic AI SaaS" identity goal, but changes the race header's most prominent element — needs explicit sign-off, not a default.
3. **Card-vs-table usage in results/advisor/pedigree tabs** — still the same open question from Phase 0/0.5 (KPI-card density, wins-only gallery), untouched here.
4. **List-nav visual-weight tradeoff** (2C item 2): confirming the lighter `.vt.on`-style indicator stays legible enough as "selected" once applied to venue tabs and race pills, which see far more use than view tabs do.

## Recommended Phase 2B scope

Given the above, the safest, highest-value next slice is: **Phase 2B items 1–3 only** (add the three new tokens, tokenize the ~15 already-matching `8px` sites and the one `999px` site) — a single commit, zero visual diff, fully mechanical, and it clears the ground for items 4–6 (the real visual radius decisions) to be scoped and reviewed on their own once you've seen this land cleanly. Items 4–6 and the Phase 2C/2D work all still need your go-ahead before any code changes, per this phase's own instructions.

---

## Git status

```
$ git status --porcelain -- site/
```
(see accompanying chat message for live output)
