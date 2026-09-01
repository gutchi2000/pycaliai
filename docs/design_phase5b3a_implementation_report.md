# PyCaLiAI Design System — Phase 5B-3A: Remaining `vs_market` Removal

> Implements the audited minimal removal from `docs/design_phase5b3_audit_and_5c_preview.md` §5B-3A exactly. Not implemented until this phase, per prior instruction.
> Date: 2026-09-01

## Changes

**`site/js/app.js`**
- `renderTable()`: removed the `.hh` header span `<span class="ta-r c-vs">市場</span>` and the per-row span `<span class="ta-r c-vs">${vsChip(h.vs_market)}</span>`.
- `umamiTableHtml()`: removed the header span `<span class="um-reason">市場</span>` and the per-row span `<span class="um-reason">${vsChip(h.vs_market)}</span>`.
- `vsChip()` itself is now unused (its third and last call site was already removed in Phase 5B-2). Left defined, per the same minimal-footprint reasoning already applied to `.dw-odds`'s CSS rule in that phase.

**`site/css/style.css`**
- Dropped the trailing vs-column width from all 8 shutsuba `.htable.hasres`/`.htable.nores` `grid-template-columns` declarations (base, ≤1000px, ≤880px, ≤560px).
- Dropped the trailing `minmax(0,2fr)` from the one base `.um-head, .um-row` declaration. The ≤880px UMAMI override needed no change — `.um-reason` was already excluded from it by a pre-existing, unrelated rule.

**`site/index.html`**: cache-busting bump only (`app.js?v=20260901c`, `style.css?v=20260901b`).

No Python file touched. `h.vs_market` remains present in `data/{date}.json`, untouched.

## Verification

| Check | Desktop (1440px) | Mobile (375px) |
|---|---|---|
| `.hh` grid track count | 9→8 (`32px 56px minmax(0,1fr) 44px 158px 70px 62px 78px`) | 6→5 (`18px 26px minmax(0,1fr) 34px 36px`) |
| Name column width, before → after | 514px → 587px (+73px = the removed 64px column + 9px freed gap, exactly as predicted by CSS Grid mechanics) | grew proportionally, no overflow |
| Row height | 65.30px — unchanged from every prior baseline of this race | 53.05px — exact match to the value established in the Phase 2D `.hsub` sweep |
| Empty gap where column was | None — screenshot confirms rows end cleanly after 勝率, no trailing space | None — confirmed via screenshot |
| UMAMI `.um-head` grid track count | 8→7 (`30px 52px minmax(96px,1.1fr) 56px 56px 42px 56px`) | unchanged (5 tracks) — was already `vs_market`-free |
| UMAMI name column width | grew to 788px (previously sharing flexible space at roughly 1.1-to-2 against the removed 2fr track, now takes the full remainder) | unchanged (mobile UMAMI layout untouched) |
| Page horizontal overflow | none | none |
| Console errors | clean (verified via a fresh single-click tab, isolated from this session's scripted-click noise) | clean |
| Numerical values unchanged | ◎エリカビアリッツ: 21.1% WIN / 46% TOP3 in both the Command Header and the shutsuba row — identical to Phase 5B-2's report | same |

Both tables verified visually via screenshot; the shutsuba table screenshot at 375px shows all 13 rows rendering cleanly with no trailing artifact.

## PASS/FAIL

| Item | Result |
|---|---|
| Shutsuba table — header + row cell removed, no empty gap, geometry balanced | **PASS** |
| UMAMI table — header + row cell removed, no empty gap, name column grows | **PASS** |
| No replacement metric introduced | **PASS** |
| Upstream `vs_market` generation untouched | **PASS** |
| No row-height regression | **PASS** |
| No console errors | **PASS** |
| No unrelated responsive behavior changed | **PASS** (only the 9 grid declarations named above were touched) |

`git status --porcelain -- site/`: `M site/css/style.css`, `M site/index.html`, `M site/js/app.js` (not yet committed — see next message).
