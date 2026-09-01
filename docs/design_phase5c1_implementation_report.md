# PyCaLiAI Design System — Phase 5C-1: Command Header Seam Cleanup

> Fixes only the bounded seam identified in the Phase 5C preview — the bare "—" judge badge below the new headline. Structure B itself is unchanged; no redesign, no new metrics, no sticky behavior.
> Date: 2026-09-01

## Change

**`site/js/app.js`**, `renderHeader()`: the `.judge` block is now wrapped in `${j.category ? \`...\` : ""}`. Since `build_site.py`'s `scrub_public()` strips `judgment.category`/`.headline`/`.detail`/`.waku_tag` **together, atomically**, based on the same "no result yet" condition (confirmed in the Phase 5B-0 trace), the presence of `category` is a reliable, exact proxy for "there is a real judge verdict to show at all." When absent, the entire `.judge` row is omitted — not just the bare badge — because the only other thing it could show pre-result (`hardness`) is already stated in the `.cmd-state` line directly above it; a judge row containing nothing but a duplicate of already-shown information would itself be a second, smaller instance of the same "filler" problem. No redacted field is read or exposed; no replacement copy was written.

**`site/css/style.css`**: the divider (hairline + spacing) that used to live on `.judge`'s `border-top`/`padding-top` was moved to `.cmd-state`'s `border-bottom`/`padding-bottom`, since `.cmd-state` is now the one element guaranteed to be the last thing in the new headline section regardless of whether `.judge` renders after it. `.judge` keeps only its `margin-top`; `.gauges`' own `margin-top` (already 14px, untouched) now supplies the correct gap whether it follows `.judge` or follows `.cmd-state` directly. No other rule touched — gauges, member-level, and class-prior styling are exactly as they were.

## Verification

Tested both states on the same set of races used throughout this project (現行 2026-08-30 for pre-result, 2026-08-16 for a settled race with real judge data), both viewports.

| Check | Pre-result (no `judgment.category`) | Post-result (`judgment.category` present) |
|---|---|---|
| `.judge` rendered? | **No** — confirmed absent from the DOM entirely | **Yes** — full content: badge ("GO"), headline+detail, hardness tag, waku_tag, byte-identical to before this change |
| Visual | Divider flows directly from the state line into the gauges row, no gap, no placeholder | Unchanged from Phase 5B-2/5B-3A — badge, tags, and detail text all present and correctly styled |

### First-viewport balance
Unchanged in spirit from the Phase 5B-2 report — race identity, ◎/WIN/TOP3, and the state line all sit within the first 375×812 viewport comfortably. What changed is everything *after* them now starts noticeably sooner.

### Distance from headline to secondary detail region
Desktop (1440×900, pre-result race): the divider now sits at ~449px from the header top, with the gauges row beginning at 483px — a single, consistent ~34px gap (padding + border + margin), whether or not a judge row is present in between. No more double-spaced or orphaned-empty-row gap.

### Do gauges visually compete with ◎/WIN/TOP3?
No. The gauges are unchanged in size/treatment and now follow immediately and cleanly after the divider — they read unambiguously as supporting detail, not as a second competing headline. This was already mostly true after Phase 5B-2's reordering; removing the empty judge row removes the one remaining thing that looked unresolved about that transition.

### Table starting position
**Measured improvement, not just a wash.** Mobile (375×812), same race, same methodology as the Phase 5B-2 report: the shutsuba table's own heading (`.sh-head`) now begins at **y≈1329px**, versus **y≈1393px** measured for the identical race/viewport in the Phase 5B-2 report — a **64px reduction**, exactly matching the height of the empty row that was removed. The "cost" disclosed in Phase 5B-2 (table pushed down by the new headline content) is now partially recovered.

### Empty-state behavior pre-result
Confirmed: no bare "—", no "N/A", no invented placeholder copy of any kind. The row simply does not exist when there's nothing real to show in it.

### Post-result behavior where judge data exists
Confirmed on a real settled race (2026-08-16, 札幌1R, テンレッドサン ◎): full judge row renders exactly as it did before this phase — badge, headline/detail narrative, hardness tag, waku tag, all present, all correctly styled, all unchanged.

### Other
- Console clean, confirmed via a fresh single-click tab.
- One network 404 was observed while testing (`data/changes_20260816.json`) — investigated and ruled out as pre-existing, unrelated behavior: `loadChanges()` fetches a same-day-changes file that legitimately doesn't exist for historical dates, and the existing code already handles this gracefully (silent no-op on a non-JSON/404 response). Not touched by, or related to, this phase's change.
- Numerical values unchanged throughout (◎エリカビアリッツ 21.1%/46%, ◎テンレッドサン 23.8%/50%, matching prior reports).

## PASS/FAIL

| Item | Result |
|---|---|
| Bare "—"/placeholder badge no longer renders pre-result | **PASS** |
| No redacted field exposed, no invented replacement copy | **PASS** |
| Post-result judge content unchanged | **PASS** |
| Divider/spacing gap closed cleanly with a small, targeted CSS adjustment | **PASS** |
| Gauges/member-level/class-prior semantics unchanged | **PASS** |
| Headline not enlarged further, no new gold/glow, no new metrics | **PASS** |
| Sticky header not implemented | **PASS** (not attempted) |
| Console clean | **PASS** |

`git diff --stat -- site/`: `style.css` (2 lines), `index.html` (cache-bust only), `app.js` (2 lines) — a genuinely small, bounded change, consistent with the seam being a bounded seam rather than a deeper problem with Structure B.
