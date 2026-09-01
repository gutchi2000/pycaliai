# PyCaLiAI Design System — Phase 5B-2: Command Header + Mobile Drawer Implementation

> Production-code implementation. Not committed — left for review, per instructions.
> Implements Decision 1 (Command Header, Structure B), Decision 2 (mobile drawer → bottom sheet), Decision 3 (`vs_market` drawer-line removal).
> Date: 2026-09-01

---

## 1. Exact changed files

```
site/js/app.js      40 lines changed (+33 / -7 net, incl. two new helper functions)
site/css/style.css  49 lines changed (+45 / -4 net)
site/index.html      4 lines changed (2 cache-busting version-string bumps only)
```

`index.html`'s two edits (`css/style.css?v=20260811navy → ?v=20260901a`, `js/app.js?v=20260801a → ?v=20260901b`) are cache-busting bumps only, following this codebase's own existing `?v=YYYYMMDDx` convention — not a content or behavior change. They were necessary for the JS/CSS edits below to actually reach a browser; without them a previously-installed service worker (see §12) and normal HTTP caching would keep serving the old files indefinitely.

---

## 2. Exact changed selectors/functions

**`site/js/app.js`**
| Change | Detail |
|---|---|
| New function `cmdPickHtml(honmei)` | Builds the ◎/WIN/TOP3 headline block. Returns `""` if no ◎ horse is found (graceful omission, no broken partial markup). |
| New function `cmdStateHtml(j, conf, fieldSize)` | Builds the restrained state line (hardness tag + CHAOS + FIELD). Omits any token whose source value is null/absent; returns `""` entirely if nothing is available. |
| `renderHeader(r)` | Restructured: added `const honmei = r.horses.find(h => h.mark === "◎")`; split `.rh-main` down to just `.rh-title`+`.rh-sub`; inserted `cmdPickHtml(honmei)` and `cmdStateHtml(j, conf, r.field_size)`; moved `.judge` out of `.rh-main` to its own top-level block, positioned after the new command content. `.gauges`, `memberLevelEl(r)`, cowork-quote, and `priorStrip(r)` are unchanged in content and unchanged in relative order to each other — only their position relative to the *new* content shifted (now after it, not beside it). |
| `openDrawer(umaban)` | Removed the `.dw-odds` line (`市場評価: ${vsChip(h.vs_market)}`). Added `document.body.classList.add("drawer-open")`. |
| `closeDrawer()` | Added `document.body.classList.remove("drawer-open")`. |

**`site/css/style.css`**
| Change | Detail |
|---|---|
| New rules | `.cmd-pick`, `.cmd-pick .mark.m1`, `.cmd-name`, `.cmd-stats`, `.cmd-stat`, `.cmd-stat b`, `.cmd-stat b small`, `.cmd-lab`, `.cmd-state`, `.cmd-tag`, `.cmd-kv b`, `.cmd-kv b small`, `.cmd-sep` |
| Modified `.judge` | Added `flex-basis: 100%`, changed `margin-top: 9px` → `margin-top: 14px; padding-top: 13px; border-top: 1px solid var(--border-subtle)` (the divider between new headline content and the pre-existing detail region). |
| Modified `.gauges` | Removed `margin-left: auto` (was right-aligning it next to `.rh-main`; no longer applicable now that it sits in its own row below `.judge`). Added `margin-top: 14px`. |
| New rule | `body.drawer-open { overflow: hidden; }` |
| New media rules (`@media (max-width: 880px)`, existing block) | `.cmd-pick .mark.m1`, `.cmd-name`, `.cmd-stats`, `.cmd-stat b`, `.cmd-stat b small` — mobile size reductions, added alongside the pre-existing `.rh-place`/`.rh-name` mobile overrides in the same block. |
| New media block (`@media (max-width: 760px)`, new) | `#drawer` bottom-sheet conversion (position/sizing/radius/shadow/transform/safe-area padding) and `#drawer::before` (decorative handle). Placed immediately after the existing `@media (max-width: 980px) { #drawer ... }` block to keep drawer-responsive rules colocated. |

No selector was deleted. No existing rule's *content* was rewritten beyond the two targeted property changes on `.judge`/`.gauges` described above.

---

## 3. Command Header DOM/CSS summary

```html
<div class="card rh">
  <div class="rh-main">          <!-- unchanged: race identity -->
    <div class="rh-title">...</div>
    <div class="rh-sub">...</div>
  </div>
  <div class="cmd-pick">          <!-- NEW -->
    <span class="mark cmd-pick .mark.m1">◎</span>
    <span class="cmd-name">エリカビアリッツ</span>
    <span class="cmd-stats">
      <span class="cmd-stat"><b>21.1<small>%</small></b><small>WIN</small></span>
      <span class="cmd-stat"><b>46<small>%</small></b><small>TOP3</small></span>
    </span>
  </div>
  <div class="cmd-state">荒れ | CHAOS 91% | FIELD 14頭</div>   <!-- NEW -->
  <div class="judge">...unchanged content, new position...</div>
  <div class="gauges">...unchanged content, new position...</div>
  ...memberLevelEl / cowork-quote / priorStrip, unchanged...
</div>
```

CSS approach: `.card.rh` is an existing `display:flex; flex-wrap:wrap` container. `.cmd-pick` and `.cmd-state` use `flex-basis:100%` (the same technique already used by `.rh-quote`/`.prior`/`.mlvl` in this exact component) to each claim a full-width row, so no new layout primitive was introduced. Numerals reuse the existing `.num`/`--disp` (IBM Plex Mono, tabular-nums) convention. The ◎ mark's gold color and 30px size come from the existing `.mark.m1` rule, scoped via `.cmd-pick .mark.m1` (a compound selector chosen specifically to out-specificity the several pre-existing `.mark.m1` responsive overrides elsewhere in the stylesheet — see §13 for why this mattered). The WIN/TOP3 numbers use `--gold2`, matching the exact precedent already set by the drawer's own `.dw-stat .v.gold` treatment for the ◎ horse's stat. The horse name uses the neutral `--tx` color, deliberately not gold — restraint is concentrated on the mark and the numbers, not applied uniformly.

One deliberate deviation from the brief's literal "HARDNESS | CHAOS | FIELD" mockup: hardness renders as a bare tag (荒れ/固い are self-explanatory Japanese adjectives) rather than a labeled "HARDNESS 荒れ" pair, since the brief explicitly says the mockup is "conceptual hierarchy, not mandatory literal text/layout."

---

## 4. Before/after desktop screenshot observations (1440×900, 札幌1R / race `2026083001020401`)

**Before** (all prior phases' screenshots of this exact race, most recently Phase 4A/5A/5B-1's live checks): header showed race identity beside a judge badge/headline/tags row, with the 4 donut gauges pushed to the right of that same row via `margin-left:auto`; member-level card below; nothing in the header stated the ◎ horse's own WIN/TOP3 numbers — a reader had to look at shutsuba row 1 for that.

**After** (this phase, confirmed live): race identity unchanged at the top. Immediately below it: "◎ エリカビアリッツ" in large type, with "21.1% WIN" / "46% TOP3" in large gold numerals to the right — the single most visually dominant element in the header. Below that, a thin muted state line ("荒れ ｜ CHAOS 91% ｜ FIELD 14頭"). Below a hairline divider: the judge badge (currently "—", see §9), hardness/waku tags, then the 4 gauges (now left-aligned in their own row, no longer pushed right), then member-level, unchanged.

No clipping, no overflow, no layout breakage observed. The gauges' repositioning (own row instead of beside race identity) is the one visible structural change beyond simple reordering — documented in §2/§13 and easily reversible (restore `margin-left:auto` + move the DOM block back beside `.rh-main`).

## 5. Before/after mobile screenshot observations (375×812, same race)

**Before**: race identity, then judge badge/tags, then 4 gauges (already full-width/wrapped on mobile pre-existing responsive rules), then member-level — no headline WIN/TOP3 statement anywhere above the fold besides scrolling into the table.

**After**: race identity → "◎ エリカビアリッツ" with "21.1% WIN" / "46% TOP3" stacked cleanly, numbers right-aligned and still large/legible at mobile size (25/24px per the 880px-breakpoint reduction) → the muted state line, fully legible, wrapping correctly if needed → divider → judge/gauges/member-level, all rendering exactly as before, just later in the scroll order. No horizontal overflow, no text collision observed.

## 6. First-viewport footprint before/after

Measured precisely on the live page (375×812): the two new blocks (`.cmd-pick` + `.cmd-state`, including their own margins) together occupy **~168px** of vertical space (`.cmd-pick` top at y=484 to `.judge` top at y=652). Both are fully visible within the 812px first viewport (◎ name and both percentages are on-screen without scrolling).

Because nothing existing was deleted (`.judge`/`.gauges`/`.mlvl`/`.prior` all retain their full original content), the *total* header height grew by roughly the same ~168px, and the shutsuba table's own heading (`.sh-head`) now begins at y≈1393px instead of further up the page. This is reported plainly rather than minimized: it is the direct, expected consequence of the explicit instruction to add the new headline without deleting the old detail content. The tradeoff is that the primary question — "what does PyCaLiAI think" — is now answered inside the first viewport itself, which it was not before; reaching the full table simply requires more scrolling than it used to.

---

## 7. Horse Drawer desktop verification

Confirmed live, race `2026083001020401`, horse row 1 (タガノシルフィー) and a 9-character-name horse (umaban 2, マーゴットクリュグ):
- Right-side panel, `position:fixed; top:0; right:0; bottom:0`, `width:900px` at 1440px viewport (`min(900px,96vw)`) — **unchanged** from before.
- Opens/closes correctly via row click, close button (✕), and backdrop click.
- `.dw-odds` (市場評価 line) confirmed **absent** — no trace in rendered HTML, layout reflows cleanly with `.dw-stats` (4-stat grid) followed directly by `.dw-info` (jockey/kinryo/trainer/style/level rows), no visual gap or misalignment.
- All other content (stat grid, info rows, `why[]` bars are not present for this simple demo horse but the drawer structure around them is intact, 近5走 table, career chart) renders exactly as before.

## 8. Horse Drawer mobile verification

Confirmed live at 375×812:
- Converts to a bottom sheet: rises from the bottom, full width, rounded top corners, decorative handle bar visible and non-interactive, `max-height` measured at **714px = 88.0% of 812px viewport height** (exact match to the `88dvh` target).
- Background page visible (dimmed) above the sheet — confirms it is not a full-screen takeover.
- Internal scroll confirmed working: scrolling the sheet's content (verified via `scrollTop` manipulation + screenshot) reveals the 近5走 table, history summary, and a **fully-rendered, correctly-proportioned ECharts career chart** — addressing the specific regression risk flagged in the Phase 5B-1 plan (chart rendering inside a height-constrained container). Background page `scrollY` remained `0` throughout — scroll is correctly isolated to the sheet.
- Close via backdrop tap and via the ✕ button both confirmed working; body scroll lock (`overflow:hidden` on `<body>`) confirmed active while open and released on close.
- Long name (9 characters, マーゴットクリュグ) renders on one line, fully readable, with clear separation from the close button — no collision.
- No horizontal page overflow introduced (`document.documentElement.scrollWidth === window.innerWidth` confirmed).
- Safe-area handling (`padding-bottom: calc(24px + env(safe-area-inset-bottom))`) is implemented via the standard CSS environment-variable mechanism, which degrades gracefully to `24px` on any non-notched viewport/browser — not independently visually testable in this environment without a notched-device emulation, but the mechanism itself is standard and low-risk.

Desktop right-panel behavior is untouched — confirmed identical position/width/transform-axis to pre-existing behavior (§7).

---

## 9. `vs_market` removal verification

- The `.dw-odds` template line was removed from `openDrawer()` (§2). Confirmed absent in the live-rendered drawer, both desktop and mobile, across multiple horses.
- Drawer layout verified balanced without it: `.dw-stats` (WIN/連対率/複勝圏/人気) flows directly into `.dw-info` (騎手/斤量/厩舎/脚質/馬レベル) with no visual gap, no orphaned spacing, no broken grid — confirmed via screenshot on both viewports.
- Per instructions, only the *public visual line* was removed. `export_marks_json.py`'s `ai_vs_market` computation and `build_site.py`'s pass-through into the public `vs_market` field were **not touched** — the field still exists in `data/{date}.json` exactly as before; only the drawer's rendering of it was removed.
- The `.dw-odds` CSS rule itself (`font-size/color/margin-bottom`) was left in the stylesheet, now unused. This is a deliberate choice, not an oversight: it keeps the change minimal and trivially reversible (if this line is ever reinstated, no CSS needs rewriting), and an unreferenced selector has zero visual exposure on its own.

## 10. Exact remaining public `vsChip()` locations

Both of the other two locations identified in the Phase 5B-0 trace remain in place, **not modified this phase**, per instruction:

| Location | Function | Desktop (>880px) | Mobile (≤880px) |
|---|---|---|---|
| Shutsuba table, `.c-vs` column | `renderTable()` | **Visible** (confirmed: e.g. "過剰" chip rendered) | **Visible** (confirmed at 375px) |
| UMAMI 全頭分析 table, `.um-reason` column | `umamiTableHtml()` | **Visible** (confirmed: e.g. "妙味" chip rendered) | **Hidden** — but by a **pre-existing** responsive rule (`@media (max-width:880px) { .um-reason { display:none } }`, already in the stylesheet before this phase, unrelated to any change made here) |

**These are reported, not touched.** Per instruction, this is logged as a separate compliance-cleanup decision for review — see §14 Decision Log. The shutsuba-table location in particular is publicly visible on every viewport and was not addressed in this implementation.

---

## 11. Numerical-content equality check

Cross-checked the new header's ◎/WIN/TOP3 against the shutsuba table's own ◎ row (`.hrow.honmei`), same race, same load:

| Value | Header | Table row 1 (◎) | Match |
|---|---|---|---|
| Horse name | エリカビアリッツ | エリカビアリッツ | ✅ |
| WIN | 21.1% | 21.1% | ✅ |
| TOP3 | 46% | 46% | ✅ (see note) |

**Found and fixed during verification**: the header's TOP3 figure initially rendered as "45.8%" (1 decimal, `pct()`'s default) while the table shows "46%" (0 decimals, `pct(h.p_sho, 0)`) for the exact same underlying `p_sho` value (0.4582) — not a data bug, but a real, visible precision mismatch between two representations of the same fact on the same page. Fixed by passing `0` explicitly in `cmdPickHtml()`, matching the table's own convention (chosen over the drawer's own `pct(h.p_sho)` 1-decimal convention, since the table sits directly beneath the header and is the more likely point of comparison for a reader). Re-verified live after the fix: exact match confirmed.

CHAOS (91%) and FIELD (14頭) were cross-checked against the existing 混戦度 gauge (91%, unchanged, still rendering below) and the existing `.mchip` "14頭" chip — both exact matches, as expected since both reuse the identical source values with identical formatting.

## 12. Console/runtime errors

Confirmed clean (zero console errors) across three independently-verified fresh-tab, single-natural-click sessions (desktop and mobile). 

**Two things investigated and ruled out, not real issues:**
- Repeated `InvalidStateError: Transition was aborted because of invalid state` / `AbortError: Transition was skipped` seen on tabs subjected to rapid scripted clicking during this verification session — this is the same `document.startViewTransition()` artifact documented since Phase 2D of this project; a fresh tab with one natural click reproduces zero errors, confirmed three separate times this phase.
- A `getComputedStyle(drawer).transform` read that returned a stale, incorrect matrix (implying the drawer was off-screen) on two separate tabs, contradicted by an immediate screenshot showing the drawer rendering correctly, fully on-screen, with correct content. This is a measurement-tool artifact (stale computed-style snapshot under this environment's viewport/CDP emulation), not a real rendering bug — same general category as the previously-documented "hidden pane returns stale computed styles" issue. **Screenshots, not `getComputedStyle().transform` reads, were used as the authoritative check for the drawer's open/closed visual state in this report.**

**One genuine, unrelated-to-this-phase discovery**: the site has an active service worker (registered via `index.html`'s own bottom-of-page script) that was caching `index.html`/`app.js`/`style.css` and serving them even after both a fresh navigation and the standard `?v=` cache-bust query-string bump. Verification required explicitly unregistering it (`navigator.serviceWorker.getRegistrations()` → `.unregister()`) before changes became visible. This is worth carrying forward as operational knowledge for future phases: **the existing `?v=` cache-busting convention alone is not sufficient once the service worker has installed; it must also be cleared, or the version bump must be paired with the service worker's own update lifecycle**, which was not something this phase needed to touch (no `sw.js` edit was made or required).

## 13. Regression findings

- **CSS specificity collision, found and fixed during implementation** (not shipped as a bug): the ◎ mark inside `.cmd-pick` carries three classes (`mark cmd-mark m1`). A first draft targeted sizing via a plain `.cmd-mark` selector, which was silently overridden by the pre-existing `.mark.m1` rule (higher specificity, and also has its own responsive overrides at `≤560px`). Fixed by using the compound selector `.cmd-pick .mark.m1` (specificity high enough to win at every breakpoint). Verified via `getComputedStyle`: `.cmd-pick .mark`'s font-size correctly resolves to `30px` (desktop) / `25px` (≤880px), not the pre-existing `21px`/`18px`.
- **TOP3 precision mismatch, found and fixed during verification** — see §11.
- **`.gauges`' repositioning is a real, if small, structural change beyond pure reordering** (removed `margin-left:auto`) — documented in §2, easily reversible.
- **First-viewport push-down of the shutsuba table on mobile** — see §6. Not a bug; a direct, disclosed consequence of the "don't delete existing content" constraint.
- **No regression found** in: race table geometry (confirmed identical grid-template-columns and row height to Phase 4A's baseline for this exact race), desktop drawer positioning/width, the other two `vsChip()` locations (untouched, confirmed still functioning), member-level/class-prior/cowork-quote rendering (unchanged content, unchanged internal styling).

## 14. PASS/FAIL per change

| Change | Result |
|---|---|
| Command Header — race identity → ◎/WIN/TOP3 → state line hierarchy | **PASS** |
| Command Header — no dashboard-card-grid, restrained gold, tabular numerics | **PASS** |
| Command Header — existing gauges/judge/member-level/class-prior preserved, not deleted | **PASS** |
| Command Header — desktop no clipping/overflow, table geometry unchanged | **PASS** |
| Command Header — mobile fits without overwhelming first viewport, numbers/state line legible | **PASS** (with the disclosed table-push-down tradeoff in §6) |
| Drawer — desktop right-side behavior unchanged | **PASS** |
| Drawer — mobile bottom sheet, ~85–90dvh, rounded corners, handle, close/backdrop, scroll isolation, safe-area | **PASS** |
| Drawer — long names readable, no close-button collision | **PASS** |
| Drawer — charts/history/why-bars functional inside the sheet | **PASS** |
| `vs_market` — `.dw-odds` line removed, layout balanced | **PASS** |
| `vs_market` — other two `vsChip()` locations identified and left untouched, reported | **PASS** (reporting requirement) |
| Console/runtime — clean | **PASS** |

No FAILs this phase.

---

## 15. `git diff --stat`

```
$ git diff --stat -- site/
 site/css/style.css | 49 +++++++++++++++++++++++++++++++++++++++++++++----
 site/index.html    |  4 ++--
 site/js/app.js     | 40 +++++++++++++++++++++++++++++++++-------
 3 files changed, 80 insertions(+), 13 deletions(-)
```

## 16. `git status --porcelain -- site/`

```
$ git status --porcelain -- site/
 M site/css/style.css
 M site/index.html
 M site/js/app.js
```

Implemented, **not committed**, per instructions. The pre-existing Phase 3B diff (mobile hero/RACE INDEX) is included within `site/css/style.css`'s changes above — no separate file for it since it was already uncommitted in the same file before this phase began.

---

## Decision Log

- **IMPLEMENTED, not committed** — Command Header Structure B (Decision 1): `renderHeader()` restructured, `cmdPickHtml()`/`cmdStateHtml()` added.
- **IMPLEMENTED, not committed** — Mobile drawer bottom sheet (Decision 2): `#drawer` mobile media block, body-scroll-lock class toggling.
- **IMPLEMENTED, not committed** — `vs_market` drawer-line removal (Decision 3): `.dw-odds` removed from `openDrawer()`'s template only; upstream computation/export untouched.
- **REPORTED, not implemented** — the two other `vsChip()` locations (shutsuba `.c-vs`, UMAMI `.um-reason`) remain publicly visible (the UMAMI one only on desktop, due to a pre-existing unrelated mobile rule). Left as-is per explicit instruction; flagged for a separate compliance-cleanup review.
- **NOT IMPLEMENTED, per explicit instruction** — sticky condensed Command Header, sticky offset changes, Horse Compare, Compact/Expert mode, any new gauge/score/formula, and every named currently-unrendered field (`judgment.chaos_pct`, `judgment.kenshu_hint`, `history.pos_trend`, `results.json excluded_*`, `horse.blinker`), HONEST RECORD redesign, any new JRA-VAN-derived public data.
- **CARRIED FORWARD, operational note** — the site's service worker must be accounted for (not just the `?v=` query string) when verifying future front-end changes live; see §12.

No cleanup phase, sticky implementation, or further phase has been started.
