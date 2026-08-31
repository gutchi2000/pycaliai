# PyCaLiAI Design System — Phase 2C Plan (Application/Data Surface Flattening)

> Planning only. No `site/` file was edited to produce this document. Every "current CSS" quote below was re-verified against the live file this turn (line numbers reflect the post-Phase-2B state), not pulled from memory.
> Date: 2026-08-31

---

## 0. Method

Drew directly from `docs/design_phase2a_audit.md`'s §2 classification (KEEP/REDUCE/REMOVE/REPLACE, 42 occurrences). Every candidate below traces to a specific row in that inventory — nothing here is a new classification. Organized against this turn's three priorities, then collapsed into one candidate table with exact current/proposed CSS and a SAFE/MODERATE/HIGH rank per item.

---

## 1. Priority 1 — Decorative excess

### Judge badge excessive glow/gradient
`.jbadge.go` / `.jbadge.caution` / `.jbadge.skip` — style.css:497–499. Phase 2A classification: **REDUCE→REPLACE**. Each is a 2-stop gradient fill + 2-layer box-shadow (inset highlight + colored glow). `.jbadge.na` (line 500) is already the flat target pattern — `background: var(--card2); color: var(--tx3);`, no gradient, no shadow — and is left untouched as the reference.

### Grade Scope badge excessive presentation
`.gs-badge` — style.css:1127–1132 (verified in Phase 2B verification pass). Gradient fill + a standalone glow shadow, same family as the judge badges.

### Purely decorative box-shadows
Hover-only glows with no informational content, where the *other* hover cues (border-color change, translateY lift) already communicate "interactive": `.date-wrap:hover`, `.venue:hover`, `.adv:hover`, `.hit-card:hover`.

### Purely decorative gradients
`.ticket::before`'s color→transparent fade (a bet-type color key — the color itself is informational, the *fade* is not) and the `.card`/`.ticket` base sheen (already flagged HIGH in Phase 2A/2B — see §3).

### Redundant multi-layer active-state glow
Three selected-state rules stack an inset highlight (the "embossed" cue) with a *second*, separate colored glow layer that adds nothing the fill+border don't already say: `.modenav button.on`, `.venue.on` (both inset+outer), `.rpill.on` (outer only, no inset to begin with — the whole shadow is the redundant layer since the gradient fill already carries the state). `.vt.on::after` (the active view-tab underline) also carries a glow beneath an already-sufficient gradient underline bar. `.tk.tk-hon` (想定隊列's "this chip is ◎" marker) glows on top of an already-sufficient border+background tint.

---

## 2. Priority 2 — Selected/status states

| Component | Current state cue | After removing excess glow, state is still obvious via |
|---|---|---|
| Judge states (`.jbadge.*`) | Gradient fill + glow | Text label (GO/慎重/見送り, literal words) **and** color-coded background+border+text — three redundant signals, never color-only |
| Mode states (`.modenav button.on`) | Gold fill + 2-layer shadow | Gold fill is untouched (only the extra glow layer drops) — state stays as obvious as it is today |
| Active pills (`.venue.on`, `.rpill.on`) | Gradient fill + glow | *Glow-only trim* (this plan's SAFE batch): fill/border untouched, so obviousness is essentially unaffected. *Full convergence to an underline-style indicator* (Phase 2A's original REPLACE proposal): a real UX-legibility question, not resolved here — see §5 |
| Filter chip (`.hit-chip.on`, results tab) | Gradient fill only (no shadow layer exists) | No safe partial trim available — either keep as-is or do the full REPLACE; deferred, see §5 |
| Badges already compliant (`.tdchip`, `.vschip`, `.mchip`, `.jtag`) | Low-alpha bg + border + text | No change proposed — already the target pattern |

Nothing in this plan removes a color cue or a text label from any status/selected element — every proposed change removes a *shadow* or *gradient-fade* layer while leaving the color-coding, border, and text untouched.

---

## 3. Priority 3 — Structural shadows

| Candidate | Current elevation cue | Can it safely move to surface-tone/border/spacing? |
|---|---|---|
| `.resbar` (result banner) | `border` + left gold accent + gradient wash + `--shadow-card` | Likely yes — border+wash+accent already do the work; the shadow is redundant. Ranked MODERATE here (isolated to result-confirmed races only, can't be verified against every page state without a race that has settled) |
| `.card` (universal, ~40+ instances/page) | 1px navy border + white sheen gradient + `--shadow-card` | The Phase 2A/2B answer stands: **yes in principle, but this is the single largest simultaneous visual change available in the whole system** — HIGH, explicitly not proposed for implementation this phase |
| `.ticket` (bet cards) | Same family as `.card` (`--shadow-card`) | Tied to the `.card` decision — HIGH, deferred with it, not decided independently |
| `.topbar` (sticky nav) | Drop shadow, no border | **Not a candidate** — this is genuine structural depth (separates the sticky nav from scrolling content beneath it); Phase 2A classified it KEEP and this plan agrees. Excluded deliberately, not an oversight |
| `#drawer` (overlay panel) | Entrance shadow | **Not a candidate** — DESIGN.md v1.0 §4's explicit true-overlay exception. Excluded deliberately |

Only `.resbar` is a plausible near-term candidate; it's ranked MODERATE (see §5) rather than SAFE because verifying it needs a race with a settled result on screen, which isn't guaranteed on every test pass the way an always-visible element is.

---

## 4. Explicit exclusions — confirmed untouched by every item in this plan

Checked each candidate below against this list; none violate it:

- AI-index bars, front-rate bars, lap-split highlights (`.bar i`, `.rb-bar i`, `.lapmini i.last`) — **not referenced anywhere in this plan**
- Advisor medals (`.adv-medal.*`) — **not referenced**. Note: `.adv:hover` (the *card's* hover shadow) is a different selector and is in scope; the medal circle itself is untouched by that change.
- Status chips already KEEP (`.tdchip`, `.vschip`, `.mchip`, `.jtag`) — **not referenced**
- Donut/gauge (`.gauge`, `.g-track`, `.g-fill`, `.g-val`) — **not referenced**
- Race-table information architecture, row height, numeric values — **not referenced**; the one table-adjacent item from Phase 2A (`.hrow.honmei`/`.intop3` gradient-fade→flat-tint) is listed below but placed in **deferred MODERATE**, not the SAFE batch, specifically because it touches the race table and this plan's stated goal is lowest-risk-first
- Prediction logic, APIs, data pipelines — no `.py`/`data/` file appears anywhere in this plan
- Marketing/homepage surfaces (`.hero-*`, `.hs-*`, `#landing`, `.lp-*`) — **not referenced**

---

## 5. Full candidate table

| # | File | Selector | Current CSS | Phase 2A class | Proposed CSS | Info preserved | Visual change | Risk | Rank |
|---|---|---|---|---|---|---|---|---|---|
| 1 | style.css:300 | `.date-wrap:hover` | `border-color: rgba(245,185,66,.55); box-shadow: 0 0 0 3px rgba(245,185,66,.08);` | REMOVE | `border-color: rgba(245,185,66,.55);` | Hover feedback (border brightens) unchanged | Hover-only; loses a faint outer glow | Very low — hover-only, no default-state change | **SAFE** |
| 2 | style.css:441 | `.venue:hover` | `...transform: translateY(-1px); box-shadow: var(--shadow-pop);` | REMOVE | `...transform: translateY(-1px);` | Lift + border/color change unchanged | Hover-only; loses drop-shadow pop | Very low | **SAFE** |
| 3 | style.css:942 | `.adv:hover` | `transform: translateY(-2px); border-color: var(--line2); box-shadow: var(--shadow-pop);` | REMOVE | `transform: translateY(-2px); border-color: var(--line2);` | Lift + border change unchanged; medal itself untouched | Hover-only | Very low | **SAFE** |
| 4 | style.css:384 | `.hit-card:hover` | `transform: translateY(-2px); border-color: var(--gold); box-shadow: 0 8px 22px -10px rgba(245,185,66,.5);` | REDUCE | `transform: translateY(-2px); border-color: var(--gold);` | Lift + gold border change unchanged | Hover-only | Very low | **SAFE** |
| 5 | style.css:313–317 | `.modenav button.on` | `box-shadow: inset 0 1px 0 rgba(255,255,255,.4), 0 2px 10px -4px rgba(245,185,66,.45);` | REDUCE | `box-shadow: inset 0 1px 0 rgba(255,255,255,.4);` | Gold fill (unchanged) is the actual state signal; embossed inset kept | Loses outer glow only; fill/color untouched | Low | **SAFE** |
| 6 | style.css:442–447 | `.venue.on` | `...box-shadow: inset 0 1px 0 rgba(255,255,255,.4), 0 4px 16px -6px rgba(245,185,66,.5);` | (part of REPLACE item — glow-layer trim only, not the full convergence) | `...box-shadow: inset 0 1px 0 rgba(255,255,255,.4);` | Gold fill + gold border unchanged | Loses outer glow only | Low | **SAFE** |
| 7 | style.css:469 | `.rpill.on` | `background: linear-gradient(...); border-color: rgba(245,185,66,.8); box-shadow: 0 4px 16px -6px rgba(245,185,66,.45);` | (part of REPLACE item — glow trim only) | `background: linear-gradient(...); border-color: rgba(245,185,66,.8);` | Gradient fill + border unchanged | Loses the (single) glow layer entirely | Low | **SAFE** |
| 8 | style.css:689–694 | `.vt.on::after` | `...border-radius: 2px 2px 0 0; box-shadow: 0 -3px 12px rgba(245,185,66,.35);` | REDUCE | drop the `box-shadow` line; everything else unchanged | Underline bar + gold text-color (`.vt.on`) unchanged | Loses glow beneath an already-visible underline | Very low | **SAFE** |
| 9 | style.css:859 | `.tk.tk-hon` | `border-color: rgba(245,185,66,.6); background: rgba(245,185,66,.08); box-shadow: 0 0 10px -4px rgba(245,185,66,.6);` | REDUCE | `border-color: rgba(245,185,66,.6); background: rgba(245,185,66,.08);` | Border + bg tint (already sufficient per Phase 2A) unchanged | Loses glow only | Very low | **SAFE** |
| 10 | style.css:976–979 | `.ticket::before` | `background: linear-gradient(180deg, var(--bcol, var(--gold)), color-mix(in srgb, var(--bcol, var(--gold)) 55%, transparent));` | REDUCE | `background: var(--bcol, var(--gold));` | Bet-type color key (the color itself) fully unchanged | 3px-wide edge stops fading to transparent, reads as a solid color bar | Low | **SAFE** |
| 11 | style.css:497–499 | `.jbadge.go/.caution/.skip` | gradient fill + 2-layer shadow (see §1) | REDUCE→REPLACE | low-alpha bg + colored text + colored border (matches `.tdchip`/`.jbadge.na` pattern) | Color-coding + text label fully unchanged; text goes bright-on-dark instead of dark-on-bright | Moderate — real background/contrast restructure on every race header; **today's data has no GO/慎重/見送り race to verify live**, only via a synthetic test element | **MODERATE** |
| 12 | style.css:1127–1132 | `.gs-badge` | gradient fill + glow (see §1) | REDUCE | low-alpha gold bg + gold text + gold border | Grade label fully unchanged | Moderate — same restructure as jbadge; *can* be verified live (today's card has a graded メイン race) | **MODERATE** |
| 13 | style.css:362 | `.hit-chip.on` | `color: #131008; background: linear-gradient(120deg, var(--gold2), var(--gold)); border-color: var(--gold);` | (REPLACE family, no shadow layer exists to trim) | Either leave as-is, or full convergence to a lighter indicator | Filter selection state | No safe partial step available | **MODERATE** |
| 14 | style.css:442–450, 469–471 | `.venue.on`/`.rpill.on` **fill removal** (full REPLACE, beyond item 6/7's glow trim) | gradient fill → text-color + underline/accent-bar, no fill | REPLACE | (not drafted — genuine UX-legibility question per Phase 2A) | Selection state, via a different mechanism | Large — changes how "selected" reads on two of the most-used nav components in the app | **HIGH** |
| 15 | style.css:651 | `.resbar` | gradient wash + left accent + `--shadow-card` | REDUCE | drop `box-shadow: var(--shadow-card)`, keep border+wash+accent | "This race has a result" signal unchanged | Small, but only verifiable on a settled race | **MODERATE** |
| 16 | style.css:738, 742–743, 1170ish | `.hrow.honmei`/`.intop3`, `.um-row.honmei` | gradient-fade background wash | REDUCE | flat low-alpha tint, same color | ◎/top-3 row highlighting unchanged | Touches the race table on every page — excluded from SAFE deliberately per §4 | **MODERATE** |
| 17 | style.css:476, 967 | `.card`/`.ticket` base (`--shadow-card` + sheen) | see §3 | REMOVE (sheen) / REDUCE (shadow) | drop sheen; lighten or drop shadow | Card boundary still readable via border + tonal step | Sitewide, simultaneous, largest single change in the whole plan | **HIGH** |
| 18 | style.css:613–626 | `.mls-mark` (level-spectrum pointer) | glow on pointer line+badge | REDUCE (soften, not remove — Phase 2A judged this partly functional) | soften blur radius, don't remove | Pointer position on the D–S spectrum unchanged | Small, but this one is *not* purely decorative (wayfinding value against a colored zone bar) — out of Priority 1's "purely decorative" scope by its own Phase 2A reasoning | **MODERATE** |
| 19 | style.css:1121ish | `.gs-bar` wash | 16%-opacity diagonal gold wash behind Grade Scope header | REDUCE | soften opacity | Header still reads as "featured" | Low-frequency (once per graded race) but a visible brand-forward treatment some may want to keep as-is | **MODERATE** |
| 20 | style.css:376, 400 | `.hit-card` background wash (not the stamp) | gradient tint across the whole win-card | REDUCE | flat card background; keep stamp gradient (explicit exception) | Win-stamp celebratory signal fully unchanged | Touches every card in the results-tab wins gallery, a prominent user-facing feature | **MODERATE** |

**Tally: 10 SAFE, 8 MODERATE, 2 HIGH.**

---

## 6. Phase 2C SAFE batch — the only items proposed for actual implementation

Items **1–10** from §5 above. All ten share the same shape: remove or flatten exactly one `box-shadow`/gradient-fade property, touch no color value that carries meaning, touch no text, touch no layout, and are verifiable on the current day's live data without needing a rare state (a GO-judged race, a settled result, a graded card) to exist.

### Exact proposed diff scope
One file: `site/css/style.css`. 10 edits, each a single-property removal or a gradient→solid substitution within an existing rule — no new selectors, no new rules, no rule reordering.

### Exact selectors
`.date-wrap:hover` · `.venue:hover` · `.adv:hover` · `.hit-card:hover` · `.modenav button.on` · `.venue.on` · `.rpill.on` · `.vt.on::after` · `.tk.tk-hon` · `.ticket::before`

### Desktop screenshot targets
1. Topbar with date-selector hovered (item 1)
2. Venue-tab row, one hovered + one active/selected (items 2, 6)
3. Race-strip pill, active/selected state (item 7)
4. View-tab row (出走表/全頭分析/コース/血統), active tab (item 8)
5. Mode-nav (予想/成績), active button (item 5)
6. Results tab: hit-card grid, one hovered (item 4)
7. Results tab: advisor card, one hovered (item 3)
8. Course tab: 想定隊列 lineup with the ◎ chip visible (item 9)
9. Any race view with a `.ticket` bet card visible (item 10)

### Mobile screenshot targets
Same list at 375×812 — items 1–10 all appear in mobile layouts (hover states are less relevant on touch but the *rest*-state CSS on the same rules is shared, so a mobile pass confirms nothing regressed for touch users); prioritize items 5–10 (visible without a mouse) over 1–4 (hover-dependent, desktop-only in practice).

### Visual acceptance criteria
- Every element in the SAFE batch remains at least as identifiable as "active/selected/hovered/interactive" as it is today — judged by: is the fill/border/color/text still present and unchanged? (It is, by construction — only shadow/gradient-fade layers are removed.)
- No layout shift: none of these properties affect box size, so no reflow is expected on any target.
- No console errors on a fresh page load, both modes, both breakpoints.
- No change to any displayed number, label, or race data.
- Side-by-side screenshots (this-session baseline vs. post-change) for each of the 9 target views show the *only* difference is the absence of a soft glow/fade — never a missing element, color, or label.

---

## 7. Deferred MODERATE/HIGH items (not in this phase's batch)

| # | Item | Why deferred |
|---|---|---|
| 11 | Judge badge flatten (`.jbadge.go/.caution/.skip`) | Real background/contrast restructure; **cannot be verified against live data today** (no GO/慎重/見送り race in the current bundle) |
| 12 | Grade Scope badge flatten (`.gs-badge`) | Same restructure family as judge badges; can be verified live, but still a real color/contrast change worth its own review pass |
| 13 | `.hit-chip.on` (results filter) | No safe partial step exists; needs the same "does the lighter indicator stay obvious" review as item 14 |
| 14 | `.venue.on`/`.rpill.on` full convergence to underline-style indicator | The single biggest UX-legibility question in this plan — deliberately separated from items 6/7's safe glow-trim |
| 15 | `.resbar` shadow removal | Only verifiable on a race with a settled result |
| 16 | `.hrow.honmei`/`.intop3`/`.um-row.honmei` gradient→flat tint | Touches the race table on every page — excluded from SAFE per this plan's own "lowest-risk first" instruction, regardless of how small the change looks in isolation |
| 17 | `.card`/`.ticket` base shadow + sheen | Sitewide, simultaneous — the largest single visual decision available in the whole system; explicitly flagged HIGH since Phase 2A/2B |
| 18 | `.mls-mark` glow soften | Not purely decorative by Phase 2A's own reasoning (has wayfinding value) — doesn't belong in a "decorative excess" batch at all |
| 19 | `.gs-bar` wash soften | Low-frequency, arguably a legitimate "featured" brand moment — a judgment call, not an obvious flatten |
| 20 | `.hit-card` background wash | Touches every card in a prominent, positive-feedback feature (the wins gallery) — worth its own dedicated look rather than bundling |

---

## 8. Decision Log

| Item | Outcome |
|---|---|
| Priority 1 (decorative excess) | 4 sub-categories identified with exact selectors; judge-badge and Grade-Scope-badge flattening drafted but ranked MODERATE (real color/contrast restructure, one unverifiable live today) |
| Priority 2 (selected/status states) | Confirmed every SAFE-batch change preserves state-obviousness via unchanged fill/border/text — never relies on the removed glow alone. Mode state already compliant per DESIGN.md v1.0 §10; only its redundant glow layer is a candidate |
| Priority 3 (structural shadows) | Only `.resbar` is a plausible near-term target (MODERATE); `.card`/`.ticket` confirmed HIGH and explicitly not attempted this phase, consistent with the Phase 2B report's own forward note; `.topbar` and `#drawer` confirmed as genuine exceptions, not oversights |
| Exclusions | Checked all 20 candidates against the explicit exclusion list; none violate it. `.adv:hover` is noted as distinct from the excluded `.adv-medal` |
| SAFE batch | 10 items, one file, each a single-property shadow/gradient removal — drafted with exact current/proposed CSS, not yet applied |
| Verification plan | 9 desktop + corresponding mobile screenshot targets and explicit acceptance criteria specified before any implementation |
| Production code | **Untouched.** No `site/` file was edited this phase. |

## `git status --porcelain -- site/`

```
$ git status --porcelain -- site/
```
(see accompanying chat message for live output — expected unchanged from the end of Phase 2B, since this phase made no code edits)
