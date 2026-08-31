# PyCaLiAI Design Decision Freeze — Phase 0.5

> Decision package only. `DESIGN.md` has **not** been edited. No production frontend files have been touched.
> Builds on: [`docs/design_audit_phase0.md`](design_audit_phase0.md) (19-item classification, accepted).
> Date: 2026-08-31

---

## 1. Governing principles for this pass

Two constraints were added on top of Phase 0's classification and are applied to every row below:

**Brand preservation.** PyCaLiAI does not abandon navy × gold to match Linear/OpenSea/Harness. Per [[project_site_lime_redesign]], navy×gold is a twice-fought-for, user-confirmed decision (a lime/turf-green alternative was built, shipped locally, and explicitly rejected). The three references supply *structure* — interaction discipline, density patterns, hierarchy, component philosophy — not a replacement palette. Where a reference's specific color or exact rule conflicts with the existing brand, the brand wins; where a reference's *structural idea* (one dominant accent used sparingly, thin dividers over per-item cards, a named type-scale for dense data) doesn't touch color, it's adopted on its merits.

**Reference weighting** — used below to judge how strongly a given rule should defer to a given source:
- **Linear 50%** — system architecture, precision, restraint discipline (spacing rhythm, single-accent-per-action, hairline structure)
- **OpenSea 35%** — dense data interfaces, numeric list presentation, tight information density
- **Harness 15%** — sparse brand/marketing surfaces, display typography confidence, one-accent-per-viewport restraint

**Surface taxonomy** (used for every rule where marketing vs. app treatment can differ):

| | Surface A — Marketing | Surface B — Application/data |
|---|---|---|
| Live components | Hero (`#hero`), landing sections 01–04 (`#landing .lp-*`, incl. HONEST RECORD stats) | Race table (`.hrow`/`.htable`), drawer, results tab (`#resultsMain`), analytics tabs (`bunseki`/`course`/`pedigree`), Grade Scope, `explain.html` cards, topbar/venue-tabs/race-strip chrome |
| Should read as | Confident, warm, a little more willing to spend a gradient/glow on 1–2 focal elements | Flatter, denser, quieter — repeated elements (badges, pills, table rows) must not each carry their own decoration |
| Governing reference lean | Harness (sparse, display-led) | OpenSea (dense, numeric, restrained) |

---

## 2. Master decision table — all 19 Phase 0 items

*Classification is carried forward from Phase 0 unless marked "flag only" — see §5 for the one place this pass questions a Phase 0 DEFER without unilaterally changing it.*

| ID | Current implementation | Current DESIGN.md rule | Reference influence | Class. | Recommended final rule | Reason | Visual impact | Implementation impact |
|---|---|---|---|---|---|---|---|---|
| 1 | `.modenav` buttons: 🏇/📊/🔬 + label; `explain.html` `<h1>` has 🏇 prefix | §9 "Remove decorative emoji from primary navigation" | Linear (text-only nav) + OpenSea (line-art icon rail, never emoji) + Harness (text+pill nav) — all three independently avoid emoji in nav | **ADOPT** | Keep §9 verbatim, no change | Exact match to DESIGN.md's own "wrong" example; unanimous reference agreement; zero brand cost | Minor — 3 buttons + 1 header lose icon prefix | Trivial, ~4 lines / 2 files (scoped in Phase 0 §5a) |
| 2 | `.modenav button.on` = gold gradient fill + glow; same idiom on `.venue.on`/`.rpill.on`; `.vt.on` (view tabs) already uses underline-only, no fill | §9 "never a large colored button" | Linear/OpenSea/Harness active-nav = color/text change or thin accent bar, never a filled button | **MODIFY** (folded into Proposal 1, §3 below) | Exclusive 2–3-way mode switches (予想/成績) may keep the filled segmented-control treatment. List-style peer selection (venue tabs, race pills) should converge toward the lighter indicator `.vt.on` already uses elsewhere in the same codebase — text-color + underline/accent-bar, not a filled glowing pill | Segmented-control convention (a toggle) is a different UI grammar than a peer list (a nav); references target the latter, not the former | None on mode-nav. Venue-tabs/race-pills: moderate future flattening | **Not zero** — venue-tabs/race-pills repeat 5–36× per screen; this is real future CSS work, not just documentation (see §5) |
| 3 | `explain.html`: own 14px/8px radii, Oswald font not shared with `style.css` tokens; mono webfont not loaded there | — (internal drift, not a DESIGN.md rule per se) | N/A — self-consistency issue | **ADOPT** | No new text; enforce existing shared tokens | Free fix, zero brand tradeoff, one page only | Card radius 14→4px, heading font Oswald→IBM Plex Mono/Zen Kaku Gothic New | Low-medium, one page (scoped in Phase 0 §5b) |
| 4 | Gold = CTA / active-nav / focus fill. Teal = semantic-positive only | §2 "Teal: primary interaction... Gold: ...use sparingly" | None of the 3 use a 2-accent interaction system with PyCaLiAI's exact split — but Harness's "two accents (Mint fill + Blue link), two different jobs" is the closest structural analogue | **MODIFY** | Gold = primary interaction & brand fill. Teal = semantic-positive & secondary informational accent. Applies uniformly, no A/B split | Brand-preservation clause overrides; borrows Harness's *structure* (two accents, two jobs), not its colors; zero implementation cost | None (doc-only) | None |
| 5 | Solid navy hex borders (`--line #1e2b4a` / `--line2 #32456e`) everywhere | §8 `rgba(255,255,255,.08/.12/.18)` universally | **No consensus among the 3**: Linear = dark hairline (not white-alpha — contradicts DESIGN.md's own citation); OpenSea = white-alpha inset ring; Harness = light-gray solid. PyCaLiAI's navy hairline is a legitimate 4th variant | **MODIFY** | Keep PyCaLiAI's navy hairline scale everywhere. OpenSea's real contribution — thin single-pixel dividers between dense rows, not per-row cards — is adopted structurally; it's already implemented correctly in `.hrow` | No reference to defer to; navy borders read cohesively against the navy canvas (same hue family) where a cold white-alpha border would not | None | None |
| 6 | 40+ `box-shadow` declarations; glow on badges/active-states/medals/every card | §3/§23 "avoid... no glow around every element... excessive shadows" | **Unanimous**: Linear/OpenSea/Harness all use near-zero shadow (hairline/inset-only elevation) — strongest cross-reference agreement of any rule in this audit | **MODIFY** (A/B split) | **A**: restrained glow on the primary CTA + ≤1 hero/HONEST-RECORD accent per viewport. **B**: shadow limited to focus ring, one consistent selected-state indicator (not glow), and the drawer's overlay-entrance shadow. Per-badge ambient glow, hover-lift-pop on every card, and glow on repeated list items move toward flat | Grounds DESIGN.md's shadow instinct (which is well-supported) in *where* it's costly — repeated data rows — vs. where 1–2 uses serve real brand signal | A: none. B: moderate future change — judge badges, medal circles, active pills read flatter/quieter | **Not zero** — ~15–20 selectors, largest single future CSS scope in this package (see §5) |
| 7 | 2-stop gold gradients on badges/active-states/win-stamps; CTA itself is solid, not gradient | §23 "no decorative gradients"; §10 "no gradient CTA" | **Unanimous**: zero gradients in any of the 3 references' documented components (Linear reserves gradient for the hero atmosphere only) | **MODIFY** (A/B split) | **A**: single-hue gold gradients permitted on hero/landing accents (matches current). **B**: gradients permitted only on win-result stamps (rare, celebratory, Harness-style "one moment, not decoration") and the mode-switch control (item 2's exception). Judge badges, advisor medals, ticket borders, Grade badge, chart area-fill move toward flat fills | Same unanimous grounding as shadows; win-stamps are infrequent "moments" (spirit of restraint honored) unlike per-row badges (repetition violates it) | A: none. B: moderate future change, same component set as item 6 | Bundled with item 6's future work — same selectors, same phase |
| 8 | `font-weight:900` on hero/marks/stamps; `700` widespread on buttons/badges/labels | §4 "avoid 700+/900" | General-UI ceiling **is** reference-grounded (Harness caps 600, Linear 590, OpenSea 500 → 600 is a realistic consensus ceiling). Hero-display weight is **not** Harness-grounded — Harness's own hero goes *light* (300) at huge size, the opposite of PyCaLiAI's smaller, heavier hero | **MODIFY** (A/B + named exceptions) | General/repeated UI text: ceiling 600. Named, unlimited exceptions: race marks (◎〇▲△), hero/landing display headlines, win-result stamps — PyCaLiAI-specific "read-this-first" moments with no reference analogue | Keeps the reference-grounded part honest (buttons/badges *can* tighten) while naming the brand-specific exceptions explicitly instead of pretending one blanket rule covers everything | Buttons/badges at 700 read very slightly lighter if later swept to 600. Marks/hero/stamps: no change | Small future sweep, ~dozen declarations, low risk — can ride with item 6/7's pass or go alone |
| 9 | 17px global body; frequent half-pixel sizes; fluid `clamp()` headings | §5 "body:14px" **and** (inconsistently) §12 "12–14px" for tables | Table-scale is OpenSea-grounded (OpenSea: "12–16px for almost all UI, reserve 20/32 for headers" — near-exact match, and OpenSea is the named dense-data reference). Prose-scale is Harness-adjacent (Harness body 16px/1.5, one step below PyCaLiAI's 17px, same "comfortable reading" register) | **MODIFY** | Two named scales instead of one number: **prose/marketing body** 14–17px (keep current 17px default). **table/data-dense** 12–15px (race table, analytics, badges/chips) | Resolves DESIGN.md's own internal §5-vs-§12 inconsistency by grounding each half in the reference it actually matches, rather than picking one arbitrary number | None — describes two scales already in production use | None |
| 10 | Focus ring: gold, 3px, box-shadow | §22 "2px brand-teal focus ring" | OpenSea names Ice-Signal-blue explicitly as its focus-ring color (a true ring, like PyCaLiAI's). Harness explicitly rejects rings — "sharp border color change, no glow ring." Linear brightens the border, no true ring either | **MODIFY** | "2px gold focus ring" — tighten width 3→2px; keep the ring mechanism (not Harness's border-brighten) since dense dark tables need a higher-contrast always-visible indicator; color follows item 4 | OpenSea's ring model fits PyCaLiAI's actual UI (a dense keyboard-navigable data table) better than Harness's model, which was designed for a sparser marketing site | Negligible — ring shrinks 1px, color already gold in practice | Trivial, one CSS custom property, whenever next touched |
| 11 | `backdrop-filter: blur()` on sticky topbar, sticky table header, drawer scrim (3 uses, all functional) | §23 "no glassmorphism cards" | None of the 3 references use `backdrop-filter` anywhere in their documented components — absence, not contradiction | **MODIFY** | Narrow to "no decorative frosted-glass card fronts." Explicitly permit blur on sticky headers/nav over scrolling content and modal/drawer scrim overlays | All 3 live uses are legibility-functional, not decorative "frosted card" chrome — the rule was clearly aimed at the latter | None — existing behavior is already correct under the narrowed reading | None |
| 12 | Infinite `pulse`/`skshine` animation, loading/skeleton states only | §21 "no continuous glowing... no decorative looping animation" | Out of scope for all 3 references (none specify loading-state treatment) — universal web pattern, addressed to close a DESIGN.md wording gap only | **MODIFY** | Add carve-out: "infinite/looping animation permitted for temporary loading/skeleton states; ban applies to static decorative chrome" | Literal current wording technically bans a necessary, universal, purely functional pattern; §21's own examples (parallax, floating blobs) show the rule's real target is ambient decoration | None | None |
| 13 | `--r-pill: 4px` — token name promises a capsule, delivers a barely-rounded rectangle | — (naming hygiene, not a DESIGN.md rule) | N/A | **Hygiene fix** (not part of the 9 MODIFY proposals) | No DESIGN.md text needed | Pure internal naming clarity | None if a pure rename | Trivial, deferred to next radius-touching pass |
| 14 | `.rs-sum-grid` (4-card ROI grid) + `.gauges` (4-donut row) | §23 "no oversized dashboard KPI cards"; §18 "one restrained strip" | OpenSea *does* use dense multi-card grids, but each card is small/compact (8px radius, 12px pad, one number+delta) — noticeably smaller/quieter than PyCaLiAI's current 4-card ROI grid (bigger padding, 23–28px numbers). OpenSea doesn't clearly bless the current treatment either | **DEFER** | — | Needs a rendered side-by-side, not resolvable from CSS reading alone | — | — |
| 15 | `.hit-grid` shows wins only; `.hit-card.big` escalates styling for ¥10,000+ wins | §18 "positive and negative results use the same component system" | N/A — information-architecture question | **DEFER** | — | Adding a losses gallery, or reframing the wins gallery, is a product decision | — | — |
| 16 | No literal 単勝/decimal-odds column in the race table | §12 example table | N/A — possible compliance constraint | **DEFER** | — | Needs cross-check against prior JRA-VAN posting-guideline work ([[project_jravan_guideline_compliance]]) before treating as a style gap | — | — |
| 17 | `--bg2/--card/--card2` don't match DESIGN.md's `bg-1/bg-2/bg-3` hex exactly | §2/§3 hex values | Canvas (`bg-0`) matches exactly — clearly the sampled anchor; the other 3 levels read as estimated, not sampled | **MODIFY** (folded into Proposal 1, §3 below) | Update DESIGN.md's bg-1/2/3 hex to the live values | Low-risk documentation correction; canvas match proves the intent was "describe reality," these 3 values just didn't get sampled | None | None |
| 18 | Single `--maxw: 1240px` for both marketing and app views | §20 "1440 app / 1200 marketing" | **Flag**: none of the 3 references actually differentiate marketing vs. app max-width either — Linear (1200, one value), OpenSea (1440, one value), Harness (1200, one value) each use a single site-wide width. DESIGN.md's dual-width idea isn't grounded in any of its own three sources | **DEFER** *(unchanged — see §5 for why this wasn't upgraded to MODIFY here)* | — | — | — | — |
| 19 | Sticky header, tabular-nums, number formatting, no-guaranteed-profit copy, marks-as-symbols, no purple/blobs/fake-terminal/gradient-text, drawer-as-secondary-panel, `prefers-reduced-motion`, landing §15–17 sections | Various (§12,14,16,17,19,21,22) | Already-compliant across all 3 references' spirit | **KEEP CURRENT** | No change | Working correctly today | None | None |

---

## 3. The 9 MODIFY DESIGN.md proposals — existing text / proposed text / rationale

*These correspond to audit items 4+17 (merged, one color proposal), 5, 6, 7, 8, 9, 10, 11, 12 above.*

### Proposal 1 — §2 Brand Colors (covers items 4, 17)

**1. Existing DESIGN.md text:**
> Teal:
> - primary interaction
> - selected states
> - focus
> - AI/model informational emphasis
>
> Gold:
> - PyCaLiAI brand highlight
> - ◎ / primary model highlight
> - high-value emphasis
> - use sparingly
>
> `--bg-1: #0B111C; --bg-2: #111927; --bg-3: #182233;`

**2. Proposed replacement text:**
> Gold:
> - primary interaction (buttons, active/selected states, focus)
> - PyCaLiAI brand fill — the interface's default "this is active/actionable" signal
> - ◎ / primary model highlight
>
> Teal:
> - semantic-positive (ROI ≥ control line, hit-rate, "GO" judge status)
> - secondary informational accent (market-value-under indicators, AI/model informational emphasis where it is not the primary action)
>
> Both accents are used deliberately and are not interchangeable — gold answers "what can I act on / what is PyCaLiAI's own," teal answers "is this number good."
>
> `--bg-1: #0A101E; --bg-2: #0D1424; --bg-3: #121C31;`

**3. Why this is better for PyCaLiAI specifically:**
Navy×gold is not a placeholder — it's the outcome of a real design fight (lime tried, shipped locally, rejected; see [[project_site_lime_redesign]]). A rule that assigns "primary interaction" to teal quietly asks every future contributor (including a future Claude session reading this file cold) to treat the CTA, active nav, and focus ring as *wrong* and slowly re-plumb them toward teal — undoing a decision the user already made twice. Swapping the two role descriptions costs nothing (it's a documentation edit) and immediately makes DESIGN.md describe the product that actually exists. The bg-1/2/3 hex correction is the same kind of fix at smaller scale: bg-0 already matches exactly, so aligning the other three levels removes the only remaining unforced numeric mismatch in the core palette.

---

### Proposal 2 — §3 / §8 Borders (covers item 5)

**1. Existing DESIGN.md text:**
> Cards: `border: 1px solid rgba(255,255,255,0.08);`
>
> Default: `1px solid rgba(255,255,255,0.08)`
> Selected: `1px solid rgba(45,212,168,0.40)`
> Strong separator: `1px solid rgba(255,255,255,0.14)`
>
> Do NOT copy Linear's 0.5px border globally. 1px is preferred for consistent rendering across Windows and non-Retina displays.

**2. Proposed replacement text:**
> Borders use PyCaLiAI's navy hairline scale, not white-alpha — this keeps every edge in the same warm/cool family as the navy canvas rather than introducing a colder, unrelated tone:
>
> Default: `1px solid var(--line)` *(navy, ≈ #1e2b4a)*
> Strong separator / elevated: `1px solid var(--line2)` *(navy, ≈ #32456e)*
> Selected: gold-tinted border or left-edge accent bar (follows Proposal 1's gold-as-primary resolution), not teal.
>
> 1px is preferred for consistent rendering across Windows and non-Retina displays (unchanged from prior guidance).
>
> Density note (from OpenSea, 35% weight): on repeated-row surfaces (race table, dense lists), prefer a single thin divider between rows over wrapping each row in its own bordered card — already correctly implemented in `.hrow`.

**3. Why this is better for PyCaLiAI specifically:**
The original rule cites "Linear" as if it supports white-alpha borders, but Linear's own reference sheet says the opposite (`#23252a` dark hairline on `#0f1011`, not `rgba(255,255,255,…)`) — the rule wasn't actually grounded in the source it named. None of the three references agree with each other on border color model, so there's no "correct" answer to defer to — which means the existing, already-shipped navy hairline (which reads cleanly against the navy canvas) is a perfectly legitimate choice to keep, and documenting it costs nothing. What *is* worth taking from OpenSea is the structural lesson — thin dividers, not per-row cards — which the site already does.

---

### Proposal 3 — §3 / §23 Shadows & glow (covers item 6)

**1. Existing DESIGN.md text:**
> Avoid drop shadows.
> No glassmorphism.
> No large glow effects.
>
> *(§23, forbidden list includes:)* glow around every element / excessive shadows

**2. Proposed replacement text:**
> **Marketing surfaces**: a restrained glow is permitted on the primary CTA's hover state and on up to one hero/HONEST-RECORD accent element per viewport — not more. This mirrors Harness's own restraint principle ("Phosphor Mint... reserved for featured cards, one per viewport maximum") applied to PyCaLiAI's gold.
>
> **Application/data surfaces**: shadow is limited to (a) the accessibility focus ring, (b) one consistent selected/active-state indicator per component family — a border-color change or thin accent bar, not an ambient colored glow, and (c) the drawer/modal's entrance shadow (a functional depth cue for a true overlay, consistent with how all three references still shadow genuine overlays). Ambient per-badge glow, hover-lift-and-pop on every card, and glow repeated across list items (venue tabs, race pills, judge badges) are not used on these surfaces.
>
> This rule is the one place in this document where "matches DESIGN.md" and "matches current production" genuinely diverge on application surfaces — see the migration note in the audit's decision log before treating this as already satisfied.

**3. Why this is better for PyCaLiAI specifically:**
This is the one rule where Linear, OpenSea, and Harness are in complete agreement — stronger consensus than on any other topic in this audit — so it's worth taking seriously rather than working around. But a blanket "avoid shadows" is too blunt for a brand built on a warm glow accent; the actual cost of glow is concentrated in *repetition* (the same badge/pill glow appearing 8, 20, 40 times on one screen reads as noisy), not in its existence. Scoping the rule to "restrained, ≤1-per-viewport on marketing; functional-only, no ambient glow on data" keeps the parts of the reference consensus that matter (a data table full of glowing badges is genuinely harder to scan) while explicitly preserving room for the brand moment (CTA hover, hero) that makes the site feel like PyCaLiAI rather than a Linear clone.

---

### Proposal 4 — §23 Gradients (covers item 7)

**1. Existing DESIGN.md text:**
> *(§23, forbidden list includes:)* decorative gradients
>
> *(§10:)* No gradient CTA.

**2. Proposed replacement text:**
> No gradient CTA remains unchanged — the primary action button stays a solid fill (already true today).
>
> **Marketing surfaces**: single-hue gold, 2-stop gradients are permitted on hero/landing accent elements (already true today).
>
> **Application/data surfaces**: gradients are permitted only for (a) win-result celebratory stamps (`hitchip`, `won-badge`, `hit-card`) — infrequent, deliberate "moments" rather than a per-row treatment — and (b) the mode-switch segmented control (see item 2). All other application-surface gradients (judge badges, advisor medals, ticket accent borders, Grade Scope badge, chart area-fill) are flat single-color fills.
>
> Any gradient anywhere remains 2-stop maximum and single-hue — no multi-color decoration.

**3. Why this is better for PyCaLiAI specifically:**
Same unanimous-reference grounding as shadows (proposal 3), and the two rules move together in practice since most of the live gradient declarations already carry a glow shadow in the same rule block. The win-stamp exception is deliberately narrow: it fires once per correct pick, not once per table row, so it stays inside the spirit of Harness's "sparingly, one per viewport" restraint even while being a gradient. Judge badges and advisor medals, by contrast, repeat on every race/every advisor comment — exactly the repetition that turns a nice accent into visual noise on a page meant to be scanned fast.

---

### Proposal 5 — §4 Typography weight (covers item 8)

**1. Existing DESIGN.md text:**
> Use:
> 400 normal
> 500 emphasis
> 600 important headings only
>
> Avoid:
> 700+
> 900

**2. Proposed replacement text:**
> Use:
> 400 normal
> 500 emphasis
> 600 general UI emphasis (headings, buttons, badges, labels) — this is the ceiling for repeated interface text
>
> Named exceptions (weight 700–900 permitted, unlimited by the 600 ceiling above):
> - race marks (◎〇▲△) — must read instantly at a glance across a dense table
> - hero / landing display headlines — a single, low-repetition brand moment
> - win-result stamps (`won-badge`, `hit-card` "的中"/"万馬券") — same rationale as the gradient exception in §23
>
> Everywhere else, treat 700+ as a signal something should probably be a named exception or should drop to 600 — not a default.

**3. Why this is better for PyCaLiAI specifically:**
The 600 ceiling for general UI text is genuinely reference-grounded — Harness caps headings at 600, Linear tops out at 590, OpenSea never exceeds 500, so 600 is a real consensus, and PyCaLiAI's current widespread use of 700 on ordinary buttons/badges/labels can reasonably tighten toward it without losing anything. But treating the hero headline the same way doesn't hold up under scrutiny: Harness's own hero goes to weight *300* at 88px — the opposite move, using scale instead of weight for impact — which only works because Harness's hero type is nearly triple PyCaLiAI's size. At PyCaLiAI's smaller, denser hero scale (clamp 30–60px), weight is genuinely doing the work that Harness's extra 30–50px of size does for it. Naming this exception explicitly is more honest than either (a) pretending Harness supports heavy hero weight, or (b) shrinking PyCaLiAI's hero weight to match a rule that doesn't actually transfer.

---

### Proposal 6 — §5 / §12 Type scale (covers item 9)

**1. Existing DESIGN.md text:**
> *(§5:)* body: 14px / 1.5
>
> *(§12:)* Use: 12–14px typography [for the race table]

**2. Proposed replacement text:**
> Two named type scales, not one:
>
> **Prose/marketing body** (landing copy, drawer notes, narrative text, HONEST RECORD copy): 14–17px / 1.5–1.9. Default 17px for primary reading copy — this product is read at a glance across dense tables, and a slightly larger base size than a typical SaaS product (Harness itself uses 16px) is deliberate, not an oversight.
>
> **Table/data-dense** (race table cells, analytics tables, badges, chips, sparklines): 12–15px, tabular-nums where the value is compared column-to-column. This matches OpenSea's own scale almost exactly ("12–16px for almost all UI, reserve 20/32 for headers") — OpenSea is the named dense-data-interface reference for a reason, and its type discipline is the one to borrow here.

**3. Why this is better for PyCaLiAI specifically:**
DESIGN.md currently contradicts itself — §5 says body is 14px, §12 says the race table (arguably the most important body content on the site) should be 12–14px, a *different* and *smaller* number, with no acknowledgment that these are two different contexts. Neither number matches what's actually shipped (17px prose, 12–17.5px table cells depending on column). Splitting the rule into two named scales resolves the self-contradiction and grounds each half in the reference that actually matches its job — OpenSea for dense tabular data, a slightly-larger-than-Harness comfortable-reading size for prose, which suits a glance-speed product better than Harness's marketing-first 16px would.

---

### Proposal 7 — §22 Focus ring (covers item 10)

**1. Existing DESIGN.md text:**
> Keyboard focus:
> 2px brand-teal focus ring

**2. Proposed replacement text:**
> Keyboard focus:
> 2px brand-gold focus ring (`box-shadow: 0 0 0 2px rgba(gold, .3–.4)`) — a visible ring, not a border-color-only change, because the primary navigable surface (the race table) is dense and dark enough that a subtle border shift risks being missed. Color follows the primary-interaction resolution in Proposal 1.

**3. Why this is better for PyCaLiAI specifically:**
Of the three references, Harness explicitly rejects a glow ring in favor of "a sharp border color change, no ring" — but Harness's UI is a sparse marketing/devtools site, not a dense keyboard-navigable data table. OpenSea, the dense-data reference, *does* use a true ring (Ice-Signal-blue, named explicitly for "focus rings" in its color-role table) — its context is closer to PyCaLiAI's actual use case (tabbing through many rows/controls) and its mechanism transfers better than Harness's. The color itself just follows from Proposal 1 rather than needing its own justification. The live implementation already uses a ring at 3px; tightening to 2px is the one small, genuinely-live-code-affecting change in this whole proposal set, and it's trivial (one custom property).

---

### Proposal 8 — §23 Glassmorphism (covers item 11)

**1. Existing DESIGN.md text:**
> No glassmorphism.
>
> *(§23, forbidden list includes:)* glassmorphism cards

**2. Proposed replacement text:**
> No decorative glassmorphism — do not use `backdrop-filter` blur to create a "frosted glass" card front as a stylistic flourish.
>
> Functional blur is permitted where it serves legibility rather than decoration: sticky headers/navigation over scrolling content, and modal/drawer scrim overlays. Neither of these is a "card" in the decorative sense the original rule targets.

**3. Why this is better for PyCaLiAI specifically:**
None of the three references use `backdrop-filter` anywhere in their documented components, so there's no reference to either confirm or contradict here — this is a case where PyCaLiAI's own implementation had to make a call the references simply don't address. All three live uses (sticky topbar, sticky table header, drawer scrim) solve a real problem — keeping controls readable while content scrolls underneath, and focusing attention on a modal — that a literal reading of "no glassmorphism" would remove without replacing. Narrowing the rule to target decorative frosted-card fronts specifically (which PyCaLiAI doesn't do) closes the gap between the rule's evident intent and its literal wording.

---

### Proposal 9 — §21 Interaction / animation (covers item 12)

**1. Existing DESIGN.md text:**
> Do not use:
> parallax
> continuous glowing animations
> floating blobs
> decorative looping animation

**2. Proposed replacement text:**
> Do not use:
> parallax
> continuous glowing animations *(on static/decorative chrome)*
> floating blobs
> decorative looping animation
>
> Exception: infinite/looping animation is permitted for temporary loading and skeleton states (shimmer, pulse) — these disappear once real data arrives and exist to signal "working," not to decorate.

**3. Why this is better for PyCaLiAI specifically:**
The rule's own examples (parallax, floating blobs) make its actual target obvious: ambient decorative motion on chrome that never changes. A loading shimmer is the opposite — temporary, tied directly to real application state, and one of the most universally-expected patterns on the web (its *absence* would read as broken, not restrained). This is purely a wording gap, not a real disagreement between the doc and the product; a one-line carve-out closes it without weakening the rule against what it's actually trying to prevent.

---

## 4. What Phase 0.5 changes relative to Phase 0

Most of the 9 proposals above are pure documentation corrections — no live CSS is implied to change (color roles, borders, type-scale naming, glassmorphism/animation wording, and the focus-ring color are all "make the doc match reality" moves, net implementation cost: zero). Two things are different from Phase 0's original framing and are called out explicitly so they aren't mistaken for already-done:

1. **Items 6 and 7 (shadow/glow, gradients) are now A/B-scoped rather than blanket-accepted.** Phase 0 classified these as MODIFY-DESIGN.md with an open allowlist ("focus rings, active-state glow, hover lift are fine"). This pass narrows that allowlist specifically for **application/data surfaces**, which means accepting Proposals 3 and 4 as written commits to *future* CSS work — flattening judge badges, advisor medals, ticket accent borders, and repeated active-pill glow — not just editing DESIGN.md. This is the single largest implementation-impact item in the whole package and should be scoped as its own follow-up phase, not assumed to ride along for free.
2. **Item 2 (modenav "large colored button")** similarly now implies venue-tabs and race-pills should eventually converge toward the lighter indicator style `.vt.on` (view tabs) already uses correctly in production — a smaller, lower-risk version of the same kind of future work as #1.
3. **Item 18 (dual max-width) was flagged, not reclassified.** Checking all three references' own layout numbers shows none of them actually use a marketing/app width split either (each uses one site-wide max-width) — which weakens DESIGN.md's original dual-1440/1200 rule more than it strengthens it. This is a legitimate DEFER→MODIFY candidate, but reclassifying it here would break from the "9 items currently classified MODIFY" scope this package was asked to freeze — surfaced for a future pass instead of decided unilaterally now.

Everything else in §2/§3 is low-risk documentation-only and can be applied to `DESIGN.md` as a single edit whenever the user signs off, with no code-side follow-up required.

---

## 5. Status

This document is a **frozen decision package** — a reviewable proposal, not an applied change. Nothing here has been written into `DESIGN.md`. Two separate future approvals remain, and neither is implied by accepting this package:

1. Applying the accepted proposals' text into `DESIGN.md` itself (a documentation-only edit).
2. Any resulting Phase 1/2 work on `site/` — most of §3's 9 proposals require none; Proposals 3 and 4 (and, more narrowly, item 2) are the exception and would need their own scoped plan (files/selectors/risk/verification, in the format already used in the Phase 0 audit's §5) before any code is touched.

No production frontend files were read-then-edited to produce this document — only `docs/design_audit_phase0.md` was read, and this file was written net-new. `git status --porcelain -- site/` output follows in the accompanying chat message as a live command result.
