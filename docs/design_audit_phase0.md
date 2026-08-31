# PyCaLiAI Design System Audit — Phase 0

> Audit only. No production frontend files were edited to produce this report.
> Scope: `site/index.html`, `site/explain.html`, `site/css/style.css`, `site/js/app.js`, `site/js/baba.js`
> Measured against: root `DESIGN.md`, `docs/design-references/{Linear,openSea,Harness}_DESIGN.md`
> Date: 2026-08-31

---

## 0. Method

Read all five frontend files in full (style.css 1383 lines / app.js 1806 lines / baba.js 146 lines / index.html 390 lines / explain.html 175 lines — no truncation, no sampling). Cross-referenced every color, radius, shadow, weight, spacing and component pattern found against `DESIGN.md` §1–24, then against the three cited references to see whether a DESIGN.md rule is actually grounded in "Linear/OpenSea/Harness" or is a stricter, self-derived addition. `DESIGN.md` and the three reference copies were **not modified**; only added `docs/design-references/*.md` (didn't exist before this session) so the audit has stable, versioned inputs to cite.

**Headline finding:** DESIGN.md is not a description of a *new* direction — three of its numeric tokens (`--bg-0`/canvas, `--brand-gold`, `--brand-teal`) match the live production CSS exactly. But a much larger set of DESIGN.md rules (borders, shadows, gradients, primary-interaction color, font family, font-weight ceiling) do **not** match production, and production's choices look deliberate rather than accidental (glow/gradient is a 40+ occurrence signature pattern; gold-as-primary is corroborated by [[project_site_lime_redesign]] memory as a hard-fought, twice-reverted brand decision). So this audit treats DESIGN.md as a **partially-retrofitted, partially-aspirational** document, and flags — rather than silently resolves — every place where "make DESIGN.md match reality" and "make reality match DESIGN.md" are both live options.

---

## 1. Frontend Architecture Inventory

| Surface | File | Role |
|---|---|---|
| Main app | `site/index.html` (390 ln) | Two "sheets" in one DOM, toggled via `body.on-landing`: a marketing/landing cover (hero + 4 numbered sections) and the data app (venue tabs → race strip → race view). Contains **~150 lines of inline `<script>`** that render the landing sections (hero spot, today's-focus table, race index) — a third home for presentation logic, separate from `app.js`. Also an inline `<style>` block (CLS-prevention + the entire 馬場バイアス/baba-bias component's CSS) that never made it into `style.css`. |
| Analysis cards | `site/explain.html` (175 ln) | Fully separate mini-app: own inline `<style>` block (own card radius, own colors-via-var reuse, own font-family choice), own inline `<script>` (own `esc()`, own row/card renderers) that does not share code with `app.js`. Loads `style.css` for CSS custom properties only. |
| Global stylesheet | `site/css/style.css` (1383 ln) | Single stylesheet for both pages. `:root` design tokens + every component class. Explicit "旧名エイリアス" (legacy-name alias) block maps old variable names onto new ones — carried over from the 2026-08-11 lime→navy rollback (see [[project_site_lime_redesign]]). |
| Render logic | `site/js/app.js` (1806 ln) | Vanilla JS, template-string HTML generation, no framework. Owns: nav, race table, drawer, results tab, 4 ECharts-based analysis tabs (scatter, career line, course bars, pedigree bars). Chart color values are **hardcoded hex literals** inside JS option objects (`"#f5b942"`, `"#a9b6d3"`, `"#32456e"`...) — the CSS custom-property token system is not read back into JS anywhere (no `getComputedStyle`), so a future token change to `style.css` will not propagate to any chart. |
| Track-bias widget | `site/js/baba.js` (146 ln) | Self-contained IIFE, own gauge-position math, reuses CSS classes from `style.css` (`.cg-*`, `.bstat`, `.kchip`) but no unique tokens. |
| Design tokens | `style.css :root` (ln 5–50) | Core bg/card/line scale, brand (`--lime`/`--lime-light`/`--lime-dark`), text scale, semantic (positive/attention/uncertain/danger/neutral), then a **second block of legacy aliases** (`--gold`→`--lime`, `--teal`→`--positive`, `--blue` new literal, etc.) that most component rules actually reference instead of the "new" names. |
| Responsive | 7 distinct breakpoints: 1000 / 980 / 880 / 760 / 680 / 620 / 560px | Organically grown, not a clean 2–3 tier system. Column-count collapse strategy for the race table (not horizontal scroll) — mark/umaban/name are the last three columns ever hidden, at every breakpoint. |
| Fonts loaded | `index.html`: Noto Sans JP, Zen Kaku Gothic New, IBM Plex Mono (400–900). `explain.html`: Noto Sans JP + **Oswald** only — does not load Zen Kaku Gothic New or IBM Plex Mono despite `style.css`'s `--sans`/`--disp` referencing them first. On `explain.html`, `var(--disp)` (used for all `.num`/mono data) silently falls back past two unloaded fonts to plain Noto Sans JP — numeric columns on that page are **not actually monospaced** in practice, even though the CSS says they should be. |
| Chart lib | ECharts 5.5.1 via jsDelivr CDN, deferred, lazy-init on first tab open | Only on `bunseki`/`course`/`pedigree` tabs and the drawer's career chart — not force-loaded on the primary race view. |

---

## 2. Inconsistencies vs. DESIGN.md, by category

### 2a. Colors — exact hex comparison

| Token | DESIGN.md | Live CSS (`style.css :root`) | Verdict |
|---|---|---|---|
| Canvas / bg-0 | `#070B14` | `--bg: #070b14` | **exact match** |
| bg-1 | `#0B111C` | `--bg2: #0a101e` | close, not exact |
| bg-2 (card) | `#111927` | `--card: #0d1424` | different — more saturated navy |
| bg-3 (elevated) | `#182233` | `--card2: #121c31` | different |
| border-subtle | `rgba(255,255,255,.08)` | `--line: #1e2b4a` (solid navy, not white-alpha) | **different model**, not just different value |
| border-default | `rgba(255,255,255,.12)` | `--line2: #32456e` (solid navy) | **different model** |
| text-primary | `#F7F8FA` | `--text: #edf1fb` | close, slight blue cast |
| text-secondary | `#A9B4C2` | `--text-2: #a9b6d3` | close |
| text-muted | `#6F7B8B` | `--text-3: #7385a8` | different — more saturated blue |
| brand-teal | `#2DD4A8` | `--positive: #2dd4a8` | **exact match** |
| brand-gold | `#F5B942` | `--lime: #f5b942` | **exact match** (note: variable is still named `--lime` — value/name mismatch inherited from the lime→navy rollback) |
| semantic-positive | `#34D399` | reuses `--positive` (`#2dd4a8`) | DESIGN.md wants a *distinct* positive-green from brand-teal; site conflates the two into one variable |
| semantic-negative | `#F87171` | `--danger: #f2555a` | different — site's red is more saturated |
| semantic-warning | `#F5B942` (= brand-gold) | `--attention: #f0a132` | different — site deliberately keeps warning ≠ gold (arguably healthier separation, see §4) |
| semantic-info | `#60A5FA` | `--blue: #5ba0f5` | close |

Colors outside the token system, used by domain necessity: JRA post-position frame colors `.w1`–`.w8` (8 fixed real-world colors), podium medal gradients (`resb.r1/r2/r3`), grade badge colors in `explain.html`. These are legitimately exempt — not brand color, they encode a real-world convention — but are worth naming as "known, intentional token-system exceptions" rather than leaving them silently unaccounted for.

### 2b. Typography

- **Primary font family**: DESIGN.md §4 wants `Geist, Inter, "Noto Sans JP", system-ui`. Live `--sans` is `"Zen Kaku Gothic New", "Noto Sans JP"` — neither Geist nor Inter is loaded or referenced anywhere in the codebase. Full conflict on the Latin/UI typeface; largely moot for JP glyphs since JP text renders via Noto Sans JP either way in both systems.
- **Mono/numeric font**: DESIGN.md wants `JetBrains Mono, IBM Plex Mono, ui-monospace`. Live `--disp` is `"IBM Plex Mono", "Zen Kaku Gothic New", "Noto Sans JP", monospace` — first choice matches. Reasonably aligned on `index.html`; **broken on `explain.html`** per §1 above (font not loaded there).
- **Font-weight ceiling**: DESIGN.md §4 says avoid 700+/900. Live CSS uses `font-weight: 900` on hero/landing headlines (`.hero-h1`, `.lp-h2`), race marks (`.mark{font-weight:900}`), win-stamps (`.hit-stamp`, `.won-badge`), plus `font-weight: 700` dozens of times across buttons, badges, labels, ticket types. This is pervasive, not incidental — heavy weight is doing real scan-speed work on the numbers/marks a user needs to find fastest in a dense table.
- **Body size**: DESIGN.md §5 "body: 14px/1.5". Live `body{font-size:17px; line-height:1.55}` (15px at ≤560px). DESIGN.md's own §12 separately specifies "12–14px" for race-table typography specifically — so DESIGN.md is not internally consistent between its global body size and its table-specific size either; neither number matches the live 17px base.
- **Type scale discipline**: DESIGN.md §5 implies a fixed discrete scale. Live CSS uses `clamp()` fluid type for hero/landing headings, plus frequent half-pixel sizes (13.5, 14.5, 15.5, 16.5, 17.5, 19, 21, 23, 25px…) tuned per component rather than snapped to a scale.

### 2c. Spacing

DESIGN.md §6: 4px grid, allowed core values {4,8,12,16,20,24,32,48,64}, explicitly "do not arbitrarily use 13/18/27/37px." Live CSS uses `clamp()` for major paddings (`--pad: clamp(14px,3vw,30px)` — neither endpoint is on the allowed list) plus many off-grid values in component rules (9px, 13px, 14px, 18px, 22px, 34px, 46px…). Pervasive but low-severity — reads as deliberate per-component hand-tuning, not carelessness.

### 2d. Border radius

| Role | DESIGN.md | Live token | Verdict |
|---|---|---|---|
| small/icon | 4px | `--r-sm: 2px` | sharper than spec |
| button/input | 6px | buttons use `--r-sm` (2px) | sharper |
| app card | 8px | `--r-md: 4px` | sharper |
| pill | 9999px | `--r-pill: 4px` | **not a pill at all** — the token name promises a capsule shape and delivers a barely-rounded rectangle |

No 20–32px "giant rounded card" violations found anywhere in race/data views (good — the one explicit §12 anti-pattern DESIGN.md calls out by number is absent). `explain.html`'s inline style defines its *own* radii (14px cards, 8px buttons/inputs) that match neither `style.css`'s tokens nor DESIGN.md — a third, drifting radius system on one page.

### 2e. Shadows / glow — the largest single conflict

DESIGN.md §3 "Avoid drop shadows... No large glow effects" and §23 forbids "glow around every element" / "excessive shadows." Live CSS has **40+ distinct `box-shadow` declarations**, and glow is a consistent, deliberate signature: `--shadow-card` on every `.card`, `--shadow-pop` on hover, colored glow on `.btn-prime:hover`, on every `.jbadge` variant, on active `.venue`/`.modenav button`/`.hit-chip`/`.rpill`, on podium medal `.resb.r1`, on the active view-tab underline `.vt.on::after`, on `.tk.tk-hon`, on `.adv-medal.gA`, on `.ticket`, on the Grade Scope `.gs-badge`, on the level-spectrum pointer badges (`.mls-mark b`, `.cg-mark b`), on `#drawer`'s entrance shadow. This is not stray CSS debt — it's the UI's "instrument-panel glow" identity, applied with the same 2–3 shadow recipes over and over.

### 2f. Cards & gradients

`.card` itself = gradient sheen background + solid navy border + drop shadow — three separate DESIGN.md §3 rules broken in one base class. §23's "no decorative gradients" is broken far more broadly: every active/selected/win state uses a 2-stop same-hue gold gradient (`.modenav button.on`, `.venue.on`, `.jbadge.*`, `.hitchip`, `.resb.r1/r2/r3`, `.ticket::before`, `.gs-badge`, `.adv-medal.gA`) plus one ECharts area-fill gradient on the career chart. The primary CTA (`.btn-prime`) itself is **not** a gradient (solid `--lime` fill) — so §10's specific "no gradient CTA" rule is technically honored even while the broader gradient pattern is not.

### 2g. Buttons & primary-interaction color

DESIGN.md §2 assigns "primary interaction / selected states / focus" to **teal** and treats **gold** as a brand accent to "use sparingly." Live reality is close to the inverse: `.btn-prime` (the hero CTA), the active mode-nav pill, the active venue tab, the active race pill, and the focus ring (`--ring: 0 0 0 3px rgba(245,185,66,.22)` — gold, 3px, box-shadow-based, not the "2px brand-teal" §22 asks for) are **all gold**. Teal in production is used almost exclusively as the *semantic-positive* signal (ROI-positive numbers, "GO" judge-badge, under-market chips) — i.e., production already treats teal the way DESIGN.md describes *semantic-positive*, and gold the way a reasonable person would expect a hard-won brand color to be used: as the default interactive/selected accent. This is a color-role swap between the doc and the app, not a token-value typo.

### 2h. Badges

DESIGN.md §11: height 20–24px, padding 4px 6px, radius 4px, low-alpha semantic background. `.tdchip` (turf/dirt) and `.vschip` (value chip) genuinely match this — low-alpha background + border, restrained. `.jbadge` (GO/慎重/見送り) does not: 15.5px text, 3px 16px padding, full gradient fill + glow, closer to a "pill button" than a badge. `.won-badge`/`.hitchip` are deliberately celebratory (solid gold gradient, bold dark text) — a "win stamp," not a status badge in DESIGN.md's sense.

### 2i. Navigation — highest-confidence finding in the whole audit

DESIGN.md §9 gives this exact pairing:

> Preferred: `予想` / `成績` / `分析` — rather than: `🏇 予想` / `📊 成績` / `🔬 分析カード`

`index.html`'s actual `.modenav` markup is, character for character:
```html
<button data-mode="races" class="on">🏇 予想</button>
<button data-mode="results">📊 成績</button>
<a href="explain.html"><button type="button">🔬 分析カード</button></a>
```
This is the exact "wrong" example from the spec, not a paraphrase — almost certainly DESIGN.md §9 was written while looking at this exact line. Domain marks (◎〇▲△) are correctly untouched elsewhere; this is specifically about the three decorative nav-icon emoji. `explain.html`'s own `<h1>🏇 レース分析カード…</h1>` is the same pattern in a page header rather than nav (softer case, same root cause).

The active-state styling compounds it: DESIGN.md wants "subtle bg-2 surface, teal indicator or text, **never a large colored button**." `.modenav button.on` is a full gold-gradient filled pill with glow — precisely the anti-pattern named.

### 2j. Race table (§12) vs. actual `.hrow` columns

| DESIGN.md example column | Actual column | Note |
|---|---|---|
| 印 | ✓ `.mark` | match |
| 馬番 | ✓ `wk()` | match |
| 馬名 | ✓ `.hname` (+ jockey/weight sub-line) | match |
| 人気 | ✓ `.c-ninki` | match |
| 単勝 (odds) | **absent** — no raw decimal-odds column | table shows AI-side probabilities only; likely intentional given the JRA-VAN posting-guideline work already done on this site ([[project_jravan_guideline_compliance]]) — needs confirming, not a pure style question |
| AI勝率 | ✓ `.pwin` | match |
| TOP3 | ✓ `.psho` (labeled 複勝圏, same concept) | match |
| Model Odds | not shown as decimal odds | shown as % probability instead |
| Market | shown as categorical chip (妙味/過剰/中立) via `vsChip()` | arguably better UX than a raw number, but not literally what DESIGN.md's example shows |
| Edge | no numeric edge value in this table | categorical only |
| *(not in DESIGN.md)* | 近5走 sparkline | additive, must be kept per hard constraint (no information removal) |

Row mechanics: sticky header ✓, right-aligned numerics ✓, subtle row dividers ✓, `font-variant-numeric: tabular-nums` applied broadly via `.num` ✓ — these all match DESIGN.md well. Row height is not verifiable from CSS alone (`.hrow` padding 7px 14px around a two-line name+sub-line cell likely renders taller than the 40–44px target) — flagged for live measurement in Phase 1, not asserted here.

### 2k. Numeric formatting (§14) — best-compliance area

`pct()`/`num()` helpers consistently format to 1 decimal (`31.4%`, not `31.423845%`); `explain.html`'s `fmtOdds` does the same. Copy is restrained ("目安"/"参考"/"投資助言ではありません" appears repeatedly) — no "guaranteed profit" language found anywhere. This section of DESIGN.md is already well satisfied by the live code.

### 2l. Responsive (§20)

Single `--maxw: 1240px` used for both app views and marketing/landing sections, where DESIGN.md wants two values (1440 app / 1200 marketing). Mark/umaban/name are never hidden at any breakpoint — the one hard requirement DESIGN.md states is honored. Several interactive chip/pill elements (`.hit-chip`, `.venue`) likely render under the 40px touch-target minimum on mobile — flagged for live measurement, not confirmed from CSS.

---

## 3. §12–14 / §23 rule-by-rule checklist

| Rule | Status | Evidence |
|---|---|---|
| §12 race table has 印/馬番/馬名/人気/AI勝率/TOP3 | ✅ pass | see 2j |
| §12 sticky header, right-aligned numerics, tabular-nums | ✅ pass | `.hh{position:sticky}`, `.ta-r`, `.num` |
| §12 40–44px row height | ⚠️ unverified | needs live measurement |
| §13 no full-gold ◎ row fill | ✅ pass | `.hrow.honmei` uses a subtle left-edge gradient wash, not a filled row |
| §13 marks readable without color | ✅ pass | ◎〇▲△ are symbols, not color-only |
| §14 stable, non-overprecise number formatting | ✅ pass | see 2k |
| §14 no "guaranteed profit" language | ✅ pass | copy audited, none found |
| §23 no AI-purple gradients | ✅ pass | no purple anywhere in the palette |
| §23 no glassmorphism | ❌ fail | `backdrop-filter: blur()` on `.topbar`, `.hh` sticky header, `#overlay` — see 2m |
| §23 no giant rounded SaaS cards | ✅ pass | max radius found is 14px (`explain.html`), far from "giant" |
| §23 no glow around every element | ❌ fail | see 2e |
| §23 no excessive shadows | ❌ fail | see 2e |
| §23 no decorative gradients | ❌ fail | see 2f |
| §23 no emoji navigation | ❌ fail | see 2i — exact match to the forbidden example |
| §23 no random color badges | ✅ pass (partial) | badge colors are semantic, not random — though badges are saturated/glow rather than DESIGN.md's low-alpha style (2h) |
| §23 no oversized dashboard KPI cards | ⚠️ borderline | `.rs-sum-grid` (4-card ROI grid) + `.gauges` (4-donut row) — needs visual judgment, see 4 |
| §23 no unnecessary charts | ✅ pass | charts are opt-in (secondary tabs), directly serve the product's own stated identity (§1: "model vs market comparison... explainability") |
| §23 no gradient text | ✅ pass | none found |
| §23 no animated background blobs | ✅ pass | none found |
| §23 no fake terminal decorations | ✅ pass | none found |
| §23 no casino/sportsbook aesthetics | ⚠️ borderline | `.hit-card.big` escalates styling for ¥10,000+ wins; see 2m |

### 2m. Two additional findings outside the strict checklist

- **Functional vs. decorative blur**: the three `backdrop-filter` uses are all functional (sticky-header legibility over scrolling content, modal scrim) — none are decorative "frosted card" chrome. The literal rule is broken; the spirit arguably is not.
- **§18 "positive and negative results use the same component system"**: aggregate stats (ROI, hit-rate) already include losses honestly, and the site's own hero copy says "的中も外れも、全部見せる" (show hits and misses alike). But the 成績 tab's card-gallery (`.hit-grid`) is a **wins-only** showcase — there's no parallel losses gallery, and `.hit-card.big` gives extra-prominent stamp styling to bigger wins specifically. This is an information-architecture question (what to show), not a token question — flagged, not resolved here.

---

## 4. Conflict classification

**KEEP CURRENT** = live CSS stays, no doc or code change needed (already fine or DESIGN.md rule doesn't actually apply well here).
**ADOPT DESIGN.md** = change the site to match the doc.
**MODIFY DESIGN.md** = the doc's rule should change to match a deliberate, already-shipped decision.
**DEFER** = genuine open question; needs a user call, not a unilateral resolution.

| # | Conflict | Classification | Why |
|---|---|---|---|
| 1 | Emoji in primary nav (§9) | **ADOPT DESIGN.md** | Exact match to DESIGN.md's own "wrong" example. Lowest risk, highest-confidence fix in the audit. |
| 2 | `.modenav button.on` = "large colored button" (§9) | **MODIFY DESIGN.md** | Bundle with #4 below — once gold is accepted as the primary-interaction color, a filled-segment toggle for a true binary/ternary mode switch is a defensible, common pattern (iOS segmented control), and the same idiom already extends consistently to `.venue.on`/`.rpill.on`. Special-casing just modenav would be inconsistent. |
| 3 | `explain.html`'s drifting inline design system (Oswald font, own 14px/8px radii) | **ADOPT DESIGN.md** (i.e. converge onto shared `style.css` tokens) | This is really "site vs. itself," the safest kind of fix — it only touches one secondary page and doesn't require picking a side in any brand debate. |
| 4 | Primary-interaction color: gold (live) vs. teal (DESIGN.md §2) | **MODIFY DESIGN.md** | Navy×gold was a deliberately fought-for, twice-reconsidered brand decision per [[project_site_lime_redesign]] (lime tried and explicitly rejected by the user, reverted to navy×gold). Reassigning primary-interaction to teal would mean re-touching the CTA, every active nav/tab/pill state, and the focus ring — a large, high-risk visual change to satisfy a color-role rule, when swapping two role *descriptions* in the doc is a zero-risk fix. Recommend: gold = primary-interaction (brand + action), teal = semantic-positive + secondary informational accent (already how it's used today). |
| 5 | Border model: solid navy hex (live) vs. `rgba(255,255,255,.08–.18)` (DESIGN.md §8) | **MODIFY DESIGN.md** | DESIGN.md's own cited references don't agree with each other here — Linear uses dark hairline borders (`#23252a` on `#0f1011`, i.e. *not* white-alpha), OpenSea uses white-alpha inset rings, Harness uses light-on-dark borders. A blanket "always white-alpha" rule contradicts one of the three named influences. Recommend narrowing §8 to describe the site's existing navy-hairline approach as the accepted "Linear-style dark border," reserving white-alpha as an option rather than a mandate. |
| 6 | Shadow/glow: pervasive (live) vs. "avoid" (§3, §23) | **MODIFY DESIGN.md** | 40+ occurrences, applied via 2–3 consistent recipes, functioning as the UI's visual identity ("instrument panel" glow). Purging it is a large, high-regression-risk undertaking, and doing so risks pushing the UI toward the *generic flat SaaS* look DESIGN.md's own §1 says to avoid. Recommend narrowing §23 to ban gratuitous/decorative shadow while explicitly allowlisting: focus rings, active/selected-state accent glow, hover lift. |
| 7 | Gradients: pervasive on active/win states (live) vs. "no decorative gradients" (§23) | **MODIFY DESIGN.md** | Same bucket as #6 — all gradients found are restrained 2-stop, single-hue (gold) fills on badges/active-states, not multi-hue decoration. Recommend permitting "monochromatic 2-stop gradient, single brand hue, on badges/active-states only," keeping the ban on multi-hue/background decorative gradients. |
| 8 | Font-weight ceiling: 700–900 used widely (live) vs. "avoid 700+" (§4) | **MODIFY DESIGN.md** | None of the three cited references support a strict "avoid 700+" reading — Harness explicitly allows 600 for headings, Linear caps at 590 (i.e. a heavy-medium, closer in spirit to 600 than to "avoid 700"), OpenSea tops out at 500. DESIGN.md's rule is stricter than any single reference and isn't grounded in one. Recommend raising the ceiling to "600 general UI; 700–900 permitted specifically for race marks (◎〇▲△), hero/landing display numbers, and win-result stamps" — a narrow, principled carve-out rather than a blanket cap. |
| 9 | Body size 17px (live) vs. "body: 14px" (§5) — and vs. §12's own "12–14px" table spec | **MODIFY DESIGN.md** | DESIGN.md is not internally consistent between its global body size and its own race-table size. Recommend explicitly splitting "prose body" (14–17px, this product's data needs glance-legibility) from "table/data-dense" (12–15px) as two named scales rather than one flat number. |
| 10 | Focus ring: gold, 3px, box-shadow (live) vs. "2px brand-teal" (§22) | **MODIFY DESIGN.md** | Same root cause as #4 — once gold is accepted as primary-interaction, the focus ring should follow it. Recommend "2px gold focus ring" (keep width discipline, swap color to match #4's resolution). |
| 11 | `backdrop-filter` blur on sticky topbar/table-header/drawer-scrim (live) vs. "no glassmorphism" (§23) | **MODIFY DESIGN.md** | All three uses are functional legibility patterns (sticky-over-scroll, modal scrim), not decorative "frosted card" chrome. Recommend narrowing §23 to ban decorative glass-card fronts specifically, while permitting blur on sticky/overlay surfaces. |
| 12 | Loading-state infinite animations (`pulse`, `skshine`) vs. "no continuous glowing/looping decorative animation" (§21) | **MODIFY DESIGN.md** | Standard, temporary, functional loading indicators — not decorative chrome. Recommend an explicit loading-state carve-out in §21's wording. |
| 13 | `--r-pill: 4px` names a pill but isn't one | **Doc-independent hygiene fix** | Not really a DESIGN.md conflict — a misleading internal variable name. Low-priority rename/introduce-a-real-`--r-pill` candidate whenever CSS is next touched. |
| 14 | `.rs-sum-grid` (4 KPI cards) + `.gauges` (4 donut rings) vs. "no oversized dashboard KPI cards" (§23) / "one restrained metrics strip" (§18) | **DEFER** | Genuinely borderline; needs a rendered screenshot side-by-side, not a CSS-reading judgment call. |
| 15 | `.hit-grid` wins-only gallery, `.hit-card.big` escalation for big wins vs. §18 "positive and negative use same component system" | **DEFER** | Information-architecture decision (add a losses gallery? reframe the wins gallery?), not a style-token fix. |
| 16 | Missing literal `単勝`/odds column in race table (§12 example) | **DEFER** | Possibly intentional given prior JRA-VAN posting-guideline compliance work ([[project_jravan_guideline_compliance]]) — needs confirming against that constraint before treating as a gap to fill. |
| 17 | bg-1/bg-2/bg-3 hex values don't match DESIGN.md exactly (2a) | **MODIFY DESIGN.md** | Canvas (bg-0) matches exactly and was clearly the sampled anchor; the other three levels read as estimated rather than sampled. Low-risk documentation correction to the site's real values. |
| 18 | Dual max-width (1440 app / 1200 marketing, §20) vs. single `--maxw:1240px` (live) | **DEFER** | Cosmetic, low priority; needs a call on whether marketing sections should visually narrow relative to app views. |
| 19 | §12 sticky header, tabular-nums, row dividers, right-alignment; §14 number formatting; §16 marks-as-list (not cards); §17 Today's Focus (no "BEST BET" language); §19 drawer-as-secondary-panel; §21 `prefers-reduced-motion`; §22 marks/badges never color-only | **KEEP CURRENT** | Already compliant — see 2j/2k and the checklist in §3. `#landing`'s "01/02/03" sections in particular read as near-literal implementations of DESIGN.md §15–17's own prescriptions; worth preserving as reference examples of what "compliant" looks like elsewhere in the same codebase. |

---

## 5. Phase 1 migration plan (ADOPT-classified items only)

Only items #1 and #3 above are classified ADOPT (site changes to match the doc). Everything else in this table is deferred to *after* the MODIFY-classified DESIGN.md text is actually revised (can't migrate code toward rules that are themselves still being corrected).

### 5a. Remove decorative emoji from primary navigation

- **Files**: `site/index.html` (lines 125–127, `.modenav` buttons), `site/explain.html` (line 67, `<h1>` prefix)
- **Selectors/markup affected**: `.modenav button[data-mode="races"]`, `.modenav button[data-mode="results"]`, the `分析カード` button/link, `explain.html`'s `<h1>`
- **Change**: `🏇 予想` → `予想`, `📊 成績` → `成績`, `🔬 分析カード` → `分析カード`; drop the `🏇` prefix from `explain.html`'s `<h1>`. No CSS changes needed — button padding/layout is unaffected by removing a leading glyph+space.
- **Expected visual change**: nav labels lose their icon prefix; slightly less visual noise in the topbar. Nothing else moves (buttons aren't icon-sized, just text+emoji today, so no layout reflow expected).
- **Regression risk**: very low. No logic depends on the emoji characters (JS reads `data-mode`, not textContent). Only risk is aesthetic — worth a quick screenshot comparison, not a functional test.
- **Verification**: `preview_start` the site (or open `index.html` via a local server, since `fetch()` calls need HTTP not `file://`), screenshot the topbar before/after, confirm `explain.html` header renders without the emoji, confirm `data-mode` click handlers still work (`setMode()` in `app.js` is untouched).

### 5b. Converge `explain.html`'s inline design system onto shared tokens

- **Files**: `site/explain.html` (lines 20–60, the inline `<style>` block)
- **Selectors affected**: `.card` (14px→ align to `--r-md`), `select`/`.toolbar button`/`.backlink` (8px→ align to `--r-md` or a shared `--r-input`), the `Oswald` font `<link>` (line 18) and any rule relying on it
- **Change**: replace hardcoded `border-radius: 14px` / `8px` with `var(--r-md)` (or whatever radius token the MODIFY-DESIGN.md pass settles on for cards/inputs); drop the Oswald Google Fonts `<link>` and let headings fall back to `var(--disp)` (which, once Zen Kaku Gothic New / IBM Plex Mono are also linked on this page — currently missing, see 2b — will render as intended rather than silently degrading to Noto Sans JP).
- **Expected visual change**: `explain.html` card corners get slightly sharper (14px→4px, a real visible change), page headings (`.ctitle .nm`, `.ehead h1`) switch typeface from Oswald to IBM Plex Mono/Zen Kaku Gothic New — a more noticeable visual shift than 5a, since Oswald is a distinctly different letterform (condensed grotesque) from the mono/JP-gothic pairing used everywhere else.
- **Regression risk**: low-medium. Purely visual, but the font swap is the most visible change in this whole Phase 1 list — worth a full side-by-side screenshot of `explain.html` before/after, checking that Japanese race names, English abbreviations, and numeric columns all still read cleanly at the existing column widths (Oswald is narrower than the mono fallback; some truncated `white-space:nowrap; overflow:hidden` cells could wrap differently).
- **Verification**: load `explain.html` with real data (a date that has `data/explain/{date}.json`), screenshot before/after, specifically check the `.ctitle .nm` (race name) and table header row for any new truncation/overflow.

### 5c. (Optional, low priority) Fix `--r-pill` naming

- **Files**: `site/css/style.css` (token declaration ln 46, ~15 call sites using `var(--r-pill)`)
- **Change**: either (a) rename `--r-pill` → `--r-chip` and leave its value at 4px, introducing a new, genuinely-9999px `--r-pill` for future use, or (b) leave the name and just document that "pill" here means "chip" in this codebase. Not urgent; bundle into whichever future pass touches radius tokens for the MODIFY-DESIGN.md items.
- **Regression risk**: none if done as a pure rename (find/replace, no value change).

---

## 6. Recommended DESIGN.md revisions (not yet applied)

DESIGN.md was intentionally left untouched this turn — see the note at the end of this document for why. These are concrete, ready-to-apply proposed edits corresponding to the MODIFY-classified rows in §4, for the user to accept, adjust, or reject before they're written into the file:

1. **§2 Brand Colors** — swap the stated roles of teal and gold: gold becomes "primary interaction / selected states / focus / CTA fill," teal becomes "semantic-positive (ROI, hit-rate, 'GO' status) / secondary informational accent." Update `--semantic-positive` to equal the live `--positive` value (`#2DD4A8`) explicitly rather than implying it should differ from brand-teal. Update bg-1/bg-2/bg-3 hex to the live values (`#0A101E` / `#0D1424` / `#121C31`) or explicitly mark them as a "target palette, not yet migrated" if the user wants the *doc* values to eventually become the *real* values instead.
2. **§3/§8 Borders** — replace the blanket "always `rgba(255,255,255,.08–.18)`" rule with: "borders may be either white-alpha (OpenSea-style, for dense data surfaces) or a dark navy hairline consistent with the surface scale (Linear-style) — pick one per page/section and stay consistent within it; the current site uses the dark-hairline approach throughout and should keep doing so unless a specific surface calls for the white-alpha alternative."
3. **§3/§23 Shadows & glow** — replace "avoid drop shadows... no large glow effects... no glow around every element" with an allowlist: focus rings, active/selected-state accent glow (badges, pills, tabs), hover lift shadows are permitted; ambient/background shadow and shadow used purely as decoration on static, non-interactive elements is not.
4. **§23 Gradients** — replace blanket "no decorative gradients" with: "gradients are permitted as 2-stop, single-brand-hue fills on badges, active/selected states, and win/result indicators; multi-hue or background decorative gradients remain forbidden."
5. **§4 Font weight** — replace "avoid 700+/900" with: "600 for general UI emphasis; 700–900 permitted specifically for race marks (◎〇▲△), hero/landing display numbers, and win-result stamps, where scan-speed on a small number of high-priority glyphs matters more than general restraint."
6. **§5/§12 Type scale** — split into two named scales: "prose/UI body: 14–17px" and "table/data-dense: 12–15px," rather than one flat 14px body rule that matches neither current usage context.
7. **§22 Focus ring** — change "2px brand-teal focus ring" to "2px brand-gold focus ring" (keep the 2px width discipline; align color to the #1 resolution above). Note the live ring is currently 3px — flag for the ADOPT/MODIFY decision on whether to tighten the ring to 2px live, or accept 3px in the doc too.
8. **§23 Glassmorphism** — narrow "no glassmorphism cards" to "no decorative frosted-glass card fronts; `backdrop-filter` blur is permitted on sticky headers/nav over scrolling content and on modal/drawer scrim overlays, where it serves legibility rather than decoration."
9. **§21 Animation** — add an explicit carve-out: "infinite/looping animation is permitted for temporary loading and skeleton states; it remains forbidden for static decorative chrome."

Items **not** in this list (the DEFER rows in §4 — KPI-card density, wins-only results gallery, the missing odds column, dual max-width) are genuine open questions rather than proposed text, and are called out separately so they don't get silently folded into a documentation edit.

---

## 7. Decision Log

| Item | Verdict |
|---|---|
| Copy the three external reference docs into the repo (`docs/design-references/*.md`) | **Done this session** — didn't exist before, needed as stable audit input |
| Root `DESIGN.md` content | **Left unchanged.** "Keep DESIGN.md as a draft" plus the scale of what this audit found (many rules read as aspirational/self-derived rather than descriptive) argued against silently rewriting the user's brand-color-role and visual-language decisions inside this same turn. §6 above gives the concrete proposed edits for explicit sign-off instead. |
| Emoji in primary nav (§9) | **ADOPT DESIGN.md** — fix in Phase 1, see 5a. Highest confidence, lowest risk finding in the audit. |
| `explain.html` drifting inline styles/fonts | **ADOPT DESIGN.md** (converge onto shared tokens) — fix in Phase 1, see 5b. |
| `--r-pill` naming | Doc-independent hygiene fix, low priority, see 5c. |
| Primary-interaction color (gold vs. teal), borders, shadows/glow, gradients, font-weight ceiling, body/type-scale, focus-ring color, glassmorphism, loading-animation wording, bg-1/2/3 hex | **MODIFY DESIGN.md** — 9 concrete proposed edits in §6, none applied yet, all pending user sign-off. |
| KPI-card/gauge density, wins-only results gallery + big-win escalation, missing literal odds column, dual max-width | **DEFER** — genuine open questions (visual judgment call, product/IA decision, or compliance cross-check), not resolvable from a token audit alone. |
| Everything in §3's checklist marked ✅ (sticky header, tabular-nums, number formatting, no-guaranteed-profit copy, marks-as-symbols, no purple/blobs/fake-terminal/gradient-text, drawer-as-secondary-panel, `prefers-reduced-motion`, the landing page's §15–17 sections) | **KEEP CURRENT** — already compliant, no action. |

---

## Git status (end of Phase 0)

Only documentation was written this session:
- `docs/design-references/Linear_DESIGN.md` (new)
- `docs/design-references/openSea_DESIGN.md` (new)
- `docs/design-references/Harness_DESIGN.md` (new)
- `docs/design_audit_phase0.md` (new, this file)

**No file under `site/` was read-then-edited to produce this report — only read.** `git status` / `git diff --stat` output confirming this is appended to the end of the chat message that accompanies this file, not duplicated here since it is a live command result, not a design artifact.
