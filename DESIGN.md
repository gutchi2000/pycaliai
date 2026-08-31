# PyCaLiAI Design System v1.0

> Quantitative horse-racing intelligence terminal.
> Precision, evidence, probability, restraint.

Theme: Dark

Status: **Frozen v1.0.** Resolves all findings from [docs/design_audit_phase0.md](docs/design_audit_phase0.md) and [docs/design_decision_freeze_phase05.md](docs/design_decision_freeze_phase05.md). Supersedes the unversioned draft this file replaces.

Design direction — reference weighting (structure and philosophy, not literal color or brand):
- Linear 50% — system architecture, precision, restraint discipline
- OpenSea 35% — dense data interfaces, numeric presentation, density
- Harness.io 15% — sparse brand/marketing surfaces, display typography confidence
- Never clone any reference literally. Where a reference's specific rule conflicts with PyCaLiAI's existing navy × gold identity, **the brand wins** — see §1 and §3.

---

## 1. Product Identity

PyCaLiAI is not a generic betting site.

It is a quantitative horse-racing analysis product built around:
- machine-learning predictions
- calibrated probability
- model vs market comparison
- transparent historical performance
- explainability

The UI must feel:
- analytical
- precise
- transparent
- professional
- information-dense

It must NOT feel:
- casino-like
- gambling-promotional
- crypto-hype
- generic AI SaaS
- neon cyberpunk
- playful consumer app

Data is the visual content.

---

## 2. Surfaces: Marketing vs Application/Data

Every rule below that says "Marketing" or "Application/Data" refers to this split. A rule that names neither applies to both.

**Marketing surfaces** — homepage hero, the HONEST RECORD / HOW TO READ / TODAY'S FOCUS / RACE INDEX landing sections, any future brand or about page. Read as confident and warm. May spend a restrained gradient or glow accent on up to one focal element per viewport.

**Application/Data surfaces** — the race table and its detail drawer, results tab, analytics tabs (全頭分析 / コース / 血統), Grade Scope, explainability cards, and the shell chrome around them (topbar, venue tabs, race strip, view tabs). Read as flat, dense, and quiet. Repeated elements — badges, pills, table rows, cards — must not each carry their own decoration; decoration that looks good once looks noisy at 20×.

One page can contain both: a homepage's hero and landing sections are Marketing; the race table beneath them is Application/Data.

---

## 3. Brand Colors

Use the existing PyCaLiAI identity.

```
--bg-0: #070B14;
--bg-1: #0A101E;
--bg-2: #0D1424;
--bg-3: #121C31;

--border-subtle: rgba(30,43,74,0.5);   /* navy, lightest — lightest dividers only */
--border-default: #1E2B4A;             /* navy — standard card/table border */
--border-strong: #32456E;              /* navy — emphasis dividers */

--text-primary: #F7F8FA;
--text-secondary: #A9B4C2;
--text-muted: #6F7B8B;

--brand-gold: #F5B942;
--brand-teal: #2DD4A8;

--semantic-positive: #2DD4A8;   /* == --brand-teal: one value, not two competing meanings */
--semantic-negative: #F2555A;
--semantic-warning: #F0A132;    /* deliberately ≠ brand-gold, see Rules below */
--semantic-info: #5BA0F5;
```

Rules:

Gold:
- primary interaction — buttons, active/selected states, focus
- PyCaLiAI brand fill — the interface's default "this is active/actionable" signal
- ◎ / primary model highlight
- Marketing surfaces: also the accent for hero and HONEST RECORD focal numbers

Teal:
- semantic-positive (ROI at/above the control line, hit-rate, "GO" judge status)
- secondary informational accent (market-value-under indicators, AI/model informational emphasis where it is *not* the primary action)

Gold and teal are not interchangeable: gold answers "what can I act on, what is PyCaLiAI's own," teal answers "is this number good."

Warning is intentionally its own color, not a gold reuse — a warning state must never render identically to "this is clickable."

Red:
- semantic-negative only, never decorative

Never introduce AI-purple as a new brand color.

---

## 4. Surface System

Use tonal hierarchy first. Shadow only where this section and §22 (Interaction) explicitly allow it.

Level 0: `#070B14` — Page canvas.
Level 1: `#0A101E` — Navigation / main containers.
Level 2: `#0D1424` — Cards / table surfaces.
Level 3: `#121C31` — Hover / selected / elevated detail surfaces.

Cards:
`border: 1px solid var(--border-default);` *(navy hairline — see §9 Borders. Not white-alpha.)*

**Shadow and glow, by surface** (see §2 for the split):
- **Marketing**: a restrained glow is allowed on the primary CTA's hover state and on up to one hero/HONEST RECORD accent element per viewport — not more.
- **Application/Data**: no ambient or decorative shadow/glow. Shadow is limited to: the accessibility focus ring (§23), one consistent selected/active-state indicator per component family (a border-color change or thin accent bar — never a colored ambient glow), and a true overlay's entrance shadow (the detail drawer, a modal scrim).

**Glassmorphism:**
No decorative frosted-glass card fronts. `backdrop-filter` blur is permitted specifically on sticky headers/navigation over scrolling content, and on modal/drawer scrim overlays — legibility, not decoration.

---

## 5. Typography

Primary: Geist, Inter, "Noto Sans JP", system-ui, sans-serif

Numeric/Data: "JetBrains Mono", "IBM Plex Mono", ui-monospace, monospace

Japanese glyph fallback: "Noto Sans JP"

Use:
400 normal
500 emphasis
600 general UI emphasis (headings, buttons, badges, labels) — the ceiling for repeated interface text

Named exceptions — weight 700–900 permitted, unlimited by the 600 ceiling above. These are **PyCaLiAI-specific, not derived from Linear, OpenSea, or Harness** — none of the three use heavy weight this way:
- race marks (◎〇▲△) — must read instantly at a glance across a dense table
- Marketing hero / landing display headlines — a single, low-repetition brand moment. Note this specifically does *not* follow Harness's own hero treatment, which goes to weight 300 at a much larger (56–88px) size than PyCaLiAI's hero uses — the exception is PyCaLiAI's own, not borrowed
- win-result stamps (的中 / 万馬券) on Application/Data surfaces — an infrequent celebratory moment, not a per-row treatment (see §4's gradient/glow allowance and §24)

Outside these three named cases, treat 700+ as a signal the element should either become a named exception or drop to 600.

All quantitative columns must use:

`font-variant-numeric: tabular-nums;`

Numeric values requiring column comparison may use monospace.

Examples:
31.4%
3.20
4.60
+43.8%
448R
84.2%

---

## 6. Type Scale

Two named scales — do not measure everything against one flat number. Which one applies depends on the §2 surface split.

**Prose / Marketing body** (landing copy, drawer notes, narrative text, HONEST RECORD copy):
body: 14–17px / 1.5–1.9 — default 17px for primary reading copy. PyCaLiAI is read at a glance across dense tables; a slightly larger base size than a typical SaaS product is deliberate, not an oversight.
section-heading: 20px / 1.3 / 500
page-heading: 32px / 1.15 / 500
marketing-heading: 48px / 1.05 / 500 desktop, 36px mobile — Marketing surfaces only.

**Table / data-dense** (race table cells, analytics tables, badges, chips, sparklines) — Application/Data surfaces, OpenSea-influenced:
caption: 12px / 1.4
data: 13–15px / 1.4 — tabular-nums where the value is compared column-to-column

Do not make Application/Data headings 48–72px. Large typography is reserved for Marketing surfaces only.

The type scale defines core reusable tokens. Component-specific values explicitly defined elsewhere in this document may use intermediate sizes when required by hierarchy or density.

The nominal minimum type size for standard UI text is 12px. Dense, width-constrained secondary metadata may use a smaller size when necessary to preserve complete information and avoid truncation. Such exceptions must be justified by rendered-content measurement using representative real data — not assumed. Do not increase secondary metadata size when doing so causes meaningful content clipping.

---

## 7. Spacing

Base grid:
4px

Allowed core values:
4
8
12
16
20
24
32
48
64

Dense application UI:
element gap: 8px
card padding: 12–16px
section gap: 32–48px

Marketing UI:
card padding: 24px
section gap: 64–80px

Do not arbitrarily use:
13px
18px
27px
37px

---

## 8. Radius

small / icon:
4px

button / input:
6px

application card:
8px

marketing card:
12px

pill:
9999px

Do not use 20–32px rounded cards in race/data views.

---

## 9. Borders

On Application/Data surfaces, prefer borders over shadow for structure (see §4).

Default: `1px solid var(--border-default)` *(navy, `#1E2B4A`)*
Subtle: `1px solid var(--border-subtle)` *(navy, lightest — reserve for the least prominent dividers)*
Strong separator / elevated: `1px solid var(--border-strong)` *(navy, `#32456E`)*
Selected: `1px solid rgba(245,185,66,0.40)` *(gold — follows §3's primary-interaction color, not teal)*

PyCaLiAI uses a navy hairline, not white-alpha — every edge stays in the same hue family as the navy canvas. This is a deliberate PyCaLiAI choice, **not a literal copy of any one reference**: Linear itself uses a dark hairline (not white-alpha), OpenSea uses white-alpha, and Harness uses light-gray — the three references don't agree with each other here, so there is no single "correct" reference answer to defer to.

What *is* adopted from the references — structurally, not chromatically, mainly from OpenSea: on repeated-row surfaces, prefer a single thin divider between rows over wrapping each row in its own bordered card.

Do NOT copy Linear's 0.5px border globally.
1px is preferred for consistent rendering across Windows and non-Retina displays.

---

## 10. Navigation

Navigation should be text + restrained line icons.

Remove decorative emoji from primary navigation.

Preferred:

予想
成績
分析

rather than:

🏇 予想
📊 成績
🔬 分析カード

Horse-racing symbols such as:
◎ 〇 ▲ △
are domain information and must remain.

**List-style navigation** (venue tabs, race selection, view tabs — peer options a user picks one of many from):
- subtle bg-2 surface
- gold indicator or text (follows §3's primary-interaction color)
- never a large filled/glowing button

**Exclusive mode switches** (e.g. 予想/成績 — a true 2–3-way toggle, not a peer list) may use a filled segmented-control treatment. This is a different UI grammar than list navigation, and the distinction — not a blanket ban — is what the reference systems actually demonstrate: Linear, OpenSea, and Harness all avoid *filled list navigation*; none rule out a filled exclusive-choice toggle.

---

## 11. Buttons

Primary:
gold background
dark text (on-gold, e.g. `#2D2002`)

Secondary:
transparent
1px border (navy — see §9)
primary text

Ghost:
transparent
muted text

Height:
32–36px desktop
40–44px touch/mobile

Radius:
6px

Only one visually dominant primary action per region.

No gradient CTA. Primary stays a solid fill on both Marketing and Application/Data surfaces — see §4 for where gradient accents are permitted elsewhere.

---

## 12. Badges

Badges communicate state, never decoration.

Examples:
LIVE
UNDER
OVER
GO
SKIP
G1
G2
G3

Height:
20–24px

Padding:
4px 6px

Radius:
4px

Use semantic colors with low-alpha background.

Never display five unrelated bright badge colors in one row.

---

## 13. Race Table

Race data is the most important interface.

Desktop layout should prioritize scanning speed.

Recommended columns:

印
馬番
馬名
人気
単勝
AI勝率
TOP3
Model Odds
Market
Edge

Use:
12–15px typography (table/data-dense scale, §6)
tabular numbers
right alignment for numeric values
sticky header
subtle row separators (navy, §9)
40–44px row height

40–44px is the preferred row height for single-line dense data rows. Compound race rows containing a primary horse-name line plus secondary metadata may exceed this range when required for readability and hierarchy. Do not compress multi-line race rows solely to satisfy the nominal 40–44px target.

Horse name is primary text.

Secondary metadata:
jockey
weight
frame
running style

must be visually subordinate.

Do not convert every metric into a card.

Tables are preferred when users compare horses.

---

## 14. Race Row Hierarchy

Normal horse:
neutral surface

Hover:
bg-3

◎:
gold micro-accent only

〇:
teal/neutral secondary accent

Selected:
gold border or left indicator (follows §3)

Do not fill the complete ◎ row bright gold.

Marks must remain readable without relying on color.

---

## 15. Quantitative Values

Probability:
31.4%

TOP3:
64.8%

Odds:
3.20

Edge:
+43.8%

Formatting must be stable between rows.

Never write:

31.423845%

when UI precision does not require it.

Align decimals where practical.

Positive edge does not automatically mean success.
Avoid visual language implying guaranteed profitability.

---

## 16. Homepage

*(Entirely a Marketing surface — see §2.)*

Homepage should be less dense than race views.

Order:

1. Hero
2. Current status / LIVE
3. HONEST RECORD
4. HOW TO READ
5. TODAY'S FOCUS
6. RACE INDEX

Hero:
- minimal
- no generic AI illustration
- product UI or real quantitative data should be visual focus

HONEST RECORD:
Do not use three giant SaaS metric cards.

Prefer one restrained horizontal metrics strip or compact grid.

Primary values may be 28–36px.

Supporting labels 11–13px.

Confidence interval / sample definition must remain visible.

---

## 17. HOW TO READ

The five racing marks should behave as a compact legend/table.

Example structure:

◎ 本命      TOP3 62%
〇 対抗      TOP3 51%
▲ 単穴      TOP3 43%
△ 連下      TOP3 33%
△ 連下      TOP3 25%

Keep explanation available below or through expandable detail.

Do not create five oversized colorful cards.

---

## 18. TODAY'S FOCUS

One focused analytical panel.

Hierarchy:

Race
◎ horse
AI probability
market difference
short model rationale

Do not turn it into promotional "BEST BET" language.

No casino styling.

---

## 19. HONEST RECORD

Performance reporting must visually emphasize credibility.

Include:
- sample size
- evaluation period
- metric definition
- actual ROI/hit rate
- confidence intervals where available

Do not highlight only positive metrics.

Positive and negative historical results use the same component system.

---

## 20. Explainability View

*(An Application/Data surface — see §2.)*

Desktop:
central content + optional secondary detail panel

Do not create dozens of independent cards.

Group related information:

Model probability
Market
Past performance
Running style / pace
Track state
Ratings

Charts must have subdued grid lines.

Data takes precedence over decorative visualization.

---

## 21. Responsive Rules

Desktop, Application/Data surfaces:
max width 1440px

Desktop, Marketing surfaces:
max width 1200px

This dual max-width is a **PyCaLiAI product-specific decision — not reference-derived.** None of the three references actually differentiate marketing vs. application width; each uses one site-wide max-width. Race/data interfaces need horizontal room for dense comparative information (many columns, sparklines, side-by-side stats); marketing pages read better at a narrower, more composed width. Keep the two values distinct rather than converging to one.

Tablet:
collapse secondary side panels

Mobile:
single-column shell

Race tables may horizontally scroll.

Keep:
mark
horse number
horse name

visible or sticky whenever practical.

Touch target:
minimum 40px
prefer 44px

Do not shrink desktop tables until text becomes unreadable.

---

## 22. Interaction

Transition:
120–180ms

Use animation for:
hover
selected
drawer
tab switch
loading transition

Do not use:
parallax
continuous glowing animations on static or decorative chrome
floating blobs
decorative looping animation

Exception: infinite/looping animation (shimmer, pulse) is permitted for temporary loading and skeleton states — these disappear once real data arrives and signal "working," not decoration.

Respect:
prefers-reduced-motion

---

## 23. Accessibility

Never encode important meaning by color only.

Marks and status require text/symbol equivalents.

Keyboard focus:
2px brand-gold focus ring (follows §3's primary-interaction color; a visible ring, not a border-only change — the primary navigable surface, the race table, is dense and dark enough that a subtle border shift risks being missed)

Ensure readable contrast for:
body
muted text
tables
disabled controls

Avoid extremely low-opacity gray text.

---

## 24. Forbidden Patterns

Do NOT use:

- generic AI purple gradients
- decorative glassmorphism (frosted-glass card fronts) — functional blur on sticky/overlay surfaces is defined in §4, not covered by this ban
- giant rounded SaaS cards
- glow or gradient beyond what §4 permits per surface (Application/Data: functional only; Marketing: restrained, ≤1 accent per viewport)
- excessive shadows
- emoji navigation
- random color badges
- oversized dashboard KPI cards
- unnecessary charts
- gradient text
- animated background blobs
- fake terminal decorations
- casino / sportsbook aesthetics

---

## 25. Design Principle

When deciding between:

more decoration
or
more information clarity

always choose information clarity.

When deciding between:

another card
or
structured rows/table

prefer structured rows/table when comparison is the task.

The interface should look credible even if all brand color is removed.
Brand color is punctuation, not structure.
