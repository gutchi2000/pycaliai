# PyCaLiAI Design System — Phase 2D Implementation Report (smallest batch)

> Implements exactly the two approved items: the DESIGN.md row-height clarification and the mobile `.hsub` font-size bump. Not committed — left for review, per instructions.
> Date: 2026-08-31

---

## ⚠️ Headline finding: the `.hsub` change measurably fails one of its own named acceptance criteria

Implemented exactly as instructed — `10px → 12px`, mobile breakpoint only. But live measurement shows **9 of 14 rows (64%) in the sampled race now truncate with an ellipsis** where none did before (visually confirmed by screenshot, not just computed). This is real clipping, not a false alarm: at 10px, every jockey name + "(替)" jockey-change marker fit inside the fixed column width; at 12px, the same text no longer fits for rows with longer names or the "(替)" suffix. `no clipping` was one of the four things I was asked to confirm — it does not hold. I did not revert or pick a different value myself, since 12px was given as an explicit target (unlike `.ticket::before` last phase, this task carried no "if a criterion fails, revert" clause) — flagging it clearly here instead, for you to decide: keep 12px and accept the truncation trade-off, pick a smaller intermediate value, or revert. Full detail in §2.

---

## 1. Exact diff

```diff
--- a/DESIGN.md
+++ b/DESIGN.md
@@ (§13 Race Table, "Use:" list)
 subtle row separators (navy, §9)
 40–44px row height
 
+40–44px is the preferred row height for single-line dense data rows. Compound race rows containing a primary horse-name line plus secondary metadata may exceed this range when required for readability and hierarchy. Do not compress multi-line race rows solely to satisfy the nominal 40–44px target.
+
 Horse name is primary text.

--- a/site/css/style.css
+++ b/site/css/style.css
@@ (@media (max-width: 560px) block)
   .hname { font-size: 12.5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
-  .hsub { font-size: 10px; }
+  .hsub { font-size: 12px; }
   .kyaku { font-size: 9px; padding: 0 4px; margin-right: 4px; }
```

Two files, three lines added/changed total. Desktop `.hsub` (base rule, `14px`) was not touched. No other selector, breakpoint, or file was touched.

---

## 2. Mobile before/after measurements

Measured on the live rendered page at a confirmed `innerWidth: 375`, same race (札幌1R, 14 horses), same rows sampled both times. (One tooling note: this session's Browser pane caches `style.css?v=20260811navy` aggressively since the URL doesn't change on edit — a plain reload kept serving the pre-edit file even though the server had the new content. Worked around by injecting a uniquely cache-busted stylesheet link before measuring "after," confirmed via `document.styleSheets` that only the fresh sheet was active.)

| Metric | Before (10px) | After (12px) | Delta |
|---|---|---|---|
| `.hsub` computed font-size | 10px | **12px** | as instructed |
| `.hsub` line-height | 15.5px | 18.6px | +3.1px |
| `.hsub` rendered height | 18.67px | 20.17px | +1.5px |
| Race-row rendered height (uniform, all 14 rows) | 53.05px | **54.55px** | +1.5px (exactly matches the hsub height delta — clean, predictable growth) |
| `.hname` rendered height | 19.38px | 19.38px | unchanged (not touched) |
| Row-to-row overlap | none | **none** | confirmed via bounding-rect gap check across all 14 rows |
| Rows with `.hsub` text truncated (ellipsis) | **0 of 14** | **9 of 14 (64%)** | **regression — see headline finding above** |
| Column boundaries / grid-template-columns | unaffected | unaffected | not touched by this edit; truncation happens *within* the existing fixed-width column, columns themselves don't shift |
| Mobile viewport width confirmed | 375px | 375px | via `window.innerWidth` |

### Confirmation against the four named criteria

| Criterion | Result |
|---|---|
| Text no longer falls below the 12px design-system floor | **PASS** — `.hsub` now measures exactly 12px, at the floor named in DESIGN.md §6 |
| No clipping | **FAIL** — 9/14 rows now truncate; 0/14 did before. Visually confirmed by screenshot (jockey names like "横山和生(替)" render as "横山和…") |
| No overlap | **PASS** — zero row-to-row overlap at any of the 14 rows |
| No column misalignment | **PASS** — column widths are set by `grid-template-columns`, untouched by this edit; truncation is content overflow within a fixed column, not a layout shift |
| No unacceptable density regression | **Judgment call, leaning PASS** — row height grew 1.5px (53.05→54.55px, +2.8%), imperceptible on its own. The truncation is the actual user-facing cost here, not density |

---

## 3. Desktop regression check

Confirmed **zero effect on desktop** — this edit only touches the `@media (max-width: 560px)` block:

| Metric | Desktop (1440px) |
|---|---|
| `.hsub` computed font-size | 14px (base rule, unchanged) |
| Rows with truncation | 0 of 14 |
| Row height (all 14 rows) | 65.3px, identical to every prior phase's measurement |

---

## 4. Mobile screenshot / browser observation

Screenshotted the shutsuba table at 375px after the change. Visually confirms the JS measurement precisely: row 1 (タガノシルフィー) shows jockey text truncated to "横山和…"; row 4 (テンレッドサン) "横山武…"; row 7 (タガノラヴズワン) "和田陽…"; row 9 (セントピュアベース) "荻野琢…"; row 10 (モンテパステル) "古川吉…". Rows without truncation (マーゴットクリュグ "松本大輝", ゴーゴーエイミー "斎藤新", サンセリテ "武豊(替)", ウォータースパウト "吉田隼人", ブライトエンジェル "川又賢治") render exactly as their full text, matching the per-row JSON data exactly. No layout breakage, no overlapping rows, marks/frame-colors/AI指数/勝率/市場 columns all intact and legible.

---

## 5. Numerical-value equality check

Spot-checked against every prior phase's screenshot of this same race: タガノシルフィー 7.6%/81, ◎エリカビアリッツ 21.1% — identical. Gauges, member-level, and class-stats numbers (checked earlier in this session before the edit) were not touched by either change and were not re-verified redundantly, since neither edit touches any data binding, computation, or JS — only two CSS/doc text properties. No console errors on a fresh load, desktop or mobile.

---

## 6. DESIGN.md section changed

**§13 Race Table** — added the row-height clarification paragraph immediately after the existing "40–44px row height" bullet, before "Horse name is primary text." The 40–44px figure itself was not changed; the new paragraph scopes it to single-line rows and explicitly licenses compound rows (like the race table's actual hname+hsub structure) to exceed it, matching what Phase 2D's measurement pass already found and classified KEEP.

---

## 7. `git diff --stat`

```
$ git diff --stat
 DESIGN.md           |  2 ++
 site/css/style.css  |  2 +-
 (+ the same 18 pre-existing unrelated files every prior phase has reported)
```

## 8. `git status --porcelain -- site/`

```
$ git status --porcelain -- site/
 M site/css/style.css
```

(`DESIGN.md` also shows modified via plain `git status`, outside the `-- site/` filter.) Neither committed, per instructions.

---

## What happens next is your call

Given the clipping finding in §2, before this gets committed I'd want your read on: keep 12px as instructed (truncation trade-off accepted), try an intermediate value (e.g. 11px — untested, would need its own measurement pass), or revert this one change and leave `.hsub` at 10px. I implemented exactly what was specified and measured the true result rather than assuming success; the DESIGN.md clarification (§1's item) has no such issue and needs no further decision.

---

## Final Outcome (accepted)

The 12px implementation described above **failed** its own "no clipping" acceptance criterion (9/14 rows, 64.3%, truncated) and was not accepted.

A follow-up measurement-only sweep (`docs/design_phase2d_hsub_sweep.md`) tested five candidate sizes — 10px, 10.5px, 11px, 11.5px, 12px — against the same race, the same 14 rows, the same mobile viewport, using the same fresh-navigation/cache-busted measurement method established in this report. **10px was the only value producing zero clipped rows** (0/14). Every value above it clipped at least 8 of 14 rows (57.1%), with no gradual transition — the jump from 0% to 57.1% clipping happens in the single step between 10px and 10.5px, because the binding constraint is a fixed-width column against one specific substring (the "(替)" jockey-change marker), not a gradual legibility curve.

**Final decision: keep mobile `.hsub` at 10px.** `site/css/style.css` was fully restored to this exact state after the sweep — confirmed via `git diff -- site/css/style.css` producing no output. **Production visual behavior is unchanged from before this entire `.hsub` investigation began**; nothing shipped differently to a real user at any point.

**DESIGN.md was refined based on this measured evidence**, not reverted without learning anything from the attempt: §6 Type Scale now carries a generic exception — deliberately not naming `.hsub` or `10px`, since the principle (dense width-constrained metadata may render below the nominal 12px floor when the alternative is silent truncation, provided the exception is justified by measurement against real content rather than assumption) is meant to outlive this specific component and this specific measured number.
