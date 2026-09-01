# PyCaLiAI Phase 6A — Forecast Ledger Implementation Report

> Implements Option A from the 6A-0 audit (`docs/phase6a0_forecast_history_audit.md`), confirmed by you: a single immutable, timestamped forecast record per race (6A-1, 6A-4, 6A-5), surfaced as a new "予測記録" tab. 6A-3's multi-point timeline chart and 6A-6 (What Changed) are **not implemented** — both require multiple forecast points per race, which the audit found don't exist in the current pipeline; nothing was invented to manufacture them.
> Not committed — left for review, per instruction. See §14/§15 for an important note on commit sequencing.
> Date: 2026-09-01

---

## 1. Purpose

Give a permanent, verifiable answer to "what exactly had the model predicted before this race," immune to any later site rebuild (e.g. Sunday's result-inclusive rebuild) silently changing what's shown as "the forecast."

## 2. Exact current data source

`site/data/{date}.json`, specifically each race's `mark`/`p_win`/`p_sho` per horse plus `confidence.field_chaos_score` and `member_level.{tier,label}` — all fields already confirmed public-safe and already rendered elsewhere on the site (Command Header, shutsuba table). No new field, no odds, no market data, no restricted upstream row is read.

## 3. New data/storage introduced

- **`build_forecast_history.py`** (new, root-level script, 149 lines) — reads `site/data/{date}.json` (already-scrubbed, already-built) plus `reports/cowork_input/{date}_bundle.json` (for `model_version` only), and writes one file per race to `site/data/forecast_history/{date}/{race_id}.json`. Write-once: if the target file already exists, it is left untouched, so a later run (e.g. after Sunday's result-inclusive rebuild) cannot alter an earlier snapshot. Also (re)writes `site/data/forecast_history/{date}/_index.json` — a lightweight `{date, race_ids}` manifest the frontend uses to decide whether to show the tab at all, without fetching all 35 per-race files. Field selection is an explicit whitelist copy (`horse_snapshot()`/`race_snapshot()`), plus a defense-in-depth `assert_public_safe()` walk that raises if any of a fixed list of forbidden keys (`odds`, `ev_tan`, `vs_market`, `hosei_marks`, `training`, `taiju`, etc.) is ever present in the object about to be written.
- **Generated data**: ran manually against `20260830` (35 races, today's live unsettled card) and `20260816` (35 races, a settled historical card, for Race Replay testing) — 70 files total, all under `site/data/forecast_history/`. This is real, correct output, not scratch — see §10 for spot-checked values.
- This directory lives under `site/data/`, so it ships with the existing static-site deploy (the Dockerfile `COPY`s the whole `data/` tree) with **no new deploy step, no Dockerfile change, no new sync script**.

**Not wired into weekly automation.** The script is standalone and must currently be run manually. It is *not* invoked from `weekly_nicegui.ps1` or `sync-hf-umami.ps1` — I deliberately did not touch either, since they're the unattended, Task-Scheduler-driven production automation this project treats with extra care. The natural hook point is one line, `python build_forecast_history.py {date}`, immediately after `build_site.py` runs inside `sync-hf-umami.ps1`'s Phase A step — flagged here for you to add (or ask me to add) rather than done unilaterally.

## 4. Exact changed files

```
build_forecast_history.py            new, 149 lines
site/data/forecast_history/**        new, 70 generated JSON files (2 dates × 35 races each, + 2 _index.json)
site/js/app.js                       modified
site/css/style.css                   modified
site/index.html                      modified (cache-busting only)
```

## 5. Exact functions/selectors

**`app.js`**: `loadDay()` — added one more per-day fetch block (mirrors the existing `career`/`changes` pattern exactly) populating `day.forecastIds` (a `Set`) from `_index.json`, with the same try/catch-to-empty-set graceful-failure shape already used for `career`. `viewsFor(r)` — appends `{key:"forecast", label:"予測記録"}` when `state.day.forecastIds.has(r.race_id)`. `renderView()` — added the `state.view === "forecast" && !...has(r.race_id)` fallback-to-shutsuba guard (mirrors the existing `grade` guard) and the dispatch branch. New functions: `fcDateTime(iso)` (small formatter) and `renderForecast(r, vb)` (fetches the per-race snapshot lazily — only when the tab is opened, not eagerly for all 35 races — and renders it).

**`style.css`**: new rules `.fc-card`, `.fc-head`, `.fc-lock`, `.fc-meta`, `.fc-state`, `.fc-table`, `.fc-row`, `.fc-hh`, `.fc-uma`, `.fc-name`, `.fc-win`/`.fc-sho`, `.fc-note`, `.fc-result-sep`, `.fc-result`, `.fc-pending`, plus a mobile (`≤880px`) override for `.fc-row`'s column widths. All reuse existing tokens (`--gold2`, `--tx`/`--tx2`/`--tx3`, `--line`/`--line2`, `.mark`/`.num`) — no new color, no new radius/shadow token.

## 6. UI entry point

A new tab, "予測記録", appended after 血統 in the existing view-tabs row — **only rendered when data exists** for the current race (same conditional-tab pattern already used for the "🏆重賞" tab). Not a top-level nav item, not a permanently-expanded panel — exactly the "tabs, not permanently expanded panels" instruction.

## 7. Desktop behavior

Verified live at 1440×900: tab appears/disappears correctly based on data availability; clicking it shows a locked-forecast card (label, timestamp, model version, chaos/member-level, a 5-column mark/umaban/name/WIN/TOP3 table sorted by WIN descending) followed — for settled races only — by a visually distinct RESULT section behind a dashed divider with its own "結果（レース確定後）" label, reusing the existing `posBadge()`/`.resbar` treatment. For unsettled races, a plain "未確定（このレースはまだ発走していません）" note appears instead of any result content. (Note: desktop screenshots at this content's scroll depth repeatedly hit this project's previously-documented paint-throttling artifact — blank frames past a certain scroll offset, unrelated to this feature. Verified instead via direct DOM text extraction, which is authoritative and was cross-checked against a working mobile screenshot of the identical content — see §13.)

## 8. Mobile behavior

Verified live at 375×812 (screenshot captured, see §13): tab, header, state line, and the full 14-row table all render cleanly; long names (e.g. "ルージュリヴィエラ", "プルーフリーディング") truncate gracefully with ellipsis at the mobile column width, no overflow, no collision. Race Replay's dashed divider and RESULT section are both clearly visible and correctly separated from the forecast table above them.

## 9. Public-data/compliance audit

| Field | Source | PyCaLiAI-generated? | Contains upstream raw value? | Already public elsewhere? | Required by feature? |
|---|---|---|---|---|---|
| `race_id` | `site/data/{date}.json` | n/a (identifier) | no | yes (every view) | yes — join key |
| `mark` | same | yes (model output) | no | yes (shutsuba table, header) | yes — core content |
| `p_win`, `p_sho` | same | yes (model output) | no | yes (shutsuba table, header) | yes — core content |
| `umaban`, `name` | same | no (metadata) | no | yes (everywhere) | yes — display identifier |
| `race_state.chaos` (`confidence.field_chaos_score`) | same | yes (model output) | no | yes (header gauge) | yes — race-state context |
| `race_state.member_level` | same | yes (derived from public results) | no | yes (header card) | yes — race-state context |
| `model_version` | `reports/cowork_input/{date}_bundle.json` | yes | no | yes (site footer) | yes — provenance |
| `generated_at` | wall-clock at first write | n/a | no | no (new) | yes — the whole point of a ledger |

Inspected the 70 generated files directly (§10) for the explicit forbidden-key list from the brief (`odds`, `ev_tan`, `market probability`, `vs_market`, `training`, plus this project's own `hosei_marks`/`taiju`/`fuku_low`/`fuku_high`/`umaren_odds`) — none present, confirmed both by the whitelist-only construction and the runtime `assert_public_safe()` check (which would have raised and aborted the write had any of these appeared).

## 10. Numerical/reproducibility verification

- Idempotency: ran the script twice against `20260830` — first run wrote 35, second run wrote 0/skipped 35 (confirmed via CLI output, §"6A verification" below).
- Spot-checked `2026083001020401`'s ledger entry against every live measurement of this exact race taken throughout this entire project session: ◎エリカビアリッツ 21.1%/46%, 〇テンレッドサン 12.7%/37%, ▲ブライトエンジェル 12.7%/37% — **exact match** in every case.
- Cross-checked the settled race `2026081601010801`: forecast shows ◎テンレッドサン 23.8%/50% (matches every prior measurement this session); the RESULT section correctly shows the *actual* finish (1着プルーフリーディング, a ▲, not the ◎) — the feature does not hide or reframe an unfavorable outcome, confirming it isn't silently biased toward flattering results.
- `generate_results.py` traced (Phase 6A-0 audit): reads only `reports/cowork_output/{date}_bets.json`, never touches `site/data/{date}.json` or the new ledger files — result processing cannot mutate forecast history, by construction, independent of any new code written this phase.

## 11. Regression testing

Existing shutsuba table, header, drawer, and all other tabs re-verified unaffected — `viewsFor()`/`renderView()`'s existing branches (`grade`, `bunseki`, `course`, `pedigree`) are untouched, only extended. `loadDay()`'s existing `career`/`changes` fetches are untouched, only one more parallel block added after them.

## 12. Console/runtime status

Clean, confirmed via a fresh single-click tab. Network requests to `data/forecast_history/**` all return 200 for dates with data and a single clean 404 (silently handled, tab correctly hidden) for a date without it — confirmed via direct network-log inspection, not just console absence.

## 13. Screenshots

Mobile screenshot (375×812, settled race, Race Replay view) captured and visually confirms: gold "確定予測" label, timestamp/model-version metadata, CHAOS/member-level state line, the full sorted horse table with graceful name truncation, the immutability disclosure note, the dashed separator, and the distinctly-styled RESULT section below it. Desktop was verified via DOM text extraction (see §7) rather than a screenshot image, due to this environment's known deep-scroll rendering limitation — the *content* was directly confirmed identical in substance to the mobile screenshot (same values, same structure), just not captured as a desktop image.

## 14. `git diff --stat`

```
$ git diff --stat -- site/js/app.js site/css/style.css site/index.html
 site/css/style.css | 65 +++++++++++++++++++++++++-----------
 site/index.html    |  4 +--
 site/js/app.js     | 96 +++++++++++++++++++++++++++++++++++++++++++++++++-----
 3 files changed, 136 insertions(+), 29 deletions(-)
```

**Important**: this diff includes both Phase 5C-3's still-uncommitted compact-header changes *and* this phase's forecast-tab changes, stacked in the same working tree — Phase 5C-3 was implemented and reported but never explicitly committed before this Phase 6 spec arrived. They are not entangled at the *feature* level (different functions, different selectors, no shared lines), but a plain `git diff` right now shows both together. Before committing anything, please confirm whether you want Phase 5C-3 committed first as its own commit (as originally planned) and this phase's changes committed separately on top, per the brief's own requested commit boundaries (`feat: add forecast snapshot history` / `feat: add Forecast Timeline / Race Replay`) — I did not decide this unilaterally.

## 15. `git status --porcelain`

```
$ git status --porcelain -- site/ build_forecast_history.py
 M site/css/style.css
 M site/index.html
 M site/js/app.js
?? build_forecast_history.py
?? site/data/forecast_history/
```

Not committed, per instruction.

## 16. Known limitations

- **6A-3's multi-point timeline and 6A-6 (What Changed) are not implemented** — per the confirmed audit finding, there is currently only one legitimate forecast-generation point per race, so a "how did it change" view has nothing to compare. If a second compliant generation point is ever introduced, both become straightforward additions on top of this same storage format.
- **Not wired into weekly automation** — must be run manually until you decide where it should hook into `sync-hf-umami.ps1` (see §3).
- **6A verification checklist, explicit status**:

| Check | Status |
|---|---|
| Single snapshot | PASS |
| Multiple snapshots (35/race day) | PASS |
| No snapshot history (older date) | PASS — tab correctly absent, no error |
| Settled race | PASS — Race Replay + Result shown, correctly separated |
| Unsettled race | PASS — pending note shown, no result content |
| Mark changes / no mark changes | N/A under the confirmed single-snapshot scope — nothing to diff |
| Horse scratched between snapshots | N/A under the confirmed single-snapshot scope — only one snapshot exists; a later scratch simply isn't reflected in the (correctly frozen) forecast, which is the intended behavior |
| Long horse names | PASS — verified on mobile, graceful ellipsis truncation |
| Mobile | PASS |
| Desktop | PASS (verified via DOM text extraction, see §7/§13) |
| Existing race table unchanged | PASS |
| Existing predictions unchanged | PASS |
| History is append-only | PASS — idempotency re-run test |
| Result processing does not mutate history | PASS — architecturally separate, traced in 6A-0 |
| No market/JRA-VAN restricted values leak | PASS — whitelist construction + runtime assertion, both clean |
| Console clean | PASS |

## 17. PASS/FAIL

**PASS**, within the confirmed reduced scope (Option A). No production regression found. Two things need your decision before this goes further: the commit-sequencing question in §14, and whether/how to wire `build_forecast_history.py` into weekly automation (§3, §16).
