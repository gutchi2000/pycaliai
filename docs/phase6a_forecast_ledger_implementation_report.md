# PyCaLiAI Phase 6A — Forecast Ledger Implementation Report

> Implements Option A from the 6A-0 audit (`docs/phase6a0_forecast_history_audit.md`), confirmed by you: a single immutable, timestamped forecast record per race (6A-1, 6A-4, 6A-5), surfaced as a new "予測記録" tab. 6A-3's multi-point timeline chart and 6A-6 (What Changed) are **not implemented** — both require multiple forecast points per race, which the audit found don't exist in the current pipeline; nothing was invented to manufacture them.
> **Superseded in part by a provenance rework — see §18.** The single-`generated_at`-field design described in §3/§9 below was replaced before commit with an explicit `provenance`/`captured_at`/`source_generated_at`/`source_generated_at_basis` schema, after review found the original design couldn't distinguish a genuine pre-race capture from a same-day-or-later backfill. §14/§15's commit-sequencing question was resolved: committed as an isolated `design: Phase 6A Forecast Ledger` commit, after Phase 5C-3 and before Phase 6B, per your explicit ordering.
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
| `provenance`, `captured_at`, `source_generated_at`, `source_generated_at_basis` | see §18 (provenance rework) — superseded the single `generated_at` field originally listed here | n/a | no | no (new) | yes — the whole point of a ledger |

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

## 14. `git diff --stat` / commit sequencing — resolved (see §18)

The commit-sequencing question originally raised here (Phase 5C-3's header changes and this phase's forecast-tab changes were stacked, uncommitted, in the same working tree) was resolved exactly as you instructed: the three phases were split into isolated commits via hunk-level reconstruction (verified byte-for-byte against the final working tree before each commit), in the order 5C-3 → 6A → 6B. This phase's isolated diff, as actually committed:

```
$ git diff --stat HEAD~2..HEAD~1 -- site/js/app.js site/css/style.css site/index.html build_forecast_history.py
 build_forecast_history.py |  91 (new file)
 site/css/style.css        |  27 +++++++++++++++
 site/index.html           |   4 +-
 site/js/app.js            |  91 +++++++++++++++++++++++++++++++++++++++++++
 4 files changed, insertions only except index.html's cache-bust bump
```

## 15. `git status --porcelain`

Committed as `design: Phase 6A Forecast Ledger (append-only forecast history)`. See §18 for the full commit list and verification.

## 16. Known limitations

- **6A-3's multi-point timeline and 6A-6 (What Changed) are not implemented** — per the confirmed audit finding, there is currently only one legitimate forecast-generation point per race, so a "how did it change" view has nothing to compare. If a second compliant generation point is ever introduced, both become straightforward additions on top of this same storage format.
- **Not wired into weekly automation** — must be run manually. A specific hook location (inside `sync-hf-umami.ps1`, immediately after its existing `build_site.py` call, gated to only pass `--live` from Phase A's own invocation) has been proposed and reported separately for your approval; not implemented yet (see §18).
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

**PASS**, within the confirmed reduced scope (Option A). No production regression found. Two things needed your decision before this went further: the commit-sequencing question in §14, and whether/how to wire `build_forecast_history.py` into weekly automation (§3, §16) — both addressed in §18.

## 18. Addendum — provenance rework (post-review correction)

Your review of the first pass found the single-`generated_at`-field design insufficient: it recorded only *when this script wrote the file*, with no way to tell "captured before the race" apart from "backfilled today by reading an old archived file." Fixed before any commit, by replacing that one field with four:

- **`provenance`**: `"live_generation"` (this run was explicitly invoked with `--live` as part of a genuine same-day Phase-A run) or `"archived_bundle_backfill"` (the default — read an already-existing `site/data/{date}.json` at some later point).
- **`captured_at`**: wall-clock time this script wrote the record. Never claimed to be the original forecast time.
- **`source_generated_at`**: a best-effort estimate of when the forecast content actually first went public, or `null` if it can't be proven — never guessed.
- **`source_generated_at_basis`**: how `source_generated_at` was derived. `"git_first_commit_content_verified"` — found `site/data/{date}.json`'s earliest git commit via `git log --follow`, then confirmed (via `git show <hash>:<path>`) that every race's `mark`/`p_win`/`p_sho` in that commit is byte-identical to current content, i.e. nothing has changed since — making that commit's timestamp a trustworthy proxy. `"content_diverged_from_earliest_known_commit"` or `"no_git_history"` otherwise, with `source_generated_at` left `null`.

The frontend (`renderForecast()` in `app.js`) surfaces this directly rather than only in metadata: a badge ("事後記録（アーカイブから復元）" for backfill / "レース当日リアルタイム記録" for live), a "予測確定 …（記録改ざん検証済）" line when the git-verified basis applies, a separate "台帳記録 …" line for `captured_at`, and a disclosure note that explicitly states a backfilled record is not proof the ledger was locked pre-race.

**Full re-audit of every currently-generated ledger file** (both dates, 70 race files + 2 `_index.json`): all 70 classify as `provenance=archived_bundle_backfill`, `source_generated_at_basis=git_first_commit_content_verified` — verified programmatically (not sampled) by loading every file and tallying `(provenance, source_generated_at_basis)`. `20260830`'s `source_generated_at` = `2026-08-29T22:49:01+09:00`; `20260816`'s = `2026-08-15T23:16:07+09:00` (confirmed to be the *earliest* of several commits touching that date's file, not a later refresh). No `live_generation` records exist yet — expected, since `--live` has never actually been invoked from automation (it isn't wired in yet, see below).

Live-verified on both desktop (1440×900) and mobile (375×812) against `20260816`: badge, predicted-at line, captured-at line, and disclosure note all render exactly as designed, with a clean console (the one 404 seen — `data/changes_20260816.json` — is this project's pre-existing, unrelated day-of-changes overlay correctly finding nothing for a historical date, not a regression).

**Automation hook — proposed, not implemented** (per your instruction to report before touching unattended automation): `build_forecast_history.py`'s true dependency is `site/data/{date}.json`, which `sync-hf-umami.ps1` produces via its own `build_site.py` call (line ~54) — `weekly_post.ps1` never touches `site/data/`. Proposed change: add a `-Date` and `-Live` param to `sync-hf-umami.ps1`; call `build_forecast_history.py $Date $(if($Live){'--live'})` immediately after its `build_site.py` step, before any commit/deploy step. `weekly_nicegui.ps1`'s Phase A block would pass `-Date $Date -Live`; BetsOnly/Post would pass `-Date $Date` only (no `-Live`) — so if Phase A's capture is ever missed, a later phase's call still captures the race, but honestly as a backfill rather than mislabeling it live. Not yet implemented — awaiting your review alongside the Phase 6B hook proposal.

**Commit**: landed as its own isolated commit, `design: Phase 6A Forecast Ledger (append-only forecast history)`, positioned after Phase 5C-3 and before Phase 6B per your explicit ordering — reconstructed via hunk-level surgery on a scratch copy and verified byte-for-byte against the final working tree before commit, not a squash of the stacked working-tree diff.
