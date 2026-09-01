# PyCaLiAI Phase 6A-0 — Forecast History Audit

> Audit only, per explicit instruction ("Before creating a new persistence format, trace the current prediction pipeline... Do not create duplicate history storage if reliable historical snapshots already exist"). No code written this phase.
> Date: 2026-09-01

---

## Headline finding

**The feature's core premise — that PyCaLiAI's own forecast changes at multiple points across race day (09:00 / 12:00 / 14:00 / FINAL) — does not correspond to how the pipeline actually works.** The model-derived prediction (mark, `p_win`, `p_sho`) for a given race is computed **exactly once per week**, on Saturday morning, and never recomputed before that race runs. The only system that executes repeatedly across race day (T-10, once per race, ~10 minutes before post) operates on **live JV-Link odds** specifically — the exact data category this feature is explicitly prohibited from touching. This is detailed below, then followed by a concrete recommendation and a question this audit cannot resolve unilaterally.

---

## What was traced

### 1. Weekly bundle generation (`export_weekly_marks.py` → `reports/cowork_input/{date}_bundle.json`)

Invoked once, manually/via `weekly_nicegui.ps1` Phase A, Saturday morning, from a single TARGET-exported CSV (`data/weekly/{date}.csv`). Confirmed live: the bundle's top-level object contains only `{date, model, race_count, races}` — **no `generated_at` or any timestamp field at all**, at the bundle level or per-race. Nothing in this script's structure implies or supports being re-run intraday for the same date — it takes a CSV snapshot of the week's card as input, and that card doesn't change during the day.

### 2. Existing JSON archives

- `reports/cowork_input/{date}_bundle.json` and its per-race split (`reports/cowork_input/{date}/{race_id}.json`) — both are direct products of the single Phase-A run above. The per-race files are a **split of the same single generation**, not independent snapshots.
- `reports/cowork_output/{date}_bets.json` — this is the one file genuinely touched multiple times per race day (once per race, by T-10's `compute_bets.py --apply` call). But it is **betting output, not forecast output**: it's built by combining the (static) bundle with `reports/live_odds/{race_id}.json` (live JV-Link odds), and per `t10_runner.py`'s own docstring, each race's entry is merged **in place** — the file is mutated, not appended to. It also carries `hosei_marks` — a display-only "T-10 corrected mark" that blends `log(p_win) + λ·log(市場π)` (confirmed in this project's own memory of the T-10 hosei-mark feature) — i.e., a mark that is partly **market-derived**. This file cannot be a source for a compliant public snapshot, on two independent grounds: it isn't immutable, and part of its content is odds-derived.
- `reports/wide_residual_shadow/{date}_shadow.json` — also explicitly compares `p_model` against `p_market_fair`; same problem.
- `reports/live_odds/{race_id}.json` — raw JV-Link odds, obviously out of scope entirely.

### 3. Previous deployment artifacts

`site/data/{date}.json` is committed to git once per race weekend (Phase A push) and I found no evidence of the same date's file being regenerated with different `p_win`/`mark` values later in the week — `data/{date}.json`'s marks are the same Phase-A bundle, re-shaped by `build_site.py`. `data/manifest.json` carries a single `built_at` timestamp for the **whole weekly site build** — one value, not per-race, not multiple-times-a-day.

### 4. Prediction output directories

Covered above (`reports/cowork_input/`, `reports/cowork_output/`, `reports/live_odds/`, `reports/wide_residual_shadow/`). None constitute a forecast-only, immutable, multi-point history.

### 5. Result-generation path

`generate_results.py` reads `reports/cowork_output/{date}_bets.json` (settled bets) and writes `data/results.json`/`data/cowork_results.json` — aggregate P&L only. It does **not** read or rewrite `data/{date}.json` or `reports/cowork_input/*_bundle.json` at all. Forecast and result are already architecturally separate files, written by separate scripts, at separate times. This is a genuinely useful existing invariant: **"a later result cannot rewrite an earlier forecast" is already true today, by construction, before any new code is written.**

### 6. Existing timestamps

None found at race granularity. The only timestamp in the whole pipeline relevant to "when was this generated" is `manifest.json`'s `built_at` — one value for the entire week's site build.

---

## Why this matters for 6A-1 through 6A-6

- **6A-3 (Timeline UI)** and **6A-6 (What Changed)** are specified around multiple forecast points across a single race day. With the model computed once per week, there is exactly **one** compliant data point per race — a sparkline or "09:00→12:00→14:00→FINAL" chart would either be empty or, if artificially populated by re-recording the same static bundle at several points in the day, would show a **perfectly flat line every time** (the underlying `p_win`/`mark` values would be byte-identical at every "snapshot," since nothing recomputes them). Shipping that would technically satisfy the UI spec while providing no real information — and worse, it would visually imply the model is being actively re-evaluated through the day when it is not, which cuts against this project's own established "honest disclosure" ethos (the same standard applied to HONEST RECORD, the compliance redactions, etc.).
- **6A-4 (Race Replay)** and **6A-5 (Final Forecast Lock)** do **not** depend on there being multiple points — a single, permanently preserved, timestamped forecast record already answers "what exactly had the model predicted beforehand," which is the stated purpose of the feature (question 5 in the Phase 6 goal list). This part of the spec is fully achievable with what exists today.

---

## Recommendation

Do not build a multi-point intraday timeline against data that doesn't have multiple points. Two honest paths forward, and this audit cannot pick between them alone:

**Option A — Scope 6A down to a "Forecast Ledger."** Capture the single existing forecast-generation moment (Phase A) per race as a permanent, immutable, timestamped record — smallest possible new storage, uses only already-compliant fields (mark, `p_win`, `p_sho`, `confidence`/chaos, `member_level`), and directly delivers Race Replay + Final Forecast Lock (6A-4, 6A-5) with full integrity guarantees. The Timeline UI (6A-3) becomes a single-point "this is the forecast, locked at generation time" display rather than a multi-dot chart; "What Changed" (6A-6) is dropped as inapplicable until a second legitimate generation event exists for the same race. This is a real, honestly-scoped feature — smaller than specified, but everything in it is true.

**Option B — Discuss introducing a genuine second (or third) compliant generation point** before building any storage — e.g., deciding whether re-running the mark/probability computation (not the odds-mixed betting layer) at a fixed point or two before race day would ever produce a materially different result given the model's inputs are pedigree/history/course-based rather than live-market-based, and if so, designing that as its own, separately-scoped pipeline change. This is meaningfully bigger than "the smallest append-only snapshot layer" the brief asks for, and isn't something to decide unilaterally.

I'd recommend Option A as the next step: it's the smallest, fully-compliant, immediately-buildable slice that still delivers real product value (a permanent, tamper-evident forecast record, verifiable after the fact), and it doesn't foreclose Option B later if a genuine second generation point is ever introduced.

No storage format, file, or code has been created in this phase.
