# EXP21 — Equal-information odds-band market audit

Status: **Stage 1 complete (2026-09-27)** — tansho G1 FAIL (Spearman 0.49), umaren G1 FAIL (0.35),
fukusho G1 PASS (0.86) and G2 PASS (evaluation D1 uplift +0.051 over the all-ticket pool, ROI 0.784). No band in
any type, period or layer exceeds ROI 1.0; not a profit edge. See `STAGE1_REPORT.md`.

Frozen before outcomes: **v0.3-frozen (2026-09-27)**; Stage 1 limited to tansho, fukusho and umaren.
Tansho/umaren bands are compared with the label-free calibrated null (1/overround from terminal prices); fukusho
uses Spearman of band ROI (primary) and the pool ROI as baseline. 2026 OD is never called T-10. See `SPEC.md` §13
and `PROVENANCE.md`.

Previous status: **Stage 0 complete (2026-09-27); G1/G2 not started, no band ROI computed**

Stage 0 (see `STAGE0_DATA_AUDIT.md`): G0 PASS for tansho, fukusho, umaren (full terminal and ~T-28 coverage
2013-2023), wakuren, wide, umatan (2023 91-column identified as terminal; layout reverse-mapped against official
payouts) and sanrenpuku (2026 terminal days only, 179 races). **Sanrentan G0 FAIL (no price source).**
**Premise correction needed**: the 2026 OD 227-column files are not T-10 (5 terminal days, 36 pre-race exports,
4 without payouts), so D1 forward for the other five types is unavailable. G1 MDE (80% power): tansho ~10pt,
fukusho ~6pt, umaren ~12pt favorite-minus-longshot ROI spread; false pass at delta=0 <= 0.6%.

EXP21 is a model-independent description of favorite-longshot-bias shapes across eight
single-race JRA bet types. It does not search for a betting policy.

Formal G1/G2 replication is possible only for tansho, fukusho and umaren. Wakuren,
wide, umatan, sanrenpuku and sanrentan are limited to a 2023 terminal description and
a 2026 T-10 forward description because multi-year pre/terminal full-ticket prices are
not available.

Primary bins equalize market-implied expected-hit mass. Equal ticket-count and fixed
human-readable bands are secondary. Terminal diagnostic and decision-time analysis are
strictly separated.

- SPEC.md — frozen human-readable contract
- spec.json — frozen machine-readable contract
- FABLE_REVIEW_REQUEST.md — completed review request
- OPUS_IMPLEMENTATION_REQUEST.md — next handoff

No result aggregation, parser extension, 2024/2025 opening, production change, model
training, candidate generation, staking or ROI-policy change has been performed.
