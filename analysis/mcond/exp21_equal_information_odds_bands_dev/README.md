# EXP21 — Equal-information odds-band market audit

Status: **v0.2-frozen after Fable review; implementation not started**

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
