# EXP21 — Equal-information odds-band audit

Status: **draft for Fable review; no aggregation has been run**

This experiment audits whether the apparently favorable return bands shown in the
AI Keiba Masters 2026 winner's presentation survive finer, statistically comparable
odds bands across all standard single-race JRA bet types.

The primary table does **not** use fixed-width odds bands.  It allocates equal market-
implied expected-hit mass to every band.  Two secondary tables use equal ticket counts
and human-readable fixed bands (`1.0-5.0`, `5.1-15.0`, ...).

The first gate is data provenance and parser correctness.  No ROI table may be produced
until every ticket type passes its own price/payout contract.  Diagnostic terminal-price
tables and actionable decision-time tables must never be mixed.

Files:

- `SPEC.md` — frozen-design candidate
- `spec.json` — machine-readable contract
- `FABLE_REVIEW_REQUEST.md` — reviewer handoff

No production, model, staking, candidate-generation, or 2024/2025 holdout change is in
scope.
