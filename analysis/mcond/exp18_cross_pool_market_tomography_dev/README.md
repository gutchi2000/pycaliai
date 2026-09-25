# EXP18 — Cross-Pool Market Tomography

Status: **v0.1 draft / independent review required / not frozen**

This experiment asks whether different JRA betting pools imply mutually inconsistent
probability distributions over the same race outcome.  It does not begin with ROI,
ticket selection, stake allocation, or a new horse model.

Primary initial target: **UMAREN (unordered top-two pair)**.  The target UMAREN pool is
withheld from the tomography input.  TANSHO/FUKUSHO and any other pool that passes the
Stage 0 provenance audit are used to construct a coherent latent ranking distribution.
The resulting unordered-pair probabilities are compared with the terminal UMAREN market
using race-level categorical logloss.

The archived TANPUK and UMAREN files contain pool-total vote-count columns, but no
combination-level vote counts.  Pool totals are therefore optional liquidity metadata;
they are not treated as exact combination-level flow.  The failed new export attempt is
recorded as unavailable and is not a prerequisite.

The draft specification is in `SPEC.md`; the machine-readable mirror is `spec.json`.

No implementation, outcome evaluation, 2024/2025 opening, ROI analysis, ticket generation,
or production change is authorized by this draft.
