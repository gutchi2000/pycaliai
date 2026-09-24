# EXP17 — Dynamic Common-Opponent Hodge–Plackett–Luce

## Status

**Stage 0 specification draft / not frozen / no implementation or evaluation run.**

This experiment tests whether two horses that have never met can be compared through
their shared historical opponents, and whether that pair-specific information adds
anything beyond the existing dynamic ratings, the clean tabular model, and the
terminal close win market.

The naive version of the idea is not new in this repository. EXP02 already estimated
a time-varying scalar horse ability with a Plackett–Luce update. A single scalar
ability automatically creates transitive comparisons through common opponents.
EXP12 also tested one-hop opponent-identity summaries and failed its degree/experience
preserving placebo. EXP17 therefore does **not** repeat either experiment.

The new hypothesis is narrower:

> A target pair may have path-specific evidence through particular common opponents
> that is not recoverable from one scalar rating per horse. Preserve that pairwise
> evidence until the final race-level projection instead of averaging it into each
> horse's row features.

The primary representation is a time-safe two-hop common-opponent matrix. For every
unordered pair in the current field, evidence from shared opponents is aggregated into
an antisymmetric comparison value. Weighted Hodge projection converts that matrix into
one coherent score per horse, and a Plackett–Luce softmax converts the scores into a
race probability distribution.

## Files

- `SPEC.md` — human-readable frozen-design candidate
- `spec.json` — machine-readable copy of the core protocol

## Hard boundaries

- No 2024/2025 result evaluation.
- No ROI, ticket generation, staking, or production connection.
- No PageRank, embedding, GNN, or paths longer than two hops in the primary experiment.
- No claim that all graph methods failed if this exact two-hop hypothesis fails.
- No claim of novelty if Stage 0 shows mathematical or empirical equivalence to EXP02.

## Planned decision sequence

1. Stage 0: prior-art equivalence, data/time-safety, graph coverage, and power audit.
2. Stage 1: synthetic invariants and exact as-of graph construction.
3. Stage 2: direct mechanism test on previously-unmet pairs.
4. Stage 3: terminal-close residual test, only if Stage 2 passes.
5. Stage 4: historical-pre-snapshot diagnostic, only as pre-registered.
6. Economic evaluation is allowed only after a practical terminal-close residual pass.

The specification remains a draft until independently reviewed and committed as a
numbered frozen version.
