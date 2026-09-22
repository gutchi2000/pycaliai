# Codex for Open Source — application draft

This document is a working draft. Replace bracketed fields with exact facts
shown by the application form before submission.

## Project

- Repository: https://github.com/gutchi2000/pycaliai
- Project: PyCaLiAI
- Role: founder and primary maintainer
- Public application: https://pycaliai.com
- License: MIT for original source code and documentation; third-party data is
  explicitly excluded in `NOTICE.md`.

## Short project description

PyCaLiAI is an actively maintained, leak-aware machine-learning research and
operations system for Japanese horse racing. It combines learning-to-rank,
Plackett–Luce probabilities, calibration, time-based validation, forward shadow
evaluation, fail-closed safety gates, reproducible research records, and a
public static results interface. The reusable contribution is the methodology
and operational tooling for evaluating predictive systems without temporal
leakage or silent training/serving skew.

## Why it matters

Many predictive examples stop at an offline score. PyCaLiAI documents the less
visible work required to operate a probabilistic model responsibly: strict
as-of feature semantics, chronological holdouts, calibration monitoring,
training/serving parity checks, rejected-experiment records, pre-registered
promotion gates, forward-only shadow trials, and fail-closed publishing and
bet-construction validation.

The repository is also a concrete Japanese-language reference implementation
for maintainers and independent researchers working on ranking, calibration,
decision systems, and reproducible evaluation under market feedback.

## Evidence of active maintenance

- Maintained continuously since March 2026.
- More than 1,300 maintainer commits as of 2026-09-22.
- A live weekly production workflow and public results site.
- Active issue prevention through tests, audit documents, model version
  ledgers, rollback artifacts, and explicit rejection of leaked experiments.
- Ongoing release, deployment, documentation, and data-compliance work by the
  primary maintainer.

Before submission, update these facts from GitHub and add any public usage,
external contributor, citation, discussion, or download evidence available at
that time. Do not overstate adoption.

## How Codex would be used

1. Review pull requests for temporal leakage, unsafe defaults, and
   training/serving skew.
2. Triage issues and turn production incidents into minimal regression tests.
3. Maintain architecture, schema, release, and experiment-ledger documentation.
4. Generate and validate synthetic fixtures so more of the project can be
   reproduced without redistributing licensed data.
5. Audit dependencies and harden secret, data-provenance, and publication
   boundaries.
6. Automate release notes and consistency checks across code, tests, and docs.

## Honest limitations

The project is currently maintained primarily by one person and has limited
public adoption signals. Some full-pipeline inputs require separately licensed
data and Windows-specific integrations. The near-term open-source roadmap is to
separate the reusable core from private operational data, add synthetic test
fixtures, improve packaging, publish versioned releases, and make contribution
paths clearer.

## Suggested form answer: why should this project be selected?

PyCaLiAI is a small but unusually active real-world ML maintenance project. Its
value is not the claim that a racing model predicts perfectly; it is the public
engineering record of how to prevent leakage, measure calibration, detect
training/serving skew, reject weak ideas, validate decisions forward in time,
and fail closed when production evidence is incomplete. Codex would directly
reduce the review and documentation load on a solo maintainer while helping
turn a working private-data pipeline into a cleaner, reproducible open-source
core with synthetic fixtures and stronger security boundaries.
