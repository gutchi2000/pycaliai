# Open-source scope and reproducibility

PyCaLiAI's open-source value is the engineering around a real, continuously
operated ranking system: time-aware validation, leakage audits, probability
calibration, fail-closed production gates, forward shadow evaluation, and a
static publication layer. It is not a bundled redistribution of commercial
race databases.

## Included under MIT

- Original Python and PowerShell source code
- Original tests, schemas, and architecture documentation
- Static-site source under `site/`
- Configuration examples that contain no secrets or restricted data

See `NOTICE.md` for exclusions and third-party rights.

## Inputs users must provide

The full operational pipeline expects race cards, historical results, training
records, and market snapshots. The maintainer currently obtains some inputs
through separately licensed services. Contributors must use data they are
authorized to access and must not submit raw third-party exports.

The project should progressively add small synthetic fixtures so core parsing,
ranking, calibration, settlement, and guardrail logic can be tested without
commercial data. Until that work is complete, the full production run is not a
one-command public reproduction.

## Reproducibility contract

A research result should record:

1. the prediction timestamp and the information available at that time;
2. train, validation, and test date boundaries;
3. feature provenance and missing-value behavior;
4. model and calibrator versions;
5. evaluation metrics, uncertainty, and rejected hypotheses;
6. whether the result is retrospective, shadow, or live-forward evidence.

Features derived from outcomes must be computed as-of or out-of-fold. Static
aggregates that include the evaluated race are treated as leakage and rejected.

## Responsible use

Predictions are probabilistic and may be wrong. PyCaLiAI is research software,
not financial advice and not a promise of profit. The production workflow keeps
bet placement under human control and uses validation gates that fail closed.
Users must comply with local law, provider terms, and age restrictions.
