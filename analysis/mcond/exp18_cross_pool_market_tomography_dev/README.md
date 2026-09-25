# EXP18 — Cross-Pool Market Tomography

Status: **Stage 0 complete — stopped before Stage 1, awaiting Fable review**
(spec v0.4-frozen-stage0, commit `67b874ec`; the only change to frozen fields is
`stage0.power.practical_floor_nats` going from null to a number, commit `ee5b585a`)

The primary question is an **incremental residual test**, not direct replacement
of the UMAREN market. For year Y, the null freely fits the terminal market's
temperature using Y-1 and earlier. The nested alternative adds a separate cross-pool
coefficient. The Gate compares this alternative with the temperature-only null, so
market sharpness drift cannot be mistaken for cross-pool information.

T0 Harville and T1 Stern head-to-head comparisons are not novel: `crux_joint.py`
already evaluated the 9h version on 2024-2025 and found the market better
(3.343 vs 3.380). They are reproduction anchors only. Novel scope is limited to:

- calibrated offset residual information;
- T2 soft FUKUSHO constraints under a declared MaxEnt prior;
- historical-pre actionability, only after a practical terminal residual passes.

Archived TANPUK and UMAREN contain pool-total vote counts, not combination-level
counts. The failed new export is not required. Pool totals remain optional liquidity
metadata.

2024/2025 are already opened for the old T0/T1 head-to-head. They remain sealed only
for T2 and the new offset-residual hypothesis.

## Stage 0 results (2026-09-25)

No 2019-2023 finish, payout or realized top2 was read. The outcome loader asserts
`max(year) <= 2018`; the structure loader asserts no outcome columns; 2024/2025 rows
are dropped right after reading.

| Audit | Result | Where |
|---|---|---|
| S0-A prior art | **Not equivalent** (T0/T1 already done → anchor only; offset residual and T2 not done anywhere) | `PRIOR_ART_EQUIVALENCE_AUDIT.md` |
| S0-A anchor (9h, ≤2018) | **PASS**: λ*=1.1, Δ = LL(Harville) − LL(market) = **+0.0416** (market better; range 0.0185–0.074) | `out/anchor_le2018.json` |
| S0-B provenance | **PASS**: TANPUK/UMAREN 2013–2023, cp932, 区分 1/4, exact (rid16, 区分, 月日時分) join; 227-column unprefixed files not used | `MARKET_POOL_PROVENANCE.md`, `POOL_SCHEMA_MANIFEST.json` |
| S0-C exactness | **33/33 synthetic oracle / invariant tests PASS** | `out/invariant_tests.json` |
| S0-C FUKUSHO roundtrip | **FAIL** (horse-row pass rate 0.02–0.27% per year vs 99% required) → **T2 unimplemented, UB3 dropped**; Gate arm = UB2 only | `TOMOGRAPHY_IDENTIFIABILITY_AUDIT.md` §6 |
| S0-D coverage | **PASS**: 2019–2023 16,644 races, terminal / complete grid / pre+terminal all **1.000** (floors 0.95 / 0.99 / 0.90) | `RACE_AND_TICKET_COVERAGE.json` |
| S0-E practical floor | **0.8843 nats/race** (committed before any outcome performance) | `POWER_AUDIT.md`, `out/power_audit.json` |
| S0-E power at floor | **PASS**: SIGNAL power **1.000** (400/400, Wilson95 [0.990, 1.000]); also 1.000 at 0.5× and 2×. Median estimate at the floor is 0.742 (attenuation 0.84) | `POWER_AUDIT.md`, `out/power_audit.json` |
| S0-F compute | 1 year build 12 s, peak RSS 498 MB; P1/P2 200 draws ≈ 20–25 min each | `COMPUTE_DRY_RUN.json` |

**Review item for Fable (not changed here):** the declared noise convention
`SD(noise)=sqrt(2Δ)` makes the estimate about as far from the truth as the market is
(expected logloss gain of the estimate ≈ 8% of Δ, `out/power_diagnostics.json`), so
expected Kelly growth stays negative up to Δ≈0.88 nats. The resulting floor is ~3.5×
the full-investment reference 0.255 nats and ~21× the whole anchor gap between the
Harville T0 and the market (0.042). Under the frozen spec, PASS-PRACTICAL at Stage 1 is
therefore practically unattainable; any revision of the convention needs a new version.

`stage0_checks.py` re-verifies deliverables, frozen-field integrity, floor consistency,
test results and wording rules.

The specification is in `SPEC.md`; the machine-readable mirror is `spec.json`.

No Stage 1, outcome evaluation on 2019-2023, T2/offset 2024-2025 opening, ROI analysis,
ticket generation, stake optimization or production change has been run.
