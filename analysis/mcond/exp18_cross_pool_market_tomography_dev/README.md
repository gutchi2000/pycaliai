# EXP18 — Cross-Pool Market Tomography

Status: **v0.3 draft / second Fable review incorporated / difference re-review required / not frozen**

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

The specification is in `SPEC.md`; the machine-readable mirror is `spec.json`.

No implementation, outcome evaluation, T2/offset 2024-2025 opening, ROI analysis,
ticket generation, or production change is authorized by this draft.
