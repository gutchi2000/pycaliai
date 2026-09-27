# EXP21 — Equal-information odds-band market audit

Final status: **closed after Stage 1 (2026-09-27, approved in Fable final review)**

> 単勝・馬連の等期待的中質量10帯では、最大穴の1帯だけに較正null比−0.13〜−0.27の崖が再現し、残る9帯はおおむね±0.03前後で平坦だった（例外は馬連の帯1 +0.046〜+0.064 と、期間で位置が変わる単勝の帯8または帯9 +0.068〜+0.069。D0）。複勝は帯順位が再現したが、最大穴帯を除いたROIも0.784で、法定参考値0.80と損益分岐1.0の双方に届かなかった。全券種・期間・価格層でROI 1.0超の帯は存在しなかった。
>
> 射程外: 枠連・ワイド・馬単はD0記述のみ、三連複は標本不足、三連単は価格源なし、T−10・2024/2025は未検証。

Stage 1 summary: **Stage 1 complete (2026-09-27)** — tansho G1 FAIL (Spearman 0.49), umaren G1 FAIL (0.35),
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
**Premise correction (resolved in v0.3, `SPEC.md` §13)**: the 2026 OD 227-column files are not T-10 (5 terminal
days, 36 pre-race exports with undefined timing, 4 without payouts; T-10 acquired 0 days). The v0.2 "D1 forward"
layer is superseded; the only valid D1 is the historical ~T-28 snapshot. G1 MDE (80% power): tansho ~10pt,
fukusho ~6pt, umaren ~12pt favorite-minus-longshot ROI spread; false pass at delta=0 <= 0.6%.

EXP21 is a model-independent description of favorite-longshot-bias shapes across eight
single-race JRA bet types. It does not search for a betting policy.

Formal G1/G2 replication is possible only for tansho, fukusho and umaren. Wakuren,
wide and umatan are limited to a D0 description (2023 terminal and 2026 terminal 5 days);
sanrenpuku is insufficient and sanrentan has no price source. (The v0.2 "2026 T-10 forward
description" was withdrawn in v0.3: 2026 OD is not T-10.)

Primary bins equalize market-implied expected-hit mass. Equal ticket-count and fixed
human-readable bands are secondary. Terminal diagnostic and decision-time analysis are
strictly separated.

- STAGE1_REPORT.md — Stage 1 report and final conclusion (latest)
- STAGE1_TABLES.md, out/stage1_results.json — full Stage 1 tables
- SPEC.md (§13 = v0.3-frozen) / spec.json (`v03`, `stage1_results`, `final_status`) — contract; §13 / `v03` supersede conflicting v0.2 text
- STAGE0_DATA_AUDIT.md, WINNER_CLAIM_UNCERTAINTY.md — Stage 0 audit
- PROVENANCE.md — worktree/branch provenance and the `aec789bb` record
- FABLE_REVIEW_REQUEST.md, OPUS_IMPLEMENTATION_REQUEST.md — historical (Stage 0 handoff, completed and superseded by v0.3)

Outcomes opened: 2023 payouts in Stage 0 for price-layout identification only; 2013-2023 finishes/payouts for
tansho, fukusho and umaren in Stage 1 (commit `7cd7d452`). 2024/2025 remain sealed. No parser extension,
production change, model training, candidate generation, staking or ROI-policy change has been performed.
