# -*- coding: utf-8 -*-
"""
_trio_faith_top2_conc.py
========================
FAITHFUL classifier backtest of the user's 4-tier 三連複 formation system,
using the PRODUCTION race_confidence() classifier — specifically
CLASSIFIER B = top2_concentration (top1+top2 win prob) = the signal the
user actually keys on for 固さ (the 阪神12R "集中 0.92" example).

Tiering: highest-concentration 25% -> F4 (固い), ... lowest 25% -> F1 (大波乱).

Baselines:
  B0 = flat prob-first top-6 PL combos @¥1000, ALL races (established best odds-blind line)
  B2 = user system: classifier tier -> matched formation + per-tier stake
  B4 = REVERSED control: 固い->F1 (¥400 wide), 大波乱->F4 (¥2000 narrow)

p_win basis: BOTH raw PL and CALIBRATED (models/pl_calibrators_v6.pkl 'tansho')
are tried for the classifier; we report which we keyed the verdict on.

Metrics: blended (money-weighted) ROI%, capture%, per-race bootstrap 95% CI
(2000 draws, default_rng(12345)), per_tier breakdown, era split (2024/2025),
fixed-threshold (valid-derived cutoffs applied to test) for leak-safety.
Percentile transform is done per-split (mildly in-sample favorable) AND
valid->test (leak-safe). Both reported.
"""
from __future__ import annotations
import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, "E:/PyCaLiAI")
import pl_probs as PL
from export_marks_json import race_confidence
import joblib

RNG_SEED = 12345
SUB = "E:/PyCaLiAI/analysis/_trio_substrate.parquet"
CAL_PKL = "E:/PyCaLiAI/models/pl_calibrators_v6.pkl"

# ---- 4-tier formations over MODEL_RANK positions (A=1着,B=2着,C=3着 cand sets) ----
# A combo (3 bans) is bought iff its 3 model_ranks fill (A,B,C) under SOME permutation.
FORMATIONS = {
    "F1": dict(A={1, 2, 3},  B={1, 2, 3, 4},     C={1, 2, 3, 4, 5, 6, 7},       stake=400),   # 荒れ ~22pt
    "F2": dict(A={1, 2, 3},  B={1, 2, 3},        C={1, 2, 3, 4, 5, 6, 7, 8},    stake=600),   # 中荒れ ~16pt
    "F3": dict(A={1, 2},     B={1, 2, 3},        C={1, 2, 3, 4, 5, 6},          stake=1000),  # 本命来そう ~10pt
    "F4": dict(A={1},        B={2, 3},           C={2, 3, 4, 5},                stake=2000),  # 本命堅い ~5pt
}
# tier order from 固い(highest conc) to 大波乱(lowest conc)
TIER_BY_QUARTILE_FORWARD = {3: "F4", 2: "F3", 1: "F2", 0: "F1"}  # q=3 top conc -> F4
TIER_BY_QUARTILE_REVERSED = {3: "F1", 2: "F2", 1: "F3", 0: "F4"}  # reversed control


def combo_bought(ranks3, formation):
    """ranks3 = the 3 model_ranks of a combo. Bought iff fills (A,B,C) under some perm."""
    A, Bs, C = formation["A"], formation["B"], formation["C"]
    import itertools
    for perm in itertools.permutations(ranks3):
        if perm[0] in A and perm[1] in Bs and perm[2] in C:
            return True
    return False


def precompute_formation_combos():
    """For each formation, precompute the SET of model-rank-triples (sorted) that are bought.
    model_rank candidates only go up to 8 (max C in any formation)."""
    import itertools
    out = {}
    maxr = 8
    for name, fm in FORMATIONS.items():
        bought = set()
        cand = sorted(fm["A"] | fm["B"] | fm["C"])
        for trip in itertools.combinations(cand, 3):
            if combo_bought(trip, fm):
                bought.add(trip)  # trip is sorted ascending model_rank
        out[name] = bought
    return out


FORMATION_BOUGHT = precompute_formation_combos()


def build_race_table(df):
    """Group substrate by rid. For each race compute classifier signals (raw + calibrated)
    and the winning-trio model-rank-set, plus per-formation outcome (return per ¥100 staked,
    n combos bought)."""
    cal = joblib.load(CAL_PKL)
    cals = cal.get("calibrators", cal)
    tansho_cal = cals["tansho"]

    rows = []
    for rid, g in df.groupby("rid", sort=False):
        g = g.sort_values("ban").reset_index(drop=True)
        n = len(g)
        if n < 3:
            continue
        w = g["pl_w"].to_numpy(dtype=float)
        if not np.isfinite(w).all() or w.sum() <= 0:
            continue
        model_rank = g["model_rank"].to_numpy(dtype=int)
        finish = g["finish"].to_numpy()
        tri_pay = float(g["tri_pay"].iloc[0])
        split = g["split"].iloc[0]
        year = int(g["year"].iloc[0])
        pop_rank = g["pop_rank"].to_numpy(dtype=float)

        p_win_raw = PL.all_tansho(w)
        p_plc = PL.all_fukusho(w)
        # calibrated win prob then renormalize
        p_win_cal = tansho_cal.predict(p_win_raw)
        s = p_win_cal.sum()
        if s > 0:
            p_win_cal = p_win_cal / s

        ai_rank_order = model_rank  # 1=best, per-position
        # market rank by ban ascending ou_lastlog (1=most backed). skip if any NaN
        if np.isfinite(pop_rank).all():
            market_rank = pop_rank
        else:
            market_rank = None

        rc_raw = race_confidence(p_win_raw, p_plc, ai_rank_order, market_rank)
        rc_cal = race_confidence(p_win_cal, p_plc, ai_rank_order, market_rank)

        # winning trio: the 3 bans whose finish in {1,2,3}; map to their model_ranks
        win_mask = np.isin(finish, [1, 2, 3])
        win_ranks = tuple(sorted(int(r) for r in model_rank[win_mask]))
        has_valid_trio = (win_mask.sum() == 3)

        # B0: the 6 highest-PL-probability 三連複 combos (prob-first top-6), as rank-triples.
        # all_sanrenpuku keys are POSITION indices into w; map idx->model_rank.
        trio_probs = PL.all_sanrenpuku(w)  # {(i,j,k): prob}
        top6 = sorted(trio_probs.items(), key=lambda kv: -kv[1])[:6]
        b0_rank_set = set()
        for (i, j, k), _p in top6:
            rs = tuple(sorted((int(model_rank[i]), int(model_rank[j]), int(model_rank[k]))))
            b0_rank_set.add(rs)

        rows.append(dict(
            rid=rid, split=split, year=year, n=n, tri_pay=tri_pay,
            conc_raw=rc_raw["top2_concentration"],
            conc_cal=rc_cal["top2_concentration"],
            win_ranks=win_ranks, has_valid_trio=has_valid_trio,
            b0_rank_set=b0_rank_set,
        ))
    return pd.DataFrame(rows)


def formation_outcome(win_ranks, has_valid_trio, formation_name):
    """Return (n_combos_bought, hit:bool). hit iff winning-trio rank-set is in bought set.
    Per-race stake = n_combos * stake_per_combo. Return = hit * tri_pay * stake_per_combo/100."""
    bought = FORMATION_BOUGHT[formation_name]
    n_combos = len(bought)
    if not has_valid_trio:
        return n_combos, False
    hit = (win_ranks in bought)
    return n_combos, hit


def b0_outcome(win_ranks, has_valid_trio, b0_rank_set):
    """B0: flat prob-first top-6 PL combos @¥1000 (the 6 highest-PL-prob 三連複 combos).
    n_combos = 6 (may be <6 if field tiny). hit iff winning-trio rank-set in top-6."""
    n_combos = len(b0_rank_set)
    if not has_valid_trio:
        return n_combos, False
    return n_combos, (win_ranks in b0_rank_set)


def blended_roi(stakes, returns):
    s = np.sum(stakes)
    return 100.0 * np.sum(returns) / s if s > 0 else float("nan")


def bootstrap_ci(stakes, returns, n_boot=2000, seed=RNG_SEED):
    stakes = np.asarray(stakes, float)
    returns = np.asarray(returns, float)
    n = len(stakes)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    rois = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        s = stakes[idx].sum()
        rois[b] = 100.0 * returns[idx].sum() / s if s > 0 else np.nan
    lo, hi = np.nanpercentile(rois, [2.5, 97.5])
    return float(lo), float(hi)


def assign_tier(conc_series, tier_map):
    """Percentile-bucket into quartiles by concentration. Returns tier-name series.
    q=3 (top 25% conc) ... q=0 (bottom 25%)."""
    # rank-based quartile (handles ties); pd.qcut on rank for robustness
    ranks = conc_series.rank(method="first")
    q = pd.qcut(ranks, 4, labels=[0, 1, 2, 3]).astype(int)
    return q.map(tier_map)


def assign_tier_fixed(conc_values, cutoffs, tier_map):
    """Apply fixed cutoffs (3 thresholds = quartile boundaries from valid) to test conc."""
    # cutoffs = [c25, c50, c75]; q=0 if <c25, 1 if <c50, 2 if <c75, else 3
    q = np.digitize(conc_values, cutoffs)  # 0..3
    return pd.Series(q).map(tier_map).to_numpy()


def eval_system(rt, conc_col, tier_assigner, tier_map, label):
    """rt = race table. Returns dict with stakes/returns arrays per race + summary."""
    tiers = tier_assigner(rt[conc_col], tier_map) if callable(tier_assigner) else tier_assigner
    tiers = np.asarray(tiers)
    stakes = np.empty(len(rt))
    returns = np.empty(len(rt))
    hits = np.empty(len(rt), dtype=bool)
    for i, (_, r) in enumerate(rt.iterrows()):
        fm = tiers[i]
        n_combos, hit = formation_outcome(r["win_ranks"], r["has_valid_trio"], fm)
        stake_per = FORMATIONS[fm]["stake"]
        stakes[i] = n_combos * stake_per
        returns[i] = (r["tri_pay"] * stake_per / 100.0) if hit else 0.0
        hits[i] = hit
    return dict(tiers=tiers, stakes=stakes, returns=returns, hits=hits)


def summarize(stakes, returns, hits, n_races, label, n_boot=2000):
    roi = blended_roi(stakes, returns)
    lo, hi = bootstrap_ci(stakes, returns, n_boot=n_boot)
    cap = 100.0 * np.mean(hits) if len(hits) else float("nan")
    avg_stake = np.mean(stakes) if len(stakes) else float("nan")
    return dict(label=label, roi=roi, ci_low=lo, ci_high=hi, capture=cap,
                stake_per_race=avg_stake, n=int(n_races))


def main():
    df = pd.read_parquet(SUB)
    rt = build_race_table(df)
    n_dropped = df["rid"].nunique() - len(rt)
    print(f"[build] races in substrate={df['rid'].nunique()}  race-table={len(rt)}  dropped={n_dropped}", file=sys.stderr)
    # drops: only races with n<3, bad pl_w, or no valid trio recorded
    no_trio = (~rt["has_valid_trio"]).sum()
    print(f"[build] races without a valid 3-finisher trio (treated as no-hit): {no_trio}", file=sys.stderr)

    rt_test = rt[rt["split"] == "test"].reset_index(drop=True)
    rt_valid = rt[rt["split"] == "valid"].reset_index(drop=True)

    # choose classifier basis: try both, report. Use CALIBRATED as primary (system uses calibrated).
    results = {}
    for basis, col in [("raw", "conc_raw"), ("calibrated", "conc_cal")]:
        out = {}

        # ---- B0 flat prob-first top-6 (all races) ----
        def b0_eval(rtx):
            stakes = np.empty(len(rtx)); returns = np.empty(len(rtx)); hits = np.empty(len(rtx), bool)
            for i, (_, r) in enumerate(rtx.iterrows()):
                nc, hit = b0_outcome(r["win_ranks"], r["has_valid_trio"], r["b0_rank_set"])
                stakes[i] = nc * 1000
                returns[i] = (r["tri_pay"] * 1000 / 100.0) if hit else 0.0
                hits[i] = hit
            return stakes, returns, hits

        b0s_t, b0r_t, b0h_t = b0_eval(rt_test)
        b0s_v, b0r_v, b0h_v = b0_eval(rt_valid)
        out["B0_test"] = summarize(b0s_t, b0r_t, b0h_t, len(rt_test), "B0 test")
        out["B0_valid"] = summarize(b0s_v, b0r_v, b0h_v, len(rt_valid), "B0 valid")

        # ---- B2 forward tiering (per-split percentile) ----
        b2_t = eval_system(rt_test, col, assign_tier, TIER_BY_QUARTILE_FORWARD, "B2 test")
        b2_v = eval_system(rt_valid, col, assign_tier, TIER_BY_QUARTILE_FORWARD, "B2 valid")
        out["B2_test"] = summarize(b2_t["stakes"], b2_t["returns"], b2_t["hits"], len(rt_test), "B2 test (per-split pctile)")
        out["B2_valid"] = summarize(b2_v["stakes"], b2_v["returns"], b2_v["hits"], len(rt_valid), "B2 valid (per-split pctile)")

        # ---- B4 reversed control (per-split percentile) ----
        b4_t = eval_system(rt_test, col, assign_tier, TIER_BY_QUARTILE_REVERSED, "B4 test")
        out["B4_test"] = summarize(b4_t["stakes"], b4_t["returns"], b4_t["hits"], len(rt_test), "B4 reversed test")

        # ---- per_tier for B2 (test) ----
        per_tier = {}
        for fm in ["F1", "F2", "F3", "F4"]:
            mask = (b2_t["tiers"] == fm)
            st = b2_t["stakes"][mask]; rr = b2_t["returns"][mask]; hh = b2_t["hits"][mask]
            if mask.sum() == 0:
                continue
            roi = blended_roi(st, rr)
            lo, hi = bootstrap_ci(st, rr)
            cap = 100.0 * np.mean(hh)
            avg_pts = np.mean(st / FORMATIONS[fm]["stake"])  # avg n_combos
            per_tier[fm] = dict(n=int(mask.sum()), roi=roi, ci_low=lo, ci_high=hi,
                                capture=cap, stake_per_combo=FORMATIONS[fm]["stake"],
                                avg_points=float(avg_pts))
        out["per_tier_test"] = per_tier

        # ---- fixed-threshold: valid-derived cutoffs applied to test ----
        # quartile boundaries of valid concentration
        cutoffs = np.percentile(rt_valid[col].to_numpy(), [25, 50, 75])
        fixed_tiers = assign_tier_fixed(rt_test[col].to_numpy(), cutoffs, TIER_BY_QUARTILE_FORWARD)
        b2_fixed = eval_system(rt_test, col, fixed_tiers, None, "B2 fixed-threshold test")
        out["B2_fixed_test"] = summarize(b2_fixed["stakes"], b2_fixed["returns"], b2_fixed["hits"],
                                         len(rt_test), "B2 fixed-threshold (valid cutoffs) test")
        out["valid_cutoffs"] = [float(c) for c in cutoffs]

        # ---- era split for B2 forward per-split (2024 / 2025) ----
        era = {}
        for yr in [2024, 2025]:
            m = (rt_test["year"] == yr).to_numpy()
            if m.sum() == 0:
                continue
            era[yr] = blended_roi(b2_t["stakes"][m], b2_t["returns"][m])
        out["B2_era_test"] = era

        results[basis] = out

    # ---- print full summary ----
    def pr(d):
        return f"ROI={d['roi']:.1f}% CI[{d['ci_low']:.1f},{d['ci_high']:.1f}] cap={d['capture']:.1f}% stk/R=¥{d['stake_per_race']:.0f} n={d['n']}"

    print("\n================ FAITHFUL CLASSIFIER B = top2_concentration ================")
    for basis in ["raw", "calibrated"]:
        o = results[basis]
        print(f"\n--- p_win basis: {basis} ---")
        print("  B0 flat top-6 @¥1000     test :", pr(o["B0_test"]))
        print("  B0 flat top-6 @¥1000     valid:", pr(o["B0_valid"]))
        print("  B2 tiered (per-split)    test :", pr(o["B2_test"]))
        print("  B2 tiered (per-split)    valid:", pr(o["B2_valid"]))
        print("  B2 tiered (fixed valid)  test :", pr(o["B2_fixed_test"]))
        print("  B4 REVERSED control      test :", pr(o["B4_test"]))
        print(f"  vs B0 delta (test): {o['B2_test']['roi']-o['B0_test']['roi']:+.1f}pt")
        print(f"  vs B4 delta (test): {o['B2_test']['roi']-o['B4_test']['roi']:+.1f}pt")
        print(f"  era 2024/2025: {o['B2_era_test']}")
        print("  per_tier (test):")
        for fm in ["F4", "F3", "F2", "F1"]:
            if fm in o["per_tier_test"]:
                t = o["per_tier_test"][fm]
                print(f"    {fm}: ROI={t['roi']:.1f}% CI[{t['ci_low']:.1f},{t['ci_high']:.1f}] cap={t['capture']:.1f}% "
                      f"pts~{t['avg_points']:.1f} @¥{t['stake_per_combo']} n={t['n']}")

    # dump json for the calling agent
    with open("E:/PyCaLiAI/analysis/_trio_faith_top2_conc_result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=float)
    print("\n[done] wrote _trio_faith_top2_conc_result.json", file=sys.stderr)


if __name__ == "__main__":
    main()
