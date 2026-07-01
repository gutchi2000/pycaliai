"""
_trio_faith_composite.py
========================
Faithful test of the USER's 4-tier 三連複 formation system, classified by the
SYSTEM'S ACTUAL race_confidence gauges (top1_dominance, top2_concentration,
field_chaos_score, ai_market_agreement) combined into a COMPOSITE 固さ score.

Substrate: analysis/_trio_substrate.parquet (1 row/horse/race).
Classifier: production export_marks_json.race_confidence + pl_probs.
Calibrated p_win option: models/pl_calibrators_v6.pkl 'tansho' isotonic.

Metrics: blended (money-weighted) ROI%, capture%, per-race bootstrap 95% CI.
Verdict split = test(2024-25); valid(2023) = agreement.
"""
from __future__ import annotations
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, "E:/PyCaLiAI")
import pl_probs
from export_marks_json import race_confidence
import joblib

RNG = np.random.default_rng(12345)
SUB = "E:/PyCaLiAI/analysis/_trio_substrate.parquet"

# ---- USER's 4-tier formations over model_rank positions (1=best) -----------
# A=1着cand, B=2着cand, C=3着cand rank-sets. combo {x,y,z} BOUGHT iff its 3
# model_ranks fill (A,B,C) under some permutation. 三連複 order-free.
FORMATIONS = {
    "F1": dict(A={1, 2, 3},    B={1, 2, 3, 4},    C={1, 2, 3, 4, 5, 6, 7},    stake=400),
    "F2": dict(A={1, 2, 3},    B={1, 2, 3},       C={1, 2, 3, 4, 5, 6, 7, 8}, stake=600),
    "F3": dict(A={1, 2},       B={1, 2, 3},       C={1, 2, 3, 4, 5, 6},       stake=1000),
    "F4": dict(A={1},          B={2, 3},          C={2, 3, 4, 5},             stake=2000),
}
# tier -> formation:  固い(high composite)->F4 narrow/thick; 大波乱(low)->F1 wide/thin
TIER_TO_FORM_SYSTEM = {"Q1_low": "F1", "Q2": "F2", "Q3": "F3", "Q4_high": "F4"}
# reversed control:  固い->F1, 大波乱->F4
TIER_TO_FORM_REV = {"Q1_low": "F4", "Q2": "F3", "Q3": "F2", "Q4_high": "F1"}


def formation_combos(form, n_horses):
    """Return set of frozensets of model_ranks bought under (A,B,C) with the
    standard 三連複 formation semantics: pick one rank from A, one from B, one
    from C, all distinct -> the 3-rank SET is bought (order-free)."""
    A = {r for r in form["A"] if r <= n_horses}
    B = {r for r in form["B"] if r <= n_horses}
    C = {r for r in form["C"] if r <= n_horses}
    combos = set()
    for a in A:
        for b in B:
            for c in C:
                s = frozenset((a, b, c))
                if len(s) == 3:
                    combos.add(s)
    return combos


def load():
    df = pd.read_parquet(SUB)
    cal = joblib.load("E:/PyCaLiAI/models/pl_calibrators_v6.pkl")
    tansho_cal = cal["calibrators"]["tansho"]
    return df, tansho_cal


def per_race_signals(df, tansho_cal):
    """For each race compute the 4 race_confidence gauges (raw p_win and
    calibrated p_win), plus the winning trio (set of model_ranks) and tri_pay.
    Returns DataFrame indexed by rid."""
    rows = []
    drops = {"lt3_field": 0, "no_trio": 0}
    for rid, g in df.groupby("rid", sort=False):
        g = g.sort_values("model_rank")
        n = len(g)
        if n < 3:
            drops["lt3_field"] += 1
            continue
        w = g["pl_w"].to_numpy(dtype=float)
        # raw win/place vectors (ordered by model_rank ascending)
        p_win = pl_probs.all_tansho(w)
        p_plc = pl_probs.all_fukusho(w)
        ai_rank_order = g["model_rank"].to_numpy()
        # market rank by ban -> we already have pop_rank per horse; race_confidence
        # only uses spearman of ai_rank_order vs market_rank, both as vectors in
        # the SAME horse order, so pass pop_rank aligned to g order.
        pop = g["pop_rank"].to_numpy(dtype=float)
        if np.any(np.isnan(pop)):
            market = None
        else:
            market = pop  # rank already; race_confidence re-ranks internally
        rc_raw = race_confidence(p_win, p_plc, ai_rank_order, market)
        # calibrated win: apply isotonic to raw p_win then renormalize
        p_win_cal = tansho_cal.predict(p_win)
        s = p_win_cal.sum()
        p_win_cal = p_win_cal / s if s > 0 else p_win
        rc_cal = race_confidence(p_win_cal, p_plc, ai_rank_order, market)

        # winning trio as a set of MODEL_RANKS
        win_ranks = frozenset(g.loc[g["finish"].isin([1, 2, 3]), "model_rank"].tolist())
        if len(win_ranks) != 3:
            drops["no_trio"] += 1
            continue
        tri_pay = float(g["tri_pay"].iloc[0])
        rows.append(dict(
            rid=rid, split=g["split"].iloc[0], year=int(g["year"].iloc[0]),
            n_horses=n,
            top1_dominance=rc_raw["top1_dominance"],
            top2_concentration=rc_raw["top2_concentration"],
            field_chaos_score=rc_raw["field_chaos_score"],
            ai_market_agreement=rc_raw["ai_market_agreement"],
            c_top1_dominance=rc_cal["top1_dominance"],
            c_top2_concentration=rc_cal["top2_concentration"],
            c_field_chaos_score=rc_cal["field_chaos_score"],
            c_ai_market_agreement=rc_cal["ai_market_agreement"],
            win_ranks=win_ranks, tri_pay=tri_pay,
        ))
    return pd.DataFrame(rows), drops


def pct_rank(s):
    """percentile transform within the given series (NaN-safe: NaN->NaN)."""
    return s.rank(pct=True)


def composite_score(d, prefix=""):
    """COMPOSITE 固さ = mean of [(1-chaos_pct), top1_dom_pct, top2_conc_pct,
    ai_market_agreement_pct]. If market NaN, mean over other 3. Percentiles are
    computed by the CALLER (per-split or fixed valid->test) before calling.
    Here d must already contain *_pct columns. Returns composite series."""
    chaos = d[prefix + "field_chaos_score_pct"]
    t1 = d[prefix + "top1_dominance_pct"]
    t2 = d[prefix + "top2_concentration_pct"]
    am = d[prefix + "ai_market_agreement_pct"]
    parts_fixed = [(1 - chaos), t1, t2]
    base = np.nansum(np.vstack([(1 - chaos).to_numpy(), t1.to_numpy(), t2.to_numpy()]), axis=0)
    cnt = np.full(len(d), 3.0)
    amv = am.to_numpy()
    has_am = ~np.isnan(amv)
    base = base + np.where(has_am, np.nan_to_num(amv), 0.0)
    cnt = cnt + has_am.astype(float)
    return pd.Series(base / cnt, index=d.index)


def assign_pct_perSplit(d, sigcols):
    """percentile each signal within each split separately."""
    out = d.copy()
    for col in sigcols:
        out[col + "_pct"] = out.groupby("split")[col].transform(pct_rank)
    return out


def assign_pct_fixed(d, sigcols):
    """fit percentile mapping on valid, apply to test (leak-safe).
    Implementation: for each test value, its pct = fraction of VALID values <= it."""
    out = d.copy()
    valid = d[d.split == "valid"]
    for col in sigcols:
        vv = valid[col].dropna().to_numpy()
        vv_sorted = np.sort(vv)
        def mapper(x):
            if np.isnan(x):
                return np.nan
            # fraction of valid <= x
            return np.searchsorted(vv_sorted, x, side="right") / len(vv_sorted)
        # valid gets its own per-split pct (agreement set); test gets fixed map
        col_pct = np.full(len(out), np.nan)
        is_v = (out.split == "valid").to_numpy()
        is_t = (out.split == "test").to_numpy()
        # valid: in-sample pct within valid
        col_pct[is_v] = valid[col].rank(pct=True).reindex(out.index[is_v]).to_numpy()
        # test: fixed map
        tvals = out.loc[is_t, col].to_numpy()
        col_pct[is_t] = np.array([mapper(x) for x in tvals])
        out[col + "_pct"] = col_pct
    return out


def quartile_bucket(score):
    """Q1_low (lowest 25%) .. Q4_high (highest 25%). Returns labels."""
    q = score.quantile([0.25, 0.5, 0.75]).to_numpy()
    def lab(x):
        if x <= q[0]:
            return "Q1_low"
        if x <= q[1]:
            return "Q2"
        if x <= q[2]:
            return "Q3"
        return "Q4_high"
    return score.apply(lab), q


def quartile_bucket_fixedcuts(score, cuts):
    def lab(x):
        if x <= cuts[0]:
            return "Q1_low"
        if x <= cuts[1]:
            return "Q2"
        if x <= cuts[2]:
            return "Q3"
        return "Q4_high"
    return score.apply(lab)


# ---- betting evaluation ----------------------------------------------------
def eval_strategy(races, tier_col=None, tier_to_form=None, fixed_form=None,
                  b0=False):
    """Return per-race (stake, ret) arrays.
    b0: flat prob-first top-6 PL combos @1000 ALL races.
    else: for each race use formation = (fixed_form) or tier_to_form[race[tier_col]].
    """
    stakes = []
    rets = []
    caps = []  # 1 if >=1 winning combo
    for _, r in races.iterrows():
        n = int(r["n_horses"])
        win = r["win_ranks"]
        pay = r["tri_pay"]  # yen per 100yen
        if b0:
            # top-6 PL combos: among model_rank 1..6 take ALL C(6,3)=20? NO.
            # "flat prob-first top-6 PL combos" = the 6 highest-probability
            # 三連puku combos by PL. Build PL combos and take top 6.
            combos = _top_pl_combos(r, n, 6)
            stake_each = 1000
        else:
            form_key = fixed_form if fixed_form else tier_to_form[r[tier_col]]
            form = FORMATIONS[form_key]
            combos = formation_combos(form, n)
            stake_each = form["stake"]
        if not combos:
            continue
        race_stake = stake_each * len(combos)
        hit = win in combos
        race_ret = pay * (stake_each / 100.0) if hit else 0.0
        stakes.append(race_stake)
        rets.append(race_ret)
        caps.append(1 if hit else 0)
    return np.array(stakes, float), np.array(rets, float), np.array(caps, float)


# precompute PL combo ranking per race for B0
_PLCACHE = {}


def _top_pl_combos(r, n, k):
    rid = r["rid"]
    if rid in _PLCACHE:
        ranked = _PLCACHE[rid]
    else:
        w = r["_pl_w"]
        sp = pl_probs.all_sanrenpuku(w)  # keys = positional index 0-based
        # positions are sorted by model_rank ascending in our build, so
        # position i corresponds to model_rank i+1.
        items = sorted(sp.items(), key=lambda kv: kv[1], reverse=True)
        ranked = [frozenset((i + 1, j + 1, kk + 1)) for (i, j, kk), _ in items]
        _PLCACHE[rid] = ranked
    return set(ranked[:k])


def blended_roi(stakes, rets):
    if stakes.sum() == 0:
        return float("nan")
    return 100.0 * rets.sum() / stakes.sum()


def boot_ci(stakes, rets, n=2000):
    m = len(stakes)
    if m == 0:
        return (float("nan"), float("nan"))
    idx = RNG.integers(0, m, size=(n, m))
    bs = stakes[idx].sum(axis=1)
    br = rets[idx].sum(axis=1)
    roi = 100.0 * br / np.where(bs == 0, np.nan, bs)
    return (float(np.nanpercentile(roi, 2.5)), float(np.nanpercentile(roi, 97.5)))


def capture(caps):
    return float(100.0 * caps.mean()) if len(caps) else float("nan")


def main():
    df, tansho_cal = load()
    races, drops = per_race_signals(df, tansho_cal)
    print("races built:", len(races), "drops:", drops, file=sys.stderr)

    # attach pl_w arrays per race for B0 combo ranking
    plw = {}
    for rid, g in df.groupby("rid", sort=False):
        g = g.sort_values("model_rank")
        plw[rid] = g["pl_w"].to_numpy(float)
    races["_pl_w"] = races["rid"].map(plw)

    sigcols = ["top1_dominance", "top2_concentration", "field_chaos_score", "ai_market_agreement"]

    results = {}

    # We run TWO p_win bases: RAW and CALIBRATED, and report. Primary = CALIBRATED
    # (faithful to production calibrator). Build composite for each.
    for basis, prefix in [("raw", ""), ("calibrated", "c_")]:
        scols = [prefix + c for c in sigcols]
        # PER-SPLIT percentile
        dps = assign_pct_perSplit(races, scols)
        comp_ps = composite_score(dps, prefix=prefix)
        dps = dps.assign(composite=comp_ps)

        test = dps[dps.split == "test"].copy()
        valid = dps[dps.split == "valid"].copy()

        # bucket within each split by composite quartile
        t_lab, t_cuts = quartile_bucket(test["composite"])
        v_lab, v_cuts = quartile_bucket(valid["composite"])
        test["tier"] = t_lab
        valid["tier"] = v_lab

        # ---- B0 flat prob-first top6 (test & valid) ----
        s0t, r0t, c0t = eval_strategy(test, b0=True)
        s0v, r0v, c0v = eval_strategy(valid, b0=True)
        # ---- B2 system tiering ----
        s2t, r2t, c2t = eval_strategy(test, tier_col="tier", tier_to_form=TIER_TO_FORM_SYSTEM)
        s2v, r2v, c2v = eval_strategy(valid, tier_col="tier", tier_to_form=TIER_TO_FORM_SYSTEM)
        # ---- B4 reversed ----
        s4t, r4t, c4t = eval_strategy(test, tier_col="tier", tier_to_form=TIER_TO_FORM_REV)
        s4v, r4v, c4v = eval_strategy(valid, tier_col="tier", tier_to_form=TIER_TO_FORM_REV)

        # per-tier (B2, test)
        per_tier = []
        for tier in ["Q4_high", "Q3", "Q2", "Q1_low"]:
            sub = test[test.tier == tier]
            form_key = TIER_TO_FORM_SYSTEM[tier]
            st, rt, ct = eval_strategy(sub, fixed_form=form_key)
            avg_pts = float(np.mean([len(formation_combos(FORMATIONS[form_key], int(n)))
                                     for n in sub["n_horses"]])) if len(sub) else 0.0
            ci = boot_ci(st, rt)
            per_tier.append(dict(
                tier=f"{tier}->{form_key}", n=int(len(sub)),
                avg_points=round(avg_pts, 2),
                stake_per_combo=FORMATIONS[form_key]["stake"],
                capture_pct=round(capture(ct), 2),
                roi_pct=round(blended_roi(st, rt), 2),
                ci_low=round(ci[0], 2), ci_high=round(ci[1], 2),
            ))

        # era split (test 2024 vs 2025)
        t24 = test[test.year == 2024]; t25 = test[test.year == 2025]
        s24, r24, _ = eval_strategy(t24, tier_col="tier", tier_to_form=TIER_TO_FORM_SYSTEM)
        s25, r25, _ = eval_strategy(t25, tier_col="tier", tier_to_form=TIER_TO_FORM_SYSTEM)

        # ---- FIXED-THRESHOLD: valid-derived composite cuts applied to test ----
        # use fixed valid->test percentile too (leak-safe) for composite, then
        # use valid quartile cuts.
        dfx = assign_pct_fixed(races, scols)
        comp_fx = composite_score(dfx, prefix=prefix)
        dfx = dfx.assign(composite=comp_fx)
        valid_fx = dfx[dfx.split == "valid"]
        _, vcuts_fx = quartile_bucket(valid_fx["composite"])
        test_fx = dfx[dfx.split == "test"].copy()
        test_fx["tier"] = quartile_bucket_fixedcuts(test_fx["composite"], vcuts_fx)
        sfx, rfx, _ = eval_strategy(test_fx, tier_col="tier", tier_to_form=TIER_TO_FORM_SYSTEM)

        ci2t = boot_ci(s2t, r2t)
        ci0t = boot_ci(s0t, r0t)
        ci4t = boot_ci(s4t, r4t)

        roi2t = blended_roi(s2t, r2t)
        roi0t = blended_roi(s0t, r0t)
        roi4t = blended_roi(s4t, r4t)
        roi2v = blended_roi(s2v, r2v)
        roi0v = blended_roi(s0v, r0v)

        # tiers_separate: 固いF4 ROI > 荒れF1 ROI with non-overlapping CIs
        f4 = next(p for p in per_tier if p["tier"].startswith("Q4_high"))
        f1 = next(p for p in per_tier if p["tier"].startswith("Q1_low"))
        tiers_sep = (f4["roi_pct"] > f1["roi_pct"]) and (f4["ci_low"] > f1["ci_high"])

        # beats_probfirst: B2>B0 clearly (CIs not fully overlapping) AND valid agrees
        cis_overlap = not (ci2t[0] > ci0t[1] or ci0t[0] > ci2t[1])
        beats_pf = (roi2t > roi0t) and (not cis_overlap) and (roi2v > roi0v)
        beats_be = ci2t[0] > 100.0

        results[basis] = dict(
            n_test=len(test), n_valid=len(valid),
            B0=dict(test_roi=round(roi0t, 2), ci=[round(ci0t[0], 2), round(ci0t[1], 2)],
                    valid_roi=round(roi0v, 2), capture=round(capture(c0t), 2),
                    stake_per_race=round(float(s0t.mean()), 1), n=int(len(s0t))),
            B2=dict(test_roi=round(roi2t, 2), ci=[round(ci2t[0], 2), round(ci2t[1], 2)],
                    valid_roi=round(roi2v, 2), capture=round(capture(c2t), 2),
                    stake_per_race=round(float(s2t.mean()), 1), n=int(len(s2t)),
                    era2024=round(blended_roi(s24, r24), 2),
                    era2025=round(blended_roi(s25, r25), 2),
                    fixed_thr_roi=round(blended_roi(sfx, rfx), 2)),
            B4=dict(test_roi=round(roi4t, 2), ci=[round(ci4t[0], 2), round(ci4t[1], 2)],
                    valid_roi=round(blended_roi(s4v, r4v), 2), capture=round(capture(c4t), 2),
                    stake_per_race=round(float(s4t.mean()), 1), n=int(len(s4t))),
            per_tier=per_tier,
            vs_probfirst_delta_pt=round(roi2t - roi0t, 2),
            vs_reversed_delta_pt=round(roi2t - roi4t, 2),
            tiers_separate=bool(tiers_sep),
            beats_probfirst=bool(beats_pf),
            beats_breakeven=bool(beats_be),
        )

    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
