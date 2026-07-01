"""
Independent verification of the 4-tier 三連複 chaos-matched system.
ADVERSARIAL: default is_real_edge = false. Refute the canonical B2 finding.

Substrate: analysis/_trio_substrate.parquet (1 row/horse/race).
Tiering = formations over model_rank positions. A combo {x,y,z} bought iff its
3 ranks can fill (A,B,C) by some permutation. 三連複 order-free: wins iff its
ban-set == winning trio set, returns tri_pay (per 100yen) scaled by stake.

ROI% = sum(returns)/sum(stakes)*100 money-weighted. capture% = races with >=1
winning bought combo / races bet. Per-race bootstrap 95% CI (2000 draws, rng 12345).
"""
import sys
from itertools import permutations, combinations
import numpy as np
import pandas as pd

sys.path.insert(0, "E:/PyCaLiAI")
import pl_probs

SUB = "E:/PyCaLiAI/analysis/_trio_substrate.parquet"
RNG_SEED = 12345

# ---- Tier formations: sets of RANK positions (1=best) ----
TIERS = {
    "F1": dict(A={1,2,3},   B={1,2,3,4},   C={1,2,3,4,5,6,7}, stake=400),   # 荒れ wide
    "F2": dict(A={1,2,3},   B={1,2,3},     C={1,2,3,4,5,6,7,8}, stake=600), # 中荒れ
    "F3": dict(A={1,2},     B={1,2,3},     C={1,2,3,4,5,6}, stake=1000),    # 本命来そう
    "F4": dict(A={1},       B={2,3},       C={2,3,4,5}, stake=2000),        # 本命堅い narrow
}

def combos_for_tier(tier, max_rank):
    """Return set of frozenset({rank_x,rank_y,rank_z}) bought under this formation,
    restricted to ranks <= max_rank (field size). Order-free dedup."""
    A = {r for r in tier["A"] if r <= max_rank}
    B = {r for r in tier["B"] if r <= max_rank}
    C = {r for r in tier["C"] if r <= max_rank}
    bought = set()
    # a combo of 3 distinct ranks is bought iff some assignment to (A,B,C) works
    # enumerate candidate ranks = union
    cand = sorted(A | B | C)
    for trio in combinations(cand, 3):
        for perm in permutations(trio):
            if perm[0] in A and perm[1] in B and perm[2] in C:
                bought.add(frozenset(trio))
                break
    return bought

def race_eval(rgrp, tier):
    """Evaluate one race under one tier formation. Return (stake_total, return_total, hit, n_combos)."""
    n = len(rgrp)
    # rank -> ban  (model_rank is 1..n unique per race)
    rank2ban = dict(zip(rgrp["model_rank"], rgrp["ban"]))
    bought_rank_sets = combos_for_tier(tier, n)
    stake = tier["stake"]
    n_combos = len(bought_rank_sets)
    stake_total = n_combos * stake
    # winning trio = set of ban with finish in {1,2,3}
    win_bans = frozenset(rgrp.loc[rgrp["finish"].isin([1,2,3]), "ban"].tolist())
    ret = 0.0
    hit = 0
    if len(win_bans) == 3:
        tri_pay = rgrp["tri_pay"].iloc[0]
        for rs in bought_rank_sets:
            bans = frozenset(rank2ban[r] for r in rs)
            if bans == win_bans:
                ret += stake * tri_pay / 100.0
                hit = 1
                break
    return stake_total, ret, hit, n_combos

# ---- chaos proxy per race ----
def race_chaos(rgrp):
    """Lower = more solid (favorite dominant). We return 'solidity' = top PL win prob.
    High solidity -> narrow tier. Use entropy alternative too."""
    w = rgrp.sort_values("model_rank")["pl_w"].to_numpy()
    p = pl_probs.all_tansho(w)
    top = float(p.max())
    ent = float(-(p*np.log(p+1e-12)).sum())
    return top, ent

def main():
    df = pd.read_parquet(SUB)
    # precompute per-race chaos + cache groups
    races = {}
    for rid, g in df.groupby("rid"):
        g = g.copy()
        split = g["split"].iloc[0]
        year = int(g["year"].iloc[0])
        top, ent = race_chaos(g)
        races[rid] = dict(g=g, split=split, year=year, solidity=top, entropy=ent)

    rids = list(races.keys())
    valid_rids = [r for r in rids if races[r]["split"]=="valid"]
    test_rids  = [r for r in rids if races[r]["split"]=="test"]

    # ---- precompute per-race eval under EACH tier (so we can assign later) ----
    eval_cache = {}  # rid -> {tier: (stake,ret,hit,ncombos)}
    for rid in rids:
        g = races[rid]["g"]
        eval_cache[rid] = {t: race_eval(g, TIERS[t]) for t in TIERS}

    # ---- assignment policies ----
    # solidity-quantile -> tier. We use 'solidity' (top win prob). High solidity=solid=narrow F4.
    # 4 quantile bins. Map: most chaotic(low solidity) -> F1, ... most solid(high) -> F4.
    def make_bins(rid_list, key="solidity"):
        vals = np.array([races[r][key] for r in rid_list])
        # 25/50/75 percentiles
        q = np.quantile(vals, [0.25,0.5,0.75])
        return q
    def assign_tier(val, q, reversed_=False):
        # q = [q25,q50,q75] of solidity. low solidity -> chaotic -> F1
        if val <= q[0]:   tier = "F1"
        elif val <= q[1]: tier = "F2"
        elif val <= q[2]: tier = "F3"
        else:             tier = "F4"
        if reversed_:
            tier = {"F1":"F4","F2":"F3","F3":"F2","F4":"F1"}[tier]
        return tier

    def system_results(rid_list, assign_fn):
        """assign_fn(rid)->tier. Returns arrays of per-race stake,ret,hit + tier."""
        stakes, rets, hits, tiers_used = [], [], [], []
        for rid in rid_list:
            t = assign_fn(rid)
            s,r,h,nc = eval_cache[rid][t]
            stakes.append(s); rets.append(r); hits.append(h); tiers_used.append(t)
        return np.array(stakes), np.array(rets), np.array(hits), tiers_used

    def roi(stakes, rets):
        return rets.sum()/stakes.sum()*100 if stakes.sum()>0 else float('nan')

    def boot_ci(stakes, rets, n=2000):
        rng = np.random.default_rng(RNG_SEED)
        m = len(stakes)
        out = np.empty(n)
        idx_all = np.arange(m)
        for b in range(n):
            idx = rng.choice(idx_all, size=m, replace=True)
            ss = stakes[idx].sum()
            out[b] = rets[idx].sum()/ss*100 if ss>0 else np.nan
        return float(np.nanpercentile(out,2.5)), float(np.nanpercentile(out,97.5))

    def capture(hits):
        return float(hits.mean()*100)

    # =========================================================
    # B0: prob-first flat baseline. Established best = flat top-4..6 combos.
    # We implement flat prob-first: buy top-K combos by PL sanrenpuku prob, flat stake.
    # Use the "single fixed formation" analog: F-flat = buy all C(top6,3) style?
    # Canonical "prob-first top-4..6": rank the C(n,3) combos by PL prob, take top N.
    # We'll do prob-first top-4,5,6 ranks box equivalents AND a true prob-rank top-20.
    # =========================================================
    def probfirst_eval(g, topk_combos, stake=500):
        """Buy the topk_combos highest-PL-prob 三連複 combos, flat stake."""
        n = len(g)
        w = g.sort_values("model_rank")["pl_w"].to_numpy()
        # map sorted-by-rank index -> ban
        bans_by_rank = g.sort_values("model_rank")["ban"].tolist()
        sp = pl_probs.all_sanrenpuku(w)  # keys are indices into sorted-by-rank order
        ranked = sorted(sp.items(), key=lambda kv: kv[1], reverse=True)[:topk_combos]
        win_bans = frozenset(g.loc[g["finish"].isin([1,2,3]),"ban"].tolist())
        stake_total = len(ranked)*stake
        ret=0.0; hit=0
        if len(win_bans)==3:
            tri_pay = g["tri_pay"].iloc[0]
            for (i,j,k),_ in ranked:
                bans = frozenset([bans_by_rank[i],bans_by_rank[j],bans_by_rank[k]])
                if bans==win_bans:
                    ret += stake*tri_pay/100.0; hit=1; break
        return stake_total, ret, hit

    def probfirst_system(rid_list, topk, stake=500):
        ss,rr,hh=[],[],[]
        for rid in rid_list:
            s,r,h = probfirst_eval(races[rid]["g"], topk, stake)
            ss.append(s);rr.append(r);hh.append(h)
        return np.array(ss),np.array(rr),np.array(hh)

    # ---- VALID-fixed bins (leak-safe) ----
    q_valid = make_bins(valid_rids, "solidity")

    results = {}

    # B2 tiered (per-split-quantile = in-sample on TEST) -- canonical
    q_test = make_bins(test_rids,"solidity")
    assign_B2_test = lambda rid: assign_tier(races[rid]["solidity"], q_test)
    assign_B2_valid = lambda rid: assign_tier(races[rid]["solidity"], q_valid)
    s,r,h,tu = system_results(test_rids, assign_B2_test)
    results["B2_test_insample"] = dict(stakes=s,rets=r,hits=h,tiers=tu)
    s,r,h,tu = system_results(valid_rids, assign_B2_valid)
    results["B2_valid"] = dict(stakes=s,rets=r,hits=h,tiers=tu)

    # B2-leaksafe: bins FIXED from valid, applied to test
    assign_B2_leaksafe = lambda rid: assign_tier(races[rid]["solidity"], q_valid)
    s,r,h,tu = system_results(test_rids, assign_B2_leaksafe)
    results["B2_test_leaksafe"] = dict(stakes=s,rets=r,hits=h,tiers=tu)

    # B3: same tier ASSIGNMENT as B2 (in-sample test bins) but FLAT stake (kill staking tilt)
    FLAT = 700
    def system_flat(rid_list, assign_fn, flat=FLAT):
        ss,rr,hh,tu=[],[],[],[]
        for rid in rid_list:
            t=assign_fn(rid)
            g=races[rid]["g"]
            # recompute with flat stake
            n=len(g)
            bought=combos_for_tier(TIERS[t], n)
            nc=len(bought)
            stake_total=nc*flat
            rank2ban=dict(zip(g["model_rank"],g["ban"]))
            win_bans=frozenset(g.loc[g["finish"].isin([1,2,3]),"ban"].tolist())
            ret=0.0;hit=0
            if len(win_bans)==3:
                tp=g["tri_pay"].iloc[0]
                for rs in bought:
                    if frozenset(rank2ban[x] for x in rs)==win_bans:
                        ret+=flat*tp/100.0;hit=1;break
            ss.append(stake_total);rr.append(ret);hh.append(hit);tu.append(t)
        return np.array(ss),np.array(rr),np.array(hh),tu
    s,r,h,tu = system_flat(test_rids, assign_B2_test)
    results["B3_flatstake_test"] = dict(stakes=s,rets=r,hits=h,tiers=tu)

    # B4: REVERSED tilt (chaotic->narrow F4, solid->wide F1), in-sample test bins, normal stakes
    assign_B4 = lambda rid: assign_tier(races[rid]["solidity"], q_test, reversed_=True)
    s,r,h,tu = system_results(test_rids, assign_B4)
    results["B4_reversed_test"] = dict(stakes=s,rets=r,hits=h,tiers=tu)

    # B0: prob-first flat baselines, test
    for k in (4,6,10,20):
        s,r,h = probfirst_system(test_rids,k)
        results[f"B0_probfirst_top{k}_test"] = dict(stakes=s,rets=r,hits=h,tiers=None)

    # ---- single-tier homogeneous baselines on test (sanity) ----
    for t in TIERS:
        s,r,h,tu = system_results(test_rids, (lambda tt: (lambda rid: tt))(t))
        results[f"single_{t}_test"] = dict(stakes=s,rets=r,hits=h,tiers=[t]*len(test_rids))

    # ---- compute metrics ----
    def summary(d):
        roi_v = roi(d["stakes"],d["rets"])
        lo,hi = boot_ci(d["stakes"],d["rets"])
        cap = capture(d["hits"])
        stake_per_race = float(d["stakes"].mean())
        return dict(roi=roi_v, ci_low=lo, ci_high=hi, capture=cap,
                    stake_per_race=stake_per_race, n=len(d["stakes"]))

    S = {k: summary(v) for k,v in results.items()}

    # B2 era split
    def era_roi(year):
        rl = [r for r in test_rids if races[r]["year"]==year]
        s,r,h,tu = system_results(rl, assign_B2_test)
        return roi(s,r)
    era24 = era_roi(2024); era25 = era_roi(2025)

    # paired bootstrap: B2 - B4 and B2 - B0(best) and B2 - B3, per-race money-weighted diff
    def paired_diff_ci(dA, dB, n=2000):
        # both indexed over test_rids in same order
        rng = np.random.default_rng(RNG_SEED)
        m=len(dA["stakes"]); idx_all=np.arange(m)
        diffs=np.empty(n)
        for b in range(n):
            idx=rng.choice(idx_all,size=m,replace=True)
            rA=dA["rets"][idx].sum()/dA["stakes"][idx].sum()*100
            rB=dB["rets"][idx].sum()/dB["stakes"][idx].sum()*100
            diffs[b]=rA-rB
        return float(np.nanpercentile(diffs,2.5)), float(np.nanpercentile(diffs,97.5)), float(diffs.mean())

    # pick best B0
    b0_keys=[k for k in results if k.startswith("B0_")]
    best_b0=max(b0_keys, key=lambda k: S[k]["roi"])

    pd_b2_b4 = paired_diff_ci(results["B2_test_insample"], results["B4_reversed_test"])
    pd_b2_b0 = paired_diff_ci(results["B2_test_insample"], results[best_b0])
    pd_b2_b3 = paired_diff_ci(results["B2_test_insample"],
                              {"stakes":results["B3_flatstake_test"]["stakes"],
                               "rets":results["B3_flatstake_test"]["rets"]})
    pd_ls_b0 = paired_diff_ci(results["B2_test_leaksafe"], results[best_b0])

    out = dict(
        summaries=S, era24=era24, era25=era25,
        best_b0=best_b0,
        paired_B2_B4=pd_b2_b4, paired_B2_B0=pd_b2_b0,
        paired_B2_B3=pd_b2_b3, paired_leaksafe_B0=pd_ls_b0,
        q_valid=q_valid.tolist(), q_test=q_test.tolist(),
    )
    import json
    print(json.dumps(out, indent=2, default=float))

if __name__=="__main__":
    main()
