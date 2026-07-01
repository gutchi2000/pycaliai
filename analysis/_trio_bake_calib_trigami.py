"""
METHOD F - CALIBRATION & TRIGAMI TRIM for 三連複 (trio) buying.

(f1) RAW prob-first top-8 vs CALIBRATED top-8.
     - sanrenpuku calibrator is a monotone IsotonicRegression p_raw -> p_cal,
       so it CANNOT change the within-race ranking of combos => same top-8.
       (We still verify this empirically.)
     - "Calibrated single-horse" variant: map each horse's p_tansho through the
       tansho IsotonicRegression -> calibrated win prob, renormalize to adjusted
       weights w', recompute trio probs from w' via pl_probs.all_sanrenpuku(w').
       This CAN re-rank combos. Reported as CALIB-SINGLE.

(f2) TRIGAMI-TRIM: prob-first, vary point count N from 12 down to 4.
     Report ROI / capture / stake-per-race tradeoff curve to locate the
     variance-minimizing (best ROI) point count. (True trigami floor needs
     per-combo odds which we do not have; this is the point-count proxy.)

Stake = 100 yen per combo. ROI% = total_return/total_stake*100.
Race-level bootstrap 95% CI on ROI, 2000 draws, default_rng(12345).
test = verdict split; valid = agreement check.
"""
import sys
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, "E:/PyCaLiAI")
import pl_probs as PL

SUB = "E:/PyCaLiAI/analysis/_trio_substrate.parquet"
CALPKL = "E:/PyCaLiAI/models/pl_calibrators_v6.pkl"
RNG_SEED = 12345
N_BOOT = 2000

df = pd.read_parquet(SUB)
cal = joblib.load(CALPKL)
tansho_iso = cal["calibrators"]["tansho"]
trio_iso = cal["calibrators"]["sanrenpuku"]

# ----------------------------------------------------------------------------
# Build per-race structures once.
# For each race: arrays of ban, pl_w, finish; winning trio set; tri_pay.
# ----------------------------------------------------------------------------
races = {}  # rid -> dict
dropped_no_trio = 0
dropped_no_pay = 0
dropped_small = 0

for rid, g in df.groupby("rid", sort=False):
    g = g.sort_values("model_rank")  # not strictly needed
    ban = g["ban"].to_numpy()
    w = g["pl_w"].to_numpy(dtype=np.float64)
    finish = g["finish"].to_numpy()
    split = g["split"].iloc[0]
    tri_pay = float(g["tri_pay"].iloc[0])
    n = len(ban)
    # winning trio = set of ban whose finish in {1,2,3}
    win_mask = np.isin(finish, [1, 2, 3])
    win_ban = set(ban[win_mask].tolist())
    if len(win_ban) != 3:
        dropped_no_trio += 1
        continue
    if not np.isfinite(tri_pay) or tri_pay <= 0:
        dropped_no_pay += 1
        continue
    if n < 4:
        dropped_small += 1
        continue
    races[rid] = dict(
        split=split, ban=ban, w=w, win_ban=win_ban, tri_pay=tri_pay, n=n
    )

print(f"[races] kept={len(races)}  dropped_no_trio={dropped_no_trio} "
      f"dropped_no_pay={dropped_no_pay} dropped_small={dropped_small}")
nv = sum(1 for r in races.values() if r["split"] == "valid")
nt = sum(1 for r in races.values() if r["split"] == "test")
print(f"[races] valid={nv} test={nt}")


def trio_combos_raw(w, ban, topn):
    """Return list of frozenset(ban-trio) for top-N combos by RAW PL trio prob."""
    probs = PL.all_sanrenpuku(w)  # {(i,j,k): p} over indices
    items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:topn]
    out = []
    for (i, j, k), _p in items:
        out.append(frozenset((int(ban[i]), int(ban[j]), int(ban[k]))))
    return out


def trio_combos_calib_single(w, ban, topn):
    """Recompute trio probs from CALIBRATED single-horse win probs.
    p_tansho_i = w_i/sum(w); map through tansho isotonic; renormalize -> w'.
    Then rank combos by all_sanrenpuku(w')."""
    s = w.sum()
    p_tan = w / s
    p_cal = tansho_iso.predict(p_tan)
    # guard: if all zero (degenerate), fall back to raw
    tot = p_cal.sum()
    if not np.isfinite(tot) or tot <= 0:
        wp = w
    else:
        wp = p_cal / tot  # adjusted weights (proportional to calibrated win prob)
        wp = np.clip(wp, 1e-12, None)
    probs = PL.all_sanrenpuku(wp)
    items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:topn]
    out = []
    for (i, j, k), _p in items:
        out.append(frozenset((int(ban[i]), int(ban[j]), int(ban[k]))))
    return out


def eval_config(combos_fn, topn, split):
    """Return per-race arrays: stake, ret (yen), hit(bool), for given split.
    combos_fn(w,ban,topn) -> list of frozenset trios."""
    stakes = []
    rets = []
    hits = []
    for rid, r in races.items():
        if r["split"] != split:
            continue
        n = r["n"]
        topn_eff = min(topn, n * (n - 1) * (n - 2) // 6)  # cap by available combos
        combos = combos_fn(r["w"], r["ban"], topn_eff)
        k = len(combos)
        stake = 100 * k
        win = r["win_ban"]
        ret = 0.0
        hit = False
        for c in combos:
            if c == win:
                ret += r["tri_pay"]  # tri_pay is yen per 100yen stake
                hit = True
        stakes.append(stake)
        rets.append(ret)
        hits.append(hit)
    return np.array(stakes, float), np.array(rets, float), np.array(hits, bool)


def boot_ci(stakes, rets, n_boot=N_BOOT, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    m = len(stakes)
    if m == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, m, size=(n_boot, m))
    rois = []
    for b in range(n_boot):
        ii = idx[b]
        st = stakes[ii].sum()
        rt = rets[ii].sum()
        rois.append(rt / st * 100 if st > 0 else np.nan)
    rois = np.array(rois, float)
    lo, hi = np.nanpercentile(rois, [2.5, 97.5])
    return float(lo), float(hi)


def summarize(name, combos_fn, topn, split):
    stakes, rets, hits = eval_config(combos_fn, topn, split)
    n_races = len(stakes)
    tot_st = stakes.sum()
    tot_rt = rets.sum()
    roi = tot_rt / tot_st * 100 if tot_st > 0 else float("nan")
    cap = hits.mean() * 100 if n_races else float("nan")
    spr = stakes.mean() if n_races else float("nan")
    lo, hi = boot_ci(stakes, rets)
    return dict(config=name, split=split, n_races=n_races, capture_pct=cap,
                roi_pct=roi, roi_ci_low=lo, roi_ci_high=hi, stake_per_race=spr)


results = []

# ---- f1: RAW top-8 vs CALIB top-8 (both splits) ----
for split in ["valid", "test"]:
    results.append(summarize("RAW_probfirst_top8", trio_combos_raw, 8, split))
    results.append(summarize("CALIBsingle_probfirst_top8",
                             trio_combos_calib_single, 8, split))

# Verify monotone-trio-iso claim: top-8 by raw == top-8 by sanrenpuku-calibrated
# (ranking-invariant). Do it on a sample of test races.
def _verify_trio_iso_invariant():
    rng = np.random.default_rng(0)
    test_rids = [rid for rid, r in races.items() if r["split"] == "test"]
    samp = rng.choice(len(test_rids), size=min(300, len(test_rids)), replace=False)
    mism = 0
    for s in samp:
        r = races[test_rids[s]]
        w = r["w"]; ban = r["ban"]
        probs = PL.all_sanrenpuku(w)
        keys = list(probs.keys())
        praw = np.array([probs[k] for k in keys])
        pcal = trio_iso.predict(praw)
        order_raw = [keys[i] for i in np.argsort(-praw)][:8]
        order_cal = [keys[i] for i in np.argsort(-pcal)][:8]
        if set(order_raw) != set(order_cal):
            mism += 1
    return mism, len(samp)

mism, ns = _verify_trio_iso_invariant()
print(f"[verify] sanrenpuku-iso top8 set mismatch vs raw: {mism}/{ns} races "
      f"(monotone => expect ~0; ties only)")

# ---- f2: TRIGAMI-TRIM curve, prob-first, N = 12,10,8,6,5,4 (both splits) ----
trim_curve = []
for split in ["valid", "test"]:
    for topn in [12, 10, 8, 6, 5, 4]:
        res = summarize(f"PROBFIRST_top{topn}", trio_combos_raw, topn, split)
        trim_curve.append(res)
        results.append(res)

# ---- print everything ----
print("\n=== ALL CONFIGS ===")
hdr = f"{'config':28s} {'split':5s} {'n':>5s} {'cap%':>6s} {'roi%':>7s} {'ci_lo':>7s} {'ci_hi':>7s} {'spr':>7s}"
print(hdr)
for r in results:
    print(f"{r['config']:28s} {r['split']:5s} {r['n_races']:5d} "
          f"{r['capture_pct']:6.2f} {r['roi_pct']:7.2f} {r['roi_ci_low']:7.2f} "
          f"{r['roi_ci_high']:7.2f} {r['stake_per_race']:7.0f}")

print("\n=== TRIGAMI-TRIM CURVE (test) ===")
for r in trim_curve:
    if r["split"] == "test":
        print(f"  top{r['config'].replace('PROBFIRST_top',''):>3s}  "
              f"roi={r['roi_pct']:6.2f}  cap={r['capture_pct']:5.2f}  "
              f"ci=[{r['roi_ci_low']:.1f},{r['roi_ci_high']:.1f}]  "
              f"spr={r['stake_per_race']:.0f}")

# pop_rank coverage note
pop_cov = df["pop_rank"].notna().mean() * 100
print(f"\n[note] pop_rank coverage = {pop_cov:.1f}% of horse-rows "
      f"(not used by Method F; reported for honesty)")

# emit a compact dict for the agent
import json
print("\n=== JSON ===")
print(json.dumps(results, default=float))
