"""
METHOD A - rank-based BOX (anchor/control) for 三連複.
Buy every combo among the top-k horses by model_rank, k in [3,4,5,6,7,8].
Cost = C(k,3) combos * 100 yen.
Market-blind baseline (reproduces trio_box_width).
"""
import sys
from itertools import combinations
import numpy as np
import pandas as pd

sys.path.insert(0, "E:/PyCaLiAI")

PARQUET = "E:/PyCaLiAI/analysis/_trio_substrate.parquet"
KS = [3, 4, 5, 6, 7, 8]
N_BOOT = 2000
RNG_SEED = 12345
STAKE = 100.0


def build_races(df):
    """Return dict rid -> race info needed for box betting."""
    races = {}
    dropped_no_tri = 0
    dropped_small = 0
    for rid, g in df.groupby("rid", sort=False):
        g = g.sort_values("model_rank")
        tri_pay = g["tri_pay"].iloc[0]
        if not np.isfinite(tri_pay) or tri_pay <= 0:
            dropped_no_tri += 1
            continue
        winners = frozenset(g.loc[g["finish"].isin([1, 2, 3]), "ban"].tolist())
        n_horses = int(g["n_horses"].iloc[0])
        # ranked ban list by model_rank ascending (1 = best)
        ranked_ban = g["ban"].tolist()
        races[rid] = {
            "ranked_ban": ranked_ban,
            "winners": winners,
            "tri_pay": float(tri_pay),
            "n_horses": n_horses,
            "split": g["split"].iloc[0],
        }
    return races, dropped_no_tri, dropped_small


def eval_box(races, k):
    """For each race, buy box of top-k by model_rank. Return per-race stake/return arrays.
    Races where fewer than k horses exist (or winning trio not fully determined) handled:
    - need at least k ranked horses to form the box; if field < k, buy box of all available (still C(avail,3)).
    - a race counts as 'bet' if we bought >=1 combo (i.e. avail>=3).
    """
    stakes = []
    returns = []
    hit = []
    for rid, r in races.items():
        ranked = r["ranked_ban"]
        avail = ranked[: min(k, len(ranked))]
        if len(avail) < 3:
            continue
        combos = list(combinations(avail, 3))
        stake = len(combos) * STAKE
        # winning trio must be exactly 3 horses
        winners = r["winners"]
        ret = 0.0
        won = 0
        if len(winners) == 3:
            for c in combos:
                if frozenset(c) == winners:
                    ret = r["tri_pay"]  # per 100yen stake on that combo
                    won = 1
                    break
        stakes.append(stake)
        returns.append(ret)
        hit.append(won)
    return np.array(stakes), np.array(returns), np.array(hit)


def bootstrap_roi_ci(stakes, returns, n_boot=N_BOOT, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    n = len(stakes)
    if n == 0:
        return (float("nan"), float("nan"))
    rois = np.empty(n_boot)
    idx_all = np.arange(n)
    for b in range(n_boot):
        idx = rng.choice(idx_all, size=n, replace=True)
        s = stakes[idx].sum()
        r = returns[idx].sum()
        rois[b] = (r / s * 100.0) if s > 0 else float("nan")
    lo, hi = np.nanpercentile(rois, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    df = pd.read_parquet(PARQUET)
    print("loaded", df.shape)

    out_rows = []
    for split in ["valid", "test"]:
        sub = df[df["split"] == split]
        races, dropped_no_tri, dropped_small = build_races(sub)
        n_races_total = len(races)
        print(f"\n=== split={split}  races_with_valid_tripay={n_races_total} (dropped_no_tri={dropped_no_tri}) ===")
        for k in KS:
            stakes, returns, hit = eval_box(races, k)
            n_bet = len(stakes)
            tot_stake = stakes.sum()
            tot_ret = returns.sum()
            roi = (tot_ret / tot_stake * 100.0) if tot_stake > 0 else float("nan")
            capture = (hit.sum() / n_bet * 100.0) if n_bet > 0 else float("nan")
            spr = (stakes.mean()) if n_bet > 0 else float("nan")
            lo, hi = bootstrap_roi_ci(stakes, returns)
            out_rows.append(dict(
                config=f"box_top{k}", split=split, n_races=n_bet,
                capture_pct=capture, roi_pct=roi, roi_ci_low=lo, roi_ci_high=hi,
                stake_per_race=spr, n_combos=int(stakes[0] / STAKE) if n_bet else 0,
            ))
            print(f"  box_top{k:>1} (C={int(round(spr/STAKE)) if n_bet else 0:>2}combos): "
                  f"n={n_bet:>4} cap={capture:5.2f}% roi={roi:6.2f}% CI[{lo:6.2f},{hi:6.2f}] spr={spr:.0f}")

    res = pd.DataFrame(out_rows)
    res.to_csv("E:/PyCaLiAI/analysis/_trio_bake_box_rank_results.csv", index=False)
    print("\nwrote results csv")
    print(res.to_string())


if __name__ == "__main__":
    main()
