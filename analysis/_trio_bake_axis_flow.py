"""
METHOD C - 三連複 AXIS流し backtest.

(c1) ◎ fixed (model_rank 1) + 2 partners from model top-m by score, m in [5,6,7,8] -> C(m-1,2) combos.
(c2) ◎〇 both fixed (rank 1&2) + 1 partner from model top-m, m in [5,6,7,8] -> (m-2) combos.

Stake = 100 yen per combo. A combo wins iff its set of 3 ban == winning trio (finish in {1,2,3}).
ROI% = total_return / total_stake * 100. capture% = races with >=1 winning combo / races bet * 100.
stake_per_race = mean combos bought * 100.
Race-level bootstrap 95% CI on ROI (2000 draws, rng(12345)).
"""
import sys
from itertools import combinations
import numpy as np
import pandas as pd

sys.path.insert(0, "E:/PyCaLiAI")

PARQUET = "E:/PyCaLiAI/analysis/_trio_substrate.parquet"
RNG_SEED = 12345
N_BOOT = 2000


def build_races(df):
    """Return list of per-race dicts for a split, sorted by model rank."""
    races = []
    for rid, g in df.groupby("rid", sort=False):
        g = g.sort_values("model_rank")
        ban = g["ban"].to_numpy()
        finish = g["finish"].to_numpy()
        tri_pay = float(g["tri_pay"].iloc[0])
        # winning trio = set of ban whose finish in {1,2,3}
        win_mask = np.isin(finish, [1, 2, 3])
        win_trio = frozenset(ban[win_mask].tolist())
        n = len(ban)
        races.append(
            {
                "rid": rid,
                "ban": ban,          # ordered by model_rank ascending (index0 = rank1 = ◎)
                "win_trio": win_trio,
                "tri_pay": tri_pay,
                "n": n,
                "win_ok": len(win_trio) == 3,  # well-formed winning trio
            }
        )
    return races


def combos_c1(race, m):
    """◎ fixed + 2 partners from model top-m. Need at least m horses; axis is index0."""
    if race["n"] < m or m < 3:
        return None  # can't form; race not bet
    ban = race["ban"]
    axis = ban[0]
    pool = ban[1:m]  # ranks 2..m  -> (m-1) horses
    combos = [frozenset((axis, a, b)) for a, b in combinations(pool, 2)]
    return combos


def combos_c2(race, m):
    """◎〇 fixed + 1 partner from model top-m. axis = index0,1; partners ranks 3..m."""
    if race["n"] < m or m < 3:
        return None
    ban = race["ban"]
    a0, a1 = ban[0], ban[1]
    pool = ban[2:m]  # ranks 3..m -> (m-2) horses
    combos = [frozenset((a0, a1, p)) for p in pool]
    return combos


def eval_config(races, combo_fn, m):
    """Return per-race arrays: stake, ret (yen returned), win(bool), n_combos. Only bet races included."""
    stakes, rets, wins, ncombos = [], [], [], []
    skipped = 0
    for r in races:
        combos = combo_fn(r, m)
        if combos is None or len(combos) == 0:
            skipped += 1
            continue
        nc = len(combos)
        stake = nc * 100.0
        ret = 0.0
        hit = False
        if r["win_ok"]:
            for c in combos:
                if c == r["win_trio"]:
                    ret += r["tri_pay"]
                    hit = True
                    break  # exactly one combo can match a 3-set winning trio
        stakes.append(stake)
        rets.append(ret)
        wins.append(hit)
        ncombos.append(nc)
    return (
        np.array(stakes),
        np.array(rets),
        np.array(wins, dtype=bool),
        np.array(ncombos, dtype=int),
        skipped,
    )


def bootstrap_roi_ci(stakes, rets, n_boot=N_BOOT, seed=RNG_SEED):
    if len(stakes) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = len(stakes)
    rois = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        s = stakes[idx].sum()
        rt = rets[idx].sum()
        rois[b] = rt / s * 100.0 if s > 0 else float("nan")
    lo, hi = np.nanpercentile(rois, [2.5, 97.5])
    return float(lo), float(hi)


def run():
    df = pd.read_parquet(PARQUET)
    out_rows = []
    races_by_split = {}
    for split in ["valid", "test"]:
        sub = df[df["split"] == split]
        races_by_split[split] = build_races(sub)
        nbad = sum(1 for r in races_by_split[split] if not r["win_ok"])
        print(f"[{split}] total races={len(races_by_split[split])}  malformed_win_trio={nbad}")

    configs = []
    for m in [5, 6, 7, 8]:
        configs.append((f"c1_axis◎_top{m}", combos_c1, m))
    for m in [5, 6, 7, 8]:
        configs.append((f"c2_axis◎〇_top{m}", combos_c2, m))

    results = {}
    for name, fn, m in configs:
        for split in ["valid", "test"]:
            races = races_by_split[split]
            stakes, rets, wins, ncombos, skipped = eval_config(races, fn, m)
            n_bet = len(stakes)
            if n_bet == 0:
                continue
            total_stake = stakes.sum()
            total_ret = rets.sum()
            roi = total_ret / total_stake * 100.0
            capture = wins.mean() * 100.0
            stake_per_race = stakes.mean()
            ci_lo, ci_hi = bootstrap_roi_ci(stakes, rets)
            key = (name, split)
            results[key] = dict(
                config=name,
                split=split,
                n_races=n_bet,
                skipped=skipped,
                capture_pct=capture,
                roi_pct=roi,
                ci_lo=ci_lo,
                ci_hi=ci_hi,
                stake_per_race=stake_per_race,
                mean_combos=ncombos.mean(),
            )
            print(
                f"{name:18s} [{split:5s}] n={n_bet:5d} skip={skipped:4d} "
                f"combos/race={ncombos.mean():.1f} cap={capture:5.2f}% "
                f"ROI={roi:6.2f}% CI[{ci_lo:6.2f},{ci_hi:6.2f}] stake/race={stake_per_race:.0f}"
            )

    # Best test config by ROI
    test_keys = [k for k in results if k[1] == "test"]
    best = max(test_keys, key=lambda k: results[k]["roi_pct"])
    print("\nBEST TEST CONFIG:", results[best])

    import json
    payload = {f"{k[0]}|{k[1]}": v for k, v in results.items()}
    with open("E:/PyCaLiAI/analysis/_trio_axis_flow_result.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("\nwrote E:/PyCaLiAI/analysis/_trio_axis_flow_result.json")


if __name__ == "__main__":
    run()
