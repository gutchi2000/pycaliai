"""
_trio_tier_marketproxy.py
=========================
ROBUSTNESS RUN of the USER'S 4-TIER 三連複 SYSTEM using a MARKET-based chaos
proxy instead of PL entropy.

Chaos proxy: per-race 9am MARKET favorite strength = min(ou_lastlog) over the
race's horses (ou_lastlog lower = more popular). A small min => one very strong
favorite => SOLID race. A large min => no strong favorite => CHAOTIC race.

This is closer to how a human eyeballs "this looks solid": look at how short the
top favorite is. Quartile cutoffs on min(ou_lastlog) computed SEPARATELY per
split (valid=2023, test=2024-25), over races with pop coverage.

  strongest-favorite 25% (smallest min_ou) -> F4 本命堅い   (narrow, expensive)
  ...
  weakest-favorite   25% (largest  min_ou) -> F1 荒れ       (wide, cheap)

Formations are over RANK positions (model_rank, 1=◎). A 3-ban combo {x,y,z} is
bought iff its three ranks can be assigned to (A,B,C) by SOME permutation. Ranks
beyond field size simply don't exist. 三連複 = order-free; a bought combo WINS iff
its 3-ban set == winning trio; returns tri_pay yen per 100yen, scaled by the
per-combo stake.

Reports B2 (user system), B0 (flat prob-first top-6 1000/combo, ALL races with
pop coverage), B4 (reversed market call). Plus per-tier breakdown, era split,
and verdicts. Same metrics/CI/era convention as the canonical (entropy) run.

Run: PYTHONUTF8=1 E:/PyCaLiAI/venv311/Scripts/python.exe analysis/_trio_tier_marketproxy.py
"""
from __future__ import annotations

import json
import sys
from itertools import permutations, combinations

import numpy as np
import pandas as pd

sys.path.insert(0, "E:/PyCaLiAI")
import pl_probs  # noqa: E402

SUBSTRATE = "E:/PyCaLiAI/analysis/_trio_substrate.parquet"
RNG_SEED = 12345
N_BOOT = 2000

# ---------------------------------------------------------------------------
# Formations: candidate RANK sets (1-indexed model_rank), per-combo stake yen.
# ---------------------------------------------------------------------------
FORMATIONS = {
    "F1": dict(A={1, 2, 3}, B={1, 2, 3, 4}, C={1, 2, 3, 4, 5, 6, 7}, stake=400),
    "F2": dict(A={1, 2, 3}, B={1, 2, 3}, C={1, 2, 3, 4, 5, 6, 7, 8}, stake=600),
    "F3": dict(A={1, 2}, B={1, 2, 3}, C={1, 2, 3, 4, 5, 6}, stake=1000),
    "F4": dict(A={1}, B={2, 3}, C={2, 3, 4, 5}, stake=2000),
}


def formation_combos(formation: dict, n_horses: int) -> set[frozenset]:
    """Set of bought 3-RANK combos (frozenset of ranks) given field size."""
    A = {r for r in formation["A"] if r <= n_horses}
    B = {r for r in formation["B"] if r <= n_horses}
    C = {r for r in formation["C"] if r <= n_horses}
    bought = set()
    cand = sorted(A | B | C)
    for trio in combinations(cand, 3):
        for perm in permutations(trio):
            if perm[0] in A and perm[1] in B and perm[2] in C:
                bought.add(frozenset(trio))
                break
    return bought


# ---------------------------------------------------------------------------
# Build per-race structures (carry market favorite strength = min ou_lastlog)
# ---------------------------------------------------------------------------
def build_races(df: pd.DataFrame):
    """Return list of race dicts. Drops dead-heat-ambiguous races and races
    with NO pop coverage (no priced horse => no market call possible)."""
    races = []
    dropped_ambiguous = 0
    dropped_nopop = 0
    for rid, g in df.groupby("rid", sort=False):
        g = g.sort_values("model_rank")
        n = len(g)
        finish = g["finish"].values
        rank = g["model_rank"].values
        pl_w = g["pl_w"].values.astype(np.float64)
        tri_pay = float(g["tri_pay"].iloc[0])
        split = g["split"].iloc[0]
        year = int(g["year"].iloc[0])
        ou = g["ou_lastlog"].values.astype(np.float64)

        win_mask = np.isin(finish, [1, 2, 3])
        if win_mask.sum() != 3:
            dropped_ambiguous += 1
            continue
        win_ranks = frozenset(int(r) for r in rank[win_mask])

        # MARKET chaos proxy: strongest favorite = min ou_lastlog (lower=popular)
        if np.all(np.isnan(ou)):
            dropped_nopop += 1
            continue
        min_ou = float(np.nanmin(ou))

        races.append(dict(
            rid=rid, split=split, year=year, n=n,
            win_ranks=win_ranks, tri_pay=tri_pay,
            min_ou=min_ou, pl_w=pl_w,
        ))
    return races, dropped_ambiguous, dropped_nopop


# ---------------------------------------------------------------------------
# Strategy evaluators -> per-race (stake, ret, hit, n_pts)
# ---------------------------------------------------------------------------
def eval_formation_on_race(race: dict, formation: dict):
    combos = formation_combos(formation, race["n"])
    n_pts = len(combos)
    stake = n_pts * formation["stake"]
    ret = 0.0
    hit = False
    if race["win_ranks"] in combos:
        hit = True
        ret = race["tri_pay"] * (formation["stake"] / 100.0)
    return stake, ret, hit, n_pts


def eval_probfirst_topk_on_race(race: dict, k_combos: int, stake_per_combo: float):
    probs = pl_probs.all_sanrenpuku(race["pl_w"])  # {(i,j,k): prob}, 0-based rows
    if not probs:
        return 0.0, 0.0, False, 0
    items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:k_combos]
    bought = set()
    for (i, j, kk), _p in items:
        bought.add(frozenset((i + 1, j + 1, kk + 1)))  # row idx -> rank
    n_pts = len(bought)
    stake = n_pts * stake_per_combo
    ret = 0.0
    hit = False
    if race["win_ranks"] in bought:
        hit = True
        ret = race["tri_pay"] * (stake_per_combo / 100.0)
    return stake, ret, hit, n_pts


# ---------------------------------------------------------------------------
# Tiering by MARKET favorite strength (min ou_lastlog) quartiles, per split.
# Default: strongest favorite (smallest min_ou) -> F4 (本命堅い, narrow/expensive),
#          weakest favorite  (largest  min_ou) -> F1 (荒れ, wide/cheap).
# Reversed: strongest favorite -> F1, weakest -> F4 (deliberate mismatch).
# ---------------------------------------------------------------------------
def assign_tiers(races: list, reversed_map: bool = False):
    for split in ("valid", "test"):
        sub = [r for r in races if r["split"] == split]
        if not sub:
            continue
        mo = np.array([r["min_ou"] for r in sub])
        q25, q50, q75 = np.quantile(mo, [0.25, 0.50, 0.75])
        for r in sub:
            m = r["min_ou"]
            # solidity quartile by min_ou: 0=strongest fav (solid) .. 3=weakest (chaotic)
            if m <= q25:
                qi = 0  # strongest favorite => most solid
            elif m <= q50:
                qi = 1
            elif m <= q75:
                qi = 2
            else:
                qi = 3  # weakest favorite => most chaotic
            if not reversed_map:
                # solid -> F4, chaotic -> F1  (correct human-eyeball call)
                tier = {0: "F4", 1: "F3", 2: "F2", 3: "F1"}[qi]
            else:
                # reversed mismatch: solid -> F1, chaotic -> F4
                tier = {0: "F1", 1: "F2", 2: "F3", 3: "F4"}[qi]
            r["_qi"] = qi
            r["tier"] = tier


# ---------------------------------------------------------------------------
# Bootstrap CI on money-weighted blended ROI
# ---------------------------------------------------------------------------
def blended_roi(stakes: np.ndarray, returns: np.ndarray) -> float:
    s = stakes.sum()
    if s <= 0:
        return float("nan")
    return float(returns.sum() / s * 100.0)


def bootstrap_ci(stakes: np.ndarray, returns: np.ndarray, n_boot=N_BOOT, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    m = len(stakes)
    if m == 0:
        return float("nan"), float("nan")
    rois = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, m, m)
        s = stakes[idx].sum()
        rois[b] = returns[idx].sum() / s * 100.0 if s > 0 else np.nan
    lo, hi = np.nanpercentile(rois, [2.5, 97.5])
    return float(lo), float(hi)


def summarize(stakes_list, returns_list, hits_list, name):
    stakes = np.array(stakes_list, dtype=np.float64)
    returns = np.array(returns_list, dtype=np.float64)
    hits = np.array(hits_list, dtype=bool)
    roi = blended_roi(stakes, returns)
    lo, hi = bootstrap_ci(stakes, returns)
    cap = float(hits.mean() * 100.0) if len(hits) else float("nan")
    stake_per_race = float(stakes.mean()) if len(stakes) else float("nan")
    return dict(name=name, roi=roi, ci_low=lo, ci_high=hi, capture_pct=cap,
                stake_per_race=stake_per_race, n_races=int(len(stakes)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = pd.read_parquet(SUBSTRATE)
    races, dropped, dropped_nopop = build_races(df)
    n_valid_races = sum(1 for r in races if r["split"] == "valid")
    n_test_races = sum(1 for r in races if r["split"] == "test")
    print(f"[build] usable races (with pop coverage): valid={n_valid_races} "
          f"test={n_test_races} (dropped {dropped} dead-heat, "
          f"{dropped_nopop} no-pop-coverage)")

    test = [r for r in races if r["split"] == "test"]
    valid = [r for r in races if r["split"] == "valid"]

    results = {}

    # ---- B0: flat prob-first top-6, 1000/combo, ALL (pop-covered) races ----
    def run_b0(rs):
        S, R, H = [], [], []
        for r in rs:
            s, ret, hit, _ = eval_probfirst_topk_on_race(r, 6, 1000.0)
            S.append(s); R.append(ret); H.append(hit)
        return S, R, H
    b0_t = summarize(*run_b0(test), "B0")
    b0_v = summarize(*run_b0(valid), "B0")
    results["B0"] = (b0_t, b0_v)

    # ---- B1_Fi: each formation standalone on ALL races ----
    def run_formation_all(rs, fkey):
        S, R, H, PTS = [], [], [], []
        f = FORMATIONS[fkey]
        for r in rs:
            s, ret, hit, npt = eval_formation_on_race(r, f)
            S.append(s); R.append(ret); H.append(hit); PTS.append(npt)
        return S, R, H, PTS
    for fkey in ("F1", "F2", "F3", "F4"):
        St, Rt, Ht, _ = run_formation_all(test, fkey)
        Sv, Rv, Hv, _ = run_formation_all(valid, fkey)
        results[f"B1_{fkey}"] = (summarize(St, Rt, Ht, f"B1_{fkey}"),
                                 summarize(Sv, Rv, Hv, f"B1_{fkey}"))

    # ---- B2: USER SYSTEM (market quartile -> matched formation + stake) ----
    assign_tiers(races, reversed_map=False)

    def run_tiered(rs, flat_stake=None):
        S, R, H = [], [], []
        per_tier = {f: dict(S=[], R=[], H=[], PTS=[]) for f in FORMATIONS}
        for r in rs:
            fkey = r["tier"]
            f = FORMATIONS[fkey]
            if flat_stake is None:
                s, ret, hit, npt = eval_formation_on_race(r, f)
            else:
                combos = formation_combos(f, r["n"])
                npt = len(combos)
                s = npt * flat_stake
                hit = r["win_ranks"] in combos
                ret = r["tri_pay"] * (flat_stake / 100.0) if hit else 0.0
            S.append(s); R.append(ret); H.append(hit)
            pt = per_tier[fkey]
            pt["S"].append(s); pt["R"].append(ret); pt["H"].append(hit); pt["PTS"].append(npt)
        return S, R, H, per_tier

    St, Rt, Ht, per_tier_t = run_tiered(test)
    Sv, Rv, Hv, _ = run_tiered(valid)
    b2_t = summarize(St, Rt, Ht, "B2")
    b2_v = summarize(Sv, Rv, Hv, "B2")
    results["B2"] = (b2_t, b2_v)

    # per-tier breakdown on TEST
    per_tier_rows = []
    for fkey in ("F1", "F2", "F3", "F4"):
        pt = per_tier_t[fkey]
        S = np.array(pt["S"], float); R = np.array(pt["R"], float); H = np.array(pt["H"], bool)
        PTS = np.array(pt["PTS"], float)
        if len(S) == 0:
            continue
        roi = blended_roi(S, R)
        lo, hi = bootstrap_ci(S, R)
        per_tier_rows.append(dict(
            tier=fkey,
            n_races_test=int(len(S)),
            avg_points=float(PTS.mean()),
            stake_per_combo=int(FORMATIONS[fkey]["stake"]),
            capture_pct=float(H.mean() * 100.0),
            roi_pct=roi, roi_ci_low=lo, roi_ci_high=hi,
        ))

    # ---- B3: tiered formation map but FLAT 1000/combo ----
    St3, Rt3, Ht3, _ = run_tiered(test, flat_stake=1000.0)
    Sv3, Rv3, Hv3, _ = run_tiered(valid, flat_stake=1000.0)
    results["B3"] = (summarize(St3, Rt3, Ht3, "B3"),
                     summarize(Sv3, Rv3, Hv3, "B3"))

    # ---- B4: REVERSED market call (formation + stake both reversed) ----
    assign_tiers(races, reversed_map=True)
    St4, Rt4, Ht4, _ = run_tiered(test)
    Sv4, Rv4, Hv4, _ = run_tiered(valid)
    results["B4"] = (summarize(St4, Rt4, Ht4, "B4"),
                     summarize(Sv4, Rv4, Hv4, "B4"))

    # restore default map
    assign_tiers(races, reversed_map=False)

    # ---- era split on TEST for B2 (2024 vs 2025) ----
    def run_tiered_subset(rs):
        S, R, H, _ = run_tiered(rs)
        return summarize(S, R, H, "B2era")
    test_2024 = [r for r in test if r["year"] == 2024]
    test_2025 = [r for r in test if r["year"] == 2025]
    b2_2024 = run_tiered_subset(test_2024)
    b2_2025 = run_tiered_subset(test_2025)

    # ---- verdicts ----
    vs_probfirst = b2_t["roi"] - b0_t["roi"]
    vs_reversed = b2_t["roi"] - results["B4"][0]["roi"]

    tier_roi = {row["tier"]: row for row in per_tier_rows}
    f4 = tier_roi.get("F4"); f1 = tier_roi.get("F1")
    tiers_separate = False
    if f4 and f1:
        monotone_ish = f4["roi_pct"] > f1["roi_pct"]
        ci_separate = (f4["roi_ci_low"] > f1["roi_ci_high"]) or (f1["roi_ci_high"] < f4["roi_ci_low"])
        tiers_separate = bool(monotone_ish and ci_separate)

    non_overlap_test = (b2_t["ci_low"] > b0_t["ci_high"]) or (b0_t["ci_low"] > b2_t["ci_high"])
    valid_agrees = b2_v["roi"] > b0_v["roi"]
    beats_probfirst = bool(b2_t["roi"] > b0_t["roi"] and non_overlap_test and valid_agrees)
    beats_breakeven = bool(b2_t["ci_low"] > 100.0)

    # ---- assemble structured output ----
    baselines = []
    baselines.append(dict(
        name="B0", test_roi=b0_t["roi"], ci_low=b0_t["ci_low"], ci_high=b0_t["ci_high"],
        capture_pct=b0_t["capture_pct"], stake_per_race=b0_t["stake_per_race"],
        valid_roi=b0_v["roi"], n_races=b0_t["n_races"]))
    for fkey in ("F1", "F2", "F3", "F4"):
        bt, bv = results[f"B1_{fkey}"]
        baselines.append(dict(
            name=f"B1_{fkey}", test_roi=bt["roi"], ci_low=bt["ci_low"], ci_high=bt["ci_high"],
            capture_pct=bt["capture_pct"], stake_per_race=bt["stake_per_race"],
            valid_roi=bv["roi"], n_races=bt["n_races"]))
    b3t, b3v = results["B3"]
    baselines.append(dict(
        name="B3", test_roi=b3t["roi"], ci_low=b3t["ci_low"], ci_high=b3t["ci_high"],
        capture_pct=b3t["capture_pct"], stake_per_race=b3t["stake_per_race"],
        valid_roi=b3v["roi"], n_races=b3t["n_races"]))
    b4t, b4v = results["B4"]
    baselines.append(dict(
        name="B4", test_roi=b4t["roi"], ci_low=b4t["ci_low"], ci_high=b4t["ci_high"],
        capture_pct=b4t["capture_pct"], stake_per_race=b4t["stake_per_race"],
        valid_roi=b4v["roi"], n_races=b4t["n_races"]))

    tiered_system = dict(
        test_roi=b2_t["roi"], ci_low=b2_t["ci_low"], ci_high=b2_t["ci_high"],
        valid_roi=b2_v["roi"], stake_per_race=b2_t["stake_per_race"],
        era_2024_roi=b2_2024["roi"], era_2025_roi=b2_2025["roi"])

    out = dict(
        proxy="market_min_ou_lastlog_quartiles_per_split",
        column_basis="model_rank (1=◎) for formations; chaos call by per-race "
                     "min(ou_lastlog) (strongest 9am favorite). Smallest min_ou "
                     "quartile -> F4 本命堅い (solid), largest -> F1 荒れ (chaotic). "
                     "Quartiles per split. Races without pop coverage dropped.",
        baselines=baselines,
        per_tier=per_tier_rows,
        tiered_system=tiered_system,
        vs_probfirst_delta_pt=vs_probfirst,
        vs_reversed_delta_pt=vs_reversed,
        tiers_separate=tiers_separate,
        beats_probfirst=beats_probfirst,
        beats_breakeven=beats_breakeven,
        notes="",
    )

    print(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    with open("E:/PyCaLiAI/analysis/_trio_tier_marketproxy_out.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=float)
    return out


if __name__ == "__main__":
    main()
