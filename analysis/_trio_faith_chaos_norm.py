# -*- coding: utf-8 -*-
"""
_trio_faith_chaos_norm.py
=========================
FAITHFUL classifier run for the user's 4-tier 三連複 formation system,
using the PRODUCTION race_confidence() field_chaos_score (NORMALIZED
entropy / log(n)) as CLASSIFIER A.

Question: does field-size normalization (vs the canonical RAW-entropy run,
which was the user's exact objection) change the verdict that tiering LOSES
to flat prob-first top-6 (B0)?

Baselines / systems on test(verdict 2024-25) + valid(agreement 2023):
  B0  flat prob-first top-6 PL combos @ 1000 yen, ALL races
  B2  user system: classifier tier -> matched formation + tier stake
  B4  REVERSED: 固い->F1 wide/400, 大波乱->F4 narrow/2000

Formations over model_rank positions (A=1着 / B=2着 / C=3着 candidate rank-sets):
  F1 荒れ:   A={1,2,3}  B={1,2,3,4}      C={1,2,3,4,5,6,7}    ~22pts  400/combo
  F2 中荒れ: A={1,2,3}  B={1,2,3}        C={1,2,3,4,5,6,7,8}  ~16pts  600/combo
  F3 本命来そう: A={1,2} B={1,2,3}       C={1,2,3,4,5,6}      ~10pts  1000/combo
  F4 本命堅い: A={1}   B={2,3}           C={2,3,4,5}          ~5pts   2000/combo

TILT: 固い->F4 (narrow/thick 2000), 大波乱->F1 (wide/thin 400).

三連複 is order-free: a 3-ban combo {x,y,z} is BOUGHT iff its 3 model_ranks
fill (A,B,C) under SOME permutation. Combo WINS iff its ban-set == winning trio
(3 ban whose finish in {1,2,3}); returns tri_pay yen/100 * (stake/100).

METRICS: blended ROI% = sum(returns)/sum(stakes)*100 (money-weighted; per-combo
stake varies by tier). capture% = races with >=1 winning combo / races bet.
Per-race bootstrap 95% CI (carry per-race stake & return), 2000 draws,
numpy default_rng(12345).

TAKEOUT: JRA 三連複 25% -> odds-blind floor ~75%, break-even = 100%.
"""
from __future__ import annotations
import sys
import json
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "E:/PyCaLiAI")

import pl_probs as PL                       # noqa: E402
from export_marks_json import race_confidence  # noqa: E402
import joblib                                # noqa: E402

SUB = "E:/PyCaLiAI/analysis/_trio_substrate.parquet"
CAL_PATH = "E:/PyCaLiAI/models/pl_calibrators_v6.pkl"
RNG_SEED = 12345
N_BOOT = 2000

# ---------------------------------------------------------------------------
# Formation definitions over model_rank positions.
# A = 1着候補ランク集合, B = 2着候補, C = 3着候補.
# ---------------------------------------------------------------------------
FORMATIONS = {
    "F1": {"A": {1, 2, 3}, "B": {1, 2, 3, 4}, "C": {1, 2, 3, 4, 5, 6, 7}, "stake": 400},
    "F2": {"A": {1, 2, 3}, "B": {1, 2, 3}, "C": {1, 2, 3, 4, 5, 6, 7, 8}, "stake": 600},
    "F3": {"A": {1, 2}, "B": {1, 2, 3}, "C": {1, 2, 3, 4, 5, 6}, "stake": 1000},
    "F4": {"A": {1}, "B": {2, 3}, "C": {2, 3, 4, 5}, "stake": 2000},
}
# tier label -> formation (the user's system B2). 固い->F4, 荒れ->F1.
# Quartile by chaos: lowest chaos 25% = 固い = F4; highest 25% = 大波乱 = F1.
TIER_TO_FORM = {"固い": "F4", "やや固": "F3", "やや荒": "F2", "荒れ": "F1"}
# Reversed system B4: 固い->F1 wide/400, 大波乱->F4 narrow/2000.
TIER_TO_FORM_REV = {"固い": "F1", "やや固": "F2", "やや荒": "F3", "荒れ": "F4"}


def formation_combos(form: dict, ranks_present: set[int]) -> set[frozenset[int]]:
    """Set of 3-model_rank combos (as frozenset) bought by a formation,
    restricted to ranks actually present in the race. Order-free: a combo
    {r1,r2,r3} is bought iff it can fill (A,B,C) under some permutation."""
    A, B, C = form["A"], form["B"], form["C"]
    bought = set()
    # all unordered triples of distinct present ranks
    rp = sorted(r for r in ranks_present)
    for trip in itertools.combinations(rp, 3):
        for perm in itertools.permutations(trip):
            if perm[0] in A and perm[1] in B and perm[2] in C:
                bought.add(frozenset(trip))
                break
    return bought


def b0_combos_topk(pl_w: np.ndarray, model_rank_by_row: np.ndarray,
                   k: int = 6) -> set[frozenset[int]]:
    """B0 prob-first TOP-k COMBOS (matches canonical): the k highest-probability
    三連複 combos by PL probability (NOT a top-k-horse box).
    all_sanrenpuku returns {(i,j,l): prob} with 0-based row indices; rows are
    ordered by ascending model_rank, so row i -> model_rank_by_row[i].
    Returns set of frozenset(model_rank) for the top-k combos."""
    probs = PL.all_sanrenpuku(pl_w)              # {(i,j,l): prob}, 0-based rows
    if not probs:
        return set()
    items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:k]
    bought = set()
    for (i, j, l), _p in items:
        bought.add(frozenset((int(model_rank_by_row[i]),
                              int(model_rank_by_row[j]),
                              int(model_rank_by_row[l]))))
    return bought


def load_calibrator():
    o = joblib.load(CAL_PATH)
    return o["calibrators"]["tansho"]


def build_races(df: pd.DataFrame, tansho_cal):
    """One dict per race with everything needed for backtest."""
    races = []
    for rid, g in df.groupby("rid", sort=False):
        g = g.sort_values("model_rank")
        ban = g["ban"].to_numpy()
        model_rank = g["model_rank"].to_numpy()
        pl_w = g["pl_w"].to_numpy(dtype=float)
        finish = g["finish"].to_numpy()
        pop_rank = g["pop_rank"].to_numpy(dtype=float)
        split = g["split"].iloc[0]
        year = int(g["year"].iloc[0])
        tri_pay = float(g["tri_pay"].iloc[0])
        n = len(g)
        if n < 3:
            continue
        # winning trio = ban set with finish in {1,2,3}
        win_mask = (finish >= 1) & (finish <= 3)
        win_bans = ban[win_mask]
        if len(win_bans) != 3:
            # malformed result (dead heat / missing) -> drop
            continue

        # PL probability vectors (in g order = ascending model_rank)
        p_win_raw = PL.all_tansho(pl_w)             # win-prob vec
        p_plc = PL.all_fukusho(pl_w)                # place-prob vec
        # calibrated win prob (apply isotonic then renormalize)
        p_win_cal = tansho_cal.predict(p_win_raw)
        s = p_win_cal.sum()
        p_win_cal = p_win_cal / s if s > 0 else p_win_raw.copy()

        # market rank by ban (ascending ou_lastlog = most backed first)
        mkt_rank_by_ban = None
        if not np.isnan(pop_rank).any():
            mkt_rank_by_ban = pop_rank  # already a rank aligned to ban order

        ai_rank_order = model_rank  # 1 = best

        rc_raw = race_confidence(p_win_raw, p_plc, ai_rank_order, mkt_rank_by_ban)
        rc_cal = race_confidence(p_win_cal, p_plc, ai_rank_order, mkt_rank_by_ban)

        # map ban -> model_rank, winning trio as model_rank set
        ban_to_rank = {int(b): int(r) for b, r in zip(ban, model_rank)}
        win_rankset = frozenset(ban_to_rank[int(b)] for b in win_bans)
        ranks_present = set(int(r) for r in model_rank)
        ranks_sorted = sorted(ranks_present)

        races.append({
            "rid": rid,
            "split": split,
            "year": year,
            "tri_pay": tri_pay,
            "n": n,
            "pl_w": pl_w,                 # row order = ascending model_rank
            "rank_by_row": model_rank,    # row i -> model_rank
            "ranks_present": ranks_present,
            "ranks_sorted": ranks_sorted,
            "win_rankset": win_rankset,
            "chaos_raw": rc_raw["field_chaos_score"],
            "chaos_cal": rc_cal["field_chaos_score"],
        })
    return races


def eval_system(races, combo_fn, tier_of=None):
    """combo_fn(race) -> set[frozenset model_rank]. Returns per-race
    arrays of (stake, ret, won) and tier label list (if tier_of given)."""
    stakes, rets, wons, tiers = [], [], [], []
    for r in races:
        if tier_of is not None:
            tier = tier_of(r)
        else:
            tier = None
        combos, stake_per = combo_fn(r, tier)
        if not combos:
            continue
        n_combos = len(combos)
        stake = n_combos * stake_per
        won = r["win_rankset"] in combos
        ret = (r["tri_pay"] * stake_per / 100.0) if won else 0.0
        stakes.append(stake)
        rets.append(ret)
        wons.append(1 if won else 0)
        tiers.append(tier)
    return (np.array(stakes, dtype=float), np.array(rets, dtype=float),
            np.array(wons, dtype=int), tiers)


def blended_roi(stakes, rets):
    tot_s = stakes.sum()
    return (rets.sum() / tot_s * 100.0) if tot_s > 0 else float("nan")


def boot_ci(stakes, rets, n_boot=N_BOOT, seed=RNG_SEED):
    """Per-race bootstrap 95% CI on blended (money-weighted) ROI."""
    rng = np.random.default_rng(seed)
    m = len(stakes)
    if m == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, m, size=(n_boot, m))
    bs = stakes[idx].sum(axis=1)
    br = rets[idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rois = np.where(bs > 0, br / bs * 100.0, np.nan)
    rois = rois[~np.isnan(rois)]
    if len(rois) == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(rois, 2.5)), float(np.percentile(rois, 97.5)))


# ---- combo functions ----
def make_b0_fn(k=6):
    def fn(r, tier):
        return b0_combos_topk(r["pl_w"], r["rank_by_row"], k), 1000
    return fn


def make_tier_fn(tier_to_form):
    # cache formation combos per (formation, ranks_present) to speed up
    cache = {}

    def fn(r, tier):
        form_name = tier_to_form[tier]
        form = FORMATIONS[form_name]
        key = (form_name, tuple(r["ranks_sorted"]))
        combos = cache.get(key)
        if combos is None:
            combos = formation_combos(form, r["ranks_present"])
            cache[key] = combos
        return combos, form["stake"]
    return fn


def quartile_tiers(races, signal_key, edges=None):
    """Assign tier label per race by chaos quartile.
    If edges given (from valid), use fixed thresholds; else compute per-set."""
    vals = np.array([r[signal_key] for r in races], dtype=float)
    if edges is None:
        q = np.percentile(vals, [25, 50, 75])
    else:
        q = edges
    labels = []
    for v in vals:
        if v <= q[0]:
            labels.append("固い")     # lowest chaos
        elif v <= q[1]:
            labels.append("やや固")
        elif v <= q[2]:
            labels.append("やや荒")
        else:
            labels.append("荒れ")    # highest chaos
    return labels, q


def summarize(name, stakes, rets, wons, stake_per_label=None):
    roi = blended_roi(stakes, rets)
    lo, hi = boot_ci(stakes, rets)
    cap = (wons.sum() / len(wons) * 100.0) if len(wons) else float("nan")
    avg_stake = float(stakes.mean()) if len(stakes) else float("nan")
    return {
        "name": name,
        "test_roi": round(roi, 3),
        "ci_low": round(lo, 3),
        "ci_high": round(hi, 3),
        "capture_pct": round(cap, 3),
        "stake_per_race": round(avg_stake, 1),
        "n_races": int(len(stakes)),
    }


def main():
    df = pd.read_parquet(SUB)
    n_rows0 = len(df)
    cal = load_calibrator()

    print(f"[load] substrate rows={n_rows0} races={df['rid'].nunique()}", flush=True)
    races_all = build_races(df, cal)
    test = [r for r in races_all if r["split"] == "test"]
    valid = [r for r in races_all if r["split"] == "valid"]
    print(f"[build] usable races: test={len(test)} valid={len(valid)} "
          f"(dropped {df['rid'].nunique() - len(races_all)} races for n<3 / bad trio)",
          flush=True)

    # ===================== choose chaos variant =====================
    # Primary classifier uses CALIBRATED chaos (per prompt: try calibrated win).
    # We also report raw-win chaos as a sensitivity check.
    SIG = "chaos_cal"

    out = {}

    # ---- B0: flat prob-first top-6, ALL races ----
    b0_fn = make_b0_fn(6)
    s, rt, w, _ = eval_system(test, b0_fn)
    b0_test = summarize("B0_probfirst_top6", s, rt, w)
    sv, rtv, wv, _ = eval_system(valid, b0_fn)
    b0_valid_roi = blended_roi(sv, rtv)

    # ---- per-set quartile tiers (in-sample favorable) on CALIBRATED chaos ----
    test_tiers, test_edges = quartile_tiers(test, SIG)
    valid_tiers, valid_edges = quartile_tiers(valid, SIG)
    test_tier_map = {id(r): t for r, t in zip(test, test_tiers)}
    valid_tier_map = {id(r): t for r, t in zip(valid, valid_tiers)}

    def test_tier_of(r):
        return test_tier_map[id(r)]

    def valid_tier_of(r):
        return valid_tier_map[id(r)]

    # ---- B2: user system (classifier tier -> matched formation) ----
    b2_fn = make_tier_fn(TIER_TO_FORM)
    s2, rt2, w2, t2 = eval_system(test, b2_fn, tier_of=test_tier_of)
    b2_test = summarize("B2_user_system", s2, rt2, w2)
    s2v, rt2v, w2v, t2v = eval_system(valid, b2_fn, tier_of=valid_tier_of)
    b2_valid_roi = blended_roi(s2v, rt2v)

    # ---- B4: reversed ----
    b4_fn = make_tier_fn(TIER_TO_FORM_REV)
    s4, rt4, w4, t4 = eval_system(test, b4_fn, tier_of=test_tier_of)
    b4_test = summarize("B4_reversed", s4, rt4, w4)
    s4v, rt4v, w4v, _ = eval_system(valid, b4_fn, tier_of=valid_tier_of)
    b4_valid_roi = blended_roi(s4v, rt4v)

    # ---- per-tier breakdown for B2 (test) ----
    per_tier = []
    tarr = np.array(t2)
    for tier_label, form_name in TIER_TO_FORM.items():
        mask = tarr == tier_label
        st, rtt, wt = s2[mask], rt2[mask], w2[mask]
        if len(st) == 0:
            continue
        roi = blended_roi(st, rtt)
        lo, hi = boot_ci(st, rtt)
        cap = wt.sum() / len(wt) * 100.0
        avg_pts = float(st.mean() / FORMATIONS[form_name]["stake"])
        per_tier.append({
            "tier": f"{tier_label}->{form_name}",
            "n_races_test": int(len(st)),
            "avg_points": round(avg_pts, 2),
            "stake_per_combo": FORMATIONS[form_name]["stake"],
            "capture_pct": round(cap, 3),
            "roi_pct": round(roi, 3),
            "roi_ci_low": round(lo, 3),
            "roi_ci_high": round(hi, 3),
        })

    # tiers_separate: 固いF4 ROI > 荒れF1 ROI with non-overlapping CIs
    f4 = next((x for x in per_tier if x["tier"].startswith("固い")), None)
    f1 = next((x for x in per_tier if x["tier"].startswith("荒れ")), None)
    tiers_separate = False
    if f4 and f1:
        tiers_separate = (f4["roi_pct"] > f1["roi_pct"]
                          and f4["roi_ci_low"] > f1["roi_ci_high"])

    # ---- era split for B2 (test 2024 / 2025) ----
    def era_roi(yr):
        idx = [i for i, r in enumerate(test) if r["year"] == yr]
        # rebuild stake/ret aligned to eval order (eval skips none for B2 since
        # every race gets a tier+combos), so order matches `test` order.
        return blended_roi(s2[idx], rt2[idx]) if idx else float("nan")
    # NOTE eval_system iterates races in order and B2 never skips -> alignment ok
    era_2024 = era_roi(2024)
    era_2025 = era_roi(2025)

    # ---- fixed-threshold: valid-derived chaos cutoffs applied to test ----
    fixed_tiers, _ = quartile_tiers(test, SIG, edges=valid_edges)
    fixed_map = {id(r): t for r, t in zip(test, fixed_tiers)}

    def fixed_tier_of(r):
        return fixed_map[id(r)]
    sF, rtF, wF, tF = eval_system(test, b2_fn, tier_of=fixed_tier_of)
    fixed_roi = blended_roi(sF, rtF)

    # ---- sensitivity: RAW-win chaos B2 (to compare normalization-of-prob) ----
    raw_tiers, _ = quartile_tiers(test, "chaos_raw")
    raw_map = {id(r): t for r, t in zip(test, raw_tiers)}
    sR, rtR, wR, _ = eval_system(test, b2_fn, tier_of=lambda r: raw_map[id(r)])
    raw_chaos_roi = blended_roi(sR, rtR)

    # ===================== assemble + verdicts =====================
    vs_probfirst = b2_test["test_roi"] - b0_test["test_roi"]
    vs_reversed = b2_test["test_roi"] - b4_test["test_roi"]

    # beats_probfirst: B2 > B0 clearly (CIs not fully overlapping) AND valid agrees
    cis_overlap_fully = (b2_test["ci_low"] <= b0_test["ci_high"]
                         and b0_test["ci_low"] <= b2_test["ci_high"])
    # "clearly" => B2 ci_low > B0 ci_high (B2 strictly above)
    b2_strictly_above = b2_test["ci_low"] > b0_test["ci_high"]
    valid_agrees = b2_valid_roi > b0_valid_roi
    beats_probfirst = bool(b2_strictly_above and valid_agrees)
    beats_breakeven = bool(b2_test["ci_low"] > 100.0)

    result = {
        "classifier": "field_chaos_score (NORMALIZED entropy/log(n)) from production race_confidence; per-split chaos quartiles; primary on CALIBRATED p_win (tansho isotonic v6, renormalized)",
        "p_win_basis": "CALIBRATED (models/pl_calibrators_v6.pkl ['calibrators']['tansho'] isotonic .predict applied to all_tansho(pl_w), then renormalized). Raw-win chaos also tested as sensitivity.",
        "baselines": [
            {**b0_test, "valid_roi": round(b0_valid_roi, 3)},
            {**b2_test, "valid_roi": round(b2_valid_roi, 3)},
            {**b4_test, "valid_roi": round(b4_valid_roi, 3)},
        ],
        "per_tier": per_tier,
        "tiered_system": {
            "test_roi": b2_test["test_roi"],
            "ci_low": b2_test["ci_low"],
            "ci_high": b2_test["ci_high"],
            "valid_roi": round(b2_valid_roi, 3),
            "stake_per_race": b2_test["stake_per_race"],
            "era_2024_roi": round(era_2024, 3),
            "era_2025_roi": round(era_2025, 3),
            "fixed_threshold_test_roi": round(fixed_roi, 3),
        },
        "vs_probfirst_delta_pt": round(vs_probfirst, 3),
        "vs_reversed_delta_pt": round(vs_reversed, 3),
        "tiers_separate": bool(tiers_separate),
        "beats_probfirst": beats_probfirst,
        "beats_breakeven": beats_breakeven,
        "notes": (
            f"B0 valid={b0_valid_roi:.2f}; B2 valid={b2_valid_roi:.2f}; "
            f"raw-win-chaos B2 test ROI={raw_chaos_roi:.2f} (sensitivity); "
            f"fixed(valid-edges) B2={fixed_roi:.2f}. "
            f"Per-split percentile is mildly in-sample favorable (reported alongside "
            f"valid-derived fixed-threshold). "
            f"Dropped {df['rid'].nunique()-len(races_all)} races (n<3 or bad trio). "
            f"pop_rank NaN -> market_agreement skipped for those races (chaos unaffected)."
        ),
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    Path("E:/PyCaLiAI/analysis/_trio_faith_chaos_norm_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    main()
