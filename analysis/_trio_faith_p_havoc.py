# -*- coding: utf-8 -*-
"""
_trio_faith_p_havoc.py
======================
Faithful test of the USER'S 4-tier 三連複 formation system, classified by the
SYSTEM'S OWN classifiers:
  (A) race_confidence (production) field_chaos_score percentile -> tier
  (D) p_havoc from the dedicated upset model (havoc_v1) -> real thresholds.

Backtest 三連複 buys. Compare:
  B0 = flat prob-first top-6 PL combos @¥1000 (ALL races)
  B2 = user system: classifier tier -> matched formation+stake
  B4 = REVERSED control (固い->F1 wide/¥400 ... 大波乱->F4 narrow/¥2000)

Metrics: money-weighted blended ROI%, capture%, per-race bootstrap 95% CI.
Verdict on test(2024-25); valid(2023) = agreement.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path("E:/PyCaLiAI")
sys.path.insert(0, str(BASE))

import pl_probs as PL
import joblib
import bundle_signals
from export_marks_json import race_confidence

RNG = np.random.default_rng(12345)
SUB = BASE / "analysis/_trio_substrate.parquet"
MASTER = BASE / "data/master_v2_20130105-20251228.csv"
CAL_PKL = BASE / "models/pl_calibrators_v6.pkl"

# -----------------------------------------------------------------
# Formation definitions: rank-sets (1=best=model_rank). order-free trio.
# A combo {x,y,z} (3 distinct model_ranks) is BOUGHT iff there is a permutation
# assigning the three ranks into (A,B,C) sets.
# -----------------------------------------------------------------
FORMS = {
    "F1": {"A": {1,2,3}, "B": {1,2,3,4},   "C": set(range(1,8)),  "stake": 400,  "label": "荒れ"},
    "F2": {"A": {1,2,3}, "B": {1,2,3},     "C": set(range(1,9)),  "stake": 600,  "label": "中荒れ"},
    "F3": {"A": {1,2},   "B": {1,2,3},     "C": set(range(1,7)),  "stake": 1000, "label": "本命来そう"},
    "F4": {"A": {1},     "B": {2,3},       "C": {2,3,4,5},        "stake": 2000, "label": "本命堅い"},
}
# classifier-class -> formation
CLASS_TO_FORM   = {"固い":"F4", "中庸":"F3", "波乱":"F2", "大波乱":"F1"}      # user system (B2)
CLASS_TO_FORM_R = {"固い":"F1", "中庸":"F2", "波乱":"F3", "大波乱":"F4"}      # reversed (B4)


def can_match_permutation(ranks3, form):
    """ranks3 = the 3 model_ranks (set of 3 ints). Return True if some assignment
    of the 3 ranks to (A,B,C) sets is feasible (each rank to a distinct slot)."""
    A, B, C = form["A"], form["B"], form["C"]
    rs = list(ranks3)
    import itertools
    for perm in itertools.permutations(rs):
        if perm[0] in A and perm[1] in B and perm[2] in C:
            return True
    return False


def enumerate_form_combos(rank_to_ban, form):
    """Return set of frozenset(ban-triples) bought under `form` given the race's
    rank->ban mapping. Iterate over candidate rank triples (all in A|B|C up to max)."""
    A, B, C = form["A"], form["B"], form["C"]
    cand_ranks = sorted((A | B | C) & set(rank_to_ban.keys()))
    bought = set()
    import itertools
    for tri in itertools.combinations(cand_ranks, 3):
        if can_match_permutation(set(tri), form):
            bought.add(frozenset(rank_to_ban[r] for r in tri))
    return bought


# -----------------------------------------------------------------
# Load substrate
# -----------------------------------------------------------------
sub = pd.read_parquet(SUB)
sub["rid"] = sub["rid"].astype(str)

# Build per-race structures.
print("[load] substrate races:", sub["rid"].nunique(), file=sys.stderr)

# Load calibrator (tansho isotonic) for calibrated p_win variant.
cal = joblib.load(CAL_PKL)
iso_tansho = cal["calibrators"]["tansho"]


def calibrate_pwin(p_win):
    pc = iso_tansho.predict(np.clip(p_win, 0, None))
    s = pc.sum()
    if s <= 0:
        return p_win
    return pc / s


# -----------------------------------------------------------------
# p_havoc: build df from master_v2 for substrate rids, merge _v6_score from substrate
# -----------------------------------------------------------------
print("[load] master_v2 (filtered to substrate rids)...", file=sys.stderr)
need_cols = ["レースID(新/馬番無)", "馬番", "場所", "芝・ダ", "距離", "馬場状態",
             "天気", "クラス名", "コース区分", "年齢限定", "限定", "性別限定", "重量種別"]
sub_rids = set(sub["rid"].unique())
chunks = []
for ch in pd.read_csv(MASTER, usecols=need_cols, encoding="utf-8",
                      dtype={"レースID(新/馬番無)": str}, chunksize=200000):
    ch["_rid16"] = ch["レースID(新/馬番無)"].astype(str).str[:16]
    ch = ch[ch["_rid16"].isin(sub_rids)]
    if len(ch):
        chunks.append(ch)
mdf = pd.concat(chunks, ignore_index=True)
# use 16-digit rid as the grouping COL_RID so it matches substrate rid
mdf["レースID(新/馬番無)"] = mdf["_rid16"]
mdf["馬番"] = pd.to_numeric(mdf["馬番"], errors="coerce").astype("Int64")
# merge v6 score from substrate
score_map = sub.set_index(["rid", "ban"])["score"].to_dict()
mdf["_v6_score"] = [score_map.get((r, int(b)) if pd.notna(b) else None, np.nan)
                    for r, b in zip(mdf["レースID(新/馬番無)"], mdf["馬番"])]
before = len(mdf)
mdf = mdf[mdf["_v6_score"].notna()].copy()
print(f"[havoc] master rows with v6 score: {len(mdf)}/{before}", file=sys.stderr)

print("[havoc] predicting p_havoc...", file=sys.stderr)
p_havoc_map = bundle_signals.predict_p_havoc(mdf, "_v6_score")
print(f"[havoc] races scored: {len(p_havoc_map)}", file=sys.stderr)


# -----------------------------------------------------------------
# Per-race precompute: rank_to_ban, ban_to_idx, winning trio, tri_pay, split,
# year, classifier raw signals.
# -----------------------------------------------------------------
races = {}  # rid -> dict
drop_no_havoc = 0
drop_small = 0
for rid, g in sub.groupby("rid", sort=False):
    g = g.sort_values("model_rank").reset_index(drop=True)
    n = len(g)
    if n < 3:
        drop_small += 1
        continue
    bans = g["ban"].astype(int).values
    w = g["pl_w"].values.astype(float)
    # PL probs over the race (index order = sorted by model_rank)
    p_win = PL.all_tansho(w)
    p_plc = PL.all_fukusho(w)
    sanren = PL.all_sanrenpuku(w)  # keys are (i,j,k) idx into this sorted order
    finish = g["finish"].values
    # winning trio = set of ban with finish in {1,2,3}
    win_mask = pd.to_numeric(pd.Series(finish), errors="coerce").isin([1, 2, 3]).values
    win_bans = frozenset(int(b) for b, m in zip(bans, win_mask) if m)
    rank_to_ban = {int(r): int(b) for r, b in zip(g["model_rank"], bans)}
    ban_to_idx = {int(b): i for i, b in enumerate(bans)}
    # market rank by ban (1=most backed = lowest ou_lastlog). NaN-> skip agreement
    pop = g["pop_rank"].values
    if np.isnan(pop).any():
        market_rank_by_ban = None
    else:
        market_rank_by_ban = pop  # already a rank
    # classifier C: race_confidence on raw + calibrated p_win
    ai_rank_order = g["model_rank"].values
    rc_raw = race_confidence(p_win, p_plc, ai_rank_order, market_rank_by_ban)
    p_win_cal = calibrate_pwin(p_win)
    rc_cal = race_confidence(p_win_cal, p_plc, ai_rank_order, market_rank_by_ban)
    ph = p_havoc_map.get(rid, np.nan)
    if np.isnan(ph):
        drop_no_havoc += 1
    races[rid] = dict(
        split=g["split"].iloc[0], year=int(g["year"].iloc[0]),
        n=n, rank_to_ban=rank_to_ban, ban_to_idx=ban_to_idx,
        win_bans=win_bans, tri_pay=float(g["tri_pay"].iloc[0]),
        sanren=sanren, p_havoc=ph,
        chaos_raw=rc_raw["field_chaos_score"], chaos_cal=rc_cal["field_chaos_score"],
        has_win=(len(win_bans) == 3),
    )

print(f"[races] usable {len(races)}; drop_small {drop_small}; no_havoc {drop_no_havoc}", file=sys.stderr)

# Some races may have winning trio not size 3 (DQ/scratch) — exclude from scoring
# but keep for diagnostics. We require has_win to settle returns.
bad_win = sum(1 for r in races.values() if not r["has_win"])
print(f"[races] with valid 3-horse winning trio: {sum(r['has_win'] for r in races.values())} (bad {bad_win})", file=sys.stderr)


# need global idx_to_ban per race for topk; refactor: store maps
def settle(bought_combos, race):
    """Given set of frozenset(ban-triples) bought at a per-combo stake, return
    (stake_total, return_total, hit) for this race. stake handled by caller."""
    win = race["win_bans"]
    hit = win in bought_combos
    return hit


# -----------------------------------------------------------------
# Strategy evaluators -> per-race (stake, ret)
# -----------------------------------------------------------------
def eval_B0(race):
    """flat prob-first top-6 PL combos @¥1000."""
    rtb = race["rank_to_ban"]
    bidx = race["ban_to_idx"]
    sanren = race["sanren"]
    ranks = sorted(rtb.keys())[:6]
    idxs = [bidx[rtb[r]] for r in ranks]
    import itertools
    scored = []
    for tri in itertools.combinations(idxs, 3):
        key = tuple(sorted(tri))
        scored.append((sanren.get(key, 0.0), key))
    scored.sort(reverse=True)
    idx_to_ban = {i: b for b, i in bidx.items()}
    bought = set(frozenset(idx_to_ban[i] for i in key) for _, key in scored[:6])
    stake = 1000 * len(bought)
    ret = 0.0
    if race["win_bans"] in bought:
        ret = race["tri_pay"] * (1000 / 100.0)
    return stake, ret


def eval_form(race, form):
    rtb = race["rank_to_ban"]
    bought = enumerate_form_combos(rtb, form)
    stake = form["stake"] * len(bought)
    ret = 0.0
    if race["win_bans"] in bought:
        ret = race["tri_pay"] * (form["stake"] / 100.0)
    return stake, ret, len(bought)


def tier_of(race, classifier, mapping):
    if classifier == "havoc":
        ph = race["p_havoc"]
        if np.isnan(ph):
            return None
        cls = bundle_signals.havoc_class(ph)
        return mapping[cls]
    # percentile classifiers handled separately (need set-wide percentile)
    raise ValueError


# -----------------------------------------------------------------
# Build per-split race lists with valid winning trio
# -----------------------------------------------------------------
def split_races(split):
    return [rid for rid, r in races.items() if r["split"] == split and r["has_win"]]


# -----------------------------------------------------------------
# Percentile classifier (race_confidence field_chaos). Low chaos = 固い.
# Map chaos percentile to class: <0.25 固い, <0.5 中庸, <0.75 波乱, else 大波乱.
# Two variants: per-split percentile (in-sample favorable) and valid->test fixed.
# -----------------------------------------------------------------
def chaos_class_from_pct(pct):
    if pct < 0.25: return "固い"
    if pct < 0.50: return "中庸"
    if pct < 0.75: return "波乱"
    return "大波乱"


def build_chaos_tiers(chaos_key):
    """Return dict rid->class for both per-split and fixed(valid->test)."""
    test_ids = split_races("test")
    valid_ids = split_races("valid")
    per_split = {}
    for ids in (test_ids, valid_ids):
        vals = np.array([races[r][chaos_key] for r in ids])
        order = vals.argsort()
        ranks = np.empty(len(vals)); ranks[order] = np.arange(len(vals))
        pct = ranks / max(len(vals) - 1, 1)
        for r, p in zip(ids, pct):
            per_split[r] = chaos_class_from_pct(p)
    # fixed: derive cut points from valid distribution, apply to test
    vvals = np.array([races[r][chaos_key] for r in valid_ids])
    cuts = np.quantile(vvals, [0.25, 0.50, 0.75])
    def fixed_class(v):
        if v < cuts[0]: return "固い"
        if v < cuts[1]: return "中庸"
        if v < cuts[2]: return "波乱"
        return "大波乱"
    fixed = {r: fixed_class(races[r][chaos_key]) for r in test_ids}
    # valid fixed = its own (== per_split valid)
    for r in valid_ids:
        fixed[r] = per_split[r]
    return per_split, fixed, cuts


# -----------------------------------------------------------------
# Bootstrap CI on money-weighted ROI given per-race (stake, ret) arrays
# -----------------------------------------------------------------
def boot_ci(stakes, rets, n=2000):
    stakes = np.asarray(stakes, float); rets = np.asarray(rets, float)
    m = len(stakes)
    if m == 0 or stakes.sum() == 0:
        return (float("nan"), float("nan"))
    idx = RNG.integers(0, m, size=(n, m))
    bs = stakes[idx].sum(axis=1)
    br = rets[idx].sum(axis=1)
    roi = np.where(bs > 0, br / bs * 100.0, np.nan)
    return float(np.nanpercentile(roi, 2.5)), float(np.nanpercentile(roi, 97.5))


def roi_of(stakes, rets):
    s = np.sum(stakes)
    return (np.sum(rets) / s * 100.0) if s > 0 else float("nan")


def capture_of(rets):
    rets = np.asarray(rets)
    bet = rets.size  # all races here have stake>0
    return float((rets > 0).sum() / bet * 100.0) if bet else float("nan")


# -----------------------------------------------------------------
# Run a strategy that maps rid->form (per-race formation) over a race id list
# -----------------------------------------------------------------
def run_form_strategy(ids, rid_to_form):
    stakes, rets, npts = [], [], []
    for rid in ids:
        race = races[rid]
        form = FORMS[rid_to_form[rid]]
        s, r, npt = eval_form(race, form)
        if s == 0:
            continue
        stakes.append(s); rets.append(r); npts.append(npt)
    return np.array(stakes), np.array(rets), np.array(npts)


def run_B0(ids):
    stakes, rets = [], []
    for rid in ids:
        s, r = eval_B0(races[rid])
        if s == 0:
            continue
        stakes.append(s); rets.append(r)
    return np.array(stakes), np.array(rets)


# =================================================================
# MAIN: run for each classifier
# =================================================================
results = {}

# B0 baseline (both splits)
for split in ("test", "valid"):
    ids = split_races(split)
    s, r = run_B0(ids)
    results[f"B0_{split}"] = dict(roi=roi_of(s, r), ci=boot_ci(s, r),
                                  capture=capture_of(r), n=len(ids),
                                  stake_per_race=float(s.mean()))

# --- Classifier C: race_confidence chaos (raw + calibrated) ---
for chaos_key, cname in (("chaos_raw", "C_raw"), ("chaos_cal", "C_cal")):
    per_split, fixed, cuts = build_chaos_tiers(chaos_key)
    test_ids = split_races("test"); valid_ids = split_races("valid")
    # B2 user-system, per-split percentile
    rid2form = {r: CLASS_TO_FORM[per_split[r]] for r in test_ids}
    s2, r2, n2 = run_form_strategy(test_ids, rid2form)
    rid2form_v = {r: CLASS_TO_FORM[per_split[r]] for r in valid_ids}
    s2v, r2v, n2v = run_form_strategy(valid_ids, rid2form_v)
    # B4 reversed
    rid4form = {r: CLASS_TO_FORM_R[per_split[r]] for r in test_ids}
    s4, r4, n4 = run_form_strategy(test_ids, rid4form)
    # fixed threshold (valid->test)
    rid2form_fx = {r: CLASS_TO_FORM[fixed[r]] for r in test_ids}
    sfx, rfx, _ = run_form_strategy(test_ids, rid2form_fx)
    # per-tier breakdown (test, per-split)
    per_tier = {}
    for cls, fkey in CLASS_TO_FORM.items():
        tids = [r for r in test_ids if per_split[r] == cls]
        st, rt, nt = run_form_strategy(tids, {r: fkey for r in tids})
        per_tier[cls] = dict(form=fkey, n=len(tids), roi=roi_of(st, rt),
                             ci=boot_ci(st, rt), capture=capture_of(rt),
                             avg_pts=float(nt.mean()) if len(nt) else 0.0,
                             stake_per_combo=FORMS[fkey]["stake"])
    # era split for B2 test
    era = {}
    for yr in (2024, 2025):
        eids = [r for r in test_ids if races[r]["year"] == yr]
        se, re, _ = run_form_strategy(eids, {r: CLASS_TO_FORM[per_split[r]] for r in eids})
        era[yr] = roi_of(se, re)
    results[cname] = dict(
        B2_test=dict(roi=roi_of(s2, r2), ci=boot_ci(s2, r2), capture=capture_of(r2),
                     n=len(test_ids), stake_per_race=float(s2.mean())),
        B2_valid=dict(roi=roi_of(s2v, r2v), ci=boot_ci(s2v, r2v), capture=capture_of(r2v),
                      n=len(valid_ids), stake_per_race=float(s2v.mean())),
        B4_test=dict(roi=roi_of(s4, r4), ci=boot_ci(s4, r4), capture=capture_of(r4)),
        fixed_test_roi=roi_of(sfx, rfx),
        era=era, per_tier=per_tier, cuts=list(map(float, cuts)),
    )

# --- Classifier D: p_havoc (dedicated upset model) ---
test_ids = [r for r in split_races("test") if not np.isnan(races[r]["p_havoc"])]
valid_ids = [r for r in split_races("valid") if not np.isnan(races[r]["p_havoc"])]
# bucket counts
bucket_counts = {"test": {}, "valid": {}}
for split, ids in (("test", test_ids), ("valid", valid_ids)):
    for rid in ids:
        cls = bundle_signals.havoc_class(races[rid]["p_havoc"])
        bucket_counts[split][cls] = bucket_counts[split].get(cls, 0) + 1

rid2form = {r: CLASS_TO_FORM[bundle_signals.havoc_class(races[r]["p_havoc"])] for r in test_ids}
s2, r2, n2 = run_form_strategy(test_ids, rid2form)
rid2form_v = {r: CLASS_TO_FORM[bundle_signals.havoc_class(races[r]["p_havoc"])] for r in valid_ids}
s2v, r2v, n2v = run_form_strategy(valid_ids, rid2form_v)
rid4form = {r: CLASS_TO_FORM_R[bundle_signals.havoc_class(races[r]["p_havoc"])] for r in test_ids}
s4, r4, n4 = run_form_strategy(test_ids, rid4form)
# B0 restricted to havoc-scored races (apples-to-apples for delta)
s0, r0 = run_B0(test_ids)
# p_havoc thresholds are model-fixed => fixed_test == per-split (no percentile)
per_tier = {}
for cls, fkey in CLASS_TO_FORM.items():
    tids = [r for r in test_ids if bundle_signals.havoc_class(races[r]["p_havoc"]) == cls]
    st, rt, nt = run_form_strategy(tids, {r: fkey for r in tids})
    per_tier[cls] = dict(form=fkey, n=len(tids), roi=roi_of(st, rt),
                         ci=boot_ci(st, rt), capture=capture_of(rt),
                         avg_pts=float(nt.mean()) if len(nt) else 0.0,
                         stake_per_combo=FORMS[fkey]["stake"])
era = {}
for yr in (2024, 2025):
    eids = [r for r in test_ids if races[r]["year"] == yr]
    se, re, _ = run_form_strategy(eids, {r: CLASS_TO_FORM[bundle_signals.havoc_class(races[r]["p_havoc"])] for r in eids})
    era[yr] = roi_of(se, re)

results["D_havoc"] = dict(
    B0_test=dict(roi=roi_of(s0, r0), ci=boot_ci(s0, r0), capture=capture_of(r0)),
    B2_test=dict(roi=roi_of(s2, r2), ci=boot_ci(s2, r2), capture=capture_of(r2),
                 n=len(test_ids), stake_per_race=float(s2.mean())),
    B2_valid=dict(roi=roi_of(s2v, r2v), ci=boot_ci(s2v, r2v), capture=capture_of(r2v),
                  n=len(valid_ids), stake_per_race=float(s2v.mean())),
    B4_test=dict(roi=roi_of(s4, r4), ci=boot_ci(s4, r4), capture=capture_of(r4)),
    fixed_test_roi=roi_of(s2, r2),  # model thresholds are already fixed/leak-safe
    era=era, per_tier=per_tier, bucket_counts=bucket_counts,
)

print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
