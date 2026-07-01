"""
_trio_faith_verify.py
Independent re-run of the FAITHFUL-classifier verification of the user's 4-tier
三連複 formation system, using the PRODUCTION race_confidence() and CALIBRATED p_win.

Adversarial default: is_real_edge = False. We try to refute B2 (tiered, composite
固さ classifier) vs B0 (flat prob-first top-6 @1000) and B4 (reversed control).
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, "E:/PyCaLiAI")
import pl_probs
from export_marks_json import race_confidence

RNG = np.random.default_rng(12345)
SUB = "E:/PyCaLiAI/analysis/_trio_substrate.parquet"
CAL = "E:/PyCaLiAI/models/pl_calibrators_v6.pkl"

# tier formations over model_rank (1=best). (A=1着 set, B=2着 set, C=3着 set), stake/combo
FORMS = {
    "F1": (set([1, 2, 3]), set([1, 2, 3, 4]), set(range(1, 8)), 400),
    "F2": (set([1, 2, 3]), set([1, 2, 3]), set(range(1, 9)), 600),
    "F3": (set([1, 2]), set([1, 2, 3]), set(range(1, 7)), 1000),
    "F4": (set([1]), set([2, 3]), set([2, 3, 4, 5]), 2000),
}

# ----- load -----
df = pd.read_parquet(SUB)
calib = joblib.load(CAL)["calibrators"]
tansho_cal = calib["tansho"]


def build_race(g):
    """Return per-race dict with calibrated p_win vec, place vec, ranks, market ranks,
    winning trio (set of bans), tri_pay, n_horses, and combos buyable per tier."""
    g = g.sort_values("model_rank")
    bans = g["ban"].to_numpy()
    w = g["pl_w"].to_numpy(dtype=float)
    ranks = g["model_rank"].to_numpy()
    # calibrated p_win
    p_raw = pl_probs.all_tansho(w)
    p_cal = tansho_cal.predict(p_raw)
    s = p_cal.sum()
    p_cal = p_cal / s if s > 0 else p_raw
    p_plc = pl_probs.all_fukusho(w)
    # market rank by ban (1=most backed = lowest ou_lastlog)
    pop = g["pop_rank"].to_numpy()
    if np.isnan(pop).any():
        market_rank = None
    else:
        market_rank = pop  # already a rank
    return {
        "bans": bans,
        "ranks": ranks,
        "p_cal": p_cal,
        "p_plc": p_plc,
        "market_rank": market_rank,
        "tri_pay": float(g["tri_pay"].iloc[0]),
        "n_horses": int(g["n_horses"].iloc[0]),
        "win_set": frozenset(g.loc[g["finish"].isin([1, 2, 3]), "ban"].tolist()),
        "g": g,
    }


def conf(r):
    return race_confidence(r["p_cal"], r["p_plc"], r["ranks"], r["market_rank"])


def tier_combos(r, tier):
    """Set of buyable 3-ban frozensets for given tier formation, using model_rank slots."""
    A, B, C, _ = FORMS[tier]
    rank_to_ban = dict(zip(r["ranks"], r["bans"]))
    # candidate bans per slot
    out = set()
    avail = set(rank_to_ban.keys())
    Aset = A & avail
    Bset = B & avail
    Cset = C & avail
    for a in Aset:
        for b in Bset:
            if b == a:
                continue
            for c in Cset:
                if c == a or c == b:
                    continue
                combo = frozenset([rank_to_ban[a], rank_to_ban[b], rank_to_ban[c]])
                out.add(combo)
    return out


def b0_combos(r):
    """flat prob-first: top-6 PL combos by all_sanrenpuku prob, @1000 each."""
    w = r["g"]["pl_w"].to_numpy(dtype=float)
    bans = r["bans"]
    sp = pl_probs.all_sanrenpuku(w)  # {(i,j,k):prob} index-based on sorted order? -> index into w order
    # w here is in model_rank order (g sorted). all_sanrenpuku uses positional indices.
    items = sorted(sp.items(), key=lambda kv: kv[1], reverse=True)[:6]
    out = []
    for (i, j, k), _ in items:
        out.append(frozenset([bans[i], bans[j], bans[k]]))
    return out


# ----- precompute per race -----
def collect(split):
    races = []
    for rid, g in df[df["split"] == split].groupby("rid"):
        if len(g) < 3:
            continue
        r = build_race(g)
        if r["tri_pay"] <= 0 or not r["win_set"]:
            continue
        r["rid"] = rid
        r["conf"] = conf(r)
        races.append(r)
    return races


valid_races = collect("valid")
test_races = collect("test")
print(f"valid races {len(valid_races)}  test races {len(test_races)}")


def composite_score(c):
    """mean of [(1-chaos), top1_dom, top2_conc, market_agree]; drop agree if None.
    Note: percentile transform applied later; here we return the raw gauges as a tuple."""
    return c


# Build a DataFrame of gauges per race for percentile transforms
def gauges_df(races):
    rows = []
    for r in races:
        c = r["conf"]
        rows.append({
            "rid": r["rid"],
            "chaos": c["field_chaos_score"],
            "top1": c["top1_dominance"],
            "top2": c["top2_concentration"],
            "agree": c["ai_market_agreement"],
        })
    return pd.DataFrame(rows).set_index("rid")


def composite_from_pct(gd, ref=None):
    """Percentile-transform each gauge (vs ref df, or self) then composite mean.
    (1-chaos_pct), top1_pct, top2_pct, agree_pct (drop agree where NaN)."""
    if ref is None:
        ref = gd
    def pct(col, vals):
        rv = ref[col].dropna().to_numpy()
        # percentile rank of each val within rv
        rv_sorted = np.sort(rv)
        return np.searchsorted(rv_sorted, vals, side="right") / max(len(rv_sorted), 1)
    chaos_pct = pct("chaos", gd["chaos"].to_numpy())
    top1_pct = pct("top1", gd["top1"].to_numpy())
    top2_pct = pct("top2", gd["top2"].to_numpy())
    agree_vals = gd["agree"].to_numpy()
    agree_pct = pct("agree", np.where(np.isnan(agree_vals), 0.0, agree_vals))
    comp = np.zeros(len(gd))
    for i in range(len(gd)):
        parts = [1 - chaos_pct[i], top1_pct[i], top2_pct[i]]
        if not np.isnan(agree_vals[i]):
            parts.append(agree_pct[i])
        comp[i] = np.mean(parts)
    return pd.Series(comp, index=gd.index)


def assign_tier(comp_series):
    """Highest composite quartile -> F4 (固い), lowest -> F1 (荒れ)."""
    q = comp_series.rank(pct=True)
    tier = pd.Series(index=comp_series.index, dtype=object)
    tier[q <= 0.25] = "F1"
    tier[(q > 0.25) & (q <= 0.50)] = "F2"
    tier[(q > 0.50) & (q <= 0.75)] = "F3"
    tier[q > 0.75] = "F4"
    return tier


REVERSED = {"F1": "F4", "F2": "F3", "F3": "F2", "F4": "F1"}


def race_stake_return(r, tier):
    A, B, C, stake = FORMS[tier]
    combos = tier_combos(r, tier)
    n = len(combos)
    total_stake = n * stake
    ret = 0.0
    if r["win_set"] in combos:
        ret = r["tri_pay"] * (stake / 100.0)
    return total_stake, ret, n


def b0_stake_return(r):
    combos = b0_combos(r)
    total_stake = len(combos) * 1000
    ret = 0.0
    if r["win_set"] in combos:
        ret = r["tri_pay"] * (1000 / 100.0)
    return total_stake, ret


def blended_roi(stakes, returns):
    ts = np.sum(stakes)
    return 100.0 * np.sum(returns) / ts if ts > 0 else 0.0


def boot_ci_paired(deltas_stake_a, deltas_ret_a, stake_b, ret_b, n=2000):
    """Paired per-race bootstrap of (ROI_a - ROI_b) delta. Each race carries
    (stake_a, ret_a, stake_b, ret_b). Resample races."""
    sa = np.asarray(deltas_stake_a, float)
    ra = np.asarray(deltas_ret_a, float)
    sb = np.asarray(stake_b, float)
    rb = np.asarray(ret_b, float)
    m = len(sa)
    out = np.empty(n)
    for d in range(n):
        idx = RNG.integers(0, m, m)
        roia = 100.0 * ra[idx].sum() / sa[idx].sum() if sa[idx].sum() > 0 else 0.0
        roib = 100.0 * rb[idx].sum() / sb[idx].sum() if sb[idx].sum() > 0 else 0.0
        out[d] = roia - roib
    return np.percentile(out, [2.5, 97.5])


def boot_ci_roi(stakes, returns, n=2000):
    s = np.asarray(stakes, float)
    r = np.asarray(returns, float)
    m = len(s)
    out = np.empty(n)
    for d in range(n):
        idx = RNG.integers(0, m, m)
        ts = s[idx].sum()
        out[d] = 100.0 * r[idx].sum() / ts if ts > 0 else 0.0
    return np.percentile(out, [2.5, 97.5])


# ---------- PRIMARY: per-split percentile ----------
gd_test = gauges_df(test_races)
gd_valid = gauges_df(valid_races)
comp_test_self = composite_from_pct(gd_test)
tier_test_self = assign_tier(comp_test_self)

# leak-safe fixed: derive composite percentiles & quartile cuts on valid, apply to test
comp_valid = composite_from_pct(gd_valid)
# fixed cuts: percentile-transform test gauges using VALID as reference, then quartile
# cuts taken from valid composite distribution
comp_test_fixed = composite_from_pct(gd_test, ref=gd_valid)
vq = comp_valid.quantile([0.25, 0.5, 0.75]).to_numpy()
def assign_tier_fixed(comp_series, cuts):
    tier = pd.Series(index=comp_series.index, dtype=object)
    tier[comp_series <= cuts[0]] = "F1"
    tier[(comp_series > cuts[0]) & (comp_series <= cuts[1])] = "F2"
    tier[(comp_series > cuts[1]) & (comp_series <= cuts[2])] = "F3"
    tier[comp_series > cuts[2]] = "F4"
    return tier
tier_test_fixed = assign_tier_fixed(comp_test_fixed, vq)


def run_system(races, tier_map, reversed_=False):
    """Return per-race stakes, returns arrays + per-tier breakdown."""
    stakes, returns = [], []
    per_tier = {t: {"stk": [], "ret": [], "n": [], "rid": []} for t in FORMS}
    for r in races:
        t = tier_map.get(r["rid"])
        if t is None:
            continue
        if reversed_:
            t = REVERSED[t]
        stk, ret, ncombo = race_stake_return(r, t)
        stakes.append(stk)
        returns.append(ret)
        per_tier[t]["stk"].append(stk)
        per_tier[t]["ret"].append(ret)
        per_tier[t]["n"].append(ncombo)
        per_tier[t]["rid"].append(r["rid"])
    return np.array(stakes), np.array(returns), per_tier


# B2 = tiered (per-split percentile, primary)
b2_stk, b2_ret, b2_pt = run_system(test_races, tier_test_self)
# B4 = reversed control
b4_stk, b4_ret, _ = run_system(test_races, tier_test_self, reversed_=True)
# B2 fixed leak-safe
b2f_stk, b2f_ret, _ = run_system(test_races, tier_test_fixed)
# B2 valid (agreement)
b2v_stk, b2v_ret, _ = run_system(valid_races, assign_tier(comp_valid))

# B0 flat prob-first top-6
b0_stk, b0_ret = [], []
b0_rid = []
for r in test_races:
    s, rt = b0_stake_return(r)
    b0_stk.append(s); b0_ret.append(rt); b0_rid.append(r["rid"])
b0_stk = np.array(b0_stk); b0_ret = np.array(b0_ret)
b0v_stk, b0v_ret = [], []
for r in valid_races:
    s, rt = b0_stake_return(r)
    b0v_stk.append(s); b0v_ret.append(rt)
b0v_stk = np.array(b0v_stk); b0v_ret = np.array(b0v_ret)

# ROIs
roi_b2 = blended_roi(b2_stk, b2_ret)
roi_b4 = blended_roi(b4_stk, b4_ret)
roi_b0 = blended_roi(b0_stk, b0_ret)
roi_b2f = blended_roi(b2f_stk, b2f_ret)
roi_b2v = blended_roi(b2v_stk, b2v_ret)
roi_b0v = blended_roi(b0v_stk, b0v_ret)

ci_b2 = boot_ci_roi(b2_stk, b2_ret)
ci_b4 = boot_ci_roi(b4_stk, b4_ret)
ci_b0 = boot_ci_roi(b0_stk, b0_ret)

# capture %
def capture(stk, ret):
    r = np.asarray(ret); s = np.asarray(stk)
    bet = s > 0
    return 100.0 * np.sum((r > 0) & bet) / np.sum(bet)

cap_b2 = capture(b2_stk, b2_ret)
cap_b0 = capture(b0_stk, b0_ret)
cap_b4 = capture(b4_stk, b4_ret)

print(f"B0 test ROI {roi_b0:.2f} CI{ci_b0}  cap {cap_b0:.1f}")
print(f"B2 test ROI {roi_b2:.2f} CI{ci_b2}  cap {cap_b2:.1f}  fixed {roi_b2f:.2f}  valid {roi_b2v:.2f}")
print(f"B4 rev  ROI {roi_b4:.2f} CI{ci_b4}  cap {cap_b4:.1f}")
print(f"B0 valid ROI {roi_b0v:.2f}")

# ----- ATTACK 1: B2 vs B0 paired delta CI -----
# align races present in both (b2 buys every race that has a tier => all test races)
# build per-race arrays aligned by rid
rid_order = [r["rid"] for r in test_races]
b2_map_s = dict(zip([r["rid"] for r in test_races], b2_stk))
b2_map_r = dict(zip([r["rid"] for r in test_races], b2_ret))
b0_map_s = dict(zip(b0_rid, b0_stk)); b0_map_r = dict(zip(b0_rid, b0_ret))
b4_map_s = dict(zip([r["rid"] for r in test_races], b4_stk))
b4_map_r = dict(zip([r["rid"] for r in test_races], b4_ret))
common = [rid for rid in rid_order if rid in b0_map_s and rid in b2_map_s]
a_s = np.array([b2_map_s[x] for x in common]); a_r = np.array([b2_map_r[x] for x in common])
o_s = np.array([b0_map_s[x] for x in common]); o_r = np.array([b0_map_r[x] for x in common])
r4_s = np.array([b4_map_s[x] for x in common]); r4_r = np.array([b4_map_r[x] for x in common])

delta_b2_b0 = boot_ci_paired(a_s, a_r, o_s, o_r)
delta_b2_b4 = boot_ci_paired(a_s, a_r, r4_s, r4_r)
pt_b2_b0 = blended_roi(a_s, a_r) - blended_roi(o_s, o_r)
pt_b2_b4 = blended_roi(a_s, a_r) - blended_roi(r4_s, r4_r)
print(f"ATTACK1 B2-B0 delta {pt_b2_b0:+.2f}pt CI{delta_b2_b0} straddle0={delta_b2_b0[0]<0<delta_b2_b0[1]}")
print(f"ATTACK2 B2-B4 delta {pt_b2_b4:+.2f}pt CI{delta_b2_b4} straddle0={delta_b2_b4[0]<0<delta_b2_b4[1]}")

# ----- ATTACK 3: per-tier winner difficulty diagnosis -----
def winner_diag(races, tier_map, tier):
    mr = []  # mean model_rank of winning-trio horses (in races bet at this tier that HIT)
    le7 = []  # fraction of winners with rank<=7
    for r in races:
        if tier_map.get(r["rid"]) != tier:
            continue
        g = r["g"]
        wmask = g["finish"].isin([1, 2, 3])
        ranks = g.loc[wmask, "model_rank"].to_numpy()
        if len(ranks) == 0:
            continue
        mr.extend(ranks.tolist())
        le7.extend((ranks <= 7).tolist())
    return (np.mean(mr) if mr else float("nan"),
            100.0 * np.mean(le7) if le7 else float("nan"))

f4_mr, f4_le7 = winner_diag(test_races, tier_test_self, "F4")
f1_mr, f1_le7 = winner_diag(test_races, tier_test_self, "F1")
print(f"ATTACK3 F4(固) winner meanRank {f4_mr:.2f} <=7 {f4_le7:.1f}% | F1(荒) winner meanRank {f1_mr:.2f} <=7 {f1_le7:.1f}%")

# ----- per-tier ROI table -----
def tier_table(per_tier, races, tier_map):
    rows = []
    for t in ["F1", "F2", "F3", "F4"]:
        stk = np.array(per_tier[t]["stk"]); ret = np.array(per_tier[t]["ret"])
        nco = np.array(per_tier[t]["n"])
        if len(stk) == 0:
            continue
        roi = blended_roi(stk, ret)
        ci = boot_ci_roi(stk, ret)
        cap = 100.0 * np.sum(ret > 0) / len(ret)
        rows.append({
            "tier": t, "n_races_test": int(len(stk)),
            "avg_points": float(np.mean(nco)),
            "stake_per_combo": FORMS[t][3],
            "capture_pct": round(cap, 2),
            "roi_pct": round(roi, 2),
            "roi_ci_low": round(ci[0], 2), "roi_ci_high": round(ci[1], 2),
        })
    return rows

per_tier_rows = tier_table(b2_pt, test_races, tier_test_self)
for row in per_tier_rows:
    print(row)

# ----- era stability -----
def era_roi(races, tier_map, yr):
    stk, ret = [], []
    for r in races:
        if int(str(r["rid"])[:4]) != yr:
            continue
        t = tier_map.get(r["rid"])
        if t is None: continue
        s, rt, _ = race_stake_return(r, t)
        stk.append(s); ret.append(rt)
    return blended_roi(np.array(stk), np.array(ret))

era24 = era_roi(test_races, tier_test_self, 2024)
era25 = era_roi(test_races, tier_test_self, 2025)
print(f"era 2024 {era24:.2f}  2025 {era25:.2f}")

# stake per race (avg)
spr_b2 = float(np.mean(b2_stk))
spr_b0 = float(np.mean(b0_stk))
spr_b4 = float(np.mean(b4_stk))

import json
result = {
    "roi_b0": roi_b0, "ci_b0": list(ci_b0), "cap_b0": cap_b0, "spr_b0": spr_b0,
    "roi_b2": roi_b2, "ci_b2": list(ci_b2), "cap_b2": cap_b2, "spr_b2": spr_b2,
    "roi_b4": roi_b4, "ci_b4": list(ci_b4), "cap_b4": cap_b4, "spr_b4": spr_b4,
    "roi_b2f": roi_b2f, "roi_b2v": roi_b2v, "roi_b0v": roi_b0v,
    "n_test": len(test_races), "n_valid": len(valid_races),
    "delta_b2_b0": pt_b2_b0, "ci_d_b2_b0": list(delta_b2_b0),
    "delta_b2_b4": pt_b2_b4, "ci_d_b2_b4": list(delta_b2_b4),
    "f4_mr": f4_mr, "f4_le7": f4_le7, "f1_mr": f1_mr, "f1_le7": f1_le7,
    "era24": era24, "era25": era25,
    "per_tier": per_tier_rows,
}
with open("E:/PyCaLiAI/analysis/_trio_faith_verify_out.json", "w") as f:
    json.dump(result, f, indent=2, default=float)
print("WROTE out json")
