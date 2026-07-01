"""
METHOD E - Market-aware partner selection for 三連複 (trio) buying.

Tests whether steering trio partners toward mid-popularity bands (that the
v6 model still rates) beats a prob-first selection and clears the ~75% takeout
floor. The favorite band (pop 1-3) is a CONTROL; if a real "public overbets
favorites" bias exists, the favorite band should be WORSE than mid-pop bands.

One row per horse per race. WINNING TRIO = the set of 3 ban with finish in {1,2,3}.
Stake = 100 yen per combo. A combo returns tri_pay yen iff its 3-ban set == winning trio.
ROI% = total_return/total_stake*100. Verdict split = 'test' (2024-2025); 'valid'=2023 agreement.
"""
import sys, itertools
sys.path.insert(0, 'E:/PyCaLiAI')
import numpy as np
import pandas as pd
import pl_probs

SUB = 'E:/PyCaLiAI/analysis/_trio_substrate.parquet'
RNG_SEED = 12345
N_BOOT = 2000

df = pd.read_parquet(SUB)

# --- build per-race structures ---
# Each race -> dict with arrays indexed by a stable horse ordering.
def build_races(frame):
    races = []
    for rid, g in frame.groupby('rid', sort=False):
        g = g.reset_index(drop=True)
        ban = g['ban'].to_numpy()
        mrank = g['model_rank'].to_numpy()
        plw = g['pl_w'].to_numpy(dtype=float)
        finish = g['finish'].to_numpy()
        pop = g['pop_rank'].to_numpy(dtype=float)
        oul = g['ou_lastlog'].to_numpy(dtype=float)
        tri_pay = float(g['tri_pay'].iloc[0])
        # winning trio = set of ban whose finish in {1,2,3}
        win_mask = np.isin(finish, [1, 2, 3])
        win_ban = frozenset(ban[win_mask].tolist())
        races.append(dict(
            rid=rid, ban=ban, mrank=mrank, plw=plw, finish=finish,
            pop=pop, oul=oul, tri_pay=tri_pay, win_ban=win_ban,
            valid_trio=(len(win_ban) == 3),
            has_pop=(~np.isnan(pop)).all() and (~np.isnan(oul)).all(),
        ))
    return races

races_all = {sp: build_races(df[df['split'] == sp]) for sp in ['valid', 'test']}

# --- combo enumeration helpers ---
def combos_to_banset(race, idx_combos):
    """map list of index-triples (into race arrays) to list of frozenset(ban)."""
    ban = race['ban']
    out = []
    for c in idx_combos:
        out.append(frozenset(int(ban[i]) for i in c))
    return out

def eval_config(races, select_fn):
    """select_fn(race) -> list of frozenset(ban) combos to buy (each 3 ban), or [] to skip.
    Returns per-race arrays: stake, ret (yen), hit(0/1), bet(0/1)."""
    stakes, rets, hits, bets = [], [], [], []
    for r in races:
        combos = select_fn(r)
        if not combos:
            stakes.append(0.0); rets.append(0.0); hits.append(0); bets.append(0)
            continue
        # dedup combos
        combos = list({c for c in combos if len(c) == 3})
        if not combos:
            stakes.append(0.0); rets.append(0.0); hits.append(0); bets.append(0)
            continue
        stake = 100.0 * len(combos)
        won = any(c == r['win_ban'] for c in combos)
        ret = r['tri_pay'] if won else 0.0
        stakes.append(stake); rets.append(ret); hits.append(1 if won else 0); bets.append(1)
    return (np.array(stakes), np.array(rets), np.array(hits), np.array(bets))

def summarize(stakes, rets, hits, bets):
    bet_mask = bets == 1
    n_races = int(bet_mask.sum())
    tot_stake = stakes.sum()
    tot_ret = rets.sum()
    roi = (tot_ret / tot_stake * 100.0) if tot_stake > 0 else float('nan')
    capture = (hits[bet_mask].sum() / n_races * 100.0) if n_races > 0 else float('nan')
    spr = (stakes[bet_mask].mean()) if n_races > 0 else float('nan')
    return n_races, capture, roi, spr

def bootstrap_roi_ci(stakes, rets, bets, n_boot=N_BOOT, seed=RNG_SEED):
    """race-level bootstrap on ROI over BET races only."""
    bet_mask = bets == 1
    s = stakes[bet_mask]; r = rets[bet_mask]
    n = len(s)
    if n == 0:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    rois = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        ts = s[idx].sum()
        rois[b] = (r[idx].sum() / ts * 100.0) if ts > 0 else np.nan
    lo, hi = np.nanpercentile(rois, [2.5, 97.5])
    return float(lo), float(hi)

# --- market-implied trio prob via softmax(-ou_lastlog) + Harville-style trio approx ---
def market_win_p(race):
    oul = race['oul']
    # softmax of -ou_lastlog within race (lower oul = more popular = higher prob)
    x = -oul
    x = x - np.nanmax(x)
    e = np.exp(x)
    return e / e.sum()

def harville_trio_prob(p, idx_combos):
    """Harville-style P(set of 3 = top3 in any order) = sum over the 6 orderings of
    p_a * p_b/(1-p_a) * p_c/(1-p_a-p_b). Returns dict combo->prob (unnormalized over chosen set)."""
    out = {}
    for (i, j, k) in idx_combos:
        tot = 0.0
        for a, b, c in itertools.permutations((i, j, k)):
            pa, pb, pc = p[a], p[b], p[c]
            d1 = 1.0 - pa
            if d1 <= 1e-12:
                continue
            d2 = 1.0 - pa - pb
            if d2 <= 1e-12:
                continue
            tot += pa * (pb / d1) * (pc / d2)
        out[(i, j, k)] = tot
    return out

# === selection functions ===

def banset_from_modelrank(race, rank):
    """ban of the horse at given model_rank (1-based)."""
    m = race['mrank']
    pos = np.where(m == rank)[0]
    return int(race['ban'][pos[0]]) if len(pos) else None

def model_topN_idx(race, N):
    """array indices of model top-N by model_rank."""
    m = race['mrank']
    order = np.argsort(m)  # rank 1 first
    return order[:N]

def in_pop_band(race, idx, lo, hi):
    p = race['pop'][idx]
    return (not np.isnan(p)) and (lo <= p <= hi)

# --- METHOD e1: ◎〇 axis (rank1&2) + 1 partner in pop band B ∩ model top-8 ---
def make_e1(band):
    lo, hi = band
    def fn(race):
        if not race['has_pop']:
            return []
        ax1 = banset_from_modelrank(race, 1)
        ax2 = banset_from_modelrank(race, 2)
        if ax1 is None or ax2 is None:
            return []
        ax1_pos = np.where(race['ban'] == ax1)[0][0]
        ax2_pos = np.where(race['ban'] == ax2)[0][0]
        top8 = model_topN_idx(race, 8)
        partners = [i for i in top8 if i not in (ax1_pos, ax2_pos) and in_pop_band(race, i, lo, hi)]
        combos = [frozenset((ax1, ax2, int(race['ban'][i]))) for i in partners]
        return combos
    return fn

# --- METHOD e2: ◎ axis (rank1) + 2 partners both in pop band B ∩ model top-8 ---
def make_e2(band):
    lo, hi = band
    def fn(race):
        if not race['has_pop']:
            return []
        ax1 = banset_from_modelrank(race, 1)
        if ax1 is None:
            return []
        ax1_pos = np.where(race['ban'] == ax1)[0][0]
        top8 = model_topN_idx(race, 8)
        partners = [i for i in top8 if i != ax1_pos and in_pop_band(race, i, lo, hi)]
        combos = [frozenset((ax1, int(race['ban'][a]), int(race['ban'][b])))
                  for a, b in itertools.combinations(partners, 2)]
        return combos
    return fn

# --- METHOD e3: PL/market divergence ratio, buy top-8 combos by ratio ---
def make_e3(topk=8):
    def fn(race):
        if not race['has_pop']:
            return []
        n = len(race['ban'])
        if n < 3:
            return []
        # PL trio probs over ALL index combos
        pl = pl_probs.all_sanrenpuku(race['plw'])  # dict idx-triple -> prob
        # market win prob -> harville trio over same combos
        p = market_win_p(race)
        idxs = list(pl.keys())
        mk = harville_trio_prob(p, idxs)
        # ratio PL/market, guard tiny market
        scored = []
        for c in idxs:
            mp = mk.get(c, 0.0)
            if mp <= 1e-12:
                continue
            scored.append((pl[c] / mp, c))
        if not scored:
            return []
        scored.sort(reverse=True)
        chosen = [c for _, c in scored[:topk]]
        return combos_to_banset(race, chosen)
    return fn

# --- baselines for context ---
def make_probfirst(topk=8):
    """prob-first: buy top-k trio combos by PL prob (odds-blind)."""
    def fn(race):
        n = len(race['ban'])
        if n < 3:
            return []
        pl = pl_probs.all_sanrenpuku(race['plw'])
        items = sorted(pl.items(), key=lambda kv: kv[1], reverse=True)[:topk]
        chosen = [c for c, _ in items]
        return combos_to_banset(race, chosen)
    return fn

def make_probfirst_poponly(topk=8):
    """prob-first but only on pop-covered races (apples-to-apples vs e3)."""
    base = make_probfirst(topk)
    def fn(race):
        if not race['has_pop']:
            return []
        return base(race)
    return fn

CONFIGS = [
    # e1 bands (1 partner)
    ('e1_pop1-3_FAVCTRL', make_e1((1, 3))),
    ('e1_pop4-6_mid',     make_e1((4, 6))),
    ('e1_pop7-9_himo',    make_e1((7, 9))),
    ('e1_pop4-9_wide',    make_e1((4, 9))),
    # e2 bands (2 partners)
    ('e2_pop4-9',  make_e2((4, 9))),
    ('e2_pop7-12', make_e2((7, 12))),
    # e3 divergence
    ('e3_div_top8', make_e3(8)),
    # baselines
    ('base_probfirst_top8_all',     make_probfirst(8)),
    ('base_probfirst_top8_poponly', make_probfirst_poponly(8)),
]

def main():
    rows = []
    for name, fn in CONFIGS:
        line = {'config': name}
        for sp in ['valid', 'test']:
            races = races_all[sp]
            stakes, rets, hits, bets = eval_config(races, fn)
            n_races, capture, roi, spr = summarize(stakes, rets, hits, bets)
            lo, hi = bootstrap_roi_ci(stakes, rets, bets)
            line[f'{sp}_n'] = n_races
            line[f'{sp}_cap'] = capture
            line[f'{sp}_roi'] = roi
            line[f'{sp}_lo'] = lo
            line[f'{sp}_hi'] = hi
            line[f'{sp}_spr'] = spr
        rows.append(line)
        print(f"{name:28s} | test n={line['test_n']:5d} cap={line['test_cap']:5.1f}% "
              f"roi={line['test_roi']:6.1f}% CI[{line['test_lo']:6.1f},{line['test_hi']:6.1f}] "
              f"spr={line['test_spr']:5.0f} || valid roi={line['valid_roi']:6.1f}% "
              f"CI[{line['valid_lo']:6.1f},{line['valid_hi']:6.1f}]")
    out = pd.DataFrame(rows)
    out.to_csv('E:/PyCaLiAI/analysis/_trio_bake_market_band_results.csv', index=False)

    # data hygiene report
    print("\n--- data hygiene ---")
    for sp in ['valid', 'test']:
        races = races_all[sp]
        tot = len(races)
        invalid = sum(1 for r in races if not r['valid_trio'])
        nopop = sum(1 for r in races if not r['has_pop'])
        print(f"{sp}: total_races={tot} invalid_trio(not exactly 3 in finish123)={invalid} "
              f"no_full_pop_coverage={nopop}")

if __name__ == '__main__':
    main()
