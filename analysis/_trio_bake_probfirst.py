import sys, json
sys.path.insert(0, 'E:/PyCaLiAI')
import numpy as np
import pandas as pd
from itertools import combinations
import pl_probs

SUB = 'E:/PyCaLiAI/analysis/_trio_substrate.parquet'
N_LIST = [4, 6, 8, 10, 12, 16, 20, 30]
STAKE = 100.0
BOOT = 2000

df = pd.read_parquet(SUB)

# Build per-race structures once: for each rid, the ranked list of combos (by PL trio prob)
# combos expressed as frozenset of ban; the winning trio set; tri_pay; split.
races = {}
dropped_no_pay = 0
dropped_small = 0
dropped_no_trio = 0

for rid, g in df.groupby('rid', sort=False):
    g = g.reset_index(drop=True)
    n = len(g)
    split = g['split'].iloc[0]
    tri_pay = g['tri_pay'].iloc[0]
    # winning trio = set of 3 ban with finish in {1,2,3}
    win = g.loc[g['finish'].isin([1, 2, 3]), 'ban'].tolist()
    if n < 3:
        dropped_small += 1
        continue
    if len(win) != 3:
        dropped_no_trio += 1
        continue
    if pd.isna(tri_pay) or tri_pay <= 0:
        dropped_no_pay += 1
        continue
    w = g['pl_w'].to_numpy(dtype=float)
    ban = g['ban'].to_numpy()
    probs = pl_probs.all_sanrenpuku(w)  # dict {(i,j,k): prob} over array indices
    # rank combos by prob desc; map idx->ban as frozenset
    items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    ranked = [frozenset((int(ban[i]), int(ban[j]), int(ban[k]))) for (i, j, k), _ in items]
    win_set = frozenset(int(x) for x in win)
    races[rid] = {
        'split': split,
        'tri_pay': float(tri_pay),
        'win_set': win_set,
        'ranked': ranked,
        'ncombos': len(ranked),
    }

# Precompute per-race per-N: stake, return for each config
# return = tri_pay if win_set in top-N combos else 0; stake = min(N, ncombos)*STAKE
def race_outcome(r, N):
    top = r['ranked'][:N]
    ncombo = min(N, r['ncombos'])
    stake = ncombo * STAKE
    ret = r['tri_pay'] if r['win_set'] in top else 0.0
    return stake, ret

splits = ['valid', 'test']
rids_by_split = {s: [rid for rid, r in races.items() if r['split'] == s] for s in splits}

rng_master = np.random.default_rng(12345)

results = []
for s in splits:
    rids = rids_by_split[s]
    for N in N_LIST:
        stakes = np.array([race_outcome(races[rid], N)[0] for rid in rids])
        rets = np.array([race_outcome(races[rid], N)[1] for rid in rids])
        n_races = len(rids)
        total_stake = stakes.sum()
        total_ret = rets.sum()
        roi = total_ret / total_stake * 100.0 if total_stake > 0 else float('nan')
        wins = (rets > 0).sum()
        capture = wins / n_races * 100.0 if n_races > 0 else float('nan')
        stake_per_race = stakes.mean()
        # bootstrap CI on ROI: resample races with replacement
        rng = np.random.default_rng(12345)
        idx_all = np.arange(n_races)
        boot_rois = np.empty(BOOT)
        for b in range(BOOT):
            samp = rng.integers(0, n_races, n_races)
            ts = stakes[samp].sum()
            tr = rets[samp].sum()
            boot_rois[b] = tr / ts * 100.0 if ts > 0 else np.nan
        ci_low = float(np.nanpercentile(boot_rois, 2.5))
        ci_high = float(np.nanpercentile(boot_rois, 97.5))
        results.append({
            'split': s, 'N': N, 'n_races': int(n_races),
            'capture_pct': round(capture, 2), 'roi_pct': round(roi, 2),
            'roi_ci_low': round(ci_low, 2), 'roi_ci_high': round(ci_high, 2),
            'stake_per_race': round(float(stake_per_race), 1),
        })

print('dropped_small', dropped_small, 'dropped_no_trio', dropped_no_trio, 'dropped_no_pay', dropped_no_pay)
print('n races valid', len(rids_by_split['valid']), 'test', len(rids_by_split['test']))
for row in results:
    print(row)

with open('E:/PyCaLiAI/analysis/_trio_bake_probfirst_out.json', 'w') as f:
    json.dump({'results': results,
               'dropped': {'small': dropped_small, 'no_trio': dropped_no_trio, 'no_pay': dropped_no_pay},
               'n_races': {s: len(rids_by_split[s]) for s in splits}}, f, indent=2)
