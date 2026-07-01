"""Wide pair popularity-band analysis (2024 fit / 2025 OOS).
Classify EVERY wide pair (i<j) that actually exists in the payout table by the o9 of its two horses:
  honmei-honmei : both o9 < 5
  honmei-chuana : one < 5, other in [5,15)
  chuana-chuana : both in [5,15)
  (also report 'other'/with-longshot bands for completeness)
ROI/hit/n per band, on year==2025. Then: pairs containing the AI honmei (mark_rank==1) vs not.

NOTE: this is a pair-level enumeration over all available wide payouts, NOT a strategy-selection
backtest. cost=100/pair, ret = payout if both top3. ROI = ret/cost*100. hit = (#winning pairs)/(#pairs).
"""
import pandas as pd, json, itertools
from collections import defaultdict

H = pd.read_parquet(r'E:\PyCaLiAI\data\_wide\oos_horses.parquet')
WP = json.load(open(r'E:\PyCaLiAI\data\_wide\oos_wide_pay.json', encoding='utf-8'))

def band_of(o):
    if o < 5: return 'h'      # honmei (favorite)
    if o < 15: return 'c'     # chuana (mid-longshot)
    return 'l'                 # longshot

PAIR_LABEL = {
    frozenset(['h','h']): 'honmei-honmei (both o9<5)',
    frozenset(['h','c']): 'honmei-chuana (one<5, one 5-15)',
    frozenset(['c','c']): 'chuana-chuana (both 5-15)',
    frozenset(['h','l']): 'honmei-longshot (one<5, one>=15)',
    frozenset(['c','l']): 'chuana-longshot (one 5-15, one>=15)',
    frozenset(['l','l']): 'longshot-longshot (both>=15)',
}

def analyze(df, label_filter=None):
    """Enumerate all existing wide pairs. Accumulate by band.
    label_filter: None=all pairs; 'with_axis'=>at least one of pair is mark_rank==1;
                  'no_axis'=>neither is mark_rank==1."""
    agg = defaultdict(lambda: dict(cost=0.0, ret=0.0, n=0, hit=0))
    for rid, g in df.groupby('rid'):
        wp = WP.get(rid, {})
        if not wp: continue
        o9 = dict(zip(g.ban, g.o9))
        mr = dict(zip(g.ban, g.mark_rank))
        top3 = set(g[g.top3 == 1].ban)
        axis_bans = set(g[g.mark_rank == 1].ban)
        for a, b in itertools.combinations(sorted(o9.keys()), 2):
            oa, ob = o9[a], o9[b]
            if pd.isna(oa) or pd.isna(ob): continue
            # axis filter
            in_axis = (a in axis_bans) or (b in axis_bans)
            if label_filter == 'with_axis' and not in_axis: continue
            if label_filter == 'no_axis' and in_axis: continue
            lab = PAIR_LABEL[frozenset([band_of(oa), band_of(ob)])]
            pay = wp.get(f'{a}-{b}', 0.0)
            d = agg[lab]
            d['cost'] += 100.0
            d['n'] += 1
            if a in top3 and b in top3:
                d['ret'] += pay
                d['hit'] += 1
    out = {}
    for lab, d in agg.items():
        out[lab] = dict(n=d['n'],
                        roi=round(d['ret'] / d['cost'] * 100, 1) if d['cost'] else 0.0,
                        hit=round(d['hit'] / d['n'] * 100, 1) if d['n'] else 0.0)
    return out

oos = H[H.year == 2025]
print(f"=== 2025 OOS: {oos.rid.nunique()} races, {len(oos)} horses ===\n")

print("--- ALL wide pairs by popularity band (2025 OOS) ---")
allb = analyze(oos)
order = list(PAIR_LABEL.values())
for lab in order:
    if lab in allb:
        d = allb[lab]
        print(f"  {lab:42s} n={d['n']:6d}  ROI={d['roi']:6.1f}%  hit={d['hit']:5.1f}%")

print("\n--- pairs CONTAINING AI honmei (mark_rank==1) ---")
wa = analyze(oos, 'with_axis')
for lab in order:
    if lab in wa:
        d = wa[lab]
        print(f"  {lab:42s} n={d['n']:6d}  ROI={d['roi']:6.1f}%  hit={d['hit']:5.1f}%")

print("\n--- pairs NOT containing AI honmei ---")
na = analyze(oos, 'no_axis')
for lab in order:
    if lab in na:
        d = na[lab]
        print(f"  {lab:42s} n={d['n']:6d}  ROI={d['roi']:6.1f}%  hit={d['hit']:5.1f}%")

# overall axis vs not
def overall(df, lf):
    cost = ret = n = hit = 0
    for rid, g in df.groupby('rid'):
        wp = WP.get(rid, {})
        if not wp: continue
        o9 = dict(zip(g.ban, g.o9))
        top3 = set(g[g.top3 == 1].ban)
        axis_bans = set(g[g.mark_rank == 1].ban)
        for a, b in itertools.combinations(sorted(o9.keys()), 2):
            in_axis = (a in axis_bans) or (b in axis_bans)
            if lf == 'with_axis' and not in_axis: continue
            if lf == 'no_axis' and in_axis: continue
            cost += 100; n += 1
            if a in top3 and b in top3:
                ret += wp.get(f'{a}-{b}', 0.0); hit += 1
    return dict(n=n, roi=round(ret/cost*100,1) if cost else 0, hit=round(hit/n*100,1) if n else 0)

print("\n--- OVERALL: ◎含む vs 含まない (all bands) ---")
print(f"  with axis : {overall(oos,'with_axis')}")
print(f"  no  axis  : {overall(oos,'no_axis')}")
print(f"  all       : {overall(oos,None)}")
