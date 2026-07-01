"""Robustness: does honmei-honmei (both o9<5) hold the floor in 2024 too?
And sensitivity to the favorite cutoff."""
import pandas as pd, json, itertools
from collections import defaultdict

H = pd.read_parquet(r'E:\PyCaLiAI\data\_wide\oos_horses.parquet')
WP = json.load(open(r'E:\PyCaLiAI\data\_wide\oos_wide_pay.json', encoding='utf-8'))

def hh_roi(df, fav_cut):
    cost=ret=n=hit=0
    for rid,g in df.groupby('rid'):
        wp=WP.get(rid,{})
        if not wp: continue
        o9=dict(zip(g.ban,g.o9)); top3=set(g[g.top3==1].ban)
        favs=[b for b in o9 if not pd.isna(o9[b]) and o9[b]<fav_cut]
        for a,b in itertools.combinations(sorted(favs),2):
            cost+=100; n+=1
            if a in top3 and b in top3:
                ret+=wp.get(f'{a}-{b}',0.0); hit+=1
    return dict(n=n, roi=round(ret/cost*100,1) if cost else 0, hit=round(hit/n*100,1) if n else 0)

for yr in [2024,2025]:
    d=H[H.year==yr]
    print(f"=== honmei-honmei band, both o9<cut, year={yr} ({d.rid.nunique()} races) ===")
    for cut in [4,5,6,7]:
        print(f"  o9<{cut}: {hh_roi(d,cut)}")
    print()
