# Part 2: clean band claims, confounders, era robustness, bootstrap CI
import pandas as pd, json, itertools, numpy as np

H = pd.read_parquet(r'E:\PyCaLiAI\data\_wide\oos_horses.parquet')
WP = json.load(open(r'E:\PyCaLiAI\data\_wide\oos_wide_pay.json', encoding='utf-8'))


def wide_roi(races_df, pick_fn):
    cost = ret = hit = nbet = 0; pts_sum = 0
    for rid, g in races_df.groupby('rid'):
        sel = pick_fn(g)
        if not sel or len(sel) < 2:
            continue
        wp = WP.get(rid, {})
        pairs = list(itertools.combinations(sorted(sel), 2))
        if not pairs:
            continue
        top3 = set(g[g.top3 == 1].ban)
        rr = 0; h = False
        for a, b in pairs:
            if a in top3 and b in top3:
                rr += wp.get(f'{a}-{b}', 0.0); h = True
        cost += len(pairs) * 100; ret += rr; hit += int(h); nbet += 1
        pts_sum += len(pairs)
    return dict(n=nbet, roi=round(ret / cost * 100, 1) if cost else 0,
                hit=round(hit / nbet * 100, 1) if nbet else 0,
                avg_pts=round(pts_sum / max(nbet, 1), 1))


def pick_topk(k):
    def f(g):
        return g.sort_values('mark_rank').head(k).ban.tolist()
    return f


# CLEAN BAND. The claim: clean band box k=3 = 84.3%/63.8%, flow p=2 = 85.4%/55.6%.
# "Clean" likely = low entropy (堅いレース). Need to define the band on 2024, eval on 2025.
print('=== ENTROPY distribution ===')
race_ent = H.groupby('rid').agg(ent=('ent', 'first'), year=('year', 'first'))
print(race_ent.ent.describe())
print('quartiles:', race_ent.ent.quantile([.1, .25, .5, .75, .9]).round(3).to_dict())

# Fit threshold on 2024: find ent cutoff. Test a few quantiles.
print('\n=== CLEAN BAND (low entropy) by ent threshold, year-split ===')
for q in [0.10, 0.20, 0.25, 0.33, 0.50]:
    cut = race_ent[race_ent.year == 2024].ent.quantile(q)  # fit cut on 2024
    for y in [2024, 2025]:
        clean_rids = set(race_ent[(race_ent.year == y) & (race_ent.ent <= cut)].index)
        sub = H[H.rid.isin(clean_rids)]
        r3 = wide_roi(sub, pick_topk(3))
        print(f'q={q} cut={cut:.3f} year={y} (cleanN={len(clean_rids)}) box3: {r3}')
    print()

print('=== Clean band, box k and flow, eval 2025 with 2024-fit cut(q=0.25) ===')
cut = race_ent[race_ent.year == 2024].ent.quantile(0.25)
clean25 = set(race_ent[(race_ent.year == 2025) & (race_ent.ent <= cut)].index)
sub = H[H.rid.isin(clean25)]
print('clean 2025 races:', len(clean25), 'cut=', round(cut, 4))
for k in [2, 3, 4, 5]:
    print(f'  box k={k}:', wide_roi(sub, pick_topk(k)))


# CONFOUNDER: is "clean" just favorite-heavy? check o9 of ◎ in clean vs all
print('\n=== CONFOUNDER: favorite odds in clean vs all (2025) ===')
H25 = H[H.year == 2025]
fav = H25[H25.mark_rank == 1]
print('all   ◎ median o9:', round(fav.o9.median(), 2), 'mean p_win', round(fav.p_win.mean(), 3))
favc = fav[fav.rid.isin(clean25)]
print('clean ◎ median o9:', round(favc.o9.median(), 2), 'mean p_win', round(favc.p_win.mean(), 3))


# BOOTSTRAP CI on the headline ROIs (2025 OOS). Resample races.
print('\n=== BOOTSTRAP 95% CI of ROI (race-level resample, 2025 OOS) ===')
def boot_ci(races_df, pick_fn, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    # precompute per-race cost & return
    recs = []
    for rid, g in races_df.groupby('rid'):
        sel = pick_fn(g)
        if not sel or len(sel) < 2:
            continue
        wp = WP.get(rid, {})
        pairs = list(itertools.combinations(sorted(sel), 2))
        if not pairs:
            continue
        top3 = set(g[g.top3 == 1].ban)
        rr = sum(wp.get(f'{a}-{b}', 0.0) for a, b in pairs if a in top3 and b in top3)
        recs.append((len(pairs) * 100, rr))
    recs = np.array(recs, dtype=float)
    n = len(recs)
    point = recs[:, 1].sum() / recs[:, 0].sum() * 100
    rois = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        s = recs[idx]
        rois.append(s[:, 1].sum() / s[:, 0].sum() * 100)
    lo, hi = np.percentile(rois, [2.5, 97.5])
    return round(point, 1), round(lo, 1), round(hi, 1), n

for k in [2, 3, 4, 5]:
    pt, lo, hi, n = boot_ci(H25, pick_topk(k))
    over = 'CROSSES 77.5' if lo < 77.5 < hi or hi < 77.5 else ('>floor' if lo > 77.5 else '<floor')
    print(f'2025 box k={k}: ROI={pt} CI[{lo},{hi}] n={n}  vs floor77.5 -> {over}')

# clean band box3 CI
pt, lo, hi, n = boot_ci(sub, pick_topk(3))
print(f'2025 CLEAN box3: ROI={pt} CI[{lo},{hi}] n={n}')
