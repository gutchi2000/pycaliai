# Part 3: monotonicity test, flow CI, grade confounder, broad-box claim, trigami
import pandas as pd, json, itertools, numpy as np

H = pd.read_parquet(r'E:\PyCaLiAI\data\_wide\oos_horses.parquet')
WP = json.load(open(r'E:\PyCaLiAI\data\_wide\oos_wide_pay.json', encoding='utf-8'))


def race_recs(races_df, pick_fn):
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
    return np.array(recs, dtype=float)


def roi_of(recs):
    return recs[:, 1].sum() / recs[:, 0].sum() * 100


def boot(recs, B=4000, seed=1):
    rng = np.random.default_rng(seed)
    n = len(recs)
    out = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n)
        s = recs[idx]
        out[i] = s[:, 1].sum() / s[:, 0].sum() * 100
    return out


def pick_topk(k):
    def f(g):
        return g.sort_values('mark_rank').head(k).ban.tolist()
    return f


H24, H25 = H[H.year == 2024], H[H.year == 2025]

# 1) MONOTONICITY claim: "絞るほどROI高". Test box k=2 vs k=3 vs k=4 paired-by-race on 2025.
print('=== MONOTONICITY: is box k=2 > k=3 > k=4 a real ordering? (2025) ===')
r2, r3, r4 = race_recs(H25, pick_topk(2)), race_recs(H25, pick_topk(3)), race_recs(H25, pick_topk(4))
print('point ROIs: k2=%.1f k3=%.1f k4=%.1f' % (roi_of(r2), roi_of(r3), roi_of(r4)))
# paired diff bootstrap k2 - k4 (same races, aligned order from groupby is stable)
rng = np.random.default_rng(2)
n = len(r2)
diffs = []
for _ in range(4000):
    idx = rng.integers(0, n, n)
    d = roi_of(r2[idx]) - roi_of(r4[idx])
    diffs.append(d)
lo, hi = np.percentile(diffs, [2.5, 97.5])
print('k2 - k4 ROI diff: %.2f  CI[%.2f, %.2f]  (claim: k2>k4 strict)' % (np.mean(diffs), lo, hi))
diffs2 = []
for _ in range(4000):
    idx = rng.integers(0, n, n)
    diffs2.append(roi_of(r2[idx]) - roi_of(r3[idx]))
lo2, hi2 = np.percentile(diffs2, [2.5, 97.5])
print('k2 - k3 ROI diff: %.2f  CI[%.2f, %.2f]' % (np.mean(diffs2), lo2, hi2))

# 2) Broad box claim: 20pt+ box ROI 74.5%. box of 6 = 15, box of 7 = 21.
print('\n=== BROAD BOX claim (20pt+ = box7=21pts) 2025 ===')
for k in [6, 7, 8]:
    rr = race_recs(H25, pick_topk(k))
    b = boot(rr)
    print(f'box k={k} ({k*(k-1)//2}pts): ROI={roi_of(rr):.1f} CI[{np.percentile(b,2.5):.1f},{np.percentile(b,97.5):.1f}]')

# 3) GRADE confounder: does UMAMI grade selection beat naive mark_rank box? grade fit on 2024.
print('\n=== GRADE (UMAMI) vs mark_rank: pick S/A graded horses, box, 2025 ===')
def pick_grade(grades):
    def f(g):
        sub = g[g.grade.isin(grades)]
        return sub.ban.tolist()
    return f
for gr in [['S'], ['S', 'A'], ['S', 'A', 'B']]:
    rr = race_recs(H25, pick_grade(gr))
    if len(rr) == 0:
        print(gr, 'no bets'); continue
    b = boot(rr)
    print(f'grade {gr}: ROI={roi_of(rr):.1f} CI[{np.percentile(b,2.5):.1f},{np.percentile(b,97.5):.1f}] nbet={len(rr)} avgpairs={rr[:,0].mean()/100:.1f}')

# 4) FLOW p=2 clean-band claim vs box3 clean-band: is flow really 85.4%?
print('\n=== Clean-band FLOW vs BOX (2024-fit cut q=0.25, eval 2025) ===')
race_ent = H.groupby('rid').agg(ent=('ent', 'first'), year=('year', 'first'))
cut = race_ent[race_ent.year == 2024].ent.quantile(0.25)
clean25 = set(race_ent[(race_ent.year == 2025) & (race_ent.ent <= cut)].index)
csub = H25[H25.rid.isin(clean25)]

def wide_roi_flow_recs(races_df, anchor_rank, partner_ranks):
    recs = []
    for rid, g in races_df.groupby('rid'):
        gs = g.sort_values('mark_rank')
        rank2ban = dict(zip(gs.mark_rank, gs.ban))
        a = rank2ban.get(anchor_rank)
        partners = [rank2ban.get(r) for r in partner_ranks if rank2ban.get(r) is not None]
        if a is None or not partners:
            continue
        wp = WP.get(rid, {})
        top3 = set(g[g.top3 == 1].ban)
        rr = sum(wp.get('%d-%d' % tuple(sorted([a, b])), 0.0) for b in partners if a in top3 and b in top3)
        recs.append((len(partners) * 100, rr))
    return np.array(recs, dtype=float)

rrflow = wide_roi_flow_recs(csub, 1, [2, 3])
b = boot(rrflow)
print(f'CLEAN flow ◎+2: ROI={roi_of(rrflow):.1f} CI[{np.percentile(b,2.5):.1f},{np.percentile(b,97.5):.1f}] n={len(rrflow)}')
rrbox3 = race_recs(csub, pick_topk(3))
b = boot(rrbox3)
print(f'CLEAN box3:     ROI={roi_of(rrbox3):.1f} CI[{np.percentile(b,2.5):.1f},{np.percentile(b,97.5):.1f}] n={len(rrbox3)}')

# 5) How many distinct clean-band strategies were searched? multiplicity / overfit risk
print('\n=== MULTIPLICITY: best-of-many on 2025 directly (overfit demo) ===')
# scan ent cut x k on 2025 itself, report the max ROI found (what cherry-picking yields)
best = (0, None)
for q in np.arange(0.05, 0.6, 0.05):
    cut2 = race_ent.ent.quantile(q)
    rids = set(race_ent[(race_ent.year == 2025) & (race_ent.ent <= cut2)].index)
    s = H25[H25.rid.isin(rids)]
    for k in [2, 3, 4]:
        rr = race_recs(s, pick_topk(k))
        if len(rr) < 100: continue
        roi = roi_of(rr)
        if roi > best[0]:
            best = (round(roi, 1), f'q={q:.2f} k={k} n={len(rr)}')
print('cherry-picked max on 2025:', best, '(this is the overfit ceiling, not a real edge)')
