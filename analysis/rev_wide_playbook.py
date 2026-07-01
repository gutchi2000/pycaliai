# Adversarial reproduction of the Wide Playbook claims
import pandas as pd, json, itertools, numpy as np

H = pd.read_parquet(r'E:\PyCaLiAI\data\_wide\oos_horses.parquet')
WP = json.load(open(r'E:\PyCaLiAI\data\_wide\oos_wide_pay.json', encoding='utf-8'))

print('=== SANITY ===')
print('total rows', len(H), 'races', H.rid.nunique())
print('races by year', H.groupby('year').rid.nunique().to_dict())
print('top3 base rate', round(H.top3.mean(), 4))
for y in [2024, 2025]:
    rids = set(H[H.year == y].rid.unique())
    cov = sum(1 for r in rids if r in WP)
    print(y, 'races', len(rids), 'in WP', cov)


# wide settlement helper (as given by the task)
def wide_roi(races_df, pick_fn):
    cost = ret = hit = nbet = 0
    pts_sum = 0
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
                avg_pts=round(pts_sum / max(nbet, 1), 1),
                cost=cost, ret=int(ret))


# ---- pick functions by mark_rank ----
def pick_topk(k):
    def f(g):
        return g.sort_values('mark_rank').head(k).ban.tolist()
    return f


def pick_flow(p):
    # axis = mark_rank 1 (anchor), partners = mark_rank 2..p+1
    # produces p pairs (anchor with each partner)
    def f(g):
        sub = g.sort_values('mark_rank').head(p + 1)
        return sub.ban.tolist()  # box of anchor+partners; but flow != box
    return f


# Claim uses "◎軸流し2点" = anchor flow to 2 partners = 2 bets (◎-〇, ◎-▲)
# But wide_roi treats sel as a BOX (all pairs). To do a true flow we need a different settler.
def wide_roi_flow(races_df, anchor_rank, partner_ranks):
    """anchor flows to each partner. n pairs = len(partner_ranks)."""
    cost = ret = hit = nbet = 0; pts_sum = 0
    for rid, g in races_df.groupby('rid'):
        gs = g.sort_values('mark_rank')
        if len(gs) < max([anchor_rank] + partner_ranks):
            continue
        rank2ban = dict(zip(gs.mark_rank, gs.ban))
        a = rank2ban.get(anchor_rank)
        partners = [rank2ban.get(r) for r in partner_ranks if rank2ban.get(r) is not None]
        if a is None or not partners:
            continue
        wp = WP.get(rid, {})
        top3 = set(g[g.top3 == 1].ban)
        rr = 0; h = False
        for b in partners:
            i, j = sorted([a, b])
            if a in top3 and b in top3:
                rr += wp.get(f'{i}-{j}', 0.0); h = True
        cost += len(partners) * 100; ret += rr; hit += int(h); nbet += 1
        pts_sum += len(partners)
    return dict(n=nbet, roi=round(ret / cost * 100, 1) if cost else 0,
                hit=round(hit / nbet * 100, 1) if nbet else 0,
                avg_pts=round(pts_sum / max(nbet, 1), 1),
                cost=cost, ret=int(ret))


print('\n=== CLAIM A: box by top-k, FULL OOS (both years) ===')
for k in [2, 3, 4, 5, 6]:
    r = wide_roi(H, pick_topk(k))
    print(f'box k={k} (pairs={k*(k-1)//2}):', r)

print('\n=== Same, split by year ===')
for y in [2024, 2025]:
    sub = H[H.year == y]
    print(f'--- year {y} ---')
    for k in [2, 3, 4, 5, 6]:
        r = wide_roi(sub, pick_topk(k))
        print(f'box k={k}:', r)

print('\n=== CLAIM: ◎-〇 1 pair (top2) by year ===')
for y in [2024, 2025]:
    sub = H[H.year == y]
    print(y, wide_roi(sub, pick_topk(2)))

print('\n=== CLAIM: ◎軸流し partners=2 (◎-〇,◎-▲) vs partners=4 (10pt? no, flow=4 pairs) ===')
for y in [2024, 2025]:
    sub = H[H.year == y]
    print(f'--- year {y} ---')
    for npart in [1, 2, 3, 4, 5]:
        r = wide_roi_flow(sub, 1, list(range(2, 2 + npart)))
        print(f'  flow ◎+{npart} partners ({npart} pairs):', r)

print('\n=== "相手4頭以上(10点+)" interpretation: box of ◎〇▲△ + more = which gives 10pts? ===')
# 10 pairs = box of 5 (C(5,2)=10). "相手4" likely = anchor + 4 others as box = 5 horses = 10 pairs
for y in [2024, 2025]:
    sub = H[H.year == y]
    print(y, 'box5(10pts)', wide_roi(sub, pick_topk(5)))
