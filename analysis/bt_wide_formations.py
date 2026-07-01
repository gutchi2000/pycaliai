"""Wide formation/points backtest on 2024-2025 OOS.
Eval on 2025; any fit/threshold uses 2024 only (circular avoidance).
Floor: wide takeout ~22.5% -> break-even 77.5% (recoup of stake);
100% = profit.
"""
import pandas as pd, json, itertools

H = pd.read_parquet(r'E:\PyCaLiAI\data\_wide\oos_horses.parquet')
WP = json.load(open(r'E:\PyCaLiAI\data\_wide\oos_wide_pay.json', encoding='utf-8'))

# grade '罠' (trap) is the mojibake single-char category
TRAP = [g for g in H.grade.unique() if g not in ('S', 'A', 'B', 'C')][0]
H['under'] = H['under'].astype(bool)

OOS = H[H.year == 2025].copy()
FIT = H[H.year == 2024].copy()


def wide_eval(races_df, pick_fn):
    cost = ret = hit = nbet = 0
    pts_sum = 0
    for rid, g in races_df.groupby('rid'):
        sel = pick_fn(g)
        if not sel or len(sel) < 2:
            continue
        pairs = list(itertools.combinations(sorted(set(sel)), 2))
        if not pairs:
            continue
        wp = WP.get(rid, {})
        top3 = set(g[g.top3 == 1].ban)
        rr = 0.0; h = False
        for a, b in pairs:
            if a in top3 and b in top3:
                rr += wp.get(f'{a}-{b}', 0.0); h = True
        cost += len(pairs) * 100
        ret += rr
        hit += int(h)
        nbet += 1
        pts_sum += len(pairs)
    return dict(n=nbet,
                roi=round(ret / cost * 100, 1) if cost else 0.0,
                hit=round(hit / nbet * 100, 1) if nbet else 0.0,
                avg_pts=round(pts_sum / nbet, 1) if nbet else 0.0)


def axis_others(g, n_others):
    """axis = mark_rank==1, others = next n by mark_rank (top2..topK)."""
    gg = g.sort_values('mark_rank')
    axis = gg[gg.mark_rank == 1].ban.tolist()
    if not axis:
        return []
    others = gg[gg.mark_rank > 1].head(n_others).ban.tolist()
    return axis + others


def box_topk(g, k):
    gg = g.sort_values('mark_rank')
    return gg.head(k).ban.tolist()


def probfirst_box(g, k):
    """top-k horses by p_win -> box."""
    gg = g.sort_values('p_win', ascending=False)
    return gg.head(k).ban.tolist()


def umami_axis_others(g, n_others, grades=('S', 'A')):
    """axis=◎(mark_rank1); others = top S/A grade horses (excl axis) by mark_rank, fill with mark_rank order."""
    gg = g.sort_values('mark_rank')
    axis = gg[gg.mark_rank == 1].ban.tolist()
    if not axis:
        return []
    pool = gg[gg.mark_rank > 1]
    pref = pool[pool.grade.isin(grades)].ban.tolist()
    sel = pref[:n_others]
    if len(sel) < n_others:
        rest = [b for b in pool.ban.tolist() if b not in sel]
        sel += rest[:n_others - len(sel)]
    return axis + sel


def umami_box(g, k, grades=('S', 'A')):
    """box of top-k UMAMI S/A graded horses by mark_rank, fall back to mark_rank order."""
    gg = g.sort_values('mark_rank')
    pref = gg[gg.grade.isin(grades)].ban.tolist()
    sel = pref[:k]
    if len(sel) < k:
        rest = [b for b in gg.ban.tolist() if b not in sel]
        sel += rest[:k - len(sel)]
    return sel


def cap_pts(pick_fn, cap):
    """wrap a pick_fn so resulting pair count <= cap (trim others, keep axis logic by dropping last selections)."""
    def f(g):
        sel = pick_fn(g)
        if len(sel) < 2:
            return sel
        # reduce sel size until C(len,2) <= cap
        s = sel[:]
        while len(s) >= 2 and len(list(itertools.combinations(s, 2))) > cap:
            s = s[:-1]
        return s
    return f


print('=== A) axis ◎ nagashi: others top2..top8 (no cap) | 2025 OOS ===')
print(f"{'others':<8}{'pts':<6}{'n':<7}{'roi%':<8}{'hit%':<8}")
for no in range(1, 8):
    r = wide_eval(OOS, lambda g, no=no: axis_others(g, no))
    print(f"{no:<8}{r['avg_pts']:<6}{r['n']:<7}{r['roi']:<8}{r['hit']:<8}")

print()
print('=== B) axis ◎ nagashi with point cap 1..10 (others=top8 then capped) | 2025 OOS ===')
print(f"{'cap':<6}{'pts':<6}{'n':<7}{'roi%':<8}{'hit%':<8}")
for cap in range(1, 11):
    r = wide_eval(OOS, cap_pts(lambda g: axis_others(g, 8), cap))
    print(f"{cap:<6}{r['avg_pts']:<6}{r['n']:<7}{r['roi']:<8}{r['hit']:<8}")

print()
print('=== C) box by mark_rank: ◎〇▲(3pts) / ◎〇▲△△(10pts) | 2025 OOS ===')
for k in (3, 5):
    r = wide_eval(OOS, lambda g, k=k: box_topk(g, k))
    label = '◎〇▲' if k == 3 else '◎〇▲△△'
    print(f"{label:<10} k={k} pts={r['avg_pts']} n={r['n']} roi={r['roi']} hit={r['hit']}")

print()
print('=== D) prob-first box (top-k by p_win) k=2..5 | 2025 OOS ===')
print(f"{'k':<6}{'pts':<6}{'n':<7}{'roi%':<8}{'hit%':<8}")
for k in range(2, 6):
    r = wide_eval(OOS, lambda g, k=k: probfirst_box(g, k))
    print(f"{k:<6}{r['avg_pts']:<6}{r['n']:<7}{r['roi']:<8}{r['hit']:<8}")

print()
print('=== E) UMAMI axis ◎ + S/A others (n=1..6) | 2025 OOS ===')
print(f"{'others':<8}{'pts':<6}{'n':<7}{'roi%':<8}{'hit%':<8}")
for no in range(1, 7):
    r = wide_eval(OOS, lambda g, no=no: umami_axis_others(g, no))
    print(f"{no:<8}{r['avg_pts']:<6}{r['n']:<7}{r['roi']:<8}{r['hit']:<8}")

print()
print('=== F) UMAMI box (top-k S/A) k=2..5 | 2025 OOS ===')
print(f"{'k':<6}{'pts':<6}{'n':<7}{'roi%':<8}{'hit%':<8}")
for k in range(2, 6):
    r = wide_eval(OOS, lambda g, k=k: umami_box(g, k))
    print(f"{k:<6}{r['avg_pts']:<6}{r['n']:<7}{r['roi']:<8}{r['hit']:<8}")

print()
print('=== G) UMAMI axis◎+S/A others, capped 3/4/6 | 2025 OOS (memory best-op candidate) ===')
for cap in (3, 4, 6, 10):
    r = wide_eval(OOS, cap_pts(lambda g: umami_axis_others(g, 6), cap))
    print(f"cap={cap:<3} pts={r['avg_pts']} n={r['n']} roi={r['roi']} hit={r['hit']}")
