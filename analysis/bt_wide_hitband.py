"""Fill the 30-40% hit gap: try axis◎ + asymmetric small fans, and
UMAMI-gated 'fire only when axis is strong' selection. Eval 2025."""
import pandas as pd, json, itertools

H = pd.read_parquet(r'E:\PyCaLiAI\data\_wide\oos_horses.parquet')
WP = json.load(open(r'E:\PyCaLiAI\data\_wide\oos_wide_pay.json', encoding='utf-8'))
TRAP = [g for g in H.grade.unique() if g not in ('S', 'A', 'B', 'C')][0]
H['under'] = H['under'].astype(bool)
OOS = H[H.year == 2025].copy()
FIT = H[H.year == 2024].copy()


def wide_eval(races_df, pick_fn):
    cost = ret = hit = nbet = pts_sum = 0
    for rid, g in races_df.groupby('rid'):
        sel = pick_fn(g)
        if not sel or len(sel) < 2:
            continue
        pairs = list(itertools.combinations(sorted(set(sel)), 2))
        if not pairs:
            continue
        wp = WP.get(rid, {}); top3 = set(g[g.top3 == 1].ban); rr = 0.0; h = False
        for a, b in pairs:
            if a in top3 and b in top3:
                rr += wp.get(f'{a}-{b}', 0.0); h = True
        cost += len(pairs) * 100; ret += rr; hit += int(h); nbet += 1; pts_sum += len(pairs)
    return dict(n=nbet, roi=round(ret / cost * 100, 1) if cost else 0.0,
                hit=round(hit / nbet * 100, 1) if nbet else 0.0,
                avg_pts=round(pts_sum / nbet, 1) if nbet else 0.0,
                pnl=round(ret - cost))


# axis◎ + 2 others, but FIRE ONLY when race entropy low (堅い) -> raises hit by selecting
def axis2_entgate(g, ent_max):
    if g.ent.iloc[0] > ent_max:
        return []
    gg = g.sort_values('mark_rank')
    return gg.head(3).ban.tolist()  # ◎〇▲ 3pts


print('=== H) ◎〇▲ 3pts, fire only when ent<=thr (gate on堅さ) | 2025 OOS ===')
print(f"{'ent<=':<8}{'pts':<6}{'n':<7}{'roi%':<8}{'hit%':<8}")
for thr in (0.55, 0.60, 0.65, 0.70, 0.75, 1.0):
    r = wide_eval(OOS, lambda g, thr=thr: axis2_entgate(g, thr))
    print(f"{thr:<8}{r['avg_pts']:<6}{r['n']:<7}{r['roi']:<8}{r['hit']:<8}")

print()
# axis◎ + 1 other (1pt, the ◎-〇 single pair) -> 28% hit. Pair ◎ with 〇 only.
def axis_top2_pair(g):
    gg = g.sort_values('mark_rank')
    return gg.head(2).ban.tolist()


# ◎ pair with best-of(〇,▲) is same; try ◎ + (〇 and ▲) but only 2 pairs containing axis (nagashi 2pts)
def axis_nagashi2(g):
    gg = g.sort_values('mark_rank')
    axis = gg[gg.mark_rank == 1].ban.tolist()
    others = gg[gg.mark_rank > 1].head(2).ban.tolist()
    # nagashi: only axis-other pairs, NOT other-other
    return axis, others


def wide_eval_nagashi(races_df, pick_fn):
    """pick_fn -> (axis_list, others_list); only axis×other pairs are bet (true流し)."""
    cost = ret = hit = nbet = pts_sum = 0
    for rid, g in races_df.groupby('rid'):
        ax, ot = pick_fn(g)
        if not ax or not ot:
            continue
        pairs = [tuple(sorted((a, b))) for a in ax for b in ot if a != b]
        pairs = list(set(pairs))
        if not pairs:
            continue
        wp = WP.get(rid, {}); top3 = set(g[g.top3 == 1].ban); rr = 0.0; h = False
        for a, b in pairs:
            if a in top3 and b in top3:
                rr += wp.get(f'{a}-{b}', 0.0); h = True
        cost += len(pairs) * 100; ret += rr; hit += int(h); nbet += 1; pts_sum += len(pairs)
    return dict(n=nbet, roi=round(ret / cost * 100, 1) if cost else 0.0,
                hit=round(hit / nbet * 100, 1) if nbet else 0.0,
                avg_pts=round(pts_sum / nbet, 1) if nbet else 0.0,
                pnl=round(ret - cost))


print()
print('=== I) TRUE nagashi ◎-others only (no other-other), others=2,3,4 | 2025 OOS ===')
print(f"{'others':<8}{'pts':<6}{'n':<7}{'roi%':<8}{'hit%':<8}")
for no in (2, 3, 4):
    def f(g, no=no):
        gg = g.sort_values('mark_rank')
        ax = gg[gg.mark_rank == 1].ban.tolist()
        ot = gg[gg.mark_rank > 1].head(no).ban.tolist()
        return ax, ot
    r = wide_eval_nagashi(OOS, f)
    print(f"{no:<8}{r['avg_pts']:<6}{r['n']:<7}{r['roi']:<8}{r['hit']:<8}")
