# -*- coding: utf-8 -*-
"""Read-only forensics on the settled ledger: which bet classes are
statistically below takeout, split by era. Race-day block bootstrap."""
import json, sys, io, random
from collections import defaultdict
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

TAKEOUT = {'単勝': 80.0, '複勝': 80.0, 'ワイド': 77.5, '馬連': 77.5, '馬単': 77.5,
           '三連複': 75.0, '三連単': 72.5, '枠連': 77.5}

d = json.load(open('data/cowork_results.json', encoding='utf-8'))
bets = d['bets']

def boot(rows, reps=10000, seed=42):
    """race-day block bootstrap on ROI (%)"""
    rnd = random.Random(seed)
    byday = defaultdict(list)
    for r in rows:
        byday[r['date']].append(r)
    days = list(byday)
    if not days:
        return None
    out = []
    for _ in range(reps):
        s = t = 0.0
        for _ in days:
            for r in byday[rnd.choice(days)]:
                s += r['購入額'] - r.get('返還', 0)
                t += r['払戻']
        if s > 0:
            out.append(100 * t / s)
    out.sort()
    return round(out[int(.025*len(out))], 1), round(out[int(.975*len(out))], 1)

def stat(rows, label, takeout=None, ci=True):
    s = sum(r['購入額'] - r.get('返還', 0) for r in rows)
    t = sum(r['払戻'] for r in rows)
    h = sum(1 for r in rows if r['的中'])
    if s <= 0:
        return
    roi = 100*t/s
    c = boot(rows) if ci else None
    verdict = ''
    if c and takeout:
        verdict = 'BELOW TAKEOUT (確定漏れ)' if c[1] < takeout else ('ABOVE' if c[0] > takeout else 'inconclusive')
    print(f"{label:<28} n={len(rows):>5} hit={100*h/len(rows):>5.1f}%  "
          f"bet={s:>10,.0f} ROI={roi:>6.1f}%  CI95={c}  {verdict}")

ERAS = [('全期間 0418-0906', '00000000', '99999999'),
        ('v5期 0418-0517',  '20260418', '20260517'),
        ('shape期 0518-0809','20260518', '20260809'),
        ('topdown期 0810-',  '20260810', '99999999')]

for name, a, b in ERAS:
    rows = [r for r in bets if a <= r['date'] <= b]
    print(f"\n===== {name} =====")
    stat(rows, '合計', None)
    for ty in sorted({r['馬券種'] for r in rows}):
        sub = [r for r in rows if r['馬券種'] == ty]
        stat(sub, f'  {ty}', TAKEOUT.get(ty, 77.5))

print("\n===== 反実仮想: 確定漏れ券種を落とした場合（全期間） =====")
for drop in ([], ['複勝'], ['馬単'], ['複勝', '馬単'], ['複勝', '馬単', '馬連']):
    rows = [r for r in bets if r['馬券種'] not in drop]
    stat(rows, 'drop=' + (','.join(drop) if drop else 'なし'), None)

print("\n===== topdown期だけで同じ反実仮想 =====")
td = [r for r in bets if r['date'] >= '20260810']
for drop in ([], ['複勝'], ['馬単'], ['複勝', '馬単']):
    rows = [r for r in td if r['馬券種'] not in drop]
    stat(rows, 'drop=' + (','.join(drop) if drop else 'なし'), None)
