# -*- coding: utf-8 -*-
"""「馬券構築層を全部外し、◎の複勝を全レース均等額で買う」= null policy の 2026 実測。
本番実績(実額加重 71.8%)と比較し、選別層/配分層がそれぞれ何pt削っているかを分解する。
実行: python -m analysis.null_policy_2026
"""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(__file__).resolve().parents[1]

def ci(p, seed=1, nb=4000):
    p = np.asarray(p, float)
    if len(p) == 0: return None
    rng = np.random.default_rng(seed)
    b = np.array([p[rng.integers(0, len(p), len(p))].mean() for _ in range(nb)])
    return dict(n=len(p), roi=round(100*p.mean(), 1), hit=round(100*(p > 0).mean(), 1),
                ci95=[round(100*np.percentile(b, 2.5), 1), round(100*np.percentile(b, 97.5), 1)])

def load():
    rows = []
    for bp in sorted(glob.glob(str(BASE/'reports/cowork_input/*_bundle.json'))):
        date = Path(bp).name[:8]
        kp = BASE/'data'/'kekka'/f'{date}.csv'
        if not date.startswith('2026') or not kp.exists(): continue
        k = pd.read_csv(kp, encoding='cp932', low_memory=False)
        k['rid16'] = k['レースID(新)'].astype(str).str[:16]
        k['ban'] = pd.to_numeric(k['馬番'], errors='coerce')
        k['fuku'] = pd.to_numeric(k['複勝配当'], errors='coerce')
        k['tan'] = pd.to_numeric(k['単勝配当'], errors='coerce')
        k['fin'] = pd.to_numeric(k['確定着順'], errors='coerce')
        fuku = {(r.rid16, int(r.ban)): r.fuku/100 for r in k.itertuples()
                if pd.notna(r.fuku) and r.fuku > 0 and pd.notna(r.ban)}
        tan = {(r.rid16, int(r.ban)): r.tan/100 for r in k.itertuples()
               if r.fin == 1 and pd.notna(r.tan) and pd.notna(r.ban)}
        raced = {r.rid16 for r in k.itertuples()}
        b = json.loads(Path(bp).read_text(encoding='utf-8'))
        races = b['races'] if isinstance(b['races'], list) else list(b['races'].values())
        for r in races:
            rid = ''.join(c for c in str(r.get('race_id', '')) if c.isdigit())[:16]
            if rid not in raced: continue
            hs = r.get('horses', [])
            hon = next((h for h in hs if h.get('mark') == '◎'), None)
            if not hon: continue
            ban = int(hon['umaban'])
            rows.append(dict(date=date, rid=rid, field=len(hs),
                             p=float(hon.get('p_win') or 0),
                             fuku=fuku.get((rid, ban), 0.0), tan=tan.get((rid, ban), 0.0)))
    return pd.DataFrame(rows)

def main():
    df = load()
    led = json.load(open(BASE/'data/cowork_results.json', encoding='utf-8'))
    bet_rids = {''.join(c for c in str(r['race_id']) if c.isdigit())[:16]
                for r in led['bets'] if r['馬券種'] == '複勝'}
    df['participated'] = df.rid.isin(bet_rids)

    print(f"母集団 {len(df)}R / {df.date.nunique()}開催日 {df.date.min()}-{df.date.max()}\n")
    print("=== null policy: ◎複勝 全レース均等額 ===")
    print(' 全体        ', ci(df.fuku))
    print(' 参戦した     ', ci(df[df.participated].fuku))
    print(' 見送った     ', ci(df[~df.participated].fuku))
    print("\n=== 比較用: ◎単勝 全レース均等額 ===")
    print(' 全体        ', ci(df.tan))
    print("\n=== 月別 ◎複勝 均等額（regime 安定性） ===")
    for m, g in df.assign(m=df.date.str[:6]).groupby('m'):
        print(f'  {m}', ci(g.fuku))
    print("\n=== 頭数別 ===")
    for lbl, g in df.groupby(pd.cut(df.field, [0, 9, 13, 16, 99],
                                    labels=['〜9頭', '10-13頭', '14-16頭', '17頭〜']), observed=True):
        print(f'  {lbl}', ci(g.fuku))

if __name__ == '__main__':
    main()
