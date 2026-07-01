# -*- coding: utf-8 -*-
"""
wide_anaba.py — ◎軸ワイドに穴馬を絡めるとROIは上がるか (OOS実検証)
================================================================================
◎軸ワイドの天井~86%は「当たるが小配当(チャルク)」が原因。◎に穴を絡めれば配当が跳ねるか?
実オッズ(data/odds)で人気帯を定義し、◎×{人気2-3 / 中穴4-7 / 大穴8+ / 全}のワイドを比較。
ヘッジ(◎-人気2点+◎-穴1点)も検証。valid2023→test2024-25 + bootstrap。

反証仮説: [[project_longshot_weakness]] AIは穴に弱い→穴絡めるとROI低下のはず。データで決着。
実行: PYTHONUTF8=1 ./venv311/Scripts/python.exe wide_anaba.py
"""
from __future__ import annotations
import glob, io, json, sys, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import backtest_pl_ev as BT
import pl_probs as PL

BASE = Path(__file__).parent
SCORE_CACHE = BASE / "data/_ev_grid_scores.parquet"
V6_CAL = BASE / "models/pl_calibrators_v6.pkl"
OUT_TXT = BASE / "reports/wide_anaba_summary.txt"
COL_RID = "レースID(新/馬番無)"; COL_BAN = "馬番"
BET = 100; NBOOT = 1500; MIN_RACES = 200; CHAOS_MAX = 0.7943


def log(m): print(m, flush=True)
def _f(x):
    try: return float(x)
    except Exception: return np.nan


def load_odds_pop():
    """{rid16: {ban: 人気rank}} 実単勝オッズ昇順=人気1位。"""
    out = {}
    for f in sorted(glob.glob(str(BASE/"data/odds/odds_*.csv"))):
        df = pd.read_csv(f, encoding="cp932", low_memory=False)
        df["rid16"] = df["レースID(新)"].astype(str).str[:16]
        df["ban"] = pd.to_numeric(df["馬番"], errors="coerce")
        df["o"] = pd.to_numeric(df["単勝オッズ"], errors="coerce")
        for rid, g in df.groupby("rid16"):
            g = g.dropna(subset=["ban", "o"]).sort_values("o")
            out[rid] = {int(b): r for r, b in enumerate(g["ban"].values, start=1)}
    return out


def boot_ci(pairs, seed=42):
    if not pairs: return (0., 0., 0.)
    st=np.array([p[0] for p in pairs],float); rt=np.array([p[1] for p in pairs],float)
    roi=rt.sum()/st.sum() if st.sum() else 0.
    rng=np.random.default_rng(seed); m=len(pairs); o=np.empty(NBOOT)
    for b in range(NBOOT):
        i=rng.integers(0,m,m); s=st[i].sum(); o[b]=rt[i].sum()/s if s else 0.
    return (float(roi), float(np.percentile(o,2.5)), float(np.percentile(o,97.5)))


def band_of(pop):
    if pop is None: return None
    if pop <= 3: return "fav"      # 人気2-3(◎除く)
    if pop <= 7: return "mid"      # 中穴4-7
    return "ana"                   # 大穴8+


def precompute(sc, pop_map, widepay, cal):
    races=[]; t0=time.time(); n=0
    for rid, g in sc.groupby(COL_RID, sort=False):
        rid_s=str(rid)
        if rid_s not in widepay or rid_s not in pop_map: continue
        g=g.sort_values(COL_BAN).reset_index(drop=True)
        ban=g[COL_BAN].astype(int).values; w=PL.pl_weights(g["_score"].values); N=len(w)
        if N<5: continue
        p=w/w.sum(); pc=np.clip(p,1e-9,1); ent=float(-(pc*np.log(pc)).sum()/np.log(N))
        hon=int(np.argmax(w)); hon_ban=int(ban[hon])
        Wm=cal["wide"].predict(BT.all_wide_mat(w).ravel()).reshape(N,N)
        winp={frozenset((int(a),int(b))):float(pay) for (a,b,pay) in widepay[rid_s]}
        pops=pop_map[rid_s]
        # ◎絡みペア: (相手ban, p_wide, band)
        aite=[]
        for j in range(N):
            if j==hon: continue
            bj=int(ban[j]); pw=float(Wm[hon,j]); bd=band_of(pops.get(bj))
            aite.append((bj, pw, bd))
        races.append({"ent":ent,"hon_ban":hon_ban,"aite":aite,"win":winp,
                      "period":"valid" if g["split"].iloc[0]=="valid" else "test"})
        n+=1
        if n%2000==0: log(f"  pre {n} ({time.time()-t0:.0f}s)")
    log(f"  precompute {n} races ({time.time()-t0:.0f}s)")
    return races


def settle_pairs(r, picks):
    hb=r["hon_ban"]; stake=len(picks)*BET; ret=0.0
    for bj in picks:
        ret += r["win"].get(frozenset((hb,bj)), 0.0)*(BET/100.0)
    return stake, ret


def select(r, band, cap, hedge_ana):
    """◎絡み相手から band の馬を p_wide降順 cap。hedge_ana: fav cap2 + ana 1点。"""
    a=r["aite"]
    if hedge_ana:
        favs=sorted([x for x in a if x[2]=="fav"], key=lambda z:-z[1])[:2]
        anas=sorted([x for x in a if x[2]=="ana"], key=lambda z:-z[1])[:1]
        return [x[0] for x in favs+anas]
    pool=[x for x in a if (band=="all" or x[2]==band)]
    pool.sort(key=lambda z:-z[1])
    return [x[0] for x in pool[:cap]]


def main():
    cal=joblib.load(V6_CAL)["calibrators"]
    widepay=BT.load_wide_payouts(); log(f"[wide] {len(widepay):,}")
    pop=load_odds_pop(); log(f"[odds pop] {len(pop):,} races")
    sc=pd.read_parquet(SCORE_CACHE); sc[COL_RID]=sc[COL_RID].astype(str)
    sc=sc[sc["year"].isin([2023,2024,2025])]
    races=precompute(sc, pop, widepay, cal)

    CONFIGS=[
        ("◎-人気(fav) 2点", dict(band="fav",cap=2,hedge_ana=False)),
        ("◎-人気(fav) 3点", dict(band="fav",cap=3,hedge_ana=False)),
        ("◎-中穴(mid) 2点", dict(band="mid",cap=2,hedge_ana=False)),
        ("◎-中穴(mid) 3点", dict(band="mid",cap=3,hedge_ana=False)),
        ("◎-大穴(ana) 2点", dict(band="ana",cap=2,hedge_ana=False)),
        ("◎-大穴(ana) 1点", dict(band="ana",cap=1,hedge_ana=False)),
        ("◎-全 p_wide上位3", dict(band="all",cap=3,hedge_ana=False)),
        ("◎-全 p_wide上位4", dict(band="all",cap=4,hedge_ana=False)),
        ("ヘッジ:◎人気2+◎穴1", dict(band=None,cap=3,hedge_ana=True)),
    ]
    GATES=[("全レース", lambda r: True), ("堅い回のみ(chaos≤.79)", lambda r: r["ent"]<=CHAOS_MAX)]

    L=[]; P=lambda s:(L.append(s), log(s))
    P("="*94)
    P("◎軸ワイドに穴を絡めるか  [v6 p_wide / valid2023→test2024-25 / flat¥100 / 実オッズ人気帯]")
    P("="*94)
    P(f"{'ゲート':22}{'config':22}{'vROI':>7}{'tROI':>7}{'  tCI95':>16}{'tN':>6}{'的中%':>7}")
    for gname, gf in GATES:
        for cname, cfg in CONFIGS:
            vr=[]; tr=[]; hit=0; nt=0
            for r in races:
                if not gf(r): continue
                picks=select(r, **cfg)
                if not picks: continue
                st,rt=settle_pairs(r, picks)
                (vr if r["period"]=="valid" else tr).append((st,rt))
                if r["period"]=="test":
                    nt+=1; hit+= 1 if rt>0 else 0
            if len(vr)<MIN_RACES or len(tr)<MIN_RACES: continue
            vroi=sum(x[1] for x in vr)/sum(x[0] for x in vr)
            troi,lo,hi=boot_ci(tr)
            P(f"{gname:22}{cname:22}{vroi*100:6.1f}%{troi*100:6.1f}% [{lo*100:5.1f},{hi*100:5.1f}]{nt:>6}{hit/nt*100 if nt else 0:>6.1f}%")
    P("="*94)
    P("基準: ◎-人気2点・堅い回 = test 85.9% (wide_lab最良)。これを穴絡めが超えるか?")
    OUT_TXT.write_text("\n".join(L), encoding="utf-8")
    log(f"[saved] {OUT_TXT}")


if __name__ == "__main__":
    main()
