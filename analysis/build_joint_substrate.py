# -*- coding: utf-8 -*-
"""
build_joint_substrate.py — 市場内部(Harville/Stern/Dr.Z)検定の共通基盤【モデル不要】
================================================================================
目的: benter_singlewin_noedge が名指しした唯一の生存路 =
  「市場の連勝/複勝オッズは、市場自身の単勝marginalと構造的に矛盾(mispriced)しているか」
を撃つための per-race / per-pair 基盤を焼く。**モデルを一切使わない** = 市場 vs 市場の
純粋な一貫性裁定。決定材料は当日9時、決済は確定。leak-safe。

入力:
  data/Time _series_odds/TANPUK_*.csv   単勝+複勝(Lo/Hi)時系列 → 9時de-vig π_i, 複勝odds
  data/Time _series_odds/UMAREN_*.csv   馬連 全ペアodds時系列 → 9時/確定 o_ij
  (load_payouts)                        確定着順 top3 馬番 + 実払戻

出力:
  data/_joint/mkt_horses.parquet   per-horse: rid,year,n,ban,pi(9時単marginal),o_win9,
       fuku_lo9,fuku_hi9,win_overround,jyun(確定),top3(0/1),is_win,is_plc,is_sho
  data/_joint/mkt_umaren.parquet   per-pair: rid,year,i,j,o_um9,o_umfin,is_winpair,
       um_overround9
実行: PYTHONUTF8=1 python analysis/build_joint_substrate.py
"""
import sys, io, glob, re, os
from pathlib import Path
import numpy as np, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
sys.path.insert(0, "E:/PyCaLiAI")
from backtest_pl_ev import load_payouts

BASE = Path("E:/PyCaLiAI")
ODIR = BASE / "data" / "Time _series_odds"
OUT = BASE / "data" / "_joint"; os.makedirs(OUT, exist_ok=True)
YEARS = None  # None=全年。重い時は (2022,2023,2024,2025)


def rid16(x): return re.sub(r"\D", "", str(x))[:16]


def load_tanpuk():
    """per-race: 9時(区分1, race日0900最寄) の 単勝odds dict, 複勝Lo/Hi dict。"""
    files = sorted(glob.glob(str(ODIR / "TANPUK_*.csv")))
    tp = pd.concat([pd.read_csv(f, encoding="cp932", low_memory=False) for f in files],
                   ignore_index=True)
    c = list(tp.columns); RID, KB, TM = c[0], c[1], c[2]
    tan, flo, fhi = {}, {}, {}
    for x in c:
        m = re.match(r"^\s*(\d+)\s*単\s*$", str(x));            m and tan.__setitem__(int(m.group(1)), x)
        m = re.match(r"^\s*(\d+)\s*複\s*Lo\s*$", str(x));        m and flo.__setitem__(int(m.group(1)), x)
        m = re.match(r"^\s*(\d+)\s*複\s*Hi\s*$", str(x));        m and fhi.__setitem__(int(m.group(1)), x)
    tp["rid16"] = tp[RID].map(rid16)
    tp[KB] = pd.to_numeric(tp[KB], errors="coerce")
    tp[TM] = pd.to_numeric(tp[TM], errors="coerce")
    tp = tp[tp["rid16"].str.len() == 16]
    tp = tp[tp["rid16"].str[:4].astype(int) >= 2013]
    if YEARS:
        tp = tp[tp["rid16"].str[:4].astype(int).isin(YEARS)]
    # 全odds列を一括数値化 (セル単位 to_numeric を回避)
    allcols = list(tan.values()) + list(flo.values()) + list(fhi.values())
    tp[allcols] = tp[allcols].apply(pd.to_numeric, errors="coerce")
    out = {}
    for rid, g in tp.groupby("rid16", sort=False):
        mmdd = int(rid[4:8]); g1 = g[g[KB] == 1]
        if len(g1) == 0: continue
        hh = g1[TM] % 10000; dd = g1[TM] // 10000
        base = g1[dd == mmdd]; base = base if len(base) > 0 else g1
        row = base.loc[(hh.loc[base.index] - 900).abs().idxmin()]
        t9 = {b: float(row[col]) for b, col in tan.items() if pd.notna(row[col]) and row[col] > 1.0}
        fl = {}; fh = {}
        for b, col in flo.items():
            lo = row[col]; hi = row[fhi[b]] if b in fhi else np.nan
            if pd.notna(lo) and lo > 1.0: fl[b] = float(lo); fh[b] = float(hi) if pd.notna(hi) else float(lo)
        if len(t9) >= 5:
            out[rid] = {"t9": t9, "fl": fl, "fh": fh}
    return out


def load_umaren():
    """per-race: 9時 / 確定 の 馬連ペアodds dict {(i,j):odds}。"""
    files = sorted(glob.glob(str(ODIR / "UMAREN_*.csv")))
    um = pd.concat([pd.read_csv(f, encoding="cp932", low_memory=False) for f in files],
                   ignore_index=True)
    c = list(um.columns); RID, KB, TM, TOU = c[0], c[1], c[2], c[3]
    pair_cols = {}
    for x in c[5:]:
        m = re.search(r"(\d+)\s*-\s*(\d+)", str(x))
        if m: pair_cols[(int(m.group(1)), int(m.group(2)))] = x
    um["rid16"] = um[RID].map(rid16)
    um[KB] = pd.to_numeric(um[KB], errors="coerce")
    um[TM] = pd.to_numeric(um[TM], errors="coerce")
    um = um[um["rid16"].str.len() == 16]
    um = um[um["rid16"].str[:4].astype(int) >= 2013]
    if YEARS:
        um = um[um["rid16"].str[:4].astype(int).isin(YEARS)]
    pair_items = list(pair_cols.items())
    um[[c for c in pair_cols.values()]] = um[[c for c in pair_cols.values()]].apply(pd.to_numeric, errors="coerce")

    def extract(row, tou):
        out = {}
        for (i, j), col in pair_items:
            if i <= tou and j <= tou:
                v = row[col]
                if pd.notna(v) and v > 1.0: out[(i, j)] = float(v)
        return out

    out = {}
    for rid, g in um.groupby("rid16", sort=False):
        try:
            tou = int(pd.to_numeric(g[TOU], errors="coerce").dropna().iloc[0])
        except Exception:
            continue
        mmdd = int(rid[4:8]); g1 = g[g[KB] == 1]; g4 = g[g[KB] == 4]
        if len(g1) == 0 or len(g4) == 0: continue
        hh = g1[TM] % 10000; dd = g1[TM] // 10000
        base = g1[dd == mmdd]; base = base if len(base) > 0 else g1
        am9 = base.loc[(hh.loc[base.index] - 900).abs().idxmin()]
        fin = g4.loc[g4[TM].idxmax()]
        o9 = extract(am9, tou); ofin = extract(fin, tou)
        if len(o9) >= 5 and len(ofin) >= 5:
            out[rid] = {"o9": o9, "ofin": ofin, "tou": tou}
    return out


def main():
    print("[load] TANPUK (単複 9時 snapshot) ...")
    tan = load_tanpuk();  print(f"  races(単複9時): {len(tan):,}")
    print("[load] UMAREN (馬連 9時/確定 snapshot) ...")
    um = load_umaren();   print(f"  races(馬連9時+確定): {len(um):,}")
    print("[load] payouts (確定着順 top3) ...")
    pays = load_payouts(); print(f"  races(払戻): {len(pays):,}")

    common = sorted(set(tan) & set(um) & set(pays))
    print(f"  3者共通レース: {len(common):,}")

    h_rows = []; p_rows = []
    for rid in common:
        t = tan[rid]; u = um[rid]; po = pays[rid]
        bans = sorted(t["t9"].keys())
        n = len(bans)
        if n < 5: continue
        inv = np.array([1.0 / t["t9"][b] for b in bans])
        wovr = float(inv.sum())                 # 単勝 overround (~1.25)
        pi = inv / wovr                          # de-vig 単勝 marginal
        win_b, plc_b, sho_b = po["win"], po["plc"], po["sho"]
        top3set = {win_b, plc_b, sho_b}
        year = int(rid[:4])
        for k, b in enumerate(bans):
            h_rows.append(dict(
                rid=rid, year=year, n=n, ban=int(b), pi=float(pi[k]),
                o_win9=float(t["t9"][b]),
                fuku_lo9=(float(t["fl"][b]) if b in t["fl"] else np.nan),
                fuku_hi9=(float(t["fh"][b]) if b in t["fh"] else np.nan),
                win_overround=wovr,
                is_win=int(b == win_b), is_plc=int(b == plc_b), is_sho=int(b == sho_b),
                top3=int(b in top3set),
            ))
        # 馬連ペア
        wp = (min(win_b, plc_b), max(win_b, plc_b))
        ovr9 = float(np.sum([1.0 / v for v in u["o9"].values()]))   # 馬連 overround (~1.29)
        for (i, j), o9 in u["o9"].items():
            ofin = u["ofin"].get((i, j))
            p_rows.append(dict(
                rid=rid, year=year, i=int(i), j=int(j), n=n,
                o_um9=float(o9), o_umfin=(float(ofin) if ofin else np.nan),
                is_winpair=int((i, j) == wp), um_overround9=ovr9,
                umaren_pay=(float(po["umaren"]) if (i, j) == wp and po["umaren"] == po["umaren"] else np.nan),
            ))

    H = pd.DataFrame(h_rows); P = pd.DataFrame(p_rows)
    H.to_parquet(OUT / "mkt_horses.parquet")
    P.to_parquet(OUT / "mkt_umaren.parquet")
    print(f"\n[out] mkt_horses.parquet rows={len(H):,} races={H.rid.nunique():,} "
          f"(test2024-25 races={H[H.year>=2024].rid.nunique():,})")
    print(f"[out] mkt_umaren.parquet rows={len(P):,} pairs/race~{len(P)/max(P.rid.nunique(),1):.1f}")
    # --- サニティ ---
    print("\n=== サニティ ===")
    print(f"  単勝 overround 中央値 {H.groupby('rid').win_overround.first().median():.4f} (~1.25=控除20%)")
    print(f"  馬連 overround 中央値 {P.groupby('rid').um_overround9.first().median():.4f} (~1.29=控除22.5%)")
    print(f"  複勝odds カバレッジ {H.fuku_lo9.notna().mean():.3f}")
    print(f"  馬連確定odds カバレッジ {P.o_umfin.notna().mean():.3f}")
    # 単勝 calibration ざっくり: pi 帯別の実勝率
    H["wbin"] = pd.cut(H.pi, [0, .05, .1, .2, .35, 1.0])
    cal = H.groupby("wbin", observed=True).agg(n=("is_win", "size"), pi=("pi", "mean"), win=("is_win", "mean"))
    print("  単勝marginal calibration (pi帯→平均pi vs 実勝率):")
    for b, r in cal.iterrows():
        print(f"    {str(b):>14}: n{int(r.n):>6} pi{r.pi:.3f} 実{r.win:.3f}")


if __name__ == "__main__":
    main()
