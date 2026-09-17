# -*- coding: utf-8 -*-
"""
build_bet_substrate.py — 全券種の厳密決済サブストレート
========================================================
目的: 「単勝/複勝/枠連/馬連/馬単/ワイド/三連複/三連単 × 買い方 × 点数 × 金額」を
      総当たりで探索するための、OOS 確率 + 実払戻 の完全テーブルを1本に焼く。

入力:
  data/_policy/v6_scores.parquet             本番v6 OOS の p_win/p_fuku (2023-25, 10364R)
  E:/競馬過去走データ/kekka_20130105-20251228_v2.csv  全頭オッズ + 全券種払戻
  data/wide_payouts_2016-2025.parquet        ワイド3組の払戻

出力:
  data/_policy/bet_substrate.pkl

決済の厳密性:
  - 同着レースは除外 (払戻組が複数になり厳密決済できないため。全体の <1%)
  - 券種の的中判定は 確定着順 から組を復元して照合。払戻は当該行の実配当。
  - 単勝/複勝オッズは「指時系2 = 当日9時」= 発走前に取得可能な値のみ使用 (リーク無し)。
実行: python -m analysis.build_bet_substrate
"""
from __future__ import annotations
import sys
from itertools import combinations, permutations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
KEKKA = Path(r"E:\競馬過去走データ\kekka_20130105-20251228_v2.csv")
OUT = BASE / "data/_policy/bet_substrate.pkl"

USECOLS = ["日付", "場所", "Ｒ", "枠番", "馬番", "確定着順", "レースID(新)",
           "単勝配当", "複勝配当", "枠連", "馬連", "馬単", "３連複", "３連単",
           "指時系2・単勝", "指時系2・複下", "複上2"]


def main() -> None:
    src = BASE / "data/_policy/v6_scores.parquet"   # 本番 v6 の OOS 確率 (analysis/dump_v6_scores.py)
    marks = pd.read_parquet(src)
    marks["rid"] = marks["rid"].astype(str)
    print(f"prob source: {src.name}")
    want = set(marks.rid)
    print(f"marks: {len(marks):,} rows / {len(want):,} races")

    print("reading kekka master (large, cp932)...")
    k = pd.read_csv(KEKKA, encoding="cp932", low_memory=False, usecols=USECOLS)
    k["rid"] = k["レースID(新)"].astype(str).str[:16]
    k = k[k.rid.isin(want)].copy()
    for c, n in [("枠番", "waku"), ("馬番", "ban"), ("確定着順", "fin"),
                 ("複勝配当", "fuku_pay"), ("枠連", "wakuren"), ("馬連", "umaren"),
                 ("馬単", "umatan"), ("３連複", "sanpuku"), ("３連単", "sanrentan"),
                 ("指時系2・単勝", "odds9"), ("指時系2・複下", "fuku_lo"), ("複上2", "fuku_hi")]:
        k[n] = pd.to_numeric(k[c], errors="coerce")
    # 単勝配当は勝ち馬行のみ数値、他は "(odds)" 文字列
    k["tan_pay"] = pd.to_numeric(k["単勝配当"], errors="coerce")
    print(f"kekka joined rows: {len(k):,} / races {k.rid.nunique():,}")

    wide = pd.read_parquet(BASE / "data/wide_payouts_2016-2025.parquet")
    wide["race_id"] = wide["race_id"].astype(str)
    wmap = {r.race_id: ((int(r.w1_i), int(r.w1_j), float(r.w1_pay)),
                        (int(r.w2_i), int(r.w2_j), float(r.w2_pay)),
                        (int(r.w3_i), int(r.w3_j), float(r.w3_pay)))
            for r in wide.itertuples()
            if not any(pd.isna(x) for x in (r.w1_i, r.w1_j, r.w1_pay,
                                            r.w2_i, r.w2_j, r.w2_pay,
                                            r.w3_i, r.w3_j, r.w3_pay))}
    print(f"wide payouts available for {len(wmap):,} races")

    pm = {(r.rid, int(r.ban)): (float(r.p_win), float(r.p_fuku))
          for r in marks.itertuples() if not pd.isna(r.ban)}

    races, skipped = [], {"deadheat": 0, "no_top3": 0, "no_prob": 0, "bad_odds": 0}
    for rid, g in k.groupby("rid", sort=True):
        g = g.dropna(subset=["ban"])
        # 確定着順 0 = 出走取消/除外。出走していないので母集団から外す
        g = g[g.fin.notna() & (g.fin >= 1)]
        if len(g) < 5:
            skipped["no_top3"] += 1
            continue
        fin = g["fin"].to_numpy()
        # 同着レースは払戻組が複数になり厳密決済できないので落とす
        top = g[(g.fin >= 1) & (g.fin <= 3)]
        if len(top) != 3 or sorted(top.fin.tolist()) != [1, 2, 3]:
            skipped["deadheat" if len(top) > 3 else "no_top3"] += 1
            continue
        ban = g["ban"].to_numpy(int)
        probs = np.array([pm.get((rid, b), (np.nan, np.nan))[0] for b in ban], float)
        pfuku = np.array([pm.get((rid, b), (np.nan, np.nan))[1] for b in ban], float)
        if np.isnan(probs).any() or probs.sum() <= 0:
            skipped["no_prob"] += 1
            continue
        odds9 = g["odds9"].to_numpy(float)
        if np.isnan(odds9).all():
            skipped["bad_odds"] += 1
            continue

        o = {int(r.fin): int(r.ban) for r in top.itertuples()}
        w1 = g.loc[g.fin == 1]
        w2 = g.loc[g.fin == 2]
        w3 = g.loc[g.fin == 3]
        rec = dict(
            rid=rid, date=rid[:8], place=str(g["場所"].iloc[0]),
            rno=int(pd.to_numeric(g["Ｒ"].iloc[0], errors="coerce") or 0),
            n=len(ban), ban=ban, waku=g["waku"].to_numpy(float),
            p=probs / probs.sum(), p_fuku=pfuku, odds9=odds9,
            fuku_lo=g["fuku_lo"].to_numpy(float), fuku_hi=g["fuku_hi"].to_numpy(float),
            fin=fin, first=o[1], second=o[2], third=o[3],
            tan_pay=float(w1["tan_pay"].iloc[0]) if not pd.isna(w1["tan_pay"].iloc[0]) else np.nan,
            fuku_pay={int(r.ban): float(r.fuku_pay) for r in top.itertuples()
                      if not pd.isna(r.fuku_pay)},
            wakuren=float(w1["wakuren"].iloc[0]), umaren=float(w1["umaren"].iloc[0]),
            umatan=float(w1["umatan"].iloc[0]), sanpuku=float(w3["sanpuku"].iloc[0]),
            sanrentan=float(w3["sanrentan"].iloc[0]),
            waku_first=float(w1["waku"].iloc[0]), waku_second=float(w2["waku"].iloc[0]),
            wide=wmap.get(rid),
        )
        races.append(rec)

    print(f"built {len(races):,} races   skipped={skipped}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(races, OUT, compress=3)
    print(f"saved -> {OUT}")

    yrs = pd.Series([r["date"][:6] for r in races]).value_counts().sort_index()
    print(yrs.to_string())
    cov = {k2: float(np.mean([not pd.isna(r[k2]) if not isinstance(r[k2], (dict, tuple, type(None)))
                              else bool(r[k2]) for r in races]))
           for k2 in ["tan_pay", "fuku_pay", "wakuren", "umaren", "umatan",
                      "sanpuku", "sanrentan", "wide"]}
    print("payout coverage:", {k2: round(v, 4) for k2, v in cov.items()})


if __name__ == "__main__":
    main()
