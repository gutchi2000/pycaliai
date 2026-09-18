# -*- coding: utf-8 -*-
"""
market.py — 購入可能時点の単勝市場確率 (取得時刻つき)
====================================================
入力: data/Time _series_odds/TANPUK_*.csv
  列: レースID, 区分 (1=途中 / 4=確定), 月日時分 (MMDDhhmm), 頭数, 単勝票数, 複勝票数, {n}単, {n}複Lo, {n}複Hi
  1レースあたり 区分1 がほぼ3回 (前日23時台 / 当日9時前後 / 発走約35分前) + 区分4 (確定) 1回。

出す時点 (snap):
  pre   : 当日の区分1のうち、確定(区分4)時刻より 15 分以上前で最も遅いもの
          = 本番の購入判断 (T-10) より前に確実に取れた最後の価格。主検定用。
  am9   : 当日の区分1のうち 09:30 以前で最も遅いもの (感度分析用)
  final : 区分4 (確定)。**特徴量には使わない**。参照・監査用のみ。

保存列 (馬単位):
  rid16, ban, snap, snap_mmddhhmm, final_mmddhhmm, min_before_final,
  odds, implied_raw (=1/odds), race_implied_sum (=overround), pi (=implied_raw/sum),
  flag_missing (オッズ無し/1.0以下), flag_prevday (当日スナップが無い), n_field
overround の除去: レース内で implied_raw を比例正規化 (proportional)。
出力: data/_research/mcond/market.parquet
実行: python -m analysis.mcond.market
"""
from __future__ import annotations
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
ODIR = BASE / "data" / "Time _series_odds"
OUT = BASE / "data" / "_research" / "mcond" / "market.parquet"
MIN_GAP_PRE = 15      # 確定の何分前までを「購入判断前」とみなすか
AM9_LIMIT = 930       # am9 スナップの上限時刻 (hhmm)


def _minutes(mmddhhmm: np.ndarray, year: np.ndarray) -> np.ndarray:
    """MMDDhhmm → 年内通し分 (年またぎは rid の年で揃える前提)。"""
    mm = mmddhhmm // 1000000
    dd = (mmddhhmm // 10000) % 100
    hh = (mmddhhmm // 100) % 100
    mi = mmddhhmm % 100
    ts = pd.to_datetime(dict(year=year, month=mm, day=dd, hour=hh, minute=mi), errors="coerce")
    return ts


def main() -> None:
    files = sorted(glob.glob(str(ODIR / "TANPUK_*.csv")))
    tp = pd.concat([pd.read_csv(f, encoding="cp932", low_memory=False) for f in files],
                   ignore_index=True)
    c = list(tp.columns)
    RID, KB, TM = c[0], c[1], c[2]
    tan = {}
    for x in c:
        m = re.match(r"^\s*(\d+)\s*単\s*$", str(x))
        if m:
            tan[int(m.group(1))] = x
    tp["rid16"] = tp[RID].astype(str).str.replace(r"\D", "", regex=True).str[:16]
    tp = tp[tp["rid16"].str.len() == 16].copy()
    tp["year"] = tp["rid16"].str[:4].astype(int)
    tp = tp[tp["year"] >= 2013]
    tp[KB] = pd.to_numeric(tp[KB], errors="coerce")
    tp[TM] = pd.to_numeric(tp[TM], errors="coerce")
    tp = tp.dropna(subset=[KB, TM])
    tp[TM] = tp[TM].astype(np.int64)
    # 年末開催の前日23時 (12/27 等) は rid の年と同じ。1/5 開催の前日 (1/4) も同年なので問題なし。
    tp["ts"] = _minutes(tp[TM].to_numpy(), tp["year"].to_numpy())
    tp["race_mmdd"] = tp["rid16"].str[4:8].astype(int)
    tp["snap_mmdd"] = tp[TM] // 10000
    tp["hhmm"] = tp[TM] % 10000
    tancols = list(tan.values())
    tp[tancols] = tp[tancols].apply(pd.to_numeric, errors="coerce")
    print(f"TANPUK rows={len(tp):,} races={tp.rid16.nunique():,}")

    fin = tp[tp[KB] == 4].groupby("rid16").agg(final_ts=("ts", "max"), final_mmddhhmm=(TM, "max"))
    mid = tp[tp[KB] == 1].merge(fin, left_on="rid16", right_index=True, how="inner")
    mid["gap_min"] = (mid["final_ts"] - mid["ts"]).dt.total_seconds() / 60.0
    sameday = mid["snap_mmdd"] == mid["race_mmdd"]

    pre = mid[sameday & (mid["gap_min"] >= MIN_GAP_PRE)].sort_values("ts").groupby("rid16").tail(1)
    am9 = mid[sameday & (mid["hhmm"] <= AM9_LIMIT) & (mid["gap_min"] >= MIN_GAP_PRE)] \
        .sort_values("ts").groupby("rid16").tail(1)
    final = tp[tp[KB] == 4].sort_values("ts").groupby("rid16").tail(1).merge(
        fin, left_on="rid16", right_index=True)
    final["gap_min"] = 0.0

    rows = []
    for snap, df in (("pre", pre), ("am9", am9), ("final", final)):
        long = df.melt(id_vars=["rid16", "year", TM, "final_mmddhhmm", "gap_min"],
                       value_vars=tancols, var_name="col", value_name="odds")
        long["ban"] = long["col"].str.extract(r"(\d+)").astype(int)
        long = long.dropna(subset=["odds"])
        long = long[long["odds"] > 0]
        long["snap"] = snap
        rows.append(long)
    m = pd.concat(rows, ignore_index=True).rename(columns={TM: "snap_mmddhhmm",
                                                          "gap_min": "min_before_final"})
    m["flag_missing"] = m["odds"] <= 1.0
    m["implied_raw"] = np.where(m["flag_missing"], np.nan, 1.0 / m["odds"])
    g = m.groupby(["rid16", "snap"])
    m["race_implied_sum"] = g["implied_raw"].transform("sum")
    m["n_field"] = g["ban"].transform("count")
    m["pi"] = m["implied_raw"] / m["race_implied_sum"]
    m["flag_prevday"] = False  # 当日スナップのみ採用しているので常に False (記録用)
    m = m.drop(columns=["col"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    m.to_parquet(OUT, index=False)
    print(f"saved {len(m):,} rows -> {OUT}")

    for s in ("pre", "am9", "final"):
        x = m[m.snap == s].drop_duplicates("rid16")
        print(f"  {s:<5} races={len(x):,}  確定までの分: 中央値 {x.min_before_final.median():.0f} "
              f"/ 5%点 {x.min_before_final.quantile(.05):.0f} / 95%点 {x.min_before_final.quantile(.95):.0f}"
              f"  overround 中央値 {x.race_implied_sum.median():.3f}")


if __name__ == "__main__":
    main()
