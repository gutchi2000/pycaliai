"""
build_course_affinity.py
========================
train 期間 (~2022) から「種牡馬 / 母父馬 / 騎手 / 馬個体 × コース」の集計テーブルを生成。
リーク防止のため train のみで fit。 学習・推論時は course_affinity_feats.py で JOIN。

集計テーブル (data/ 以下に parquet 4 つ):
  course_affinity_sire.parquet
    key = (sire, place, td, dist_bin)        => n, top3, top3_rate, win_rate
  course_affinity_msire.parquet
    key = (msire, place, td, dist_bin)       => 同上
  course_affinity_jockey.parquet
    key = (jockey_code, place, td)           => 同上
  course_affinity_horse_dist.parquet
    key = (horse_id, dist_bin)               => 同上

距離区分:
  short:  <= 1400
  mile:   1401-1800
  middle: 1801-2200
  long:   >= 2201

最小サンプル: n < 5 の組合せは除外 (信頼性低い → 推論時 NaN)。

実行: python build_course_affinity.py
"""
from __future__ import annotations
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE       = Path(__file__).parent
MASTER_CSV = BASE / "data/master_v2_20130105-20251228.csv"
OUT_SIRE   = BASE / "data/course_affinity_sire.parquet"
OUT_MSIRE  = BASE / "data/course_affinity_msire.parquet"
OUT_JOCKEY = BASE / "data/course_affinity_jockey.parquet"
OUT_HORSE  = BASE / "data/course_affinity_horse_dist.parquet"

MIN_N = 5

DIST_BINS = [
    ("short",  0,    1400),
    ("mile",   1401, 1800),
    ("middle", 1801, 2200),
    ("long",   2201, 9999),
]


def dist_bin_label(d: float) -> str | None:
    if pd.isna(d): return None
    d = int(d)
    for name, lo, hi in DIST_BINS:
        if lo <= d <= hi:
            return name
    return None


def aggregate(df: pd.DataFrame, group_cols: list, label: str, out_path: Path):
    g = df.groupby(group_cols, sort=False)
    agg = g.agg(
        n=("着順", "size"),
        top3=("is_top3", "sum"),
        wins=("is_win", "sum"),
    ).reset_index()
    agg = agg[agg["n"] >= MIN_N].copy()
    agg["top3_rate"] = agg["top3"] / agg["n"]
    agg["win_rate"]  = agg["wins"] / agg["n"]
    print(f"  [{label}]  groups={len(agg):,}  "
          f"n_total={int(agg['n'].sum()):,}  "
          f"top3_rate_avg={agg['top3_rate'].mean():.4f}  "
          f"win_rate_avg={agg['win_rate'].mean():.4f}")
    agg.to_parquet(out_path, index=False)
    print(f"  saved: {out_path}  ({out_path.stat().st_size/1024:.1f} KB)")


def main():
    print("=" * 70)
    print("build_course_affinity.py  (train 期間 ~2022 のみで fit)")
    print("=" * 70)

    cols = ["種牡馬", "母父馬", "騎手コード", "血統登録番号",
            "場所", "芝・ダ", "距離", "着順", "split"]
    print(f"[load] {MASTER_CSV}  (usecols=9)")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", usecols=cols, low_memory=False)
    print(f"  rows={len(df):,}")

    df = df[df["split"] == "train"].copy()
    print(f"  train rows={len(df):,}")
    df["着順"] = pd.to_numeric(df["着順"], errors="coerce")
    df = df.dropna(subset=["着順", "場所", "芝・ダ", "距離"])
    df["距離"] = df["距離"].astype(int)
    df["dist_bin"] = df["距離"].apply(dist_bin_label)
    df = df.dropna(subset=["dist_bin"])
    df["is_top3"] = (df["着順"].astype(int) <= 3).astype(int)
    df["is_win"]  = (df["着順"].astype(int) == 1).astype(int)
    print(f"  after dropna rows={len(df):,}")

    print(f"\n[1/4] sire × 場所 × 芝・ダ × dist_bin")
    sub = df.dropna(subset=["種牡馬"]).copy()
    aggregate(sub, ["種牡馬", "場所", "芝・ダ", "dist_bin"],
              "sire", OUT_SIRE)

    print(f"\n[2/4] msire × 場所 × 芝・ダ × dist_bin")
    sub = df.dropna(subset=["母父馬"]).copy()
    aggregate(sub, ["母父馬", "場所", "芝・ダ", "dist_bin"],
              "msire", OUT_MSIRE)

    print(f"\n[3/4] jockey × 場所 × 芝・ダ")
    sub = df.dropna(subset=["騎手コード"]).copy()
    aggregate(sub, ["騎手コード", "場所", "芝・ダ"],
              "jockey", OUT_JOCKEY)

    print(f"\n[4/4] horse_id × dist_bin")
    sub = df.dropna(subset=["血統登録番号"]).copy()
    aggregate(sub, ["血統登録番号", "dist_bin"],
              "horse_dist", OUT_HORSE)

    print("\nDONE")


if __name__ == "__main__":
    main()
