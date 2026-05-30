"""
course_affinity_feats.py
=========================
build_course_affinity.py で生成した集計テーブル 4 つを df に JOIN するヘルパー。
race_relative_feats と同じパターン: bundle に保存される mode を見て学習・推論
両方で同じ列を追加する。

公開関数:
  add_course_affinity_feats(df, mode='v1')
  added_columns(mode='v1')

追加列 (mode='v1', 12 列):
  sire_n, sire_top3_rate, sire_win_rate
  msire_n, msire_top3_rate, msire_win_rate
  jockey_course_n, jockey_course_top3_rate, jockey_course_win_rate
  horse_dist_n, horse_dist_top3_rate, horse_dist_win_rate

NaN 処理: テーブルにマッチしない (n<5 で除外、新規馬等) は NaN のままにし、
         LightGBM の欠損値処理に任せる (-9999 fillna は呼び出し側で実施)。
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent

PATH_SIRE   = BASE / "data/course_affinity_sire.parquet"
PATH_MSIRE  = BASE / "data/course_affinity_msire.parquet"
PATH_JOCKEY = BASE / "data/course_affinity_jockey.parquet"
PATH_HORSE  = BASE / "data/course_affinity_horse_dist.parquet"

DIST_BINS = [
    ("short",  0,    1400),
    ("mile",   1401, 1800),
    ("middle", 1801, 2200),
    ("long",   2201, 9999),
]


def _dist_bin_label(d: float) -> str | None:
    if pd.isna(d): return None
    d = int(d)
    for name, lo, hi in DIST_BINS:
        if lo <= d <= hi:
            return name
    return None


_CACHE: dict = {}


def _load_tables():
    if "loaded" in _CACHE:
        return
    _CACHE["sire"]   = pd.read_parquet(PATH_SIRE)
    _CACHE["msire"]  = pd.read_parquet(PATH_MSIRE)
    _CACHE["jockey"] = pd.read_parquet(PATH_JOCKEY)
    _CACHE["horse"]  = pd.read_parquet(PATH_HORSE)
    _CACHE["loaded"] = True


def add_course_affinity_feats(df: pd.DataFrame, mode: str = "v1") -> pd.DataFrame:
    """df に集計テーブル 4 つを JOIN。

    必要カラム: 種牡馬, 母父馬, 騎手コード, 血統登録番号, 場所, 芝・ダ, 距離
    """
    _load_tables()
    df = df.copy()

    # 距離 → dist_bin
    if "距離" in df.columns:
        df["_dist_bin_tmp"] = pd.to_numeric(df["距離"], errors="coerce").apply(_dist_bin_label)

    # 1. sire
    if "種牡馬" in df.columns and "_dist_bin_tmp" in df.columns:
        sire = _CACHE["sire"].rename(columns={
            "n": "sire_n", "top3_rate": "sire_top3_rate", "win_rate": "sire_win_rate",
        })[["種牡馬", "場所", "芝・ダ", "dist_bin", "sire_n", "sire_top3_rate", "sire_win_rate"]]
        df = df.merge(
            sire,
            left_on=["種牡馬", "場所", "芝・ダ", "_dist_bin_tmp"],
            right_on=["種牡馬", "場所", "芝・ダ", "dist_bin"],
            how="left",
        )
        df = df.drop(columns=[c for c in ["dist_bin"] if c in df.columns])

    # 2. msire
    if "母父馬" in df.columns and "_dist_bin_tmp" in df.columns:
        msire = _CACHE["msire"].rename(columns={
            "n": "msire_n", "top3_rate": "msire_top3_rate", "win_rate": "msire_win_rate",
        })[["母父馬", "場所", "芝・ダ", "dist_bin", "msire_n", "msire_top3_rate", "msire_win_rate"]]
        df = df.merge(
            msire,
            left_on=["母父馬", "場所", "芝・ダ", "_dist_bin_tmp"],
            right_on=["母父馬", "場所", "芝・ダ", "dist_bin"],
            how="left",
        )
        df = df.drop(columns=[c for c in ["dist_bin"] if c in df.columns])

    # 3. jockey × course (no dist_bin)
    if "騎手コード" in df.columns:
        jk = _CACHE["jockey"].rename(columns={
            "n": "jockey_course_n",
            "top3_rate": "jockey_course_top3_rate",
            "win_rate":  "jockey_course_win_rate",
        })[["騎手コード", "場所", "芝・ダ",
            "jockey_course_n", "jockey_course_top3_rate", "jockey_course_win_rate"]]
        df = df.merge(jk, on=["騎手コード", "場所", "芝・ダ"], how="left")

    # 4. horse × dist_bin
    if "血統登録番号" in df.columns and "_dist_bin_tmp" in df.columns:
        hd = _CACHE["horse"].rename(columns={
            "n": "horse_dist_n",
            "top3_rate": "horse_dist_top3_rate",
            "win_rate":  "horse_dist_win_rate",
        })[["血統登録番号", "dist_bin",
            "horse_dist_n", "horse_dist_top3_rate", "horse_dist_win_rate"]]
        df = df.merge(
            hd,
            left_on=["血統登録番号", "_dist_bin_tmp"],
            right_on=["血統登録番号", "dist_bin"],
            how="left",
        )
        df = df.drop(columns=[c for c in ["dist_bin"] if c in df.columns])

    if "_dist_bin_tmp" in df.columns:
        df = df.drop(columns=["_dist_bin_tmp"])

    return df


def added_columns(mode: str = "v1"):
    return [
        "sire_n", "sire_top3_rate", "sire_win_rate",
        "msire_n", "msire_top3_rate", "msire_win_rate",
        "jockey_course_n", "jockey_course_top3_rate", "jockey_course_win_rate",
        "horse_dist_n", "horse_dist_top3_rate", "horse_dist_win_rate",
    ]


if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    # スモークテスト: master_v2 valid 行に対して JOIN し、 NaN 率を測定
    print("[smoke] master_v2 valid 行で JOIN")
    df = pd.read_csv(
        "data/master_v2_20130105-20251228.csv", encoding="utf-8-sig",
        usecols=["種牡馬", "母父馬", "騎手コード", "血統登録番号",
                 "場所", "芝・ダ", "距離", "split"],
        low_memory=False,
    )
    df = df[df["split"] == "valid"].head(10000).copy()
    out = add_course_affinity_feats(df)
    added = added_columns()
    print(f"  rows: {len(out):,}")
    print(f"  added cols: {len(added)}")
    for c in added:
        nan_rate = out[c].isna().mean() * 100
        valid = out[c].dropna()
        print(f"  {c:32s}  NaN={nan_rate:5.1f}%  "
              f"mean={valid.mean():.4f}  min={valid.min():.4f}  max={valid.max():.4f}")
