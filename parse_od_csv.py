"""
parse_od_csv.py
===============
OD*.CSV (JRA-VAN形式オッズファイル) から単勝・複勝オッズを抽出し、
weekly CSVのレースIDに変換する。

OD CSVフォーマット (91列):
  col0: レースID(10桁) = PP YY KD RR HH
    PP=場コード, YY=年下2桁, K=回, D=日, RR=R番号, HH=馬番
  col1: 出走頭数
  col4: 馬番
  col7: 単勝オッズ
  col8: 複勝オッズ下限
  col9: 複勝オッズ上限

レースIDマッピング:
  OD  "06 26 32 01" → 場06, 2026年, 3回2日目, R01
  Weekly "20260329 06 03 02 01" → 同じレース

Usage:
  from parse_od_csv import load_od_odds
  odds_df = load_od_odds("E:/競馬過去走データ/OD260329.CSV", date="20260329")
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path


def parse_od_race_key(od_race_id_str: str) -> dict:
    """OD形式の10桁レースIDをパースする。"""
    s = od_race_id_str.zfill(10)
    return {
        "place": s[:2],
        "year": s[2:4],
        "kai": int(s[4]) if len(s) > 4 else 0,    # 回
        "nichi": int(s[5]) if len(s) > 5 else 0,   # 日
        "race_num": s[6:8],
        "horse_num": int(s[8:10]),
    }


def od_to_weekly_race_id(od_race_id_str: str, date: str) -> str:
    """OD形式レースID → weekly CSV形式の16桁レースIDに変換。

    Args:
        od_race_id_str: "0626320101" (10桁)
        date: "20260329" (YYYYMMDD)

    Returns:
        "2026032906030201" (16桁、馬番なし)
    """
    p = parse_od_race_key(od_race_id_str)
    return f"{date}{p['place']}{p['kai']:02d}{p['nichi']:02d}{p['race_num']}"


def load_od_odds(od_csv_path: str | Path, date: str | None = None) -> pd.DataFrame:
    """OD CSVから単勝・複勝オッズを読み込み、weekly形式のrace_idに変換。

    Args:
        od_csv_path: OD*.CSVのパス
        date: YYYYMMDD形式の日付。Noneの場合ファイル名から推定。

    Returns:
        DataFrame with columns:
          race_id (16桁), horse_num, shutsuu, tan_odds, fuku_low, fuku_high
    """
    od_csv_path = Path(od_csv_path)

    # 日付をファイル名から推定: OD260329.CSV → 20260329
    if date is None:
        stem = od_csv_path.stem  # "OD260329"
        ymd = stem.replace("OD", "")
        date = f"20{ymd}"

    od = pd.read_csv(od_csv_path, encoding="cp932", header=None, dtype={0: str})
    od[0] = od[0].str.zfill(10)

    # 抽出
    result = pd.DataFrame({
        "od_id": od[0],
        "horse_num": od[4].astype(int),
        "shutsuu": od[1].astype(int),
        "tan_odds": pd.to_numeric(od[7], errors="coerce"),
        "fuku_low": pd.to_numeric(od[8], errors="coerce"),
        "fuku_high": pd.to_numeric(od[9], errors="coerce"),
    })

    # race_id変換
    result["race_id"] = result["od_id"].apply(
        lambda x: od_to_weekly_race_id(x, date)
    )

    # 基本的なバリデーション
    result = result[result["tan_odds"] > 0].copy()
    result["fuku_mid"] = (result["fuku_low"] + result["fuku_high"]) / 2
    result["ninki"] = result.groupby("race_id")["tan_odds"].rank(
        method="first", ascending=True
    )

    return result[["race_id", "horse_num", "shutsuu", "tan_odds",
                    "fuku_low", "fuku_high", "fuku_mid", "ninki"]]


if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    # テスト
    test_path = r"E:\競馬過去走データ\OD260329.CSV"
    df = load_od_odds(test_path)
    print(f"Loaded: {len(df)} rows, {df['race_id'].nunique()} races")
    print(f"\nSample (first race):")
    race1 = df[df["race_id"] == df["race_id"].iloc[0]].sort_values("ninki")
    print(f"  race_id: {race1['race_id'].iloc[0]}")
    print(f"  {'#':>3} {'Ninki':>5} {'Tansho':>8} {'FukuLo':>8} {'FukuHi':>8}")
    for _, r in race1.head(5).iterrows():
        print(f"  {r['horse_num']:>3} {r['ninki']:>5.0f} {r['tan_odds']:>8.1f} "
              f"{r['fuku_low']:>8.1f} {r['fuku_high']:>8.1f}")

    # Verify mapping against weekly CSV
    import glob
    weekly_path = r"E:\PyCaLiAI\data\weekly\20260329.csv"
    try:
        w = pd.read_csv(weekly_path, encoding="cp932", on_bad_lines="skip", dtype=str)
        w_rids = set(w.iloc[:, 0].dropna().unique())
        od_rids = set(df["race_id"].unique())
        overlap = w_rids & od_rids
        print(f"\nMapping verification:")
        print(f"  OD races: {len(od_rids)}")
        print(f"  Weekly races: {len([r for r in w_rids if r.startswith('20260329')])}")
        print(f"  Overlap: {len(overlap)}")
        if overlap:
            print(f"  Example match: {sorted(overlap)[0]}")
    except Exception as e:
        print(f"\nMapping verification skipped: {e}")
