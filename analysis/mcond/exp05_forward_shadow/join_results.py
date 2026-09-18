# -*- coding: utf-8 -*-
"""
join_results.py — 確定結果を予測レコードとは別テーブルへ結合する (spec §13)
================================================================================
予測レコード (exp05fs_predictions/{date}/{rid}_{hash}_rev{n}.json) 自体は一切編集しない。
結果は data/_research/mcond/exp05fs_results/{date}.parquet に別途保存し、主評価スクリプトが
race_id×horse_id で結合する。
実行: python -m analysis.mcond.exp05_forward_shadow.join_results --date 20260919
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))

RESULTS_DIR = BASE / "data" / "_research" / "mcond" / "exp05fs_results"


def build(date_str: str, save: bool = True) -> pd.DataFrame:
    kekka_path = BASE / "data" / "kekka" / f"{date_str}.csv"
    if not kekka_path.exists():
        raise FileNotFoundError(f"{kekka_path} が無い (結果未確定、または日付誤り)")
    for enc in ("cp932", "utf-8-sig", "utf-8"):
        try:
            k = pd.read_csv(kekka_path, encoding=enc, low_memory=False)
            break
        except Exception:
            continue
    else:
        raise ValueError(f"{kekka_path} を読めるエンコーディングが無い")

    rid_col = next((c for c in k.columns if "レースID" in c), None)
    ban_col = next((c for c in k.columns if c == "馬番"), None)
    fin_col = next((c for c in k.columns if "着順" in c and "確定" in c), None) or \
        next((c for c in k.columns if c == "着順"), None)
    if not (rid_col and ban_col and fin_col):
        raise ValueError(f"kekka CSVの列を特定できない: {list(k.columns)[:20]}")

    out = pd.DataFrame({
        "rid16": k[rid_col].astype(str).str.replace(r"\D", "", regex=True).str[:16],
        "ban": pd.to_numeric(k[ban_col], errors="coerce"),
        "finish_position": pd.to_numeric(k[fin_col], errors="coerce"),
    }).dropna(subset=["ban"])
    out["ban"] = out["ban"].astype(int)
    out["win_flag"] = (out["finish_position"] == 1).astype("Int64")
    out["top3_flag"] = (out["finish_position"] <= 3).astype("Int64")
    win_col = next((c for c in k.columns if "単勝配当" in c or "単勝払戻" in c), None)
    fuku_col = next((c for c in k.columns if "複勝配当" in c or "複勝払戻" in c), None)
    if win_col:
        out["win_payout"] = pd.to_numeric(
            k[win_col].astype(str).str.extract(r"([\d.]+)")[0], errors="coerce")
    if fuku_col:
        out["place_payout"] = pd.to_numeric(
            k[fuku_col].astype(str).str.extract(r"([\d.]+)")[0], errors="coerce")
    out["result_source"] = str(kekka_path.relative_to(BASE))
    out["result_source_hash"] = None  # 週次CSVは確定後も更新され得るためファイルhashは付与しない
    out["settlement_time"] = pd.Timestamp.now().isoformat()

    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out.to_parquet(RESULTS_DIR / f"{date_str}.parquet", index=False)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    args = ap.parse_args()
    out = build(args.date)
    print(f"[join_results] {args.date}: {len(out)}行  win={out['win_flag'].sum()}  "
          f"top3={out['top3_flag'].sum()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
