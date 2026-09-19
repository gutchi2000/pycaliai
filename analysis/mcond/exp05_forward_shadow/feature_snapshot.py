# -*- coding: utf-8 -*-
"""
feature_snapshot.py — serve-safe 117特徴を週次バッチで生成し保存する (spec §9)
================================================================================
市場オッズは発走直前でないと存在しないため per-race real-time が必要だが、表特徴は
production の週次パイプライン (export_weekly_marks.py) と同じカデンスで良い ——
週末の出走表が来た時点で計算しても対象レースより十分前であり時点安全性は保たれる。

再利用 (変更しない):
  - predict_weekly.parse_csv        週次CSVパース
  - export_weekly_marks.py の _SERVE_RENAME と同じマッピング (ここで複製、値は同期を要コメント)
  - serve_history_feats.fill_history_features  hist_*/course_*/jockey_*/騎手・調教師コード
  - analysis.mcond.exp05_forward_shadow.frozen_encode  C1の凍結エンコード (カテゴリ正規化含む)
  - analysis.mcond.exp05_forward_shadow.live_history  C2/C3 (時点安全なchain逐次計算、v2)

出力: data/_research/mcond/exp05fs_features/{date}.parquet (gitignore)
     各列について _missing_<col> フラグ、メタに coverage レポート
実行: python -m analysis.mcond.exp05_forward_shadow.feature_snapshot --date 20260919
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from predict_weekly import parse_csv  # noqa: E402
from grade_feats import class_name_to_ord  # noqa: E402
from analysis.mcond.exp05_forward_shadow import frozen_encode  # noqa: E402
from analysis.mcond.exp05_forward_shadow import live_history as LH  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT_DIR = BASE / "data/_research/mcond/exp05fs_features"
FEATURE_SCHEMA_VERSION = "exp05fs_v2_live_history"

# export_weekly_marks.py の _SERVE_RENAME と同じ定義。あちらを変更したらここも同期すること
# (意図的に複製: exp05_forward_shadow は本番週次処理に依存せず単独で動く設計、spec §11)。
SERVE_RENAME = {
    "R": "Ｒ",
    "前走補正": "prev_hosei", "前走補9": "prev_hosei9",
    "trn_hanro_4f": "trnH_Time1", "trn_hanro_3f": "trnH_Time2",
    "trn_hanro_2f": "trnH_Time3", "trn_hanro_1f": "trnH_Time4",
    "trn_hanro_lap1": "trnH_Lap1", "trn_hanro_lap2": "trnH_Lap2",
    "trn_hanro_lap3": "trnH_Lap3", "trn_hanro_lap4": "trnH_Lap4",
    "trn_hanro_days": "trnH_days_ago",
    "trn_wc_5f": "trnW_5F", "trn_wc_4f": "trnW_4F", "trn_wc_3f": "trnW_3F",
    "trn_wc_lap1": "trnW_Lap1", "trn_wc_lap2": "trnW_Lap2",
    "trn_wc_lap3": "trnW_Lap3", "trn_wc_days": "trnW_days_ago",
}

def load_and_prepare(date_str: str) -> pd.DataFrame:
    path = BASE / "data/weekly" / f"{date_str}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = parse_csv(path)
    rename = {k: v for k, v in SERVE_RENAME.items() if k in df.columns and v not in df.columns}
    df = df.rename(columns=rename)
    try:
        from serve_history_feats import fill_history_features
        fill_history_features(df)
    except Exception as exc:
        print(f"[feature_snapshot] fill_history_features失敗 (fail-open, hist/course/jockey欠損のまま): {exc}")
    return df


def build_chain_target(df: pd.DataFrame, date_str: str) -> pd.DataFrame:
    """live_history.compute_c2_from_chain / compute_dyn_skill_live が要求するスキーマへ変換。"""
    name2hid = LH.name_to_hid_2025()
    names = df.get("馬名", pd.Series([""] * len(df), index=df.index)).astype(str)
    ident = names.map(lambda n: name2hid.get(n, f"NEW:{n}"))
    rid_col = "レースID(新/馬番無)" if "レースID(新/馬番無)" in df.columns else "レースID(新)"
    return pd.DataFrame({
        "ident": ident, "date": pd.Timestamp(date_str),
        "rid16": df[rid_col].astype(str).str[:16],
        "ban": pd.to_numeric(df["馬番"], errors="coerce"),
        "venue": df.get("場所", pd.Series("", index=df.index)),
        "surface": df.get("芝・ダ", pd.Series("", index=df.index)).astype(str).replace({"ダート": "ダ"}),
        "dist": pd.to_numeric(df.get("距離"), errors="coerce"),
        "cls_ord": df.get("クラス名", pd.Series("", index=df.index)).map(class_name_to_ord),
        "jockey": df.get("騎手コード", pd.Series("", index=df.index)).astype(str),
        "trainer": df.get("調教師コード", pd.Series("", index=df.index)).astype(str),
    })


def build(date_str: str, save: bool = True, chain: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    df = load_and_prepare(date_str)
    c1, found_c1 = frozen_encode.encode_c1(df)

    target = build_chain_target(df, date_str)
    as_of = max((d for d in LH.available_2026_result_dates() if d < date_str), default=None)
    if chain is None:
        chain = LH.build_combined_chain(as_of_date=as_of)
    c2 = LH.compute_c2_from_chain(target, chain)
    dyn_state = LH.compute_dyn_skill_live(chain)

    dyn_mu = target["ident"].map(dyn_state["g_mu"])
    MU0 = 25.0
    dyn_mu_filled = dyn_mu.fillna(MU0)
    tmp = pd.DataFrame({"rid": target["rid16"], "mu": dyn_mu_filled})
    g = tmp.groupby("rid")["mu"]
    n, tot = g.transform("count"), g.transform("sum")
    others_mean = (tot - dyn_mu_filled) / (n - 1).clip(lower=1)
    c3 = pd.DataFrame({
        "dyn_skill_mu": dyn_mu_filled,
        "horse_skill_minus_field": dyn_mu_filled - others_mean,
        "raw_career_runs": target["ident"].map(dyn_state["n"]).fillna(0.0),
        "raw_days_since": pd.to_numeric(df.get("間隔"), errors="coerce"),
    })
    dyn_match_rate = float(dyn_mu.notna().mean()) if len(dyn_mu) else 0.0

    found_c2 = {c: bool(c2[c].notna().any()) for c in c2.columns}
    found_c3 = {"dyn_skill_mu": dyn_match_rate > 0, "horse_skill_minus_field": dyn_match_rate > 0,
               "raw_career_runs": dyn_match_rate > 0, "raw_days_since": "間隔" in df.columns}

    rid_col = "レースID(新/馬番無)" if "レースID(新/馬番無)" in df.columns else "レースID(新)"
    key = pd.DataFrame({
        "rid16": df[rid_col].astype(str).str[:16],
        "ban": pd.to_numeric(df["馬番"], errors="coerce").astype("Int64"),
        "horse_name": df.get("馬名", ""),
    })
    feats = pd.concat([key.reset_index(drop=True), c1.reset_index(drop=True),
                       c2.reset_index(drop=True), c3.reset_index(drop=True)], axis=1)
    feats["feature_snapshot_generated_at"] = pd.Timestamp.now().isoformat()
    feats["feature_schema_version"] = FEATURE_SCHEMA_VERSION
    feats["source_weekly_csv"] = f"data/weekly/{date_str}.csv"
    feats["chain_as_of_date"] = as_of

    coverage = {**{f"c1__{k}": v for k, v in found_c1.items()},
               **{k: v for k, v in found_c2.items()},
               **{k: v for k, v in found_c3.items()}}
    meta = {"date": date_str, "n_rows": int(len(feats)), "n_races": int(key["rid16"].nunique()),
           "column_found": coverage,
           "n_columns_missing_entirely": int(sum(1 for v in coverage.values() if not v)),
           "horse_state_match_rate": dyn_match_rate, "chain_as_of_date": as_of,
           "chain_n_rows": int(len(chain))}
    if save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        feats.to_parquet(OUT_DIR / f"{date_str}.parquet", index=False)
        (OUT_DIR / f"{date_str}_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return feats, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYYMMDD (data/weekly/{date}.csv)")
    args = ap.parse_args()
    feats, meta = build(args.date)
    print(f"[feature_snapshot] {args.date}: {meta['n_rows']}頭 / {meta['n_races']}レース  "
          f"完全欠損列={meta['n_columns_missing_entirely']}  "
          f"dyn_skill状態一致率={meta.get('horse_state_match_rate', float('nan')):.3f}")
    missing_cols = [k for k, v in meta["column_found"].items() if not v]
    if missing_cols:
        print(f"  今週欠損の列 ({len(missing_cols)}): {missing_cols}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
