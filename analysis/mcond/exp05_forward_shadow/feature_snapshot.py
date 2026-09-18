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
  - analysis.mcond.exp05_forward_shadow.frozen_encode  C1の凍結エンコード
  - analysis.mcond.exp05_forward_shadow.horse_identity の 2025年末状態 (dyn_skill, career_runs)

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
from analysis.mcond.exp05_forward_shadow import frozen_encode  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT_DIR = BASE / "data/_research/mcond/exp05fs_features"
HORSE_STATE = HERE / "out/horse_state_2025.json"
FEATURE_SCHEMA_VERSION = "exp05fs_v1"

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

C2_UNAVAILABLE = ["raw_jockey_same", "raw_cls_chg", "raw_jq_delta", "raw_jt_pair"]


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


def compute_c2(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = pd.DataFrame(index=df.index)
    found = {}
    interval = pd.to_numeric(df.get("間隔"), errors="coerce")
    out["raw_log_int"] = np.log(interval.clip(lower=1))
    found["raw_log_int"] = "間隔" in df.columns

    dist = pd.to_numeric(df.get("距離"), errors="coerce")
    prev_dist = pd.to_numeric(df.get("前距離"), errors="coerce")
    out["raw_dist_chg"] = dist - prev_dist
    found["raw_dist_chg"] = ("距離" in df.columns) and ("前距離" in df.columns)

    if "場所" in df.columns and "前走場所" in df.columns:
        out["raw_venue_chg"] = (df["場所"].astype(str) != df["前走場所"].astype(str)).astype(float)
        found["raw_venue_chg"] = True
    else:
        out["raw_venue_chg"] = np.nan
        found["raw_venue_chg"] = False

    if "芝・ダ" in df.columns and "前芝・ダ" in df.columns:
        out["raw_surface_chg"] = (df["芝・ダ"].astype(str) != df["前芝・ダ"].astype(str)).astype(float)
        found["raw_surface_chg"] = True
    else:
        out["raw_surface_chg"] = np.nan
        found["raw_surface_chg"] = False

    for c in C2_UNAVAILABLE:
        out[c] = np.nan
        found[c] = False  # v1の既知の制約 (README.md参照): 履歴突合が必要でweekly CSV単体では出せない
    return out, found


def compute_c3(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    state = json.loads(HORSE_STATE.read_text(encoding="utf-8"))["horses"] if HORSE_STATE.exists() else {}
    names = df.get("馬名", pd.Series([None] * len(df), index=df.index)).astype(str)
    dyn_mu = names.map(lambda n: state.get(n, {}).get("dyn_mu"))
    career = names.map(lambda n: state.get(n, {}).get("career_runs"))
    found_rate = float(dyn_mu.notna().mean()) if len(dyn_mu) else 0.0

    out = pd.DataFrame(index=df.index)
    out["dyn_skill_mu"] = pd.to_numeric(dyn_mu, errors="coerce")
    MU0 = 25.0
    out["dyn_skill_mu"] = out["dyn_skill_mu"].fillna(MU0)  # 新馬等は共通初期値 (dyn_skill.pyのMU0と同じ)

    rid_col = "レースID(新/馬番無)" if "レースID(新/馬番無)" in df.columns else "レースID(新)"
    tmp = pd.DataFrame({"rid": df[rid_col].astype(str), "mu": out["dyn_skill_mu"]})
    g = tmp.groupby("rid")["mu"]
    n = g.transform("count")
    tot = g.transform("sum")
    others_mean = (tot - out["dyn_skill_mu"]) / (n - 1).clip(lower=1)
    out["horse_skill_minus_field"] = out["dyn_skill_mu"] - others_mean

    out["raw_career_runs"] = pd.to_numeric(career, errors="coerce").fillna(0.0)
    out["raw_days_since"] = pd.to_numeric(df.get("間隔"), errors="coerce")

    found = {"dyn_skill_mu": found_rate > 0, "horse_skill_minus_field": found_rate > 0,
            "raw_career_runs": found_rate > 0, "raw_days_since": "間隔" in df.columns}
    meta = {"horse_state_match_rate": found_rate,
           "note": "dyn_skill_mu/raw_career_runsは2025年末状態からの繰越、2026年の既走分は未反映 (README.md)"}
    return out, found, meta


def build(date_str: str, save: bool = True) -> tuple[pd.DataFrame, dict]:
    df = load_and_prepare(date_str)
    c1, found_c1 = frozen_encode.encode_c1(df)
    c2, found_c2 = compute_c2(df)
    c3, found_c3, c3_meta = compute_c3(df)

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

    coverage = {**{f"c1__{k}": v for k, v in found_c1.items()},
               **{k: v for k, v in found_c2.items()},
               **{k: v for k, v in found_c3.items()}}
    meta = {"date": date_str, "n_rows": int(len(feats)), "n_races": int(key["rid16"].nunique()),
           "column_found": coverage,
           "n_columns_missing_entirely": int(sum(1 for v in coverage.values() if not v)),
           **c3_meta}
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
