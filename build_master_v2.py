"""
build_master_v2.py
===================
master_kako5.csv に以下をマージして master_v2_20130105-20251228.csv を作る。

  Stage 1-3: 補正タイム (data/hosei/H_20130105-20251228.csv)
             - 前走補正, 前走補9 のみ (補正=今回レース時点 = 目的変数と同時決定 → リーク)
  Stage 1-4: 調教タイム (data/training/H,W-*.csv)
             - 馬名 + 直前の年月日 <= 日付 で時系列マッチ
  Stage 1-5: 血統登録番号ベースの自前履歴集計
             - 同コース(場所x芝ダx距離) 着度数 1-3着
             - 同騎手 直近N走着度数
             - 時系列カットオフ付き (date < current date)

決定的・再現可能: seed 無し (純粋な merge/集計)、入力ファイルが同じなら同じ出力。

実行:
    python build_master_v2.py [--stage 3|4|5|all]

出力:
    data/master_v2_20130105-20251228.csv
    reports/master_v2_build_log.json
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE          = Path(__file__).parent
MASTER_IN     = BASE / "data/master_kako5.csv"
MASTER_OUT    = BASE / "data/master_v2_20130105-20251228.csv"
HOSEI_CSV     = BASE / "data/hosei/H_20130105-20251228.csv"
TRAINING_H    = BASE / "data/training/H-20150401-20260313.csv"
TRAINING_W    = BASE / "data/training/W-20150401-20260313.csv"
BUILD_LOG     = BASE / "reports/master_v2_build_log.json"

COL_RID       = "レースID(新)"
COL_BAN       = "馬番"
COL_DATE      = "日付"
COL_NAME      = "馬名"
COL_PEDIGREE  = "血統登録番号"
COL_PLACE     = "場所"
COL_SURFACE   = "芝・ダ"
COL_DIST      = "距離"
COL_JOCKEY    = "騎手コード"
COL_JYUN      = "着順"


def log(msg: str):
    print(msg, flush=True)


# ============================================================
# Stage 1-3: 補正タイム merge
# ============================================================
def merge_hosei(df: pd.DataFrame) -> pd.DataFrame:
    log(f"\n=== Stage 1-3: 補正タイム merge ({HOSEI_CSV.name}) ===")
    h = pd.read_csv(HOSEI_CSV, encoding="cp932",
                    usecols=["レースID(新)", "馬番", "前走補正", "前走補9"])
    h = h.rename(columns={"前走補正": "prev_hosei", "前走補9": "prev_hosei9"})
    h["馬番"] = pd.to_numeric(h["馬番"], errors="coerce").astype("Int64")
    h = h.drop_duplicates(subset=["レースID(新)", "馬番"])
    log(f"  hosei rows: {len(h):,}")

    df = df.copy()
    df["馬番"] = pd.to_numeric(df["馬番"], errors="coerce").astype("Int64")
    before = len(df)
    df = df.merge(h, on=["レースID(新)", "馬番"], how="left")
    assert len(df) == before, f"行数変化 {before} → {len(df)}"

    cov_hosei  = df["prev_hosei"].notna().mean() * 100
    cov_hosei9 = df["prev_hosei9"].notna().mean() * 100
    log(f"  カバレッジ: prev_hosei={cov_hosei:.1f}% / prev_hosei9={cov_hosei9:.1f}%")
    return df


# ============================================================
# Stage 1-4: 調教タイム merge
# ============================================================
def merge_training(df: pd.DataFrame) -> pd.DataFrame:
    log(f"\n=== Stage 1-4: 調教 H (坂路) + W (CW) merge ===")
    df = df.copy()
    df[COL_DATE] = pd.to_numeric(df[COL_DATE], errors="coerce").astype("Int64")

    # 馬名正規化 (training ファイルは JRA-VAN 系で少し表記ずれ可能性あり)
    df["_name_key"] = df[COL_NAME].astype(str).str.strip()

    # --- H (坂路) ---
    log(f"  読み込み: {TRAINING_H.name}")
    H = pd.read_csv(TRAINING_H, encoding="cp932", low_memory=False)
    # 列名: 場所, 年月日, 馬名, Time1-4, Lap1-4
    H = H.rename(columns={"年月日": "train_date", "馬名": "_name_key"})
    H["train_date"] = pd.to_numeric(H["train_date"], errors="coerce").astype("Int64")
    H["_name_key"]  = H["_name_key"].astype(str).str.strip()
    H = H.dropna(subset=["train_date", "_name_key"])
    H = H[["_name_key", "train_date", "Time1", "Time2", "Time3", "Time4",
           "Lap1", "Lap2", "Lap3", "Lap4"]].copy()
    # Lap1..Time4 数値化
    for c in ["Time1", "Time2", "Time3", "Time4", "Lap1", "Lap2", "Lap3", "Lap4"]:
        H[c] = pd.to_numeric(H[c], errors="coerce")

    # --- W (CW) ---
    log(f"  読み込み: {TRAINING_W.name}")
    W = pd.read_csv(TRAINING_W, encoding="cp932", low_memory=False)
    W = W.rename(columns={"年月日": "train_date", "馬名": "_name_key"})
    W["train_date"] = pd.to_numeric(W["train_date"], errors="coerce").astype("Int64")
    W["_name_key"]  = W["_name_key"].astype(str).str.strip()
    W = W.dropna(subset=["train_date", "_name_key"])
    keep_w = ["_name_key", "train_date"]
    for c in ["5F", "4F", "3F", "Lap1", "Lap2", "Lap3"]:
        if c in W.columns:
            W[c] = pd.to_numeric(W[c], errors="coerce")
            keep_w.append(c)
    W = W[keep_w].copy()

    log(f"  H rows={len(H):,}  W rows={len(W):,}")

    # --- 馬×日付ごと最終調教を 1 行集約 (最大 2 本/週の想定で、直近 1 件を採用) ---
    def _latest_before(tr: pd.DataFrame, race_df: pd.DataFrame,
                       prefix: str, value_cols: list[str]) -> pd.DataFrame:
        """race_df の (_name_key, 日付) に対して tr の train_date < 日付 で最も近い行を付与."""
        tr = tr.sort_values(["_name_key", "train_date"]).reset_index(drop=True)
        key = race_df[["_name_key", COL_DATE]].drop_duplicates().sort_values([COL_DATE]).reset_index(drop=True)
        # merge_asof 用
        tr_m = tr.rename(columns={"train_date": "_t_date"})
        tr_m = tr_m.sort_values("_t_date")
        key_m = key.sort_values(COL_DATE)
        merged = pd.merge_asof(
            key_m,
            tr_m,
            left_on=COL_DATE, right_on="_t_date",
            by="_name_key",
            direction="backward",
            allow_exact_matches=False,  # 同日は除く (調教が当日の場合がまれにあるので安全策)
        )
        # days diff
        merged["train_days_ago"] = merged[COL_DATE].astype("Int64") - merged["_t_date"].astype("Int64")
        out_cols = ["_name_key", COL_DATE] + value_cols + ["train_days_ago"]
        merged = merged[out_cols].copy()
        merged.columns = ["_name_key", COL_DATE] + [f"{prefix}_{c}" for c in value_cols] + [f"{prefix}_days_ago"]
        return merged

    h_feat = _latest_before(H, df, "trnH",
                            ["Time1", "Time2", "Time3", "Time4", "Lap1", "Lap2", "Lap3", "Lap4"])
    w_val_cols = [c for c in ["5F", "4F", "3F", "Lap1", "Lap2", "Lap3"] if c in W.columns]
    w_feat = _latest_before(W, df, "trnW", w_val_cols)

    before = len(df)
    df = df.merge(h_feat, on=["_name_key", COL_DATE], how="left")
    df = df.merge(w_feat, on=["_name_key", COL_DATE], how="left")
    assert len(df) == before

    cov_h = df["trnH_Time4"].notna().mean() * 100 if "trnH_Time4" in df.columns else 0
    cov_w = df[f"trnW_{w_val_cols[0]}"].notna().mean() * 100 if w_val_cols else 0
    log(f"  カバレッジ: 坂路={cov_h:.1f}% / CW={cov_w:.1f}%")

    df = df.drop(columns=["_name_key"])
    return df


# ============================================================
# Stage 1-5: 血統登録番号ベースの自前履歴集計
# ============================================================
def compute_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """時系列カットオフ (日付 < current) で以下を集計:
    - 同コース (場所×芝ダ×距離帯) 出走数, 1着率, 3着以内率
    - 同騎手 直近 10 走 1着率, 3着以内率
    """
    log("\n=== Stage 1-5: 自前履歴集計 (血統登録番号ベース) ===")
    df = df.copy()
    df[COL_DATE] = pd.to_numeric(df[COL_DATE], errors="coerce").astype("Int64")
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df["_is_win"]  = (df[COL_JYUN] == 1).astype("Int8")
    df["_is_top3"] = (df[COL_JYUN] <= 3).astype("Int8")

    # 距離帯
    def dist_band(d):
        if pd.isna(d): return "?"
        d = int(d)
        if d <= 1400: return "短"
        if d <= 1700: return "マ"
        if d <= 2200: return "中"
        return "長"
    df["_dist_b"] = df[COL_DIST].apply(dist_band)
    df["_course_key"] = (df[COL_PLACE].astype(str) + "|" +
                         df[COL_SURFACE].astype(str) + "|" + df["_dist_b"])

    log(f"  対象: {len(df):,} 行 / {df[COL_PEDIGREE].nunique():,} 頭")

    # --- 同コース累積 (cumsum 系, 馬ごと時系列) ---
    df = df.sort_values([COL_PEDIGREE, COL_DATE]).reset_index(drop=True)
    g = df.groupby([COL_PEDIGREE, "_course_key"])
    df["course_n_prev"]    = g.cumcount()
    df["course_wins_prev"] = g["_is_win"].cumsum().astype("Int64") - df["_is_win"].astype("Int64")
    df["course_top3_prev"] = g["_is_top3"].cumsum().astype("Int64") - df["_is_top3"].astype("Int64")
    df["course_win_rate"]  = np.where(df["course_n_prev"] > 0,
                                      df["course_wins_prev"] / df["course_n_prev"], np.nan)
    df["course_top3_rate"] = np.where(df["course_n_prev"] > 0,
                                      df["course_top3_prev"] / df["course_n_prev"], np.nan)

    # --- 同騎手 直近 10 走 (馬-騎手ペアの cumulative) ---
    gj = df.groupby([COL_PEDIGREE, COL_JOCKEY])
    df["jockey_n_prev"]    = gj.cumcount()
    df["jockey_wins_prev"] = gj["_is_win"].cumsum().astype("Int64") - df["_is_win"].astype("Int64")
    df["jockey_top3_prev"] = gj["_is_top3"].cumsum().astype("Int64") - df["_is_top3"].astype("Int64")
    df["jockey_win_rate"]  = np.where(df["jockey_n_prev"] > 0,
                                      df["jockey_wins_prev"] / df["jockey_n_prev"], np.nan)
    df["jockey_top3_rate"] = np.where(df["jockey_n_prev"] > 0,
                                      df["jockey_top3_prev"] / df["jockey_n_prev"], np.nan)

    df = df.drop(columns=["_is_win", "_is_top3", "_dist_b", "_course_key",
                          "course_wins_prev", "course_top3_prev",
                          "jockey_wins_prev", "jockey_top3_prev"])

    log("  加えた列: course_n_prev, course_win_rate, course_top3_rate, "
        "jockey_n_prev, jockey_win_rate, jockey_top3_rate")
    return df


# ============================================================
# main
# ============================================================
def main(stages: set[int]):
    log(f"=== build_master_v2 開始 stages={sorted(stages)} ===")
    log(f"読み込み: {MASTER_IN}")
    df = pd.read_csv(MASTER_IN, encoding="utf-8-sig", low_memory=False)
    log(f"  元 master: {len(df):,} 行 × {len(df.columns)} 列")

    stats = {"in_rows": len(df), "in_cols": len(df.columns), "stages": sorted(stages)}

    if 3 in stages:
        df = merge_hosei(df)
    if 4 in stages:
        df = merge_training(df)
    if 5 in stages:
        df = compute_history_features(df)

    stats["out_rows"] = len(df)
    stats["out_cols"] = len(df.columns)
    new_cols = [c for c in df.columns if c not in
                pd.read_csv(MASTER_IN, encoding="utf-8-sig", nrows=0).columns]
    stats["new_cols"] = new_cols
    log(f"\n=== 最終 ===")
    log(f"  out: {len(df):,} 行 × {len(df.columns)} 列")
    log(f"  追加列 ({len(new_cols)}): {new_cols}")

    # 日付順に戻す
    df = df.sort_values([COL_DATE, COL_RID, COL_BAN]).reset_index(drop=True)

    log(f"  保存: {MASTER_OUT}")
    df.to_csv(MASTER_OUT, index=False, encoding="utf-8-sig")

    BUILD_LOG.parent.mkdir(exist_ok=True)
    with open(BUILD_LOG, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
    log(f"  log:  {BUILD_LOG}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    help="3 | 4 | 5 | all  (複数は 3,4,5)")
    args = ap.parse_args()
    if args.stage == "all":
        stages = {3, 4, 5}
    else:
        stages = {int(s) for s in args.stage.split(",")}
    main(stages)
