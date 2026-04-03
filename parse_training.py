"""
parse_training.py
=================
調教データ (坂路H / ウッドチップW) から特徴量を抽出する。

坂路 (H-*.csv): 年月日, 馬名, Time1, Time2, Time3, Time4, Lap4, Lap3, Lap2, Lap1
  - 800m坂路コース, 4ハロン計測
  - Time1=全体タイム, Time4=最後の1F

ウッドチップ (W-*.csv): 年月日, 馬名, 5F, 4F, 3F, Lap3, Lap2, Lap1
  - ウッドチップコース, 3-5F計測
  - 5Fは欠損あり(短い調教), 3Fは常にあり

抽出特徴量:
  h_best_time: 坂路ベストタイム (Time1の最小値)
  h_best_lap1: 坂路ベスト最終1Fラップ
  h_accel: 坂路加速度 (Lap1 - Lap4, 負=末脚強い)
  h_count: 坂路追い切り本数
  w_best_3f: ウッドベスト3Fタイム
  w_best_lap1: ウッドベスト最終1Fラップ
  w_accel: ウッド加速度 (Lap1 - Lap3)
  w_count: ウッド追い切り本数
  total_workouts: 合計追い切り本数
  has_training: 調教データ有無

Usage:
  from parse_training import load_training_features
  feats = load_training_features("data/training", week_date="20260322")
  # Returns DataFrame: horse_name → features
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
import glob


def _load_hill(training_dir: Path, week_date: str | None = None) -> pd.DataFrame:
    """坂路データ (H-*.csv) を読み込む。"""
    pattern = str(training_dir / "H-*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame()

    # week_dateが指定されたら、そのレース日の直前の週ファイルを選ぶ
    if week_date and files:
        target = files[-1]  # デフォルトは最新
        for f in files:
            # ファイル名: H-20260321-20260327.csv → 終了日20260327
            stem = Path(f).stem
            parts = stem.split("-")
            if len(parts) >= 3:
                end_date = parts[2]
                if end_date <= week_date:
                    target = f
        files = [target]

    dfs = []
    for f in files:
        df = pd.read_csv(f, encoding="cp932")
        df.columns = ["date", "horse_name", "time1", "time2", "time3", "time4",
                       "lap4", "lap3", "lap2", "lap1"]
        # 異常値除外 (time1=0 or > 100秒は除外)
        df = df[(df["time1"] > 40) & (df["time1"] < 100)]
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _load_wood(training_dir: Path, week_date: str | None = None) -> pd.DataFrame:
    """ウッドチップデータ (W-*.csv) を読み込む。"""
    pattern = str(training_dir / "W-*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame()

    if week_date and files:
        target = files[-1]
        for f in files:
            stem = Path(f).stem
            parts = stem.split("-")
            if len(parts) >= 3:
                end_date = parts[2]
                if end_date <= week_date:
                    target = f
        files = [target]

    dfs = []
    for f in files:
        df = pd.read_csv(f, encoding="cp932")
        df.columns = ["date", "horse_name", "f5", "f4", "f3", "lap3", "lap2", "lap1"]
        # 3Fが有効なものだけ
        df = df[df["f3"].notna() & (df["f3"] > 20) & (df["f3"] < 60)]
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_training_features(
    training_dir: str | Path,
    week_date: str | None = None,
) -> pd.DataFrame:
    """調教データから馬ごとの特徴量を抽出。

    Args:
        training_dir: data/training/ ディレクトリパス
        week_date: YYYYMMDD形式のレース日 (該当週のファイルを自動選択)

    Returns:
        DataFrame indexed by horse_name with training features
    """
    training_dir = Path(training_dir)
    hill = _load_hill(training_dir, week_date)
    wood = _load_wood(training_dir, week_date)

    features = {}

    # 坂路特徴量
    if len(hill) > 0:
        hill["horse_name"] = hill["horse_name"].str.strip()
        h_grp = hill.groupby("horse_name")
        h_feats = pd.DataFrame({
            "h_best_time": h_grp["time1"].min(),
            "h_best_lap1": h_grp["lap1"].min(),
            "h_count": h_grp["time1"].count(),
        })
        # 加速度: 最新の追い切りのLap1 - Lap4
        h_latest = hill.sort_values("date").groupby("horse_name").last()
        h_feats["h_accel"] = h_latest["lap1"] - h_latest["lap4"]
        features["hill"] = h_feats

    # ウッドチップ特徴量
    if len(wood) > 0:
        wood["horse_name"] = wood["horse_name"].str.strip()
        w_grp = wood.groupby("horse_name")
        w_feats = pd.DataFrame({
            "w_best_3f": w_grp["f3"].min(),
            "w_best_lap1": w_grp["lap1"].min(),
            "w_count": w_grp["f3"].count(),
        })
        w_latest = wood.sort_values("date").groupby("horse_name").last()
        w_feats["w_accel"] = w_latest["lap1"] - w_latest["lap3"]
        features["wood"] = w_feats

    # 結合
    if not features:
        return pd.DataFrame()

    if "hill" in features and "wood" in features:
        result = features["hill"].join(features["wood"], how="outer")
    elif "hill" in features:
        result = features["hill"]
    else:
        result = features["wood"]

    # 欠損を0/NaN埋め
    for col in ["h_count", "w_count"]:
        if col in result.columns:
            result[col] = result[col].fillna(0).astype(int)
        else:
            result[col] = 0

    result["total_workouts"] = result.get("h_count", 0) + result.get("w_count", 0)
    result["has_training"] = 1

    return result


if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    feats = load_training_features(r"E:\PyCaLiAI\data\training", week_date="20260327")
    print(f"Training features: {len(feats)} horses")
    print(f"\nColumns: {list(feats.columns)}")
    print(f"\nStats:")
    print(feats.describe().to_string())

    print(f"\nSample (top 10 by h_best_time):")
    if "h_best_time" in feats.columns:
        sample = feats.dropna(subset=["h_best_time"]).nsmallest(10, "h_best_time")
        for name, row in sample.iterrows():
            print(f"  {name:<15} H={row.get('h_best_time',''):.1f}s "
                  f"Lap1={row.get('h_best_lap1',''):.1f}s "
                  f"Accel={row.get('h_accel',''):+.1f}s "
                  f"N={row.get('h_count',0):.0f}")
