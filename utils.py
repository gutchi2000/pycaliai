"""
utils.py
PyCaLiAI - 共通ユーティリティ
"""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


def backup_model(path: Path) -> None:
    """既存モデルファイルを上書き前にタイムスタンプ付きでバックアップする。

    例: lgbm_optuna_v1.pkl → lgbm_optuna_v1_20250101_120000.pkl
    ファイルが存在しない場合は何もしない。
    """
    if not path.exists():
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_stem(f"{path.stem}_{ts}")
    shutil.move(str(path), str(backup_path))
    print(f"[backup_model] {path.name} → {backup_path.name}")


def add_meta(df: pd.DataFrame) -> pd.DataFrame:
    """日付・曜日・土日フラグ・同日同会場R数・週末10Rフラグを付与する。"""
    df = df.copy()
    df["date"] = pd.to_datetime(df["日付"].astype(str), format="%Y%m%d")
    df["曜日"]  = df["date"].dt.dayofweek
    df["土日"]  = df["曜日"].isin([5, 6])
    rc = (
        df.groupby(["日付", "場所"])["race_id"]
        .nunique()
        .reset_index()
        .rename(columns={"race_id": "R数"})
    )
    df = df.merge(rc, on=["日付", "場所"], how="left")
    df["週末10R"] = df["土日"] & (df["R数"] >= 10)
    return df


def parse_time_str(series: pd.Series) -> pd.Series:
    """走破タイム文字列を秒数（float）に変換する。

    形式: "M.SS.T"（分.秒.1/10秒）→ float秒
    例: "1.34.5" → 94.5秒
    """
    def _convert(val: str) -> float | None:
        try:
            parts = str(val).strip().split(".")
            if len(parts) == 3:
                return int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 10
            return float(val)
        except (ValueError, TypeError, AttributeError):
            return None
    return series.apply(_convert)
