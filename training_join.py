# -*- coding: utf-8 -*-
"""
training_join.py — 坂路 / WC 調教 CSV の読み込みと出走表への JOIN (旧 optuna_lgbm.py 148〜262 行を無改変で移動)
================================================================================
docs/LEGACY_RETIREMENT_PLAN_20261004.md 段階 1 (2026-11-04)。weekly_parse.parse_csv (本番) と旧学習スクリプト
(optuna_lgbm / optuna_catboost) の両方から使う。E:\競馬過去走データ\ は読み取りのみ。
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
HANRO_MASTER  = Path(r"E:\競馬過去走データ\H-20150401-20260313.csv")
WC_MASTER     = Path(r"E:\競馬過去走データ\W-20150401-20260313.csv")
CHUKYO_DIR    = DATA_DIR / "training"  # 週次調教CSV置き場（H-*.csv / W-*.csv）




# =========================================================
# 調教データロード・マージ
# =========================================================
def load_chukyo():
    """坂路・WCマスターCSV + 週次追加ファイル（H-*.csv / W-*.csv）を読み込む。

    毎週、レース週（土〜金）の調教CSVを以下の命名規則で置くだけで自動取り込みされる:
        E:\\PyCaLiAI\\data\\training\\H-YYYYMMDD-YYYYMMDD.csv  （坂路）
        E:\\PyCaLiAI\\data\\training\\W-YYYYMMDD-YYYYMMDD.csv  （WC）
    例: H-20260314-20260320.csv, W-20260314-20260320.csv
    重複行は自動で除去するので、マスターと日付が被っても問題ない。
    """
    chukyo_dir = CHUKYO_DIR
    CHUKYO_DIR.mkdir(parents=True, exist_ok=True)

    def _read_hanro(path: Path):
        return pd.read_csv(path, encoding="cp932",
                           usecols=["年月日", "馬名", "Time1", "Time2", "Time3", "Time4",
                                    "Lap4", "Lap3", "Lap2", "Lap1"])

    def _read_wc(path: Path):
        return pd.read_csv(path, encoding="cp932",
                           usecols=["年月日", "馬名", "5F", "4F", "3F", "Lap3", "Lap2", "Lap1"])

    def _collect(master: Path, glob_pat: str, reader, rename_map: dict, label: str):
        files = []
        if master.exists():
            files.append(master)
        for f in sorted(chukyo_dir.glob(glob_pat)):
            if f != master:
                files.append(f)
        if not files:
            logger.warning(f"{label} CSV未検出: {chukyo_dir}")
            return None
        dfs = []
        for f in files:
            try:
                dfs.append(reader(f))
            except Exception as e:
                logger.warning(f"{label} CSV読み込みスキップ: {f.name} ({e})")
        if not dfs:
            return None
        result = pd.concat(dfs, ignore_index=True).drop_duplicates()
        result = result.rename(columns=rename_map)
        result["_dt"] = pd.to_datetime(result["年月日"].astype(str), format="%Y%m%d")
        result = result.sort_values(["馬名", "_dt"]).reset_index(drop=True)
        weekly = [f.name for f in files if f != master]
        extra = f" + 週次{len(weekly)}件: {weekly}" if weekly else ""
        logger.info(f"{label} CSV読み込み: {len(result):,}行（マスター{extra}）")
        return result

    hanro = _collect(
        HANRO_MASTER, "H-*.csv", _read_hanro,
        {"Time1": "hanro_4f", "Time2": "hanro_3f", "Time3": "hanro_2f", "Time4": "hanro_1f",
         "Lap4": "hanro_lap4", "Lap3": "hanro_lap3", "Lap2": "hanro_lap2", "Lap1": "hanro_lap1"},
        "坂路",
    )
    wc = _collect(
        WC_MASTER, "W-*.csv", _read_wc,
        {"5F": "wc_5f", "4F": "wc_4f", "3F": "wc_3f",
         "Lap3": "wc_lap3", "Lap2": "wc_lap2", "Lap1": "wc_lap1"},
        "WC",
    )
    return hanro, wc


def merge_chukyo(df: pd.DataFrame, hanro, wc) -> pd.DataFrame:
    """グループ別searchsortedで最終追い切り（レース14日前以内）をJOIN。"""
    import numpy as np

    df = df.copy()
    df["_race_dt"] = pd.to_datetime(df["日付"].astype(str), format="%Y%m%d")
    df_idx = df.reset_index(drop=True)

    for trn, feat_cols, out_cols, prefix in [
        (hanro,
         ["hanro_4f", "hanro_3f", "hanro_2f", "hanro_1f",
          "hanro_lap4", "hanro_lap3", "hanro_lap2", "hanro_lap1"],
         ["trn_hanro_4f", "trn_hanro_3f", "trn_hanro_2f", "trn_hanro_1f",
          "trn_hanro_lap4", "trn_hanro_lap3", "trn_hanro_lap2", "trn_hanro_lap1"],
         "hanro"),
        (wc,
         ["wc_5f", "wc_4f", "wc_3f", "wc_lap3", "wc_lap2", "wc_lap1"],
         ["trn_wc_5f", "trn_wc_4f", "trn_wc_3f", "trn_wc_lap3", "trn_wc_lap2", "trn_wc_lap1"],
         "wc"),
    ]:
        days_col = f"trn_{prefix}_days"
        result = {c: np.full(len(df_idx), np.nan) for c in out_cols + [days_col]}

        if trn is None:
            for col in out_cols + [days_col]:
                df[col] = float("nan")
            continue

        trn_sorted = trn.sort_values(["馬名", "_dt"])
        trn_g = trn_sorted.groupby("馬名", sort=True)
        race_g = df_idx.groupby("馬名", sort=True)

        for horse, race_rows in race_g:
            if horse not in trn_g.groups:
                continue
            trn_h = trn_g.get_group(horse)
            trn_dates = trn_h["_dt"].values
            trn_feats = trn_h[feat_cols].values
            race_idxs = race_rows.index.values
            race_dates = race_rows["_race_dt"].values

            positions = np.searchsorted(trn_dates, race_dates, side="right") - 1
            cutoffs   = race_dates - np.timedelta64(14, "D")
            valid     = (positions >= 0) & (trn_dates[np.maximum(positions, 0)] >= cutoffs)

            vp = positions[valid]
            vi = race_idxs[valid]
            for j, col in enumerate(out_cols):
                result[col][vi] = trn_feats[vp, j]
            if valid.any():
                result[days_col][vi] = (
                    race_dates[valid] - trn_dates[vp]
                ).astype("timedelta64[D]").astype(float)

        for col in out_cols + [days_col]:
            df[col] = result[col]

    df = df.drop(columns=["_race_dt"])
    cov_h = df["trn_hanro_4f"].notna().mean() * 100
    cov_w = df["trn_wc_3f"].notna().mean() * 100
    logger.info(f"調教JOIN完了: 坂路カバレッジ={cov_h:.1f}%  WC={cov_w:.1f}%")
    return df
