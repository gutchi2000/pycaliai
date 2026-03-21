"""
predict_weekly.py
週末出走表CSV → モデル予想結果CSV出力

Usage:
    python predict_weekly.py --csv data/weekly/20260301.csv
    python predict_weekly.py --csv data/weekly/20260301.csv --budget 10000
    python predict_weekly.py --csv data/weekly/20260301.csv --out reports/pred_20260301.csv

Output columns:
    日付, 場所, R, クラス, 距離, レースID, 馬番, 馬名, 騎手,
    スコア, 印, 複勝_購入額, 馬連_買い目, 馬連_購入額,
    三連複_買い目, 三連複_購入額, 三連単_買い目, 三連単_購入額,
    戦略対象
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR      = Path(__file__).parent
MODEL_DIR     = BASE_DIR / "models"
STRATEGY_JSON = BASE_DIR / "data" / "strategy_weights.json"
LGBM_PATH      = MODEL_DIR / "lgbm_optuna_v1.pkl"
CAT_PATH       = MODEL_DIR / "catboost_optuna_v1.pkl"
RANK_PATH      = MODEL_DIR / "catboost_rank_v1.pkl"
CAL_PATH       = MODEL_DIR / "ensemble_calibrator_v1.pkl"
TORCH_PATH     = MODEL_DIR / "transformer_pl_v2.pkl"
META_PATH      = MODEL_DIR / "stacking_meta_v1.pkl"
STACK_CAL_PATH = MODEL_DIR / "stacking_calibrator_v1.pkl"

# モデルキャッシュ（プロセス内で1回だけロード）
_model_cache: dict = {}

def _get_cached(path: Path, key: str):
    if key not in _model_cache and path.exists():
        _model_cache[key] = joblib.load(path)
    return _model_cache.get(key)

MIN_UNIT        = 100
EXCLUDE_PLACES  = {"東京", "小倉"}
EXCLUDE_CLASSES = {"新馬", "障害"}

# =========================================================
# 列定義
# =========================================================
RACE_COLS = [
    "レースID(新)","日付S","曜日","場所","開催","R","レース名","クラス名",
    "芝・ダート","距離","コース区分","コーナー回数","馬場状態(暫定)","天候(暫定)",
    "フルゲート頭数","発走時刻","性別限定","重量種別","年齢限定",
]
HORSE_COLS_33 = [
    "枠番","B","馬番","馬名S","性別","年齢","人気_今走","単勝","ZI印","ZI","ZI順",
    "斤量","減M","替","騎手","所属","調教師","父","母父","父タイプ","母父タイプ",
    "前走月","前走日","前走場所","前走TD","前走距離","前走馬場状態","前走着順",
    "前走人気","前走レース名","前走上り3F","前走決手","前走間隔",
]
HORSE_COLS_46 = [
    "枠番","B","馬番","馬名S","性別","年齢","人気_今走","単勝","ZI印","ZI","ZI順",
    "斤量","減M","替","騎手","所属","調教師","父","母父","父タイプ","母父タイプ",
    "前走月","前走日","前走開催","前走間隔","前走レース名","前走TD","前走距離","前走馬場状態",
    "前走B","前走騎手","前走斤量","前走減","前走人気","前走単勝オッズ","前走着順","前走着差",
    "マイニング順位","前走通過1","前走通過2","前走通過3","前走通過4","前走Ave3F",
    "前走上り3F","前走上り3F順位","前走1_2着馬",
]
HORSE_COLS_48 = [
    "枠番","B","馬番","馬名S","性別","年齢","人気_今走","単勝","ZI印","ZI","ZI順",
    "斤量","減M","替","騎手","所属","調教師","父","母父","父タイプ","母父タイプ",
    "前走月","前走日","前走開催","前走間隔","前走レース名","前走TD","前走距離","前走馬場状態",
    "前走B","前走騎手","前走斤量","前走減","前走人気","前走単勝オッズ","前走着順","前走着差",
    "マイニング順位","前走通過1","前走通過2","前走通過3","前走通過4","前走Ave3F",
    "前走上り3F","前走上り3F順位","前走1_2着馬",
    "騎手コード","調教師コード",   # 48列形式: 末尾2列にコード追加
]
HORSE_COLS_49 = [
    "枠番","B","馬番","馬名S","性別","年齢","馬体重","馬体重増減_raw","馬体重増減",
    "人気_今走","単勝","ZI印","ZI","ZI順","斤量","減M","替","騎手","所属","調教師",
    "父","母父","父タイプ","母父タイプ",
    "前走月","前走日","前走開催","前走間隔","前走レース名","前走TD","前走距離","前走馬場状態",
    "前走B","前走騎手","前走斤量","前走減","前走人気","前走単勝オッズ","前走着順","前走着差",
    "マイニング順位","前走通過1","前走通過2","前走通過3","前走通過4","前走Ave3F",
    "前走上り3F","前走上り3F順位","前走1_2着馬",
]
HORSE_COLS_99 = [
    "枠番","B","馬番","馬名S","性別","年齢","馬体重","馬体重増減_raw","馬体重増減",
    "人気_今走","単勝","ZI印","ZI","ZI順","斤量","減M","替","騎手","所属","調教師",
    "父","母父","父タイプ","母父タイプ",
    # 1走前
    "前走月","前走日","前走開催","前走間隔","前走レース名","前走TD","前走距離","前走馬場状態",
    "前走B","前走騎手","前走斤量","前走減","前走人気","前走単勝オッズ","前走着順","前走着差",
    "マイニング順位","前走通過1","前走通過2","前走通過3","前走通過4","前走Ave3F",
    "前走上り3F","前走上り3F順位","前走1_2着馬",
    # 2走前
    "二走前月","二走前日","二走前開催","二走前間隔","二走前レース名","二走前TD","二走前距離","二走前馬場状態",
    "二走前B","二走前騎手","二走前斤量","二走前減","二走前人気","二走前単勝オッズ","二走前着順","二走前着差",
    "二走前マイニング順位","二走前通過1","二走前通過2","二走前通過3","二走前通過4","二走前Ave3F",
    "二走前上り3F","二走前上り3F順位","二走前1_2着馬",
    # 3走前
    "三走前月","三走前日","三走前開催","三走前間隔","三走前レース名","三走前TD","三走前距離","三走前馬場状態",
    "三走前B","三走前騎手","三走前斤量","三走前減","三走前人気","三走前単勝オッズ","三走前着順","三走前着差",
    "三走前マイニング順位","三走前通過1","三走前通過2","三走前通過3","三走前通過4","三走前Ave3F",
    "三走前上り3F","三走前上り3F順位","三走前1_2着馬",
]
COLUMN_MAP = {
    "馬名S":         "馬名",
    "芝・ダート":     "芝・ダ",
    "馬場状態(暫定)": "馬場状態",
    "天候(暫定)":    "天気",
    "人気_今走":     "人気",
    "ZI順":         "ZI順位",
    "父":            "種牡馬",
    "母父":          "母父馬",
    "父タイプ":       "父タイプ名",
    "母父タイプ":     "母父タイプ名",
    "前走着順":       "前走確定着順",
    "前走上り3F":     "前走上り3F",
    "前走TD":        "前芝・ダ",
    "前走間隔":       "間隔",
    "前走着差":       "前走着差タイム",
    "前走斤量":       "前走斤量",
    # master CSV の列名に合わせる（モデルが参照する名前）
    "前走Ave3F":     "前走Ave-3F",
    "前走上り3F順位":  "前走上り3F順",
    "前走通過1":      "前1角",
    "前走通過2":      "前2角",
    "前走通過3":      "前3角",
    "前走通過4":      "前4角",
    "マイニング順位":  "マイニング順位",
    "前走単勝オッズ":  "前走単勝オッズ",
    # 前走距離 → 前距離（モデルが使う列名）
    "前走距離":       "前距離",
}
CLASS_NORMALIZE = {
    "新馬":"新馬","未勝利":"未勝利","1勝":"1勝","500万":"1勝",
    "2勝":"2勝","1000万":"2勝","3勝":"3勝","1600万":"3勝",
    "OP(L)":"OP(L)","Ｇ１":"Ｇ１","Ｇ２":"Ｇ２","Ｇ３":"Ｇ３",
}


# =========================================================
# 着度数CSV パース
# =========================================================
TYAKU_DIR  = BASE_DIR / "data" / "tyaku"
HOSSEI_DIR = BASE_DIR / "data" / "hosei"

TYAKU_HORSE_COLS = [
    "枠番","B","馬番","印","M2","M3","M4","馬名S","C","性別","年齢","替","騎手","斤量","減M","単勝",
    "馬体重","増減",
    "中央平地全:1着","中央平地全:2着","中央平地全:3着","中央平地全:外","中央平地全:連対率",
    "同騎手:1着","同騎手:2着","同騎手:3着","同騎手:外","同騎手:連対率",
    "全ダート:1着","全ダート:2着","全ダート:3着","全ダート:外","全ダート:連対率",
    "同距離:1着","同距離:2着","同距離:3着","同距離:外","同距離:連対率",
    "同場所:1着","同場所:2着","同場所:3着","同場所:外","同場所:連対率",
    "同コース:1着","同コース:2着","同コース:3着","同コース:外","同コース:連対率",
    "同クラス:1着","同クラス:2着","同クラス:3着","同クラス:外","同クラス:連対率",
    "穴傾向","BESTタイム",
]


def _load_tyaku(date_str: str) -> pd.DataFrame | None:
    """data/tyaku/YYYYMMDD.csv を読み込み、馬番→着度数の対応表を返す。
    ファイルが存在しない場合は None を返す。"""
    path = TYAKU_DIR / f"{date_str}.csv"
    if not path.exists():
        return None

    for enc in ["cp932", "shift_jis", "utf-8"]:
        try:
            text = path.read_bytes().decode(enc)
            break
        except Exception:
            continue
    else:
        return None

    rows: list[dict] = []
    current_race_id: str | None = None

    for line in text.splitlines():
        cols = line.split(",")
        if len(cols) == 19 and cols[0] not in ("レースID(新)", ""):
            current_race_id = cols[0].strip()[:16]  # 16桁に切る
        elif len(cols) == 55 and cols[0] not in ("枠番", "") and current_race_id:
            row = dict(zip(TYAKU_HORSE_COLS, cols))
            row["レースID(新/馬番無)"] = current_race_id
            rows.append(row)

    if not rows:
        return None

    df = pd.DataFrame(rows)
    # 数値変換
    for col in ["馬番","馬体重","増減",
                "中央平地全:1着","中央平地全:2着","中央平地全:3着","中央平地全:外",
                "同コース:1着","同コース:2着","同コース:3着","同コース:外",
                "同クラス:1着","同クラス:2着","同クラス:3着","同クラス:外"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 馬ごとの複勝率を計算（ベイズ平滑化: 少走経験馬でも 0 にならないよう事前分布で補正）
    # prior = 訓練valid中央値 0.286, 仮想サンプル数 = 5
    # smoothed = (着内回数 + 1.43) / (総走数 + 5.0)
    # → 0走: 1.43/5.00 = 0.286, 1走0着: 1.43/6.00 = 0.238, 10走3着内: 4.43/15.0 = 0.295
    _PRIOR_ALPHA = 1.43   # 0.286 × 5
    _PRIOR_BETA  = 5.0
    for prefix, out_col in [("中央平地全", "horse_fuku_career"),
                             ("同コース",   "horse_fuku_course"),
                             ("同クラス",   "horse_fuku_class")]:
        w = df[f"{prefix}:1着"].fillna(0) + df[f"{prefix}:2着"].fillna(0) + df[f"{prefix}:3着"].fillna(0)
        total = w + df[f"{prefix}:外"].fillna(0)
        df[out_col] = (w + _PRIOR_ALPHA) / (total + _PRIOR_BETA)

    # 馬体重の増減を数値化（"- 8" → -8, "+ 2" → 2）
    if "増減" in df.columns:
        df["増減"] = df["増減"].astype(str).str.replace(" ", "").str.replace("－","-").str.replace("＋","+")
        df["増減"] = pd.to_numeric(df["増減"], errors="coerce")

    keep = ["レースID(新/馬番無)","馬番","馬体重","増減",
            "horse_fuku_career","horse_fuku_course","horse_fuku_class"]
    return df[[c for c in keep if c in df.columns]]


def _load_hosei(date_str: str) -> pd.DataFrame | None:
    """data/hosei/H_*.csv を全て結合してレースID×馬番→補正タイムの対応表を返す。
    毎週 H_YYYYMMDD-YYYYMMDD.csv 形式でフォルダに追加するだけで自動取り込みされる。
    """
    hosei_files = sorted(HOSSEI_DIR.glob("H_*.csv"))
    if not hosei_files:
        return None
    dfs = []
    for path in hosei_files:
        for enc in ["cp932", "utf-8-sig", "utf-8"]:
            try:
                tmp = pd.read_csv(path, encoding=enc,
                                  usecols=["レースID(新)", "前走補9", "前走補正"],
                                  dtype={"レースID(新)": str})
                dfs.append(tmp)
                break
            except Exception:
                continue
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True).drop_duplicates()
    df["レースID(新)"] = df["レースID(新)"].astype(str)
    for col in ["前走補9", "前走補正"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["レースID(新)", "前走補9", "前走補正"]]


# =========================================================
# CSV パース
# =========================================================
def parse_csv(path: Path) -> pd.DataFrame:
    """ターゲット形式CSV（33列 or 46列）をDataFrameに変換する。"""
    for enc in ["cp932", "shift_jis", "utf-8"]:
        try:
            text = path.read_bytes().decode(enc)
            break
        except Exception:
            continue

    races: list[dict] = []
    current_race: dict | None = None

    for line in text.splitlines():
        cols = line.split(",")
        if cols[0] in ("レースID(新)", "枠番", "番", ""):
            continue
        if len(cols) == 19:
            current_race = dict(zip(RACE_COLS, cols))
        elif len(cols) == 33 and current_race:
            horse = dict(zip(HORSE_COLS_33, cols))
            horse.update(current_race)
            races.append(horse)
        elif len(cols) == 46 and current_race:
            horse = dict(zip(HORSE_COLS_46, cols))
            horse.update(current_race)
            races.append(horse)
        elif len(cols) == 48 and current_race:
            horse = dict(zip(HORSE_COLS_48, cols))
            horse.update(current_race)
            races.append(horse)
        elif len(cols) == 49 and current_race:
            horse = dict(zip(HORSE_COLS_49, cols))
            horse.update(current_race)
            races.append(horse)
        elif len(cols) == 99 and current_race:
            horse = dict(zip(HORSE_COLS_99, cols))
            horse.update(current_race)
            races.append(horse)

    df = pd.DataFrame(races).rename(columns=COLUMN_MAP)
    df["レースID(新/馬番無)"] = df["レースID(新)"].astype(str).str[:16]

    for col in ["枠番","馬番","斤量","ZI","ZI順位","距離","人気","単勝",
                "前走確定着順","前走上り3F","前走距離","間隔","前走人気",
                "前走着差タイム","前走斤量","前走Ave-3F","前走上り3F順",
                "マイニング順位","前走単勝オッズ",
                "前1角","前2角","前3角","前4角",
                "フルゲート頭数","年齢","出走頭数","コーナー回数",
                # COLUMN_MAP で rename された後の列名
                "前距離","前走馬体重","前走馬体重増減"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 出走頭数：週次CSVは直接持たないため、レースIDごとの馬数から算出
    if "出走頭数" not in df.columns:
        df["出走頭数"] = df.groupby("レースID(新/馬番無)")["馬番"].transform("count")
    df["出走頭数"] = pd.to_numeric(df["出走頭数"], errors="coerce").fillna(
        pd.to_numeric(df.get("フルゲート頭数"), errors="coerce")
    )

    # 脚質特徴量：前走コーナー通過順位から計算（モデルと同じ定義）
    if "前1角" in df.columns and "前4角" in df.columns:
        n = df["出走頭数"].clip(lower=2)
        df["prev_pos_rel"]   = (df["前1角"] - 1) / (n - 1)
        df["closing_power"]  = (df["前1角"] - df["前4角"]) / (n - 1)

    # 騎手・調教師ローリング成績（週次CSVに騎手コードなし → 訓練データ中央値で補完）
    # 訓練 valid 中央値: jockey_fuku30=0.200, jockey_fuku90=0.200,
    #                    trainer_fuku30=0.200, trainer_fuku90=0.211
    _ROLLING_TRAIN_MEDIANS = {
        "jockey_fuku30": 0.200,
        "jockey_fuku90": 0.200,
        "trainer_fuku30": 0.200,
        "trainer_fuku90": 0.211,
    }
    for fname, code_col, stat_cols in [
        ("jockey_stats.csv",  "騎手コード",  ["jockey_fuku30", "jockey_fuku90"]),
        ("trainer_stats.csv", "調教師コード", ["trainer_fuku30", "trainer_fuku90"]),
    ]:
        stats_path = BASE_DIR / "data" / fname
        if stats_path.exists():
            stats = pd.read_csv(stats_path, encoding="utf-8-sig")
            if code_col in df.columns:
                # "01122" 形式（先頭ゼロ付き文字列）→ int変換（1122）してマージ
                df[code_col] = pd.to_numeric(df[code_col], errors="coerce")
                df = df.merge(stats[[code_col] + stat_cols], on=code_col, how="left")
                for col in stat_cols:
                    if col in df.columns:
                        df[col] = df[col].fillna(_ROLLING_TRAIN_MEDIANS.get(col, 0.200))
            else:
                for col in stat_cols:
                    df[col] = _ROLLING_TRAIN_MEDIANS.get(col, 0.200)
        else:
            for col in stat_cols:
                df[col] = _ROLLING_TRAIN_MEDIANS.get(col, 0.200)

    # ── 着度数CSV（data/tyaku/YYYYMMDD.csv）があればマージ ──
    date_str = path.stem   # ファイル名の日付部分（例: "20260308"）
    tyaku_df = _load_tyaku(date_str)
    if tyaku_df is not None:
        df = df.merge(tyaku_df, on=["レースID(新/馬番無)", "馬番"], how="left",
                      suffixes=("", "_tyaku"))
        # 馬体重: tyaku が優先（週次CSVに馬体重がある場合は上書き）
        if "馬体重_tyaku" in df.columns:
            df["馬体重"] = df["馬体重_tyaku"].combine_first(
                pd.to_numeric(df.get("馬体重"), errors="coerce"))
            df.drop(columns=["馬体重_tyaku"], inplace=True)
        if "増減_tyaku" in df.columns:
            df["馬体重増減"] = df["増減_tyaku"]
            df.drop(columns=["増減_tyaku"], inplace=True)
        # horse_fuku: tyaku CSV 由来 (horse_fuku_career_tyaku) を優先、
        #             なければ週次CSV由来 (horse_fuku_career) を使用。
        #             app.py も tyaku CSV を優先するので両者の予測を一致させる。
        if "horse_fuku_career_tyaku" in df.columns:
            fuku_src = df["horse_fuku_career_tyaku"].combine_first(df.get("horse_fuku_career", pd.Series(dtype=float)))
            df.drop(columns=["horse_fuku_career_tyaku"], inplace=True, errors="ignore")
        elif "horse_fuku_career" in df.columns:
            fuku_src = df["horse_fuku_career"]
        else:
            fuku_src = None
        if fuku_src is not None:
            df["horse_fuku10"] = fuku_src.fillna(0.286)
            df["horse_fuku30"] = fuku_src.fillna(0.312)
            logger.info(
                f"着度数CSV読み込み済: horse_fuku mean={df['horse_fuku10'].mean():.3f}, "
                f"std={df['horse_fuku10'].std():.3f}"
            )
        else:
            df["horse_fuku10"] = 0.286
            df["horse_fuku30"] = 0.312
    else:
        # 着度数CSVなし → 訓練データ中央値で補完
        df["horse_fuku10"] = 0.286
        df["horse_fuku30"] = 0.312

    # 週次CSVに含まれないペース指数・体重比を訓練データ中央値で補完
    # （0 のままだと分布外でモデルが誤った方向に流れる）
    _PACE_MEDIANS = {
        "前PCI": 49.0,
        "前走PCI3": 50.2,
        "前走RPCI": 48.5,
        "前走平均1Fタイム": 12.26,
    }
    for col, med in _PACE_MEDIANS.items():
        if col not in df.columns:
            df[col] = med

    # 斤量体重比 = 斤量 / 馬体重（馬体重 0 or 欠損のときは訓練中央値で補完）
    if "斤量体重比" not in df.columns:
        wt = pd.to_numeric(df["馬体重"], errors="coerce").replace(0, np.nan) if "馬体重" in df.columns else np.nan
        jk = pd.to_numeric(df["斤量"], errors="coerce") if "斤量" in df.columns else np.nan
        if isinstance(wt, pd.Series) and isinstance(jk, pd.Series):
            df["斤量体重比"] = (jk / wt).fillna(11.8)
        else:
            df["斤量体重比"] = 11.8

    # 訓練データに含まれるが週次CSVにない特徴量を中央値で補完
    # （0 のままだと out-of-distribution でモデルが誤判断する）
    _MISSING_FEATURE_MEDIANS = {
        "馬齢斤量差":            -1,    # 訓練 valid 中央値
        "トラックコード(JV)":      23,   # 訓練 valid 中央値（芝・内=23）
        "前走トラックコード(JV)":  23,   # 訓練 valid 中央値
        "前走競走種別":           13,   # 訓練 valid 中央値
        "前走出走頭数":           15,   # 訓練 valid 中央値
        "前走馬体重":            472,   # 訓練 valid 中央値
        "前走馬体重増減":           0,   # 訓練 valid 中央値（増減ゼロが最多）
        "騎手年齢":              30,   # 訓練 valid 中央値
        "調教師年齢":             53,   # 訓練 valid 中央値
        "休み明け～戦目":           2,   # 訓練 valid 中央値
    }
    for col, med in _MISSING_FEATURE_MEDIANS.items():
        if col not in df.columns:
            df[col] = med

    for col in ["馬体重","馬体重増減","前走斤量","生産者"]:
        if col not in df.columns:
            df[col] = 0
    for col in ["前走走破タイム","前走着差タイム"]:
        if col not in df.columns:
            df[col] = float("nan")  # LGBMのNaN処理に任せる（0だと分布が大きく外れる）

    # ── 補正タイムCSV（data/hosei/H_*.csv）があればマージ ──
    hosei_df = _load_hosei(date_str)
    if hosei_df is not None:
        # hosei は18桁ID（レース16桁+馬番2桁）なので一致させてマージ
        df["_hosei_key"] = (
            df["レースID(新/馬番無)"].astype(str)
            + df["馬番"].astype(int).astype(str).str.zfill(2)
        )
        hosei_df = hosei_df.rename(columns={"レースID(新)": "_hosei_key"})
        df = df.merge(hosei_df, on="_hosei_key", how="left")
        df = df.drop(columns=["_hosei_key"])
        logger.info(
            f"hosei CSV読み込み済: 前走補9 mean={df['前走補9'].mean():.1f}, "
            f"カバレッジ={df['前走補9'].notna().mean()*100:.1f}%"
        )
    else:
        df["前走補9"]  = float("nan")
        df["前走補正"] = float("nan")

    # ── 調教データ（坂路・WCマスターCSVからJOIN）──
    try:
        from optuna_lgbm import load_chukyo, merge_chukyo
        hanro, wc = load_chukyo()
        df = merge_chukyo(df, hanro, wc)
    except Exception as e:
        logger.warning(f"調教JOIN失敗（スキップ）: {e}")
        for col in ["trn_hanro_4f","trn_hanro_3f","trn_hanro_2f","trn_hanro_1f",
                    "trn_hanro_lap1","trn_hanro_lap2","trn_hanro_lap3","trn_hanro_lap4",
                    "trn_hanro_days",
                    "trn_wc_5f","trn_wc_4f","trn_wc_3f",
                    "trn_wc_lap1","trn_wc_lap2","trn_wc_lap3",
                    "trn_wc_days"]:
            df[col] = float("nan")

    df = df[~df["距離"].astype(str).str.contains("障", na=False)].copy()

    if "前走単勝オッズ" not in df.columns:
        df["前走単勝オッズ"] = float("nan")

    logger.info(f"パース完了（障害除外済）: {len(df)}頭 / {df['レースID(新/馬番無)'].nunique()}レース")
    return df


# =========================================================
# 予測
# =========================================================
from utils import parse_time_str


def predict_lgbm(df: pd.DataFrame, obj: dict) -> np.ndarray:
    model, encoders, feature_cols = obj["model"], obj["encoders"], obj["feature_cols"]
    df = df.copy()
    for col in ["前走走破タイム","前走着差タイム"]:
        if col in df.columns:
            df[col] = parse_time_str(df[col])
    for col, le in encoders.items():
        if col not in df.columns:
            df[col] = 0
            continue
        df[col] = df[col].astype(str).fillna("__NaN__")
        known = set(le.classes_)
        df[col] = df[col].apply(lambda x: x if x in known else "__NaN__")
        if "__NaN__" not in le.classes_:
            le.classes_ = np.append(le.classes_, "__NaN__")
        df[col] = le.transform(df[col])
    # rolling stats 系は 0 ではなく NaN（モデルの「データなし」分岐を利用）
    _ROLLING_COLS = {"jockey_fuku30","jockey_fuku90","trainer_fuku30","trainer_fuku90",
                     "horse_fuku10","horse_fuku30","前走補9","前走補正",
                     "trn_hanro_4f","trn_hanro_3f","trn_hanro_2f","trn_hanro_1f",
                     "trn_hanro_lap1","trn_hanro_lap2","trn_hanro_lap3","trn_hanro_lap4",
                     "trn_hanro_days",
                     "trn_wc_5f","trn_wc_4f","trn_wc_3f",
                     "trn_wc_lap1","trn_wc_lap2","trn_wc_lap3",
                     "trn_wc_days",
                     "前走単勝オッズ"}
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan if col in _ROLLING_COLS else 0
    # lgb.Booster と LGBMClassifier 両対応
    import lightgbm as lgb
    if isinstance(model, lgb.Booster):
        return model.predict(df[feature_cols])
    return model.predict_proba(df[feature_cols])[:, 1]


def predict_catboost(df: pd.DataFrame, obj: dict) -> np.ndarray:
    from catboost import Pool
    model, feature_cols = obj["model"], obj["feature_cols"]
    cat_list = [
        "種牡馬","父タイプ名","母父馬","母父タイプ名","毛色",
        "馬主(最新/仮想)","生産者","芝・ダ","コース区分","芝(内・外)",
        "馬場状態","天気","クラス名","場所","性別","斤量",
        "ブリンカー","重量種別","年齢限定","限定","性別限定","指定条件",
        "前走場所","前芝・ダ","前走馬場状態","前走斤量","前好走",
    ]
    df = df.copy()
    for col in ["前走走破タイム","前走着差タイム"]:
        if col in df.columns:
            df[col] = parse_time_str(df[col])
    for col in cat_list:
        df[col] = df[col].fillna("__NaN__").astype(str) if col in df.columns else "__NaN__"
    _ROLLING_COLS = {"jockey_fuku30","jockey_fuku90","trainer_fuku30","trainer_fuku90",
                     "horse_fuku10","horse_fuku30","前走補9","前走補正",
                     "trn_hanro_4f","trn_hanro_3f","trn_hanro_2f","trn_hanro_1f",
                     "trn_hanro_lap1","trn_hanro_lap2","trn_hanro_lap3","trn_hanro_lap4",
                     "trn_hanro_days",
                     "trn_wc_5f","trn_wc_4f","trn_wc_3f",
                     "trn_wc_lap1","trn_wc_lap2","trn_wc_lap3",
                     "trn_wc_days",
                     "前走単勝オッズ"}
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan if col in _ROLLING_COLS else 0.0
    cat_idx = [i for i, c in enumerate(feature_cols) if c in cat_list]
    pool = Pool(df[feature_cols], cat_features=cat_idx)
    return model.predict_proba(pool)[:, 1]


def predict_catboost_rank(df: pd.DataFrame, obj: dict) -> np.ndarray:
    """YetiRankモデルで予測。スコアをレース内min-max正規化して[0,1]の擬似確率に変換。"""
    from catboost import Pool
    model, feature_cols = obj["model"], obj["feature_cols"]
    cat_list = [
        "種牡馬","父タイプ名","母父馬","母父タイプ名","毛色",
        "馬主(最新/仮想)","生産者","芝・ダ","コース区分","芝(内・外)",
        "馬場状態","天気","クラス名","場所","性別","斤量",
        "ブリンカー","重量種別","年齢限定","限定","性別限定","指定条件",
        "前走場所","前芝・ダ","前走馬場状態","前走斤量","前好走",
    ]
    df = df.copy()
    for col in ["前走走破タイム","前走着差タイム"]:
        if col in df.columns:
            df[col] = parse_time_str(df[col])
    for col in cat_list:
        df[col] = df[col].fillna("__NaN__").astype(str) if col in df.columns else "__NaN__"
    _ROLLING_COLS = {"jockey_fuku30","jockey_fuku90","trainer_fuku30","trainer_fuku90",
                     "horse_fuku10","horse_fuku30","前走補9","前走補正",
                     "trn_hanro_4f","trn_hanro_3f","trn_hanro_2f","trn_hanro_1f",
                     "trn_hanro_lap1","trn_hanro_lap2","trn_hanro_lap3","trn_hanro_lap4",
                     "trn_hanro_days",
                     "trn_wc_5f","trn_wc_4f","trn_wc_3f",
                     "trn_wc_lap1","trn_wc_lap2","trn_wc_lap3",
                     "trn_wc_days",
                     "前走単勝オッズ"}
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan if col in _ROLLING_COLS else 0.0
    cat_idx = [i for i, c in enumerate(feature_cols) if c in cat_list]
    pool = Pool(df[feature_cols], cat_features=cat_idx)
    scores = model.predict(pool)  # 生のランキングスコア

    # レース内min-max正規化で[0,1]に変換（レース内相対評価を保持）
    result = np.full(len(df), 0.5)
    df_reset = df.reset_index(drop=True)
    for race_id, group in df_reset.groupby("レースID(新/馬番無)"):
        idx = group.index.tolist()
        s = scores[idx]
        s_min, s_max = s.min(), s.max()
        if s_max > s_min:
            result[idx] = (s - s_min) / (s_max - s_min)
    return result


_META_EXTRA = ["芝・ダ", "距離", "クラス名", "場所", "馬場状態", "出走頭数", "枠番", "馬番"]


def predict_transformer_local(df: pd.DataFrame) -> np.ndarray:
    """Transformer PL予測。レース内スコアをmin-max正規化して[0,1]に変換。モデル未存在 or 失敗時はゼロ配列を返す。"""
    torch_obj = _get_cached(TORCH_PATH, "torch")
    if torch_obj is None:
        return np.zeros(len(df))
    try:
        import torch
        from train_transformer import RaceTransformer
        from optuna_transformer_pl import preprocess as pl_preprocess, RaceDatasetPL, MAX_HORSES as PL_MAX_HORSES
        from torch.utils.data import DataLoader

        DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_config = torch_obj["model_config"]
        encoders     = torch_obj["encoders"]
        num_stats    = torch_obj["num_stats"]
        num_cols     = torch_obj["num_cols"]
        cat_cols     = torch_obj["cat_cols"]

        df_copy = df.copy()
        if "fukusho_flag" not in df_copy.columns:
            df_copy["fukusho_flag"] = 0
        if "着順" not in df_copy.columns:
            df_copy["着順"] = 0
        df2, _, _ = pl_preprocess(df_copy, encoders=encoders, fit=False, num_stats=num_stats)

        # モデルをキャッシュ（起動後1回だけビルド）
        if "torch_model" not in _model_cache:
            m = RaceTransformer(
                cat_vocab_sizes=model_config["cat_vocab_sizes"],
                cat_cols=model_config["cat_cols"],
                n_num=model_config["n_num"],
                d_model=model_config.get("d_model", 64),
                n_heads=model_config.get("n_heads", 2),
                n_layers=model_config.get("n_layers", 4),
                d_ff=model_config.get("d_ff", 256),
                dropout=model_config.get("dropout", 0.1),
            ).to(DEVICE)
            m.load_state_dict(torch_obj["model_state"])
            m.eval()
            _model_cache["torch_model"] = (m, DEVICE)
        model, DEVICE = _model_cache["torch_model"]

        ds     = RaceDatasetPL(df2, cat_cols, num_cols, model_config["cat_vocab_sizes"])
        loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)

        all_scores: list[float] = []
        with torch.no_grad():
            for batch in loader:
                out   = model(batch["cat"].to(DEVICE), batch["num"].to(DEVICE), batch["mask"].to(DEVICE))
                valid = ~batch["mask"]
                all_scores.extend(out.cpu()[valid].numpy().tolist())

        # race_id 順に並べ替えて元 index に戻す
        result  = np.zeros(len(df))
        df_sort = df.sort_values("レースID(新/馬番無)").reset_index(drop=True)
        idx = 0
        for _, group in df_sort.groupby("レースID(新/馬番無)", sort=True):
            valid_n = min(len(group), PL_MAX_HORSES)
            for orig_idx in list(group.index)[:valid_n]:
                if idx < len(all_scores):
                    result[orig_idx] = all_scores[idx]
                    idx += 1

        # レース内 min-max 正規化で [0, 1] に変換
        df_reset = df.reset_index(drop=True)
        for _, group in df_reset.groupby("レースID(新/馬番無)"):
            idxs = group.index.tolist()
            s = result[idxs]
            s_min, s_max = s.min(), s.max()
            if s_max > s_min:
                result[idxs] = (s - s_min) / (s_max - s_min)
            else:
                result[idxs] = 0.5
        return result
    except Exception as e:
        logger.warning(f"Transformer PL予測失敗（0で埋め）: {e}")
        return np.zeros(len(df))


def predict_stacking(df: pd.DataFrame, lgbm_obj: dict, cat_obj: dict) -> np.ndarray | None:
    """スタッキングモデルで予測（4モデル対応）。未存在 or 失敗時は None を返す。"""
    if not META_PATH.exists():
        return None
    try:
        from stacking import build_meta_features

        p_lgbm  = predict_lgbm(df, lgbm_obj)
        p_cat   = predict_catboost(df, cat_obj)
        rank_obj = _get_cached(RANK_PATH, "rank")
        p_rank  = predict_catboost_rank(df, rank_obj) \
                  if rank_obj is not None else np.full(len(df), 0.5)
        p_trans = predict_transformer_local(df)

        meta_obj      = _get_cached(META_PATH, "meta")
        meta_model    = meta_obj["meta_model"]
        meta_encoders = meta_obj["meta_encoders"]
        meta_cols     = meta_obj["meta_cols"]

        df_tmp = df.copy()
        if "出走頭数" not in df_tmp.columns or df_tmp["出走頭数"].isna().all():
            df_tmp["出走頭数"] = df_tmp.groupby("レースID(新/馬番無)")["馬番"].transform("count")

        meta_df, _ = build_meta_features(
            df_tmp, p_lgbm, p_cat, p_rank, p_trans,
            meta_encoders=meta_encoders, fit=False,
            race_col="レースID(新/馬番無)",
        )

        return meta_model.predict_proba(meta_df[meta_cols])[:, 1]
    except Exception as e:
        logger.warning(f"スタッキング予測失敗（フォールバック）: {e}")
        return None


def ensemble_predict(df: pd.DataFrame, lgbm_obj: dict, cat_obj: dict) -> np.ndarray:
    # スタッキング優先（キャリブレーター妥当性チェック付き）
    stacking = predict_stacking(df, lgbm_obj, cat_obj)
    if stacking is not None:
        cal_obj = _get_cached(STACK_CAL_PATH, "stack_cal")
        if cal_obj is not None:
            calibrated = cal_obj["calibrator"].transform(stacking)
            # キャリブレーター出力の妥当性チェック：平均 5% 以上かつ最大 50% 未満
            if calibrated.mean() >= 0.05 and calibrated.max() < 0.50:
                return calibrated
            logger.warning(
                f"スタッキングキャリブレーター出力が異常（mean={calibrated.mean():.3f}, "
                f"max={calibrated.max():.3f}）→ エンサンブルにフォールバック"
            )
    # YetiRank + Transformer PL が存在すれば4モデル加重平均
    # キャリブレーターに最適重みが保存されていればそちらを使用（なければデフォルト値）
    rank_obj  = _get_cached(RANK_PATH, "rank")
    torch_obj = _get_cached(TORCH_PATH, "torch")
    cal_obj_pre = _get_cached(CAL_PATH, "ens_cal")
    w_lgbm  = cal_obj_pre.get("w_lgbm",  0.262) if cal_obj_pre else 0.262
    w_cat   = cal_obj_pre.get("w_cat",   0.479) if cal_obj_pre else 0.479
    w_rank  = cal_obj_pre.get("w_rank",  0.156) if cal_obj_pre else 0.156
    w_trans = cal_obj_pre.get("w_trans", 0.103) if cal_obj_pre else 0.103
    if rank_obj is not None and torch_obj is not None:
        try:
            p_lgbm  = predict_lgbm(df, lgbm_obj)
            p_cat   = predict_catboost(df, cat_obj)
            p_rank  = predict_catboost_rank(df, rank_obj)
            p_trans = predict_transformer_local(df)
            raw = w_lgbm * p_lgbm + w_cat * p_cat + w_rank * p_rank + w_trans * p_trans
        except Exception as e:
            logger.warning(f"4モデル予測失敗（2モデルにフォールバック）: {e}")
            raw = 0.5 * predict_lgbm(df, lgbm_obj) + 0.5 * predict_catboost(df, cat_obj)
    elif rank_obj is not None:
        try:
            p_lgbm = predict_lgbm(df, lgbm_obj)
            p_cat  = predict_catboost(df, cat_obj)
            p_rank = predict_catboost_rank(df, rank_obj)
            raw = 0.4 * p_lgbm + 0.4 * p_cat + 0.2 * p_rank
        except Exception as e:
            logger.warning(f"YetiRank予測失敗（2モデルにフォールバック）: {e}")
            raw = 0.5 * predict_lgbm(df, lgbm_obj) + 0.5 * predict_catboost(df, cat_obj)
    else:
        # フォールバック: 2モデル平均
        raw = 0.5 * predict_lgbm(df, lgbm_obj) + 0.5 * predict_catboost(df, cat_obj)
    cal_obj = _get_cached(CAL_PATH, "ens_cal")
    if cal_obj is not None:
        return cal_obj["calibrator"].transform(raw)
    logger.warning("キャリブレーター未生成。calibrate.py を先に実行してください。")
    return raw


def assign_marks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mark"] = ""
    ranked = df["prob"].rank(ascending=False, method="first")
    for idx, r in ranked.items():
        if r <= 5:
            df.at[idx, "mark"] = {1:"◎",2:"◯",3:"▲",4:"△",5:"×"}[int(r)]
    return df


# =========================================================
# 買い目生成
# =========================================================
def floor_to_unit(x: int, unit: int = MIN_UNIT) -> int:
    return max((x // unit) * unit, unit)


def get_bets(race_df: pd.DataFrame, place: str, cls_raw: str,
             strategy: dict, budget: int) -> dict:
    """HAHO（馬連◎軸2点+三連複ボックス1点）/ HALO（三連複ボックス1点のみ）
       / LALO（複勝◎1点のみ）/ CQC（単勝◎1点のみ）を生成する。"""
    cls      = CLASS_NORMALIZE.get(cls_raw, cls_raw)
    bet_info = strategy.get(place, {}).get(cls) or strategy.get(place, {}).get(cls_raw, {})

    result: dict = {
        "HAHO_戦略対象":      False,   # 実際にベットが生成されたときのみ True
        "HAHO_馬連_買い目":   "", "HAHO_馬連_購入額":   0,
        "HAHO_三連複_買い目": "", "HAHO_三連複_購入額": 0,
        "HALO_戦略対象":      False,
        "HALO_三連複_買い目": "", "HALO_三連複_購入額": 0,
        "LALO_戦略対象":      False,
        "LALO_複勝_買い目":   "", "LALO_複勝_購入額":   0,
        "CQC_戦略対象":       False,
        "CQC_単勝_買い目":    "", "CQC_単勝_購入額":    0,
    }
    if not bet_info:
        return result

    marks_df = {m: race_df[race_df["mark"] == m] for m in ["◎","◯","▲"]}
    hon    = marks_df["◎"]
    taikou = marks_df["◯"]
    sabo   = marks_df["▲"]
    if hon.empty:
        return result

    h1 = int(hon.iloc[0]["馬番"])
    h2 = int(taikou.iloc[0]["馬番"]) if not taikou.empty else None
    h3 = int(sabo.iloc[0]["馬番"])   if not sabo.empty  else None

    # ── HAHO: 馬連◎軸2点 + 三連複ボックス1点 ──────────────────────────
    haho_types = {k: v for k, v in bet_info.items() if k in ("馬連", "三連複")}
    if haho_types and "馬連" in haho_types and h2 and h3:  # 馬連が戦略にない場合はHALOに任せる
        total_ratio = sum(v["bet_ratio"] for v in haho_types.values()) or 1.0
        # 馬連: ◎-◯, ◎-▲ の2点
        if "馬連" in haho_types:
            r   = haho_types["馬連"]["bet_ratio"]
            amt = floor_to_unit(int(budget * r / total_ratio))
            cbs = [(h1, h2), (h1, h3)]
            per = floor_to_unit(amt // len(cbs))
            result["HAHO_馬連_買い目"] = " / ".join(f"{min(a,b)}-{max(a,b)}" for a, b in cbs)
            result["HAHO_馬連_購入額"] = per * len(cbs)
        # 三連複: ◎◯▲ボックス1点
        if "三連複" in haho_types:
            r    = haho_types["三連複"]["bet_ratio"]
            amt  = floor_to_unit(int(budget * r / total_ratio))
            c_sf = tuple(sorted([h1, h2, h3]))
            result["HAHO_三連複_買い目"] = "-".join(map(str, c_sf))
            result["HAHO_三連複_購入額"] = amt
        # ベットが1件でも生成されたら対象フラグを立てる
        if result["HAHO_馬連_買い目"] or result["HAHO_三連複_買い目"]:
            result["HAHO_戦略対象"] = True

    # ── HALO: 三連複ボックス1点のみ（全予算）──────────────────────────
    if "三連複" in bet_info and h2 and h3:
        result["HALO_戦略対象"] = True
        c_sf = tuple(sorted([h1, h2, h3]))
        result["HALO_三連複_買い目"] = "-".join(map(str, c_sf))
        result["HALO_三連複_購入額"] = floor_to_unit(budget)

    # ── LALO: 複勝◎1点のみ（全予算）────────────────────────────────────
    if bet_info:   # 戦略対象レースなら常に発動
        result["LALO_戦略対象"]    = True
        result["LALO_複勝_買い目"] = str(h1)
        result["LALO_複勝_購入額"] = floor_to_unit(budget)

    # ── CQC: 単勝◎1点のみ（全予算）─────────────────────────────────────
    if bet_info:   # 戦略対象レースなら常に発動
        result["CQC_戦略対象"]   = True
        result["CQC_単勝_買い目"] = str(h1)
        result["CQC_単勝_購入額"] = floor_to_unit(budget)

    return result


# =========================================================
# メイン
# =========================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="週末CSV予想結果出力")
    parser.add_argument("--csv",    required=True, help="入力CSVパス (例: data/weekly/20260301.csv)")
    parser.add_argument("--budget", type=int, default=10000, help="1レース予算（円）")
    parser.add_argument("--out",    default="", help="出力CSVパス（省略時は自動命名）")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error(f"CSVが見つかりません: {csv_path}")
        sys.exit(1)

    # モデル・戦略読み込み
    logger.info("モデル読み込み中...")
    lgbm_obj = joblib.load(LGBM_PATH)
    cat_obj  = joblib.load(CAT_PATH)
    with open(STRATEGY_JSON, encoding="utf-8") as f:
        strategy = json.load(f)

    # CSV パース
    df = parse_csv(csv_path)

    # 全レース予測
    logger.info("予測開始...")
    rows = []
    race_id_col = "レースID(新/馬番無)"

    for race_id, race_df in df.groupby(race_id_col):
        race_df = race_df.copy().reset_index(drop=True)
        meta    = race_df.iloc[0]
        place   = str(meta.get("場所",""))
        cls_raw = str(meta.get("クラス名",""))
        shida   = str(meta.get("芝・ダ",""))
        dist    = str(meta.get("距離",""))
        date    = str(meta.get("日付S",""))
        r_num   = str(meta.get("R",""))

        try:
            race_df["prob"]  = ensemble_predict(race_df, lgbm_obj, cat_obj)
            race_df          = assign_marks(race_df)
            race_df["score"] = (race_df["prob"] * 100).round(1)
            # 期待値スコア = 複勝確率 × 単勝オッズ（市場の過小評価を検出）
            tansho = pd.to_numeric(race_df.get("単勝", pd.Series(dtype=float)), errors="coerce")
            race_df["ev_score"] = (race_df["prob"] * tansho).round(2)
        except Exception as e:
            logger.warning(f"予測失敗 {race_id}: {e}")
            race_df["prob"]     = 0.0
            race_df["mark"]     = ""
            race_df["score"]    = 0.0
            race_df["ev_score"] = 0.0

        # 除外フィルタ: 東京・小倉・新馬・障害
        if place in EXCLUDE_PLACES or cls_raw in EXCLUDE_CLASSES:
            bets = {
                "HAHO_戦略対象": False,
                "HAHO_馬連_買い目": "", "HAHO_馬連_購入額": 0,
                "HAHO_三連複_買い目": "", "HAHO_三連複_購入額": 0,
                "HALO_戦略対象": False,
                "HALO_三連複_買い目": "", "HALO_三連複_購入額": 0,
                "LALO_戦略対象": False,
                "LALO_複勝_買い目": "", "LALO_複勝_購入額": 0,
                "CQC_戦略対象":  False,
                "CQC_単勝_買い目": "", "CQC_単勝_購入額":  0,
            }
        else:
            # 買い目生成
            bets = get_bets(race_df, place, cls_raw, strategy, args.budget)

        for _, row in race_df.sort_values("馬番").iterrows():
            is_hon = row["mark"] == "◎"
            rows.append({
                "日付":                date,
                "場所":                place,
                "R":                   r_num,
                "クラス":              cls_raw,
                "距離":                f"{shida}{dist}m",
                "レースID":            race_id,
                "馬番":                int(row["馬番"]) if pd.notna(row["馬番"]) else "",
                "馬名":                str(row.get("馬名","")),
                "騎手":                str(row.get("騎手","")),
                "スコア":              float(row["score"]),
                "単勝オッズ":          float(row["単勝"]) if pd.notna(row.get("単勝")) else "",
                "期待値スコア":        float(row.get("ev_score", 0.0)),
                "印":                  str(row["mark"]),
                "HAHO_戦略対象":       "✅" if bets["HAHO_戦略対象"] else "",
                "HAHO_馬連_買い目":    bets["HAHO_馬連_買い目"]    if is_hon else "",
                "HAHO_馬連_購入額":    bets["HAHO_馬連_購入額"]    if is_hon else "",
                "HAHO_三連複_買い目":  bets["HAHO_三連複_買い目"]  if is_hon else "",
                "HAHO_三連複_購入額":  bets["HAHO_三連複_購入額"]  if is_hon else "",
                "HALO_戦略対象":       "✅" if bets["HALO_戦略対象"] else "",
                "HALO_三連複_買い目":  bets["HALO_三連複_買い目"]  if is_hon else "",
                "HALO_三連複_購入額":  bets["HALO_三連複_購入額"]  if is_hon else "",
                "LALO_戦略対象":       "✅" if bets["LALO_戦略対象"] else "",
                "LALO_複勝_買い目":    bets["LALO_複勝_買い目"]    if is_hon else "",
                "LALO_複勝_購入額":    bets["LALO_複勝_購入額"]    if is_hon else "",
                "CQC_戦略対象":        "✅" if bets["CQC_戦略対象"]  else "",
                "CQC_単勝_買い目":     bets["CQC_単勝_買い目"]     if is_hon else "",
                "CQC_単勝_購入額":     bets["CQC_単勝_購入額"]     if is_hon else "",
            })

    out_df = pd.DataFrame(rows)

    # 出力
    out_path = Path(args.out) if args.out else BASE_DIR / "reports" / f"pred_{csv_path.stem}.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"出力完了: {out_path}")

    # サマリ表示
    haho_races = out_df[out_df["HAHO_戦略対象"]=="✅"]["レースID"].nunique()
    halo_races = out_df[out_df["HALO_戦略対象"]=="✅"]["レースID"].nunique()
    lalo_races = out_df[out_df["LALO_戦略対象"]=="✅"]["レースID"].nunique()
    cqc_races  = out_df[out_df["CQC_戦略対象"] =="✅"]["レースID"].nunique()
    print(f"\n{'='*50}")
    print(f"予想完了: {out_df['レースID'].nunique()}レース / {len(out_df)}頭")
    print(f"HAHO対象: {haho_races}R  HALO対象: {halo_races}R  LALO対象: {lalo_races}R  CQC対象: {cqc_races}R")
    print(f"出力先:   {out_path}")
    print(f"{'='*50}")

    # 戦略対象レースの買い目サマリ
    hon_rows = out_df[out_df["印"]=="◎"].copy()
    if not hon_rows.empty:
        haho_disp = hon_rows[hon_rows["HAHO_戦略対象"]=="✅"][[
            "日付","場所","R","クラス","距離","馬名",
            "HAHO_馬連_買い目","HAHO_馬連_購入額",
            "HAHO_三連複_買い目","HAHO_三連複_購入額",
        ]]
        if not haho_disp.empty:
            print("\n【HAHO 買い目一覧（馬連◎軸2点 + 三連複ボックス1点）】")
            print(haho_disp.to_string(index=False))
        halo_disp = hon_rows[hon_rows["HALO_戦略対象"]=="✅"][[
            "日付","場所","R","クラス","距離","馬名",
            "HALO_三連複_買い目","HALO_三連複_購入額",
        ]]
        if not halo_disp.empty:
            print("\n【HALO 買い目一覧（三連複ボックス1点のみ）】")
            print(halo_disp.to_string(index=False))
        lalo_disp = hon_rows[hon_rows["LALO_戦略対象"]=="✅"][[
            "日付","場所","R","クラス","距離","馬名",
            "LALO_複勝_買い目","LALO_複勝_購入額",
        ]]
        if not lalo_disp.empty:
            print("\n【LALO 買い目一覧（複勝◎1点のみ）】")
            print(lalo_disp.to_string(index=False))
        cqc_disp = hon_rows[hon_rows["CQC_戦略対象"]=="✅"][[
            "日付","場所","R","クラス","距離","馬名",
            "CQC_単勝_買い目","CQC_単勝_購入額",
        ]]
        if not cqc_disp.empty:
            print("\n【CQC 買い目一覧（単勝◎1点のみ）】")
            print(cqc_disp.to_string(index=False))


if __name__ == "__main__":
    main()