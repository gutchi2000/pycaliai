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

from ev_filter import BetFilter, EVCalibrator, is_upgrade_race

# ── フィルタ・キャリブレーター（グローバルインスタンス） ──
_bet_filter   = BetFilter()
_ev_calibrator = EVCalibrator()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR      = Path(__file__).parent
MODEL_DIR     = BASE_DIR / "models"
STRATEGY_JSON = BASE_DIR / "data" / "strategy_weights.json"
LGBM_PATH      = MODEL_DIR / "lgbm_optuna_v1.pkl"
CAT_PATH       = MODEL_DIR / "catboost_optuna_v1.pkl"
RANK_PATH      = MODEL_DIR / "catboost_rank_v1.pkl"
CAL_PATH       = MODEL_DIR / "ensemble_calibrator_v4.pkl"   # Test 2024-fit (2026-03-25更新)
CAL_PATH_V3    = MODEL_DIR / "ensemble_calibrator_v3.pkl"   # Train-based fallback
CAL_PATH_V2    = MODEL_DIR / "ensemble_calibrator_v2.pkl"
CAL_PATH_V1    = MODEL_DIR / "ensemble_calibrator_v1.pkl"
WIN_PATH       = MODEL_DIR / "lgbm_win_v1.pkl"
FUKU_LGBM_PATH = MODEL_DIR / "lgbm_fukusho_v1.pkl"
FUKU_CAT_PATH  = MODEL_DIR / "catboost_fukusho_v1.pkl"
RANK_LGBM_PATH = MODEL_DIR / "lgbm_rank_v1.pkl"        # LambdaRank (Phase 5)
REGRESS_PATH   = MODEL_DIR / "lgbm_regression_v1.pkl"   # 着順回帰 (Phase 5)
ENS_WEIGHTS_PATH = MODEL_DIR / "ensemble_weights.json"  # 最適化重み (Phase 5)
# Phase 5+: 距離別 Mixture of Experts
EXPERT_PATHS = {
    "turf_short": MODEL_DIR / "expert_turf_short.pkl",
    "turf_mid":   MODEL_DIR / "expert_turf_mid.pkl",
    "turf_long":  MODEL_DIR / "expert_turf_long.pkl",
    "dirt":       MODEL_DIR / "expert_dirt.pkl",
}
VALUE_MODEL_PATH = MODEL_DIR / "value_model_v2.pkl"
ORDER_MODEL_PATH = MODEL_DIR / "order_model_v1.pkl"  # 着順3クラス予測モデル（HALOフォーメーション用）
RETURN_RATE_FUKU = 0.75   # 複勝控除率
TORCH_PATH     = MODEL_DIR / "transformer_pl_v2.pkl"
META_PATH      = MODEL_DIR / "stacking_meta_v1.pkl"
STACK_CAL_PATH = MODEL_DIR / "stacking_calibrator_v1.pkl"
RETURN_RATE_TAN = 0.80  # 単勝控除率
LIVE_CSV = BASE_DIR / "data" / "live_results_2026.csv"  # ライブ予測記録（追記式）

# モデルキャッシュ（プロセス内で1回だけロード）
_model_cache: dict = {}

def _get_cached(path: Path, key: str):
    if key not in _model_cache and path.exists():
        _model_cache[key] = joblib.load(path)
    return _model_cache.get(key)

MIN_UNIT        = 100
EXCLUDE_PLACES  = {"東京", "小倉"}
EXCLUDE_CLASSES = {"新馬", "障害"}

# SegmentBetFilter（app.py と同一定義）
SEGMENT_BET_BLACKLIST = {
    ("dirt",       "三連複"),
    ("turf_short", "馬連"),
    ("turf_short", "複勝"),
    ("turf_mid",   "馬連"),
}
SEGMENT_CLASS_BET_BLACKLIST = {
    ("dirt", "未勝利", "馬連"),
    ("dirt", "オープン","馬連"),
    ("dirt", "3勝",   "馬連"),
    ("dirt", "GⅠ",   "馬連"),
    ("dirt", "GⅡ",   "馬連"),
    ("dirt", "GⅢ",   "馬連"),
    ("dirt", "OP(L)", "馬連"),
    ("dirt", "1勝", "複勝"),
}


def _race_segment(td: str, dist) -> str:
    try:
        d = int(dist)
    except (TypeError, ValueError):
        return "unknown"
    if td == "ダ":
        return "dirt"
    if d <= 1400:
        return "turf_short"
    if d <= 2200:
        return "turf_mid"
    return "turf_long"


def _is_class_blacklisted(seg: str, cls_raw: str, bt: str) -> bool:
    return (seg, cls_raw, bt) in SEGMENT_CLASS_BET_BLACKLIST

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
KAKO5_DIR  = BASE_DIR / "data" / "kako5"

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


def _load_kako5_warnings(date_str: str) -> dict[str, str]:
    """kako5/YYYYMMDD.csv を読み、過去5走に中止(止)・除外(外)・取消(消)がある馬名→警告文 を返す。

    警告例: {"アドマイヤテラ": "2走前中止", "○○": "1走前除外・3走前中止"}
    ファイルが存在しない場合は空dictを返す。
    """
    # 日付変換: "2026.3.8" → "20260308"
    try:
        from datetime import datetime
        d = datetime.strptime(date_str.strip(), "%Y.%m.%d") if "-" not in date_str else \
            datetime.strptime(date_str.strip(), "%Y-%m-%d")
    except ValueError:
        # "2026.3.8" のようにゼロ埋めなし
        parts = date_str.strip().replace("-", ".").split(".")
        d = datetime(int(parts[0]), int(parts[1]), int(parts[2]))

    kako5_path = KAKO5_DIR / f"{d.strftime('%Y%m%d')}.csv"
    if not kako5_path.exists():
        return {}

    STOP_CODES = {"止", "外", "消"}
    CODE_LABEL = {"止": "中止", "外": "除外", "消": "取消"}
    # 着順の列インデックス（1走前=18, 2走前=30, ..., 5走前=66）
    CHAKU_IDXS = {1: 18, 2: 30, 3: 42, 4: 54, 5: 66}
    UMANAME_IDX = 7  # 馬名S

    warnings: dict[str, str] = {}
    in_horse_section = False

    try:
        with open(kako5_path, encoding="cp932", errors="replace") as f:
            for line in f:
                parts = line.rstrip("\n").split(",")
                # 馬ヘッダー行を検出
                if parts and parts[0].strip() == "枠番":
                    in_horse_section = True
                    continue
                # レース情報行（先頭がレースIDっぽい数字16桁）はスキップ
                if not in_horse_section:
                    continue
                if len(parts) < 20:
                    continue
                # 馬名を取得
                umaname = parts[UMANAME_IDX].strip() if len(parts) > UMANAME_IDX else ""
                if not umaname:
                    continue
                # 過去5走の着順チェック
                warns = []
                for n, idx in CHAKU_IDXS.items():
                    if idx >= len(parts):
                        break
                    val = parts[idx].strip()
                    if val in STOP_CODES:
                        warns.append(f"{n}走前{CODE_LABEL[val]}")
                if warns:
                    warnings[umaname] = "・".join(warns)
    except Exception as e:
        logger.warning(f"kako5警告読み込み失敗: {e}")

    return warnings


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

    # ── 過去5走特徴量（data/kako5/YYYYMMDD.csv）があればマージ ──
    kako5_path = KAKO5_DIR / f"{date_str}.csv"
    if kako5_path.exists():
        try:
            from parse_kako5 import build_from_kako5, KAKO5_COLS
            kako5_df = build_from_kako5(kako5_path)
            if not kako5_df.empty:
                # kako5_df: [レースID(新), 馬番, ...kako5特徴量]
                # df: レースID(新/馬番無) = 16桁, 馬番 = int
                # kako5: レースID(新) = 18桁(レース16+馬番2)
                # kako5のレースID(新)は18桁なので16桁に変換
                if kako5_df["レースID(新)"].astype(str).str.len().mode().iloc[0] > 16:
                    kako5_df["レースID(新)"] = kako5_df["レースID(新)"].astype(str).str[:16]
                df = df.merge(
                    kako5_df.rename(columns={"レースID(新)": "レースID(新/馬番無)"}),
                    on=["レースID(新/馬番無)", "馬番"],
                    how="left",
                    suffixes=("", "_kako5"),
                )
                valid_pct = df["kako5_avg_pos"].notna().mean() * 100
                logger.info(f"kako5 CSV読み込み済: カバレッジ={valid_pct:.1f}%")
        except Exception as e:
            logger.warning(f"kako5マージ失敗: {e}")
            for col in ["kako5_avg_pos", "kako5_std_pos", "kako5_best_pos",
                         "kako5_avg_agari3f", "kako5_best_agari3f",
                         "kako5_same_td_ratio", "kako5_same_dist_ratio", "kako5_same_place_ratio",
                         "kako5_pos_trend", "kako5_race_count"]:
                if col not in df.columns:
                    df[col] = float("nan")
    else:
        logger.info(f"kako5 CSV なし: {kako5_path}")
        for col in ["kako5_avg_pos", "kako5_std_pos", "kako5_best_pos",
                     "kako5_avg_agari3f", "kako5_best_agari3f",
                     "kako5_same_td_ratio", "kako5_same_dist_ratio", "kako5_same_place_ratio",
                     "kako5_pos_trend", "kako5_race_count"]:
            if col not in df.columns:
                df[col] = float("nan")

    # ── 調教データ（坂路・WCマスターCSVからJOIN）──
    try:
        from optuna_lgbm import load_chukyo, merge_chukyo
        hanro, wc = load_chukyo()
        # merge_chukyo は "日付"(YYYYMMDD) を参照するが週次CSVは "日付S"(YYYY.M.D) なので変換
        if "日付" not in df.columns and "日付S" in df.columns:
            df["日付"] = df["日付S"].apply(
                lambda s: "{:04d}{:02d}{:02d}".format(*[int(x) for x in str(s).split(".")])
            )
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


def predict_lgbm_win(df: pd.DataFrame, obj: dict) -> np.ndarray:
    """lgbm_win_v1 (is_1st_place) 予測。predict_lgbm と同じロジック。"""
    return predict_lgbm(df, obj)


def predict_lgbm_fukusho(df: pd.DataFrame, obj: dict) -> np.ndarray:
    """lgbm_fukusho_v1 (複勝/is_top3) 予測。predict_lgbm と同じロジック。"""
    return predict_lgbm(df, obj)


def predict_catboost_fukusho(df: pd.DataFrame, obj: dict) -> np.ndarray:
    """catboost_fukusho_v1 (複勝/is_top3) 予測。predict_catboost と同じロジック。"""
    return predict_catboost(df, obj)


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


def predict_lgbm_rank(df: pd.DataFrame, obj: dict) -> np.ndarray:
    """LambdaRankモデルで予測。スコアをレース内min-max正規化して[0,1]に変換。"""
    model = obj["model"]
    encoders = obj["encoders"]
    feature_cols = obj["feature_cols"]
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
    _ROLLING_COLS = {"jockey_fuku30","jockey_fuku90","trainer_fuku30","trainer_fuku90",
                     "horse_fuku10","horse_fuku30","前走補9","前走補正",
                     "trn_hanro_4f","trn_hanro_3f","trn_hanro_2f","trn_hanro_1f",
                     "trn_hanro_lap1","trn_hanro_lap2","trn_hanro_lap3","trn_hanro_lap4",
                     "trn_hanro_days",
                     "trn_wc_5f","trn_wc_4f","trn_wc_3f",
                     "trn_wc_lap1","trn_wc_lap2","trn_wc_lap3",
                     "trn_wc_days","前走単勝オッズ"}
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan if col in _ROLLING_COLS else 0
    scores = model.predict(df[feature_cols])
    result = np.full(len(df), 0.5)
    df_reset = df.reset_index(drop=True)
    for race_id, group in df_reset.groupby("レースID(新/馬番無)"):
        idx = group.index.tolist()
        s = scores[idx]
        s_min, s_max = s.min(), s.max()
        if s_max > s_min:
            result[idx] = (s - s_min) / (s_max - s_min)
    return result


def predict_lgbm_regression(df: pd.DataFrame, obj: dict) -> np.ndarray:
    """着順回帰モデルで予測。着順を反転してレース内正規化で[0,1]に変換。"""
    model = obj["model"]
    encoders = obj["encoders"]
    feature_cols = obj["feature_cols"]
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
    _ROLLING_COLS = {"jockey_fuku30","jockey_fuku90","trainer_fuku30","trainer_fuku90",
                     "horse_fuku10","horse_fuku30","前走補9","前走補正",
                     "trn_hanro_4f","trn_hanro_3f","trn_hanro_2f","trn_hanro_1f",
                     "trn_hanro_lap1","trn_hanro_lap2","trn_hanro_lap3","trn_hanro_lap4",
                     "trn_hanro_days",
                     "trn_wc_5f","trn_wc_4f","trn_wc_3f",
                     "trn_wc_lap1","trn_wc_lap2","trn_wc_lap3",
                     "trn_wc_days","前走単勝オッズ"}
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan if col in _ROLLING_COLS else 0
    pred_pos = model.predict(df[feature_cols])
    result = np.full(len(df), 0.5)
    df_reset = df.reset_index(drop=True)
    for race_id, group in df_reset.groupby("レースID(新/馬番無)"):
        idx = group.index.tolist()
        s = -pred_pos[idx]  # 着順を反転（小さい着順=高スコア）
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


def _load_ensemble_weights(segment: str | None = None) -> dict[str, float] | None:
    """optimize_weights.py で生成した最適化重みをロードする。
    segment 指定時は Expert 別重み ensemble_weights_{segment}.json を優先。"""
    if segment is not None:
        seg_path = MODEL_DIR / f"ensemble_weights_{segment}.json"
        if seg_path.exists():
            try:
                with open(seg_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info(f"Expert別重みロード: {seg_path.name} valid={data.get('valid_auc')}")
                return data.get("weights", {})
            except Exception as e:
                logger.warning(f"Expert別重みロード失敗: {e}")
    if ENS_WEIGHTS_PATH.exists():
        try:
            with open(ENS_WEIGHTS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            weights = data.get("weights", {})
            logger.info(f"最適化重みロード: {ENS_WEIGHTS_PATH.name} (method={data.get('method')})")
            return weights
        except Exception as e:
            logger.warning(f"最適化重みロード失敗: {e}")
    return None


# Expert別重みに含めるべきセグメント（AUCが上がったもののみ）
SEGMENT_WEIGHTS_AVAILABLE = {"turf_mid", "dirt"}


def _select_segment(df: pd.DataFrame) -> str | None:
    """距離・トラックからセグメント名を返す（pkl 存在は見ない）。"""
    if len(df) == 0:
        return None
    td = str(df.iloc[0].get("芝・ダ", ""))
    try:
        d = int(df.iloc[0].get("距離", 0))
    except (TypeError, ValueError):
        return None
    if td == "ダ":
        seg = "dirt"
    elif d <= 1400:
        seg = "turf_short"
    elif d <= 2200:
        seg = "turf_mid"
    else:
        seg = "turf_long"
    return seg if seg in SEGMENT_WEIGHTS_AVAILABLE else None


def _select_expert(td: str, dist) -> str | None:
    """距離・トラックから Expert 名を返す。pkl が無い（=未学習/見送り）なら None。"""
    try:
        d = int(dist)
    except (TypeError, ValueError):
        return None
    if td == "ダ":
        name = "dirt"
    elif d <= 1400:
        name = "turf_short"
    elif d <= 2200:
        name = "turf_mid"
    else:
        name = "turf_long"
    if EXPERT_PATHS[name].exists():
        return name
    return None


def _predict_expert(df: pd.DataFrame, name: str) -> np.ndarray | None:
    """Expertモデルで予測。失敗時 None。"""
    try:
        obj = _get_cached(EXPERT_PATHS[name], f"expert_{name}")
        if obj is None:
            return None
        return predict_lgbm(df, obj)
    except Exception as e:
        logger.warning(f"Expert {name} 予測失敗: {e}")
        return None


def ensemble_predict(df: pd.DataFrame, lgbm_obj: dict, cat_obj: dict) -> np.ndarray:
    """
    全モデルアンサンブル予測。
    Phase 5: optimize_weights.py の最適化重みを自動ロード。
    重みファイルがなければ従来のハードコード重みにフォールバック。
    """
    # --- Phase 5+: Expert別重みを優先、なければグローバル重み ---
    segment = _select_segment(df)
    opt_weights = _load_ensemble_weights(segment)
    if opt_weights is not None:
        try:
            raw = _ensemble_with_optimized_weights(df, lgbm_obj, cat_obj, opt_weights)
        except Exception as e:
            logger.warning(f"最適化重みアンサンブル失敗（フォールバック）: {e}")
            raw = _ensemble_fallback(df, lgbm_obj, cat_obj)
    else:
        raw = _ensemble_fallback(df, lgbm_obj, cat_obj)

    # セグメント別重みを使っていない場合のみ、旧式の Expert 単純加重を適用
    if segment is None:
        try:
            if len(df) > 0:
                td = str(df.iloc[0].get("芝・ダ", ""))
                dist = df.iloc[0].get("距離", None)
                expert_name = _select_expert(td, dist)
                if expert_name is not None:
                    p_exp = _predict_expert(df, expert_name)
                    if p_exp is not None:
                        raw = 0.7 * raw + 0.3 * p_exp
                        logger.info(f"Expert適用: {expert_name} (0.7*ens + 0.3*expert)")
        except Exception as e:
            logger.warning(f"Expert適用失敗: {e}")

    # キャリブレーター: v4 -> v3 -> v2 -> v1 優先チェーン
    for cal_p, cal_key in [(CAL_PATH, "ens_cal"), (CAL_PATH_V3, "ens_cal_v3"),
                           (CAL_PATH_V2, "ens_cal_v2"), (CAL_PATH_V1, "ens_cal_v1")]:
        cal_obj = _get_cached(cal_p, cal_key)
        if cal_obj is not None:
            logger.info(f"キャリブレーター使用: {cal_p.name}")
            return cal_obj["calibrator"].transform(raw)
    logger.warning("キャリブレーター未生成。calibrate.py を先に実行してください。")
    return raw


def _ensemble_with_optimized_weights(
    df: pd.DataFrame,
    lgbm_obj: dict,
    cat_obj: dict,
    weights: dict[str, float],
) -> np.ndarray:
    """最適化重みを使った全モデルアンサンブル。"""
    # モデル名 → (ロードキー, パス, 予測関数, obj or None)
    model_map = {
        "lgbm":       ("lgbm_opt",   LGBM_PATH,       predict_lgbm,            lgbm_obj),
        "catboost":   ("cat_opt",    CAT_PATH,         predict_catboost,        cat_obj),
        "fuku_lgbm":  ("fuku_lgbm",  FUKU_LGBM_PATH,  predict_lgbm_fukusho,    None),
        "fuku_cat":   ("fuku_cat",   FUKU_CAT_PATH,    predict_catboost_fukusho, None),
        "rank_cat":   ("rank",       RANK_PATH,        predict_catboost_rank,   None),
        "rank_lgbm":  ("rank_lgbm",  RANK_LGBM_PATH,  predict_lgbm_rank,       None),
        "regression": ("regression", REGRESS_PATH,     predict_lgbm_regression, None),
        "lgbm_win":   ("lgbm_win",   WIN_PATH,         predict_lgbm_win,        None),
    }
    # Phase 5+: Expert モデルも候補に追加（predict_lgbm と同じ形式）
    for exp_name, exp_path in EXPERT_PATHS.items():
        model_map[f"expert_{exp_name}"] = (f"expert_{exp_name}", exp_path, predict_lgbm, None)

    raw = np.zeros(len(df))
    total_w = 0.0
    used_models = []

    for name, w in weights.items():
        if w < 0.001 or name not in model_map:
            continue
        cache_key, path, fn, pre_obj = model_map[name]
        # objが事前に渡されているものはそのまま使う
        if pre_obj is not None:
            obj = pre_obj
        else:
            obj = _get_cached(path, cache_key)
            if obj is None:
                logger.debug(f"  {name}: モデル未存在 ({path.name}), スキップ")
                continue
        try:
            p = fn(df, obj)
            raw += w * p
            total_w += w
            used_models.append(name)
        except Exception as e:
            logger.warning(f"  {name}: 予測失敗 ({e}), スキップ")

    if total_w < 0.01:
        raise RuntimeError("有効なモデルがありません")

    # 使われなかったモデルがある場合、重みを再正規化
    if abs(total_w - 1.0) > 0.01:
        raw = raw / total_w
        logger.info(f"重み再正規化: total_w={total_w:.3f} -> 1.0 (欠損モデルあり)")

    logger.info(f"最適化アンサンブル: {len(used_models)}モデル ({', '.join(used_models)})")
    return raw


def _ensemble_fallback(df: pd.DataFrame, lgbm_obj: dict, cat_obj: dict) -> np.ndarray:
    """最適化重みなし時のフォールバック（従来ロジック）。"""
    rank_obj  = _get_cached(RANK_PATH, "rank")
    win_obj   = _get_cached(WIN_PATH, "lgbm_win")
    fuku_lgbm_obj = _get_cached(FUKU_LGBM_PATH, "fuku_lgbm")
    fuku_cat_obj  = _get_cached(FUKU_CAT_PATH,  "fuku_cat")
    if rank_obj is not None and win_obj is not None:
        try:
            p_lgbm = predict_lgbm(df, lgbm_obj)
            p_cat  = predict_catboost(df, cat_obj)
            p_rank = predict_catboost_rank(df, rank_obj)
            p_win  = predict_lgbm_win(df, win_obj)
            if fuku_lgbm_obj is not None and fuku_cat_obj is not None:
                p_fuku_lgbm = predict_lgbm_fukusho(df, fuku_lgbm_obj)
                p_fuku_cat  = predict_catboost_fukusho(df, fuku_cat_obj)
                return (0.25 * p_lgbm + 0.25 * p_cat + 0.20 * p_rank
                        + 0.20 * p_win + 0.05 * p_fuku_lgbm + 0.05 * p_fuku_cat)
            return 0.30 * p_lgbm + 0.30 * p_cat + 0.20 * p_rank + 0.20 * p_win
        except Exception as e:
            logger.warning(f"4/6モデル予測失敗: {e}")
    return 0.5 * predict_lgbm(df, lgbm_obj) + 0.5 * predict_catboost(df, cat_obj)


def assign_marks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mark"] = ""
    ranked = df["prob"].rank(ascending=False, method="first")
    MARK_MAP = {1:"◎", 2:"◯", 3:"▲", 4:"△", 5:"☆", 6:"★"}
    for idx, r in ranked.items():
        if r <= 6:
            df.at[idx, "mark"] = MARK_MAP[int(r)]
    return df


# =========================================================
# Value Model (Layer 2) — 市場の歪みを検出して買い判定
# =========================================================
def compute_value_scores(race_df: pd.DataFrame,
                         od_odds: pd.DataFrame | None = None,
                         ) -> tuple[pd.Series, pd.Series]:
    """Value Model でレース内各馬の predicted ROI と cal_prob を計算する。

    Returns:
        (pred_roi, cal_prob) - 両方とも pd.Series
    """
    nan_result = (pd.Series(np.nan, index=race_df.index),
                  pd.Series(np.nan, index=race_df.index))

    value_obj = _get_cached(VALUE_MODEL_PATH, "value_model")
    if value_obj is None:
        return nan_result

    vm = value_obj["model"]
    feats = value_obj["features"]
    iso = value_obj.get("calibrator")

    # オッズ取得: OD CSVがあれば使う、なければweekly CSVの単勝から推定
    tansho = pd.to_numeric(race_df.get("単勝", pd.Series(dtype=float)), errors="coerce")
    fuku_low = pd.Series(np.nan, index=race_df.index)
    fuku_high = pd.Series(np.nan, index=race_df.index)

    if od_odds is not None:
        race_id = str(race_df.get("レースID(新/馬番無)", race_df.get("race_id", pd.Series())).iloc[0]
                      ) if "レースID(新/馬番無)" in race_df.columns else ""
        od_race = od_odds[od_odds["race_id"] == race_id] if race_id else pd.DataFrame()
        if len(od_race) > 0:
            # OD CSVの馬番で結合
            horse_nums = race_df["馬番"].astype(int)
            od_map = od_race.set_index("horse_num")
            for idx, hnum in horse_nums.items():
                if hnum in od_map.index:
                    tansho.at[idx] = od_map.at[hnum, "tan_odds"]
                    fuku_low.at[idx] = od_map.at[hnum, "fuku_low"]
                    fuku_high.at[idx] = od_map.at[hnum, "fuku_high"]

    # 複勝未取得 → 単勝から推定
    if fuku_low.isna().all():
        fuku_low = tansho.pow(0.6).round(1)
        fuku_high = (tansho.pow(0.6) * 1.5).round(1)

    if tansho.isna().all():
        return nan_result

    # Calibrate prob
    raw_prob = race_df["prob"].values
    if iso is not None:
        cal_prob_vals = iso.transform(raw_prob)
    else:
        cal_prob_vals = raw_prob

    # Feature engineering (match train_value_model.py)
    fuku_mid = (fuku_low + fuku_high) / 2
    model_rank = race_df["prob"].rank(ascending=False, method="first")
    ninki = tansho.rank(method="first", ascending=True)
    shutsuu = len(race_df)

    X = pd.DataFrame({
        "cal_prob": cal_prob_vals,
        "model_rank": model_rank.values,
        "ninki": ninki.values,
        "tan_odds": tansho.values,
        "fuku_mid": fuku_mid.values,
        "EV_fuku": cal_prob_vals * fuku_mid.values,
        "disagree": model_rank.values - ninki.values,
        "abs_disagree": np.abs(model_rank.values - ninki.values),
        "log_tan_odds": np.log1p(tansho.values),
        "log_fuku_mid": np.log1p(fuku_mid.values),
        "odds_rank_ratio": tansho.values / (ninki.values + 0.5),
        "model_vs_market_prob": cal_prob_vals - (1 / tansho.values),
        "shutsuu": shutsuu,
        "fuku_spread": (fuku_high - fuku_low).values,
        "fuku_spread_ratio": ((fuku_high - fuku_low) / (fuku_mid + 0.01)).values,
    }, index=race_df.index)

    # Predict
    pred_roi = vm.predict(X[feats])
    return (pd.Series(pred_roi, index=race_df.index),
            pd.Series(cal_prob_vals, index=race_df.index))


# =========================================================
# 買い目生成
# =========================================================
def floor_to_unit(x: int, unit: int = MIN_UNIT) -> int:
    return max((x // unit) * unit, unit)


def _predict_order_proba(race_df: pd.DataFrame):
    """着順予測モデル（order_model_v1.pkl）で各馬の (p_win, p_place23, p_out) を返す。
    モデルが無い場合 None。"""
    order_obj = _get_cached(ORDER_MODEL_PATH, "order_model")
    if order_obj is None:
        return None
    try:
        model = order_obj["model"]
        feats = order_obj["features"]
        encs  = order_obj["encoders"]
        df = race_df.copy()
        for col, le in encs.items():
            if col in df.columns:
                df[col] = df[col].astype(str).fillna("__NaN__")
                known = set(le.classes_)
                df[col] = df[col].apply(lambda x: x if x in known else "__unknown__")
                if "__unknown__" not in le.classes_:
                    le.classes_ = np.append(le.classes_, "__unknown__")
                df[col] = le.transform(df[col])
        from utils import parse_time_str as _pts
        for col in ["前走走破タイム", "前走着差タイム"]:
            if col in df.columns:
                df[col] = _pts(df[col])
        for f in feats:
            if f not in df.columns:
                df[f] = np.nan
        proba = model.predict_proba(df[feats])
        result = pd.DataFrame(proba, columns=["p_win", "p_place23", "p_out"], index=race_df.index)
        result["馬番"] = race_df["馬番"].values
        return result
    except Exception as e:
        logger.debug(f"order model predict failed: {e}")
        return None


def get_bets(race_df: pd.DataFrame, place: str, cls_raw: str,
             strategy: dict, budget: int) -> dict:
    """全6プラン(HAHO/HALO/STANDARD/LALO/CQC/TRIPLE)の買い目を flat dict で返す。
    app.py の get_bets() と同じ戦略ロジックだが、CSV出力用に flat 形式。"""
    import itertools
    cls      = CLASS_NORMALIZE.get(cls_raw, cls_raw)
    bet_info = strategy.get(place, {}).get(cls) or strategy.get(place, {}).get(cls_raw, {})

    result: dict = {
        "HAHO_戦略対象": False,
        "HAHO_三連複_買い目": "", "HAHO_三連複_購入額": 0, "HAHO_三連複_点数": 0,
        "HALO_戦略対象": False,
        "HALO_三連単_買い目": "", "HALO_三連単_購入額": 0, "HALO_三連単_点数": 0,
        "STANDARD_戦略対象": False,
        "STANDARD_戦略対象": False,
        "STANDARD_単勝_買い目": "", "STANDARD_単勝_購入額": 0,
        "STANDARD_複勝_買い目": "", "STANDARD_複勝_購入額": 0,
        "STANDARD_馬連_買い目": "", "STANDARD_馬連_購入額": 0,
        "TRIPLE_戦略対象": False,
        "TRIPLE_三連複_買い目": "", "TRIPLE_三連複_購入額": 0,
        "TRIPLE_複勝_買い目": "",   "TRIPLE_複勝_購入額":   0,
    }

    if place in EXCLUDE_PLACES or cls_raw in EXCLUDE_CLASSES:
        return result

    # 全マーク馬の馬番を抽出
    h_marks = {}
    hon_row = None
    for m in ["◎","◯","▲","△","☆","★"]:
        rows = race_df[race_df["mark"] == m]
        if not rows.empty:
            h_marks[m] = int(rows.iloc[0]["馬番"])
            if m == "◎":
                hon_row = rows.iloc[0]
    if "◎" not in h_marks:
        return result

    h1 = h_marks["◎"]
    h2 = h_marks.get("◯")
    others_5    = [h_marks[m] for m in ["◯","▲","△","☆","★"] if m in h_marks]
    opponents_4 = [h_marks[m] for m in ["▲","△","☆","★"]     if m in h_marks]

    # 単勝オッズガード
    hon_tansho = 0.0
    try:
        hon_tansho = float(hon_row.get("単勝オッズ", hon_row.get("単勝", 0)) or 0)
    except Exception:
        pass
    MIN_TANSHO_FOR_FUKU = 2.0
    odds_too_low = 0 < hon_tansho < MIN_TANSHO_FOR_FUKU

    # SegmentBetFilter
    _meta_row = race_df.iloc[0]
    _td   = str(_meta_row.get("芝・ダ", _meta_row.get("芝・ダート", "")))
    _dist = _meta_row.get("距離", None)
    _seg  = _race_segment(_td, _dist)
    san_blocked    = (_seg, "三連複") in SEGMENT_BET_BLACKLIST or _is_class_blacklisted(_seg, cls_raw, "三連複")
    fuku_blocked   = (_seg, "複勝")   in SEGMENT_BET_BLACKLIST or _is_class_blacklisted(_seg, cls_raw, "複勝")
    umaren_blocked = (_seg, "馬連")   in SEGMENT_BET_BLACKLIST or _is_class_blacklisted(_seg, cls_raw, "馬連")
    tansho_blocked = (_seg, "単勝")   in SEGMENT_BET_BLACKLIST
    santan_blocked = (_seg, "三連単") in SEGMENT_BET_BLACKLIST

    # ── HAHO: 三連複◎1頭軸-5頭流し（10点×¥1,000）──────────────────
    if len(others_5) >= 2 and not san_blocked:
        combos = []
        for a, b in itertools.combinations(others_5, 2):
            combos.append("-".join(map(str, sorted([h1, a, b]))))
        result["HAHO_戦略対象"]    = True
        result["HAHO_三連複_買い目"] = " / ".join(combos)
        result["HAHO_三連複_購入額"] = 1000 * len(combos)
        result["HAHO_三連複_点数"]   = len(combos)

    # ── HALO: 三連単フォーメーション ──
    # Stage 0 (2026-04-16 反転): スコア差ルールを主軸、着順モデルは
    # 高信頼ケース（◎の絶対勝率 ≥ 0.50 かつ p_hon ≥ p_tai × 2.0）のみ補強
    # 根拠: backtest 5,309R で score-rule ROI 70.37% > order-model ROI 67.58%
    if h2 and not santan_blocked:
        # === Stage 2-05: HALO playbook (条件別 NO_BET / top_n) ===
        _playbook_top_n = 3
        _playbook_no_bet = False
        _playbook_cell   = ""
        try:
            from utils import lookup_halo_policy as _lhp
            _meta_row_pw = race_df.iloc[0] if len(race_df) > 0 else None
            _shiba_da_pw = str(_meta_row_pw.get("芝・ダ", "")) if _meta_row_pw is not None else ""
            _fsize_pw    = len(race_df)
            _policy_pw   = _lhp(_shiba_da_pw, _fsize_pw)
            _playbook_no_bet = bool(_policy_pw.get("no_bet"))
            _playbook_top_n  = max(3, int(_policy_pw.get("top_n", 3)))
            _playbook_cell   = _policy_pw.get("cell", "")
        except Exception:
            pass

        # NO_BET セルは HALO 空欄のままスキップ
        _tri_done = _playbook_no_bet

        # === Stage 2-02: trifecta_model_v1 (LambdaRank + Plackett-Luce) を優先 ===
        if not _playbook_no_bet:
            try:
                _tri_path = BASE_DIR / "models" / "trifecta_model_v1.pkl"
                if _tri_path.exists():
                    import joblib as _jl
                    _tri_obj = _jl.load(_tri_path)
                    _tri_model = _tri_obj.get("model")
                    _tri_feats = _tri_obj.get("feature_cols")
                    if _tri_model is not None and _tri_feats is not None:
                        from train_trifecta_model import add_race_features as _arf, pl_combo_probs as _plc, FEATURE_COLS as _TFC
                        _rdf2 = race_df.copy()
                        if "mark" not in _rdf2.columns:
                            _mk_map = {v: k for k, v in h_marks.items()}
                            _rdf2["mark"] = pd.to_numeric(_rdf2["馬番"], errors="coerce") \
                                              .map(lambda ub: _mk_map.get(int(ub), "") if pd.notna(ub) else "")
                        if "jyun" not in _rdf2.columns:
                            _rdf2["jyun"] = float("nan")
                        if "race_id" not in _rdf2.columns:
                            _rdf2["race_id"] = "tmp"
                        if "place" not in _rdf2.columns:
                            _rdf2["place"] = place
                        if "race_santan_pay" not in _rdf2.columns:
                            _rdf2["race_santan_pay"] = 0.0
                        if "ensemble_prob" not in _rdf2.columns:
                            _rdf2["ensemble_prob"] = pd.to_numeric(
                                _rdf2.get("prob", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
                        _rdf2 = _arf(_rdf2)
                        _X2 = pd.DataFrame(
                            [{f: float(r.get(f, 0)) for f in _TFC} for _, r in _rdf2.iterrows()])
                        _ms2 = _tri_model.predict(_X2.fillna(0))
                        _ubs2 = [int(r["馬番"]) for _, r in race_df.iterrows() if pd.notna(r.get("馬番"))]
                        _sm2 = {ub: float(s) for ub, s in zip(_ubs2, _ms2)}
                        _cwp = _plc(_sm2, top_n=min(8, len(_sm2)))
                        if _cwp:
                            _sel = [c for c, _ in _cwp[:_playbook_top_n]]
                            n = len(_sel)
                            per_bet = max(100, (9600 // n // 100) * 100)
                            result["HALO_戦略対象"]    = True
                            result["HALO_三連単_買い目"] = " / ".join(f"{f}→{s}→{t}" for f, s, t in sorted(_sel))
                            result["HALO_三連単_購入額"] = per_bet * n
                            result["HALO_三連単_点数"]   = n
                            _tri_done = True
            except Exception as _te:
                pass  # フォールバック: スコア差ルールへ

        if not _tri_done:
            # === Stage 2-01b: Optuna 最適化済み閾値を halo_thresholds.json から読み込み ===
            from utils import load_halo_thresholds as _lht
            _hthr = _lht()
            _gap12_hi  = _hthr["gap_12_hi"]    # default 10 → opt 3.68
            _gap12_lo  = _hthr["gap_12_lo"]    # default  5 → opt 2.54
            _gap_top4  = _hthr["gap_top4_lo"]  # default 15 → opt 27.04
            _pw_min    = _hthr["pw_min"]       # default 0.50 → opt 0.73
            _pw_ratio  = _hthr["pw_ratio"]     # default 2.0  → opt 1.58

            # === スコア差ルール（一次採用） ===
            scores = race_df.set_index("馬番")["score"].to_dict() if "score" in race_df.columns else {}
            s_hon = float(scores.get(h1, 0))
            s_tai = float(scores.get(h2, 0))
            s_sab = float(scores.get(h_marks.get("▲", -1), 0))
            s_del = float(scores.get(h_marks.get("△", -1), 0))
            gap_12 = s_hon - s_tai
            gap_top4 = s_hon - s_del if s_del > 0 else s_hon - s_sab

            if gap_12 >= _gap12_hi:
                first  = [h1]
                second = [h_marks[m] for m in ["◯","▲"] if m in h_marks]
                third  = [h_marks[m] for m in ["◯","▲","△","☆","★"] if m in h_marks]
            elif gap_12 <= _gap12_lo and gap_top4 <= _gap_top4:
                first  = [h1, h2]
                second = [h_marks[m] for m in ["◎","◯","▲"] if m in h_marks]
                third  = [h_marks[m] for m in ["◎","◯","▲","△","☆"] if m in h_marks]
            else:
                first  = [h1, h2]
                second = [h_marks[m] for m in ["◎","◯","▲","△"] if m in h_marks]
                third  = [h_marks[m] for m in ["◎","◯","▲","△","☆","★"] if m in h_marks]

            # === 着順モデル補強（高信頼な ◎突出ケースのみ） ===
            order_proba = _predict_order_proba(race_df)
            if order_proba is not None:
                try:
                    pw = order_proba.set_index("馬番")["p_win"].to_dict()
                    pw_hon = float(pw.get(h1, 0))
                    pw_tai = float(pw.get(h2, 0))
                    if pw_hon >= _pw_min and pw_hon >= pw_tai * _pw_ratio:
                        first  = [h1]
                        second = [h_marks[m] for m in ["◯","▲"] if m in h_marks]
                        third  = [h_marks[m] for m in ["◯","▲","△","☆","★"] if m in h_marks]
                except Exception:
                    pass

            fm_combos = set()
            for f in first:
                for s in second:
                    for t in third:
                        if len({f, s, t}) == 3:
                            fm_combos.add((f, s, t))
            if len(fm_combos) > 36:
                third = third[:4]
                fm_combos = set()
                for f in first:
                    for s in second:
                        for t in third:
                            if len({f, s, t}) == 3:
                                fm_combos.add((f, s, t))

            if fm_combos:
                n = len(fm_combos)
                per_bet = max(100, (9600 // n // 100) * 100)
                combo_strs = [f"{f}→{s}→{t}" for f, s, t in sorted(fm_combos)]
                result["HALO_戦略対象"]    = True
                result["HALO_三連単_買い目"] = " / ".join(combo_strs)
                result["HALO_三連単_購入額"] = per_bet * n
                result["HALO_三連単_点数"]   = n

    # ── STANDARD: 単複馬連 EV ベース選別（Stage 1-06: 2026-04-16）─────
    # 旧仕様 (◎-◯固定 1 点 + 単勝20%+複勝60%+馬連20%) は撤廃。
    # 単勝 ◎/◯, 複勝 ◎/◯, 馬連 ◎-{◯,▲,△,☆} の各候補を EV 計算し、
    # 閾値 (utils.MIN_EV_*) を通過したものだけ採用、EV-1 を重み配分。
    if not (tansho_blocked and fuku_blocked and umaren_blocked):
        try:
            from ev_gate import (
                make_race_meta, compute_ev_tansho, compute_ev_fuku,
                compute_ev_umaren, pass_ev_gate,
            )
            try:
                _dist_int = int(float(_dist)) if _dist is not None else 1600
            except Exception:
                _dist_int = 1600
            race_meta = make_race_meta(place, _td, _dist_int, len(race_df))

            tansho_series = pd.to_numeric(race_df.get("単勝", pd.Series(dtype=float)), errors="coerce")
            pop_series = tansho_series.rank(method="min", ascending=True)
            pop_map = {int(r["馬番"]): int(p)
                       for (_, r), p in zip(race_df.iterrows(), pop_series)
                       if pd.notna(r.get("馬番")) and pd.notna(p)}

            order_proba = _predict_order_proba(race_df)
            if order_proba is not None and "馬番" in order_proba.columns:
                p_win_map  = {int(r["馬番"]): float(r["p_win"])
                              for _, r in order_proba.iterrows() if pd.notna(r.get("馬番"))}
                p_p23_map  = {int(r["馬番"]): float(r["p_place23"])
                              for _, r in order_proba.iterrows() if pd.notna(r.get("馬番"))}
                p_fuku_map = {k: min(1.0, p_win_map.get(k, 0) + p_p23_map.get(k, 0)) for k in p_win_map}
            else:
                p_win_map  = {int(r["馬番"]): float(r.get("prob", 0))
                              for _, r in race_df.iterrows() if pd.notna(r.get("馬番"))}
                p_fuku_map = {k: min(1.0, v * 2.5) for k, v in p_win_map.items()}

            cand: list[dict] = []

            if not tansho_blocked:
                for mark in ["◎", "◯"]:
                    if mark in h_marks:
                        h = h_marks[mark]
                        p = p_win_map.get(h, 0.0)
                        ev, pay = compute_ev_tansho(p, pop_map.get(h, 7), race_meta)
                        if pass_ev_gate("単勝", ev):
                            cand.append({"馬券種":"単勝","key":str(h),"ev":ev})

            if not fuku_blocked:
                for mark in ["◎", "◯"]:
                    if mark in h_marks:
                        h = h_marks[mark]
                        p = p_fuku_map.get(h, p_win_map.get(h, 0.0) * 2.5)
                        ev, pay = compute_ev_fuku(p, pop_map.get(h, 7), race_meta)
                        if pass_ev_gate("複勝", ev):
                            cand.append({"馬券種":"複勝","key":str(h),"ev":ev})

            if not umaren_blocked:
                for mark in ["◯", "▲", "△", "☆"]:
                    if mark in h_marks:
                        ho = h_marks[mark]
                        a, b = sorted([h1, ho])
                        p_a = p_fuku_map.get(h1, 0.0)
                        p_b = p_fuku_map.get(ho, 0.0)
                        p_hit = p_a * p_b / 3.0
                        ev, pay = compute_ev_umaren(p_hit, pop_map.get(a, 7), pop_map.get(b, 7), race_meta)
                        if pass_ev_gate("馬連", ev):
                            cand.append({"馬券種":"馬連","key":f"{a}-{b}","ev":ev})

            if cand:
                pos = [max(0.0, c["ev"] - 1.0) for c in cand]
                tot = sum(pos)
                weights = [w / tot for w in pos] if tot > 0 else [1.0/len(cand)] * len(cand)

                # 馬券種ごとに集約
                grouped: dict = {"単勝": [], "複勝": [], "馬連": []}
                for c, w in zip(cand, weights):
                    amt = floor_to_unit(int(budget * w))
                    if amt > 0:
                        grouped[c["馬券種"]].append((c["key"], amt))

                if any(grouped.values()):
                    result["STANDARD_戦略対象"] = True
                    if grouped["単勝"]:
                        result["STANDARD_単勝_買い目"] = " / ".join(k for k,_ in grouped["単勝"])
                        result["STANDARD_単勝_購入額"] = sum(a for _,a in grouped["単勝"])
                    if grouped["複勝"]:
                        result["STANDARD_複勝_買い目"] = " / ".join(k for k,_ in grouped["複勝"])
                        result["STANDARD_複勝_購入額"] = sum(a for _,a in grouped["複勝"])
                    if grouped["馬連"]:
                        result["STANDARD_馬連_買い目"] = " / ".join(k for k,_ in grouped["馬連"])
                        result["STANDARD_馬連_購入額"] = sum(a for _,a in grouped["馬連"])
        except Exception as e:
            logger.warning(f"STANDARD EV 計算失敗 {place} {cls_raw}: {e}")

    # ── TRIPLE: 三連複◎◯▲1点(¥1,000) + 複勝◎(残り) ──────────────
    h3 = h_marks.get("▲")
    if h2 and h3 and not san_blocked and not fuku_blocked and not odds_too_low:
        FIXED_SAN = 1000
        san_key  = "-".join(map(str, sorted([h1, h2, h3])))
        amt_fuku = max(MIN_UNIT, floor_to_unit(budget - FIXED_SAN))
        result["TRIPLE_戦略対象"]      = True
        result["TRIPLE_三連複_買い目"] = san_key
        result["TRIPLE_三連複_購入額"] = FIXED_SAN
        result["TRIPLE_複勝_買い目"]   = str(h1)
        result["TRIPLE_複勝_購入額"]   = amt_fuku

    return result


# =========================================================
# ライブ結果記録
# =========================================================
def _append_live_results(out_df: pd.DataFrame) -> None:
    """予測結果を data/live_results_2026.csv に追記する。

    実績列（着順・払戻）は空欄で記録し、後日レース結果判明後に手動または
    record_live_results.py で埋める。
    重複防止: (日付, 場所, R, 馬番) が既存行と一致する場合はスキップ。
    """
    # 記録対象列（予測列のみ。実績列は空欄で追加）
    pred_cols = [
        "日付", "場所", "R", "クラス", "距離", "レースID",
        "馬番", "馬名", "騎手",
        "スコア", "印",
        "期待値スコア", "EV補正スコア", "単勝オッズ",
        "フィルタ除外", "警告",
        "HAHO_戦略対象", "HAHO_三連複_買い目", "HAHO_三連複_購入額", "HAHO_三連複_点数",
        "HALO_戦略対象", "HALO_三連単_買い目", "HALO_三連単_購入額", "HALO_三連単_点数",
        "STANDARD_戦略対象", "STANDARD_単勝_買い目", "STANDARD_単勝_購入額",
                             "STANDARD_複勝_買い目", "STANDARD_複勝_購入額",
                             "STANDARD_馬連_買い目", "STANDARD_馬連_購入額",
        "TRIPLE_戦略対象", "TRIPLE_三連複_買い目", "TRIPLE_三連複_購入額",
                           "TRIPLE_複勝_買い目",   "TRIPLE_複勝_購入額",
    ]
    # 実際に存在する列のみ抽出
    use_cols = [c for c in pred_cols if c in out_df.columns]
    new_rows = out_df[use_cols].copy()
    # 実績列（空欄）を付加
    for col in ["着順", "単勝払戻", "複勝払戻", "馬連払戻", "三連複払戻", "三連単払戻"]:
        if col not in new_rows.columns:
            new_rows[col] = ""

    key_cols = ["日付", "場所", "R", "馬番"]

    if LIVE_CSV.exists():
        existing = pd.read_csv(LIVE_CSV, encoding="utf-8-sig", dtype=str)
        # 重複チェック
        existing_keys = set(
            zip(existing["日付"].astype(str), existing["場所"].astype(str),
                existing["R"].astype(str),    existing["馬番"].astype(str))
        )
        new_rows["_key"] = list(zip(
            new_rows["日付"].astype(str), new_rows["場所"].astype(str),
            new_rows["R"].astype(str),    new_rows["馬番"].astype(str)
        ))
        before = len(new_rows)
        new_rows = new_rows[~new_rows["_key"].isin(existing_keys)].drop(columns=["_key"])
        skipped = before - len(new_rows)
        if skipped > 0:
            logger.info(f"live_results: {skipped}行は既存レコードのためスキップ")
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows

    combined.to_csv(LIVE_CSV, index=False, encoding="utf-8-sig")
    logger.info(f"live_results_2026.csv 更新: +{len(new_rows)}行 → 累計{len(combined)}行  ({LIVE_CSV})")


# =========================================================
# メイン
# =========================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="週末CSV予想結果出力")
    parser.add_argument("--csv",    required=True, help="入力CSVパス (例: data/weekly/20260301.csv)")
    parser.add_argument("--budget", type=int, default=10000, help="1レース予算（円）")
    parser.add_argument("--out",    default="", help="出力CSVパス（省略時は自動命名）")
    parser.add_argument("--odds",   default="", help="OD CSVパス (例: E:/競馬過去走データ/OD260329.CSV)")
    parser.add_argument("--strategy", default="balanced",
                        choices=["balanced", "high_roi", "volume"],
                        help="Value Model戦略 (default: balanced)")
    parser.add_argument("--plan", default="triple",
                        choices=["triple", "legacy"],
                        help="買い目プラン: triple=三連複+複勝統一戦略, legacy=旧HAHO/HALO/LALO/CQC (default: triple)")
    parser.add_argument("--triple-type", default="safe",
                        choices=["aggressive", "standard", "safe"],
                        help="TRIPLEプランの配分: aggressive=三連複100%%, standard=三連複50%%+複勝50%%, safe=三連複1000円固定+残り複勝 (default: safe)")
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

    # OD CSV (前日オッズ) からの複勝オッズ取り込み
    od_odds = None
    if args.odds:
        try:
            from parse_od_csv import load_od_odds
            od_path = Path(args.odds)
            date_str = csv_path.stem  # "20260329" from filename
            od_odds = load_od_odds(od_path, date=date_str)
            logger.info(f"ODオッズ読込: {len(od_odds)}頭 / {od_odds['race_id'].nunique()}R")
        except Exception as e:
            logger.warning(f"OD CSV読込失敗: {e}")

    # Value Model v2 戦略設定
    value_strat_name = args.strategy
    value_obj = _get_cached(VALUE_MODEL_PATH, "value_model")
    if value_obj and "strategies" in value_obj:
        value_strat = value_obj["strategies"].get(value_strat_name, {})
        value_pr_thr = value_strat.get("pred_roi_thr", 0.88)
        value_cp_thr = value_strat.get("cal_prob_thr", 0.20)
        logger.info(f"Value戦略: {value_strat_name} (pred_roi>={value_pr_thr}, cal_prob>={value_cp_thr})")
    else:
        value_pr_thr = 0.88
        value_cp_thr = 0.20

    # kako5 警告マップ（馬名→警告文）
    first_date = str(df["日付S"].iloc[0]) if "日付S" in df.columns else ""
    kako5_warns = _load_kako5_warnings(first_date)
    if kako5_warns:
        logger.info(f"kako5警告対象馬: {len(kako5_warns)}頭 ({', '.join(kako5_warns.keys())})")

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
            # 期待値スコア = model_prob × 単勝オッズ / 控除率(0.80)
            tansho = pd.to_numeric(race_df.get("単勝", pd.Series(dtype=float)), errors="coerce")
            race_df["ev_score"] = (race_df["prob"] * tansho / RETURN_RATE_TAN).round(3)
            # EV 補正スコア（逆転現象を修正）
            race_df["ev_cal"] = race_df["ev_score"].apply(
                lambda e: round(_ev_calibrator.transform(float(e)), 3)
                if pd.notna(e) else 0.0
            )
        except Exception as e:
            logger.warning(f"予測失敗 {race_id}: {e}")
            race_df["prob"]     = 0.0
            race_df["mark"]     = ""
            race_df["score"]    = 0.0
            race_df["ev_score"] = 0.0
            race_df["ev_cal"]   = 0.0

        # ── Value Model (Layer 2) ─────────────────────────────────
        try:
            race_df["value_score"], race_df["cal_prob"] = compute_value_scores(
                race_df, od_odds=od_odds)
            # VALUE戦略判定: pred_roi >= threshold AND cal_prob >= threshold
            race_df["value_buy"] = (
                (race_df["value_score"] >= value_pr_thr) &
                (race_df["cal_prob"] >= value_cp_thr)
            )
            # 調教フィルタ（任意: データがなくても動作に影響なし）
            try:
                from parse_training import load_training_features
                training_dir = BASE_DIR / "data" / "training"
                if training_dir.exists():
                    t_feats = load_training_features(str(training_dir), week_date=date)
                    for idx, row in race_df.iterrows():
                        horse_name = str(row.get("馬名S", row.get("馬名", "")))
                        if horse_name in t_feats.index:
                            tf = t_feats.loc[horse_name]
                            # 末脚加速良好 → 閾値を0.02緩和して買い判定
                            if pd.notna(tf.get("h_accel")) and tf["h_accel"] < -1.5:
                                if race_df.at[idx, "value_score"] >= value_pr_thr - 0.02:
                                    race_df.at[idx, "value_buy"] = True
                            # 坂路最終1Fが遅すぎ → 買い判定取消
                            if pd.notna(tf.get("h_best_lap1")) and tf["h_best_lap1"] > 17.0:
                                race_df.at[idx, "value_buy"] = False
            except Exception:
                pass  # 調教データなし or エラー時はスキップ
        except Exception as e:
            logger.debug(f"Value Model スキップ {race_id}: {e}")
            race_df["value_score"] = np.nan
            race_df["cal_prob"] = np.nan
            race_df["value_buy"] = False

        # ── 昇級戦判定（◎ベース） ───────────────────────────────
        hon_row      = race_df[race_df["mark"] == "◎"]
        hon_ev       = float(hon_row["ev_score"].iloc[0]) if len(hon_row) > 0 else 0.0
        hon_ev_cal   = float(hon_row["ev_cal"].iloc[0]) if len(hon_row) > 0 else 0.0
        n_horses     = len(race_df)
        baba         = str(meta.get("馬場状態", "")).strip()
        cls_now_raw  = pd.to_numeric(meta.get("クラス区分", None), errors="coerce")
        cls_prev_raw = pd.to_numeric(
            hon_row["前走クラスコード"].iloc[0]
            if len(hon_row) > 0 and "前走クラスコード" in hon_row.columns else None,
            errors="coerce",
        )
        upgrade = is_upgrade_race(cls_now_raw, cls_prev_raw)

        # ── フィルタ適用（HELL セグメント除外 + EV 補正スコア） ───
        base_excluded = place in EXCLUDE_PLACES or cls_raw in EXCLUDE_CLASSES
        filter_result = _bet_filter.check(
            place=place, n_horses=n_horses, baba=baba,
            ev=hon_ev, is_upgrade=upgrade, ev_cal=hon_ev_cal,
        )
        filter_reason = filter_result.reason if filter_result.should_skip else ""

        if base_excluded or filter_result.should_skip:
            bets = get_bets(race_df, "", "", {}, 0)  # empty result with correct keys
            if filter_result.should_skip and not base_excluded:
                logger.info(f"[BetFilter] ケン: {place} {r_num}R → {filter_result.reason}")
        else:
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
                "EV補正スコア":        float(row.get("ev_cal", 0.0)),
                "フィルタ除外":        filter_reason if is_hon else "",
                "警告":                kako5_warns.get(str(row.get("馬名","")), ""),
                "印":                  str(row["mark"]),
                "HAHO_戦略対象":       "✅" if bets["HAHO_戦略対象"] else "",
                "HAHO_三連複_買い目":  bets["HAHO_三連複_買い目"]  if is_hon else "",
                "HAHO_三連複_購入額":  bets["HAHO_三連複_購入額"]  if is_hon else "",
                "HAHO_三連複_点数":    bets["HAHO_三連複_点数"]    if is_hon else "",
                "HALO_戦略対象":       "✅" if bets["HALO_戦略対象"] else "",
                "HALO_三連単_買い目":  bets["HALO_三連単_買い目"]  if is_hon else "",
                "HALO_三連単_購入額":  bets["HALO_三連単_購入額"]  if is_hon else "",
                "HALO_三連単_点数":    bets["HALO_三連単_点数"]    if is_hon else "",
                "STANDARD_戦略対象":   "✅" if bets["STANDARD_戦略対象"] else "",
                "STANDARD_単勝_買い目":bets["STANDARD_単勝_買い目"] if is_hon else "",
                "STANDARD_単勝_購入額":bets["STANDARD_単勝_購入額"] if is_hon else "",
                "STANDARD_複勝_買い目":bets["STANDARD_複勝_買い目"] if is_hon else "",
                "STANDARD_複勝_購入額":bets["STANDARD_複勝_購入額"] if is_hon else "",
                "STANDARD_馬連_買い目":bets["STANDARD_馬連_買い目"] if is_hon else "",
                "STANDARD_馬連_購入額":bets["STANDARD_馬連_購入額"] if is_hon else "",
                "ValueScore":          round(float(row.get("value_score", 0.0)), 3)
                                       if pd.notna(row.get("value_score")) else "",
                "CalProb":             round(float(row.get("cal_prob", 0.0)), 3)
                                       if pd.notna(row.get("cal_prob")) else "",
                "VALUE_買い":          "✅" if row.get("value_buy", False) else "",
                "TRIPLE_戦略対象":     "✅" if bets["TRIPLE_戦略対象"] else "",
                "TRIPLE_三連複_買い目": bets["TRIPLE_三連複_買い目"] if is_hon else "",
                "TRIPLE_三連複_購入額": bets["TRIPLE_三連複_購入額"] if is_hon else "",
                "TRIPLE_複勝_買い目":   bets["TRIPLE_複勝_買い目"]   if is_hon else "",
                "TRIPLE_複勝_購入額":   bets["TRIPLE_複勝_購入額"]   if is_hon else "",
            })

    out_df = pd.DataFrame(rows)

    # 出力
    out_path = Path(args.out) if args.out else BASE_DIR / "reports" / f"pred_{csv_path.stem}.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"出力完了: {out_path}")

    # ライブ結果記録（予測列を追記、実績列は後日埋める）
    _append_live_results(out_df)

    # サマリ表示
    def _count_plan(col: str) -> int:
        return out_df[out_df[col]=="✅"]["レースID"].nunique() if col in out_df.columns else 0
    print(f"\n{'='*50}")
    print(f"予想完了: {out_df['レースID'].nunique()}レース / {len(out_df)}頭")
    print(f"TRIPLE: {_count_plan('TRIPLE_戦略対象')}R  "
          f"HAHO: {_count_plan('HAHO_戦略対象')}R  "
          f"HALO: {_count_plan('HALO_戦略対象')}R  "
          f"STANDARD: {_count_plan('STANDARD_戦略対象')}R")
    # Value Model サマリ
    if "VALUE_買い" in out_df.columns:
        value_horses = (out_df["VALUE_買い"] == "✅").sum()
        value_races = out_df.loc[out_df["VALUE_買い"] == "✅", "レースID"].nunique()
        print(f"VALUE戦略({value_strat_name}): {value_horses}頭 / {value_races}R "
              f"(pred_roi>={value_pr_thr}, cal_prob>={value_cp_thr})")
    print(f"出力先:   {out_path}")
    print(f"{'='*50}")

    # 戦略対象レースの買い目サマリ
    hon_rows = out_df[out_df["印"]=="◎"].copy()
    if not hon_rows.empty:
        # TRIPLE買い目（デフォルト表示）
        if args.plan == "triple" and "TRIPLE_戦略対象" in hon_rows.columns:
            triple_disp = hon_rows[hon_rows["TRIPLE_戦略対象"]=="✅"][[
                "日付","場所","R","クラス","距離","馬名",
                "TRIPLE_三連複_買い目","TRIPLE_三連複_購入額",
                "TRIPLE_複勝_買い目","TRIPLE_複勝_購入額",
            ]]
            if not triple_disp.empty:
                print(f"\n【TRIPLE 買い目一覧（{args.triple_type}: 三連複◎◯▲1点 + 複勝◎1点）】")
                print(triple_disp.to_string(index=False))

        if "HAHO_戦略対象" in hon_rows.columns:
            haho_disp = hon_rows[hon_rows["HAHO_戦略対象"]=="✅"][[
                "日付","場所","R","クラス","距離","馬名",
                "HAHO_三連複_買い目","HAHO_三連複_購入額",
            ]]
            if not haho_disp.empty:
                print("\n【HAHO 買い目一覧（三連複◎1頭軸-5頭流し）】")
                print(haho_disp.to_string(index=False))
        if "HALO_戦略対象" in hon_rows.columns:
            halo_disp = hon_rows[hon_rows["HALO_戦略対象"]=="✅"][[
                "日付","場所","R","クラス","距離","馬名",
                "HALO_三連単_買い目","HALO_三連単_購入額",
            ]]
            if not halo_disp.empty:
                print("\n【HALO 買い目一覧（三連単フォーメーション）】")
                print(halo_disp.to_string(index=False))
        if "LALO_戦略対象" in hon_rows.columns:
            lalo_disp = hon_rows[hon_rows["LALO_戦略対象"]=="✅"][[
                "日付","場所","R","クラス","距離","馬名",
                "LALO_複勝_買い目","LALO_複勝_購入額",
            ]]
            if not lalo_disp.empty:
                print("\n【LALO 買い目一覧（複勝◎1点のみ）】")
                print(lalo_disp.to_string(index=False))
        if "CQC_戦略対象" in hon_rows.columns:
            cqc_disp = hon_rows[hon_rows["CQC_戦略対象"]=="✅"][[
                "日付","場所","R","クラス","距離","馬名",
                "CQC_単勝_買い目","CQC_単勝_購入額",
            ]]
            if not cqc_disp.empty:
                print("\n【CQC 買い目一覧（単勝◎1点のみ）】")
                print(cqc_disp.to_string(index=False))

        # ★EV推奨レース（CQC対象 かつ 期待値スコア >= 1.5）
        if "期待値スコア" in hon_rows.columns:
            ev_col = "期待値スコア"
        elif "乖離スコア" in hon_rows.columns:
            ev_col = "乖離スコア"
        else:
            ev_col = None
        if ev_col and "CQC_戦略対象" in hon_rows.columns:
            ev_cands = hon_rows[
                (hon_rows["CQC_戦略対象"] == "✅") &
                (pd.to_numeric(hon_rows[ev_col], errors="coerce") >= 1.5)
            ].copy()
            ev_cands = ev_cands.sort_values(ev_col, ascending=False)
            disp_cols = [c for c in ["日付","場所","R","クラス","距離","馬名","単勝オッズ",ev_col] if c in ev_cands.columns]
            if not ev_cands.empty:
                print(f"\n【★EV推奨（単勝◎, EV>=1.5, {len(ev_cands)}R）】")
                print(ev_cands[disp_cols].to_string(index=False))
            else:
                print("\n【★EV推奨】該当なし（EV>=1.5 のCQC対象レースなし）")


if __name__ == "__main__":
    main()