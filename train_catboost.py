"""
train_catboost.py
PyCaLiAI - CatBoost 複勝内2値分類モデルの学習・評価

Usage:
    python train_catboost.py
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    plt.rcParams["font.family"] = "MS Gothic"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# =========================================================
# パス設定
# =========================================================
BASE_DIR   = Path(r"E:\PyCaLiAI")
DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MASTER_CSV  = DATA_DIR / "master_kako5.csv"  # kako5特徴量付き
MASTER_CSV_ORIG = DATA_DIR / "master_20130105-20251228.csv"  # フォールバック
MODEL_PATH  = MODEL_DIR / "catboost_fukusho_v1.pkl"

TARGET       = "fukusho_flag"
RANDOM_STATE = 42

# =========================================================
# 特徴量定義
# =========================================================

# CatBoostにそのまま渡すカテゴリ列（LabelEncoding不要）
CAT_FEATURES = [
    # 血統
    "種牡馬", "父タイプ名", "母父馬", "母父タイプ名", "毛色",
    # 人的要素
    "馬主(最新/仮想)", "生産者",
    # レース条件
    "芝・ダ", "コース区分", "芝(内・外)", "馬場状態", "天気",
    "クラス名", "場所",
    # 馬プロフィール
    "性別", "斤量", "ブリンカー", "重量種別",
    "年齢限定", "限定", "性別限定", "指定条件",
    # 前走条件
    "前走場所", "前芝・ダ", "前走馬場状態", "前走斤量", "前好走",
]

# 数値列
NUM_FEATURES = [
    # レース条件
    "距離", "トラックコード(JV)", "出走頭数", "フルゲート頭数",
    # 騎手・調教師（コードは数値として使う）
    "騎手コード", "調教師コード", "騎手年齢", "調教師年齢",
    # 馬プロフィール
    "枠番", "馬番", "年齢", "馬齢斤量差", "斤量体重比",
    "間隔", "休み明け～戦目",
    # 前走成績
    "前走確定着順", "前距離", "前走出走頭数", "前走競走種別",
    "前走トラックコード(JV)", "前走馬体重", "前走馬体重増減",
    # 前走コーナー通過
    "前1角", "前2角", "前3角", "前4角",
    # 前走タイム・ペース
    "前走上り3F", "前走上り3F順",
    "前走Ave-3F", "前PCI", "前走PCI3", "前走RPCI",
    "前走平均1Fタイム",
    # 騎手・調教師直近複勝率
    "jockey_fuku30", "jockey_fuku90",
    "trainer_fuku30", "trainer_fuku90",
    # 馬の直近フォーム指数
    "horse_fuku10", "horse_fuku30",
    # 派生指標（build_dataset / optuna_lgbm 由来）
    "prev_pos_rel", "closing_power",
    # 補正タイム（data/hosei）
    "前走補9", "前走補正",
    # 調教データ（坂路・WC）
    "trn_hanro_4f", "trn_hanro_3f", "trn_hanro_2f", "trn_hanro_1f",
    "trn_hanro_lap1", "trn_hanro_lap2", "trn_hanro_lap3", "trn_hanro_lap4",
    "trn_hanro_days",
    "trn_wc_5f", "trn_wc_4f", "trn_wc_3f",
    "trn_wc_lap1", "trn_wc_lap2", "trn_wc_lap3",
    "trn_wc_days",
    # 前走オッズ
    "前走単勝オッズ",
    # 過去5走集約特徴量（parse_kako5.py で生成）
    "kako5_avg_pos", "kako5_std_pos", "kako5_best_pos",
    "kako5_avg_agari3f", "kako5_best_agari3f",
    "kako5_same_td_ratio", "kako5_same_dist_ratio", "kako5_same_place_ratio",
    "kako5_pos_trend", "kako5_race_count",
    # 好走3分類 + 同条件ベスト（Phase 5）
    "kako5_expected_good_count", "kako5_upset_good_count", "kako5_hidden_good_count",
    "kako5_same_cond_best_pos",
    # 全キャリア同条件ベスト（build_from_master で生成）
    "hist_same_cond_best_pos", "hist_same_cond_top3_rate", "hist_same_cond_count",
    "hist_same_place_best_pos",
]

# タイム文字列列（数値変換が必要）
TIME_STR_FEATURES = ["前走走破タイム", "前走着差タイム"]

ALL_FEATURES = CAT_FEATURES + NUM_FEATURES + TIME_STR_FEATURES

# =========================================================
# CatBoostパラメータ
# =========================================================
CAT_PARAMS = {
    "iterations":            2000,
    "learning_rate":         0.05,
    "depth":                 6,
    "eval_metric":           "AUC",
    "random_seed":           RANDOM_STATE,
    "verbose":               200,
    "early_stopping_rounds": 50,
    "task_type":             "CPU",   # GPUがあれば"GPU"に変更可
    "auto_class_weights":    "Balanced",  # クラス不均衡対応
}


# =========================================================
# 前処理
# =========================================================
from utils import parse_time_str


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """CatBoost用前処理。カテゴリ列はstr・欠損は'__NaN__'に統一。"""
    df = df.copy()

    # タイム文字列 → 数値
    for col in TIME_STR_FEATURES:
        if col in df.columns:
            df[col] = parse_time_str(df[col])

    # カテゴリ列：欠損を '__NaN__' で埋めてstr型に統一
    for col in CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("__NaN__").astype(str)
        else:
            df[col] = "__NaN__"

    # 数値列：型をfloat64に統一
    for col in NUM_FEATURES + TIME_STR_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# =========================================================
# データロード・分割
# =========================================================
def load_and_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    csv_path = MASTER_CSV if MASTER_CSV.exists() else MASTER_CSV_ORIG
    logger.info(f"マスターCSV読み込み: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    logger.info(f"  {len(df):,}行 × {len(df.columns)}列")

    train = df[df["split"] == "train"].copy()
    valid = df[df["split"] == "valid"].copy()
    test  = df[df["split"] == "test"].copy()

    logger.info(
        f"分割: train={len(train):,} / valid={len(valid):,} / test={len(test):,}"
    )
    return train, valid, test


# =========================================================
# 学習
# =========================================================
def train_model(
    train: pd.DataFrame,
    valid: pd.DataFrame,
) -> tuple[CatBoostClassifier, list[str]]:

    train = preprocess(train)
    valid = preprocess(valid)

    # 使用可能な特徴量だけ選択
    feature_cols = [c for c in ALL_FEATURES if c in train.columns]
    missing = [c for c in ALL_FEATURES if c not in train.columns]
    if missing:
        logger.warning(f"特徴量が存在しないためスキップ: {missing}")
    logger.info(f"使用特徴量数: {len(feature_cols)}")

    # カテゴリ列のインデックス（feature_cols内での位置）
    cat_feature_indices = [
        i for i, c in enumerate(feature_cols) if c in CAT_FEATURES
    ]
    logger.info(f"カテゴリ特徴量数: {len(cat_feature_indices)}")

    X_tr, y_tr = train[feature_cols], train[TARGET].astype(int)
    X_va, y_va = valid[feature_cols], valid[TARGET].astype(int)

    # CatBoost Pool（カテゴリ列を明示）
    pool_tr = Pool(X_tr, y_tr, cat_features=cat_feature_indices)
    pool_va = Pool(X_va, y_va, cat_features=cat_feature_indices)

    model = CatBoostClassifier(**CAT_PARAMS)
    model.fit(pool_tr, eval_set=pool_va, use_best_model=True)

    logger.info(f"学習完了: best_iteration={model.best_iteration_}")
    return model, feature_cols


# =========================================================
# 評価
# =========================================================
def evaluate(
    model: CatBoostClassifier,
    df: pd.DataFrame,
    feature_cols: list[str],
    split_name: str,
) -> dict[str, float]:

    df = preprocess(df)

    # カテゴリ列のインデックス
    cat_feature_indices = [
        i for i, c in enumerate(feature_cols) if c in CAT_FEATURES
    ]

    X = df[feature_cols]
    y = df[TARGET].astype(int)

    pool = Pool(X, cat_features=cat_feature_indices)
    proba = model.predict_proba(pool)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    auc    = roc_auc_score(y, proba)
    pr_auc = average_precision_score(y, proba)

    logger.info(f"[{split_name}] AUC={auc:.4f}  PR-AUC={pr_auc:.4f}")
    print(f"\n=== {split_name} 分類レポート ===")
    print(classification_report(y, pred, target_names=["圏外", "複勝内"]))

    # ROC曲線
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y, proba, ax=ax, name=split_name)
    ax.set_title(f"ROC曲線 [{split_name}] CatBoost")
    path = REPORT_DIR / f"catboost_roc_{split_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"ROC曲線保存: {path}")

    return {"auc": auc, "pr_auc": pr_auc}


# =========================================================
# 特徴量重要度
# =========================================================
def plot_importance(
    model: CatBoostClassifier,
    feature_cols: list[str],
) -> None:
    importance = model.get_feature_importance()
    feat_imp = pd.Series(importance, index=feature_cols).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    feat_imp.tail(25).plot(kind="barh", ax=ax)
    ax.set_title("CatBoost 特徴量重要度 TOP25")
    ax.set_xlabel("Importance")
    path = REPORT_DIR / "catboost_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"特徴量重要度保存: {path}")


# =========================================================
# モデル保存
# =========================================================
def save_model(
    model: CatBoostClassifier,
    feature_cols: list[str],
) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "feature_cols": feature_cols},
        MODEL_PATH,
    )
    logger.info(f"モデル保存: {MODEL_PATH}")


# =========================================================
# main
# =========================================================
def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train, valid, test = load_and_split()
    model, feature_cols = train_model(train, valid)

    metrics_valid = evaluate(model, valid, feature_cols, "Valid")
    metrics_test  = evaluate(model, test,  feature_cols, "Test")

    plot_importance(model, feature_cols)
    save_model(model, feature_cols)

    print("\n" + "=" * 50)
    print("CatBoost 学習完了サマリ")
    print("=" * 50)
    print(f"Valid AUC    : {metrics_valid['auc']:.4f}")
    print(f"Test  AUC    : {metrics_test['auc']:.4f}")
    print(f"Valid PR-AUC : {metrics_valid['pr_auc']:.4f}")
    print(f"Test  PR-AUC : {metrics_test['pr_auc']:.4f}")
    print(f"特徴量数      : {len(feature_cols)}")
    print(f"モデル保存先  : {MODEL_PATH}")


if __name__ == "__main__":
    main()