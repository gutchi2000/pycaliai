"""
train_lgbm.py
PyCaLiAI - LightGBM 複勝内2値分類モデルの学習・評価

Usage:
    python train_lgbm.py
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import (
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning)

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

MASTER_CSV  = DATA_DIR / "master_kako5.csv"  # kako5特徴量付き (parse_kako5.py --mode master で生成)
MASTER_CSV_ORIG = DATA_DIR / "master_20130105-20251228.csv"  # kako5なし（フォールバック）
MODEL_PATH  = MODEL_DIR / "lgbm_fukusho_v1.pkl"

TARGET      = "fukusho_flag"
COL_DATE    = "日付"
COL_RACE_ID = "レースID(新)"
RANDOM_STATE = 42

# =========================================================
# 特徴量定義
# =========================================================

# カテゴリ列（LabelEncoding対象）
CAT_FEATURES = [
    "芝・ダ", "コース区分", "芝(内・外)", "馬場状態", "天気",
    "クラス名", "場所",
    "前走場所", "前芝・ダ", "前走馬場状態", "前走斤量",
    "前好走", "性別", "斤量", "ブリンカー",
]

# 数値列
NUM_FEATURES = [
    # レース条件
    "距離", "トラックコード(JV)", "出走頭数", "フルゲート頭数",
    # 馬プロフィール
    "枠番", "馬番", "年齢", "馬齢斤量差", "斤量体重比",
    "間隔", "休み明け～戦目", "騎手年齢", "調教師年齢",
    # 前走成績
    "前走確定着順", "前距離", "前走出走頭数", "前走競走種別",
    "前走トラックコード(JV)", "前走馬体重", "前走馬体重増減",
    # 前走タイム・ペース
    "前1角", "前2角", "前3角", "前4角",
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
]

# 前走走破タイム・着差タイム（文字列→数値変換が必要）
TIME_STR_FEATURES = ["前走走破タイム", "前走着差タイム"]

ALL_FEATURES = CAT_FEATURES + NUM_FEATURES + TIME_STR_FEATURES

# =========================================================
# LightGBMパラメータ
# =========================================================
LGBM_PARAMS = {
    "objective":          "binary",
    "metric":             "auc",
    "learning_rate":      0.05,
    "num_leaves":         63,
    "min_child_samples":  50,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "random_state":       RANDOM_STATE,
    "n_estimators":       2000,
    "verbose":            -1,
}


# =========================================================
# 前処理
# =========================================================
from utils import parse_time_str


def encode_categoricals(
    df: pd.DataFrame,
    cat_cols: list[str],
    encoders: dict[str, LabelEncoder] | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    カテゴリ列をLabelEncodingする。
    fit=True  → encoderを新規fitして返す（学習時）
    fit=False → 渡されたencoderでtransformのみ（推論時）
    未知ラベルは -1 に変換する。
    """
    df = df.copy()
    if encoders is None:
        encoders = {}

    for col in cat_cols:
        if col not in df.columns:
            df[col] = np.nan
            continue

        df[col] = df[col].astype(str).fillna("__NaN__")

        if fit:
            le = LabelEncoder()
            le.fit(df[col])
            encoders[col] = le
        else:
            le = encoders[col]
            # 未知ラベルを既知クラスに含める
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else "__unknown__")
            if "__unknown__" not in le.classes_:
                le.classes_ = np.append(le.classes_, "__unknown__")

        df[col] = le.transform(df[col])

    return df, encoders


def preprocess(
    df: pd.DataFrame,
    encoders: dict[str, LabelEncoder] | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """特徴量の前処理を一括で行う。"""
    df = df.copy()

    # タイム文字列 → 数値
    for col in TIME_STR_FEATURES:
        if col in df.columns:
            df[col] = parse_time_str(df[col])

    # カテゴリ → LabelEncoding
    df, encoders = encode_categoricals(df, CAT_FEATURES, encoders, fit=fit)

    return df, encoders


# =========================================================
# データロード・分割
# =========================================================
def load_and_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """マスターCSVを読み込んでtrain/valid/testに分割する。"""
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
) -> tuple[LGBMClassifier, dict[str, LabelEncoder], list[str]]:
    """LightGBMモデルを学習して返す。"""

    # 前処理（fit=Trueでencoderを学習）
    train, encoders = preprocess(train, fit=True)
    valid, _        = preprocess(valid, encoders=encoders, fit=False)

    # 使用可能な特徴量だけ選択
    feature_cols = [c for c in ALL_FEATURES if c in train.columns]
    missing = [c for c in ALL_FEATURES if c not in train.columns]
    if missing:
        logger.warning(f"特徴量が存在しないためスキップ: {missing}")
    logger.info(f"使用特徴量数: {len(feature_cols)}")

    X_tr, y_tr = train[feature_cols], train[TARGET]
    X_va, y_va = valid[feature_cols], valid[TARGET]

    # クラス不均衡対応（正例率0.215 → scale_pos_weight）
    neg = (y_tr == 0).sum()
    pos = (y_tr == 1).sum()
    spw = neg / pos
    logger.info(f"scale_pos_weight: {spw:.3f} (neg={neg:,} / pos={pos:,})")

    params = {**LGBM_PARAMS, "scale_pos_weight": spw}
    model = LGBMClassifier(**params)

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[
            early_stopping(stopping_rounds=50, verbose=False),
            log_evaluation(period=200),
        ],
    )
    logger.info(f"学習完了: best_iteration={model.best_iteration_}")

    return model, encoders, feature_cols


# =========================================================
# 評価
# =========================================================
def evaluate(
    model: LGBMClassifier,
    df: pd.DataFrame,
    encoders: dict[str, LabelEncoder],
    feature_cols: list[str],
    split_name: str,
) -> dict[str, float]:
    """AUC・PR-AUC・分類レポートを出力する。"""
    df, _ = preprocess(df, encoders=encoders, fit=False)

    X = df[feature_cols]
    y = df[TARGET]
    proba = model.predict_proba(X)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    auc    = roc_auc_score(y, proba)
    pr_auc = average_precision_score(y, proba)

    logger.info(f"[{split_name}] AUC={auc:.4f}  PR-AUC={pr_auc:.4f}")
    print(f"\n=== {split_name} 分類レポート ===")
    print(classification_report(y, pred, target_names=["圏外", "複勝内"]))

    # ROC曲線
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y, proba, ax=ax, name=split_name)
    ax.set_title(f"ROC曲線 [{split_name}]")
    path = REPORT_DIR / f"lgbm_roc_{split_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"ROC曲線保存: {path}")

    return {"auc": auc, "pr_auc": pr_auc}


# =========================================================
# 特徴量重要度（SHAP）
# =========================================================
def plot_importance(
    model: LGBMClassifier,
    df: pd.DataFrame,
    encoders: dict[str, LabelEncoder],
    feature_cols: list[str],
    n_sample: int = 3000,
) -> None:
    """SHAP beeswarm plotを保存する。"""
    df, _ = preprocess(df, encoders=encoders, fit=False)
    sample = df[feature_cols].sample(
        min(n_sample, len(df)), random_state=RANDOM_STATE
    )

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    fig, _ = plt.subplots(figsize=(10, 8))
    shap.summary_plot(sv, sample, show=False)
    path = REPORT_DIR / "lgbm_shap_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"SHAP保存: {path}")


# =========================================================
# モデル保存
# =========================================================
def save_model(
    model: LGBMClassifier,
    encoders: dict[str, LabelEncoder],
    feature_cols: list[str],
) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model":        model,
            "encoders":     encoders,
            "feature_cols": feature_cols,
        },
        MODEL_PATH,
    )
    logger.info(f"モデル保存: {MODEL_PATH}")


# =========================================================
# main
# =========================================================
def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # データ準備
    train, valid, test = load_and_split()

    # 学習
    model, encoders, feature_cols = train_model(train, valid)

    # 評価
    metrics_valid = evaluate(model, valid, encoders, feature_cols, "Valid")
    metrics_test  = evaluate(model, test,  encoders, feature_cols, "Test")

    # SHAP
    plot_importance(model, test, encoders, feature_cols)

    # 保存
    save_model(model, encoders, feature_cols)

    # サマリ
    print("\n" + "=" * 50)
    print("LightGBM 学習完了サマリ")
    print("=" * 50)
    print(f"Valid AUC    : {metrics_valid['auc']:.4f}")
    print(f"Test  AUC    : {metrics_test['auc']:.4f}")
    print(f"Valid PR-AUC : {metrics_valid['pr_auc']:.4f}")
    print(f"Test  PR-AUC : {metrics_test['pr_auc']:.4f}")
    print(f"特徴量数      : {len(feature_cols)}")
    print(f"モデル保存先  : {MODEL_PATH}")


if __name__ == "__main__":
    main()