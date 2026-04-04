"""
train_regression.py
PyCaLiAI - LightGBM 着順回帰モデル

着順を連続値として直接予測する。
二値分類では区別できない「2着 vs 8着」の差を学習可能。
Huber損失でロバスト回帰（大差の外れ値に頑健）。

Usage:
    python train_regression.py
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error
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

MASTER_CSV  = DATA_DIR / "master_kako5.csv"
MASTER_CSV_ORIG = DATA_DIR / "master_20130105-20251228.csv"
MODEL_PATH  = MODEL_DIR / "lgbm_regression_v1.pkl"

COL_POS     = "着順"
COL_DATE    = "日付"
COL_RACE_ID = "レースID(新/馬番無)"  # レースレベルID（馬番なし16桁）
RANDOM_STATE = 42

# =========================================================
# 特徴量定義（train_lgbm.pyと共通）
# =========================================================
from train_lgbm import CAT_FEATURES, NUM_FEATURES, TIME_STR_FEATURES, ALL_FEATURES
from train_lgbm import preprocess

# =========================================================
# LightGBM回帰パラメータ
# =========================================================
REG_PARAMS = {
    "objective":          "huber",        # ロバスト回帰（外れ値に頑健）
    "metric":             "mae",
    "huber_delta":        3.0,            # 着順3以上の誤差は外れ値扱い
    "learning_rate":      0.05,
    "num_leaves":         63,
    "min_child_samples":  50,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "random_state":       RANDOM_STATE,
    "n_estimators":       4000,
    "verbose":            -1,
}


# =========================================================
# データロード・分割
# =========================================================
def load_and_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """マスターCSVを読み込んでtrain/valid/testに分割。"""
    csv_path = MASTER_CSV if MASTER_CSV.exists() else MASTER_CSV_ORIG
    logger.info(f"マスターCSV読み込み: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    logger.info(f"  {len(df):,}行 × {len(df.columns)}列")

    # 着順がNaNのレコードを除外
    df[COL_POS] = pd.to_numeric(df[COL_POS], errors="coerce")
    before = len(df)
    df = df.dropna(subset=[COL_POS])
    if before > len(df):
        logger.info(f"  着順NaN除外: {before - len(df):,}行")

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
) -> tuple[LGBMRegressor, dict[str, LabelEncoder], list[str]]:
    """LightGBM回帰モデルを学習して返す。"""

    # 目的変数：着順（そのまま連続値として使用）
    y_tr = train[COL_POS].values.astype(np.float32)
    y_va = valid[COL_POS].values.astype(np.float32)

    logger.info(f"着順分布 (train): mean={y_tr.mean():.2f}, "
                f"median={np.median(y_tr):.1f}, "
                f"min={y_tr.min():.0f}, max={y_tr.max():.0f}")

    # 前処理
    train, encoders = preprocess(train, fit=True)
    valid, _        = preprocess(valid, encoders=encoders, fit=False)

    # 特徴量選択
    feature_cols = [c for c in ALL_FEATURES if c in train.columns]
    missing = [c for c in ALL_FEATURES if c not in train.columns]
    if missing:
        logger.warning(f"特徴量が存在しないためスキップ: {missing}")
    logger.info(f"使用特徴量数: {len(feature_cols)}")

    X_tr = train[feature_cols]
    X_va = valid[feature_cols]

    # 回帰学習
    model = LGBMRegressor(**REG_PARAMS)

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
def evaluate_regression(
    model: LGBMRegressor,
    df: pd.DataFrame,
    encoders: dict[str, LabelEncoder],
    feature_cols: list[str],
    split_name: str,
) -> dict[str, float]:
    """MAE, RMSE, Top3精度を評価。"""
    y_true = df[COL_POS].values.astype(np.float32)
    race_ids = df[COL_RACE_ID].values

    # 前処理
    df_proc, _ = preprocess(df, encoders=encoders, fit=False)
    X = df_proc[feature_cols]

    # 予測（着順が小さいほど上位）
    y_pred = model.predict(X)

    # 全体MAE, RMSE
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Top3判定精度（予測着順 <= 3.5 のとき複勝圏内と判定）
    pred_top3 = y_pred <= 3.5
    true_top3 = y_true <= 3
    precision = np.sum(pred_top3 & true_top3) / max(np.sum(pred_top3), 1)
    recall    = np.sum(pred_top3 & true_top3) / max(np.sum(true_top3), 1)

    # BinaryAUC（-y_predをスコアとして使用、着順が小さい＝スコア高い）
    from sklearn.metrics import roc_auc_score
    if "fukusho_flag" in df.columns:
        y_fuku = df["fukusho_flag"].values
        valid_mask = ~np.isnan(y_fuku)
        if valid_mask.sum() > 0:
            auc = roc_auc_score(y_fuku[valid_mask], -y_pred[valid_mask])
        else:
            auc = 0.0
    else:
        auc = 0.0

    # レース単位Top3Hit（予測着順が最も小さい3頭に1着馬が含まれるか）
    top3_hit = 0
    total_races = 0
    unique_races = np.unique(race_ids)
    for race_id in unique_races:
        mask = race_ids == race_id
        y_t = y_true[mask]
        y_p = y_pred[mask]
        if len(y_t) < 3:
            continue
        top3_idx = np.argsort(y_p)[:3]  # 予測着順が小さい順（上位3頭）
        if np.any(y_t[top3_idx] == 1):
            top3_hit += 1
        total_races += 1
    top3_rate = top3_hit / total_races if total_races > 0 else 0.0

    logger.info(
        f"[{split_name}] MAE={mae:.3f}  RMSE={rmse:.3f}  "
        f"Top3Prec={precision:.4f}  Top3Recall={recall:.4f}  "
        f"Top3Hit={top3_rate:.4f}  BinaryAUC={auc:.4f}"
    )
    return {
        "mae": mae,
        "rmse": rmse,
        "top3_precision": precision,
        "top3_recall": recall,
        "top3_hit_rate": top3_rate,
        "binary_auc": auc,
    }


# =========================================================
# 予測値分布プロット
# =========================================================
def plot_prediction_dist(
    model: LGBMRegressor,
    df: pd.DataFrame,
    encoders: dict[str, LabelEncoder],
    feature_cols: list[str],
) -> None:
    """予測着順と実着順の分布を比較プロット。"""
    y_true = df[COL_POS].values.astype(np.float32)
    df_proc, _ = preprocess(df, encoders=encoders, fit=False)
    y_pred = model.predict(df_proc[feature_cols])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 散布図
    axes[0].scatter(y_true, y_pred, alpha=0.01, s=1)
    axes[0].plot([0, 18], [0, 18], "r--", lw=1)
    axes[0].set_xlabel("実着順")
    axes[0].set_ylabel("予測着順")
    axes[0].set_title("実着順 vs 予測着順")
    axes[0].set_xlim(0, 19)
    axes[0].set_ylim(0, 19)

    # ヒストグラム
    axes[1].hist(y_pred, bins=50, alpha=0.7, label="予測")
    axes[1].hist(y_true, bins=18, alpha=0.3, label="実際")
    axes[1].set_xlabel("着順")
    axes[1].set_title("着順分布")
    axes[1].legend()

    path = REPORT_DIR / "regression_pred_dist.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"分布プロット保存: {path}")


# =========================================================
# 特徴量重要度
# =========================================================
def plot_importance(
    model: LGBMRegressor,
    feature_cols: list[str],
    top_n: int = 30,
) -> None:
    imp = model.feature_importances_
    idx = np.argsort(imp)[-top_n:]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(idx)), imp[idx], align="center")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_cols[i] for i in idx])
    ax.set_title("Regression Feature Importance (top 30)")
    ax.set_xlabel("Importance")

    path = REPORT_DIR / "regression_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"重要度プロット保存: {path}")


# =========================================================
# モデル保存
# =========================================================
def save_model(
    model: LGBMRegressor,
    encoders: dict[str, LabelEncoder],
    feature_cols: list[str],
) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model":        model,
            "encoders":     encoders,
            "feature_cols": feature_cols,
            "model_type":   "regression",
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
    metrics_valid = evaluate_regression(model, valid, encoders, feature_cols, "Valid")
    metrics_test  = evaluate_regression(model, test,  encoders, feature_cols, "Test")

    # プロット
    plot_prediction_dist(model, test, encoders, feature_cols)
    plot_importance(model, feature_cols)

    # 保存
    save_model(model, encoders, feature_cols)

    # サマリ
    print("\n" + "=" * 50)
    print("LightGBM 着順回帰 学習完了サマリ")
    print("=" * 50)
    print(f"Valid MAE        : {metrics_valid['mae']:.3f}")
    print(f"Test  MAE        : {metrics_test['mae']:.3f}")
    print(f"Valid RMSE       : {metrics_valid['rmse']:.3f}")
    print(f"Test  RMSE       : {metrics_test['rmse']:.3f}")
    print(f"Valid BinaryAUC  : {metrics_valid['binary_auc']:.4f}")
    print(f"Test  BinaryAUC  : {metrics_test['binary_auc']:.4f}")
    print(f"Valid Top3Hit    : {metrics_valid['top3_hit_rate']:.4f}")
    print(f"Test  Top3Hit    : {metrics_test['top3_hit_rate']:.4f}")
    print(f"特徴量数          : {len(feature_cols)}")
    print(f"モデル保存先      : {MODEL_PATH}")


if __name__ == "__main__":
    main()
