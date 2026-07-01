"""
optuna_lgbm_roi.py
PyCaLiAI - LightGBM Tweedie回帰（roi_target直接予測）実験

roi_target = 複勝配当（3着以内なら実際の配当倍率、圏外は0）を直接予測する。
binary分類（fukusho_flag）との比較実験用。

Usage:
    python optuna_lgbm_roi.py
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import joblib
import optuna
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# =========================================================
# パス設定（optuna_lgbm.py と共有）
# =========================================================
BASE_DIR   = Path(r"E:\PyCaLiAI")
DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MASTER_CSV    = DATA_DIR / "master_20130105-20251228.csv"
HOSEI_DIR     = DATA_DIR / "hosei"
KEKKA_CSV     = DATA_DIR / "kekka_20130105-20251228.csv"
HANRO_MASTER  = Path(r"E:\競馬過去走データ\H-20150401-20260313.csv")
WC_MASTER     = Path(r"E:\競馬過去走データ\W-20150401-20260313.csv")
MODEL_PATH    = MODEL_DIR / "lgbm_roi_v1.pkl"
STUDY_PATH    = REPORT_DIR / "optuna_lgbm_roi_study.pkl"

# 評価用（AUCはfukusho_flagで計算）
TARGET_TRAIN = "roi_target"
TARGET_EVAL  = "fukusho_flag"
RANDOM_STATE = 42
N_TRIALS     = 50

# 特徴量・前処理は optuna_lgbm.py をそのまま流用
from optuna_lgbm import (
    CAT_FEATURES, NUM_FEATURES, TIME_STR_FEATURES, ALL_FEATURES,
    preprocess, load_data,
)
from utils import backup_model


# =========================================================
# Objective関数（Tweedie回帰）
# =========================================================
def make_objective(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_cols: list[str],
):
    X_va = valid[feature_cols]
    y_va_eval = valid[TARGET_EVAL]  # AUC評価用

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective":         "tweedie",
            "tweedie_variance_power": trial.suggest_float("tweedie_variance_power", 1.0, 1.9),
            "metric":            "rmse",
            "random_state":      RANDOM_STATE,
            "verbose":           -1,
            "num_leaves":        trial.suggest_int("num_leaves", 31, 255),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "n_estimators":      2000,
        }

        X_tr = train[feature_cols]
        y_tr = train[TARGET_TRAIN].clip(lower=0).fillna(0)

        model = LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, valid[TARGET_TRAIN].clip(lower=0).fillna(0))],
            callbacks=[
                early_stopping(stopping_rounds=50, verbose=False),
                log_evaluation(period=-1),
            ],
        )

        # 評価はfukusho_flagに対するAUC（回帰スコアをランキングとして使用）
        pred = model.predict(X_va)
        return roc_auc_score(y_va_eval, pred)

    return objective


# =========================================================
# main
# =========================================================
def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("データロード・前処理...")
    train, valid, test = load_data()
    train, encoders = preprocess(train, fit=True)
    valid, _        = preprocess(valid, encoders=encoders, fit=False)
    test,  _        = preprocess(test,  encoders=encoders, fit=False)

    feature_cols = [c for c in ALL_FEATURES if c in train.columns]
    logger.info(f"使用特徴量数: {len(feature_cols)}")
    logger.info(f"roi_target: mean={train[TARGET_TRAIN].mean():.3f}, "
                f"max={train[TARGET_TRAIN].max():.1f}, "
                f">0の割合={( train[TARGET_TRAIN] > 0).mean()*100:.1f}%")

    # ── Optuna 最適化 ──
    logger.info(f"Optuna開始: {N_TRIALS}試行（Tweedie回帰）")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        make_objective(train, valid, feature_cols),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    logger.info(f"最適パラメータ: {study.best_params}")
    logger.info(f"Best Valid AUC: {study.best_value:.4f}")

    # ── 最適パラメータで再学習 ──
    logger.info("最適パラメータで再学習中...")
    best_params = {
        **study.best_params,
        "objective":    "tweedie",
        "metric":       "rmse",
        "random_state": RANDOM_STATE,
        "verbose":      -1,
        "n_estimators": 2000,
    }

    X_tr = train[feature_cols]
    y_tr = train[TARGET_TRAIN].clip(lower=0).fillna(0)
    X_va = valid[feature_cols]
    y_va = valid[TARGET_TRAIN].clip(lower=0).fillna(0)

    best_model = LGBMRegressor(**best_params)
    best_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[
            early_stopping(stopping_rounds=50, verbose=False),
            log_evaluation(period=200),
        ],
    )

    # ── 評価（AUC on fukusho_flag） ──
    pred_va = best_model.predict(valid[feature_cols])
    pred_te = best_model.predict(test[feature_cols])
    auc_va  = roc_auc_score(valid[TARGET_EVAL], pred_va)
    auc_te  = roc_auc_score(test[TARGET_EVAL],  pred_te)

    # 比較基準（optuna_lgbm.py の最新）
    OLD_AUC_VA = 0.7412
    OLD_AUC_TE = 0.7474

    logger.info(f"[Valid] AUC={auc_va:.4f}  (binary旧: {OLD_AUC_VA})")
    logger.info(f"[Test]  AUC={auc_te:.4f}  (binary旧: {OLD_AUC_TE})")

    delta_va = auc_va - OLD_AUC_VA
    delta_te = auc_te - OLD_AUC_TE
    verdict = "改善✓" if delta_te > 0.001 else ("同等" if abs(delta_te) <= 0.001 else "悪化✗")
    logger.info(f"判定: {verdict}  ΔValid={delta_va:+.4f}  ΔTest={delta_te:+.4f}")

    # ── 保存 ──
    backup_model(MODEL_PATH)
    joblib.dump(
        {"model": best_model, "encoders": encoders, "feature_cols": feature_cols},
        MODEL_PATH,
    )
    joblib.dump(study, STUDY_PATH)
    logger.info(f"モデル保存: {MODEL_PATH}")

    print("\n" + "=" * 60)
    print("LightGBM ROI回帰 実験サマリ")
    print("=" * 60)
    print(f"Valid AUC : {auc_va:.4f}  (binary比: {delta_va:+.4f})")
    print(f"Test  AUC : {auc_te:.4f}  (binary比: {delta_te:+.4f})")
    print(f"判定      : {verdict}")
    if delta_te > 0.001:
        print("→ lgbm_roi_v1.pkl を lgbm_optuna_v1.pkl に置き換え推奨")
    else:
        print("→ binary分類を継続使用")
    print("=" * 60)


if __name__ == "__main__":
    main()
