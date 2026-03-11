"""
optuna_catboost.py
PyCaLiAI - CatBoost Optuna ハイパーパラメータ最適化

Usage:
    python optuna_catboost.py
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import joblib
import optuna
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

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

MASTER_CSV = DATA_DIR / "master_20130105-20251228.csv"
MODEL_PATH = MODEL_DIR / "catboost_optuna_v1.pkl"
STUDY_PATH = REPORT_DIR / "optuna_catboost_study.pkl"

TARGET       = "fukusho_flag"
RANDOM_STATE = 42
N_TRIALS     = 50

CAT_FEATURES_LIST = [
    "種牡馬", "父タイプ名", "母父馬", "母父タイプ名", "毛色",
    "馬主(最新/仮想)", "生産者",
    "芝・ダ", "コース区分", "芝(内・外)", "馬場状態", "天気",
    "クラス名", "場所",
    "性別", "斤量", "ブリンカー", "重量種別",
    "年齢限定", "限定", "性別限定", "指定条件",
    "前走場所", "前芝・ダ", "前走馬場状態", "前走斤量", "前好走",
]

NUM_FEATURES = [
    "距離", "トラックコード(JV)", "出走頭数", "フルゲート頭数",
    "騎手コード", "調教師コード", "騎手年齢", "調教師年齢",
    "枠番", "馬番", "年齢", "馬齢斤量差", "斤量体重比",
    "間隔", "休み明け～戦目",
    "前走確定着順", "前距離", "前走出走頭数", "前走競走種別",
    "前走トラックコード(JV)", "前走馬体重", "前走馬体重増減",
    "前1角", "前2角", "前3角", "前4角",
    "前走上り3F", "前走上り3F順",
    "前走Ave-3F", "前PCI", "前走PCI3", "前走RPCI",
    "前走平均1Fタイム",
    # 騎手・調教師の直近成績（build_dataset.py で生成）
    "jockey_fuku30", "jockey_fuku90",
    "trainer_fuku30", "trainer_fuku90",
    # 馬の直近成績（build_dataset.py で生成）
    "horse_fuku10", "horse_fuku30",
    # 脚質特徴量（build_dataset.py で生成）
    "prev_pos_rel", "closing_power",
]

TIME_STR_FEATURES = ["前走走破タイム", "前走着差タイム"]


from utils import parse_time_str, backup_model


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in TIME_STR_FEATURES:
        if col in df.columns:
            df[col] = parse_time_str(df[col])
    for col in CAT_FEATURES_LIST:
        if col in df.columns:
            df[col] = df[col].fillna("__NaN__").astype(str)
        else:
            df[col] = "__NaN__"
    for col in NUM_FEATURES + TIME_STR_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = 0.0
    return df


def make_pool(
    df: pd.DataFrame,
    feature_cols: list[str],
    with_label: bool = True,
    sample_weight: pd.Series | None = None,
) -> Pool:
    cat_indices = [
        i for i, c in enumerate(feature_cols)
        if c in CAT_FEATURES_LIST
    ]
    if with_label:
        return Pool(
            df[feature_cols], df[TARGET].astype(int),
            cat_features=cat_indices,
            weight=sample_weight,
        )
    return Pool(df[feature_cols], cat_features=cat_indices)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info(f"マスターCSV読み込み: {MASTER_CSV}")
    df    = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    train = preprocess(df[df["split"] == "train"].copy())
    valid = preprocess(df[df["split"] == "valid"].copy())
    test  = preprocess(df[df["split"] == "test"].copy())
    logger.info(
        f"分割: train={len(train):,} / valid={len(valid):,} / test={len(test):,}"
    )
    return train, valid, test


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train, valid, test = load_data()

    all_features = CAT_FEATURES_LIST + NUM_FEATURES + TIME_STR_FEATURES
    feature_cols = [c for c in all_features if c in train.columns]
    logger.info(f"使用特徴量数: {len(feature_cols)}")

    # CatBoost は auto_class_weights="Balanced" で不均衡対応済み。
    # roi_target を sample_weight に加えると正例が過剰重み付けされるため使わない。
    pool_tr = make_pool(train, feature_cols)
    pool_va = make_pool(valid, feature_cols)
    pool_te = make_pool(test,  feature_cols, with_label=False)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations":            500,
            "eval_metric":           "AUC",
            "random_seed":           RANDOM_STATE,
            "verbose":               0,
            "early_stopping_rounds": 50,
            "auto_class_weights":    "Balanced",   # 不均衡対策（roi_target と併用可能）
            "task_type":             "GPU",         # RTX 3070 Ti で高速化
            "learning_rate":         trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "depth":                 trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg":           trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
            "bagging_temperature":   trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength":       trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
            "border_count":          trial.suggest_int("border_count", 32, 255),
        }

        model = CatBoostClassifier(**params)
        model.fit(pool_tr, eval_set=pool_va, use_best_model=True)

        proba = model.predict_proba(pool_va)[:, 1]
        return roc_auc_score(valid[TARGET].astype(int), proba)

    logger.info(f"Optuna開始: {N_TRIALS}試行")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    logger.info(f"最適パラメータ: {study.best_params}")
    logger.info(f"Best Valid AUC: {study.best_value:.4f}")

    # 最適パラメータで再学習（iterationsを2000に増やす）
    logger.info("最適パラメータで再学習中...")
    best_params = {
        **study.best_params,
        "iterations":            2000,
        "eval_metric":           "AUC",
        "random_seed":           RANDOM_STATE,
        "verbose":               200,
        "early_stopping_rounds": 50,
        "auto_class_weights":    "Balanced",   # 不均衡対策（roi_target と併用可能）
        "task_type":             "GPU",         # RTX 3070 Ti で高速化
    }

    best_model = CatBoostClassifier(**best_params)
    best_model.fit(pool_tr, eval_set=pool_va, use_best_model=True)

    proba_va = best_model.predict_proba(pool_va)[:, 1]
    proba_te = best_model.predict_proba(pool_te)[:, 1]
    auc_va   = roc_auc_score(valid[TARGET].astype(int), proba_va)
    auc_te   = roc_auc_score(test[TARGET].astype(int),  proba_te)

    logger.info(f"[Valid] AUC={auc_va:.4f}  (旧: 0.7433)")
    logger.info(f"[Test]  AUC={auc_te:.4f}  (旧: 0.7431)")

    # 既存モデルをバックアップしてから上書き
    backup_model(MODEL_PATH)
    joblib.dump(
        {"model": best_model, "feature_cols": feature_cols},
        MODEL_PATH,
    )
    joblib.dump(study, STUDY_PATH)
    logger.info(f"モデル保存: {MODEL_PATH}")
    logger.info(f"Study保存: {STUDY_PATH}")

    print("\n" + "=" * 50)
    print("CatBoost Optuna 最適化完了サマリ")
    print("=" * 50)
    print(f"Valid AUC : {auc_va:.4f}  (旧: 0.7433)")
    print(f"Test  AUC : {auc_te:.4f}  (旧: 0.7431)")
    print(f"Best試行  : Trial {study.best_trial.number}")
    print(f"\n最適パラメータ:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()