"""
optuna_lgbm.py
PyCaLiAI - LightGBM Optuna ハイパーパラメータ最適化

Usage:
    python optuna_lgbm.py
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import joblib
import optuna
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
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
# パス設定
# =========================================================
BASE_DIR   = Path(r"E:\PyCaLiAI")
DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MASTER_CSV = DATA_DIR / "master_20130105-20251228.csv"
MODEL_PATH = MODEL_DIR / "lgbm_optuna_v1.pkl"
STUDY_PATH = REPORT_DIR / "optuna_lgbm_study.pkl"

TARGET       = "fukusho_flag"
RANDOM_STATE = 42
N_TRIALS     = 50

# =========================================================
# 特徴量定義（train_lgbm.pyと同じ）
# =========================================================
CAT_FEATURES = [
    "芝・ダ", "コース区分", "芝(内・外)", "馬場状態", "天気",
    "クラス名", "場所",
    "前走場所", "前芝・ダ", "前走馬場状態", "前走斤量",
    "前好走", "性別", "斤量", "ブリンカー",
]

NUM_FEATURES = [
    "距離", "トラックコード(JV)", "出走頭数", "フルゲート頭数",
    "枠番", "馬番", "年齢", "馬齢斤量差", "斤量体重比",
    "間隔", "休み明け～戦目", "騎手年齢", "調教師年齢",
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
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES + TIME_STR_FEATURES


# =========================================================
# 前処理
# =========================================================
from utils import parse_time_str, backup_model


def encode_categoricals(
    df: pd.DataFrame,
    cat_cols: list[str],
    encoders: dict[str, LabelEncoder] | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    df = df.copy()
    if encoders is None:
        encoders = {}
    for col in cat_cols:
        if col not in df.columns:
            df[col] = 0
            continue
        df[col] = df[col].astype(str).fillna("__NaN__")
        if fit:
            le = LabelEncoder()
            vals = df[col].tolist()
            if "__NaN__" not in vals:
                vals.append("__NaN__")
            le.fit(vals)
            encoders[col] = le
        else:
            le = encoders[col]
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else "__NaN__")
        df[col] = le.transform(df[col])
    return df, encoders


def preprocess(
    df: pd.DataFrame,
    encoders: dict | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    for col in TIME_STR_FEATURES:
        if col in df.columns:
            df[col] = parse_time_str(df[col])
    df, encoders = encode_categoricals(df, CAT_FEATURES, encoders, fit=fit)
    return df, encoders


# =========================================================
# データロード（1回だけ読み込んでキャッシュ）
# =========================================================
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info(f"マスターCSV読み込み: {MASTER_CSV}")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    train = df[df["split"] == "train"].copy()
    valid = df[df["split"] == "valid"].copy()
    test  = df[df["split"] == "test"].copy()
    logger.info(
        f"分割: train={len(train):,} / valid={len(valid):,} / test={len(test):,}"
    )
    return train, valid, test


# =========================================================
# Objective関数
# =========================================================
def make_objective(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    encoders: dict,
    feature_cols: list[str],
):
    """Optunaのobjective関数を生成して返す。"""

    X_va = valid[feature_cols]
    y_va = valid[TARGET]

    # roi_target を sample_weight として使用（圏外=1.0, 高配当3着以内=高weight）
    # min_periods=5未満の行は NaN → 1.0 で補完
    w_tr = train["roi_target"].clip(lower=1.0).fillna(1.0) if "roi_target" in train.columns else None

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective":           "binary",
            "metric":              "auc",
            "random_state":        RANDOM_STATE,
            "verbose":             -1,
            # scale_pos_weight は roi_target の sample_weight で代替
            # 探索するパラメータ
            "num_leaves":          trial.suggest_int("num_leaves", 31, 255),
            "learning_rate":       trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_child_samples":   trial.suggest_int("min_child_samples", 20, 200),
            "subsample":           trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":    trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":           trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":          trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_split_gain":      trial.suggest_float("min_split_gain", 0.0, 1.0),
            "n_estimators":        2000,
        }

        X_tr = train[feature_cols]
        y_tr = train[TARGET]

        model = LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[
                early_stopping(stopping_rounds=50, verbose=False),
                log_evaluation(period=-1),
            ],
        )

        proba = model.predict_proba(X_va)[:, 1]
        return roc_auc_score(y_va, proba)

    return objective


# =========================================================
# main
# =========================================================
def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # データロード・前処理
    train, valid, test = load_data()
    train, encoders = preprocess(train, fit=True)
    valid, _        = preprocess(valid, encoders=encoders, fit=False)
    test,  _        = preprocess(test,  encoders=encoders, fit=False)

    feature_cols = [c for c in ALL_FEATURES if c in train.columns]
    logger.info(f"使用特徴量数: {len(feature_cols)}")

    # Optuna最適化
    logger.info(f"Optuna開始: {N_TRIALS}試行")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        make_objective(train, valid, encoders, feature_cols),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    logger.info(f"最適パラメータ: {study.best_params}")
    logger.info(f"Best Valid AUC: {study.best_value:.4f}")

    # 最適パラメータで再学習
    logger.info("最適パラメータで再学習中...")
    best_params = {
        **study.best_params,
        "objective":    "binary",
        "metric":       "auc",
        "random_state": RANDOM_STATE,
        "verbose":      -1,
        "n_estimators": 2000,
    }

    X_tr, y_tr = train[feature_cols], train[TARGET]
    X_va, y_va = valid[feature_cols], valid[TARGET]
    w_tr = train["roi_target"].clip(lower=1.0).fillna(1.0) if "roi_target" in train.columns else None

    best_model = LGBMClassifier(**best_params)
    best_model.fit(
        X_tr, y_tr,
        sample_weight=w_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[
            early_stopping(stopping_rounds=50, verbose=False),
            log_evaluation(period=200),
        ],
    )

    # Test評価
    X_te, y_te = test[feature_cols], test[TARGET]
    proba_va   = best_model.predict_proba(X_va)[:, 1]
    proba_te   = best_model.predict_proba(X_te)[:, 1]
    auc_va     = roc_auc_score(y_va, proba_va)
    auc_te     = roc_auc_score(y_te, proba_te)

    logger.info(f"[Valid] AUC={auc_va:.4f}  (旧: 0.7412)")
    logger.info(f"[Test]  AUC={auc_te:.4f}  (旧: 0.7474)")

    # 保存（既存モデルをバックアップしてから上書き）
    backup_model(MODEL_PATH)
    joblib.dump(
        {"model": best_model, "encoders": encoders, "feature_cols": feature_cols},
        MODEL_PATH,
    )
    joblib.dump(study, STUDY_PATH)
    logger.info(f"モデル保存: {MODEL_PATH}")
    logger.info(f"Study保存: {STUDY_PATH}")

    print("\n" + "=" * 50)
    print("LightGBM Optuna 最適化完了サマリ")
    print("=" * 50)
    print(f"Valid AUC : {auc_va:.4f}  (旧: 0.7412)")
    print(f"Test  AUC : {auc_te:.4f}  (旧: 0.7474)")
    print(f"Best試行  : Trial {study.best_trial.number}")
    print(f"\n最適パラメータ:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()