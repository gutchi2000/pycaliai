"""
stacking.py
PyCaLiAI - 4モデルスタッキング（メタ学習）

Level 0: LGBM / CatBoost / YetiRank / Transformer PL（既存モデル）
Level 1: MetaLightGBM（4モデルの予測 + レース内順位 + レース条件）

Valid セット（2023年）= メタ学習データ（ベースモデルの訓練データに含まれない）
Test セット（2024年〜）= 評価データ

Usage:
    python stacking.py
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
BASE_DIR         = Path(r"E:\PyCaLiAI")
DATA_DIR         = BASE_DIR / "data"
MODEL_DIR        = BASE_DIR / "models"
REPORT_DIR       = BASE_DIR / "reports"

MASTER_CSV       = DATA_DIR  / "master_20130105-20251228.csv"
HOSEI_DIR        = DATA_DIR  / "hosei"
KEKKA_CSV        = DATA_DIR  / "kekka_20130105-20251228.csv"
LGBM_MODEL_PATH  = MODEL_DIR / "lgbm_optuna_v1.pkl"
CAT_MODEL_PATH   = MODEL_DIR / "catboost_optuna_v1.pkl"
RANK_MODEL_PATH  = MODEL_DIR / "catboost_rank_v1.pkl"
TORCH_MODEL_PATH = MODEL_DIR / "transformer_pl_v2.pkl"
META_MODEL_PATH  = MODEL_DIR / "stacking_meta_v1.pkl"

TARGET       = "fukusho_flag"
COL_RACE_ID  = "レースID(新)"
RANDOM_STATE = 42
N_TRIALS     = 50

# メタモデルに追加するレース条件特徴量
META_EXTRA_FEATURES = [
    "芝・ダ", "距離", "クラス名", "場所", "馬場状態",
    "出走頭数", "枠番", "馬番",
]

# =========================================================
# ベースモデル予測関数（calibrate.py から移植）
# =========================================================
from calibrate import (
    _predict_lgbm,
    _predict_catboost_rank,
    _predict_transformer,
)
from utils import parse_time_str


def _predict_catboost(df: pd.DataFrame, obj: dict) -> np.ndarray:
    from catboost import Pool
    model, feature_cols = obj["model"], obj["feature_cols"]
    cat_list = [
        "種牡馬", "父タイプ名", "母父馬", "母父タイプ名", "毛色",
        "馬主(最新/仮想)", "生産者", "芝・ダ", "コース区分", "芝(内・外)",
        "馬場状態", "天気", "クラス名", "場所", "性別", "斤量",
        "ブリンカー", "重量種別", "年齢限定", "限定", "性別限定", "指定条件",
        "前走場所", "前芝・ダ", "前走馬場状態", "前走斤量", "前好走",
    ]
    df = df.copy()
    for col in ["前走走破タイム", "前走着差タイム"]:
        if col in df.columns:
            df[col] = parse_time_str(df[col])
    for col in cat_list:
        df[col] = df[col].fillna("__NaN__").astype(str) if col in df.columns else "__NaN__"
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    cat_idx = [i for i, c in enumerate(feature_cols) if c in cat_list]
    pool = Pool(df[feature_cols], cat_features=cat_idx)
    return model.predict_proba(pool)[:, 1]


# =========================================================
# データロード
# =========================================================
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info(f"マスターCSV読み込み: {MASTER_CSV}")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)

    # hosei JOIN（data/hosei/H_*.csv を全て結合）
    hosei_files = sorted(HOSEI_DIR.glob("H_*.csv"))
    if hosei_files:
        hosei = pd.concat([
            pd.read_csv(f, encoding="cp932",
                        usecols=["レースID(新)", "前走補9", "前走補正"])
            for f in hosei_files
        ], ignore_index=True).drop_duplicates()
        df = df.merge(hosei, on="レースID(新)", how="left")
        logger.info(f"  hosei JOIN ({len(hosei_files)}ファイル): 前走補9カバレッジ={df['前走補9'].notna().mean()*100:.1f}%")
    else:
        df["前走補9"] = float("nan")
        df["前走補正"] = float("nan")

    # 前走単勝オッズ JOIN
    if KEKKA_CSV.exists():
        kekka = pd.read_csv(KEKKA_CSV, encoding="cp932",
                            usecols=["レースID(新)", "馬番", "単勝配当"])
        def _parse_tansho(s):
            s = str(s).strip()
            if s.startswith("(") and s.endswith(")"):
                return float(s[1:-1])
            try:
                return float(s) / 100
            except Exception:
                return float("nan")
        kekka["単勝オッズ"] = kekka["単勝配当"].apply(_parse_tansho)
        kekka = kekka.drop(columns=["単勝配当"])
        kekka_prev = kekka.rename(columns={"レースID(新)": "前走レースID(新)", "単勝オッズ": "前走単勝オッズ"})
        kekka_prev["前走レースID(新)"] = kekka_prev["前走レースID(新)"].astype("int64")
        kekka_prev["馬番"] = kekka_prev["馬番"].astype("int64")
        df["前走レースID(新)"] = pd.to_numeric(df["前走レースID(新)"], errors="coerce")
        df = df.merge(kekka_prev, on=["前走レースID(新)", "馬番"], how="left")
    else:
        df["前走単勝オッズ"] = float("nan")

    # レースID(新/馬番無)
    if "レースID(新/馬番無)" not in df.columns:
        df["レースID(新/馬番無)"] = df["レースID(新)"].astype(str).str[:16]

    train = df[df["split"] == "train"].copy().reset_index(drop=True)
    valid = df[df["split"] == "valid"].copy().reset_index(drop=True)
    test  = df[df["split"] == "test"].copy().reset_index(drop=True)
    logger.info(f"  分割: train={len(train):,} / valid={len(valid):,} / test={len(test):,}")
    return train, valid, test


# =========================================================
# メタ特徴量構築
# =========================================================
def build_meta_features(
    df: pd.DataFrame,
    p_lgbm: np.ndarray,
    p_cat: np.ndarray,
    p_rank: np.ndarray,
    p_trans: np.ndarray,
    meta_encoders: dict | None = None,
    fit: bool = True,
    race_col: str = "レースID(新/馬番無)",
) -> tuple[pd.DataFrame, dict]:
    """
    メタ特徴量：
    - 4モデルの予測確率
    - 4モデルそれぞれのレース内順位（1=最高スコア）
    - レース条件特徴量
    """
    meta = df[META_EXTRA_FEATURES].copy()

    # 予測確率
    meta["p_lgbm"]  = p_lgbm
    meta["p_cat"]   = p_cat
    meta["p_rank"]  = p_rank
    meta["p_trans"] = p_trans

    # レース内順位（1=最高評価）
    df_tmp = df[[race_col]].copy()
    for col_name, proba in [("p_lgbm", p_lgbm), ("p_cat", p_cat),
                             ("p_rank", p_rank), ("p_trans", p_trans)]:
        df_tmp[col_name] = proba
        rank_col = f"rank_{col_name.split('_')[1]}"
        meta[rank_col] = df_tmp.groupby(race_col)[col_name].rank(
            ascending=False, method="min"
        ).values

    # 予測の分散・最大差（モデル間の一致度）
    preds = np.stack([p_lgbm, p_cat, p_rank, p_trans], axis=1)
    meta["pred_mean"]  = preds.mean(axis=1)
    meta["pred_std"]   = preds.std(axis=1)
    meta["pred_range"] = preds.max(axis=1) - preds.min(axis=1)

    # カテゴリエンコード
    if meta_encoders is None:
        meta_encoders = {}
    for col in meta.select_dtypes(include="object").columns:
        meta[col] = meta[col].fillna("__NaN__").astype(str)
        if fit:
            le   = LabelEncoder()
            vals = meta[col].tolist()
            if "__NaN__" not in vals:
                vals.append("__NaN__")
            le.fit(vals)
            meta_encoders[col] = le
        else:
            le    = meta_encoders.get(col)
            if le is None:
                meta[col] = 0
                continue
            known = set(le.classes_)
            meta[col] = meta[col].apply(lambda x: x if x in known else "__NaN__")
        meta[col] = meta_encoders[col].transform(meta[col])

    # 数値列のNaN補完
    for col in meta.select_dtypes(include=[float, int]).columns:
        meta[col] = pd.to_numeric(meta[col], errors="coerce").fillna(0)

    return meta, meta_encoders


# =========================================================
# Optuna 目標関数
# =========================================================
def make_meta_objective(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective":         "binary",
            "metric":            "auc",
            "random_state":      RANDOM_STATE,
            "verbose":           -1,
            "num_leaves":        trial.suggest_int("num_leaves", 16, 127),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "n_estimators":      2000,
        }
        model = LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[
                early_stopping(stopping_rounds=50, verbose=False),
                log_evaluation(period=-1),
            ],
        )
        return roc_auc_score(y_va, model.predict_proba(X_va)[:, 1])
    return objective


# =========================================================
# main
# =========================================================
def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # データロード
    _, valid, test = load_data()

    # ベースモデル読み込み
    logger.info("ベースモデル読み込み...")
    lgbm_obj = joblib.load(LGBM_MODEL_PATH)
    cat_obj  = joblib.load(CAT_MODEL_PATH)
    rank_obj = joblib.load(RANK_MODEL_PATH) if RANK_MODEL_PATH.exists() else None
    use_rank = rank_obj is not None

    # ── Valid セット予測（メタ学習データ） ──
    logger.info("Valid セット予測（4モデル）...")
    pv_lgbm  = _predict_lgbm(valid, lgbm_obj)
    pv_cat   = _predict_catboost(valid, cat_obj)
    pv_rank  = _predict_catboost_rank(valid, rank_obj) if use_rank else np.full(len(valid), 0.5)
    pv_trans = _predict_transformer(valid)

    logger.info(f"  LGBM  AUC={roc_auc_score(valid[TARGET], pv_lgbm):.4f}")
    logger.info(f"  Cat   AUC={roc_auc_score(valid[TARGET], pv_cat):.4f}")
    if use_rank:
        logger.info(f"  Rank  AUC={roc_auc_score(valid[TARGET], pv_rank):.4f}")
    if pv_trans.any():
        logger.info(f"  Trans AUC={roc_auc_score(valid[TARGET], pv_trans):.4f}")

    # ── Test セット予測（評価データ） ──
    logger.info("Test セット予測（4モデル）...")
    pt_lgbm  = _predict_lgbm(test, lgbm_obj)
    pt_cat   = _predict_catboost(test, cat_obj)
    pt_rank  = _predict_catboost_rank(test, rank_obj) if use_rank else np.full(len(test), 0.5)
    pt_trans = _predict_transformer(test)

    # ── メタ特徴量構築 ──
    logger.info("メタ特徴量構築...")
    meta_valid, meta_encoders = build_meta_features(
        valid, pv_lgbm, pv_cat, pv_rank, pv_trans, fit=True
    )
    meta_test, _ = build_meta_features(
        test, pt_lgbm, pt_cat, pt_rank, pt_trans,
        meta_encoders=meta_encoders, fit=False
    )

    meta_cols = meta_valid.columns.tolist()
    logger.info(f"  メタ特徴量数: {len(meta_cols)}")

    X_va = meta_valid
    y_va = valid[TARGET]
    X_te = meta_test
    y_te = test[TARGET]

    # ── Optuna 最適化 ──
    logger.info(f"Optuna開始: {N_TRIALS}試行...")

    # valid を train/val に分割（時系列順）してOptunaで調整
    split_idx = int(len(X_va) * 0.7)
    X_opt_tr, y_opt_tr = X_va.iloc[:split_idx], y_va.iloc[:split_idx]
    X_opt_va, y_opt_va = X_va.iloc[split_idx:], y_va.iloc[split_idx:]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        make_meta_objective(X_opt_tr, y_opt_tr, X_opt_va, y_opt_va),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )
    logger.info(f"最適パラメータ: {study.best_params}")
    logger.info(f"Best Optuna AUC: {study.best_value:.4f}")

    # ── メタモデル再学習（valid全体で） ──
    logger.info("メタモデル最終学習（valid全体）...")
    best_params = {
        **study.best_params,
        "objective":    "binary",
        "metric":       "auc",
        "random_state": RANDOM_STATE,
        "verbose":      -1,
        "n_estimators": 2000,
    }
    meta_model = LGBMClassifier(**best_params)
    meta_model.fit(
        X_opt_tr, y_opt_tr,
        eval_set=[(X_opt_va, y_opt_va)],
        callbacks=[
            early_stopping(stopping_rounds=50, verbose=False),
            log_evaluation(period=200),
        ],
    )

    # ── 評価 ──
    proba_va = meta_model.predict_proba(X_va)[:, 1]
    proba_te = meta_model.predict_proba(X_te)[:, 1]
    auc_va   = roc_auc_score(y_va, proba_va)
    auc_te   = roc_auc_score(y_te, proba_te)

    # 比較基準：現行アンサンブル加重平均
    OLD_AUC_VA = 0.7712
    OLD_AUC_TE = 0.7756

    delta_va = auc_va - OLD_AUC_VA
    delta_te = auc_te - OLD_AUC_TE
    verdict = "改善" if delta_te > 0.001 else ("同等" if abs(delta_te) <= 0.001 else "悪化")

    logger.info(f"[Valid] AUC={auc_va:.4f}  (加重平均: {OLD_AUC_VA}  差: {delta_va:+.4f})")
    logger.info(f"[Test]  AUC={auc_te:.4f}  (加重平均: {OLD_AUC_TE}  差: {delta_te:+.4f})")
    logger.info(f"判定: {verdict}")

    # 特徴量重要度
    importance = pd.Series(
        meta_model.feature_importances_, index=meta_cols
    ).sort_values(ascending=False)
    logger.info("特徴量重要度 (上位10):")
    for feat, imp in importance.head(10).items():
        logger.info(f"  {feat}: {imp}")

    # ── 保存 ──
    joblib.dump(
        {
            "meta_model":    meta_model,
            "meta_encoders": meta_encoders,
            "meta_cols":     meta_cols,
        },
        META_MODEL_PATH,
    )
    logger.info(f"メタモデル保存: {META_MODEL_PATH}")

    print("\n" + "=" * 60)
    print("Stacking 完了サマリ")
    print("=" * 60)
    print(f"Valid AUC  : {auc_va:.4f}  (加重平均比: {delta_va:+.4f})")
    print(f"Test  AUC  : {auc_te:.4f}  (加重平均比: {delta_te:+.4f})")
    print(f"判定       : {verdict}")
    if delta_te > 0.001:
        print("-> stacking_meta_v1.pkl を calibrate.py / predict_weekly.py で有効化推奨")
    print("=" * 60)


if __name__ == "__main__":
    main()
