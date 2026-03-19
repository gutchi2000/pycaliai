"""
optimize_ensemble_weights.py
PyCaLiAI - アンサンブル重みのOptuna最適化

ValidセットでLGBM/CatBoost/YetiRank/TransformerPLの重みを探索し、
AUC最大化・Brier最小化を同時に最適化する。

Usage:
    python optimize_ensemble_weights.py
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import joblib
import optuna
import numpy as np
import pandas as pd
from catboost import Pool
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

from utils import parse_time_str

BASE_DIR   = Path(r"E:\PyCaLiAI")
DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MASTER_CSV      = DATA_DIR  / "master_20130105-20251228.csv"
HOSSEI_CSV      = DATA_DIR  / "hossei" / "H_20130105-20251228.csv"
LGBM_MODEL_PATH = MODEL_DIR / "lgbm_optuna_v1.pkl"
CAT_MODEL_PATH  = MODEL_DIR / "catboost_optuna_v1.pkl"
RANK_MODEL_PATH = MODEL_DIR / "catboost_rank_v1.pkl"
TORCH_MODEL_PATH= MODEL_DIR / "transformer_pl_v2.pkl"
CAL_OUT_PATH    = MODEL_DIR / "ensemble_calibrator_v1.pkl"

TARGET     = "fukusho_flag"
N_TRIALS   = 100
RANDOM_STATE = 42

_model_cache: dict = {}


# =========================================================
# 予測関数（calibrate.py から流用）
# =========================================================
def _predict_lgbm(df: pd.DataFrame, obj: dict) -> np.ndarray:
    model, encoders, feature_cols = obj["model"], obj["encoders"], obj["feature_cols"]
    df = df.copy()
    for col in ["前走走破タイム", "前走着差タイム"]:
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
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    import lightgbm as lgb
    if isinstance(model, lgb.Booster):
        return model.predict(df[feature_cols])
    return model.predict_proba(df[feature_cols])[:, 1]


def _predict_catboost(df: pd.DataFrame, obj: dict) -> np.ndarray:
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


def _predict_catboost_rank(df: pd.DataFrame, obj: dict) -> np.ndarray:
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
    scores = model.predict(pool)

    race_id_col = "レースID(新/馬番無)"
    result = np.full(len(df), 0.5)
    df_reset = df.reset_index(drop=True)
    for _, group in df_reset.groupby(race_id_col):
        idxs = group.index.tolist()
        s = scores[idxs]
        s_min, s_max = s.min(), s.max()
        if s_max > s_min:
            result[idxs] = (s - s_min) / (s_max - s_min)
    return result


def _predict_transformer(df: pd.DataFrame) -> np.ndarray:
    if not TORCH_MODEL_PATH.exists():
        return np.zeros(len(df))
    if "torch" not in _model_cache:
        _model_cache["torch"] = joblib.load(TORCH_MODEL_PATH)
    torch_obj = _model_cache["torch"]
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

        race_id_col = "レースID(新/馬番無)"
        result  = np.zeros(len(df))
        df_sort = df.sort_values(race_id_col).reset_index(drop=True)
        idx = 0
        for _, group in df_sort.groupby(race_id_col, sort=True):
            valid_n = min(len(group), PL_MAX_HORSES)
            for orig_idx in list(group.index)[:valid_n]:
                if idx < len(all_scores):
                    result[orig_idx] = all_scores[idx]
                    idx += 1

        df_reset = df.reset_index(drop=True)
        for _, group in df_reset.groupby(race_id_col):
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


# =========================================================
# メイン
# =========================================================
def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # データ読み込み
    logger.info(f"マスターCSV読み込み: {MASTER_CSV}")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)

    if HOSSEI_CSV.exists():
        hossei = pd.read_csv(HOSSEI_CSV, encoding="cp932",
                             usecols=["レースID(新)", "馬番", "前走補9", "前走補正"])
        df = df.merge(hossei, on=["レースID(新)", "馬番"], how="left")

    valid = df[df["split"] == "valid"].copy()
    test  = df[df["split"] == "test"].copy()
    logger.info(f"Valid: {len(valid):,} / Test: {len(test):,}")

    y_valid = valid[TARGET].values
    y_test  = test[TARGET].values

    # 全モデル予測（1回だけ計算してキャッシュ）
    logger.info("モデル読み込み・予測中...")
    lgbm_obj = joblib.load(LGBM_MODEL_PATH)
    cat_obj  = joblib.load(CAT_MODEL_PATH)
    rank_obj = joblib.load(RANK_MODEL_PATH)

    logger.info("  LGBM...")
    p_lgbm_val  = _predict_lgbm(valid, lgbm_obj)
    p_lgbm_test = _predict_lgbm(test,  lgbm_obj)

    logger.info("  CatBoost...")
    p_cat_val  = _predict_catboost(valid, cat_obj)
    p_cat_test = _predict_catboost(test,  cat_obj)

    logger.info("  YetiRank...")
    p_rank_val  = _predict_catboost_rank(valid, rank_obj)
    p_rank_test = _predict_catboost_rank(test,  rank_obj)

    logger.info("  Transformer PL...")
    p_trans_val  = _predict_transformer(valid)
    p_trans_test = _predict_transformer(test)

    # 各モデル単体AUCを表示
    logger.info("\n--- 各モデル単体 Valid AUC ---")
    for name, p in [("LGBM", p_lgbm_val), ("CatBoost", p_cat_val),
                    ("YetiRank", p_rank_val), ("TransPL", p_trans_val)]:
        logger.info(f"  {name}: {roc_auc_score(y_valid, p):.4f}")

    # =========================================================
    # Optuna: 重みを探索（合計1に正規化）
    # =========================================================
    def objective(trial: optuna.Trial) -> float:
        w1 = trial.suggest_float("w_lgbm",  0.0, 1.0)
        w2 = trial.suggest_float("w_cat",   0.0, 1.0)
        w3 = trial.suggest_float("w_rank",  0.0, 1.0)
        w4 = trial.suggest_float("w_trans", 0.0, 1.0)
        total = w1 + w2 + w3 + w4
        if total < 1e-8:
            raise optuna.TrialPruned()
        w1, w2, w3, w4 = w1/total, w2/total, w3/total, w4/total

        ens = w1*p_lgbm_val + w2*p_cat_val + w3*p_rank_val + w4*p_trans_val

        # キャリブレーション後のAUCで評価
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(ens, y_valid)
        ens_cal = ir.transform(ens)
        return roc_auc_score(y_valid, ens_cal)

    logger.info(f"\nOptuna重み最適化: {N_TRIALS}試行...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    bp = study.best_params
    total = bp["w_lgbm"] + bp["w_cat"] + bp["w_rank"] + bp["w_trans"]
    w_lgbm  = bp["w_lgbm"]  / total
    w_cat   = bp["w_cat"]   / total
    w_rank  = bp["w_rank"]  / total
    w_trans = bp["w_trans"] / total

    logger.info(f"\n--- 最適重み ---")
    logger.info(f"  LGBM:      {w_lgbm:.3f}")
    logger.info(f"  CatBoost:  {w_cat:.3f}")
    logger.info(f"  YetiRank:  {w_rank:.3f}")
    logger.info(f"  TransPL:   {w_trans:.3f}")

    # Valid / Test での最終評価
    ens_val  = w_lgbm*p_lgbm_val  + w_cat*p_cat_val  + w_rank*p_rank_val  + w_trans*p_trans_val
    ens_test = w_lgbm*p_lgbm_test + w_cat*p_cat_test + w_rank*p_rank_test + w_trans*p_trans_test

    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(ens_val, y_valid)
    ens_val_cal  = ir.transform(ens_val)
    ens_test_cal = ir.transform(ens_test)

    auc_val  = roc_auc_score(y_valid, ens_val_cal)
    auc_test = roc_auc_score(y_test,  ens_test_cal)
    brier_val  = brier_score_loss(y_valid, ens_val_cal)
    brier_test = brier_score_loss(y_test,  ens_test_cal)

    # 手決め重み（現在）との比較
    ens_manual_val  = 0.30*p_lgbm_val  + 0.30*p_cat_val  + 0.20*p_rank_val  + 0.20*p_trans_val
    ens_manual_test = 0.30*p_lgbm_test + 0.30*p_cat_test + 0.20*p_rank_test + 0.20*p_trans_test
    ir_m = IsotonicRegression(out_of_bounds="clip")
    ir_m.fit(ens_manual_val, y_valid)
    auc_manual_val  = roc_auc_score(y_valid, ir_m.transform(ens_manual_val))
    auc_manual_test = roc_auc_score(y_test,  ir_m.transform(ens_manual_test))

    logger.info(f"\n--- Valid AUC 比較 ---")
    logger.info(f"  手決め（0.30/0.30/0.20/0.20）: {auc_manual_val:.4f}")
    logger.info(f"  Optuna最適化:                  {auc_val:.4f}  ({auc_val - auc_manual_val:+.4f})")
    logger.info(f"\n--- Test AUC 比較 ---")
    logger.info(f"  手決め（0.30/0.30/0.20/0.20）: {auc_manual_test:.4f}")
    logger.info(f"  Optuna最適化:                  {auc_test:.4f}  ({auc_test - auc_manual_test:+.4f})")

    # キャリブレーター保存（最適重み込み）
    payload = {
        "calibrator":      ir,
        "w_lgbm":          w_lgbm,
        "w_cat":           w_cat,
        "w_rank":          w_rank,
        "w_trans":         w_trans,
        "valid_auc":       auc_val,
        "test_auc":        auc_test,
        "valid_brier":     brier_val,
        "test_brier":      brier_test,
        "valid_pos_rate":  y_valid.mean(),
    }
    joblib.dump(payload, CAL_OUT_PATH)
    logger.info(f"\nキャリブレーター保存（最適重み込み）: {CAL_OUT_PATH}")

    print("\n" + "=" * 50)
    print("アンサンブル重み最適化 完了サマリ")
    print("=" * 50)
    print(f"最適重み: LGBM={w_lgbm:.3f}  CatBoost={w_cat:.3f}  YetiRank={w_rank:.3f}  TransPL={w_trans:.3f}")
    print(f"Valid AUC: {auc_manual_val:.4f} → {auc_val:.4f}  ({auc_val - auc_manual_val:+.4f})")
    print(f"Test  AUC: {auc_manual_test:.4f} → {auc_test:.4f}  ({auc_test - auc_manual_test:+.4f})")


if __name__ == "__main__":
    main()
