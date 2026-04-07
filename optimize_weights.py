"""
optimize_weights.py
PyCaLiAI - アンサンブル重み自動最適化

全モデルのValidation予測を生成し、Nelder-Mead最適化で
AUC（+ optionally PR-AUC）を最大化する重みを自動探索する。

結果は models/ensemble_weights.json に保存し、
predict_weekly.py の ensemble_predict() がロードして使用する。

Usage:
    python optimize_weights.py
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

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
MASTER_CSV = DATA_DIR / "master_kako5.csv"
WEIGHTS_PATH = MODEL_DIR / "ensemble_weights.json"

TARGET = "fukusho_flag"
COL_RACE_ID = "レースID(新/馬番無)"

# =========================================================
# モデルパス
# =========================================================
MODELS = {
    "lgbm":        MODEL_DIR / "lgbm_optuna_v1.pkl",
    "catboost":    MODEL_DIR / "catboost_optuna_v1.pkl",
    "fuku_lgbm":   MODEL_DIR / "lgbm_fukusho_v1.pkl",
    "fuku_cat":    MODEL_DIR / "catboost_fukusho_v1.pkl",
    "rank_cat":    MODEL_DIR / "catboost_rank_v1.pkl",     # YetiRank
    "rank_lgbm":   MODEL_DIR / "lgbm_rank_v1.pkl",         # LambdaRank
    "regression":  MODEL_DIR / "lgbm_regression_v1.pkl",   # 着順回帰
    "lgbm_win":    MODEL_DIR / "lgbm_win_v1.pkl",          # 単勝
}


# =========================================================
# 予測関数
# =========================================================
from train_lgbm import preprocess as lgbm_preprocess, ALL_FEATURES, CAT_FEATURES, TIME_STR_FEATURES
from utils import parse_time_str


def _predict_lgbm_like(df: pd.DataFrame, obj: dict) -> np.ndarray:
    """LightGBM系モデル（binary classification）の予測。"""
    import lightgbm as lgb
    model = obj["model"]
    encoders = obj["encoders"]
    feature_cols = obj["feature_cols"]

    df = df.copy()
    for col in TIME_STR_FEATURES:
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
            df[col] = np.nan
    if isinstance(model, lgb.Booster):
        return model.predict(df[feature_cols])
    return model.predict_proba(df[feature_cols])[:, 1]


def _predict_catboost_like(df: pd.DataFrame, obj: dict) -> np.ndarray:
    """CatBoost系モデル（binary classification）の予測。"""
    from catboost import Pool
    model = obj["model"]
    feature_cols = obj["feature_cols"]
    cat_list = [
        "種牡馬","父タイプ名","母父馬","母父タイプ名","毛色",
        "馬主(最新/仮想)","生産者","芝・ダ","コース区分","芝(内・外)",
        "馬場状態","天気","クラス名","場所","性別","斤量",
        "ブリンカー","重量種別","年齢限定","限定","性別限定","指定条件",
        "前走場所","前芝・ダ","前走馬場状態","前走斤量","前好走",
    ]
    df = df.copy()
    for col in TIME_STR_FEATURES:
        if col in df.columns:
            df[col] = parse_time_str(df[col])
    for col in cat_list:
        df[col] = df[col].fillna("__NaN__").astype(str) if col in df.columns else "__NaN__"
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan
    cat_idx = [i for i, c in enumerate(feature_cols) if c in cat_list]
    pool = Pool(df[feature_cols], cat_features=cat_idx)
    return model.predict_proba(pool)[:, 1]


def _predict_catboost_rank(df: pd.DataFrame, obj: dict) -> np.ndarray:
    """CatBoost YetiRankの予測 → レース内min-max正規化。"""
    from catboost import Pool
    model = obj["model"]
    feature_cols = obj["feature_cols"]
    cat_list = [
        "種牡馬","父タイプ名","母父馬","母父タイプ名","毛色",
        "馬主(最新/仮想)","生産者","芝・ダ","コース区分","芝(内・外)",
        "馬場状態","天気","クラス名","場所","性別","斤量",
        "ブリンカー","重量種別","年齢限定","限定","性別限定","指定条件",
        "前走場所","前芝・ダ","前走馬場状態","前走斤量","前好走",
    ]
    df = df.copy()
    for col in TIME_STR_FEATURES:
        if col in df.columns:
            df[col] = parse_time_str(df[col])
    for col in cat_list:
        df[col] = df[col].fillna("__NaN__").astype(str) if col in df.columns else "__NaN__"
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan
    cat_idx = [i for i, c in enumerate(feature_cols) if c in cat_list]
    pool = Pool(df[feature_cols], cat_features=cat_idx)
    scores = model.predict(pool)
    # レース内min-max正規化
    result = np.full(len(df), 0.5)
    for race_id in df[COL_RACE_ID].unique():
        mask = (df[COL_RACE_ID] == race_id).values
        s = scores[mask]
        s_min, s_max = s.min(), s.max()
        if s_max > s_min:
            result[mask] = (s - s_min) / (s_max - s_min)
    return result


def _predict_lgbm_rank(df: pd.DataFrame, obj: dict) -> np.ndarray:
    """LightGBM LambdaRankの予測 → レース内min-max正規化。"""
    model = obj["model"]
    encoders = obj["encoders"]
    feature_cols = obj["feature_cols"]

    df = df.copy()
    for col in TIME_STR_FEATURES:
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
            df[col] = np.nan

    scores = model.predict(df[feature_cols])
    # レース内min-max正規化
    result = np.full(len(df), 0.5)
    for race_id in df[COL_RACE_ID].unique():
        mask = (df[COL_RACE_ID] == race_id).values
        s = scores[mask]
        s_min, s_max = s.min(), s.max()
        if s_max > s_min:
            result[mask] = (s - s_min) / (s_max - s_min)
    return result


def _predict_regression(df: pd.DataFrame, obj: dict) -> np.ndarray:
    """回帰モデル予測 → 着順を反転して[0,1]に正規化。"""
    model = obj["model"]
    encoders = obj["encoders"]
    feature_cols = obj["feature_cols"]

    df = df.copy()
    for col in TIME_STR_FEATURES:
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
            df[col] = np.nan

    pred_pos = model.predict(df[feature_cols])
    # レース内: 予測着順を反転して正規化（小さい着順 → 高いスコア）
    result = np.full(len(df), 0.5)
    for race_id in df[COL_RACE_ID].unique():
        mask = (df[COL_RACE_ID] == race_id).values
        s = -pred_pos[mask]  # 着順を反転
        s_min, s_max = s.min(), s.max()
        if s_max > s_min:
            result[mask] = (s - s_min) / (s_max - s_min)
    return result


# モデル名 → 予測関数マッピング
PREDICT_FN = {
    "lgbm":       _predict_lgbm_like,
    "catboost":   _predict_catboost_like,
    "fuku_lgbm":  _predict_lgbm_like,
    "fuku_cat":   _predict_catboost_like,
    "rank_cat":   _predict_catboost_rank,
    "rank_lgbm":  _predict_lgbm_rank,
    "regression": _predict_regression,
    "lgbm_win":   _predict_lgbm_like,
}


# =========================================================
# データ準備
# =========================================================
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validation & Test データを読み込み。"""
    logger.info(f"マスターCSV読み込み: {MASTER_CSV}")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    logger.info(f"  {len(df):,}行 × {len(df.columns)}列")

    valid = df[df["split"] == "valid"].copy()
    test  = df[df["split"] == "test"].copy()
    logger.info(f"Valid={len(valid):,} / Test={len(test):,}")
    return valid, test


def generate_predictions(
    df: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """全モデルの予測を生成。存在するモデルのみ。"""
    preds = {}
    available = []

    for name, path in MODELS.items():
        if not path.exists():
            logger.warning(f"  {name}: モデルファイルなし ({path.name})")
            continue
        try:
            obj = joblib.load(path)
            fn = PREDICT_FN[name]
            p = fn(df, obj)
            preds[name] = p
            available.append(name)
            logger.info(f"  {name}: OK (mean={p.mean():.4f}, std={p.std():.4f})")
        except Exception as e:
            logger.warning(f"  {name}: 予測失敗 - {e}")

    return preds, available


# =========================================================
# Nelder-Mead最適化
# =========================================================
def optimize_nelder_mead(
    preds: dict[str, np.ndarray],
    y: np.ndarray,
    available: list[str],
    metric: str = "auc",
    n_restarts: int = 10,
) -> tuple[dict[str, float], float]:
    """
    Nelder-Mead法でアンサンブル重みを最適化。
    制約: 全重み >= 0, 合計 = 1
    """
    n_models = len(available)
    pred_matrix = np.column_stack([preds[name] for name in available])

    def objective(w_raw):
        """softmax変換で制約を自動的に満たす。"""
        w = np.exp(w_raw) / np.exp(w_raw).sum()
        ensemble = pred_matrix @ w
        if metric == "auc":
            return -roc_auc_score(y, ensemble)
        elif metric == "pr_auc":
            return -average_precision_score(y, ensemble)
        else:
            # AUC + PR-AUC 複合
            auc = roc_auc_score(y, ensemble)
            pr  = average_precision_score(y, ensemble)
            return -(0.7 * auc + 0.3 * pr)

    best_score = float("inf")
    best_w = None

    rng = np.random.RandomState(42)

    for i in range(n_restarts):
        if i == 0:
            # 初期値: 均等重み
            w0 = np.zeros(n_models)
        else:
            # ランダム初期値
            w0 = rng.randn(n_models) * 0.5

        result = minimize(objective, w0, method="Nelder-Mead",
                         options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-8})

        if result.fun < best_score:
            best_score = result.fun
            best_w = result.x

    # softmax変換で最終重みを算出
    final_w = np.exp(best_w) / np.exp(best_w).sum()

    weights = {name: float(round(w, 4)) for name, w in zip(available, final_w)}
    return weights, -best_score


# =========================================================
# 検証
# =========================================================
def evaluate_weights(
    preds: dict[str, np.ndarray],
    y: np.ndarray,
    weights: dict[str, float],
    label: str,
) -> dict[str, float]:
    """指定重みでのアンサンブルを評価。"""
    ensemble = np.zeros(len(y))
    for name, w in weights.items():
        if name in preds:
            ensemble += w * preds[name]

    auc = roc_auc_score(y, ensemble)
    pr_auc = average_precision_score(y, ensemble)
    logger.info(f"[{label}] AUC={auc:.4f}  PR-AUC={pr_auc:.4f}")
    return {"auc": auc, "pr_auc": pr_auc}


# =========================================================
# main
# =========================================================
def main():
    valid, test = load_data()

    logger.info("--- Validation 予測生成 ---")
    valid_preds, available = generate_predictions(valid)

    if len(available) < 2:
        logger.error(f"利用可能モデルが{len(available)}個しかありません。最低2個必要。")
        return

    logger.info(f"\n利用可能モデル ({len(available)}個): {available}")
    y_valid = valid[TARGET].values

    # 個別モデルのAUC
    print("\n=== 個別モデルAUC (Validation) ===")
    for name in available:
        auc = roc_auc_score(y_valid, valid_preds[name])
        pr  = average_precision_score(y_valid, valid_preds[name])
        print(f"  {name:15s}: AUC={auc:.4f}  PR-AUC={pr:.4f}")

    # 均等重みのベースライン
    print("\n=== 均等重みベースライン ===")
    equal_w = {name: 1.0 / len(available) for name in available}
    eval_equal = evaluate_weights(valid_preds, y_valid, equal_w, "Valid-Equal")

    # Nelder-Mead最適化
    print("\n=== Nelder-Mead最適化 (AUC目的) ===")
    opt_weights_auc, opt_score_auc = optimize_nelder_mead(
        valid_preds, y_valid, available, metric="auc", n_restarts=20
    )
    print(f"  最適AUC: {opt_score_auc:.4f}")
    print(f"  重み: {json.dumps(opt_weights_auc, indent=2)}")

    # 複合目的（AUC + PR-AUC）
    print("\n=== Nelder-Mead最適化 (AUC+PR-AUC複合) ===")
    opt_weights_combo, opt_score_combo = optimize_nelder_mead(
        valid_preds, y_valid, available, metric="combo", n_restarts=20
    )
    print(f"  最適スコア: {opt_score_combo:.4f}")
    print(f"  重み: {json.dumps(opt_weights_combo, indent=2)}")

    # Testセットで検証（過学習チェック）
    print("\n=== Testセット検証 ===")
    logger.info("--- Test 予測生成 ---")
    test_preds, _ = generate_predictions(test)
    y_test = test[TARGET].values

    eval_test_auc = evaluate_weights(test_preds, y_test, opt_weights_auc, "Test-AUC-opt")
    eval_test_combo = evaluate_weights(test_preds, y_test, opt_weights_combo, "Test-Combo-opt")
    eval_test_equal = evaluate_weights(test_preds, y_test, equal_w, "Test-Equal")

    # Valid-Test gap チェック
    gap_auc   = abs(opt_score_auc - eval_test_auc["auc"])
    gap_combo = abs(opt_score_combo - eval_test_combo["auc"])
    print(f"\n  AUC最適化 Valid-Test gap: {gap_auc:.4f} {'OK' if gap_auc < 0.01 else 'WARN'}")
    print(f"  複合最適化 Valid-Test gap: {gap_combo:.4f} {'OK' if gap_combo < 0.01 else 'WARN'}")

    # 最良の重みセットを選択（AUC最適化を優先、gapが大きければ複合を選択）
    if gap_auc < 0.01:
        best_weights = opt_weights_auc
        best_method = "auc"
    elif gap_combo < 0.01:
        best_weights = opt_weights_combo
        best_method = "combo"
    else:
        # 両方gapが大きい場合は均等重みにフォールバック
        best_weights = equal_w
        best_method = "equal"
        logger.warning("最適化重みのValid-Testが0.01超。均等重みにフォールバック。")

    # 保存
    output = {
        "method": best_method,
        "weights": best_weights,
        "valid_auc": float(round(evaluate_weights(valid_preds, y_valid, best_weights, "Final-Valid")["auc"], 4)),
        "test_auc":  float(round(evaluate_weights(test_preds, y_test, best_weights, "Final-Test")["auc"], 4)),
        "available_models": available,
        "all_results": {
            "equal":   {"weights": equal_w, "valid_auc": eval_equal["auc"], "test_auc": eval_test_equal["auc"]},
            "auc_opt": {"weights": opt_weights_auc, "valid_auc": opt_score_auc, "test_auc": eval_test_auc["auc"]},
            "combo_opt": {"weights": opt_weights_combo, "valid_auc": opt_score_combo, "test_auc": eval_test_combo["auc"]},
        }
    }

    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(WEIGHTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"重み保存: {WEIGHTS_PATH}")

    # サマリ
    print("\n" + "=" * 60)
    print("アンサンブル重み最適化 完了サマリ")
    print("=" * 60)
    print(f"選択手法    : {best_method}")
    print(f"Valid AUC   : {output['valid_auc']:.4f}")
    print(f"Test AUC    : {output['test_auc']:.4f}")
    print(f"V-T gap     : {abs(output['valid_auc'] - output['test_auc']):.4f}")
    print(f"モデル数    : {len(available)}")
    print(f"保存先      : {WEIGHTS_PATH}")
    print("\n最終重み:")
    for name, w in sorted(best_weights.items(), key=lambda x: -x[1]):
        bar = "#" * int(w * 40)
        print(f"  {name:15s}: {w:.4f} {bar}")


# =========================================================
# Phase 5+: Expert別の最適化
# =========================================================
EXPERT_FILTERS = {
    "turf_short": {"td": "芝", "dist_max": 1400},
    "turf_mid":   {"td": "芝", "dist_min": 1600, "dist_max": 2200},
    "turf_long":  {"td": "芝", "dist_min": 2400},
    "dirt":       {"td": "ダ"},
}


def _filter_expert(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    m = pd.Series(True, index=df.index)
    if "td" in cfg:
        m &= (df["芝・ダ"] == cfg["td"])
    if "dist_min" in cfg:
        m &= (pd.to_numeric(df["距離"], errors="coerce") >= cfg["dist_min"])
    if "dist_max" in cfg:
        m &= (pd.to_numeric(df["距離"], errors="coerce") <= cfg["dist_max"])
    return df[m].copy()


def optimize_for_expert(expert_name: str, valid: pd.DataFrame, test: pd.DataFrame):
    """Expert別に valid/test をフィルタして重み最適化。Expert モデルが存在すれば候補に追加。"""
    global MODELS
    cfg = EXPERT_FILTERS[expert_name]
    v = _filter_expert(valid, cfg)
    t = _filter_expert(test, cfg)
    logger.info(f"\n========= Expert: {expert_name} =========")
    logger.info(f"  Valid={len(v):,} / Test={len(t):,}")
    if len(v) < 5000 or len(t) < 1000:
        logger.warning(f"  サンプル不足、スキップ")
        return

    # Expert モデルを候補に追加
    expert_path = MODEL_DIR / f"expert_{expert_name}.pkl"
    models_local = dict(MODELS)
    if expert_path.exists():
        models_local[f"expert_{expert_name}"] = expert_path
        PREDICT_FN[f"expert_{expert_name}"] = _predict_lgbm_like

    # 一時的に MODELS を差し替えて generate_predictions を再利用
    saved = MODELS
    MODELS = models_local
    try:
        valid_preds, available = generate_predictions(v)
        test_preds, _ = generate_predictions(t)
    finally:
        MODELS = saved

    if len(available) < 2:
        logger.warning(f"  利用可能モデル不足({len(available)})、スキップ")
        return

    y_valid = v[TARGET].values
    y_test  = t[TARGET].values

    opt_w, opt_score = optimize_nelder_mead(valid_preds, y_valid, available, metric="auc", n_restarts=15)
    test_eval = evaluate_weights(test_preds, y_test, opt_w, f"Test-{expert_name}")
    gap = abs(opt_score - test_eval["auc"])

    if gap >= 0.015:
        logger.warning(f"  V-T gap={gap:.4f} 大、均等重みにフォールバック")
        opt_w = {n: 1.0 / len(available) for n in available}
        opt_score = roc_auc_score(y_valid, np.column_stack([valid_preds[n] for n in available]) @
                                  np.array([opt_w[n] for n in available]))
        test_eval = evaluate_weights(test_preds, y_test, opt_w, f"Test-{expert_name}-equal")

    out_path = MODEL_DIR / f"ensemble_weights_{expert_name}.json"
    output = {
        "method": "auc",
        "expert": expert_name,
        "weights": opt_w,
        "valid_auc": float(round(opt_score, 4)),
        "test_auc": float(round(test_eval["auc"], 4)),
        "n_valid": int(len(v)),
        "n_test": int(len(t)),
        "available_models": available,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"  保存: {out_path.name}  Valid={opt_score:.4f}  Test={test_eval['auc']:.4f}")


def main_experts():
    """全Expertについて重み最適化を実行。"""
    valid, test = load_data()
    for name in EXPERT_FILTERS.keys():
        try:
            optimize_for_expert(name, valid, test)
        except Exception as e:
            logger.error(f"Expert {name} 失敗: {e}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--experts":
        main_experts()
    else:
        main()
