"""
ensemble.py
PyCaLiAI - 3モデル加重平均アンサンブル

LightGBM / CatBoost / Transformer の予測確率を
ValidAUCベースの重みで加重平均して最終予測を生成する。

Usage:
    python ensemble.py
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder

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

MASTER_CSV        = DATA_DIR  / "master_20130105-20251228.csv"
LGBM_MODEL_PATH   = MODEL_DIR / "lgbm_optuna_v1.pkl"
CAT_MODEL_PATH    = MODEL_DIR / "catboost_optuna_v1.pkl"
TORCH_MODEL_PATH  = MODEL_DIR / "transformer_optuna_v1.pkl"
ENSEMBLE_OUT      = REPORT_DIR / "ensemble_predictions.csv"

TARGET       = "fukusho_flag"
COL_RACE_ID  = "レースID(新/馬番無)"
RANDOM_STATE = 42

# =========================================================
# Valid AUCベースの重み（単体モデルの結果から設定）
# =========================================================
MODEL_WEIGHTS = {
    "lgbm":        0.7425,   # Valid AUC
    "catboost":    0.7472,
    "transformer": 0.7496,
}

# 印の閾値（レース内順位ベースで後述）
MARK_THRESHOLDS = {
    "◎": 1,   # 1位
    "◯": 2,   # 2位
    "▲": 3,   # 3位
    "△": 4,   # 4位
    "×": 5,   # 5位
}

# =========================================================
# 各モデルの前処理（train時と同じ処理を再現）
# =========================================================

from utils import parse_time_str


# =========================================================
# LightGBM予測
# =========================================================
def predict_lgbm(df: pd.DataFrame) -> np.ndarray:
    """LightGBMモデルで複勝確率を予測する。"""
    logger.info("LightGBM予測中...")
    if not LGBM_MODEL_PATH.exists():
        raise FileNotFoundError(f"LightGBMモデルが見つかりません: {LGBM_MODEL_PATH}")
    obj      = joblib.load(LGBM_MODEL_PATH)
    model    = obj["model"]
    encoders = obj["encoders"]
    feature_cols = obj["feature_cols"]

    df = df.copy()

    # タイム変換
    for col in ["前走走破タイム", "前走着差タイム"]:
        if col in df.columns:
            df[col] = parse_time_str(df[col])

    # カテゴリEncoding
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

    # 欠損列補完
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    proba = model.predict_proba(df[feature_cols])[:, 1]
    logger.info(f"  LightGBM予測完了: {len(proba):,}件")
    return proba


# =========================================================
# CatBoost予測
# =========================================================
def predict_catboost(df: pd.DataFrame) -> np.ndarray:
    """CatBoostモデルで複勝確率を予測する。"""
    logger.info("CatBoost予測中...")
    if not CAT_MODEL_PATH.exists():
        raise FileNotFoundError(f"CatBoostモデルが見つかりません: {CAT_MODEL_PATH}")
    obj          = joblib.load(CAT_MODEL_PATH)
    model        = obj["model"]
    feature_cols = obj["feature_cols"]

    # train_catboost.pyと同じCAT_FEATURES
    cat_features_list = [
        "種牡馬", "父タイプ名", "母父馬", "母父タイプ名", "毛色",
        "馬主(最新/仮想)", "生産者",
        "芝・ダ", "コース区分", "芝(内・外)", "馬場状態", "天気",
        "クラス名", "場所",
        "性別", "斤量", "ブリンカー", "重量種別",
        "年齢限定", "限定", "性別限定", "指定条件",
        "前走場所", "前芝・ダ", "前走馬場状態", "前走斤量", "前好走",
    ]

    df = df.copy()

    # タイム変換
    for col in ["前走走破タイム", "前走着差タイム"]:
        if col in df.columns:
            df[col] = parse_time_str(df[col])

    # カテゴリ列処理
    for col in cat_features_list:
        if col in df.columns:
            df[col] = df[col].fillna("__NaN__").astype(str)
        else:
            df[col] = "__NaN__"

    # 数値列処理
    num_cols = [c for c in feature_cols if c not in cat_features_list]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = 0.0

    # 欠損列補完
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    cat_indices = [
        i for i, c in enumerate(feature_cols)
        if c in cat_features_list
    ]
    pool  = Pool(df[feature_cols], cat_features=cat_indices)
    proba = model.predict_proba(pool)[:, 1]
    logger.info(f"  CatBoost予測完了: {len(proba):,}件")
    return proba


# =========================================================
# Transformer予測
# =========================================================
def predict_transformer(df: pd.DataFrame) -> np.ndarray:
    """Transformerモデルで複勝確率を予測する。"""
    logger.info("Transformer予測中...")

    # train_transformer.pyと同じimport
    from train_transformer import (
        RaceTransformer, RaceDataset,
        CAT_FEATURES as TORCH_CAT,
        NUM_FEATURES as TORCH_NUM,
        TIME_STR_FEATURES as TORCH_TIME,
        MAX_HORSES, DEVICE,
        preprocess as torch_preprocess,
    )
    from torch.utils.data import DataLoader

    if not TORCH_MODEL_PATH.exists():
        raise FileNotFoundError(f"Transformerモデルが見つかりません: {TORCH_MODEL_PATH}")
    obj          = joblib.load(TORCH_MODEL_PATH)
    model_state  = obj["model_state"]
    model_config = obj["model_config"]
    encoders     = obj["encoders"]
    num_stats    = obj["num_stats"]
    num_cols     = obj["num_cols"]
    cat_cols     = obj["cat_cols"]

    df = df.copy()
    df, _, _ = torch_preprocess(
        df, encoders=encoders, fit=False, num_stats=num_stats
    )

    # モデル復元
    model = RaceTransformer(
        cat_vocab_sizes=model_config["cat_vocab_sizes"],
        cat_cols=model_config["cat_cols"],
        n_num=model_config["n_num"],
        d_model=model_config.get("d_model", 128),
        n_heads=model_config.get("n_heads", 4),
        n_layers=model_config.get("n_layers", 2),
        d_ff=model_config.get("d_ff", 256),
        dropout=model_config.get("dropout", 0.1),
    ).to(DEVICE)
    model.load_state_dict(model_state)
    model.eval()

    cat_vocab_sizes = model_config["cat_vocab_sizes"]
    ds     = RaceDataset(df, cat_cols, num_cols, cat_vocab_sizes)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

    all_proba = []
    with torch.no_grad():
        for batch in loader:
            cat    = batch["cat"].to(DEVICE)
            num    = batch["num"].to(DEVICE)
            mask   = batch["mask"].to(DEVICE)
            logits = model(cat, num, mask)
            probas = torch.sigmoid(logits).cpu().numpy()  # [B, MAX_HORSES]
            valid  = ~batch["mask"].numpy()               # [B, MAX_HORSES]
            for b in range(len(probas)):
                for h in range(MAX_HORSES):
                    if valid[b, h]:
                        all_proba.append(probas[b, h])

    # dfの行順に対応させる
    # RaceDatasetはgroupbyの順序で処理するため
    # dfをレースID順にソートして対応
    df_sorted = df.sort_values(COL_RACE_ID).reset_index(drop=True)
    result    = np.zeros(len(df))

    idx = 0
    for _, group in df_sorted.groupby(COL_RACE_ID, sort=True):
        n = min(len(group), MAX_HORSES)
        for i, orig_idx in enumerate(group.index[:n]):
            if idx < len(all_proba):
                result[orig_idx] = all_proba[idx]
                idx += 1

    logger.info(f"  Transformer予測完了: {len(result):,}件")
    return result


# =========================================================
# アンサンブル
# =========================================================
def ensemble_predict(
    proba_lgbm: np.ndarray,
    proba_cat:  np.ndarray,
    proba_torch: np.ndarray,
) -> np.ndarray:
    """ValidAUCベースの加重平均でアンサンブル予測を生成する。"""
    w = MODEL_WEIGHTS
    total = sum(w.values())
    weights = {k: v / total for k, v in w.items()}

    logger.info(
        f"重み: LGBM={weights['lgbm']:.3f} / "
        f"CatBoost={weights['catboost']:.3f} / "
        f"Transformer={weights['transformer']:.3f}"
    )

    proba = (
        weights["lgbm"]        * proba_lgbm  +
        weights["catboost"]    * proba_cat   +
        weights["transformer"] * proba_torch
    )
    return proba


# =========================================================
# 印の付与（レース内順位ベース）
# =========================================================
def assign_marks(df: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    """
    レース内での確率順位に基づいて印を付与する。
    同一レース内で1位=◎、2位=◯、3位=▲、4位=△、5位=×
    """
    df = df.copy()
    df["ensemble_prob"] = proba
    df["mark"] = ""

    for _, group in df.groupby(COL_RACE_ID, sort=False):
        ranked = group["ensemble_prob"].rank(ascending=False, method="first")
        for idx, rank in ranked.items():
            for mark, threshold in MARK_THRESHOLDS.items():
                if rank <= threshold:
                    df.at[idx, "mark"] = mark
                    break

    return df


# =========================================================
# 評価
# =========================================================
def evaluate(
    proba: np.ndarray,
    y: np.ndarray,
    split_name: str,
) -> dict[str, float]:

    auc    = roc_auc_score(y, proba)
    pr_auc = average_precision_score(y, proba)
    pred   = (proba >= 0.5).astype(int)

    logger.info(f"[{split_name}] AUC={auc:.4f}  PR-AUC={pr_auc:.4f}")
    print(f"\n=== {split_name} 分類レポート ===")
    print(classification_report(y, pred, target_names=["圏外", "複勝内"]))

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y, proba, ax=ax, name=split_name)
    ax.set_title(f"ROC曲線 [{split_name}] Ensemble")
    path = REPORT_DIR / f"ensemble_roc_{split_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"ROC曲線保存: {path}")

    return {"auc": auc, "pr_auc": pr_auc}


# =========================================================
# 印の的中率集計
# =========================================================
def mark_accuracy(df: pd.DataFrame) -> None:
    """印ごとの実複勝率を集計して表示する。"""
    print("\n=== 印別 実複勝率 ===")
    print(f"{'印':<4} {'予測件数':>8} {'複勝内':>8} {'実複勝率':>8}")
    print("-" * 34)
    for mark in ["◎", "◯", "▲", "△", "×"]:
        subset = df[df["mark"] == mark]
        if len(subset) == 0:
            continue
        n_hit  = subset[TARGET].sum()
        rate   = n_hit / len(subset)
        print(f"{mark:<4} {len(subset):>8,} {int(n_hit):>8,} {rate:>8.3f}")

    # 印なし
    subset = df[df["mark"] == ""]
    if len(subset) > 0:
        n_hit = subset[TARGET].sum()
        rate  = n_hit / len(subset)
        print(f"{'なし':<4} {len(subset):>8,} {int(n_hit):>8,} {rate:>8.3f}")


# =========================================================
# main
# =========================================================
def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- データロード ----
    logger.info(f"マスターCSV読み込み: {MASTER_CSV}")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    logger.info(f"  {len(df):,}行 × {len(df.columns)}列")

    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)
    logger.info(f"Testデータ: {len(test_df):,}行")

    # ---- 各モデルで予測 ----
    proba_lgbm  = predict_lgbm(test_df)
    proba_cat   = predict_catboost(test_df)
    proba_torch = predict_transformer(test_df)

    # ---- アンサンブル ----
    proba_ensemble = ensemble_predict(proba_lgbm, proba_cat, proba_torch)

    # ---- 評価 ----
    y = test_df[TARGET].astype(int).values

    print("\n" + "=" * 50)
    print("単体モデル vs アンサンブル 比較")
    print("=" * 50)
    for name, proba in [
        ("LightGBM",    proba_lgbm),
        ("CatBoost",    proba_cat),
        ("Transformer", proba_torch),
        ("Ensemble",    proba_ensemble),
    ]:
        auc    = roc_auc_score(y, proba)
        pr_auc = average_precision_score(y, proba)
        print(f"{name:<12} AUC={auc:.4f}  PR-AUC={pr_auc:.4f}")

    metrics = evaluate(proba_ensemble, y, "Test_Ensemble")

    # ---- 印付与・的中率 ----
    test_df = assign_marks(test_df, proba_ensemble)
    mark_accuracy(test_df)

    # ---- 結果保存 ----
    out_cols = [
        "レースID(新/馬番無)", "馬番", "馬名",
        "ensemble_prob", "mark", TARGET,
    ]
    out_cols = [c for c in out_cols if c in test_df.columns]
    test_df[out_cols].to_csv(ENSEMBLE_OUT, index=False, encoding="utf-8-sig")
    logger.info(f"予測結果保存: {ENSEMBLE_OUT}")

    print("\n" + "=" * 50)
    print("アンサンブル完了サマリ")
    print("=" * 50)
    print(f"Test AUC    : {metrics['auc']:.4f}")
    print(f"Test PR-AUC : {metrics['pr_auc']:.4f}")
    print(f"結果保存先  : {ENSEMBLE_OUT}")


if __name__ == "__main__":
    main()