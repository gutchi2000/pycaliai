"""
stacking.py
PyCaLiAI - Stackingアンサンブル

Out-of-Fold (OOF) 予測でリークを防ぎながら
LightGBM / CatBoost / Transformer の予測確率を
メタ特徴量としてメタモデル（LightGBM）で学習する。

構造:
    Level 0: LightGBM / CatBoost / Transformer（既存モデル）
    Level 1: MetaLightGBM（3モデルの予測 + レース条件）

Usage:
    python stacking.py
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import (
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier, Pool
import matplotlib.pyplot as plt

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

MASTER_CSV       = DATA_DIR  / "master_20130105-20251228.csv"
LGBM_MODEL_PATH  = MODEL_DIR / "lgbm_optuna_v1.pkl"
CAT_MODEL_PATH   = MODEL_DIR / "catboost_optuna_v1.pkl"
TORCH_MODEL_PATH = MODEL_DIR / "transformer_optuna_v1.pkl"
META_MODEL_PATH  = MODEL_DIR / "stacking_meta_v1.pkl"

TARGET       = "fukusho_flag"
COL_RACE_ID  = "レースID(新/馬番無)"
RANDOM_STATE = 42
N_FOLDS      = 5

# メタモデルに追加するレース条件特徴量
META_EXTRA_FEATURES = [
    "芝・ダ", "距離", "クラス名", "場所", "馬場状態",
    "出走頭数", "枠番", "馬番",
]

# =========================================================
# 前処理ユーティリティ
# =========================================================
def parse_time_str(series: pd.Series) -> pd.Series:
    def _convert(val: str) -> float | None:
        try:
            parts = str(val).strip().split(".")
            if len(parts) == 3:
                return int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 10
            return float(val)
        except Exception:
            return None
    return series.apply(_convert)


# =========================================================
# LightGBM予測
# =========================================================
def predict_lgbm(df: pd.DataFrame) -> np.ndarray:
    obj          = joblib.load(LGBM_MODEL_PATH)
    model        = obj["model"]
    encoders     = obj["encoders"]
    feature_cols = obj["feature_cols"]

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

    return model.predict_proba(df[feature_cols])[:, 1]


# =========================================================
# CatBoost予測
# =========================================================
def predict_catboost(df: pd.DataFrame) -> np.ndarray:
    obj          = joblib.load(CAT_MODEL_PATH)
    model        = obj["model"]
    feature_cols = obj["feature_cols"]

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
    for col in ["前走走破タイム", "前走着差タイム"]:
        if col in df.columns:
            df[col] = parse_time_str(df[col])

    for col in cat_features_list:
        if col in df.columns:
            df[col] = df[col].fillna("__NaN__").astype(str)
        else:
            df[col] = "__NaN__"

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    cat_indices = [i for i, c in enumerate(feature_cols) if c in cat_features_list]
    pool = Pool(df[feature_cols], cat_features=cat_indices)
    return model.predict_proba(pool)[:, 1]


# =========================================================
# Transformer予測
# =========================================================
def predict_transformer(df: pd.DataFrame) -> np.ndarray:
    import torch
    from train_transformer import RaceTransformer, RaceDataset
    from torch.utils.data import DataLoader

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obj          = joblib.load(TORCH_MODEL_PATH)
    model_state  = obj["model_state"]
    model_config = obj["model_config"]
    encoders     = obj["encoders"]
    num_stats    = obj["num_stats"]
    num_cols     = obj["num_cols"]
    cat_cols     = obj["cat_cols"]

    from train_transformer import (
        CAT_FEATURES as TORCH_CAT,
        NUM_FEATURES as TORCH_NUM,
        TIME_STR_FEATURES as TORCH_TIME,
        MAX_HORSES,
        preprocess as torch_preprocess,
    )

    df = df.copy()
    df, _, _ = torch_preprocess(df, encoders=encoders, fit=False, num_stats=num_stats)

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
            probas = torch.sigmoid(logits).cpu().numpy()
            valid  = ~batch["mask"].numpy()
            for b in range(len(probas)):
                for h in range(MAX_HORSES):
                    if valid[b, h]:
                        all_proba.append(probas[b, h])

    df_sorted = df.sort_values(COL_RACE_ID).reset_index(drop=True)
    result    = np.zeros(len(df))
    idx = 0
    for _, group in df_sorted.groupby(COL_RACE_ID, sort=True):
        n = min(len(group), MAX_HORSES)
        for i, orig_idx in enumerate(group.index[:n]):
            if idx < len(all_proba):
                result[orig_idx] = all_proba[idx]
                idx += 1

    return result


# =========================================================
# メタ特徴量のLabelEncoding
# =========================================================
def encode_meta_features(
    df: pd.DataFrame,
    extra_cols: list[str],
    encoders: dict[str, LabelEncoder] | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    df = df.copy()
    if encoders is None:
        encoders = {}

    str_cols = df[extra_cols].select_dtypes(include="object").columns.tolist()
    for col in str_cols:
        df[col] = df[col].fillna("__NaN__").astype(str)
        if fit:
            le   = LabelEncoder()
            vals = df[col].tolist()
            if "__NaN__" not in vals:
                vals.append("__NaN__")
            le.fit(vals)
            encoders[col] = le
        else:
            le    = encoders[col]
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else "__NaN__")
        df[col] = le.transform(df[col])

    return df, encoders


# =========================================================
# メイン処理
# =========================================================
def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ---- データロード ----
    logger.info(f"マスターCSV読み込み: {MASTER_CSV}")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    logger.info(f"  {len(df):,}行 × {len(df.columns)}列")

    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    valid_df = df[df["split"] == "valid"].copy().reset_index(drop=True)
    test_df  = df[df["split"] == "test"].copy().reset_index(drop=True)

    # ---- Level0: Train全体で予測（Valid/Test用）----
    logger.info("Level0予測: Valid / Test...")
    proba_lgbm_va  = predict_lgbm(valid_df)
    proba_cat_va   = predict_catboost(valid_df)
    proba_torch_va = predict_transformer(valid_df)

    proba_lgbm_te  = predict_lgbm(test_df)
    proba_cat_te   = predict_catboost(test_df)
    proba_torch_te = predict_transformer(test_df)

    # ---- Level0: Train OOF予測（メタモデル学習用）----
    logger.info(f"Level0 OOF予測: {N_FOLDS}分割...")
    oof_lgbm  = np.zeros(len(train_df))
    oof_cat   = np.zeros(len(train_df))
    oof_torch = np.zeros(len(train_df))

    # 時系列順を保持したままKFold（シャッフルなし）
    kf = KFold(n_splits=N_FOLDS, shuffle=False)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train_df)):
        logger.info(f"  Fold {fold + 1}/{N_FOLDS}...")
        fold_va = train_df.iloc[va_idx].copy()

        oof_lgbm[va_idx]  = predict_lgbm(fold_va)
        oof_cat[va_idx]   = predict_catboost(fold_va)
        oof_torch[va_idx] = predict_transformer(fold_va)

    logger.info(f"OOF AUC LGBM      : {roc_auc_score(train_df[TARGET], oof_lgbm):.4f}")
    logger.info(f"OOF AUC CatBoost  : {roc_auc_score(train_df[TARGET], oof_cat):.4f}")
    logger.info(f"OOF AUC Transformer: {roc_auc_score(train_df[TARGET], oof_torch):.4f}")

    # ---- メタ特徴量の構築 ----
    logger.info("メタ特徴量構築...")

    # trainのメタ特徴量（OOF予測 + レース条件）
    meta_train = train_df[META_EXTRA_FEATURES].copy()
    meta_train, meta_encoders = encode_meta_features(
        meta_train, META_EXTRA_FEATURES, fit=True
    )
    meta_train["lgbm"]        = oof_lgbm
    meta_train["catboost"]    = oof_cat
    meta_train["transformer"] = oof_torch

    # validのメタ特徴量
    meta_valid = valid_df[META_EXTRA_FEATURES].copy()
    meta_valid, _ = encode_meta_features(
        meta_valid, META_EXTRA_FEATURES, encoders=meta_encoders, fit=False
    )
    meta_valid["lgbm"]        = proba_lgbm_va
    meta_valid["catboost"]    = proba_cat_va
    meta_valid["transformer"] = proba_torch_va

    # testのメタ特徴量
    meta_test = test_df[META_EXTRA_FEATURES].copy()
    meta_test, _ = encode_meta_features(
        meta_test, META_EXTRA_FEATURES, encoders=meta_encoders, fit=False
    )
    meta_test["lgbm"]        = proba_lgbm_te
    meta_test["catboost"]    = proba_cat_te
    meta_test["transformer"] = proba_torch_te

    meta_cols = META_EXTRA_FEATURES + ["lgbm", "catboost", "transformer"]

    # ---- メタモデル学習 ----
    logger.info("メタモデル学習...")
    y_train = train_df[TARGET]
    y_valid = valid_df[TARGET]
    y_test  = test_df[TARGET]

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()

    meta_model = LGBMClassifier(
        objective="binary",
        metric="auc",
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=1000,
        scale_pos_weight=neg / pos,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    meta_model.fit(
        meta_train[meta_cols], y_train,
        eval_set=[(meta_valid[meta_cols], y_valid)],
        callbacks=[
            early_stopping(stopping_rounds=50, verbose=False),
            log_evaluation(period=100),
        ],
    )

    # ---- 評価 ----
    proba_meta_va = meta_model.predict_proba(meta_valid[meta_cols])[:, 1]
    proba_meta_te = meta_model.predict_proba(meta_test[meta_cols])[:, 1]

    auc_va    = roc_auc_score(y_valid, proba_meta_va)
    auc_te    = roc_auc_score(y_test,  proba_meta_te)
    pr_auc_te = average_precision_score(y_test, proba_meta_te)

    logger.info(f"[Valid] AUC={auc_va:.4f}")
    logger.info(f"[Test]  AUC={auc_te:.4f}  PR-AUC={pr_auc_te:.4f}")

    print(f"\n=== Test 分類レポート ===")
    pred = (proba_meta_te >= 0.5).astype(int)
    print(classification_report(y_test, pred, target_names=["圏外", "複勝内"]))

    # ROC曲線
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, proba_meta_te, ax=ax, name="Stacking")
    ax.set_title("ROC曲線 [Test] Stacking")
    fig.savefig(REPORT_DIR / "stacking_roc_test.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 印別的中率
    test_df["stack_prob"] = proba_meta_te
    test_df["mark"]       = ""
    for race_id, group in test_df.groupby(COL_RACE_ID, sort=False):
        ranked = group["stack_prob"].rank(ascending=False, method="first")
        marks  = {1: "◎", 2: "◯", 3: "▲", 4: "△", 5: "×"}
        for idx, rank in ranked.items():
            if rank <= 5:
                test_df.at[idx, "mark"] = marks[int(rank)]

    print("\n=== 印別 実複勝率 ===")
    print(f"{'印':<4} {'予測件数':>8} {'複勝内':>8} {'実複勝率':>8}")
    print("-" * 34)
    for mark in ["◎", "◯", "▲", "△", "×"]:
        sub  = test_df[test_df["mark"] == mark]
        if len(sub) == 0:
            continue
        hit  = sub[TARGET].sum()
        rate = hit / len(sub)
        print(f"{mark:<4} {len(sub):>8,} {int(hit):>8,} {rate:>8.3f}")

    # ---- 保存 ----
    joblib.dump(
        {
            "meta_model":    meta_model,
            "meta_encoders": meta_encoders,
            "meta_cols":     meta_cols,
        },
        META_MODEL_PATH,
    )
    logger.info(f"メタモデル保存: {META_MODEL_PATH}")

    print("\n" + "=" * 50)
    print("Stacking 完了サマリ")
    print("=" * 50)
    print(f"Valid AUC    : {auc_va:.4f}")
    print(f"Test  AUC    : {auc_te:.4f}  (加重平均: 0.7616)")
    print(f"Test  PR-AUC : {pr_auc_te:.4f}  (加重平均: 0.4826)")
    print(f"メタモデル保存: {META_MODEL_PATH}")


if __name__ == "__main__":
    main()