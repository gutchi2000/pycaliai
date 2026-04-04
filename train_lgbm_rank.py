"""
train_lgbm_rank.py
PyCaLiAI - LightGBM LambdaRank ランキング学習モデル

着順をrelevanceスコアに変換し、レース内での順位予測精度を最大化する。
二値分類（fukusho_flag）とは異なり、「何着か」の順序関係を直接学習する。

Usage:
    python train_lgbm_rank.py
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker, early_stopping, log_evaluation
from sklearn.metrics import ndcg_score
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
MODEL_PATH  = MODEL_DIR / "lgbm_rank_v1.pkl"

COL_POS     = "着順"          # ランキング対象（確定着順）
COL_DATE    = "日付"
COL_RACE_ID = "レースID(新/馬番無)"  # レースレベルID（馬番なし16桁）
RANDOM_STATE = 42

# =========================================================
# 特徴量定義（train_lgbm.pyと共通）
# =========================================================
from train_lgbm import CAT_FEATURES, NUM_FEATURES, TIME_STR_FEATURES, ALL_FEATURES
from train_lgbm import preprocess

# =========================================================
# LambdaRankパラメータ
# =========================================================
RANK_PARAMS = {
    "objective":          "lambdarank",
    "metric":             "ndcg",
    "ndcg_eval_at":       [3, 5],       # 複勝圏(top3)を重視
    "learning_rate":      0.05,
    "num_leaves":         63,
    "min_child_samples":  50,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "random_state":       RANDOM_STATE,
    "n_estimators":       2000,
    "verbose":            -1,
    "lambdarank_truncation_level": 10,  # 上位10位まで重視
}


# =========================================================
# 着順 → relevanceスコア変換
# =========================================================
def pos_to_relevance(pos: pd.Series) -> np.ndarray:
    """
    着順をLambdaRank用のrelevanceスコアに変換。
    1着=4, 2着=3, 3着=2, 4-5着=1, 6着以降=0
    """
    pos_num = pd.to_numeric(pos, errors="coerce").fillna(18)
    rel = np.zeros(len(pos_num), dtype=np.float32)
    rel[pos_num == 1] = 4
    rel[pos_num == 2] = 3
    rel[pos_num == 3] = 2
    rel[(pos_num >= 4) & (pos_num <= 5)] = 1
    # 6着以降は0
    return rel


def make_group_sizes(df: pd.DataFrame) -> np.ndarray:
    """レースIDごとのグループサイズ配列を作成。"""
    return df.groupby(COL_RACE_ID).size().values


# =========================================================
# データロード・分割
# =========================================================
def load_and_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """マスターCSVを読み込んでtrain/valid/testに分割。レースID順にソート。"""
    csv_path = MASTER_CSV if MASTER_CSV.exists() else MASTER_CSV_ORIG
    logger.info(f"マスターCSV読み込み: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    logger.info(f"  {len(df):,}行 × {len(df.columns)}列")

    # 着順がNaNのレコードを除外（出走取消等）
    df[COL_POS] = pd.to_numeric(df[COL_POS], errors="coerce")
    before = len(df)
    df = df.dropna(subset=[COL_POS])
    if before > len(df):
        logger.info(f"  着順NaN除外: {before - len(df):,}行")

    # LambdaRankはレースID順にソートされている必要がある
    df = df.sort_values([COL_RACE_ID]).reset_index(drop=True)

    train = df[df["split"] == "train"].copy()
    valid = df[df["split"] == "valid"].copy()
    test  = df[df["split"] == "test"].copy()

    # 各splitもレースID順を保証
    train = train.sort_values(COL_RACE_ID).reset_index(drop=True)
    valid = valid.sort_values(COL_RACE_ID).reset_index(drop=True)
    test  = test.sort_values(COL_RACE_ID).reset_index(drop=True)

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
) -> tuple[LGBMRanker, dict[str, LabelEncoder], list[str]]:
    """LightGBM LambdaRankモデルを学習して返す。"""

    # relevanceスコア作成（前処理前に）
    y_tr_rel = pos_to_relevance(train[COL_POS])
    y_va_rel = pos_to_relevance(valid[COL_POS])

    # グループサイズ（前処理前に）
    group_tr = make_group_sizes(train)
    group_va = make_group_sizes(valid)

    logger.info(f"Relevance分布 (train): "
                f"4={np.sum(y_tr_rel==4):,} / 3={np.sum(y_tr_rel==3):,} / "
                f"2={np.sum(y_tr_rel==2):,} / 1={np.sum(y_tr_rel==1):,} / "
                f"0={np.sum(y_tr_rel==0):,}")
    logger.info(f"グループ数: train={len(group_tr):,} (avg {np.mean(group_tr):.1f}頭/R) / "
                f"valid={len(group_va):,} (avg {np.mean(group_va):.1f}頭/R)")

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

    # LambdaRank学習
    model = LGBMRanker(**RANK_PARAMS)

    model.fit(
        X_tr, y_tr_rel,
        group=group_tr,
        eval_set=[(X_va, y_va_rel)],
        eval_group=[group_va],
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
def evaluate_ranking(
    model: LGBMRanker,
    df: pd.DataFrame,
    encoders: dict[str, LabelEncoder],
    feature_cols: list[str],
    split_name: str,
) -> dict[str, float]:
    """NDCG@3, NDCG@5, MAP@3を算出。"""
    # relevance（前処理前に取得）
    y_rel = pos_to_relevance(df[COL_POS])
    race_ids = df[COL_RACE_ID].values

    # 前処理
    df_proc, _ = preprocess(df, encoders=encoders, fit=False)
    X = df_proc[feature_cols]

    # 予測（スコアが高いほど上位予測）
    scores = model.predict(X)

    # レース単位でNDCG計算
    ndcg3_list = []
    ndcg5_list = []
    top3_hit = 0
    total_races = 0

    unique_races = np.unique(race_ids)
    idx = 0
    for race_id in unique_races:
        mask = race_ids == race_id
        r = y_rel[mask]
        s = scores[mask]

        if len(r) < 2:
            continue

        # NDCG@k（sklearn expects 2D arrays）
        try:
            n3 = ndcg_score([r], [s], k=3)
            n5 = ndcg_score([r], [s], k=5)
            ndcg3_list.append(n3)
            ndcg5_list.append(n5)
        except ValueError:
            continue

        # Top3 hit rate: モデル上位3頭に実際の1着馬が含まれるか
        top3_pred = np.argsort(-s)[:3]
        if np.any(r[top3_pred] == 4):  # relevance==4 は1着
            top3_hit += 1
        total_races += 1

    ndcg3 = np.mean(ndcg3_list) if ndcg3_list else 0.0
    ndcg5 = np.mean(ndcg5_list) if ndcg5_list else 0.0
    top3_rate = top3_hit / total_races if total_races > 0 else 0.0

    # 二値AUC（fukusho_flagとの相関を確認）
    from sklearn.metrics import roc_auc_score
    if "fukusho_flag" in df.columns:
        y_fuku = df["fukusho_flag"].values
        valid_mask = ~np.isnan(y_fuku)
        if valid_mask.sum() > 0:
            auc = roc_auc_score(y_fuku[valid_mask], scores[valid_mask])
        else:
            auc = 0.0
    else:
        auc = 0.0

    logger.info(
        f"[{split_name}] NDCG@3={ndcg3:.4f}  NDCG@5={ndcg5:.4f}  "
        f"Top3Hit={top3_rate:.4f}  BinaryAUC={auc:.4f}"
    )
    return {
        "ndcg3": ndcg3,
        "ndcg5": ndcg5,
        "top3_hit_rate": top3_rate,
        "binary_auc": auc,
    }


# =========================================================
# 特徴量重要度
# =========================================================
def plot_importance(
    model: LGBMRanker,
    feature_cols: list[str],
    top_n: int = 30,
) -> None:
    """特徴量重要度を棒グラフで保存。"""
    imp = model.feature_importances_
    idx = np.argsort(imp)[-top_n:]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(idx)), imp[idx], align="center")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_cols[i] for i in idx])
    ax.set_title("LambdaRank Feature Importance (top 30)")
    ax.set_xlabel("Importance")

    path = REPORT_DIR / "lgbm_rank_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"重要度プロット保存: {path}")


# =========================================================
# モデル保存
# =========================================================
def save_model(
    model: LGBMRanker,
    encoders: dict[str, LabelEncoder],
    feature_cols: list[str],
) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model":        model,
            "encoders":     encoders,
            "feature_cols": feature_cols,
            "model_type":   "lambdarank",
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
    metrics_valid = evaluate_ranking(model, valid, encoders, feature_cols, "Valid")
    metrics_test  = evaluate_ranking(model, test,  encoders, feature_cols, "Test")

    # 重要度
    plot_importance(model, feature_cols)

    # 保存
    save_model(model, encoders, feature_cols)

    # サマリ
    print("\n" + "=" * 50)
    print("LightGBM LambdaRank 学習完了サマリ")
    print("=" * 50)
    print(f"Valid NDCG@3     : {metrics_valid['ndcg3']:.4f}")
    print(f"Test  NDCG@3     : {metrics_test['ndcg3']:.4f}")
    print(f"Valid NDCG@5     : {metrics_valid['ndcg5']:.4f}")
    print(f"Test  NDCG@5     : {metrics_test['ndcg5']:.4f}")
    print(f"Valid Top3Hit    : {metrics_valid['top3_hit_rate']:.4f}")
    print(f"Test  Top3Hit    : {metrics_test['top3_hit_rate']:.4f}")
    print(f"Valid BinaryAUC  : {metrics_valid['binary_auc']:.4f}")
    print(f"Test  BinaryAUC  : {metrics_test['binary_auc']:.4f}")
    print(f"特徴量数          : {len(feature_cols)}")
    print(f"モデル保存先      : {MODEL_PATH}")


if __name__ == "__main__":
    main()
