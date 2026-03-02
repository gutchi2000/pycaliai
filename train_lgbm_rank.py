"""
train_lgbm_rank.py
PyCaLiAI - LightGBM LambdaRank版

着順をそのままランキング学習のラベルとして使う。
「1着>2着>3着...」という順位関係を直接学習する。

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
from sklearn.metrics import average_precision_score, roc_auc_score
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

MASTER_CSV  = DATA_DIR / "master_20130105-20251228.csv"
MODEL_PATH  = MODEL_DIR / "lgbm_rank_v1.pkl"

TARGET       = "fukusho_flag"
COL_RANK     = "着順"           # ランキングラベル（小さいほど良い）
COL_RACE_ID  = "レースID(新/馬番無)"
RANDOM_STATE = 42

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
]

TIME_STR_FEATURES = ["前走走破タイム", "前走着差タイム"]
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES + TIME_STR_FEATURES

# =========================================================
# LambdaRankパラメータ
# =========================================================
LGBM_RANK_PARAMS = {
    "objective":          "lambdarank",
    "metric":             "ndcg",
    "ndcg_eval_at":       [3, 5],     # NDCG@3, NDCG@5で評価
    "learning_rate":      0.05,
    "num_leaves":         63,
    "min_child_samples":  50,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "random_state":       RANDOM_STATE,
    "n_estimators":       2000,
    "verbose":            -1,
    "label_gain":         list(range(19)),  # 着順0〜18のgain
}


# =========================================================
# 前処理
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
# ランキングラベル生成
# =========================================================
def make_rank_label(df: pd.DataFrame) -> pd.Series:
    """
    着順をLambdaRankのラベルに変換する。
    LightGBMのlambdarankは「大きいほど良い」ラベルを期待するため
    着順を反転させる。
    1着 → 最大値、最下位着 → 0
    """
    max_finish = df.groupby(COL_RACE_ID)[COL_RANK].transform("max")
    label = (max_finish - df[COL_RANK]).astype(int)
    return label


# =========================================================
# クエリグループ生成（レース単位）
# =========================================================
def make_query_groups(df: pd.DataFrame) -> np.ndarray:
    """
    LightGBM Rankerに渡すgroup配列を生成する。
    各レースの出走頭数の配列。
    例: [16, 18, 14, ...] = 1レース目16頭、2レース目18頭...
    """
    return df.groupby(COL_RACE_ID, sort=False).size().values


# =========================================================
# データロード・分割
# =========================================================
def load_and_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info(f"マスターCSV読み込み: {MASTER_CSV}")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    logger.info(f"  {len(df):,}行 × {len(df.columns)}列")

    # レースID順にソート（クエリグループの整合性のため必須）
    df = df.sort_values([COL_RACE_ID, COL_RANK]).reset_index(drop=True)

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
) -> tuple[LGBMRanker, dict, list[str]]:

    train, encoders = preprocess(train, fit=True)
    valid, _        = preprocess(valid, encoders=encoders, fit=False)

    feature_cols = [c for c in ALL_FEATURES if c in train.columns]
    missing = [c for c in ALL_FEATURES if c not in train.columns]
    if missing:
        logger.warning(f"スキップ: {missing}")
    logger.info(f"使用特徴量数: {len(feature_cols)}")

    # ランキングラベル（着順反転）
    train["rank_label"] = make_rank_label(train)
    valid["rank_label"] = make_rank_label(valid)

    # クエリグループ
    group_tr = make_query_groups(train)
    group_va = make_query_groups(valid)

    X_tr = train[feature_cols]
    y_tr = train["rank_label"]
    X_va = valid[feature_cols]
    y_va = valid["rank_label"]

    model = LGBMRanker(**LGBM_RANK_PARAMS)
    model.fit(
        X_tr, y_tr,
        group=group_tr,
        eval_set=[(X_va, y_va)],
        eval_group=[group_va],
        callbacks=[
            early_stopping(stopping_rounds=50, verbose=False),
            log_evaluation(period=200),
        ],
    )
    logger.info(f"学習完了: best_iteration={model.best_iteration_}")
    return model, encoders, feature_cols


# =========================================================
# 評価（AUCで比較できるようにscore→確率に変換）
# =========================================================
def evaluate(
    model: LGBMRanker,
    df: pd.DataFrame,
    encoders: dict,
    feature_cols: list[str],
    split_name: str,
) -> dict[str, float]:

    df, _ = preprocess(df, encoders=encoders, fit=False)

    X = df[feature_cols]
    y = df[TARGET].astype(int)

    # スコアをsigmoidで確率に変換
    scores = model.predict(X)
    proba  = 1 / (1 + np.exp(-scores / scores.std()))

    auc    = roc_auc_score(y, proba)
    pr_auc = average_precision_score(y, proba)
    logger.info(f"[{split_name}] AUC={auc:.4f}  PR-AUC={pr_auc:.4f}")

    return {"auc": auc, "pr_auc": pr_auc}


# =========================================================
# main
# =========================================================
def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train, valid, test = load_and_split()
    model, encoders, feature_cols = train_model(train, valid)

    metrics_valid = evaluate(model, valid, encoders, feature_cols, "Valid")
    metrics_test  = evaluate(model, test,  encoders, feature_cols, "Test")

    # モデル保存
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "encoders": encoders, "feature_cols": feature_cols},
        MODEL_PATH,
    )
    logger.info(f"モデル保存: {MODEL_PATH}")

    print("\n" + "=" * 50)
    print("LightGBM LambdaRank 学習完了サマリ")
    print("=" * 50)
    print(f"Valid AUC    : {metrics_valid['auc']:.4f}  (旧: 0.7412)")
    print(f"Test  AUC    : {metrics_test['auc']:.4f}  (旧: 0.7474)")
    print(f"Valid PR-AUC : {metrics_valid['pr_auc']:.4f}")
    print(f"Test  PR-AUC : {metrics_test['pr_auc']:.4f}")


if __name__ == "__main__":
    main()