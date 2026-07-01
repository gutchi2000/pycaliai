"""
train_order_model.py
PyCaLiAI - 着順予測モデル（3クラス: 1着 / 2-3着 / 4着以下）

三連単フォーメーション自動選択のために、各馬の着順分布を予測する。
出力: p_win(1着確率), p_place23(2-3着確率), p_out(4着以下確率)

Usage:
    python train_order_model.py
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import classification_report, log_loss
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    pass

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

MASTER_CSV      = DATA_DIR / "master_kako5.csv"
MASTER_CSV_ORIG = DATA_DIR / "master_20130105-20251228.csv"
MODEL_PATH      = MODEL_DIR / "order_model_v1.pkl"

COL_DATE    = "日付"
COL_RACE_ID = "レースID(新)"
RANDOM_STATE = 42

# =========================================================
# 特徴量定義（train_lgbm.py と同一）
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
    "jockey_fuku30", "jockey_fuku90",
    "trainer_fuku30", "trainer_fuku90",
    "horse_fuku10", "horse_fuku30",
    "prev_pos_rel", "closing_power",
    "前走補9", "前走補正",
    "trn_hanro_4f", "trn_hanro_3f", "trn_hanro_2f", "trn_hanro_1f",
    "trn_hanro_lap1", "trn_hanro_lap2", "trn_hanro_lap3", "trn_hanro_lap4",
    "trn_hanro_days",
    "trn_wc_5f", "trn_wc_4f", "trn_wc_3f",
    "trn_wc_lap1", "trn_wc_lap2", "trn_wc_lap3",
    "trn_wc_days",
    "前走単勝オッズ",
    "kako5_avg_pos", "kako5_std_pos", "kako5_best_pos",
    "kako5_avg_agari3f", "kako5_best_agari3f",
    "kako5_same_td_ratio", "kako5_same_dist_ratio", "kako5_same_place_ratio",
    "kako5_pos_trend", "kako5_race_count",
    "kako5_expected_good_count", "kako5_upset_good_count", "kako5_hidden_good_count",
    "kako5_same_cond_best_pos",
    "hist_same_cond_best_pos", "hist_same_cond_top3_rate", "hist_same_cond_count",
    "hist_same_place_best_pos",
]

TIME_STR_FEATURES = ["前走走破タイム", "前走着差タイム"]
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES + TIME_STR_FEATURES

# =========================================================
# LightGBMパラメータ（multiclass softmax）
# =========================================================
LGBM_PARAMS = {
    "objective":          "multiclass",
    "num_class":          3,
    "metric":             "multi_logloss",
    "learning_rate":      0.05,
    "num_leaves":         63,
    "min_child_samples":  50,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "random_state":       RANDOM_STATE,
    "n_estimators":       2000,
    "verbose":            -1,
}


# =========================================================
# 前処理
# =========================================================
from utils import parse_time_str


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
            df[col] = np.nan
            continue
        df[col] = df[col].astype(str).fillna("__NaN__")
        if fit:
            le = LabelEncoder()
            le.fit(df[col])
            encoders[col] = le
        else:
            le = encoders[col]
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else "__unknown__")
            if "__unknown__" not in le.classes_:
                le.classes_ = np.append(le.classes_, "__unknown__")
        df[col] = le.transform(df[col])
    return df, encoders


def preprocess(
    df: pd.DataFrame,
    encoders: dict[str, LabelEncoder] | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    df = df.copy()
    for col in TIME_STR_FEATURES:
        if col in df.columns:
            df[col] = parse_time_str(df[col])
    df, encoders = encode_categoricals(df, CAT_FEATURES, encoders, fit=fit)
    return df, encoders


def make_target(df: pd.DataFrame) -> pd.Series:
    """確定着順から3クラスターゲットを生成。
    0 = 1着, 1 = 2-3着, 2 = 4着以下
    """
    pos = pd.to_numeric(df["着順"], errors="coerce")
    target = pd.Series(2, index=df.index)  # default: 4着以下
    target[pos == 1] = 0
    target[(pos >= 2) & (pos <= 3)] = 1
    return target.astype(int)


# =========================================================
# データロード・分割
# =========================================================
def load_and_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    csv_path = MASTER_CSV if MASTER_CSV.exists() else MASTER_CSV_ORIG
    logger.info(f"マスターCSV読み込み: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    logger.info(f"  {len(df):,}行 × {len(df.columns)}列")

    # 確定着順が欠損の行は除外
    df = df[pd.to_numeric(df["着順"], errors="coerce").notna()].copy()
    logger.info(f"  着順有効行: {len(df):,}")

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
) -> tuple[LGBMClassifier, dict[str, LabelEncoder], list[str]]:
    train, encoders = preprocess(train, fit=True)
    valid, _        = preprocess(valid, encoders=encoders, fit=False)

    feature_cols = [c for c in ALL_FEATURES if c in train.columns]
    missing = [c for c in ALL_FEATURES if c not in train.columns]
    if missing:
        logger.warning(f"特徴量が存在しないためスキップ: {missing}")
    logger.info(f"使用特徴量数: {len(feature_cols)}")

    y_tr = make_target(train)
    y_va = make_target(valid)

    X_tr = train[feature_cols]
    X_va = valid[feature_cols]

    # クラス分布
    for cls, label in [(0, "1着"), (1, "2-3着"), (2, "4着以下")]:
        n = (y_tr == cls).sum()
        logger.info(f"  {label}: {n:,} ({n/len(y_tr)*100:.1f}%)")

    model = LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
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
def evaluate(model: LGBMClassifier, df: pd.DataFrame,
             encoders: dict, feature_cols: list[str], split_name: str) -> None:
    df, _ = preprocess(df, encoders=encoders, fit=False)
    X = df[feature_cols]
    y = make_target(df)

    proba = model.predict_proba(X)
    pred  = model.predict(X)

    ll = log_loss(y, proba)
    logger.info(f"[{split_name}] log_loss: {ll:.4f}")
    print(f"\n=== {split_name} ===")
    print(classification_report(y, pred, target_names=["1着", "2-3着", "4着以下"]))

    # 1着予測の精度（上位N頭の的中率）
    df = df.copy()
    df["p_win"] = proba[:, 0]
    df["y_win"] = (y == 0).astype(int)

    race_col = "レースID(新)"
    if race_col in df.columns:
        hits_top1 = 0
        hits_top2 = 0
        hits_top3 = 0
        n_races = 0
        for _, grp in df.groupby(race_col):
            grp_sorted = grp.sort_values("p_win", ascending=False)
            actual_winner = grp_sorted["y_win"].values
            if actual_winner[0] == 1:
                hits_top1 += 1
            if 1 in actual_winner[:2]:
                hits_top2 += 1
            if 1 in actual_winner[:3]:
                hits_top3 += 1
            n_races += 1
        print(f"1着的中率: top1={hits_top1/n_races*100:.1f}%  "
              f"top2={hits_top2/n_races*100:.1f}%  "
              f"top3={hits_top3/n_races*100:.1f}%  "
              f"({n_races}R)")


# =========================================================
# メイン
# =========================================================
def main() -> None:
    train, valid, test = load_and_split()

    model, encoders, feature_cols = train_model(train, valid)

    # 評価
    evaluate(model, valid, encoders, feature_cols, "Valid")
    evaluate(model, test,  encoders, feature_cols, "Test")

    # 保存
    artifact = {
        "model": model,
        "encoders": encoders,
        "features": feature_cols,
        "target_classes": ["1着", "2-3着", "4着以下"],
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    logger.info(f"モデル保存: {MODEL_PATH}")

    # 特徴量重要度 top20
    imp = pd.Series(model.feature_importances_, index=feature_cols)
    imp = imp.sort_values(ascending=False).head(20)
    print("\n=== 特徴量重要度 Top20 ===")
    for feat, val in imp.items():
        print(f"  {feat:30s} {val:6.0f}")


if __name__ == "__main__":
    main()
