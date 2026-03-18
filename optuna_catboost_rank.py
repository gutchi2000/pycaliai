"""
optuna_catboost_rank.py
PyCaLiAI - CatBoost YetiRank ランキング学習

二値分類の代わりにレース内順位を直接最適化する。
ラベル: 1位=3, 2位=2, 3位=1, 4位以下=0

Usage:
    python optuna_catboost_rank.py
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import joblib
import optuna
import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool
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

MASTER_CSV   = DATA_DIR / "master_20130105-20251228.csv"
HOSSEI_CSV   = DATA_DIR / "hossei" / "H_20130105-20251228.csv"
KEKKA_CSV    = DATA_DIR / "kekka_20130105-20251228.csv"
HANRO_MASTER = Path(r"E:\競馬過去走データ\H-20150401-20260313.csv")
WC_MASTER    = Path(r"E:\競馬過去走データ\W-20150401-20260313.csv")
MODEL_PATH   = MODEL_DIR / "catboost_rank_v1.pkl"
STUDY_PATH   = REPORT_DIR / "optuna_catboost_rank_study.pkl"

COL_RACE    = "レースID(新/馬番無)"
COL_ORDER   = "着順"
TARGET      = "rank_label"   # グレード付きラベル（0-3）
FUKUSHO     = "fukusho_flag" # AUC比較用

RANDOM_STATE = 42
N_TRIALS     = 50

# =========================================================
# 特徴量定義（optuna_catboost.py と同じ）
# =========================================================
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
    "jockey_fuku30", "jockey_fuku90",
    "trainer_fuku30", "trainer_fuku90",
    "horse_fuku10", "horse_fuku30",
    "prev_pos_rel", "closing_power",
    "前走補9", "前走補正",
    "trn_hanro_4f", "trn_hanro_lap1", "trn_hanro_days",
    "trn_wc_3f", "trn_wc_lap1", "trn_wc_days",
    "前走単勝オッズ",
]

TIME_STR_FEATURES = ["前走走破タイム", "前走着差タイム"]


from utils import parse_time_str, backup_model
from optuna_lgbm import load_chukyo, merge_chukyo


def make_rank_label(order: pd.Series) -> pd.Series:
    """着順 → グレード付きラベル（1位=3, 2位=2, 3位=1, 4位以下=0）"""
    order_num = pd.to_numeric(order, errors="coerce")
    label = pd.Series(0, index=order.index, dtype=int)
    label[order_num == 1] = 3
    label[order_num == 2] = 2
    label[order_num == 3] = 1
    return label


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
    df[TARGET] = make_rank_label(df[COL_ORDER])
    return df


def make_pool(
    df: pd.DataFrame,
    feature_cols: list[str],
    with_label: bool = True,
) -> Pool:
    # YetiRankはgroupが昇順ソートされている必要がある
    df = df.sort_values(COL_RACE).copy()
    group_sizes = df.groupby(COL_RACE, sort=True).size().values
    cat_indices = [
        i for i, c in enumerate(feature_cols)
        if c in CAT_FEATURES_LIST
    ]
    if with_label:
        return Pool(
            df[feature_cols],
            label=df[TARGET].values,
            group_id=df[COL_RACE].values,
            cat_features=cat_indices,
        )
    return Pool(
        df[feature_cols],
        group_id=df[COL_RACE].values,
        cat_features=cat_indices,
    )


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info(f"マスターCSV読み込み: {MASTER_CSV}")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    # hossei JOIN
    if HOSSEI_CSV.exists():
        hossei = pd.read_csv(HOSSEI_CSV, encoding="cp932",
                             usecols=["レースID(新)", "馬番", "前走補9", "前走補正"])
        df = df.merge(hossei, on=["レースID(新)", "馬番"], how="left")
        logger.info(f"hossei JOIN完了: 前走補9カバレッジ={df['前走補9'].notna().mean()*100:.1f}%")
    else:
        df["前走補9"]  = float("nan")
        df["前走補正"] = float("nan")
    # 調教データJOIN
    hanro, wc = load_chukyo()
    df = merge_chukyo(df, hanro, wc)
    # 前走単勝オッズJOIN
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
        kekka["前走単勝オッズ"] = kekka["単勝配当"].apply(_parse_tansho)
        kekka = kekka.drop(columns=["単勝配当"])
        kekka = kekka.rename(columns={"レースID(新)": "前走レースID(新)"})
        kekka["前走レースID(新)"] = kekka["前走レースID(新)"].astype("int64")
        kekka["馬番"] = kekka["馬番"].astype("int64")
        df["前走レースID(新)"] = pd.to_numeric(df["前走レースID(新)"], errors="coerce")
        df = df.merge(kekka, on=["前走レースID(新)", "馬番"], how="left")
        logger.info(f"前走単勝オッズJOIN完了: カバレッジ={df['前走単勝オッズ'].notna().mean()*100:.1f}%")
    else:
        df["前走単勝オッズ"] = float("nan")

    train = preprocess(df[df["split"] == "train"].copy())
    valid = preprocess(df[df["split"] == "valid"].copy())
    test  = preprocess(df[df["split"] == "test"].copy())
    logger.info(
        f"分割: train={len(train):,} / valid={len(valid):,} / test={len(test):,}"
    )
    # ラベル分布確認
    for name, d in [("train", train), ("valid", valid), ("test", test)]:
        dist = d[TARGET].value_counts().sort_index()
        logger.info(f"  {name} ラベル分布: {dist.to_dict()}")
    return train, valid, test


def eval_auc(df: pd.DataFrame, scores: np.ndarray) -> float:
    """予測スコア → fukusho_flag に対するAUC（既存モデルとの比較用）"""
    return roc_auc_score(df[FUKUSHO].astype(int), scores)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train, valid, test = load_data()

    all_features = CAT_FEATURES_LIST + NUM_FEATURES + TIME_STR_FEATURES
    feature_cols = [c for c in all_features if c in train.columns]
    logger.info(f"使用特徴量数: {len(feature_cols)}")

    pool_tr = make_pool(train, feature_cols)
    pool_va = make_pool(valid, feature_cols)

    # valid をソート済みに揃える（eval_auc用）
    valid_sorted = valid.sort_values(COL_RACE).copy()
    test_sorted  = test.sort_values(COL_RACE).copy()
    pool_te = make_pool(test_sorted, feature_cols)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "loss_function":         "YetiRank",
            "eval_metric":           "NDCG",
            "iterations":            500,
            "random_seed":           RANDOM_STATE,
            "verbose":               0,
            "early_stopping_rounds": 50,
            "task_type":             "GPU",
            "learning_rate":         trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "depth":                 trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg":           trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
            "random_strength":       trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
            "border_count":          trial.suggest_int("border_count", 32, 255),
        }
        model = CatBoostRanker(**params)
        model.fit(pool_tr, eval_set=pool_va, use_best_model=True)

        scores = model.predict(pool_va)
        return eval_auc(valid_sorted, scores)

    logger.info(f"Optuna開始: {N_TRIALS}試行（YetiRank、評価=fukusho AUC）")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    logger.info(f"最適パラメータ: {study.best_params}")
    logger.info(f"Best Valid AUC: {study.best_value:.4f}")

    # 最適パラメータで再学習（iterations=2000）
    logger.info("最適パラメータで再学習中...")
    best_params = {
        **study.best_params,
        "loss_function":         "YetiRank",
        "eval_metric":           "NDCG",
        "iterations":            2000,
        "random_seed":           RANDOM_STATE,
        "verbose":               200,
        "early_stopping_rounds": 50,
        "task_type":             "GPU",
    }
    best_model = CatBoostRanker(**best_params)
    best_model.fit(pool_tr, eval_set=pool_va, use_best_model=True)

    scores_va = best_model.predict(pool_va)
    scores_te = best_model.predict(pool_te)
    auc_va = eval_auc(valid_sorted, scores_va)
    auc_te = eval_auc(test_sorted,  scores_te)

    logger.info(f"[Valid] AUC={auc_va:.4f}  (CatBoost分類器: 0.7656)")
    logger.info(f"[Test]  AUC={auc_te:.4f}  (CatBoost分類器: 0.7706)")

    backup_model(MODEL_PATH)
    joblib.dump(
        {"model": best_model, "feature_cols": feature_cols},
        MODEL_PATH,
    )
    joblib.dump(study, STUDY_PATH)
    logger.info(f"モデル保存: {MODEL_PATH}")

    print("\n" + "=" * 50)
    print("CatBoost YetiRank 最適化完了サマリ")
    print("=" * 50)
    print(f"Valid AUC : {auc_va:.4f}  (分類器: 0.7656)")
    print(f"Test  AUC : {auc_te:.4f}  (分類器: 0.7706)")
    print(f"Best試行  : Trial {study.best_trial.number}")
    print(f"\n最適パラメータ:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
