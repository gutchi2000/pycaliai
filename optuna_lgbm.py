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

MASTER_CSV    = DATA_DIR / "master_20130105-20251228.csv"
HOSSEI_CSV    = DATA_DIR / "hossei" / "H_20130105-20251228.csv"
KEKKA_CSV     = DATA_DIR / "kekka_20130105-20251228.csv"
HANRO_MASTER  = Path(r"E:\競馬過去走データ\H-20150401-20260313.csv")
WC_MASTER     = Path(r"E:\競馬過去走データ\W-20150401-20260313.csv")
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
    # 補正タイム（data/hossei/ からJOIN）
    "前走補9", "前走補正",
    # 調教データ（E:\競馬過去走データ\ からJOIN）
    "trn_hanro_4f", "trn_hanro_lap1", "trn_hanro_days",
    "trn_wc_3f", "trn_wc_lap1", "trn_wc_days",
    # 前走単勝オッズ（kekka CSV からJOIN）※今走は配当データのためリーク→除外
    "前走単勝オッズ",
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
# 調教データロード・マージ
# =========================================================
def load_chukyo():
    """坂路・WCマスターCSVを読み込む。ファイルがなければNoneを返す。"""
    hanro = wc = None
    if HANRO_MASTER.exists():
        hanro = pd.read_csv(HANRO_MASTER, encoding="cp932",
                            usecols=["年月日", "馬名", "Time1", "Lap1"])
        hanro = hanro.rename(columns={"Time1": "hanro_4f", "Lap1": "hanro_lap1"})
        hanro["_dt"] = pd.to_datetime(hanro["年月日"].astype(str), format="%Y%m%d")
        hanro = hanro.sort_values(["馬名", "_dt"]).reset_index(drop=True)
        logger.info(f"坂路CSV読み込み: {len(hanro):,}行")
    else:
        logger.warning(f"坂路CSV未検出: {HANRO_MASTER}")
    if WC_MASTER.exists():
        wc = pd.read_csv(WC_MASTER, encoding="cp932",
                         usecols=["年月日", "馬名", "3F", "Lap1"])
        wc = wc.rename(columns={"3F": "wc_3f", "Lap1": "wc_lap1"})
        wc["_dt"] = pd.to_datetime(wc["年月日"].astype(str), format="%Y%m%d")
        wc = wc.sort_values(["馬名", "_dt"]).reset_index(drop=True)
        logger.info(f"WC CSV読み込み: {len(wc):,}行")
    else:
        logger.warning(f"WC CSV未検出: {WC_MASTER}")
    return hanro, wc


def merge_chukyo(df: pd.DataFrame, hanro, wc) -> pd.DataFrame:
    """グループ別searchsortedで最終追い切り（レース14日前以内）をJOIN。"""
    import numpy as np

    df = df.copy()
    df["_race_dt"] = pd.to_datetime(df["日付"].astype(str), format="%Y%m%d")
    df_idx = df.reset_index(drop=True)

    for trn, feat_cols, out_cols, prefix in [
        (hanro, ["hanro_4f", "hanro_lap1"],
                ["trn_hanro_4f", "trn_hanro_lap1"], "hanro"),
        (wc,    ["wc_3f", "wc_lap1"],
                ["trn_wc_3f", "trn_wc_lap1"],       "wc"),
    ]:
        days_col = f"trn_{prefix}_days"
        result = {c: np.full(len(df_idx), np.nan) for c in out_cols + [days_col]}

        if trn is None:
            for col in out_cols + [days_col]:
                df[col] = float("nan")
            continue

        trn_sorted = trn.sort_values(["馬名", "_dt"])
        trn_g = trn_sorted.groupby("馬名", sort=True)
        race_g = df_idx.groupby("馬名", sort=True)

        for horse, race_rows in race_g:
            if horse not in trn_g.groups:
                continue
            trn_h = trn_g.get_group(horse)
            trn_dates = trn_h["_dt"].values
            trn_feats = trn_h[feat_cols].values
            race_idxs = race_rows.index.values
            race_dates = race_rows["_race_dt"].values

            positions = np.searchsorted(trn_dates, race_dates, side="right") - 1
            cutoffs   = race_dates - np.timedelta64(14, "D")
            valid     = (positions >= 0) & (trn_dates[np.maximum(positions, 0)] >= cutoffs)

            vp = positions[valid]
            vi = race_idxs[valid]
            for j, col in enumerate(out_cols):
                result[col][vi] = trn_feats[vp, j]
            if valid.any():
                result[days_col][vi] = (
                    race_dates[valid] - trn_dates[vp]
                ).astype("timedelta64[D]").astype(float)

        for col in out_cols + [days_col]:
            df[col] = result[col]

    df = df.drop(columns=["_race_dt"])
    cov_h = df["trn_hanro_4f"].notna().mean() * 100
    cov_w = df["trn_wc_3f"].notna().mean() * 100
    logger.info(f"調教JOIN完了: 坂路カバレッジ={cov_h:.1f}%  WC={cov_w:.1f}%")
    return df


# =========================================================
# データロード（1回だけ読み込んでキャッシュ）
# =========================================================
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info(f"マスターCSV読み込み: {MASTER_CSV}")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    # 補正タイムJOIN
    if HOSSEI_CSV.exists():
        hossei = pd.read_csv(HOSSEI_CSV, encoding="cp932",
                             usecols=["レースID(新)", "馬番", "前走補9", "前走補正"])
        df = df.merge(hossei, on=["レースID(新)", "馬番"], how="left")
        logger.info(f"hossei JOIN完了: 前走補9カバレッジ={df['前走補9'].notna().mean()*100:.1f}%")
    else:
        df["前走補9"]  = float("nan")
        df["前走補正"] = float("nan")
        logger.warning(f"hossei CSV未検出: {HOSSEI_CSV}")
    # 調教データJOIN
    hanro, wc = load_chukyo()
    df = merge_chukyo(df, hanro, wc)
    # 単勝オッズJOIN（kekka CSV）
    if KEKKA_CSV.exists():
        kekka = pd.read_csv(KEKKA_CSV, encoding="cp932",
                            usecols=["レースID(新)", "馬番", "単勝配当"])
        def _parse_tansho(s):
            s = str(s).strip()
            if s.startswith("(") and s.endswith(")"):
                return float(s[1:-1])   # 既にオッズ形式
            try:
                return float(s) / 100   # 配当→オッズ
            except Exception:
                return float("nan")
        kekka["単勝オッズ"] = kekka["単勝配当"].apply(_parse_tansho)
        kekka = kekka.drop(columns=["単勝配当"])
        kekka["レースID(新)"] = kekka["レースID(新)"].astype("int64")
        kekka["馬番"] = kekka["馬番"].astype("int64")
        # 今走単勝オッズ
        df = df.merge(kekka, on=["レースID(新)", "馬番"], how="left")
        # 前走単勝オッズ（前走レースID×馬番でJOIN）
        kekka_prev = kekka.rename(columns={
            "レースID(新)": "前走レースID(新)", "単勝オッズ": "前走単勝オッズ"})
        df["前走レースID(新)"] = pd.to_numeric(df["前走レースID(新)"], errors="coerce")
        df = df.merge(kekka_prev, on=["前走レースID(新)", "馬番"], how="left")
        logger.info(
            f"単勝オッズJOIN完了: 今走={df['単勝オッズ'].notna().mean()*100:.1f}%"
            f"  前走={df['前走単勝オッズ'].notna().mean()*100:.1f}%"
        )
    else:
        df["単勝オッズ"]   = float("nan")
        df["前走単勝オッズ"] = float("nan")
        logger.warning(f"kekka CSV未検出: {KEKKA_CSV}")
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