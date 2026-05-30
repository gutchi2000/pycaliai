"""
train_lgbm_win.py
PyCaLiAI - 1着勝率直接予測モデル（単勝EV改善用）

目的変数: is_1st_place (1着=1, それ以外=0)
期待効果: 複勝フラグモデルより単勝EVの精度が向上し、
          EV閾値を1.0に下げても ROI 100%超えを維持しながら N数を拡大する

Usage:
    python train_lgbm_win.py
    python train_lgbm_win.py --n_trials 50
"""

from __future__ import annotations

import argparse
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
HOSEI_DIR     = DATA_DIR / "hosei"
HANRO_MASTER  = Path(r"E:\競馬過去走データ\H-20150401-20260313.csv")
WC_MASTER     = Path(r"E:\競馬過去走データ\W-20150401-20260313.csv")
CHUKYO_DIR    = DATA_DIR / "training"
MODEL_PATH    = MODEL_DIR / "lgbm_win_v1.pkl"

TARGET       = "is_1st_place"
RANDOM_STATE = 42
N_TRIALS     = 30

# =========================================================
# 特徴量定義（optuna_lgbm.py と同じ）
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
]

TIME_STR_FEATURES = ["前走走破タイム", "前走着差タイム"]
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES + TIME_STR_FEATURES


# =========================================================
# 前処理
# =========================================================
def parse_time_str(series: pd.Series) -> pd.Series:
    def _c(val):
        try:
            parts = str(val).strip().split(".")
            if len(parts) == 3:
                return int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 10
            return float(val)
        except Exception:
            return None
    return series.apply(_c)


def encode_categoricals(df, cat_cols, encoders=None, fit=True):
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


def preprocess(df, encoders=None, fit=True):
    df = df.copy()
    for col in TIME_STR_FEATURES:
        if col in df.columns:
            df[col] = parse_time_str(df[col])
    df, encoders = encode_categoricals(df, CAT_FEATURES, encoders, fit=fit)
    return df, encoders


# =========================================================
# 調教データ（optuna_lgbm.py と同じ実装）
# =========================================================
def load_chukyo():
    CHUKYO_DIR.mkdir(parents=True, exist_ok=True)

    def _read_hanro(path):
        return pd.read_csv(path, encoding="cp932",
                           usecols=["年月日", "馬名", "Time1", "Time2", "Time3", "Time4",
                                    "Lap4", "Lap3", "Lap2", "Lap1"])

    def _read_wc(path):
        return pd.read_csv(path, encoding="cp932",
                           usecols=["年月日", "馬名", "5F", "4F", "3F", "Lap3", "Lap2", "Lap1"])

    def _collect(master, glob_pat, reader, rename_map, label):
        files = []
        if master.exists():
            files.append(master)
        for f in sorted(CHUKYO_DIR.glob(glob_pat)):
            if f != master:
                files.append(f)
        if not files:
            logger.warning(f"{label} CSV未検出")
            return None
        dfs = []
        for f in files:
            try:
                dfs.append(reader(f))
            except Exception as e:
                logger.warning(f"{label} スキップ: {f.name} ({e})")
        if not dfs:
            return None
        result = pd.concat(dfs, ignore_index=True).drop_duplicates()
        result = result.rename(columns=rename_map)
        result["_dt"] = pd.to_datetime(result["年月日"].astype(str), format="%Y%m%d")
        result = result.sort_values(["馬名", "_dt"]).reset_index(drop=True)
        logger.info(f"{label} 読み込み: {len(result):,}行")
        return result

    hanro = _collect(HANRO_MASTER, "H-*.csv", _read_hanro,
        {"Time1": "hanro_4f", "Time2": "hanro_3f", "Time3": "hanro_2f", "Time4": "hanro_1f",
         "Lap4": "hanro_lap4", "Lap3": "hanro_lap3", "Lap2": "hanro_lap2", "Lap1": "hanro_lap1"},
        "坂路")
    wc = _collect(WC_MASTER, "W-*.csv", _read_wc,
        {"5F": "wc_5f", "4F": "wc_4f", "3F": "wc_3f",
         "Lap3": "wc_lap3", "Lap2": "wc_lap2", "Lap1": "wc_lap1"},
        "WC")
    return hanro, wc


def merge_chukyo(df, hanro, wc):
    df = df.copy()
    df["_race_dt"] = pd.to_datetime(df["日付"].astype(str), format="%Y%m%d")
    df_idx = df.reset_index(drop=True)

    for trn, feat_cols, out_cols, prefix in [
        (hanro,
         ["hanro_4f", "hanro_3f", "hanro_2f", "hanro_1f",
          "hanro_lap4", "hanro_lap3", "hanro_lap2", "hanro_lap1"],
         ["trn_hanro_4f", "trn_hanro_3f", "trn_hanro_2f", "trn_hanro_1f",
          "trn_hanro_lap4", "trn_hanro_lap3", "trn_hanro_lap2", "trn_hanro_lap1"],
         "hanro"),
        (wc,
         ["wc_5f", "wc_4f", "wc_3f", "wc_lap3", "wc_lap2", "wc_lap1"],
         ["trn_wc_5f", "trn_wc_4f", "trn_wc_3f", "trn_wc_lap3", "trn_wc_lap2", "trn_wc_lap1"],
         "wc"),
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
    return df


# =========================================================
# データロード
# =========================================================
def load_data():
    logger.info(f"マスターCSV読み込み: {MASTER_CSV}")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)

    # 補正タイムJOIN
    hosei_files = sorted(HOSEI_DIR.glob("H_*.csv"))
    if hosei_files:
        def _read_hosei(f):
            for enc in ["utf-8-sig", "cp932"]:
                try:
                    return pd.read_csv(f, encoding=enc,
                                       usecols=["レースID(新)", "前走補9", "前走補正"])
                except (UnicodeDecodeError, ValueError):
                    continue
            raise ValueError(f"hosei CSV エンコード不明: {f}")
        hosei = pd.concat([_read_hosei(f) for f in hosei_files],
                          ignore_index=True).drop_duplicates()
        df = df.merge(hosei, on="レースID(新)", how="left")
    else:
        df["前走補9"]  = float("nan")
        df["前走補正"] = float("nan")

    # 調教データJOIN
    hanro, wc = load_chukyo()
    df = merge_chukyo(df, hanro, wc)

    # is_1st_place 目的変数を作成
    df["着順_num"] = pd.to_numeric(df["着順"], errors="coerce")
    df[TARGET] = (df["着順_num"] == 1).astype(int)

    # 除外: 着順不明（取消・除外）、新馬・障害
    df = df[df["着順_num"].notna()].copy()

    train = df[df["split"] == "train"].copy()
    valid = df[df["split"] == "valid"].copy()
    test  = df[df["split"] == "test"].copy()

    pos_rate = train[TARGET].mean()
    logger.info(f"分割: train={len(train):,} / valid={len(valid):,} / test={len(test):,}")
    logger.info(f"1着率(train): {pos_rate:.4f} ({pos_rate*100:.1f}%) → scale_pos_weight={1/pos_rate - 1:.1f}")
    return train, valid, test


# =========================================================
# Objective関数
# =========================================================
def make_objective(train, valid, encoders, feature_cols, scale_pos_weight):
    X_va = valid[feature_cols]
    y_va = valid[TARGET]

    def objective(trial):
        params = {
            "objective":          "binary",
            "metric":             "auc",
            "random_state":       RANDOM_STATE,
            "verbose":            -1,
            "scale_pos_weight":   scale_pos_weight,
            "num_leaves":         trial.suggest_int("num_leaves", 31, 255),
            "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_child_samples":  trial.suggest_int("min_child_samples", 20, 200),
            "subsample":          trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":          trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":         trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_split_gain":     trial.suggest_float("min_split_gain", 0.0, 1.0),
            "n_estimators":       2000,
        }
        model = LGBMClassifier(**params)
        model.fit(
            train[feature_cols], train[TARGET],
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
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=N_TRIALS)
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train, valid, test = load_data()
    train, encoders = preprocess(train, fit=True)
    valid, _        = preprocess(valid, encoders=encoders, fit=False)
    test,  _        = preprocess(test,  encoders=encoders, fit=False)

    feature_cols = [c for c in ALL_FEATURES if c in train.columns]
    logger.info(f"使用特徴量数: {len(feature_cols)}")

    pos_rate = train[TARGET].mean()
    scale_pos_weight = (1 - pos_rate) / pos_rate  # 約13（14頭立て平均）
    logger.info(f"scale_pos_weight: {scale_pos_weight:.2f}")

    # Optuna最適化
    logger.info(f"Optuna開始: {args.n_trials}試行")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        make_objective(train, valid, encoders, feature_cols, scale_pos_weight),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    logger.info(f"最適パラメータ: {study.best_params}")
    logger.info(f"Best Valid AUC: {study.best_value:.4f}")

    # 最適パラメータで再学習
    best_params = {
        **study.best_params,
        "objective":          "binary",
        "metric":             "auc",
        "random_state":       RANDOM_STATE,
        "verbose":            -1,
        "scale_pos_weight":   scale_pos_weight,
        "n_estimators":       2000,
    }
    best_model = LGBMClassifier(**best_params)
    best_model.fit(
        train[feature_cols], train[TARGET],
        eval_set=[(valid[feature_cols], valid[TARGET])],
        callbacks=[
            early_stopping(stopping_rounds=50, verbose=False),
            log_evaluation(period=200),
        ],
    )

    proba_va = best_model.predict_proba(valid[feature_cols])[:, 1]
    proba_te = best_model.predict_proba(test[feature_cols])[:, 1]
    auc_va   = roc_auc_score(valid[TARGET], proba_va)
    auc_te   = roc_auc_score(test[TARGET],  proba_te)

    logger.info(f"[Valid] AUC={auc_va:.4f}")
    logger.info(f"[Test]  AUC={auc_te:.4f}")

    # 参考: 複勝フラグモデルとの比較（既存モデルのAUC）
    logger.info("参考: 複勝フラグモデル Valid AUC ≈ 0.7606")

    joblib.dump(
        {"model": best_model, "encoders": encoders, "feature_cols": feature_cols,
         "target": TARGET, "scale_pos_weight": scale_pos_weight},
        MODEL_PATH,
    )
    logger.info(f"モデル保存: {MODEL_PATH}")

    print("\n" + "=" * 50)
    print("lgbm_win_v1 学習完了")
    print("=" * 50)
    print(f"目的変数   : {TARGET}")
    print(f"Valid AUC  : {auc_va:.4f}")
    print(f"Test  AUC  : {auc_te:.4f}")
    print(f"特徴量数   : {len(feature_cols)}")
    print(f"参考(複勝) : Valid AUC ~= 0.7606")
    print(f"\n保存先: {MODEL_PATH}")
    print("\n次のステップ: backtest.py に lgbm_win_v1 を組み込んでEV計算を比較")


if __name__ == "__main__":
    main()
