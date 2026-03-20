"""
calibrate.py
PyCaLiAI - アンサンブル予測確率のキャリブレーション

Valid セット（2023年）の予測確率に IsotonicRegression を当て、
Kelly 基準で使用できる真の確率に変換するキャリブレーターを生成・保存する。

出力:
    models/ensemble_calibrator_v1.pkl  ← predict_weekly.py / app.py で使用

Usage:
    python calibrate.py
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import Pool
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

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

from utils import parse_time_str

# =========================================================
# パス設定
# =========================================================
BASE_DIR   = Path(r"E:\PyCaLiAI")
DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MASTER_CSV      = DATA_DIR  / "master_20130105-20251228.csv"
HOSEI_DIR       = DATA_DIR  / "hosei"
LGBM_MODEL_PATH  = MODEL_DIR / "lgbm_optuna_v1.pkl"
CAT_MODEL_PATH   = MODEL_DIR / "catboost_optuna_v1.pkl"
RANK_MODEL_PATH  = MODEL_DIR / "catboost_rank_v1.pkl"
TORCH_MODEL_PATH = MODEL_DIR / "transformer_pl_v2.pkl"
META_MODEL_PATH  = MODEL_DIR / "stacking_meta_v1.pkl"
CAL_OUT_PATH     = MODEL_DIR / "ensemble_calibrator_v1.pkl"
STACK_CAL_PATH   = MODEL_DIR / "stacking_calibrator_v1.pkl"

TARGET = "fukusho_flag"

_model_cache: dict = {}

CLASS_NORMALIZE = {
    "新馬": "新馬", "未勝利": "未勝利",
    "1勝": "1勝", "500万": "1勝",
    "2勝": "2勝", "1000万": "2勝",
    "3勝": "3勝", "1600万": "3勝",
    "OP(L)": "OP(L)", "Ｇ１": "Ｇ１", "Ｇ２": "Ｇ２", "Ｇ３": "Ｇ３",
}

_META_EXTRA = ["芝・ダ", "距離", "クラス名", "場所", "馬場状態", "出走頭数", "枠番", "馬番"]


# =========================================================
# 予測関数（predict_weekly.py と同じ実装）
# =========================================================
def _predict_lgbm(df: pd.DataFrame, obj: dict) -> np.ndarray:
    from sklearn.preprocessing import LabelEncoder

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

    # lgb.Booster と LGBMClassifier 両対応
    import lightgbm as lgb
    if isinstance(model, lgb.Booster):
        return model.predict(df[feature_cols])
    return model.predict_proba(df[feature_cols])[:, 1]


def _predict_catboost_rank(df: pd.DataFrame, obj: dict) -> np.ndarray:
    """YetiRankモデルで予測。スコアをレース内min-max正規化して[0,1]に変換。"""
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

    race_id_col = "レースID(新/馬番無)" if "レースID(新/馬番無)" in df.columns else df.columns[0]
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
    """Transformer PL予測。レース内スコアをmin-max正規化して[0,1]に変換。モデル未存在 or 失敗時はゼロ配列を返す。"""
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

        race_id_col = "レースID(新/馬番無)" if "レースID(新/馬番無)" in df.columns else df.columns[0]
        result  = np.zeros(len(df))
        df_sort = df.sort_values(race_id_col).reset_index(drop=True)
        idx = 0
        for _, group in df_sort.groupby(race_id_col, sort=True):
            valid_n = min(len(group), PL_MAX_HORSES)
            for orig_idx in list(group.index)[:valid_n]:
                if idx < len(all_scores):
                    result[orig_idx] = all_scores[idx]
                    idx += 1

        # レース内 min-max 正規化
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


def _predict_stacking(df: pd.DataFrame, lgbm_obj: dict, cat_obj: dict) -> np.ndarray | None:
    """スタッキングモデルで予測（4モデル対応）。未存在 or 失敗時は None を返す。"""
    if not META_MODEL_PATH.exists():
        return None
    try:
        from stacking import build_meta_features

        p_lgbm  = _predict_lgbm(df, lgbm_obj)
        p_cat   = _predict_catboost(df, cat_obj)
        p_rank  = _predict_catboost_rank(df, joblib.load(RANK_MODEL_PATH)) \
                  if RANK_MODEL_PATH.exists() else np.full(len(df), 0.5)
        p_trans = _predict_transformer(df)

        if "meta" not in _model_cache:
            _model_cache["meta"] = joblib.load(META_MODEL_PATH)
        meta_obj      = _model_cache["meta"]
        meta_model    = meta_obj["meta_model"]
        meta_encoders = meta_obj["meta_encoders"]
        meta_cols     = meta_obj["meta_cols"]

        # レースID列の確認
        race_id_col = "レースID(新/馬番無)" if "レースID(新/馬番無)" in df.columns else "レースID(新)"
        if "出走頭数" not in df.columns or df["出走頭数"].isna().all():
            df = df.copy()
            df["出走頭数"] = df.groupby(race_id_col)["馬番"].transform("count")

        meta_df, _ = build_meta_features(
            df, p_lgbm, p_cat, p_rank, p_trans,
            meta_encoders=meta_encoders, fit=False,
            race_col=race_id_col,
        )

        return meta_model.predict_proba(meta_df[meta_cols])[:, 1]
    except Exception as e:
        logger.warning(f"スタッキング予測失敗: {e}")
        return None


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


# =========================================================
# キャリブレーション
# =========================================================
def fit_calibrator(y_true: np.ndarray, proba: np.ndarray, name: str) -> IsotonicRegression:
    """IsotonicRegression をフィットし、Brier スコアで改善を確認する。"""
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(proba, y_true)
    proba_cal = ir.transform(proba)

    brier_before = brier_score_loss(y_true, proba)
    brier_after  = brier_score_loss(y_true, proba_cal)
    improvement  = (brier_before - brier_after) / brier_before * 100

    logger.info(
        f"[{name}] Brier: {brier_before:.4f} → {brier_after:.4f} "
        f"({improvement:+.1f}%)"
    )
    return ir


def plot_reliability(
    y_true: np.ndarray,
    proba_raw: np.ndarray,
    proba_cal: np.ndarray,
    name: str,
) -> None:
    """キャリブレーション前後の信頼性ダイアグラムを保存する。"""
    n_bins = 10
    fig, ax = plt.subplots(figsize=(6, 5))

    frac_pos_raw, mean_pred_raw = calibration_curve(y_true, proba_raw, n_bins=n_bins)
    frac_pos_cal, mean_pred_cal = calibration_curve(y_true, proba_cal, n_bins=n_bins)

    ax.plot([0, 1], [0, 1], "k--", label="完全キャリブレーション")
    ax.plot(mean_pred_raw, frac_pos_raw, "o-", label="補正前")
    ax.plot(mean_pred_cal, frac_pos_cal, "s-", label="補正後（Isotonic）")

    ax.set_xlabel("予測確率")
    ax.set_ylabel("実際の正例率")
    ax.set_title(f"信頼性ダイアグラム [{name}]")
    ax.legend()
    ax.grid(alpha=0.3)

    path = REPORT_DIR / f"calibration_{name.lower().replace(' ', '_')}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"信頼性ダイアグラム保存: {path}")


# =========================================================
# メイン
# =========================================================
def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # --- データ読み込み ---
    logger.info(f"マスターCSV読み込み: {MASTER_CSV}")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    logger.info(f"  総行数: {len(df):,}")

    # 補正タイムJOIN（data/hosei/H_*.csv を全て結合）
    hosei_files = sorted(HOSEI_DIR.glob("H_*.csv"))
    if hosei_files:
        hosei = pd.concat([
            pd.read_csv(f, encoding="cp932",
                        usecols=["レースID(新)", "前走補9", "前走補正"])
            for f in hosei_files
        ], ignore_index=True).drop_duplicates()
        df = df.merge(hosei, on="レースID(新)", how="left")
        logger.info(f"  hosei JOIN完了 ({len(hosei_files)}ファイル): 前走補9カバレッジ={df['前走補9'].notna().mean()*100:.1f}%")

    if "split" not in df.columns:
        raise ValueError(
            "'split' 列がありません。"
            "build_dataset.py で split 列（train/valid/test）を付与してください。"
        )

    valid = df[df["split"] == "valid"].copy()
    logger.info(f"  Valid: {len(valid):,} 行（2023年）")

    if len(valid) == 0:
        raise ValueError("Valid セットが空です。split 列を確認してください。")

    y_valid = valid[TARGET].values
    pos_rate = y_valid.mean()
    logger.info(f"  正例率（複勝内）: {pos_rate:.3f}")

    # --- モデル読み込み ---
    logger.info("モデル読み込み中...")
    lgbm_obj = joblib.load(LGBM_MODEL_PATH)
    cat_obj  = joblib.load(CAT_MODEL_PATH)

    # --- Valid セットで予測（生スコア）---
    logger.info("LightGBM 予測中...")
    lgbm_proba = _predict_lgbm(valid, lgbm_obj)

    logger.info("CatBoost 予測中...")
    cat_proba = _predict_catboost(valid, cat_obj)

    # YetiRank・Transformer PLが存在すれば4モデル加重平均
    rank_proba  = np.zeros(len(valid))
    trans_proba = np.zeros(len(valid))
    use_rank    = RANK_MODEL_PATH.exists()
    use_trans   = TORCH_MODEL_PATH.exists()

    if use_rank:
        logger.info("YetiRank 予測中...")
        rank_obj   = joblib.load(RANK_MODEL_PATH)
        rank_proba = _predict_catboost_rank(valid, rank_obj)
    if use_trans:
        logger.info("Transformer PL 予測中...")
        trans_proba = _predict_transformer(valid)

    if use_rank and use_trans:
        ens_proba = 0.30 * lgbm_proba + 0.30 * cat_proba + 0.20 * rank_proba + 0.20 * trans_proba
        logger.info("4モデルアンサンブル（LGBM×0.30 + CatBoost×0.30 + YetiRank×0.20 + TransPL×0.20）")
    elif use_rank:
        ens_proba = 0.40 * lgbm_proba + 0.40 * cat_proba + 0.20 * rank_proba
        logger.info("3モデルアンサンブル（LGBM×0.40 + CatBoost×0.40 + YetiRank×0.20）")
    else:
        ens_proba = 0.5 * lgbm_proba + 0.5 * cat_proba
        logger.info("2モデルアンサンブル（LGBM×0.50 + CatBoost×0.50）")

    logger.info(f"予測確率の統計（アンサンブル生）: "
                f"mean={ens_proba.mean():.3f}  std={ens_proba.std():.3f}  "
                f"min={ens_proba.min():.3f}  max={ens_proba.max():.3f}")

    # --- キャリブレーター学習 ---
    logger.info("キャリブレーター学習中...")
    ens_cal = fit_calibrator(y_valid, ens_proba, "Ensemble")

    ens_proba_cal = ens_cal.transform(ens_proba)
    logger.info(f"予測確率の統計（キャリブレーション後）: "
                f"mean={ens_proba_cal.mean():.3f}  std={ens_proba_cal.std():.3f}  "
                f"min={ens_proba_cal.min():.3f}  max={ens_proba_cal.max():.3f}")

    # 正例率との一致を確認（well-calibrated なら mean ≈ pos_rate）
    gap = abs(ens_proba_cal.mean() - pos_rate)
    logger.info(
        f"キャリブレーション後 mean={ens_proba_cal.mean():.3f} vs 正例率={pos_rate:.3f}  "
        f"乖離={gap:.4f}"
    )

    # --- 信頼性ダイアグラム ---
    plot_reliability(y_valid, ens_proba, ens_proba_cal, "Ensemble")

    # --- 保存（2モデルアンサンブル用）---
    payload = {
        "calibrator":  ens_cal,
        "valid_brier_before": brier_score_loss(y_valid, ens_proba),
        "valid_brier_after":  brier_score_loss(y_valid, ens_proba_cal),
        "valid_pos_rate":     pos_rate,
    }
    joblib.dump(payload, CAL_OUT_PATH)
    logger.info(f"キャリブレーター保存: {CAL_OUT_PATH}")

    # =========================================================
    # スタッキングキャリブレーション（モデルが存在する場合のみ）
    # =========================================================
    if META_MODEL_PATH.exists() and TORCH_MODEL_PATH.exists():
        logger.info("スタッキングキャリブレーション開始...")
        stack_proba = _predict_stacking(valid, lgbm_obj, cat_obj)
        if stack_proba is not None:
            stack_cal = fit_calibrator(y_valid, stack_proba, "Stacking")
            stack_proba_cal = stack_cal.transform(stack_proba)

            gap_stack = abs(stack_proba_cal.mean() - pos_rate)
            logger.info(
                f"[Stacking] キャリブレーション後 mean={stack_proba_cal.mean():.3f} "
                f"vs 正例率={pos_rate:.3f}  乖離={gap_stack:.4f}"
            )

            plot_reliability(y_valid, stack_proba, stack_proba_cal, "Stacking")

            stack_payload = {
                "calibrator":  stack_cal,
                "valid_brier_before": brier_score_loss(y_valid, stack_proba),
                "valid_brier_after":  brier_score_loss(y_valid, stack_proba_cal),
                "valid_pos_rate":     pos_rate,
            }
            joblib.dump(stack_payload, STACK_CAL_PATH)
            logger.info(f"スタッキングキャリブレーター保存: {STACK_CAL_PATH}")
        else:
            logger.warning("スタッキング予測失敗。STACK_CAL_PATH は生成されません。")
    else:
        logger.info(
            f"スタッキングモデル未検出。スキップ。\n"
            f"  {META_MODEL_PATH}\n  {TORCH_MODEL_PATH}"
        )


if __name__ == "__main__":
    main()
