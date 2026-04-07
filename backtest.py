"""
backtest.py
PyCaLiAI - バックテスト（実オッズ対応版）

kekka_20130105-20251228.csv の実際の払戻額を使って
正確な回収率を計算する。

払戻額の単位: 100円あたりの配当
  例）複勝120 → 100円購入で120円払戻（回収率120%）
  実際の払戻額 = 購入額 × (払戻配当 / 100)

Usage:
    python backtest.py
    python backtest.py --n_races 100
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import Pool
from tqdm import tqdm

from ev_filter import BetFilter, is_upgrade_race

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
BASE_DIR         = Path(r"E:\PyCaLiAI")
DATA_DIR         = BASE_DIR / "data"
MODEL_DIR        = BASE_DIR / "models"
REPORT_DIR       = BASE_DIR / "reports"

MASTER_CSV       = DATA_DIR  / "master_20130105-20251228.csv"
KEKKA_CSV        = DATA_DIR  / "kekka_20130105-20251228.csv"
LGBM_MODEL_PATH  = MODEL_DIR / "lgbm_optuna_v1.pkl"
CAT_MODEL_PATH   = MODEL_DIR / "catboost_optuna_v1.pkl"
RANK_MODEL_PATH  = MODEL_DIR / "catboost_rank_v1.pkl"
TORCH_MODEL_PATH = MODEL_DIR / "transformer_optuna_v1.pkl"
META_PATH        = MODEL_DIR / "stacking_meta_v1.pkl"
STRATEGY_JSON    = BASE_DIR  / "data" / "strategy_weights.json"
STACK_CAL_PATH   = MODEL_DIR / "stacking_calibrator_v1.pkl"
CAL_PATH         = MODEL_DIR / "ensemble_calibrator_v4.pkl"   # Test-based 2024 (best)
CAL_PATH_V3      = MODEL_DIR / "ensemble_calibrator_v3.pkl"   # Train-based (fallback)
CAL_PATH_V2      = MODEL_DIR / "ensemble_calibrator_v2.pkl"   # Valid-based (fallback)
CAL_PATH_V1      = MODEL_DIR / "ensemble_calibrator_v1.pkl"   # Original (fallback)
WIN_MODEL_PATH   = MODEL_DIR / "lgbm_win_v1.pkl"              # is_1st_place classifier

RETURN_RATE      = {"単勝": 0.80, "複勝": 0.75}
ODDS_CSV_DIR     = BASE_DIR / "data"

TARGET      = "fukusho_flag"
COL_RACE_ID = "レースID(新/馬番無)"
COL_RANK    = "着順"
BUDGET      = 10_000
MIN_UNIT    = 100

# app.py / predict_weekly.py と同じ除外条件
EXCLUDE_PLACES  = {"東京", "小倉"}
EXCLUDE_CLASSES = {"新馬", "障害"}

# predict_weekly.py と同じ定数
_META_EXTRA = ["芝・ダ", "距離", "クラス名", "場所", "馬場状態", "出走頭数", "枠番", "馬番"]
CLASS_NORMALIZE = {
    "新馬":"新馬","未勝利":"未勝利","1勝":"1勝","500万":"1勝",
    "2勝":"2勝","1000万":"2勝","3勝":"3勝","1600万":"3勝",
    "OP(L)":"OP(L)","Ｇ１":"Ｇ１","Ｇ２":"Ｇ２","Ｇ３":"Ｇ３",
}

# --no_strategy（組み合わせ探索）用: 全馬券種を等分でベット
BUDGET_RATIO = {
    "単勝":   0.20,
    "複勝":   0.20,
    "枠連":   0.20,
    "馬連":   0.20,
    "三連複": 0.20,
}

TAKEOUT = {
    "単勝":   0.20,
    "複勝":   0.20,
    "枠連":   0.225,
    "馬連":   0.225,
    "三連複": 0.25,
}


# =========================================================
# 結果CSV読み込み・払戻辞書構築
# =========================================================
def load_kekka(kekka_path: Path) -> dict[str, dict]:
    logger.info(f"kekka CSV読み込み: {kekka_path}")
    df = pd.read_csv(kekka_path, encoding="cp932", low_memory=False)
    logger.info(f"  {len(df):,}行 × {len(df.columns)}列")

    df["race_id"]   = df["レースID(新)"].astype(str).str[:16]
    df["確定着順"]   = pd.to_numeric(df["確定着順"], errors="coerce")
    df["馬番"]       = pd.to_numeric(df["馬番"],     errors="coerce")

    kekka_dict: dict[str, dict] = {}

    for race_id, group in df.groupby("race_id"):
        group = group.sort_values("確定着順")

        entry: dict = {
            "単勝": {}, "複勝": {}, "枠連": {}, "馬連": {}, "馬単": {}, "三連複": {}, "三連単": {}
        }

        # 複勝（1〜3着馬番→配当）
        for _, row in group[group["確定着順"] <= 3].iterrows():
            ban = row["馬番"]
            pay = row["複勝配当"]
            if pd.notna(ban) and pd.notna(pay):
                entry["複勝"][int(ban)] = int(pay)

        # 上位3頭
        top3_rows = group[group["確定着順"] <= 3].sort_values("確定着順")
        top3 = [
            int(h) for h in top3_rows["馬番"].tolist()
            if pd.notna(h)
        ]

        # 1着行から組み合わせ系を取得
        rank1 = group[group["確定着順"] == 1]
        if rank1.empty:
            kekka_dict[str(race_id)] = entry
            continue
        r1 = rank1.iloc[0]

        # 単勝（1着馬番→配当）
        ban = r1["馬番"]
        pay = r1.get("単勝配当")
        if pd.notna(ban) and pd.notna(pay):
            try:
                entry["単勝"][int(ban)] = int(pay)
            except (ValueError, TypeError):
                pass

        if len(top3) >= 2:
            # 枠連（1着・2着の枠番→配当）
            rank2 = group[group["確定着順"] == 2]
            if not rank2.empty:
                w1 = r1.get("枠番")
                w2 = rank2.iloc[0].get("枠番")
                pay_wk = r1.get("枠連")
                if pd.notna(w1) and pd.notna(w2) and pd.notna(pay_wk):
                    try:
                        wk = "-".join(map(str, sorted([int(w1), int(w2)])))
                        entry["枠連"][wk] = int(pay_wk)
                    except (ValueError, TypeError):
                        pass
            # 馬連
            key = "-".join(map(str, sorted(top3[:2])))
            pay = r1["馬連"]
            if pd.notna(pay):
                entry["馬連"][key] = int(pay)
            # 馬単
            key = f"{top3[0]}-{top3[1]}"
            pay = r1["馬単"]
            if pd.notna(pay):
                entry["馬単"][key] = int(pay)

        if len(top3) >= 3:
            # 三連複
            key = "-".join(map(str, sorted(top3[:3])))
            pay = r1["３連複"]
            if pd.notna(pay):
                entry["三連複"][key] = int(pay)
            # 三連単
            key = f"{top3[0]}-{top3[1]}-{top3[2]}"
            pay = r1["３連単"]
            if pd.notna(pay):
                entry["三連単"][key] = int(pay)

        kekka_dict[str(race_id)] = entry

    logger.info(f"払戻辞書構築完了: {len(kekka_dict):,}レース")
    return kekka_dict


# =========================================================
# 的中判定＋実払戻取得
# =========================================================
def get_actual_payout(
    combo: list[int],
    ordered: bool,
    bet_type: str,
    kekka_entry: dict,
) -> int:
    """
    実際の払戻配当（100円あたり）を返す。
    的中しない場合は0を返す。
    """
    if bet_type == "単勝":
        h = combo[0]
        pay = kekka_entry.get("単勝", {}).get(int(h), 0)
        return int(pay) if pay else 0

    elif bet_type == "複勝":
        h = combo[0]
        pay = kekka_entry["複勝"].get(h, 0)
        return int(pay) if pay else 0

    elif bet_type == "枠連":
        key = "-".join(map(str, sorted(combo)))
        pay = kekka_entry.get("枠連", {}).get(key, 0)
        return int(pay) if pay else 0

    elif bet_type == "馬連":
        key = "-".join(map(str, sorted(combo)))
        pay = kekka_entry["馬連"].get(key, 0)
        return int(pay) if pay else 0

    elif bet_type == "馬単":
        key = f"{combo[0]}-{combo[1]}"
        pay = kekka_entry["馬単"].get(key, 0)
        return int(pay) if pay else 0

    elif bet_type == "三連複":
        key = "-".join(map(str, sorted(combo)))
        pay = kekka_entry["三連複"].get(key, 0)
        return int(pay) if pay else 0

    elif bet_type == "三連単":
        # 着順通りのキー: 1着-2着-3着
        key = f"{combo[0]}-{combo[1]}-{combo[2]}"
        pay = kekka_entry["三連単"].get(key, 0)
        return int(pay) if pay else 0

    return 0


# =========================================================
# 前処理ユーティリティ
# =========================================================
from utils import parse_time_str


def floor_to_unit(amount: int, unit: int = MIN_UNIT) -> int:
    return (amount // unit) * unit


def estimate_odds(win_prob: float, bet_type: str) -> float:
    rate = TAKEOUT.get(bet_type, 0.25)
    if win_prob <= 0:
        return 0.0
    return round((1 - rate) / win_prob, 1)


def calc_win_prob_pl(
    horses: list[int],
    prob_series: pd.Series,
    ordered: bool,
) -> float:
    total = prob_series.sum()
    if total <= 0:
        return 0.0
    norm = prob_series / total
    if not ordered:
        probs = [norm.get(h, 0.0) for h in horses]
        return float(min(np.prod(probs) ** (1 / len(probs)) * len(probs), 0.99))
    else:
        remaining = 1.0
        prob_val  = 1.0
        for h in horses:
            p = norm.get(h, 0.0)
            if remaining <= 0:
                break
            prob_val  *= p / remaining
            remaining -= p
        return max(float(prob_val), 0.0)


# =========================================================
# 各モデル予測
# =========================================================
def predict_lgbm_batch(df: pd.DataFrame) -> np.ndarray:
    if not LGBM_MODEL_PATH.exists():
        raise FileNotFoundError(f"LightGBMモデルが見つかりません: {LGBM_MODEL_PATH}")
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


def predict_lgbm_win_batch(df: pd.DataFrame) -> np.ndarray:
    """lgbm_win_v1 (is_1st_place) バッチ予測。モデル未存在時は zeros を返す。"""
    if not WIN_MODEL_PATH.exists():
        return np.zeros(len(df))
    obj          = joblib.load(WIN_MODEL_PATH)
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


def predict_catboost_batch(df: pd.DataFrame) -> np.ndarray:
    if not CAT_MODEL_PATH.exists():
        raise FileNotFoundError(f"CatBoostモデルが見つかりません: {CAT_MODEL_PATH}")
    obj          = joblib.load(CAT_MODEL_PATH)
    model        = obj["model"]
    feature_cols = obj["feature_cols"]
    cat_features_list = [
        "種牡馬", "父タイプ名", "母父馬", "母父タイプ名", "毛色",
        "馬主(最新/仮想)", "生産者",
        "芝・ダ", "コース区分", "芝(内・外)", "馬場状態", "天気",
        "クラス名", "場所", "性別", "斤量", "ブリンカー", "重量種別",
        "年齢限定", "限定", "性別限定", "指定条件",
        "前走場所", "前芝・ダ", "前走馬場状態", "前走斤量", "前好走",
    ]
    df = df.copy()
    for col in ["前走走破タイム", "前走着差タイム"]:
        if col in df.columns:
            df[col] = parse_time_str(df[col])
    for col in cat_features_list:
        df[col] = df[col].fillna("__NaN__").astype(str) if col in df.columns else "__NaN__"
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    cat_indices = [i for i, c in enumerate(feature_cols) if c in cat_features_list]
    pool = Pool(df[feature_cols], cat_features=cat_indices)
    return model.predict_proba(pool)[:, 1]


def predict_catboost_rank_batch(df: pd.DataFrame) -> np.ndarray:
    """YetiRankバッチ予測。スコアをレース内min-max正規化して[0,1]に変換。"""
    if not RANK_MODEL_PATH.exists():
        return np.zeros(len(df))
    obj          = joblib.load(RANK_MODEL_PATH)
    model        = obj["model"]
    feature_cols = obj["feature_cols"]
    cat_features_list = [
        "種牡馬", "父タイプ名", "母父馬", "母父タイプ名", "毛色",
        "馬主(最新/仮想)", "生産者",
        "芝・ダ", "コース区分", "芝(内・外)", "馬場状態", "天気",
        "クラス名", "場所", "性別", "斤量", "ブリンカー", "重量種別",
        "年齢限定", "限定", "性別限定", "指定条件",
        "前走場所", "前芝・ダ", "前走馬場状態", "前走斤量", "前好走",
    ]
    _ROLLING_COLS = {"jockey_fuku30","jockey_fuku90","trainer_fuku30","trainer_fuku90",
                     "horse_fuku10","horse_fuku30","前走補9","前走補正",
                     "trn_hanro_4f","trn_hanro_lap1","trn_hanro_days",
                     "trn_wc_3f","trn_wc_lap1","trn_wc_days","前走単勝オッズ"}
    df = df.reset_index(drop=True).copy()
    for col in ["前走走破タイム", "前走着差タイム"]:
        if col in df.columns:
            df[col] = parse_time_str(df[col])
    for col in cat_features_list:
        df[col] = df[col].fillna("__NaN__").astype(str) if col in df.columns else "__NaN__"
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan if col in _ROLLING_COLS else 0.0
    cat_indices = [i for i, c in enumerate(feature_cols) if c in cat_features_list]
    pool   = Pool(df[feature_cols], cat_features=cat_indices)
    scores = model.predict(pool)
    # レース内min-max正規化
    result = np.full(len(df), 0.5)
    for _, group in df.groupby(COL_RACE_ID):
        idx  = group.index.tolist()
        s    = scores[idx]
        smin, smax = s.min(), s.max()
        if smax > smin:
            result[idx] = (s - smin) / (smax - smin)
    return result


def predict_transformer_batch(df: pd.DataFrame) -> np.ndarray:
    """Transformer予測。モデル未存在 or 失敗時はゼロ配列を返す。"""
    try:
        import torch
        from train_transformer import RaceTransformer, RaceDataset, MAX_HORSES
        from train_transformer import preprocess as torch_preprocess
        from torch.utils.data import DataLoader

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not TORCH_MODEL_PATH.exists():
            logger.warning(f"Transformerモデルが見つかりません: {TORCH_MODEL_PATH}")
            return np.zeros(len(df))
        obj          = joblib.load(TORCH_MODEL_PATH)
        model_state  = obj["model_state"]
        model_config = obj["model_config"]
        encoders     = obj["encoders"]
        num_stats    = obj["num_stats"]
        num_cols     = obj["num_cols"]
        cat_cols     = obj["cat_cols"]

        df = df.copy()
        if "fukusho_flag" not in df.columns:
            df["fukusho_flag"] = 0
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

        ds     = RaceDataset(df, cat_cols, num_cols, model_config["cat_vocab_sizes"])
        loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)

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
    except Exception as e:
        logger.warning(f"Transformer予測失敗（0で埋め）: {e}")
        return np.zeros(len(df))


def predict_stacking_batch(df: pd.DataFrame) -> np.ndarray | None:
    """スタッキング予測。モデル未存在 or torch未インストール or 失敗時は None を返す。"""
    if not META_PATH.exists() or not TORCH_MODEL_PATH.exists():
        return None
    try:
        import torch  # noqa: F401
    except ImportError:
        logger.info("torchが見つかりません。スタッキングをスキップして3モデルアンサンブルへ。")
        return None
    try:
        p_lgbm  = predict_lgbm_batch(df)
        p_cat   = predict_catboost_batch(df)
        p_torch = predict_transformer_batch(df)

        meta_obj      = joblib.load(META_PATH)
        meta_model    = meta_obj["meta_model"]
        meta_encoders = meta_obj["meta_encoders"]
        meta_cols     = meta_obj["meta_cols"]

        meta_df = df.copy()
        if "出走頭数" not in meta_df.columns or meta_df["出走頭数"].isna().all():
            meta_df["出走頭数"] = meta_df.groupby(COL_RACE_ID)["馬番"].transform("count")
        meta_df["クラス名"] = meta_df["クラス名"].map(CLASS_NORMALIZE).fillna(meta_df["クラス名"])
        meta_df = meta_df.reindex(columns=_META_EXTRA).copy()

        for col in meta_df.select_dtypes(include="object").columns:
            if col in meta_encoders:
                le = meta_encoders[col]
                meta_df[col] = meta_df[col].fillna("__NaN__").astype(str)
                known = set(le.classes_)
                meta_df[col] = meta_df[col].apply(lambda x: x if x in known else "__NaN__")
                if "__NaN__" not in le.classes_:
                    le.classes_ = np.append(le.classes_, "__NaN__")
                meta_df[col] = le.transform(meta_df[col])

        meta_df["lgbm"]        = p_lgbm
        meta_df["catboost"]    = p_cat
        meta_df["transformer"] = p_torch

        return meta_model.predict_proba(meta_df[meta_cols])[:, 1]
    except Exception as e:
        logger.warning(f"スタッキング予測失敗（フォールバック）: {e}")
        return None


def ensemble_predict_batch(df: pd.DataFrame) -> np.ndarray:
    """predict_weekly.py / app.py と同じスタッキング優先フロー。"""
    # スタッキング優先
    logger.info("スタッキング予測中...")
    stacking = predict_stacking_batch(df)
    if stacking is not None:
        if STACK_CAL_PATH.exists():
            cal_obj = joblib.load(STACK_CAL_PATH)
            logger.info("スタッキング + キャリブレーション適用")
            return cal_obj["calibrator"].transform(stacking)
        return stacking

    # Phase 5+: predict_weekly.ensemble_predict を使う（最適化重み + Expert別重み + MoE）
    try:
        from predict_weekly import ensemble_predict as pw_ensemble, _get_cached, LGBM_PATH as PW_LGBM, CAT_PATH as PW_CAT
        lgbm_obj = _get_cached(PW_LGBM, "lgbm_opt")
        cat_obj  = _get_cached(PW_CAT,  "cat_opt")
        if lgbm_obj is not None and cat_obj is not None:
            logger.info("Phase 5+ ensemble_predict (最適化重み+MoE) を使用 - レース単位ループ")
            race_col = "レースID(新/馬番無)"
            out = np.zeros(len(df))
            for rid, idx in df.groupby(race_col, sort=False).indices.items():
                sub = df.iloc[idx]
                p = pw_ensemble(sub, lgbm_obj, cat_obj)
                out[idx] = p
            return out
    except Exception as e:
        logger.warning(f"Phase 5+ ensemble失敗→旧4モデルにフォールバック: {e}")

    # 旧: 4モデル加重平均: LGBM×0.30 + CatBoost×0.30 + Rank×0.20 + Win×0.20
    logger.info("LightGBM予測中...")
    p_lgbm = predict_lgbm_batch(df)
    logger.info("CatBoost予測中...")
    p_cat  = predict_catboost_batch(df)
    if RANK_MODEL_PATH.exists() and WIN_MODEL_PATH.exists():
        logger.info("YetiRank予測中...")
        p_rank = predict_catboost_rank_batch(df)
        logger.info("lgbm_win_v1予測中...")
        p_win  = predict_lgbm_win_batch(df)
        raw = 0.30 * p_lgbm + 0.30 * p_cat + 0.20 * p_rank + 0.20 * p_win
        logger.info("4モデルアンサンブル（LGBM×0.30 + CatBoost×0.30 + Rank×0.20 + Win×0.20）")
    elif RANK_MODEL_PATH.exists():
        logger.info("YetiRank予測中...")
        p_rank = predict_catboost_rank_batch(df)
        raw = 0.4 * p_lgbm + 0.4 * p_cat + 0.2 * p_rank
        logger.info("3モデルアンサンブル（LGBM 0.4 + CatBoost 0.4 + Rank 0.2）")
    else:
        raw = 0.5 * p_lgbm + 0.5 * p_cat
        logger.info("2モデルアンサンブル（LGBM 0.5 + CatBoost 0.5）")
    # キャリブレーター: v4 → v3 → v2 → v1 の優先チェーン
    for cal_p in [CAL_PATH, CAL_PATH_V3, CAL_PATH_V2, CAL_PATH_V1]:
        if cal_p.exists():
            cal_obj = joblib.load(cal_p)
            logger.info(f"キャリブレーター適用: {cal_p.name}")
            return cal_obj["calibrator"].transform(raw)
    logger.warning("キャリブレーター未生成。calibrate.py を先に実行してください。")
    return raw


def load_odds(year: int) -> dict:
    """odds_YYYY*.csv から {レースID(新): {馬番: 単勝オッズ}} を返す。"""
    pattern = list(ODDS_CSV_DIR.glob(f"odds_{year}*.csv"))
    if not pattern:
        logger.warning(f"odds CSV が見つかりません（year={year}）。EV計算をスキップします。")
        return {}
    try:
        dfs = []
        for p in pattern:
            for enc in ("utf-8-sig", "cp932"):
                try:
                    dfs.append(pd.read_csv(p, encoding=enc, low_memory=False))
                    break
                except UnicodeDecodeError:
                    continue
        if not dfs:
            return {}
        odds_df = pd.concat(dfs, ignore_index=True)
        # 列名を標準化
        col_race = next((c for c in odds_df.columns if "レースID" in c), None)
        col_uma  = next((c for c in odds_df.columns if "馬番" in c), None)
        col_tan  = next((c for c in odds_df.columns if "単勝" in c), None)
        if not all([col_race, col_uma, col_tan]):
            logger.warning(f"odds CSV の列が不足: {odds_df.columns.tolist()}")
            return {}
        # レースID列を int64 経由で文字列変換（float精度損失を回避）
        odds_df[col_race] = (
            pd.to_numeric(odds_df[col_race], errors="coerce")
            .fillna(0).astype("int64").astype(str)
        )
        result: dict = {}
        for _, row in odds_df.iterrows():
            # レースID(新) は 18桁（16桁 race ID + 2桁 zero-padded 馬番）
            # COL_RACE_ID = レースID(新/馬番無) は 16桁
            rid  = str(row[col_race])[:16]
            uma  = int(row[col_uma]) if pd.notna(row[col_uma]) else None
            odds = float(row[col_tan]) if pd.notna(row[col_tan]) else None
            if uma and odds:
                result.setdefault(rid, {})[uma] = odds
        logger.info(f"odds 読み込み: {len(result)} レース")
        return result
    except Exception as e:
        logger.warning(f"odds CSV 読み込み失敗: {e}")
        return {}


def assign_marks_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mark"] = ""
    for _, group in df.groupby(COL_RACE_ID, sort=False):
        ranked = group["prob"].rank(ascending=False, method="first")
        marks  = {1: "◎", 2: "◯", 3: "▲", 4: "△", 5: "×"}
        for idx, rank in ranked.items():
            if rank <= 5:
                df.at[idx, "mark"] = marks[int(rank)]
    return df


# =========================================================
# 1レース分の処理
# =========================================================
def process_one_race(
    race_df: pd.DataFrame,
    kekka_entry: dict,
    budget: int = BUDGET,
    bet_info: dict | None = None,
    ev_threshold: float = 0.0,
    exclude_bets: set | None = None,
    race_odds: dict | None = None,
) -> list[dict]:
    import itertools

    prob_series = race_df.set_index("馬番")["prob"]
    mark_dict   = race_df.set_index("馬番")["mark"].to_dict()

    hon    = race_df[race_df["mark"] == "◎"]["馬番"].tolist()
    taikou = race_df[race_df["mark"] == "◯"]["馬番"].tolist()
    sabo   = race_df[race_df["mark"] == "▲"]["馬番"].tolist()
    top3   = hon + taikou + sabo

    def make_bet(combo, bet_type, ordered):
        p    = min(calc_win_prob_pl(list(combo), prob_series, ordered), 0.99)
        odds = estimate_odds(p, bet_type)
        sep  = "→" if ordered else "-"
        return {
            "買い目":      sep.join(map(str, combo)),
            "combo":       list(combo),
            "ordered":     ordered,
            "bet_type":    bet_type,
            "推定的中確率": round(p, 4),
            "推定オッズ":   odds,
            "推定期待値":   round(p * odds, 3),
        }

    candidates: dict[str, list[dict]] = {
        "単勝": [], "複勝": [], "枠連": [], "馬連": [], "三連複": [],
    }

    # 単勝: ◎のみ
    if hon:
        candidates["単勝"].append(make_bet([hon[0]], "単勝", False))

    for h in hon + taikou:
        candidates["複勝"].append(make_bet([h], "複勝", False))

    # 枠連: ◎枠番 × ◯▲枠番（重複枠番はスキップ）
    if hon and "枠番" in race_df.columns:
        waku_map = race_df.set_index("馬番")["枠番"].dropna().to_dict()
        hon_waku = waku_map.get(hon[0])
        if hon_waku:
            seen_wk = set()
            for h in taikou + sabo:
                h_waku = waku_map.get(h)
                if h_waku:
                    wk = tuple(sorted([int(hon_waku), int(h_waku)]))
                    if wk not in seen_wk:
                        seen_wk.add(wk)
                        p_est = min(calc_win_prob_pl([hon[0], h], prob_series, False) * 1.2, 0.99)
                        candidates["枠連"].append({
                            "買い目":      f"{wk[0]}-{wk[1]}",
                            "combo":       list(wk),
                            "ordered":     False,
                            "bet_type":    "枠連",
                            "推定的中確率": round(p_est, 4),
                            "推定オッズ":   estimate_odds(p_est / 1.2, "枠連"),
                            "推定期待値":   round(p_est, 3),
                        })

    if hon:
        for h in taikou + sabo:
            candidates["馬連"].append(
                make_bet(tuple(sorted([hon[0], h])), "馬連", False)
            )
    if len(top3) >= 3:
        rows = [make_bet(c, "三連複", False) for c in itertools.combinations(top3[:4], 3)]
        candidates["三連複"] = sorted(rows, key=lambda x: x["推定期待値"], reverse=True)[:3]

    # EV フィルタ: ◎の単勝 EV が閾値未満なら 単勝 をキャンセル
    if ev_threshold > 0.0 and hon and race_odds is not None:
        hon_odds = race_odds.get(hon[0])
        hon_prob = prob_series.get(hon[0], 0.0)
        if hon_odds and hon_prob > 0:
            ev = hon_prob * hon_odds / RETURN_RATE["単勝"]
            if ev < ev_threshold:
                candidates["単勝"] = []
        # ev_score を記録しておく（結果出力用）
        race_df = race_df.copy()
        if "ev_score" not in race_df.columns:
            race_df["ev_score"] = 0.0
        for idx, row in race_df.iterrows():
            uma = int(row["馬番"]) if pd.notna(row.get("馬番")) else None
            if uma and race_odds.get(uma) and prob_series.get(uma, 0) > 0:
                race_df.at[idx, "ev_score"] = round(
                    prob_series[uma] * race_odds[uma] / RETURN_RATE["単勝"], 3
                )

    # exclude_bets フィルタ
    if exclude_bets:
        for bt in exclude_bets:
            candidates.pop(bt, None)

    # Phase 5+: SegmentBetFilter (ROI<80% の (距離セグメント, 券種) を除外)
    SEGMENT_BET_BLACKLIST = {
        ("dirt",       "三連複"),
        ("turf_short", "馬連"),
        ("turf_short", "複勝"),
        ("turf_mid",   "馬連"),
    }
    try:
        _r0 = race_df.iloc[0]
        _td = str(_r0.get("芝・ダ", _r0.get("芝ダ", "")))
        _d  = int(_r0.get("距離", 0))
        if _td == "ダ":
            _seg = "dirt"
        elif _d <= 1400:
            _seg = "turf_short"
        elif _d <= 2200:
            _seg = "turf_mid"
        else:
            _seg = "turf_long"
        for bt in list(candidates.keys()):
            if (_seg, bt) in SEGMENT_BET_BLACKLIST:
                candidates[bt] = []
    except Exception:
        pass

    # 予算按分（戦略の bet_ratio を使用、未指定時は BUDGET_RATIO）
    filtered_bet_info = None
    if bet_info is not None:
        filtered_bet_info = {k: v for k, v in bet_info.items()
                             if k in candidates and len(candidates[k]) > 0}
        if not filtered_bet_info:
            return []   # 全馬券種が除外 → BUDGET_RATIO への誤フォールバック防止
    if filtered_bet_info:
        ratios = {k: v.get("bet_ratio", 0) for k, v in filtered_bet_info.items()
                  if k in candidates and len(candidates[k]) > 0}
    else:
        ratios = {k: BUDGET_RATIO[k] for k in BUDGET_RATIO
                  if k in candidates and len(candidates[k]) > 0}
    active      = {k: candidates[k] for k in ratios}
    total_ratio = sum(ratios.values())
    if total_ratio <= 0:
        return []
    for bet_type, bets in active.items():
        n       = len(bets)
        alloc   = floor_to_unit(int(budget * ratios[bet_type] / total_ratio))
        per_bet = max(floor_to_unit(alloc // n), MIN_UNIT)
        while per_bet * n > alloc and per_bet > MIN_UNIT:
            per_bet -= MIN_UNIT
        for b in bets:
            b["購入額"] = per_bet

    total_used = sum(b["購入額"] for bets in active.values() for b in bets)
    remainder  = floor_to_unit(budget - int(total_used))
    if remainder >= MIN_UNIT and "三連複" in active and active["三連複"]:
        n     = len(active["三連複"])
        extra = floor_to_unit(remainder // n)
        if extra >= MIN_UNIT:
            for b in active["三連複"]:
                b["購入額"] += extra

    # 的中判定（実払戻）
    results  = []
    race_id  = race_df[COL_RACE_ID].iloc[0]
    race_info = race_df.iloc[0]

    for bet_type, bets in active.items():
        for b in bets:
            payout_per100 = get_actual_payout(
                b["combo"], b["ordered"], bet_type, kekka_entry
            )
            hit      = payout_per100 > 0
            # 実払戻額 = 購入額 × (実配当 / 100)
            actual_pay = int(b["購入額"] * payout_per100 / 100) if hit else 0

            # EV スコア（単勝◎のみ意味あり）
            ev_score = 0.0
            if bet_type == "単勝" and hon and race_odds is not None:
                hon_odds = race_odds.get(hon[0])
                hon_prob = prob_series.get(hon[0], 0.0)
                if hon_odds and hon_prob > 0:
                    ev_score = round(hon_prob * hon_odds / RETURN_RATE["単勝"], 3)

            results.append({
                "race_id":     race_id,
                "日付":        race_info.get("日付", ""),
                "場所":        race_info.get("場所", ""),
                "距離":        race_info.get("距離", ""),
                "芝ダ":        race_info.get("芝・ダ", ""),
                "クラス":      race_info.get("クラス名", ""),
                "馬券種":      bet_type,
                "買い目":      b["買い目"],
                "推定的中確率": b["推定的中確率"],
                "推定オッズ":   b["推定オッズ"],
                "推定期待値":   b["推定期待値"],
                "乖離スコア":   ev_score,
                "購入額":      b["購入額"],
                "実配当(100円)": payout_per100,
                "実オッズ":    round(payout_per100 / 100, 1) if payout_per100 else 0,
                "的中":        int(hit),
                "実払戻額":    actual_pay,
                "収支":        actual_pay - b["購入額"],
            })

    return results


# =========================================================
# 集計・可視化
# =========================================================
def summarize(df: pd.DataFrame) -> None:
    n_races = df["race_id"].nunique()

    print("\n" + "=" * 70)
    print(f"バックテスト結果サマリ（実オッズ版）  {n_races:,}レース")
    print("=" * 70)

    total_cost = df["購入額"].sum()
    total_pay  = df["実払戻額"].sum()
    total_net  = total_pay - total_cost
    roi        = total_pay / total_cost * 100 if total_cost > 0 else 0

    print(f"\n【全体】")
    print(f"  総投資額  : {total_cost:>12,}円")
    print(f"  総払戻額  : {total_pay:>12,}円")
    print(f"  純収支    : {total_net:>+12,}円")
    print(f"  回収率    : {roi:>11.1f}%")

    print(f"\n【馬券種別】")
    print(f"{'馬券種':<6} {'投資':>10} {'払戻':>10} {'収支':>10} {'回収率':>8} {'的中率':>8} {'的中数':>6} {'点数':>6}")
    print("-" * 70)

    for bet_type in ["単勝", "複勝", "枠連", "馬連", "三連複"]:
        sub = df[df["馬券種"] == bet_type]
        if sub.empty:
            continue
        cost   = sub["購入額"].sum()
        pay    = sub["実払戻額"].sum()
        net    = pay - cost
        r      = pay / cost * 100 if cost > 0 else 0
        hits   = sub["的中"].sum()
        total_b = len(sub)
        hit_r  = hits / total_b * 100 if total_b > 0 else 0
        print(
            f"{bet_type:<6} {cost:>10,} {pay:>10,} {net:>+10,} "
            f"{r:>7.1f}% {hit_r:>7.1f}% {hits:>6,} {total_b:>6,}"
        )

    # EV スコア別 ROI（単勝のみ）
    tan = df[(df["馬券種"] == "単勝") & (df["乖離スコア"] > 0)].copy() if "乖離スコア" in df.columns else pd.DataFrame()
    if not tan.empty:
        print(f"\n【単勝 EV（乖離スコア）別 ROI】")
        print(f"{'EV閾値':<10} {'点数':>6} {'投資':>10} {'払戻':>10} {'ROI':>8}")
        print("-" * 50)
        for thr in [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0]:
            sub = tan[tan["乖離スコア"] >= thr]
            if sub.empty:
                continue
            c = sub["購入額"].sum()
            p = sub["実払戻額"].sum()
            r = p / c * 100 if c > 0 else 0
            flag = " ★★" if r >= 150 else " ★" if r >= 110 else ""
            print(f">= {thr:<7.1f} {len(sub):>6,} {c:>10,} {p:>10,} {r:>7.1f}%{flag}")


def plot_cumulative(df: pd.DataFrame, save_path: Path) -> None:
    race_pnl = (
        df.groupby("race_id")
        .agg(収支=("収支", "sum"), 投資=("購入額", "sum"))
        .reset_index()
    )
    race_pnl["累積収支"]     = race_pnl["収支"].cumsum()
    race_pnl["累積投資"]     = race_pnl["投資"].cumsum()
    race_pnl["累積回収率(%)"] = (
        (race_pnl["累積投資"] + race_pnl["累積収支"]) /
        race_pnl["累積投資"] * 100
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    axes[0].plot(race_pnl.index, race_pnl["累積収支"], color="steelblue")
    axes[0].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[0].set_title("累積収支推移（実オッズ）")
    axes[0].set_ylabel("累積収支（円）")
    axes[0].set_xlabel("レース数")

    axes[1].plot(race_pnl.index, race_pnl["累積回収率(%)"], color="tomato")
    axes[1].axhline(100, color="gray", linewidth=0.8, linestyle="--")
    axes[1].set_title("累積回収率推移（実オッズ）")
    axes[1].set_ylabel("累積回収率（%）")
    axes[1].set_xlabel("レース数")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"グラフ保存: {save_path}")


def plot_roi_by_category(df: pd.DataFrame, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, col, title in zip(
        axes,
        ["場所", "クラス", "芝ダ"],
        ["競馬場別回収率", "クラス別回収率", "芝ダート別回収率"],
    ):
        grp = df.groupby(col).agg(
            投資=("購入額", "sum"),
            払戻=("実払戻額", "sum"),
        ).reset_index()
        grp["回収率"] = grp["払戻"] / grp["投資"] * 100
        grp = grp.sort_values("回収率", ascending=True)
        colors = ["tomato" if v >= 100 else "steelblue" for v in grp["回収率"]]
        ax.barh(grp[col], grp["回収率"], color=colors)
        ax.axvline(100, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("回収率（%）")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"グラフ保存: {save_path}")


# =========================================================
# main
# =========================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="PyCaLiAI バックテスト（実オッズ版）")
    parser.add_argument("--n_races",       type=int, default=None)
    parser.add_argument("--budget",        type=int, default=BUDGET)
    parser.add_argument("--output_suffix", type=str, default="",
                        help="出力ファイル名のサフィックス（例: _train）")
    parser.add_argument("--no_strategy", action="store_true",
                        help="戦略フィルタを無効化（strategy_weights.json 再構築用）")
    parser.add_argument("--period", type=str, default=None,
                        choices=["train", "valid", "test"],
                        help="対象期間 (train: 〜2022年, valid: 2023年, test: 2024年〜)")
    parser.add_argument("--ev_threshold", type=float, default=0.0,
                        help="単勝 EV 下限（例: 1.0）。0 = フィルタなし")
    parser.add_argument("--exclude_bets", type=str, default="",
                        help="除外馬券種（カンマ区切り, 例: 複勝,枠連）")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # データロード
    logger.info(f"マスターCSV読み込み: {MASTER_CSV}")
    df      = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df["日付"] = pd.to_numeric(df["日付"], errors="coerce")
    # --period が明示されれば優先、なければ output_suffix で判定（後方互換）
    period = args.period or ("train" if "_train" in args.output_suffix
                             else "valid" if "_valid" in args.output_suffix
                             else "test")
    if period == "train":
        test_df = df[df["日付"] < 20230101].copy().reset_index(drop=True)
        logger.info("対象期間: 〜2022年（発見期）")
    elif period == "valid":
        test_df = df[df["split"] == "valid"].copy().reset_index(drop=True)
        logger.info("対象期間: 2023年（Valid セット）")
    else:
        test_df = df[df["split"] == "test"].copy().reset_index(drop=True)
        logger.info("対象期間: テストデータ（2024年〜）")
    logger.info(f"テストデータ（全体）: {len(test_df):,}行")

    # 除外フィルタ（app.py / predict_weekly.py と統一）
    if "場所" in test_df.columns and "クラス名" in test_df.columns:
        before = len(test_df)
        test_df = test_df[
            ~test_df["場所"].isin(EXCLUDE_PLACES) &
            ~test_df["クラス名"].isin(EXCLUDE_CLASSES)
        ].copy().reset_index(drop=True)
        logger.info(
            f"  除外フィルタ後: {len(test_df):,}行 "
            f"（除外: {before - len(test_df):,}行 / "
            f"会場={sorted(EXCLUDE_PLACES)} クラス={sorted(EXCLUDE_CLASSES)}）"
        )

    # 払戻辞書ロード
    kekka_dict = load_kekka(KEKKA_CSV)

    # オッズ読み込み（EV計算用）
    exclude_bets_set: set = set(filter(None, args.exclude_bets.split(","))) if args.exclude_bets else set()
    # 対象年リストを決定
    if period == "train":
        odds_years = list(range(2013, 2023))
    elif period == "valid":
        odds_years = [2023]
    else:
        odds_years = [2024]
    odds_dict: dict = {}
    if args.ev_threshold > 0.0:
        for yr in odds_years:
            odds_dict.update(load_odds(yr))
        logger.info(f"EV閾値: {args.ev_threshold} / 除外馬券種: {exclude_bets_set or 'なし'}")

    # アンサンブル予測
    logger.info("アンサンブル予測開始...")
    test_df["prob"] = ensemble_predict_batch(test_df)

    # 印付与
    logger.info("印付与中...")
    test_df = assign_marks_df(test_df)

    # レース単位処理
    race_ids = test_df[COL_RACE_ID].unique()
    if args.n_races:
        race_ids = race_ids[:args.n_races]
    logger.info(f"バックテスト開始: {len(race_ids):,}レース")

    # 戦略フィルタ読み込み
    strategy: dict = {}
    if args.no_strategy:
        logger.info("--no_strategy: 戦略フィルタ無効（全レース対象）")
    elif STRATEGY_JSON.exists():
        with open(STRATEGY_JSON, encoding="utf-8") as f:
            strategy = json.load(f)
        logger.info(f"戦略ファイル読み込み: {STRATEGY_JSON.name}")
    else:
        logger.warning(f"strategy_weights.json が見つかりません。全レースを対象にします。")

    all_results = []
    skipped     = 0
    skipped_strategy = 0
    skipped_filter   = 0
    _bet_filter      = BetFilter()
    for race_id in tqdm(race_ids, desc="バックテスト"):
        race_df     = test_df[test_df[COL_RACE_ID] == race_id].copy()

        # 戦略フィルタ（predict_weekly.py と同じ場所×クラス判定）
        current_bet_info = None
        if strategy:
            place   = race_df["場所"].iloc[0] if "場所" in race_df.columns else ""
            cls_raw = race_df["クラス名"].iloc[0] if "クラス名" in race_df.columns else ""
            cls     = CLASS_NORMALIZE.get(cls_raw, cls_raw)
            current_bet_info = strategy.get(place, {}).get(cls) or strategy.get(place, {}).get(cls_raw, {})
            if not current_bet_info:
                skipped_strategy += 1
                continue

        # BetFilter（HELL セグメント除外・EV 上限）
        place    = race_df["場所"].iloc[0] if "場所" in race_df.columns else ""
        n_horses = len(race_df)
        baba     = str(race_df["馬場状態"].iloc[0]) if "馬場状態" in race_df.columns else ""
        # ◎候補（prob 最大馬）の EV を暫定スコアとして使用
        hon_row  = race_df.loc[race_df["prob"].idxmax()] if "prob" in race_df.columns else None
        hon_ev   = 0.0  # odds 未取得のため EV=0（EV 上限チェックは predict_weekly.py に委ねる）
        cls_now  = pd.to_numeric(race_df["クラス区分"].iloc[0]
                                  if "クラス区分" in race_df.columns else None, errors="coerce")
        cls_prev = pd.to_numeric(
            hon_row["前走クラスコード"] if hon_row is not None and "前走クラスコード" in race_df.columns else None,
            errors="coerce",
        )
        upgrade  = is_upgrade_race(cls_now, cls_prev)
        f_result = _bet_filter.check(place=place, n_horses=n_horses,
                                     baba=baba, ev=hon_ev, is_upgrade=upgrade)
        if f_result.should_skip:
            skipped_filter += 1
            continue

        kekka_entry = kekka_dict.get(str(race_id), {
            "単勝": {}, "複勝": {}, "枠連": {}, "馬連": {}, "馬単": {}, "三連複": {}, "三連単": {}
        })
        try:
            race_odds = odds_dict.get(str(race_id)) if odds_dict else None
            results = process_one_race(
                race_df, kekka_entry,
                budget=args.budget,
                bet_info=current_bet_info,
                ev_threshold=args.ev_threshold,
                exclude_bets=exclude_bets_set,
                race_odds=race_odds,
            )
            all_results.extend(results)
        except Exception as e:
            logger.warning(f"レース {race_id} スキップ: {e}")
            skipped += 1

    logger.info(f"戦略対象外スキップ: {skipped_strategy}レース")
    logger.info(f"BetFilterスキップ:   {skipped_filter}レース")
    logger.info(f"エラースキップ: {skipped}レース")

    if not all_results:
        logger.error("結果が0件です。")
        return

    result_df = pd.DataFrame(all_results)

    # 集計
    summarize(result_df)

    # グラフ


    # CSV保存
    out_path = REPORT_DIR / f"backtest_results{args.output_suffix}.csv"
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"結果保存: {out_path}")

    print(f"\n保存ファイル:")
    print(f"  {out_path}")
    plot_cumulative(result_df, REPORT_DIR / f"backtest_cumulative{args.output_suffix}.png")
    plot_roi_by_category(result_df, REPORT_DIR / f"backtest_roi_by_category{args.output_suffix}.png")


if __name__ == "__main__":
    main()