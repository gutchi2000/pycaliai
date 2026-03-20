"""
backtest_longterm_ev.py
PyCaLiAI - テストセット全体（2024年〜）での長期EVバックテスト

master CSVのtest splitに対してアンサンブル予測を実行し、
kekka CSVの単勝配当・複勝配当とJOINして
「EVスコア（複勝確率×単勝オッズ）」の閾値ごとのROIを分析する。

Usage:
    python backtest_longterm_ev.py
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    plt.rcParams["font.family"] = "MS Gothic"

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from utils import parse_time_str

BASE_DIR   = Path(r"E:\PyCaLiAI")
DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MASTER_CSV   = DATA_DIR / "master_20130105-20251228.csv"
HOSEI_DIR    = DATA_DIR / "hosei"
KEKKA_CSV    = DATA_DIR / "kekka_20130105-20251228.csv"
LGBM_PATH    = MODEL_DIR / "lgbm_optuna_v1.pkl"
CAT_PATH     = MODEL_DIR / "catboost_optuna_v1.pkl"
RANK_PATH    = MODEL_DIR / "catboost_rank_v1.pkl"
TORCH_PATH   = MODEL_DIR / "transformer_pl_v2.pkl"
CAL_PATH     = MODEL_DIR / "ensemble_calibrator_v1.pkl"

TARGET     = "fukusho_flag"
BUDGET     = 10_000
RANDOM_STATE = 42

_model_cache: dict = {}


# =========================================================
# 予測関数
# =========================================================
def _predict_lgbm(df: pd.DataFrame, obj: dict) -> np.ndarray:
    from catboost import Pool
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
    import lightgbm as lgb
    if isinstance(model, lgb.Booster):
        return model.predict(df[feature_cols])
    return model.predict_proba(df[feature_cols])[:, 1]


def _predict_catboost(df: pd.DataFrame, obj: dict) -> np.ndarray:
    from catboost import Pool
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


def _predict_catboost_rank(df: pd.DataFrame, obj: dict) -> np.ndarray:
    from catboost import Pool
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

    race_id_col = "レースID(新/馬番無)"
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
    if not TORCH_PATH.exists():
        return np.zeros(len(df))
    if "torch" not in _model_cache:
        _model_cache["torch"] = joblib.load(TORCH_PATH)
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
        if TARGET not in df_copy.columns:
            df_copy[TARGET] = 0
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

        race_id_col = "レースID(新/馬番無)"
        result  = np.zeros(len(df))
        df_sort = df.sort_values(race_id_col).reset_index(drop=True)
        idx = 0
        for _, group in df_sort.groupby(race_id_col, sort=True):
            valid_n = min(len(group), PL_MAX_HORSES)
            for orig_idx in list(group.index)[:valid_n]:
                if idx < len(all_scores):
                    result[orig_idx] = all_scores[idx]
                    idx += 1

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
        logger.warning(f"Transformer PL予測失敗: {e}")
        return np.zeros(len(df))


# =========================================================
# kekka CSV の単勝オッズ読み込み
# =========================================================
def load_kekka_odds() -> pd.DataFrame:
    """kekka_20130105-20251228.csv から単勝配当・複勝配当を読み込む。"""
    logger.info(f"kekka CSV読み込み: {KEKKA_CSV}")
    kekka = pd.read_csv(KEKKA_CSV, encoding="cp932")
    kekka["レースキー"] = kekka["レースID(新)"].astype(str).str.zfill(18).str[:16]
    kekka["馬番"]       = pd.to_numeric(kekka["馬番"], errors="coerce")

    # 単勝配当: "570"(円) or "(3.7)"(倍率) の2形式 → 倍率に統一
    def _to_odds(val):
        s = str(val).strip()
        if s.startswith("(") and s.endswith(")"):
            # すでに倍率（例: "(3.7)" → 3.7）
            try: return float(s[1:-1])
            except: return None
        try:
            v = float(s)
            return v / 100.0  # 円 → 倍率（100円あたり）
        except:
            return None

    def _parse_fuku(val):
        s = str(val).strip()
        if s.startswith("(") or s in ("nan", "", "None"):
            return None
        try: return float(s)
        except: return None

    kekka["単勝オッズ"]  = kekka["単勝配当"].apply(_to_odds)
    kekka["複勝配当_n"]  = kekka["複勝配当"].apply(_parse_fuku)
    kekka["着順_n"]      = pd.to_numeric(kekka["確定着順"], errors="coerce")

    logger.info(f"  kekka: {len(kekka):,}行 / {kekka['レースキー'].nunique():,}レース")
    return kekka[["レースキー", "馬番", "単勝オッズ", "複勝配当_n", "着順_n"]]


# =========================================================
# メイン
# =========================================================
def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ── データ読み込み ──
    logger.info("マスターCSV読み込み...")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)

    hosei_files = sorted(HOSEI_DIR.glob("H_*.csv"))
    if hosei_files:
        hosei = pd.concat([
            pd.read_csv(f, encoding="cp932",
                        usecols=["レースID(新)", "前走補9", "前走補正"])
            for f in hosei_files
        ], ignore_index=True).drop_duplicates()
        df = df.merge(hosei, on="レースID(新)", how="left")

    test = df[df["split"] == "test"].copy().reset_index(drop=True)
    logger.info(f"テストセット: {len(test):,}行 / {test['レースID(新/馬番無)'].nunique():,}レース")

    # ── モデル予測 ──
    logger.info("モデル読み込み...")
    lgbm_obj = joblib.load(LGBM_PATH)
    cat_obj  = joblib.load(CAT_PATH)
    rank_obj = joblib.load(RANK_PATH)
    cal_obj  = joblib.load(CAL_PATH)

    w_lgbm  = cal_obj.get("w_lgbm",  0.262)
    w_cat   = cal_obj.get("w_cat",   0.479)
    w_rank  = cal_obj.get("w_rank",  0.156)
    w_trans = cal_obj.get("w_trans", 0.103)

    logger.info("  LGBM予測...")
    p_lgbm = _predict_lgbm(test, lgbm_obj)
    logger.info("  CatBoost予測...")
    p_cat  = _predict_catboost(test, cat_obj)
    logger.info("  YetiRank予測...")
    p_rank = _predict_catboost_rank(test, rank_obj)
    logger.info("  Transformer PL予測...")
    p_trans = _predict_transformer(test)

    # アンサンブル → キャリブレーション
    ens_raw = w_lgbm*p_lgbm + w_cat*p_cat + w_rank*p_rank + w_trans*p_trans
    prob    = cal_obj["calibrator"].transform(ens_raw)
    test["prob"]  = prob
    test["score"] = (prob * 100).round(1)

    # ── roi_target 列を利用（kekka JOIN 不要）──
    # roi_target: 0.0=外れ, N.N=N.N倍払い戻し（例: 3.4→340円/100円）
    # この列はマスターCSVに事前計算済みでリークなし（確定払い戻し）
    test["fuku_hit"]  = pd.to_numeric(test["fukusho_flag"], errors="coerce").fillna(0).astype(int)
    test["roi_return"] = pd.to_numeric(test["roi_target"],   errors="coerce").fillna(0.0)
    # roi_return = 払い戻し倍率（0.0 or 例: 3.4倍）。ROI = mean(roi_return) * 100

    # ── レース内◎（最高スコア馬）を特定 ──
    race_col = "レースID(新/馬番無)"
    idx_hon = test.groupby(race_col)["score"].idxmax()
    test["is_hon"] = False
    test.loc[idx_hon, "is_hon"] = True
    hon = test[test["is_hon"]].copy()
    hon_roi = hon["roi_return"].mean() * 100
    logger.info(f"◎対象: {len(hon):,}レース  複勝率: {hon['fuku_hit'].mean()*100:.1f}%  ROI: {hon_roi:.1f}%")

    # ── スコア閾値別ROI（全馬・複勝） ──
    logger.info("\n--- スコア閾値別ROI（全馬、複勝） ---")
    score_results = []
    for sthr in range(0, 80, 5):
        subset = test[test["score"] >= sthr]
        if len(subset) < 50:
            break
        roi = subset["roi_return"].mean() * 100
        score_results.append({
            "スコア閾値": sthr,
            "買い数": len(subset),
            "複勝率(%)": subset["fuku_hit"].mean() * 100,
            "ROI(%)": roi,
            "損益(万円)": (roi - 100) / 100 * len(subset) * BUDGET / 10000,
        })
    df_score = pd.DataFrame(score_results)
    print("\n" + "=" * 65)
    print("スコア閾値別ROI（全馬複勝、testセット 2024年〜）")
    print("=" * 65)
    print(df_score.to_string(index=False, float_format="%.1f"))

    # ── スコア閾値別ROI（◎のみ） ──
    logger.info("\n--- スコア閾値別ROI（◎のみ、複勝） ---")
    hon_results = []
    for sthr in range(0, 80, 5):
        subset = hon[hon["score"] >= sthr]
        if len(subset) < 20:
            break
        roi = subset["roi_return"].mean() * 100
        hon_results.append({
            "スコア閾値": sthr,
            "買い数(レース)": len(subset),
            "複勝率(%)": subset["fuku_hit"].mean() * 100,
            "ROI(%)": roi,
            "損益(万円)": (roi - 100) / 100 * len(subset) * BUDGET / 10000,
        })
    df_hon = pd.DataFrame(hon_results)
    print("\n" + "=" * 65)
    print("スコア閾値別ROI（◎複勝のみ、testセット 2024年〜）")
    print("=" * 65)
    print(df_hon.to_string(index=False, float_format="%.1f"))

    # 最適閾値（ROI最大 かつ 買い数≥50）
    for df_tgt, label in [(df_score, "全馬"), (df_hon, "◎")]:
        col = "買い数" if "買い数" in df_tgt.columns else "買い数(レース)"
        valid = df_tgt[df_tgt[col] >= 50]
        if not valid.empty:
            best = valid.loc[valid["ROI(%)"].idxmax()]
            logger.info(
                f"最適閾値({label}): スコア≥{int(best['スコア閾値'])}  "
                f"ROI={best['ROI(%)']:.1f}%  {col}={int(best[col])}"
            )

    # ── グラフ ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(df_score["スコア閾値"], df_score["ROI(%)"], "b-o", markersize=5, label="全馬")
    if not df_hon.empty:
        ax.plot(df_hon["スコア閾値"], df_hon["ROI(%)"], "r-s", markersize=5, label="◎のみ")
    ax.axhline(80,  color="gray",   linestyle="--", alpha=0.5, label="80%")
    ax.axhline(100, color="orange", linestyle="--", alpha=0.7, label="100%（均衡）")
    ax.set_xlabel("スコア閾値")
    ax.set_ylabel("ROI (%)")
    ax.set_title("スコア閾値 vs ROI（複勝）")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(df_score["スコア閾値"], df_score["複勝率(%)"], "b-o", markersize=5, label="全馬")
    if not df_hon.empty:
        ax.plot(df_hon["スコア閾値"], df_hon["複勝率(%)"], "r-s", markersize=5, label="◎のみ")
    ax.axhline(33.3, color="gray", linestyle="--", alpha=0.5, label="33.3%（理論値）")
    ax.set_xlabel("スコア閾値")
    ax.set_ylabel("複勝率 (%)")
    ax.set_title("スコア閾値 vs 複勝率")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = REPORT_DIR / "backtest_longterm_ev.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"\nグラフ保存: {out_path}")

    # CSV保存
    df_score.to_csv(REPORT_DIR / "backtest_longterm_ev_results.csv", index=False, encoding="utf-8-sig")
    df_hon.to_csv(REPORT_DIR / "backtest_longterm_hon_results.csv",  index=False, encoding="utf-8-sig")
    logger.info(f"結果CSV保存: {REPORT_DIR / 'backtest_longterm_ev_results.csv'}")


if __name__ == "__main__":
    main()
