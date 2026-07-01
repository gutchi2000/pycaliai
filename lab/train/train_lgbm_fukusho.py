"""
train_lgbm_fukusho.py - 複勝特化LightGBMモデル v2 (Phase 5+ Sprint 1.1)

目的:
  既存 lgbm_fukusho_v1.pkl は train_lgbm.py の汎用版を fukusho_flag に置き換えただけで
  特徴量・パラメータが複勝特化していない。本スクリプトでは
  - 複勝(3着以内)に効く安定性系特徴量を優先
  - ノイズになりやすい列を削る
  - regularization 強め（過剰自信の抑制）
  - dirt / turf_mid / turf_long の3セグメントに学習対象を限定
    (turf_short × 複勝 は SegmentBetFilter で除外済のため学習しない)
  を行い、valid AUC が baseline を超えたら lgbm_fukusho_v2.pkl として採用判定する。

採用条件:
  valid_auc > BASELINE_VALID_AUC かつ test_auc も baseline を上回る
  → models/lgbm_fukusho_v2.pkl に保存
  → 採用しない場合は lgbm_fukusho_v2_rejected.pkl に保存しメトリクスのみ残す

  predict_weekly.py / app.py / optimize_weights.py を v2 に切り替えるかは
  バックテスト前後比較で判断する（このスクリプトでは自動切替しない）。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from train_lgbm import (
    preprocess,
    train_model,
    encode_categoricals,
    parse_time_str,
    TIME_STR_FEATURES,
    TARGET,
    COL_DATE,
    COL_RACE_ID,
    RANDOM_STATE,
    MASTER_CSV,
    MODEL_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUT_PATH        = MODEL_DIR / "lgbm_fukusho_v2.pkl"
REJECT_PATH     = MODEL_DIR / "lgbm_fukusho_v2_rejected.pkl"
METRICS_PATH    = Path(__file__).parent / "reports" / "fukusho_v2_metrics.json"

# 採用基準: 既存 fuku_lgbm の AUC（train_lgbm の通常版とほぼ同じ）
# train_lgbm.py 通常モデル valid_auc ≈ 0.7551, test_auc ≈ 0.7594
BASELINE_VALID_AUC = 0.7551
BASELINE_TEST_AUC  = 0.7594

# =========================================================
# 複勝特化特徴量セット
# =========================================================
# train_lgbm.py の ALL_FEATURES から複勝に効きやすいものを厳選
CAT_FEATURES_FUKU = [
    "芝・ダ", "コース区分", "馬場状態",
    "クラス名", "場所",
    "前走場所", "前芝・ダ", "前走馬場状態",
    "前好走", "性別",
]

NUM_FEATURES_FUKU = [
    # レース条件
    "距離", "出走頭数",
    # 馬プロフィール
    "枠番", "馬番", "年齢", "斤量体重比",
    "間隔", "休み明け～戦目",
    # 前走成績（複勝に直結）
    "前走確定着順", "前距離",
    "前走馬体重", "前走馬体重増減",
    # 前走上り（末脚指標は複勝率に効く）
    "前走上り3F", "前走上り3F順",
    "前走Ave-3F", "前PCI",
    # 騎手・調教師複勝率（最重要）
    "jockey_fuku30", "jockey_fuku90",
    "trainer_fuku30", "trainer_fuku90",
    # 馬の直近フォーム（複勝特化）
    "horse_fuku10", "horse_fuku30",
    # 派生
    "prev_pos_rel", "closing_power",
    # 補正タイム
    "前走補9", "前走補正",
    # 前走オッズ
    "前走単勝オッズ",
    # 過去5走集約（安定性）
    "kako5_avg_pos", "kako5_std_pos", "kako5_best_pos",
    "kako5_same_td_ratio", "kako5_same_dist_ratio", "kako5_same_place_ratio",
    "kako5_pos_trend", "kako5_race_count",
    # 好走3分類
    "kako5_expected_good_count", "kako5_upset_good_count", "kako5_hidden_good_count",
    "kako5_same_cond_best_pos",
    # 全キャリア同条件
    "hist_same_cond_best_pos", "hist_same_cond_top3_rate", "hist_same_cond_count",
]

ALL_FEATURES_FUKU = CAT_FEATURES_FUKU + NUM_FEATURES_FUKU + TIME_STR_FEATURES

# =========================================================
# 複勝特化 LightGBM パラメータ
# regularization 強め / depth 浅め
# =========================================================
LGBM_PARAMS_FUKU = {
    "objective":          "binary",
    "metric":             "auc",
    "learning_rate":      0.04,
    "num_leaves":         31,           # ← 63 から減
    "max_depth":          6,            # ← 制限追加
    "min_child_samples":  100,          # ← 50 から増
    "subsample":          0.8,
    "colsample_bytree":   0.7,          # ← 0.8 から減
    "reg_alpha":          0.1,          # L1
    "reg_lambda":         0.5,          # L2 強め
    "random_state":       RANDOM_STATE,
    "n_estimators":       3000,
    "verbose":            -1,
}


def filter_segments(df: pd.DataFrame) -> pd.DataFrame:
    """学習対象を dirt / turf_mid / turf_long に限定 (turf_short × 複勝 は除外)"""
    td = df["芝・ダ"]
    d  = pd.to_numeric(df["距離"], errors="coerce")
    mask = (td == "ダ") | ((td == "芝") & (d >= 1600))
    out = df[mask].copy()
    logger.info(f"セグメントフィルタ: {len(df):,} → {len(out):,} 行 (turf_short除外)")
    return out


def main() -> None:
    logger.info(f"マスターCSV読み込み: {MASTER_CSV}")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    logger.info(f"  {len(df):,}行 × {len(df.columns)}列")

    # セグメントフィルタ
    df = filter_segments(df)

    # split
    train_df = df[df["split"] == "train"].copy()
    valid_df = df[df["split"] == "valid"].copy()
    test_df  = df[df["split"] == "test"].copy()
    logger.info(f"Train={len(train_df):,} / Valid={len(valid_df):,} / Test={len(test_df):,}")

    # 前処理（preprocessは ALL_FEATURES_FUKU に対して動かす）
    # train_lgbm.preprocess は ALL_FEATURES グローバル参照なので、ここでは独自に encode する
    for col in TIME_STR_FEATURES:
        for d_ in (train_df, valid_df, test_df):
            if col in d_.columns:
                d_[col] = parse_time_str(d_[col])

    train_df, encoders = encode_categoricals(train_df, CAT_FEATURES_FUKU, fit=True)
    valid_df, _        = encode_categoricals(valid_df, CAT_FEATURES_FUKU, encoders=encoders, fit=False)
    test_df,  _        = encode_categoricals(test_df,  CAT_FEATURES_FUKU, encoders=encoders, fit=False)

    # 欠損特徴量を補完
    for col in ALL_FEATURES_FUKU:
        for d_ in (train_df, valid_df, test_df):
            if col not in d_.columns:
                d_[col] = np.nan

    # train
    import lightgbm as lgb
    X_tr, y_tr = train_df[ALL_FEATURES_FUKU], train_df[TARGET]
    X_va, y_va = valid_df[ALL_FEATURES_FUKU], valid_df[TARGET]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dvalid = lgb.Dataset(X_va, label=y_va, reference=dtrain)

    logger.info("LightGBM 学習開始（複勝特化 v2）...")
    model = lgb.train(
        LGBM_PARAMS_FUKU,
        dtrain,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
    )

    # 評価
    p_va = model.predict(X_va)
    p_te = model.predict(test_df[ALL_FEATURES_FUKU])
    valid_auc = roc_auc_score(y_va, p_va)
    test_auc  = roc_auc_score(test_df[TARGET], p_te)
    valid_pr  = average_precision_score(y_va, p_va)
    test_pr   = average_precision_score(test_df[TARGET], p_te)

    print("\n" + "=" * 60)
    print("複勝特化LightGBM v2 学習結果")
    print("=" * 60)
    print(f"  Valid AUC : {valid_auc:.4f}  (baseline {BASELINE_VALID_AUC:.4f})")
    print(f"  Test  AUC : {test_auc:.4f}  (baseline {BASELINE_TEST_AUC:.4f})")
    print(f"  Valid PR  : {valid_pr:.4f}")
    print(f"  Test  PR  : {test_pr:.4f}")

    adopted = (valid_auc > BASELINE_VALID_AUC) and (test_auc > BASELINE_TEST_AUC)
    print(f"\n  採用判定 : {'[ADOPT]' if adopted else '[REJECT]'}")

    save_path = OUT_PATH if adopted else REJECT_PATH
    joblib.dump({
        "model":        model,
        "encoders":     encoders,
        "feature_cols": ALL_FEATURES_FUKU,
        "target":       TARGET,
        "valid_auc":    valid_auc,
        "test_auc":     test_auc,
        "valid_pr":     valid_pr,
        "test_pr":      test_pr,
        "adopted":      adopted,
        "params":       LGBM_PARAMS_FUKU,
        "version":      "v2_fukusho_specialized",
    }, save_path)
    print(f"  保存       : {save_path}")

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "valid_auc": valid_auc,
            "test_auc":  test_auc,
            "valid_pr":  valid_pr,
            "test_pr":   test_pr,
            "baseline_valid_auc": BASELINE_VALID_AUC,
            "baseline_test_auc":  BASELINE_TEST_AUC,
            "adopted":   adopted,
            "n_features": len(ALL_FEATURES_FUKU),
            "params":    LGBM_PARAMS_FUKU,
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"メトリクス保存: {METRICS_PATH}")


if __name__ == "__main__":
    main()
