"""
retrain_value_model.py
======================
PyCaLiAI Value Model v2 — 月次Walk-forward再学習スクリプト

概要:
  直近6ヶ月のデータでValue Modelを再学習し、validation ROIが改善した場合のみ
  models/value_model_v2.pkl を上書きする。

  毎月初週に weekly_post.ps1 から自動実行される（月初 Day <= 7）。

Usage:
  python retrain_value_model.py --end-date 20260430

データ要件:
  - reports/ensemble_predictions.csv (Layer 1 出力)
  - data/odds_*.csv (複勝・単勝オッズ CSV)
  ※ 2026年7月頃まではデータ不足のため再学習はスキップされる場合がある
"""
from __future__ import annotations

import argparse
import io
import sys
import warnings
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

OUTPUT_PATH = MODEL_DIR / "value_model_v2.pkl"
ENS_PATH = REPORT_DIR / "ensemble_predictions.csv"

# 再学習に必要な最低データ量
MIN_TRAIN_ROWS = 500
MIN_VALID_ROWS = 100

# LightGBM パラメータ（train_value_model.py と同一）
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 50,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
}

try:
    from train_value_model import VALUE_FEATURES, STRATEGIES
except ImportError:
    print("[WARN] train_value_model.py が見つかりません。定数を直接定義します。")
    VALUE_FEATURES = [
        "cal_prob", "model_rank", "ninki", "tan_odds", "fuku_mid",
        "EV_fuku", "disagree", "abs_disagree", "log_tan_odds", "log_fuku_mid",
        "odds_rank_ratio", "model_vs_market_prob", "shutsuu",
        "fuku_spread", "fuku_spread_ratio",
    ]
    STRATEGIES = {
        "balanced": {"pred_roi_thr": 0.88, "cal_prob_thr": 0.20,
                     "desc": "バランス型: 15R/週, 的中38%, ROI140%"},
        "high_roi": {"pred_roi_thr": 0.90, "cal_prob_thr": 0.22,
                     "desc": "高ROI型: 10R/週, 的中39%, ROI159%"},
        "volume":   {"pred_roi_thr": 0.88, "cal_prob_thr": 0.15,
                     "desc": "ボリューム型: 18R/週, 的中33%, ROI132%"},
    }


def parse_end_date(end_date_str: str) -> datetime:
    return datetime.strptime(end_date_str, "%Y%m%d")


def date_to_str(dt: datetime) -> str:
    return dt.strftime("%Y%m%d")


def load_odds_data() -> pd.DataFrame:
    """data/ ディレクトリ内の odds_*.csv を全て読み込む。"""
    odds_files = list(DATA_DIR.glob("odds_*.csv"))
    if not odds_files:
        return pd.DataFrame()

    dfs = []
    for f in sorted(odds_files):
        try:
            df = pd.read_csv(f, encoding="cp932")
            df.columns = ["rid18", "umaban", "shutsuu", "fullgate",
                          "tan_odds", "fuku_low", "fuku_high"]
            df["race_key"] = df["rid18"].astype(str).str[:16]
            dfs.append(df)
            print(f"  Loaded {f.name}: {len(df):,} rows")
        except Exception as e:
            print(f"  [WARN] {f.name} 読み込みエラー: {e}")

    if not dfs:
        return pd.DataFrame()

    odds = pd.concat(dfs, ignore_index=True)
    odds["ninki"] = odds.groupby("race_key")["tan_odds"].rank(
        method="first", ascending=True
    )
    return odds


def build_features(merged: pd.DataFrame, iso: IsotonicRegression) -> pd.DataFrame:
    """Value Model 特徴量を作成する。"""
    merged["cal_prob"] = iso.transform(merged["prob"].values)
    merged["fuku_mid"] = (merged["fuku_low"] + merged["fuku_high"]) / 2
    merged["model_rank"] = merged.groupby("race_id")["prob"].rank(
        ascending=False, method="first"
    )
    merged["EV_fuku"] = merged["cal_prob"] * merged["fuku_mid"]
    merged["disagree"] = merged["model_rank"] - merged["ninki"]
    merged["abs_disagree"] = merged["disagree"].abs()
    merged["log_tan_odds"] = np.log1p(merged["tan_odds"])
    merged["log_fuku_mid"] = np.log1p(merged["fuku_mid"])
    merged["odds_rank_ratio"] = merged["tan_odds"] / (merged["ninki"] + 0.5)
    merged["model_vs_market_prob"] = merged["cal_prob"] - (1 / merged["tan_odds"])
    merged["fuku_spread"] = merged["fuku_high"] - merged["fuku_low"]
    merged["fuku_spread_ratio"] = merged["fuku_spread"] / (merged["fuku_mid"] + 0.01)
    merged["roi_target"] = merged["fukusho_flag"] * merged["fuku_mid"]
    return merged


def compute_balanced_roi(df: pd.DataFrame, model: lgb.Booster,
                          strat_name: str = "balanced") -> float:
    """指定戦略のvalidation balanced ROI を計算する。"""
    strat = STRATEGIES[strat_name]
    pr_thr = strat["pred_roi_thr"]
    cp_thr = strat["cal_prob_thr"]
    pred = model.predict(df[VALUE_FEATURES])
    sel = df[(pred >= pr_thr) & (df["cal_prob"] >= cp_thr)]
    if len(sel) < 10:
        return 0.0
    return float(sel["roi_target"].sum() / len(sel))


def retrain(end_date_str: str, dry_run: bool = False) -> bool:
    """
    Walk-forward 再学習メイン処理。

    日付分割:
      cal:   end_date - 9ヶ月 〜 end_date - 6ヶ月
      train: end_date - 6ヶ月 〜 end_date - 1ヶ月
      valid: end_date - 1ヶ月 〜 end_date

    Returns:
      True if model was updated, False otherwise.
    """
    end_dt = parse_end_date(end_date_str)
    cal_start   = date_to_str(end_dt - relativedelta(months=9))
    cal_end     = date_to_str(end_dt - relativedelta(months=6))
    train_start = date_to_str(end_dt - relativedelta(months=6))
    train_end   = date_to_str(end_dt - relativedelta(months=1))
    valid_start = date_to_str(end_dt - relativedelta(months=1))
    valid_end   = end_date_str

    print(f"\n=== Value Model Walk-forward 再学習 ===")
    print(f"end-date : {end_date_str}")
    print(f"cal      : {cal_start} - {cal_end}")
    print(f"train    : {train_start} - {train_end}")
    print(f"valid    : {valid_start} - {valid_end}")
    print()

    # Load ensemble predictions
    if not ENS_PATH.exists():
        print(f"[ERROR] {ENS_PATH} が存在しません。再学習をスキップします。")
        return False

    print("Loading ensemble predictions...")
    ens = pd.read_csv(ENS_PATH)
    ens.columns = ["race_id", "horse_num", "horse_name", "prob", "mark", "fukusho_flag"]
    ens["date"] = ens["race_id"].astype(str).str[:8]

    # Filter to needed date range
    ens = ens[(ens["date"] >= cal_start) & (ens["date"] <= valid_end)]
    if len(ens) == 0:
        print(f"[WARN] 指定期間のensembleデータなし ({cal_start} - {valid_end})。スキップ。")
        return False

    # Load odds
    print("Loading odds...")
    odds = load_odds_data()
    if odds.empty:
        print("[ERROR] oddsデータが見つかりません。再学習をスキップします。")
        return False

    # Merge
    ens["race_key"] = ens["race_id"].astype(str)
    merged = ens.merge(
        odds[["race_key", "umaban", "tan_odds", "fuku_low", "fuku_high",
              "ninki", "shutsuu"]],
        left_on=["race_key", "horse_num"],
        right_on=["race_key", "umaban"],
        how="inner",
    )
    print(f"Merged: {len(merged):,} rows, {merged['race_id'].nunique():,} races")

    # Calibrate isotonic on cal period
    cal_data = merged[(merged["date"] >= cal_start) & (merged["date"] <= cal_end)]
    if len(cal_data) < 100:
        print(f"[WARN] calibration期間のデータ不足 ({len(cal_data)} rows)。スキップ。")
        return False
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(cal_data["prob"].values, cal_data["fukusho_flag"].values)

    # Build features
    merged = build_features(merged, iso)

    # Split
    train_df = merged[(merged["date"] >= train_start) & (merged["date"] < train_end)]
    valid_df = merged[(merged["date"] >= valid_start) & (merged["date"] <= valid_end)]

    print(f"  train: {len(train_df):,} rows")
    print(f"  valid: {len(valid_df):,} rows")

    if len(train_df) < MIN_TRAIN_ROWS:
        print(f"[WARN] 学習データ不足 ({len(train_df)} < {MIN_TRAIN_ROWS})。スキップ。")
        return False
    if len(valid_df) < MIN_VALID_ROWS:
        print(f"[WARN] validationデータ不足 ({len(valid_df)} < {MIN_VALID_ROWS})。スキップ。")
        return False

    # Train new model
    X_train = train_df[VALUE_FEATURES]
    y_train = train_df["roi_target"]
    X_valid = valid_df[VALUE_FEATURES]
    y_valid = valid_df["roi_target"]

    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)

    print("\nTraining new Value Model...")
    new_model = lgb.train(
        LGBM_PARAMS, dtrain,
        num_boost_round=1000,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    new_roi = compute_balanced_roi(valid_df, new_model)
    print(f"\n新モデル balanced ROI (valid): {new_roi:.3f}")

    # Compare with existing model
    existing_roi = 0.0
    if OUTPUT_PATH.exists():
        try:
            existing_obj = joblib.load(OUTPUT_PATH)
            existing_model = existing_obj["model"]
            existing_roi = compute_balanced_roi(valid_df, existing_model)
            print(f"既存モデル balanced ROI (valid): {existing_roi:.3f}")
        except Exception as e:
            print(f"[WARN] 既存モデル読み込みエラー: {e}。新モデルで上書きします。")
            existing_roi = 0.0
    else:
        print("既存モデルなし。新モデルを保存します。")

    if new_roi <= existing_roi:
        print(f"\n[SKIP] 新モデルROI ({new_roi:.3f}) <= 既存モデルROI ({existing_roi:.3f})。上書きしません。")
        return False

    print(f"\n[UPDATE] 新モデルROI ({new_roi:.3f}) > 既存モデルROI ({existing_roi:.3f})。モデルを更新します。")

    if dry_run:
        print("[DRY-RUN] 実際には保存しません。")
        return True

    # Save new model (preserve strategies/features from existing if available)
    default_strategy = "balanced"
    if OUTPUT_PATH.exists():
        try:
            default_strategy = existing_obj.get("default_strategy", "balanced")
        except Exception:
            pass

    artifact = {
        "model": new_model,
        "features": VALUE_FEATURES,
        "calibrator": iso,
        "strategies": STRATEGIES,
        "default_strategy": default_strategy,
        "version": "v2",
        "retrain_date": end_date_str,
    }
    joblib.dump(artifact, OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Value Model v2 Walk-forward 月次再学習"
    )
    parser.add_argument(
        "--end-date", required=True,
        help="再学習の基準日 (YYYYMMDD)。この日付の直近6ヶ月を学習に使用。"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="保存せずに評価のみ実施"
    )
    args = parser.parse_args()

    updated = retrain(args.end_date, dry_run=args.dry_run)
    if updated:
        print("\n✓ Value Model を更新しました。")
    else:
        print("\n- Value Model は更新されませんでした。")


if __name__ == "__main__":
    main()
