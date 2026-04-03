"""
train_value_model.py
====================
PyCaLiAI v2 - Value Model (Layer 2) 学習スクリプト

概要:
  Layer 1 (既存ensemble) が出力する base_prob に対して、
  オッズ情報と市場乖離度を加味し「期待回収率(ROI)」を予測する
  2nd-stage LightGBMモデルを学習する。

入力:
  - reports/ensemble_predictions.csv (Layer 1 出力)
  - data/odds_20240106-20241228.csv  (前日複勝・単勝オッズ)
  - data/odds_20230105-20231228.csv  (2023年オッズ、存在すれば)

出力:
  - models/value_model_v2.pkl  (joblib dict)
    keys: model, features, calibrator, strategies, version

戦略 (strategies dict):
  balanced:  pred_roi>=0.88, cal_prob>=0.20  → 15R/週, 的中38%, ROI140%
  high_roi:  pred_roi>=0.90, cal_prob>=0.22  → 10R/週, 的中39%, ROI159%
  volume:    pred_roi>=0.88, cal_prob>=0.15  → 18R/週, 的中33%, ROI132%

Usage:
  python train_value_model.py
  python train_value_model.py --strategy balanced
"""
from __future__ import annotations

import argparse
import sys
import io
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(r"E:\PyCaLiAI\data")
MODEL_DIR = Path(r"E:\PyCaLiAI\models")
REPORT_DIR = BASE_DIR / "reports"

ENS_PATH = REPORT_DIR / "ensemble_predictions.csv"
ODDS_FILES = [
    DATA_DIR / "odds_20240106-20241228.csv",
    DATA_DIR / "odds_20230105-20231228.csv",
]
OUTPUT_PATH = MODEL_DIR / "value_model_v2.pkl"

# ── Strategies ─────────────────────────────────────────────
STRATEGIES = {
    "balanced": {"pred_roi_thr": 0.88, "cal_prob_thr": 0.20,
                 "desc": "バランス型: 15R/週, 的中38%, ROI140%"},
    "high_roi": {"pred_roi_thr": 0.90, "cal_prob_thr": 0.22,
                 "desc": "高ROI型: 10R/週, 的中39%, ROI159%"},
    "volume":   {"pred_roi_thr": 0.88, "cal_prob_thr": 0.15,
                 "desc": "ボリューム型: 18R/週, 的中33%, ROI132%"},
}

# ── Feature engineering ─────────────────────────────────────
VALUE_FEATURES = [
    "cal_prob",
    "model_rank",
    "ninki",
    "tan_odds",
    "fuku_mid",
    "EV_fuku",
    "disagree",
    "abs_disagree",
    "log_tan_odds",
    "log_fuku_mid",
    "odds_rank_ratio",
    "model_vs_market_prob",
    "shutsuu",
    "fuku_spread",
    "fuku_spread_ratio",
]


def load_and_prepare() -> tuple[pd.DataFrame, IsotonicRegression]:
    """ensemble + odds をマージし、value特徴量を作成する。"""
    print("Loading ensemble predictions...")
    ens = pd.read_csv(ENS_PATH)
    ens.columns = ["race_id", "horse_num", "horse_name", "prob", "mark", "fukusho_flag"]
    ens["date"] = ens["race_id"].astype(str).str[:8]

    print("Loading odds...")
    dfs = []
    for odds_path in ODDS_FILES:
        if odds_path.exists():
            df = pd.read_csv(odds_path, encoding="cp932")
            df.columns = ["rid18", "umaban", "shutsuu", "fullgate",
                          "tan_odds", "fuku_low", "fuku_high"]
            df["race_key"] = df["rid18"].astype(str).str[:16]
            dfs.append(df)
            print(f"  Loaded {odds_path.name}: {len(df):,} rows")
    odds = pd.concat(dfs, ignore_index=True)
    odds["ninki"] = odds.groupby("race_key")["tan_odds"].rank(
        method="first", ascending=True
    )

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

    # Calibrate prob (fit on first 3 months of earliest year)
    print("Calibrating probabilities...")
    cal_mask = merged["date"] <= "20240331"
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(merged.loc[cal_mask, "prob"].values,
            merged.loc[cal_mask, "fukusho_flag"].values)
    merged["cal_prob"] = iso.transform(merged["prob"].values)

    # Feature engineering
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
    merged["fuku_spread_ratio"] = (
        merged["fuku_spread"] / (merged["fuku_mid"] + 0.01)
    )

    # Target: actual ROI per bet (fuku_mid if hit, 0 if miss)
    merged["roi_target"] = merged["fukusho_flag"] * merged["fuku_mid"]

    return merged, iso


def train_value_model(merged: pd.DataFrame) -> lgb.Booster:
    """Walk-forward的に学習データを最大化してValue Modelを訓練する。"""

    # Split: cal(~Mar) | train(Apr-Aug) | valid(Sep-Oct) | test(Nov-Dec)
    merged["split"] = np.where(
        merged["date"] <= "20240331", "cal",
        np.where(merged["date"] <= "20240831", "train",
        np.where(merged["date"] <= "20241031", "valid", "test"))
    )

    for s in ["cal", "train", "valid", "test"]:
        n = (merged["split"] == s).sum()
        nr = merged[merged["split"] == s]["race_id"].nunique()
        print(f"  {s}: {n:,} rows, {nr:,} races")

    train = merged[merged["split"] == "train"]
    valid = merged[merged["split"] == "valid"]
    test = merged[merged["split"] == "test"]

    X_train = train[VALUE_FEATURES]
    y_train = train["roi_target"]
    X_valid = valid[VALUE_FEATURES]
    y_valid = valid["roi_target"]
    X_test = test[VALUE_FEATURES]
    y_test = test["roi_target"]

    params = {
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

    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)

    print("\nTraining Value Model v2...")
    model = lgb.train(
        params, dtrain,
        num_boost_round=1000,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
    )
    print(f"Best iteration: {model.best_iteration}")

    # === Evaluation ===
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    for split_name, X, y, df_split in [
        ("VALID", X_valid, y_valid, valid),
        ("TEST", X_test, y_test, test),
    ]:
        pred = model.predict(X)
        df_eval = df_split.copy()
        df_eval["pred_roi"] = pred

        print(f"\n--- {split_name} ---")

        # Strategy evaluation
        for strat_name, strat in STRATEGIES.items():
            pr_thr = strat["pred_roi_thr"]
            cp_thr = strat["cal_prob_thr"]
            sel = df_eval[(df_eval["pred_roi"] >= pr_thr) & (df_eval["cal_prob"] >= cp_thr)]
            if len(sel) < 5:
                continue
            n = len(sel)
            nr = sel["race_id"].nunique()
            hit = sel["fukusho_flag"].mean()
            roi = sel["roi_target"].sum() / n
            months = df_eval["date"].str[:6].nunique()
            rpw = nr / (months * 4.3)
            mark = " ***" if roi > 1.0 else ""
            print(f"  {strat_name:<12} {n:>5,} bets {nr:>4} races ({rpw:.0f}R/週) "
                  f"的中={hit:.1%} ROI={roi:.1%}{mark}")

        # Raw threshold evaluation
        print(f"\n  {'Threshold':<12} {'Bets':>6} {'Races':>6} {'Hit%':>6} {'ROI':>7}")
        for thr in [0.0, 0.8, 0.9, 1.0, 1.1, 1.2]:
            sel = df_eval[df_eval["pred_roi"] >= thr]
            if len(sel) < 10:
                continue
            n = len(sel)
            nr = sel["race_id"].nunique()
            hit = sel["fukusho_flag"].mean()
            roi = sel["roi_target"].sum() / n
            mark = " ***" if roi > 1.0 else ""
            print(f"  >= {thr:<8} {n:>6,} {nr:>6,} {hit:>5.1%} {roi:>6.1%}{mark}")

    # Feature importance
    print("\n--- Feature Importance ---")
    imp = model.feature_importance(importance_type="gain")
    fimp = pd.DataFrame({"feature": VALUE_FEATURES, "gain": imp})
    fimp["pct"] = fimp["gain"] / fimp["gain"].sum() * 100
    fimp = fimp.sort_values("pct", ascending=False)
    for _, row in fimp.iterrows():
        bar = "#" * int(row["pct"] * 2)
        print(f"  {row['feature']:<25} {row['pct']:>5.1f}% {bar}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train Value Model v2 (Layer 2)")
    parser.add_argument("--strategy", type=str, default="balanced",
                        choices=list(STRATEGIES.keys()),
                        help="Default betting strategy")
    args = parser.parse_args()

    merged, calibrator = load_and_prepare()
    model = train_value_model(merged)

    # Save
    artifact = {
        "model": model,
        "features": VALUE_FEATURES,
        "calibrator": calibrator,
        "strategies": STRATEGIES,
        "default_strategy": args.strategy,
        "version": "v2",
    }
    joblib.dump(artifact, OUTPUT_PATH)
    print(f"\nSaved: {OUTPUT_PATH}")

    strat = STRATEGIES[args.strategy]
    print(f"Default strategy: {args.strategy} ({strat['desc']})")
    print(f"  pred_roi >= {strat['pred_roi_thr']}")
    print(f"  cal_prob >= {strat['cal_prob_thr']}")
    print("\nUsage in predict_weekly.py:")
    print("  value_obj = joblib.load('models/value_model_v2.pkl')")
    print("  pred_roi = value_obj['model'].predict(X[value_obj['features']])")
    print("  strat = value_obj['strategies'][value_obj['default_strategy']]")
    print("  buy = (pred_roi >= strat['pred_roi_thr']) & (cal_prob >= strat['cal_prob_thr'])")


if __name__ == "__main__":
    main()
