"""
train_expert.py — Phase 5+ 距離別 Mixture of Experts (MoE)

レース条件（芝/ダ × 距離帯）で学習データを絞り込み、特化Expert を学習する。
train_lgbm.py の preprocess / train_model / NUM_FEATURES / CAT_FEATURES / ALL_FEATURES
をそのまま再利用するため、新規ロジックは最小限。

Usage:
    python train_expert.py                # 全Expertを学習
    python train_expert.py --expert dirt  # 特定Expertだけ学習

Output:
    models/expert_turf_short.pkl
    models/expert_turf_mid.pkl
    models/expert_turf_long.pkl
    models/expert_dirt.pkl
    reports/expert_metrics.json   # 各Expertの valid/test AUC

採用判定:
    - Valid AUC が「全体モデル(0.7767)」を下回る Expert は採用見送り
    - reports/expert_metrics.json に "adopted": bool で記録
    - predict_weekly.py / app.py の Expert ルーティングは pkl 存在チェックで自動有効化
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from train_lgbm import (
    MODEL_DIR, REPORT_DIR, MASTER_CSV, MASTER_CSV_ORIG,
    ALL_FEATURES, TARGET,
    preprocess, train_model, evaluate,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 全体モデル(8アンサンブル)の Valid AUC ベンチマーク
BASELINE_VALID_AUC = 0.7767

EXPERTS = {
    "turf_short": {"label": "芝短距離(～1400m)",  "td": "芝", "dist_max": 1400},
    "turf_mid":   {"label": "芝中距離(1600-2200m)", "td": "芝", "dist_min": 1600, "dist_max": 2200},
    "turf_long":  {"label": "芝長距離(2400m～)",  "td": "芝", "dist_min": 2400},
    "dirt":       {"label": "ダート全距離",        "td": "ダ"},
}

MIN_TRAIN_ROWS = 50_000  # サンプル数 < 5万なら学習スキップ


def filter_by_expert(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Expert設定に従って df をフィルタする。"""
    m = pd.Series(True, index=df.index)
    if "td" in cfg:
        m &= (df["芝・ダ"] == cfg["td"])
    if "dist_min" in cfg:
        m &= (pd.to_numeric(df["距離"], errors="coerce") >= cfg["dist_min"])
    if "dist_max" in cfg:
        m &= (pd.to_numeric(df["距離"], errors="coerce") <= cfg["dist_max"])
    return df[m].copy()


def load_master() -> pd.DataFrame:
    csv_path = MASTER_CSV if MASTER_CSV.exists() else MASTER_CSV_ORIG
    logger.info(f"マスターCSV読み込み: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    logger.info(f"  {len(df):,}行 × {len(df.columns)}列")
    return df


def train_one_expert(df_full: pd.DataFrame, name: str, cfg: dict) -> dict:
    """1つのExpertを学習し、メトリクス dict を返す。"""
    logger.info(f"\n{'='*60}\nExpert: {name} ({cfg['label']})\n{'='*60}")

    df_exp = filter_by_expert(df_full, cfg)
    train  = df_exp[df_exp["split"] == "train"].copy()
    valid  = df_exp[df_exp["split"] == "valid"].copy()
    test   = df_exp[df_exp["split"] == "test"].copy()

    logger.info(f"分割: train={len(train):,} / valid={len(valid):,} / test={len(test):,}")

    if len(train) < MIN_TRAIN_ROWS:
        logger.warning(f"  サンプル不足 ({len(train):,} < {MIN_TRAIN_ROWS:,}) → スキップ")
        return {
            "name": name, "label": cfg["label"],
            "train_rows": int(len(train)), "valid_rows": int(len(valid)), "test_rows": int(len(test)),
            "skipped": True, "reason": "insufficient_samples",
            "adopted": False,
        }

    # 学習（train_lgbm の関数をそのまま再利用）
    model, encoders, feature_cols = train_model(train, valid)

    # 評価
    m_valid = evaluate(model, valid, encoders, feature_cols, f"Expert_{name}_Valid")
    m_test  = evaluate(model, test,  encoders, feature_cols, f"Expert_{name}_Test")

    valid_auc = m_valid["auc"]
    test_auc  = m_test["auc"]
    adopted   = valid_auc >= BASELINE_VALID_AUC

    # 保存（採用判定に関わらず保存。pkl 存在チェックは ensemble 側で自動有効化のため）
    out_path = MODEL_DIR / f"expert_{name}.pkl"
    joblib.dump(
        {"model": model, "encoders": encoders, "feature_cols": feature_cols},
        out_path,
    )
    logger.info(f"保存: {out_path}")
    logger.info(f"  Valid AUC: {valid_auc:.4f}  Test AUC: {test_auc:.4f}")
    logger.info(f"  Baseline:  {BASELINE_VALID_AUC:.4f}  → 採用: {'YES' if adopted else 'NO'}")

    # 採用見送りなら pkl をリネームして無効化（Expert ルーティングは pkl 存在で自動切替）
    if not adopted:
        rejected_path = MODEL_DIR / f"expert_{name}_rejected.pkl"
        out_path.rename(rejected_path)
        logger.info(f"  → 採用見送り。{rejected_path.name} にリネーム")

    return {
        "name": name, "label": cfg["label"],
        "train_rows": int(len(train)), "valid_rows": int(len(valid)), "test_rows": int(len(test)),
        "valid_auc": float(valid_auc), "test_auc": float(test_auc),
        "baseline_valid_auc": BASELINE_VALID_AUC,
        "adopted": adopted,
        "skipped": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", choices=list(EXPERTS.keys()), default=None,
                    help="特定Expertだけ学習（省略時は全部）")
    args = ap.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_master()
    targets = {args.expert: EXPERTS[args.expert]} if args.expert else EXPERTS

    metrics = []
    for name, cfg in targets.items():
        try:
            m = train_one_expert(df, name, cfg)
            metrics.append(m)
        except Exception as e:
            logger.error(f"Expert {name} 学習失敗: {e}", exc_info=True)
            metrics.append({"name": name, "error": str(e), "adopted": False})

    # メトリクス保存
    out_json = REPORT_DIR / "expert_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"baseline_valid_auc": BASELINE_VALID_AUC, "experts": metrics},
                  f, indent=2, ensure_ascii=False)
    logger.info(f"\nメトリクス保存: {out_json}")

    # サマリ
    print("\n" + "=" * 60)
    print("Expert 学習サマリ")
    print("=" * 60)
    for m in metrics:
        if m.get("skipped"):
            print(f"{m['name']:15s}: SKIPPED ({m.get('reason','-')})")
        elif "error" in m:
            print(f"{m['name']:15s}: ERROR  {m['error']}")
        else:
            mark = "[ADOPT]" if m["adopted"] else "[REJECT]"
            print(f"{m['name']:15s}: V={m['valid_auc']:.4f} T={m['test_auc']:.4f} {mark}")


if __name__ == "__main__":
    main()
