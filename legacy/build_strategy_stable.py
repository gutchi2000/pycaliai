"""
build_strategy_stable.py
PyCaLiAI - ウォークフォワード安定条件抽出

valid（2023）と test（2024-2025）の両期間で黒字の条件のみを採用する。
1年だけの結果でフィルタするより信頼性が大幅に向上する。

採用基準:
  - 各期間で MIN_RACES レース以上
  - valid と test の両方で ROI > MIN_ROI_PCT
  - 両期間合算の ROI も MIN_ROI_PCT 以上

Usage:
    python build_strategy_stable.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# =========================================================
# 設定
# =========================================================
BASE_DIR    = Path(r"E:\PyCaLiAI")
REPORT_DIR  = BASE_DIR / "reports"
DATA_DIR    = BASE_DIR / "data"

VALID_CSV   = REPORT_DIR / "backtest_results_valid.csv"   # 2023年（out-of-sample）
TEST_CSV    = REPORT_DIR / "backtest_results_2024.csv"    # 2024-2025年（out-of-sample）
OUT_JSON    = DATA_DIR   / "strategy_weights.json"
OUT_CSV     = REPORT_DIR / "stable_conditions.csv"

MIN_RACES        = 15    # 各期間で必要な最低レース数
MIN_ROI_VALID    = 50.0  # valid期間(2023)の最低ROI(%) ← 旧モデル期間は緩く
MIN_ROI_TEST     = 80.0  # test期間(2024-2025)の最低ROI(%) ← 土日各5R以上を毎週保証
MIN_ROI_COMBINED = 80.0  # 両期間合算の最低ROI(%)


# =========================================================
# ユーティリティ
# =========================================================
def add_weekend_filter(df: pd.DataFrame) -> pd.DataFrame:
    """土日＋10R以上の会場に絞る。"""
    df = df.copy()
    df["date"] = pd.to_datetime(df["日付"].astype(str), format="%Y%m%d")
    df["土日"] = df["date"].dt.dayofweek.isin([5, 6])
    rc = (
        df.groupby(["日付", "場所"])["race_id"]
        .nunique()
        .reset_index()
        .rename(columns={"race_id": "R数"})
    )
    df = df.merge(rc, on=["日付", "場所"], how="left")
    return df[df["土日"] & (df["R数"] >= 10)].copy()


def compute_roi_by_condition(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """場所×クラス×馬券種ごとの ROI・レース数・的中率を計算する。"""
    rows = []
    for (place, cls, bet), grp in df.groupby(["場所", "クラス", "馬券種"]):
        n_races = grp["race_id"].nunique()
        cost    = grp["購入額"].sum()
        pay     = grp["実払戻額"].sum()
        if cost == 0:
            continue
        roi = pay / cost * 100
        rows.append({
            "場所":   place,
            "クラス": cls,
            "馬券種": bet,
            f"レース数_{label}": n_races,
            f"投資_{label}":     cost,
            f"払戻_{label}":     pay,
            f"ROI_{label}":      round(roi, 1),
        })
    return pd.DataFrame(rows)


# =========================================================
# メイン
# =========================================================
def main() -> None:
    # --- データ読み込み ---
    logger.info(f"Valid CSV 読み込み: {VALID_CSV}")
    valid_raw = pd.read_csv(VALID_CSV, encoding="utf-8-sig")
    valid_raw = add_weekend_filter(valid_raw)
    logger.info(f"  Valid（土日10R以上）: {valid_raw['race_id'].nunique():,} レース")

    logger.info(f"Test CSV 読み込み: {TEST_CSV}")
    test_raw = pd.read_csv(TEST_CSV, encoding="utf-8-sig")
    test_raw = add_weekend_filter(test_raw)
    logger.info(f"  Test（土日10R以上）: {test_raw['race_id'].nunique():,} レース")

    # --- 各期間の条件別ROI ---
    valid_roi = compute_roi_by_condition(valid_raw, "valid")
    test_roi  = compute_roi_by_condition(test_raw,  "test")

    # --- マージ（両期間に存在する条件のみ）---
    merged = pd.merge(
        valid_roi, test_roi,
        on=["場所", "クラス", "馬券種"],
        how="inner",
    )
    logger.info(f"両期間共通条件数: {len(merged)}")

    # --- 合算ROI ---
    merged["投資_合算"] = merged["投資_valid"] + merged["投資_test"]
    merged["払戻_合算"] = merged["払戻_valid"] + merged["払戻_test"]
    merged["ROI_合算"]  = (merged["払戻_合算"] / merged["投資_合算"] * 100).round(1)

    # --- フィルタ ---
    stable = merged[
        (merged["レース数_valid"] >= MIN_RACES) &
        (merged["レース数_test"]  >= MIN_RACES) &
        (merged["ROI_valid"]      >= MIN_ROI_VALID) &
        (merged["ROI_test"]       >= MIN_ROI_TEST) &
        (merged["ROI_合算"]       >= MIN_ROI_COMBINED)
    ].copy().sort_values("ROI_合算", ascending=False)

    logger.info(f"安定黒字条件数: {len(stable)}")

    if stable.empty:
        logger.warning(
            "安定条件が0件です。MIN_RACES または MIN_ROI_PCT を緩めてください。"
        )
        return

    # --- 表示 ---
    display_cols = [
        "場所", "クラス", "馬券種",
        "レース数_valid", "ROI_valid",
        "レース数_test",  "ROI_test",
        "ROI_合算",
    ]
    print("\n=== 安定黒字条件（valid + test 両方で黒字）===")
    print(stable[display_cols].to_string(index=False))

    # --- CSV 保存 ---
    stable.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    logger.info(f"CSV 保存: {OUT_CSV}")

    # --- strategy_weights.json 生成 ---
    # 場所×クラスごとにROI比例ウェイトで按分
    strategy: dict = {}
    for (place, cls), grp in stable.groupby(["場所", "クラス"]):
        total_roi = grp["ROI_合算"].sum()
        bets: dict = {}
        for _, row in grp.iterrows():
            w = row["ROI_合算"] / total_roi
            bets[row["馬券種"]] = {
                "roi_valid":    row["ROI_valid"],
                "roi_test":     row["ROI_test"],
                "roi_combined": row["ROI_合算"],
                "n_races_valid": int(row["レース数_valid"]),
                "n_races_test":  int(row["レース数_test"]),
                "weight":       round(w, 4),
                "bet_ratio":    round(w, 4),
            }
        if place not in strategy:
            strategy[place] = {}
        strategy[place][cls] = bets

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(strategy, f, ensure_ascii=False, indent=2)
    logger.info(f"strategy_weights.json 保存: {OUT_JSON}")

    # --- サマリ ---
    print(f"\n=== サマリ ===")
    print(f"採用条件数  : {len(stable)}")
    print(f"会場数      : {stable['場所'].nunique()}")
    print(f"合算ROI範囲 : {stable['ROI_合算'].min():.1f}% 〜 {stable['ROI_合算'].max():.1f}%")

    # 1レース10万円での年間推計（test期間のレース数/年をベースに）
    test_years = 2  # backtest_results_2024.csv は2024-2025の約2年
    total_races_per_year = (stable["レース数_test"] / test_years).sum()
    budget_per_race = 100_000
    est_annual_inv  = total_races_per_year * budget_per_race
    est_annual_pay  = (
        stable.apply(
            lambda r: (r["レース数_test"] / test_years) * budget_per_race * r["ROI_test"] / 100,
            axis=1,
        ).sum()
    )
    print(f"\n--- 1レース10万円 年間推計（test期間ベース）---")
    print(f"対象レース数/年 : {total_races_per_year:.0f} R")
    print(f"年間総投資額    : {est_annual_inv:,.0f} 円")
    print(f"年間推計払戻    : {est_annual_pay:,.0f} 円")
    print(f"年間推計収支    : {est_annual_pay - est_annual_inv:+,.0f} 円")
    print(f"加重平均ROI     : {est_annual_pay/est_annual_inv*100:.1f}%")


if __name__ == "__main__":
    main()
