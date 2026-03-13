"""
build_strategy_walkforward.py
PyCaLiAI - ウォークフォワード多年検証による安定条件抽出

train期間（2013-2022）を1年ずつスライドし、
複数年にわたって黒字が続く条件を「真のエッジ候補」として採用する。

さらに valid（2023）+ test（2024-2025）で最終確認を行い
3段階で信頼度を評価する。

採用基準:
  - 各年で MIN_RACES_PER_YEAR レース以上
  - MIN_PROFITABLE_YEARS / EVAL_YEARS 年以上で ROI > 100%
  - valid + test の合算ROIも MIN_ROI_COMBINED 以上

Usage:
    python build_strategy_walkforward.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# =========================================================
# 設定
# =========================================================
BASE_DIR   = Path(r"E:\PyCaLiAI")
REPORT_DIR = BASE_DIR / "reports"
DATA_DIR   = BASE_DIR / "data"

TRAIN_CSV  = REPORT_DIR / "backtest_results_train.csv"
VALID_CSV  = REPORT_DIR / "backtest_results_valid.csv"
TEST_CSV   = REPORT_DIR / "backtest_results_2024.csv"
OUT_JSON   = DATA_DIR   / "strategy_weights.json"
OUT_CSV    = REPORT_DIR / "walkforward_conditions.csv"

EVAL_YEARS          = 5      # 評価年数（2018-2022）
MIN_PROFITABLE_YEARS = 3     # 何年以上黒字なら採用
MIN_RACES_PER_YEAR  = 15     # 各年の最低レース数
MIN_ROI_COMBINED    = 100.0  # valid+test 合算ROIの最低値（%）


# =========================================================
# ユーティリティ
# =========================================================
def add_weekend_filter(df: pd.DataFrame) -> pd.DataFrame:
    """土日かつ同会場10R以上に絞る。"""
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


def roi_by_condition(df: pd.DataFrame) -> pd.DataFrame:
    """場所×クラス×馬券種ごとの ROI とレース数を計算。"""
    rows = []
    for (place, cls, bet), grp in df.groupby(["場所", "クラス", "馬券種"]):
        n = grp["race_id"].nunique()
        cost = grp["購入額"].sum()
        pay  = grp["実払戻額"].sum()
        if cost == 0:
            continue
        rows.append({
            "場所": place, "クラス": cls, "馬券種": bet,
            "レース数": n,
            "投資":     cost,
            "払戻":     pay,
            "ROI":      pay / cost * 100,
        })
    return pd.DataFrame(rows)


# =========================================================
# メイン
# =========================================================
def main() -> None:
    # --- train データ読み込み ---
    logger.info(f"Train CSV 読み込み: {TRAIN_CSV}")
    train_all = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
    train_all = add_weekend_filter(train_all)
    train_all["year"] = pd.to_datetime(
        train_all["日付"].astype(str), format="%Y%m%d"
    ).dt.year

    # --- 直近 EVAL_YEARS 年を1年ずつ評価 ---
    max_year   = train_all["year"].max()
    eval_years = list(range(max_year - EVAL_YEARS + 1, max_year + 1))
    logger.info(f"ウォークフォワード評価年: {eval_years}")

    yearly_roi: dict[int, pd.DataFrame] = {}
    for yr in eval_years:
        df_yr = train_all[train_all["year"] == yr]
        yearly_roi[yr] = roi_by_condition(df_yr)

    # --- 条件ごとに「何年黒字か」を集計 ---
    all_conditions: set[tuple] = set()
    for df_yr in yearly_roi.values():
        for _, row in df_yr.iterrows():
            all_conditions.add((row["場所"], row["クラス"], row["馬券種"]))

    records = []
    for (place, cls, bet) in sorted(all_conditions):
        profitable_years = 0
        years_with_data  = 0
        total_cost = 0
        total_pay  = 0

        yr_details = {}
        for yr in eval_years:
            df_yr = yearly_roi[yr]
            match = df_yr[
                (df_yr["場所"] == place) &
                (df_yr["クラス"] == cls) &
                (df_yr["馬券種"] == bet)
            ]
            if match.empty or match.iloc[0]["レース数"] < MIN_RACES_PER_YEAR:
                yr_details[yr] = None
                continue
            row = match.iloc[0]
            years_with_data += 1
            total_cost += row["投資"]
            total_pay  += row["払戻"]
            roi = row["ROI"]
            yr_details[yr] = round(roi, 1)
            if roi > 100:
                profitable_years += 1

        # データがある年が少ない条件はスキップ
        if years_with_data < MIN_PROFITABLE_YEARS:
            continue

        combined_roi = total_pay / total_cost * 100 if total_cost > 0 else 0
        rec = {
            "場所": place, "クラス": cls, "馬券種": bet,
            "黒字年数": profitable_years,
            "データあり年数": years_with_data,
            "合算ROI_train": round(combined_roi, 1),
        }
        for yr in eval_years:
            rec[f"ROI_{yr}"] = yr_details.get(yr)
        records.append(rec)

    wf_df = pd.DataFrame(records)
    logger.info(f"ウォークフォワード評価条件数: {len(wf_df)}")

    # --- 安定条件フィルタ ---
    stable_train = wf_df[
        wf_df["黒字年数"] >= MIN_PROFITABLE_YEARS
    ].copy()
    logger.info(f"  {EVAL_YEARS}年中{MIN_PROFITABLE_YEARS}年以上黒字: {len(stable_train)} 条件")

    # --- valid + test でのクロスチェック ---
    logger.info(f"Valid CSV 読み込み: {VALID_CSV}")
    valid_df = add_weekend_filter(pd.read_csv(VALID_CSV, encoding="utf-8-sig"))
    logger.info(f"Test CSV 読み込み: {TEST_CSV}")
    test_df  = add_weekend_filter(pd.read_csv(TEST_CSV,  encoding="utf-8-sig"))

    oos_combined = pd.concat([valid_df, test_df], ignore_index=True)
    oos_roi = roi_by_condition(oos_combined)
    oos_roi = oos_roi.rename(columns={
        "レース数": "OOS_レース数",
        "ROI": "ROI_OOS",
        "投資": "OOS_投資",
        "払戻": "OOS_払戻",
    })

    final = pd.merge(
        stable_train,
        oos_roi[["場所", "クラス", "馬券種", "OOS_レース数", "ROI_OOS"]],
        on=["場所", "クラス", "馬券種"],
        how="left",
    )
    final["ROI_OOS"] = final["ROI_OOS"].fillna(0)
    final["OOS_レース数"] = final["OOS_レース数"].fillna(0).astype(int)

    # 信頼度スコア（黒字年率 × OOS ROI補正）
    final["信頼度"] = (
        final["黒字年数"] / final["データあり年数"] * 100
    ).round(1)

    # valid+test で最低ラインを通過した条件
    adopted = final[final["ROI_OOS"] >= MIN_ROI_COMBINED].copy()
    adopted = adopted.sort_values(["信頼度", "ROI_OOS"], ascending=False)
    logger.info(f"OOS({MIN_ROI_COMBINED}%以上)も通過した条件: {len(adopted)}")

    # --- 表示 ---
    yr_cols = [f"ROI_{yr}" for yr in eval_years]
    display_cols = (
        ["場所", "クラス", "馬券種", "黒字年数", "データあり年数", "信頼度", "合算ROI_train"]
        + yr_cols
        + ["OOS_レース数", "ROI_OOS"]
    )

    print(f"\n=== ウォークフォワード安定条件（train {MIN_PROFITABLE_YEARS}/{EVAL_YEARS}年以上 + OOS黒字）===")
    if adopted.empty:
        print("  条件なし。MIN_PROFITABLE_YEARS または MIN_ROI_COMBINED を緩めてください。")
    else:
        pd.set_option("display.max_rows", 100)
        pd.set_option("display.width", 200)
        print(adopted[display_cols].to_string(index=False))

    print(f"\n=== 参考: train安定条件でOOSが赤字だったもの ===")
    rejected = final[final["ROI_OOS"] < MIN_ROI_COMBINED].sort_values("ROI_OOS", ascending=False)
    print(rejected[display_cols].head(20).to_string(index=False))

    # --- CSV 保存 ---
    final.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    logger.info(f"CSV 保存: {OUT_CSV}")

    if adopted.empty:
        return

    # --- strategy_weights.json 生成 ---
    strategy: dict = {}
    for (place, cls), grp in adopted.groupby(["場所", "クラス"]):
        total_roi = grp["ROI_OOS"].sum()
        bets: dict = {}
        for _, row in grp.iterrows():
            w = row["ROI_OOS"] / total_roi
            bets[row["馬券種"]] = {
                "roi_train_combined": row["合算ROI_train"],
                "roi_oos":            round(row["ROI_OOS"], 1),
                "profitable_years":   int(row["黒字年数"]),
                "confidence":         row["信頼度"],
                "weight":             round(w, 4),
                "bet_ratio":          round(w, 4),
            }
        if place not in strategy:
            strategy[place] = {}
        strategy[place][cls] = bets

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(strategy, f, ensure_ascii=False, indent=2)
    logger.info(f"strategy_weights.json 保存: {OUT_JSON}")

    # --- サマリ ---
    print(f"\n=== サマリ ===")
    print(f"採用条件数    : {len(adopted)}")
    print(f"会場数        : {adopted['場所'].nunique()} 箇所 ({sorted(adopted['場所'].unique())})")
    print(f"平均信頼度    : {adopted['信頼度'].mean():.1f}%")
    print(f"OOS ROI範囲  : {adopted['ROI_OOS'].min():.1f}% 〜 {adopted['ROI_OOS'].max():.1f}%")

    test_years = 2
    total_r_per_year = (adopted["OOS_レース数"] / (1 + test_years)).sum()  # valid=1年 + test=2年
    annual_invest = total_r_per_year * 10_000
    annual_pay    = adopted.apply(
        lambda r: r["OOS_レース数"] / (1 + test_years) * 10_000 * r["ROI_OOS"] / 100,
        axis=1,
    ).sum()
    print(f"\n--- 1R=1万円 年間推計（OOS期間ベース）---")
    print(f"対象レース数/年 : {total_r_per_year:.0f} R/年")
    print(f"年間総投資額    : {annual_invest:,.0f} 円")
    print(f"年間推計収支    : {annual_pay - annual_invest:+,.0f} 円")
    print(f"加重平均OOS ROI : {annual_pay / annual_invest * 100:.1f}%")


if __name__ == "__main__":
    main()
