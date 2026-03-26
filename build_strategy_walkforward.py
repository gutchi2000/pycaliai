"""
build_strategy_walkforward.py
PyCaLiAI - ウォークフォワード多年検証による安定条件抽出

【設計思想】
  採用判断 と 評価 を完全に分離する。

  採用判断（選別に使うデータ）:
    1. train（2013-2022）walk-forward: EVAL_YEARS年中MIN_PROFITABLE_YEARS年以上黒字
    2. valid（2023）のみ: ROI ≥ MIN_ROI_VALID かつ n ≥ MIN_VALID_RACES

  独立評価（採用判断に一切使わない）:
    - test（2024-2025）: 採用後に独立ROIを記録するのみ

  この設計により「testデータで選んでtestで評価する」循環を解消する。

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
COMPARE_CSV = REPORT_DIR / "walkforward_vs_stable_comparison.csv"  # 旧設計との比較

EVAL_YEARS           = 5     # 評価年数（2018-2022）
MIN_PROFITABLE_YEARS = 3     # 何年以上黒字なら採用
MIN_RACES_PER_YEAR   = 15    # 各年の最低レース数
# ── 採用判断（valid 2023のみ）──
MIN_ROI_VALID        = 80.0  # valid期間の最低ROI（%） ← 採用に使う
MIN_VALID_RACES      = 8     # valid期間の最低レース数（1年なので緩め）
# ── 独立評価（test 2024-25、採用判断に使わない）──
# test ROIは記録のみ。採用基準に含めないことで循環を防ぐ。


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


    # --- valid(2023) で採用判断 ---
    logger.info(f"Valid CSV 読み込み: {VALID_CSV}")
    valid_df  = add_weekend_filter(pd.read_csv(VALID_CSV, encoding="utf-8-sig"))
    valid_roi = roi_by_condition(valid_df).rename(columns={
        "レース数": "n_valid", "ROI": "ROI_valid",
        "投資": "投資_valid", "払戻": "払戻_valid",
    })

    # --- test(2024-25) は独立評価のみ（採用判断に使わない）---
    logger.info(f"Test CSV 読み込み: {TEST_CSV}")
    test_df  = add_weekend_filter(pd.read_csv(TEST_CSV, encoding="utf-8-sig"))
    test_roi = roi_by_condition(test_df).rename(columns={
        "レース数": "n_test", "ROI": "ROI_test",
        "投資": "投資_test", "払戻": "払戻_test",
    })

    # --- マージ（train安定条件 × valid ROI）---
    final = pd.merge(stable_train, valid_roi[["場所","クラス","馬券種","n_valid","ROI_valid"]],
                     on=["場所","クラス","馬券種"], how="left")
    final = pd.merge(final, test_roi[["場所","クラス","馬券種","n_test","ROI_test"]],
                     on=["場所","クラス","馬券種"], how="left")
    final["ROI_valid"]  = final["ROI_valid"].fillna(0)
    final["ROI_test"]   = final["ROI_test"].fillna(0)
    final["n_valid"]    = final["n_valid"].fillna(0).astype(int)
    final["n_test"]     = final["n_test"].fillna(0).astype(int)

    # 信頼度スコア（train黒字年率）
    final["信頼度"] = (
        final["黒字年数"] / final["データあり年数"] * 100
    ).round(1)

    # ── 採用判断: valid のみ（testは使わない）──
    adopted = final[
        (final["ROI_valid"] >= MIN_ROI_VALID) &
        (final["n_valid"]   >= MIN_VALID_RACES)
    ].copy()
    adopted = adopted.sort_values(["信頼度", "ROI_valid"], ascending=False)
    logger.info(f"採用条件(valid ROI>={MIN_ROI_VALID}% / n>={MIN_VALID_RACES}): {len(adopted)}")
    logger.info(f"  → test ROI（独立評価）: 平均 {adopted['ROI_test'].mean():.1f}%"
                f" / 中央値 {adopted['ROI_test'].median():.1f}%")

    # --- 表示 ---
    yr_cols = [f"ROI_{yr}" for yr in eval_years]
    display_cols = (
        ["場所", "クラス", "馬券種", "黒字年数", "データあり年数", "信頼度", "合算ROI_train"]
        + yr_cols
        + ["n_valid", "ROI_valid", "n_test", "ROI_test"]
    )

    pd.set_option("display.max_rows", 100)
    pd.set_option("display.width", 220)

    print(f"\n=== 採用条件（train {MIN_PROFITABLE_YEARS}/{EVAL_YEARS}年 + valid ROI>={MIN_ROI_VALID}%）===")
    if adopted.empty:
        print("  条件なし。MIN_PROFITABLE_YEARS または MIN_ROI_VALID を緩めてください。")
    else:
        print(adopted[display_cols].to_string(index=False))

    print(f"\n=== 独立評価サマリ（test 2024-25、採用判断に使用せず）===")
    if not adopted.empty:
        over100 = (adopted["ROI_test"] >= 100).sum()
        over80  = (adopted["ROI_test"] >= 80).sum()
        print(f"  採用{len(adopted)}条件のうち test ROI>=100%: {over100}件 / >=80%: {over80}件")
        print(f"  test ROI 分布: {adopted['ROI_test'].describe().round(1).to_dict()}")

    print(f"\n=== 参考: train安定 + valid通過 だが test が低い条件 ===")
    if not adopted.empty:
        low_test = adopted[adopted["ROI_test"] < 80].sort_values("ROI_test")
        print(low_test[display_cols].to_string(index=False) if not low_test.empty else "  なし")

    print(f"\n=== 参考: train安定 だが valid未通過の条件 ===")
    rejected_valid = final[
        (final["ROI_valid"] < MIN_ROI_VALID) | (final["n_valid"] < MIN_VALID_RACES)
    ].sort_values("ROI_valid", ascending=False)
    print(rejected_valid[display_cols].head(15).to_string(index=False))

    # --- CSV 保存（全条件、採用フラグ付き）---
    final["採用"] = (
        (final["ROI_valid"] >= MIN_ROI_VALID) &
        (final["n_valid"]   >= MIN_VALID_RACES)
    )
    final.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    logger.info(f"全条件CSV保存: {OUT_CSV}")

    if adopted.empty:
        logger.warning("採用条件が0件。MIN_ROI_VALID または MIN_VALID_RACES を緩めてください。")
        return

    # --- 旧strategy_weights.jsonとの比較CSV ---
    old_json_path = DATA_DIR / "strategy_weights.json"
    if old_json_path.exists():
        with open(old_json_path, encoding="utf-8") as f:
            old_strat = json.load(f)
        old_rows = []
        for place, classes in old_strat.items():
            for cls, bets in classes.items():
                for bet, info in bets.items():
                    old_rows.append({
                        "場所": place, "クラス": cls, "馬券種": bet,
                        "旧_roi_valid": info.get("roi_valid", info.get("roi_oos", "")),
                        "旧_roi_test":  info.get("roi_test", ""),
                        "旧_採用": True,
                    })
        old_df = pd.DataFrame(old_rows)
        compare = pd.merge(
            adopted[["場所","クラス","馬券種","ROI_valid","ROI_test","n_valid","n_test","信頼度"]],
            old_df, on=["場所","クラス","馬券種"], how="outer"
        )
        compare["新_採用"] = compare["ROI_valid"].notna() & (compare["ROI_valid"] >= MIN_ROI_VALID)
        compare["旧_採用"] = compare["旧_採用"].fillna(False)
        compare["変化"] = compare.apply(
            lambda r: "維持" if r["新_採用"] and r["旧_採用"]
                      else ("新規追加" if r["新_採用"] and not r["旧_採用"]
                      else ("削除" if not r["新_採用"] and r["旧_採用"] else "対象外")),
            axis=1
        )
        compare.to_csv(COMPARE_CSV, index=False, encoding="utf-8-sig")
        logger.info(f"比較CSV保存: {COMPARE_CSV}")

        print(f"\n=== 旧設計 vs 新設計 比較 ===")
        for label, grp in compare.groupby("変化"):
            print(f"  {label}: {len(grp)}件")
            if label in ("削除", "新規追加"):
                for _, r in grp.iterrows():
                    print(f"    {r['場所']} {r['クラス']} {r['馬券種']}"
                          f"  旧valid={r.get('旧_roi_valid','-')} 旧test={r.get('旧_roi_test','-')}"
                          f"  新valid={r.get('ROI_valid','-'):.1f}% 新test={r.get('ROI_test','-'):.1f}%")

    # --- strategy_weights.json 生成（valid ROIで重み付け、testは記録のみ）---
    strategy: dict = {}
    for (place, cls), grp in adopted.groupby(["場所", "クラス"]):
        total_roi = grp["ROI_valid"].sum()
        bets: dict = {}
        for _, row in grp.iterrows():
            w = row["ROI_valid"] / total_roi
            bets[row["馬券種"]] = {
                "roi_train_combined": round(row["合算ROI_train"], 1),
                "roi_valid":          round(row["ROI_valid"], 1),   # 採用判断に使用
                "roi_test":           round(row["ROI_test"], 1),    # 独立評価（参考）
                "n_races_valid":      int(row["n_valid"]),
                "n_races_test":       int(row["n_test"]),
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
    logger.info(f"strategy_weights.json 保存（walk-forward v2）: {OUT_JSON}")

    # --- サマリ ---
    print(f"\n=== サマリ ===")
    print(f"採用条件数   : {len(adopted)}")
    print(f"会場数       : {adopted['場所'].nunique()} 箇所 ({sorted(adopted['場所'].unique())})")
    print(f"平均信頼度   : {adopted['信頼度'].mean():.1f}%")
    print(f"valid ROI範囲: {adopted['ROI_valid'].min():.1f}% 〜 {adopted['ROI_valid'].max():.1f}%")
    print(f"test ROI範囲 : {adopted['ROI_test'].min():.1f}% 〜 {adopted['ROI_test'].max():.1f}%"
          f"  ← 独立評価")
    print(f"\n【循環解消の確認】")
    print(f"  採用判断データ: valid 2023のみ")
    print(f"  独立評価データ: test 2024-25（採用後に測定）")
    print(f"  test平均ROI {adopted['ROI_test'].mean():.1f}% が真の汎化性能の推定値")


if __name__ == "__main__":
    main()
