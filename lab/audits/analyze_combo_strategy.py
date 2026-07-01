"""
analyze_combo_strategy.py
PyCaLiAI - 馬券種組み合わせのウォークフォワード検証

--no_strategy で生成した backtest_results_combo_{train,valid,test}.csv を読み込み、
以下の組み合わせについて (場所×クラス) ごとに OOS ROI を評価する。

  1. 単勝 + 三連複
  2. 複勝 + 三連複
  3. 単複 + 三連複  (単勝+複勝+三連複)
  4. 枠連 + 三連複
  5. 単勝枠連 + 三連複
  6. 単複枠連 + 三連複 (単勝+複勝+枠連+三連複)

Usage:
    python analyze_combo_strategy.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from itertools import combinations as itercombo

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

TRAIN_CSV = REPORT_DIR / "backtest_results_combo_train.csv"
VALID_CSV = REPORT_DIR / "backtest_results_combo_valid.csv"
TEST_CSV  = REPORT_DIR / "backtest_results_combo_test.csv"
OUT_JSON  = DATA_DIR   / "combo_strategy_results.json"
OUT_CSV   = REPORT_DIR / "combo_strategy_results.csv"

# ウォークフォワード設定（build_strategy_walkforward.py と同じ基準）
EVAL_YEARS           = 5      # 評価年数（2018-2022）
MIN_PROFITABLE_YEARS = 3      # 何年以上黒字なら Train 安定とみなす
MIN_RACES_PER_YEAR   = 15     # 各年の最低レース数
MIN_ROI_OOS          = 100.0  # OOS（valid+test）合算の最低 ROI（%）

# 検証する組み合わせ（馬券種のタプル → 等分予算）
COMBOS: list[tuple[str, ...]] = [
    ("単勝", "三連複"),
    ("複勝", "三連複"),
    ("単勝", "複勝", "三連複"),
]

COMBO_LABELS = {
    ("単勝", "三連複"):         "単勝+三連複",
    ("複勝", "三連複"):         "複勝+三連複",
    ("単勝", "複勝", "三連複"): "単複+三連複",
}

ADD_WEEKEND_FILTER = True   # 土日 & 同会場10R以上に限定（walkforward と統一）


# =========================================================
# ユーティリティ
# =========================================================
def add_weekend_filter(df: pd.DataFrame) -> pd.DataFrame:
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


def combo_roi_by_condition(
    df: pd.DataFrame,
    bet_types: tuple[str, ...],
) -> pd.DataFrame:
    """
    指定した馬券種の組み合わせで、(場所, クラス) ごとのROIを計算。
    各レースで全馬券種が存在する場合のみカウント（等分予算を仮定）。
    """
    sub = df[df["馬券種"].isin(bet_types)].copy()
    if sub.empty:
        return pd.DataFrame()

    # レース×馬券種で集計（購入額・払戻の合計）
    race_bet = (
        sub.groupby(["race_id", "日付", "場所", "クラス", "馬券種"])
        .agg(購入額=("購入額", "sum"), 払戻=("実払戻額", "sum"))
        .reset_index()
    )

    # 全馬券種が揃っているレースのみ使用
    race_type_counts = race_bet.groupby("race_id")["馬券種"].nunique()
    full_races = race_type_counts[race_type_counts == len(bet_types)].index
    race_bet = race_bet[race_bet["race_id"].isin(full_races)]

    if race_bet.empty:
        return pd.DataFrame()

    # 等分予算に正規化: 馬券種ごとの購入額を1に正規化してから合算
    # → ROI は購入額の比率に依存しないが、正規化することで等分の仮定を実現
    # ここでは実際の購入額をそのまま使う（どの馬券種も等分で割り振られているため）
    race_total = (
        race_bet.groupby(["race_id", "日付", "場所", "クラス"])
        .agg(total_cost=("購入額", "sum"), total_pay=("払戻", "sum"))
        .reset_index()
    )

    rows = []
    for (place, cls), grp in race_total.groupby(["場所", "クラス"]):
        n = grp["race_id"].nunique()
        cost = grp["total_cost"].sum()
        pay  = grp["total_pay"].sum()
        if cost == 0:
            continue
        rows.append({
            "場所": place, "クラス": cls,
            "レース数": n,
            "投資":     cost,
            "払戻":     pay,
            "ROI":      pay / cost * 100,
        })
    return pd.DataFrame(rows)


# =========================================================
# ウォークフォワード評価（per combo）
# =========================================================
def walkforward_for_combo(
    train_all: pd.DataFrame,
    valid_df:  pd.DataFrame,
    test_df:   pd.DataFrame,
    bet_types: tuple[str, ...],
    label:     str,
) -> pd.DataFrame:
    """
    1 つの組み合わせについてウォークフォワード + OOS 評価を行い結果を返す。
    """
    train_all = train_all.copy()
    train_all["year"] = pd.to_datetime(
        train_all["日付"].astype(str), format="%Y%m%d"
    ).dt.year

    max_year   = train_all["year"].max()
    eval_years = list(range(max_year - EVAL_YEARS + 1, max_year + 1))

    # 各年の ROI
    yearly_roi: dict[int, pd.DataFrame] = {}
    for yr in eval_years:
        df_yr = train_all[train_all["year"] == yr]
        yearly_roi[yr] = combo_roi_by_condition(df_yr, bet_types)

    # 全条件を収集
    all_conditions: set[tuple] = set()
    for df_yr in yearly_roi.values():
        for _, row in df_yr.iterrows():
            all_conditions.add((row["場所"], row["クラス"]))

    records = []
    for (place, cls) in sorted(all_conditions):
        profitable_years = 0
        years_with_data  = 0
        total_cost = 0
        total_pay  = 0
        yr_details = {}

        for yr in eval_years:
            df_yr = yearly_roi[yr]
            match = df_yr[(df_yr["場所"] == place) & (df_yr["クラス"] == cls)]
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

        if years_with_data < MIN_PROFITABLE_YEARS:
            continue

        combined_roi = total_pay / total_cost * 100 if total_cost > 0 else 0
        rec = {
            "組み合わせ": label,
            "場所": place, "クラス": cls,
            "黒字年数": profitable_years,
            "データあり年数": years_with_data,
            "合算ROI_train": round(combined_roi, 1),
        }
        for yr in eval_years:
            rec[f"ROI_{yr}"] = yr_details.get(yr)
        records.append(rec)

    wf_df = pd.DataFrame(records)
    if wf_df.empty:
        return wf_df

    stable_train = wf_df[wf_df["黒字年数"] >= MIN_PROFITABLE_YEARS].copy()
    if stable_train.empty:
        return stable_train

    # OOS 評価
    oos = pd.concat([valid_df, test_df], ignore_index=True)
    oos_roi = combo_roi_by_condition(oos, bet_types).rename(columns={
        "レース数": "OOS_レース数", "ROI": "ROI_OOS",
        "投資": "OOS_投資", "払戻": "OOS_払戻",
    })

    final = pd.merge(
        stable_train,
        oos_roi[["場所", "クラス", "OOS_レース数", "ROI_OOS"]],
        on=["場所", "クラス"], how="left",
    )
    final["ROI_OOS"]     = final["ROI_OOS"].fillna(0)
    final["OOS_レース数"] = final["OOS_レース数"].fillna(0).astype(int)
    final["信頼度"] = (
        final["黒字年数"] / final["データあり年数"] * 100
    ).round(1)

    return final


# =========================================================
# メイン
# =========================================================
def main() -> None:
    for csv_path, label in [(TRAIN_CSV, "train"), (VALID_CSV, "valid"), (TEST_CSV, "test")]:
        if not csv_path.exists():
            logger.error(
                f"{csv_path.name} が見つかりません。先に以下を実行してください:\n"
                f"  python backtest.py --no_strategy --period {label} "
                f"--output_suffix _combo_{label}"
            )
            return

    logger.info("バックテスト結果 CSV 読み込み中...")
    train_all = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
    valid_raw  = pd.read_csv(VALID_CSV, encoding="utf-8-sig")
    test_raw   = pd.read_csv(TEST_CSV,  encoding="utf-8-sig")

    if ADD_WEEKEND_FILTER:
        train_all = add_weekend_filter(train_all)
        valid_df  = add_weekend_filter(valid_raw)
        test_df   = add_weekend_filter(test_raw)
    else:
        valid_df = valid_raw
        test_df  = test_raw

    logger.info(f"  Train: {len(train_all):,}行")
    logger.info(f"  Valid: {len(valid_df):,}行")
    logger.info(f"  Test:  {len(test_df):,}行")

    all_results: list[pd.DataFrame] = []

    for bet_types in COMBOS:
        label = COMBO_LABELS[bet_types]
        logger.info(f"\n=== {label} ===")
        result = walkforward_for_combo(train_all, valid_df, test_df, bet_types, label)
        if result.empty:
            logger.info("  → 採用条件なし")
            continue

        adopted = result[result["ROI_OOS"] >= MIN_ROI_OOS].sort_values(
            ["ROI_OOS"], ascending=False
        )
        logger.info(f"  Train安定: {len(result)} 条件 / OOS黒字: {len(adopted)} 条件")

        yr_cols = [c for c in result.columns if c.startswith("ROI_20")]
        disp_cols = (
            ["組み合わせ", "場所", "クラス", "黒字年数", "信頼度", "合算ROI_train"]
            + yr_cols
            + ["OOS_レース数", "ROI_OOS"]
        )
        print(f"\n【{label}】OOS ROI {MIN_ROI_OOS}% 以上の条件:")
        if adopted.empty:
            print("  なし")
        else:
            pd.set_option("display.max_rows", 100)
            pd.set_option("display.width", 200)
            print(adopted[disp_cols].to_string(index=False))

        all_results.append(result)

    # まとめ保存
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        logger.info(f"\n全結果 CSV 保存: {OUT_CSV}")

        # OOS 黒字のみ抽出してサマリ表示
        oos_ok = combined[combined["ROI_OOS"] >= MIN_ROI_OOS].sort_values(
            ["組み合わせ", "ROI_OOS"], ascending=[True, False]
        )
        print("\n" + "=" * 80)
        print(f"【サマリ】OOS ROI {MIN_ROI_OOS}% 以上の条件（全組み合わせ）")
        print("=" * 80)
        if oos_ok.empty:
            print("  黒字条件なし")
        else:
            yr_cols = [c for c in combined.columns if c.startswith("ROI_20")]
            disp = ["組み合わせ", "場所", "クラス", "信頼度", "合算ROI_train"] + yr_cols + ["OOS_レース数", "ROI_OOS"]
            print(oos_ok[disp].to_string(index=False))

        # JSON 保存（strategy_weights.json 拡張用の素材）
        json_out = {}
        for _, row in oos_ok.iterrows():
            combo = row["組み合わせ"]
            place = row["場所"]
            cls   = row["クラス"]
            if combo not in json_out:
                json_out[combo] = {}
            if place not in json_out[combo]:
                json_out[combo][place] = {}
            json_out[combo][place][cls] = {
                "roi_oos":          round(row["ROI_OOS"], 1),
                "roi_train":        round(row["合算ROI_train"], 1),
                "profitable_years": int(row["黒字年数"]),
                "confidence":       float(row["信頼度"]),
                "oos_races":        int(row["OOS_レース数"]),
            }
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(json_out, f, ensure_ascii=False, indent=2)
        logger.info(f"JSON 保存: {OUT_JSON}")
    else:
        print("\n採用条件なし（全組み合わせ）")


if __name__ == "__main__":
    main()
