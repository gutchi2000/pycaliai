"""
filter_analysis.py
PyCaLiAI - 条件フィルタリング分析

「土日に10R以上開催される条件」を特定し
回収率100%超えの買い方を探す。

Usage:
    python filter_analysis.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    plt.rcParams["font.family"] = "MS Gothic"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR   = Path(r"E:\PyCaLiAI")
REPORT_DIR = BASE_DIR / "reports"
RESULT_CSV = REPORT_DIR / "backtest_results.csv"


# =========================================================
# ユーティリティ
# =========================================================
def roi_summary(df: pd.DataFrame, label: str = "") -> dict:
    """回収率サマリを計算して返す。"""
    cost    = df["購入額"].sum()
    pay     = df["実払戻額"].sum()
    hits    = df["的中"].sum()
    n       = len(df)
    n_races = df["race_id"].nunique()
    roi     = pay / cost * 100 if cost > 0 else 0
    hit_r   = hits / n * 100 if n > 0 else 0
    return {
        "ラベル":   label,
        "レース数": n_races,
        "点数":     n,
        "投資":     cost,
        "払戻":     pay,
        "収支":     pay - cost,
        "回収率":   round(roi, 1),
        "的中率":   round(hit_r, 1),
        "的中数":   int(hits),
    }


def print_summary(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    print(df[["ラベル", "レース数", "点数", "投資", "払戻", "収支", "回収率", "的中率"]].to_string(index=False))


# =========================================================
# 土日フィルタ
# =========================================================
def add_weekday(df: pd.DataFrame) -> pd.DataFrame:
    """日付列から曜日を追加する。"""
    df = df.copy()
    df["date"] = pd.to_datetime(df["日付"].astype(str), format="%Y%m%d")
    df["曜日"]  = df["date"].dt.dayofweek   # 0=月 5=土 6=日
    df["土日"]  = df["曜日"].isin([5, 6])
    return df


def filter_weekend_10r(df: pd.DataFrame) -> pd.DataFrame:
    """
    土日かつ当日・同会場で10R以上開催されるレースのみ残す。

    JRAの土日開催は通常12Rなので実質「土日の全レース」に近い。
    平日・地方・障害のみ開催日を除外するフィルタとして機能する。
    """
    df = add_weekday(df)

    # 土日のみ
    df_weekend = df[df["土日"]].copy()

    # 同日・同会場のレース数を集計
    race_count = (
        df_weekend.groupby(["日付", "場所"])["race_id"]
        .nunique()
        .reset_index()
        .rename(columns={"race_id": "当日同会場R数"})
    )
    df_weekend = df_weekend.merge(race_count, on=["日付", "場所"], how="left")

    # 10R以上の開催日のみ
    return df_weekend[df_weekend["当日同会場R数"] >= 10].copy()


# =========================================================
# 条件別分析
# =========================================================
def analyze_by_condition(
    df: pd.DataFrame,
    col: str,
    min_races: int = 30,
) -> pd.DataFrame:
    """
    指定列で集計し回収率を返す。
    min_races未満のレースは除外（サンプル少なすぎ）。
    """
    rows = []
    for val, sub in df.groupby(col):
        n_races = sub["race_id"].nunique()
        if n_races < min_races:
            continue
        s = roi_summary(sub, label=str(val))
        s[col] = val
        rows.append(s)
    return pd.DataFrame(rows).sort_values("回収率", ascending=False)


def analyze_by_bet_type(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bt, sub in df.groupby("馬券種"):
        rows.append(roi_summary(sub, label=bt))
    return pd.DataFrame(rows).sort_values("回収率", ascending=False)


def analyze_combined(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    min_races: int = 20,
) -> pd.DataFrame:
    """2条件の組み合わせで集計。"""
    rows = []
    for (v1, v2), sub in df.groupby([col1, col2]):
        n_races = sub["race_id"].nunique()
        if n_races < min_races:
            continue
        s = roi_summary(sub, label=f"{v1}×{v2}")
        rows.append(s)
    return pd.DataFrame(rows).sort_values("回収率", ascending=False)


# =========================================================
# 可視化
# =========================================================
def plot_bar_roi(
    df: pd.DataFrame,
    label_col: str,
    title: str,
    save_path: Path,
) -> None:
    df = df.sort_values("回収率", ascending=True)
    colors = ["tomato" if v >= 100 else "steelblue" for v in df["回収率"]]

    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.5)))
    ax.barh(df[label_col].astype(str), df["回収率"], color=colors)
    ax.axvline(100, color="gray", linewidth=1.0, linestyle="--", label="100%")
    ax.set_title(title)
    ax.set_xlabel("回収率（%）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"グラフ保存: {save_path}")


def plot_cumulative_filtered(
    df_all: pd.DataFrame,
    df_filtered: pd.DataFrame,
    save_path: Path,
) -> None:
    """全体 vs フィルタ後の累積収支比較。"""
    def cum_pnl(df):
        return (
            df.groupby("race_id")["収支"]
            .sum()
            .reset_index()["収支"]
            .cumsum()
            .values
        )

    pnl_all      = cum_pnl(df_all)
    pnl_filtered = cum_pnl(df_filtered)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pnl_all,      label=f"全体（{len(df_all['race_id'].unique()):,}R）",      color="steelblue", alpha=0.6)
    ax.plot(pnl_filtered, label=f"土日10R以上（{len(df_filtered['race_id'].unique()):,}R）", color="tomato")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title("累積収支比較：全体 vs 土日10R以上")
    ax.set_ylabel("累積収支（円）")
    ax.set_xlabel("レース数")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"グラフ保存: {save_path}")


# =========================================================
# main
# =========================================================
def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"バックテスト結果読み込み: {RESULT_CSV}")
    df = pd.read_csv(RESULT_CSV, encoding="utf-8-sig")
    logger.info(f"  {len(df):,}行")

    # ---- 土日10R以上フィルタ ----
    df_w = filter_weekend_10r(df)
    logger.info(f"土日10R以上フィルタ後: {df_w['race_id'].nunique():,}レース")

    print("\n" + "=" * 70)
    print("【全体 vs 土日10R以上】")
    print("=" * 70)
    print_summary([
        roi_summary(df,   "全体"),
        roi_summary(df_w, "土日10R以上"),
    ])

    # ---- 馬券種別 ----
    print("\n" + "=" * 70)
    print("【馬券種別（土日10R以上）】")
    print("=" * 70)
    bt_df = analyze_by_bet_type(df_w)
    print_summary(bt_df.to_dict("records"))

    # ---- 芝ダ別 ----
    print("\n" + "=" * 70)
    print("【芝ダ別（土日10R以上）】")
    print("=" * 70)
    shida_df = analyze_by_condition(df_w, "芝ダ", min_races=10)
    print_summary(shida_df.to_dict("records"))

    # ---- 競馬場別 ----
    print("\n" + "=" * 70)
    print("【競馬場別（土日10R以上）】")
    print("=" * 70)
    place_df = analyze_by_condition(df_w, "場所", min_races=30)
    print_summary(place_df.to_dict("records"))

    # ---- クラス別 ----
    print("\n" + "=" * 70)
    print("【クラス別（土日10R以上）】")
    print("=" * 70)
    cls_df = analyze_by_condition(df_w, "クラス", min_races=20)
    print_summary(cls_df.to_dict("records"))

    # ---- 馬券種×芝ダ 組み合わせ ----
    print("\n" + "=" * 70)
    print("【馬券種×芝ダ（土日10R以上）上位10】")
    print("=" * 70)
    combo_df = analyze_combined(df_w, "馬券種", "芝ダ", min_races=10)
    print_summary(combo_df.head(10).to_dict("records"))

    # ---- 馬券種×クラス 組み合わせ ----
    print("\n" + "=" * 70)
    print("【馬券種×クラス（土日10R以上）上位10】")
    print("=" * 70)
    combo2_df = analyze_combined(df_w, "馬券種", "クラス", min_races=10)
    print_summary(combo2_df.head(10).to_dict("records"))

    # ---- グラフ ----
    plot_bar_roi(place_df, "ラベル", "競馬場別回収率（土日10R以上）",
                 REPORT_DIR / "filter_place_roi.png")
    plot_bar_roi(cls_df,   "ラベル", "クラス別回収率（土日10R以上）",
                 REPORT_DIR / "filter_class_roi.png")
    plot_bar_roi(bt_df,    "ラベル", "馬券種別回収率（土日10R以上）",
                 REPORT_DIR / "filter_bettype_roi.png")
    plot_cumulative_filtered(df, df_w,
                 REPORT_DIR / "filter_cumulative_compare.png")

    # ---- CSV保存 ----
    for name, data in [
        ("filter_bettype",  bt_df),
        ("filter_place",    place_df),
        ("filter_class",    cls_df),
        ("filter_combo_bettype_shida",  combo_df),
        ("filter_combo_bettype_class",  combo2_df),
    ]:
        out = REPORT_DIR / f"{name}.csv"
        data.to_csv(out, index=False, encoding="utf-8-sig")
    logger.info("CSV保存完了")

    print(f"\n保存先: {REPORT_DIR}")


if __name__ == "__main__":
    main()