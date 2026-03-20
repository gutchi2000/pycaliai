"""
simulation.py
PyCaLiAI - 複数戦略シミュレーション

backtest_results.csv をベースに複数の買い方戦略を比較する。
予算は全戦略共通で1レース10,000円換算。

Usage:
    python simulation.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    plt.rcParams["font.family"] = "MS Gothic"

from utils import add_meta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR   = Path(r"E:\PyCaLiAI")
REPORT_DIR = BASE_DIR / "reports"
RESULT_CSV = REPORT_DIR / "backtest_results.csv"
BUDGET     = 10_000
MIN_UNIT   = 100


# =========================================================
# ユーティリティ
# =========================================================
def floor_to_unit(x: int, unit: int = MIN_UNIT) -> int:
    return (x // unit) * unit


def reallocate(df: pd.DataFrame, budget: int = BUDGET) -> pd.DataFrame:
    """
    フィルタ後のデータに対してレース×馬券種ごとに予算を再配分する。

    フィルタで馬券種が絞られた場合、残った馬券種に均等配分する。
    購入額・実払戻額・収支を再計算して返す。
    """
    df = df.copy()
    rows = []
    for race_id, group in df.groupby("race_id"):
        bet_types = group["馬券種"].unique()
        n_types   = len(bet_types)
        per_type  = floor_to_unit(budget // n_types)

        for bt, sub in group.groupby("馬券種"):
            n       = len(sub)
            per_bet = floor_to_unit(per_type // n)
            per_bet = max(per_bet, MIN_UNIT)
            while per_bet * n > per_type and per_bet > MIN_UNIT:
                per_bet -= MIN_UNIT
            sub = sub.copy()
            sub["購入額"]   = per_bet
            sub["実払戻額"] = sub.apply(
                lambda r: floor_to_unit(
                    int(per_bet * r["実配当(100円)"] / 100)
                ) if r["的中"] == 1 else 0,
                axis=1,
            )
            sub["収支"] = sub["実払戻額"] - sub["購入額"]
            rows.append(sub)

    if not rows:
        return df
    return pd.concat(rows).reset_index(drop=True)


def full_budget(df: pd.DataFrame, budget: int = BUDGET) -> pd.DataFrame:
    """
    1馬券種に予算全振りする場合の再配分。
    レースごとに買い目点数で均等配分。
    """
    df = df.copy()
    rows = []
    for race_id, group in df.groupby("race_id"):
        n       = len(group)
        per_bet = floor_to_unit(budget // n)
        per_bet = max(per_bet, MIN_UNIT)
        while per_bet * n > budget and per_bet > MIN_UNIT:
            per_bet -= MIN_UNIT
        group = group.copy()
        group["購入額"]   = per_bet
        group["実払戻額"] = group.apply(
            lambda r: floor_to_unit(
                int(per_bet * r["実配当(100円)"] / 100)
            ) if r["的中"] == 1 else 0,
            axis=1,
        )
        group["収支"] = group["実払戻額"] - group["購入額"]
        rows.append(group)

    if not rows:
        return df
    return pd.concat(rows).reset_index(drop=True)


def roi_summary(df: pd.DataFrame, label: str) -> dict:
    n_races  = df["race_id"].nunique()
    cost     = df["購入額"].sum()
    pay      = df["実払戻額"].sum()
    net      = pay - cost
    roi      = pay / cost * 100 if cost > 0 else 0
    hits     = df["的中"].sum()
    n        = len(df)
    hit_r    = hits / n * 100 if n > 0 else 0
    # 月次収支の標準偏差（安定性指標）
    monthly  = (
        df.groupby(df["date"].dt.to_period("M"))["収支"]
        .sum()
    )
    std_monthly = monthly.std() if len(monthly) > 1 else 0
    return {
        "戦略":         label,
        "レース数":     n_races,
        "点数":         n,
        "総投資":       int(cost),
        "総払戻":       int(pay),
        "純収支":       int(net),
        "回収率(%)":    round(roi, 1),
        "的中率(%)":    round(hit_r, 1),
        "的中数":       int(hits),
        "月次収支std":  int(std_monthly),
    }


# =========================================================
# 戦略定義
# =========================================================
def strategy_filter(df: pd.DataFrame, strategy_id: str) -> pd.DataFrame:
    """
    戦略IDに対応するフィルタを適用してDataFrameを返す。
    予算再配分も実施する。
    """
    base = df[df["週末10R"]].copy()  # 全戦略で土日10R以上を前提

    if strategy_id == "S00":
        # ベースライン: 現状の全馬券種
        return reallocate(base)

    elif strategy_id == "S01":
        # 新馬×馬連 全会場 予算全振り
        sub = base[(base["クラス"] == "新馬") & (base["馬券種"] == "馬連")]
        return full_budget(sub)

    elif strategy_id == "S02":
        # 3勝×三連複 中山除外 予算全振り
        sub = base[
            (base["クラス"] == "3勝") &
            (base["馬券種"] == "三連複") &
            (base["場所"] != "中山")
        ]
        return full_budget(sub)

    elif strategy_id == "S03":
        # 3勝×三連複 全会場 予算全振り
        sub = base[(base["クラス"] == "3勝") & (base["馬券種"] == "三連複")]
        return full_budget(sub)

    elif strategy_id == "S04":
        # 新馬×馬連 + 3勝×三連複（中山除外）按分
        s1 = base[(base["クラス"] == "新馬") & (base["馬券種"] == "馬連")].copy()
        s2 = base[
            (base["クラス"] == "3勝") &
            (base["馬券種"] == "三連複") &
            (base["場所"] != "中山")
        ].copy()
        s1["購入額"] = BUDGET // 2
        s2["購入額"] = BUDGET // 2
        for s in [s1, s2]:
            s["実払戻額"] = s.apply(
                lambda r: floor_to_unit(
                    int(r["購入額"] * r["実配当(100円)"] / 100)
                ) if r["的中"] == 1 else 0, axis=1
            )
            s["収支"] = s["実払戻額"] - s["購入額"]
        return pd.concat([s1, s2]).reset_index(drop=True)

    elif strategy_id == "S05":
        # G3×三連複 全会場 予算全振り
        sub = base[(base["クラス"] == "Ｇ３") & (base["馬券種"] == "三連複")]
        return full_budget(sub)

    elif strategy_id == "S06":
        # G3×馬連 全会場 予算全振り
        sub = base[(base["クラス"] == "Ｇ３") & (base["馬券種"] == "馬連")]
        return full_budget(sub)

    elif strategy_id == "S07":
        # 新馬×馬連 + G3×三連複 按分
        s1 = base[(base["クラス"] == "新馬") & (base["馬券種"] == "馬連")].copy()
        s2 = base[(base["クラス"] == "Ｇ３") & (base["馬券種"] == "三連複")].copy()
        for s in [s1, s2]:
            s["購入額"] = BUDGET // 2
            s["実払戻額"] = s.apply(
                lambda r: floor_to_unit(
                    int(r["購入額"] * r["実配当(100円)"] / 100)
                ) if r["的中"] == 1 else 0, axis=1
            )
            s["収支"] = s["実払戻額"] - s["購入額"]
        return pd.concat([s1, s2]).reset_index(drop=True)

    elif strategy_id == "S08":
        # 新馬/3勝/G3/OP(L) × 馬連のみ
        sub = base[
            (base["クラス"].isin(["新馬", "3勝", "Ｇ３", "OP(L)"])) &
            (base["馬券種"] == "馬連")
        ]
        return full_budget(sub)

    elif strategy_id == "S09":
        # 複勝 ◎のみ 予算全振り
        sub = base[
            (base["馬券種"] == "複勝") &
            (base["買い目"].apply(lambda x: len(str(x).split("-")) == 1))
        ]
        # 印が◎の1点のみ（買い目の最初の馬＝◎）
        # backtest_results.csvには印列がないため
        # 複勝2点のうち推定的中確率が高い方を◎と判断
        sub = sub.sort_values(["race_id", "推定的中確率"], ascending=[True, False])
        sub = sub.groupby("race_id").first().reset_index()
        return full_budget(sub)

    elif strategy_id == "S10":
        # 三連単 推定期待値上位1点のみ 予算全振り
        sub = base[base["馬券種"] == "三連単"]
        sub = sub.sort_values(["race_id", "推定期待値"], ascending=[True, False])
        sub = sub.groupby("race_id").first().reset_index()
        return full_budget(sub)

    elif strategy_id == "S11":
        # 東京・中山・中京・小倉のみ 全馬券種
        sub = base[base["場所"].isin(["東京", "中山", "中京", "小倉"])]
        return reallocate(sub)

    elif strategy_id == "S12":
        # 新馬×馬連 × 中山・中京（東京・小倉はアプリと合わせて除外）
        sub = base[
            (base["クラス"] == "新馬") &
            (base["馬券種"] == "馬連") &
            (base["場所"].isin(["中山", "中京"]))
        ]
        return full_budget(sub)

    elif strategy_id == "S13":
        # 3勝×馬連 全会場（馬連で安定を狙う）
        sub = base[(base["クラス"] == "3勝") & (base["馬券種"] == "馬連")]
        return full_budget(sub)

    elif strategy_id == "S14":
        # 新馬×馬連 + 3勝×馬連 + G3×馬連（馬連統一戦略）
        sub = base[
            (base["クラス"].isin(["新馬", "3勝", "Ｇ３"])) &
            (base["馬券種"] == "馬連")
        ]
        return reallocate(sub)

    return base


STRATEGIES = {
    "S00": "ベースライン（全馬券種）",
    "S01": "新馬×馬連 全会場",
    "S02": "3勝×三連複 中山除外",
    "S03": "3勝×三連複 全会場",
    "S04": "新馬×馬連 + 3勝×三連複(中山除外) 按分",
    "S05": "G3×三連複 全会場",
    "S06": "G3×馬連 全会場",
    "S07": "新馬×馬連 + G3×三連複 按分",
    "S08": "新馬/3勝/G3/OP(L)×馬連統一",
    "S09": "複勝◎1点全振り",
    "S10": "三連単期待値1位1点全振り",
    "S11": "高回収率4会場のみ 全馬券種",
    "S12": "新馬×馬連 中山・中京のみ",
    "S13": "3勝×馬連 全会場",
    "S14": "新馬/3勝/G3×馬連統一",
}


# =========================================================
# 可視化
# =========================================================
def plot_roi_comparison(summary_df: pd.DataFrame, save_path: Path) -> None:
    df = summary_df.sort_values("回収率(%)", ascending=True)
    colors = ["tomato" if v >= 100 else "steelblue" for v in df["回収率(%)"]]

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(df["戦略"], df["回収率(%)"], color=colors)
    ax.axvline(100, color="gray", linewidth=1.0, linestyle="--", label="100%")
    ax.set_title("戦略別回収率比較")
    ax.set_xlabel("回収率（%）")
    for bar, val in zip(bars, df["回収率(%)"]):
        ax.text(
            bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=8,
        )
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"グラフ保存: {save_path}")


def plot_cumulative_all(
    results: dict[str, pd.DataFrame],
    highlight: list[str],
    save_path: Path,
) -> None:
    """全戦略の累積収支をプロット。highlight戦略は太線で強調。"""
    fig, ax = plt.subplots(figsize=(12, 6))

    for sid, df in results.items():
        if df.empty:
            continue
        pnl = (
            df.groupby("race_id")["収支"]
            .sum()
            .reset_index()["収支"]
            .cumsum()
            .values
        )
        label = f"{sid}: {STRATEGIES[sid]}"
        lw    = 2.5 if sid in highlight else 0.8
        alpha = 1.0 if sid in highlight else 0.3
        ax.plot(pnl, label=label, linewidth=lw, alpha=alpha)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("戦略別累積収支推移")
    ax.set_ylabel("累積収支（円）")
    ax.set_xlabel("レース数（対象レースのみカウント）")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"グラフ保存: {save_path}")


def plot_monthly(
    results: dict[str, pd.DataFrame],
    highlight: list[str],
    save_path: Path,
) -> None:
    """highlight戦略の月次収支を棒グラフで表示。"""
    n = len(highlight)
    fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]

    for ax, sid in zip(axes, highlight):
        df = results.get(sid, pd.DataFrame())
        if df.empty:
            ax.set_title(f"{sid}: データなし")
            continue
        monthly = df.groupby(df["date"].dt.to_period("M"))["収支"].sum()
        colors  = ["tomato" if v >= 0 else "steelblue" for v in monthly.values]
        ax.bar(monthly.index.astype(str), monthly.values, color=colors)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(f"{sid}: {STRATEGIES[sid]} 月次収支")
        ax.set_ylabel("収支（円）")
        ax.tick_params(axis="x", rotation=45)

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
    df = add_meta(df)
    logger.info(f"  {len(df):,}行  対象レース: {df['race_id'].nunique():,}")

    # 全戦略を実行
    results: dict[str, pd.DataFrame] = {}
    summaries = []

    for sid, label in STRATEGIES.items():
        logger.info(f"戦略 {sid}: {label}")
        filtered = strategy_filter(df, sid)
        if filtered.empty:
            logger.warning(f"  → データなし（スキップ）")
            continue
        results[sid] = filtered
        s = roi_summary(filtered, f"{sid}: {label}")
        summaries.append(s)
        logger.info(
            f"  レース数:{s['レース数']:,}  "
            f"回収率:{s['回収率(%)']:.1f}%  "
            f"純収支:{s['純収支']:+,}円"
        )

    # サマリ表示
    summary_df = pd.DataFrame(summaries).sort_values("回収率(%)", ascending=False)

    print("\n" + "=" * 90)
    print("シミュレーション結果サマリ")
    print("=" * 90)
    print(
        summary_df[[
            "戦略", "レース数", "点数", "総投資", "総払戻",
            "純収支", "回収率(%)", "的中率(%)", "月次収支std"
        ]].to_string(index=False)
    )

    # 回収率100%超えの戦略
    over100 = summary_df[summary_df["回収率(%)"] >= 100]
    print(f"\n回収率100%超えの戦略: {len(over100)}個")
    if not over100.empty:
        print(over100[["戦略", "レース数", "回収率(%)", "純収支", "月次収支std"]].to_string(index=False))

    # 上位3戦略をhighlightに設定
    highlight = summary_df.head(5)["戦略"].str[:3].tolist()

    # グラフ
    plot_roi_comparison(summary_df, REPORT_DIR / "sim_roi_comparison.png")
    plot_cumulative_all(results, highlight, REPORT_DIR / "sim_cumulative_all.png")
    plot_monthly(results, highlight[:3], REPORT_DIR / "sim_monthly_top3.png")

    # CSV保存
    out_path = REPORT_DIR / "simulation_summary.csv"
    summary_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"サマリ保存: {out_path}")

    # 戦略別詳細CSV
    for sid, fdf in results.items():
        fdf.to_csv(
            REPORT_DIR / f"sim_{sid}.csv",
            index=False, encoding="utf-8-sig"
        )

    print(f"\n保存先: {REPORT_DIR}")
    print(f"グラフ: sim_roi_comparison.png / sim_cumulative_all.png / sim_monthly_top3.png")


if __name__ == "__main__":
    main()