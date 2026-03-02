"""
validation.py
PyCaLiAI - 検証3点セット

1. 時系列過学習チェック（2013-2022発見 → 2023-2024検証）
2. ドローダウン分析（最大連敗・必要資金・最悪期）
3. モンテカルロシミュレーション（1000試行・信頼区間）

対象戦略: S07（新馬×馬連 + G3×三連複）/ S04（新馬×馬連 + 3勝×三連複）
          全戦略も一括で確認可能

Usage:
    python validation.py
    python validation.py --strategies S07 S04 S01
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings("ignore")

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
DATA_DIR   = BASE_DIR / "data"
REPORT_DIR = BASE_DIR / "reports"

MASTER_CSV = DATA_DIR  / "master_20130105-20251228.csv"
KEKKA_CSV  = DATA_DIR  / "kekka_20130105-20251228.csv"

BUDGET     = 10_000
MIN_UNIT   = 100
N_MONTE    = 1000
RANDOM_STATE = 42

# 検証対象戦略（simulation.pyと同じ定義）
STRATEGIES = {
    "S01": "新馬×馬連 全会場",
    "S02": "3勝×三連複 中山除外",
    "S04": "新馬×馬連 + 3勝×三連複(中山除外) 按分",
    "S05": "G3×三連複 全会場",
    "S06": "G3×馬連 全会場",
    "S07": "新馬×馬連 + G3×三連複 按分",
    "S08": "新馬/3勝/G3/OP(L)×馬連統一",
    "S12": "新馬×馬連 高回収率4会場のみ",
    "S14": "新馬/3勝/G3×馬連統一",
}


# =========================================================
# ユーティリティ
# =========================================================
def floor_to_unit(x: int, unit: int = MIN_UNIT) -> int:
    return (x // unit) * unit


def add_meta(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["日付"].astype(str), format="%Y%m%d")
    df["曜日"]  = df["date"].dt.dayofweek
    df["土日"]  = df["曜日"].isin([5, 6])
    rc = (
        df.groupby(["日付", "場所"])["race_id"]
        .nunique()
        .reset_index()
        .rename(columns={"race_id": "R数"})
    )
    df = df.merge(rc, on=["日付", "場所"], how="left")
    df["週末10R"] = df["土日"] & (df["R数"] >= 10)
    return df


def full_budget_series(df: pd.DataFrame, budget: int = BUDGET) -> pd.DataFrame:
    """レースごとに予算均等配分して収支を再計算。"""
    rows = []
    for _, group in df.groupby("race_id"):
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
            ) if r["的中"] == 1 else 0, axis=1
        )
        group["収支"] = group["実払戻額"] - group["購入額"]
        rows.append(group)
    return pd.concat(rows).reset_index(drop=True) if rows else df


def apply_strategy(df: pd.DataFrame, sid: str) -> pd.DataFrame:
    """戦略フィルタを適用して収支を再計算したDataFrameを返す。"""
    base = df[df["週末10R"]].copy()

    if sid == "S01":
        sub = base[(base["クラス"] == "新馬") & (base["馬券種"] == "馬連")]
        return full_budget_series(sub)

    elif sid == "S02":
        sub = base[
            (base["クラス"] == "3勝") &
            (base["馬券種"] == "三連複") &
            (base["場所"] != "中山")
        ]
        return full_budget_series(sub)

    elif sid == "S04":
        s1 = base[(base["クラス"] == "新馬") & (base["馬券種"] == "馬連")].copy()
        s2 = base[
            (base["クラス"] == "3勝") &
            (base["馬券種"] == "三連複") &
            (base["場所"] != "中山")
        ].copy()
        for s in [s1, s2]:
            s["購入額"] = BUDGET // 2
            s["実払戻額"] = s.apply(
                lambda r: floor_to_unit(
                    int(r["購入額"] * r["実配当(100円)"] / 100)
                ) if r["的中"] == 1 else 0, axis=1
            )
            s["収支"] = s["実払戻額"] - s["購入額"]
        return pd.concat([s1, s2]).reset_index(drop=True)

    elif sid == "S05":
        sub = base[(base["クラス"] == "Ｇ３") & (base["馬券種"] == "三連複")]
        return full_budget_series(sub)

    elif sid == "S06":
        sub = base[(base["クラス"] == "Ｇ３") & (base["馬券種"] == "馬連")]
        return full_budget_series(sub)

    elif sid == "S07":
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

    elif sid == "S08":
        sub = base[
            (base["クラス"].isin(["新馬", "3勝", "Ｇ３", "OP(L)"])) &
            (base["馬券種"] == "馬連")
        ]
        return full_budget_series(sub)

    elif sid == "S12":
        sub = base[
            (base["クラス"] == "新馬") &
            (base["馬券種"] == "馬連") &
            (base["場所"].isin(["東京", "中山", "中京", "小倉"]))
        ]
        return full_budget_series(sub)

    elif sid == "S14":
        sub = base[
            (base["クラス"].isin(["新馬", "3勝", "Ｇ３"])) &
            (base["馬券種"] == "馬連")
        ]
        return full_budget_series(sub)

    return base


# =========================================================
# 1. 時系列過学習チェック
# =========================================================
def timeseries_validation(
    df: pd.DataFrame,
    strategies: list[str],
) -> pd.DataFrame:
    """
    時系列分割で過学習を検証する。

    分割:
      発見期間: 2013〜2022年
      検証期間: 2023〜2024年

    同じ戦略が未知データでも回収率100%超えを維持するか確認。
    """
    logger.info("=== 1. 時系列過学習チェック ===")

    df["year"] = df["date"].dt.year
    train_df = df[df["year"] <= 2022].copy()
    test_df  = df[df["year"] >= 2023].copy()

    rows = []
    for sid in strategies:
        for period, sub_df in [("発見期(〜2022)", train_df), ("検証期(2023〜)", test_df)]:
            filtered = apply_strategy(sub_df, sid)
            if filtered.empty:
                continue
            cost    = filtered["購入額"].sum()
            pay     = filtered["実払戻額"].sum()
            roi     = pay / cost * 100 if cost > 0 else 0
            n_races = filtered["race_id"].nunique()
            rows.append({
                "戦略ID":   sid,
                "戦略名":   STRATEGIES[sid],
                "期間":     period,
                "レース数": n_races,
                "総投資":   int(cost),
                "総払戻":   int(pay),
                "純収支":   int(pay - cost),
                "回収率":   round(roi, 1),
            })

    result = pd.DataFrame(rows)

    print("\n" + "=" * 80)
    print("【1. 時系列過学習チェック】")
    print("=" * 80)
    for sid in strategies:
        sub = result[result["戦略ID"] == sid]
        if sub.empty:
            continue
        print(f"\n{sid}: {STRATEGIES[sid]}")
        print(sub[["期間", "レース数", "総投資", "純収支", "回収率"]].to_string(index=False))

    return result


def plot_timeseries_roi(result: pd.DataFrame, save_path: Path) -> None:
    strategies = result["戦略ID"].unique()
    n = len(strategies)
    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.8)))

    x    = np.arange(n)
    w    = 0.35
    periods = ["発見期(〜2022)", "検証期(2023〜)"]
    colors  = ["steelblue", "tomato"]

    for i, (period, color) in enumerate(zip(periods, colors)):
        sub = result[result["期間"] == period]
        # 戦略順に並び替え
        rois = [
            sub[sub["戦略ID"] == sid]["回収率"].values[0]
            if sid in sub["戦略ID"].values else 0
            for sid in strategies
        ]
        ax.bar(x + i * w, rois, w, label=period, color=color, alpha=0.8)

    ax.axhline(100, color="gray", linewidth=1.0, linestyle="--")
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(strategies, rotation=30)
    ax.set_ylabel("回収率（%）")
    ax.set_title("時系列検証：発見期 vs 検証期 回収率比較")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"グラフ保存: {save_path}")


# =========================================================
# 2. ドローダウン分析
# =========================================================
def drawdown_analysis(
    df: pd.DataFrame,
    strategies: list[str],
) -> pd.DataFrame:
    """
    ドローダウン分析。

    指標:
      最大ドローダウン: ピークから谷までの最大損失額
      最大連敗数:      連続して収支マイナスになったレース数
      最大連敗損失:    連敗中の累積損失額
      必要資金:        破産しないための推奨資金（最大DDの3倍）
      回復レース数:    最大DDから回復するまでのレース数
    """
    logger.info("=== 2. ドローダウン分析 ===")

    rows = []
    print("\n" + "=" * 80)
    print("【2. ドローダウン分析】")
    print("=" * 80)

    for sid in strategies:
        filtered = apply_strategy(df, sid)
        if filtered.empty:
            continue

        # レース単位の収支系列
        race_pnl = (
            filtered.groupby("race_id")["収支"]
            .sum()
            .reset_index()
            .sort_values("race_id")
            ["収支"]
            .values
        )

        # 累積収支
        cumsum = np.cumsum(race_pnl)

        # ドローダウン計算
        peak = np.maximum.accumulate(cumsum)
        dd   = cumsum - peak   # 負の値

        max_dd     = int(dd.min())
        max_dd_idx = int(np.argmin(dd))

        # ピーク位置
        peak_idx = int(np.argmax(cumsum[:max_dd_idx + 1]))

        # 回復レース数（最大DD後に元の高値を超えるまで）
        recover = 0
        if max_dd_idx < len(cumsum) - 1:
            peak_val = cumsum[peak_idx]
            future   = cumsum[max_dd_idx:]
            recovered = np.where(future >= peak_val)[0]
            recover  = int(recovered[0]) if len(recovered) > 0 else -1

        # 最大連敗
        max_consec_loss  = 0
        max_consec_money = 0
        cur_consec       = 0
        cur_money        = 0
        for pnl in race_pnl:
            if pnl < 0:
                cur_consec += 1
                cur_money  += pnl
                if cur_consec > max_consec_loss:
                    max_consec_loss  = cur_consec
                    max_consec_money = cur_money
            else:
                cur_consec = 0
                cur_money  = 0

        # 必要資金（最大DDの3倍を推奨）
        required_capital = abs(max_dd) * 3

        rows.append({
            "戦略ID":        sid,
            "戦略名":        STRATEGIES[sid],
            "最大DD(円)":    max_dd,
            "最大連敗R":     max_consec_loss,
            "連敗損失(円)":  int(max_consec_money),
            "回復R数":       recover if recover >= 0 else "未回復",
            "推奨資金(円)":  required_capital,
        })

        print(f"\n{sid}: {STRATEGIES[sid]}")
        print(f"  最大ドローダウン : {max_dd:>+12,}円")
        print(f"  最大連敗レース数 : {max_consec_loss:>5}R")
        print(f"  連敗中累積損失   : {int(max_consec_money):>+12,}円")
        print(f"  回復レース数     : {recover if recover >= 0 else '未回復':>5}")
        print(f"  推奨必要資金     : {required_capital:>12,}円")

    return pd.DataFrame(rows)


def plot_drawdown(
    df: pd.DataFrame,
    strategies: list[str],
    save_path: Path,
) -> None:
    n   = len(strategies)
    fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]

    for ax, sid in zip(axes, strategies):
        filtered = apply_strategy(df, sid)
        if filtered.empty:
            ax.set_title(f"{sid}: データなし")
            continue

        race_pnl = (
            filtered.groupby("race_id")["収支"]
            .sum()
            .reset_index()
            .sort_values("race_id")
            ["収支"]
            .values
        )
        cumsum = np.cumsum(race_pnl)
        peak   = np.maximum.accumulate(cumsum)
        dd     = cumsum - peak

        ax.fill_between(range(len(dd)), dd, 0, alpha=0.4, color="tomato", label="ドローダウン")
        ax.plot(cumsum, color="steelblue", linewidth=1.2, label="累積収支")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(f"{sid}: {STRATEGIES[sid]}")
        ax.set_ylabel("収支（円）")
        ax.set_xlabel("レース数")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"グラフ保存: {save_path}")


# =========================================================
# 3. モンテカルロシミュレーション
# =========================================================
def monte_carlo(
    df: pd.DataFrame,
    strategies: list[str],
    n_sim: int = N_MONTE,
) -> pd.DataFrame:
    """
    モンテカルロシミュレーション（正しい設計版）。

    各買い目を以下の2段階でモデル化:
      1. 的中するか: Bernoulli(実績的中率)
      2. 的中したときの払戻倍率: 実績配当分布からランダムサンプリング

    これにより「運が良い/悪い年」の回収率分布を正確に推定する。
    """
    logger.info(f"=== 3. モンテカルロシミュレーション ({n_sim}回) ===")
    np.random.seed(RANDOM_STATE)

    rows = []
    print("\n" + "=" * 80)
    print(f"【3. モンテカルロシミュレーション（{n_sim:,}試行・実績分布ベース）】")
    print("=" * 80)

    for sid in tqdm(strategies, desc="モンテカルロ"):
        filtered = apply_strategy(df, sid)
        if filtered.empty:
            continue

        total_cost  = filtered["購入額"].sum()
        actual_roi  = filtered["実払戻額"].sum() / total_cost * 100
        n_bets      = len(filtered)

        # 馬券種×クラス単位で的中率と配当分布を計算
        group_stats = {}
        for (bt, cls), grp in filtered.groupby(["馬券種", "クラス"]):
            hit_rate   = grp["的中"].mean()
            hit_rows   = grp[grp["的中"] == 1]
            # 的中時の払戻倍率リスト（実配当/100）
            if len(hit_rows) > 0:
                odds_dist = (hit_rows["実配当(100円)"] / 100).values
            else:
                odds_dist = np.array([0.0])
            group_stats[(bt, cls)] = {
                "hit_rate":  hit_rate,
                "odds_dist": odds_dist,
                "purchase":  grp["購入額"].values,
            }

        # n_sim回シミュレーション
        sim_rois = np.zeros(n_sim)

        for _ in range(n_sim):
            total_pay = 0
            total_inv = 0
            for (bt, cls), stats in group_stats.items():
                n       = len(stats["purchase"])
                amounts = stats["purchase"]
                # 的中判定
                hits = np.random.binomial(1, stats["hit_rate"], size=n)
                # 的中した買い目の払戻倍率をサンプリング
                for i, (hit, amt) in enumerate(zip(hits, amounts)):
                    total_inv += amt
                    if hit:
                        odds = np.random.choice(stats["odds_dist"])
                        total_pay += floor_to_unit(int(amt * odds))
            sim_rois[_] = total_pay / total_inv * 100 if total_inv > 0 else 0

        p5, p25, p50, p75, p95 = np.percentile(sim_rois, [5, 25, 50, 75, 95])
        ruin_prob = (sim_rois < 50).mean() * 100  # 回収率50%未満の確率

        # 平均的中率
        overall_hit = filtered["的中"].mean()

        rows.append({
            "戦略ID":      sid,
            "戦略名":      STRATEGIES[sid],
            "実回収率":    round(actual_roi, 1),
            "5%ile":      round(p5,  1),
            "25%ile":     round(p25, 1),
            "中央値":     round(p50, 1),
            "75%ile":     round(p75, 1),
            "95%ile":     round(p95, 1),
            "大損確率(%)": round(ruin_prob, 1),
        })

        print(f"\n{sid}: {STRATEGIES[sid]}")
        print(f"  実際の回収率  : {actual_roi:.1f}%")
        print(f"  実績的中率    : {overall_hit*100:.1f}%")
        print(f"  回収率 5%ile  : {p5:.1f}%  （最悪ケース）")
        print(f"  回収率25%ile  : {p25:.1f}%")
        print(f"  回収率 中央値 : {p50:.1f}%")
        print(f"  回収率75%ile  : {p75:.1f}%")
        print(f"  回収率95%ile  : {p95:.1f}%  （最良ケース）")
        print(f"  大損確率      : {ruin_prob:.1f}%  （回収率50%未満）")

    return pd.DataFrame(rows)


def plot_monte_carlo(
    df: pd.DataFrame,
    strategies: list[str],
    mc_result: pd.DataFrame,
    save_path: Path,
    n_sim: int = 200,
) -> None:
    """モンテカルロの回収率分布を可視化。"""
    n   = len(strategies)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    np.random.seed(RANDOM_STATE)

    for ax, sid in zip(axes, strategies):
        filtered = apply_strategy(df, sid)
        if filtered.empty:
            continue

        probs   = filtered["推定的中確率"].values
        payouts = filtered.apply(
            lambda r: r["実配当(100円)"] / 100 if r["実配当(100円)"] > 0 else 0,
            axis=1,
        ).values
        amounts  = filtered["購入額"].values
        total_cost = amounts.sum()

        hits_sim = np.random.binomial(1, np.clip(probs, 0, 1), size=(n_sim, len(probs)))
        pay_sim  = hits_sim * (amounts * payouts)
        roi_sim  = pay_sim.sum(axis=1) / total_cost * 100

        actual_roi = filtered["実払戻額"].sum() / total_cost * 100

        ax.hist(roi_sim, bins=40, color="steelblue", alpha=0.7, edgecolor="white")
        ax.axvline(100,        color="gray",   linewidth=1.2, linestyle="--", label="損益分岐(100%)")
        ax.axvline(actual_roi, color="tomato", linewidth=1.5, linestyle="-",  label=f"実績({actual_roi:.1f}%)")
        p5 = np.percentile(roi_sim, 5)
        ax.axvline(p5, color="orange", linewidth=1.2, linestyle=":", label=f"5%ile({p5:.1f}%)")

        ax.set_title(f"{sid}\n{STRATEGIES[sid]}", fontsize=9)
        ax.set_xlabel("回収率（%）")
        ax.set_ylabel("頻度")
        ax.legend(fontsize=7)

    fig.suptitle(f"モンテカルロ回収率分布（{n_sim}試行）", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"グラフ保存: {save_path}")


# =========================================================
# main
# =========================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="PyCaLiAI 検証3点セット")
    parser.add_argument(
        "--strategies", nargs="+",
        default=["S07", "S04", "S01", "S02", "S05", "S12"],
        help="検証する戦略ID（デフォルト: S07 S04 S01 S02 S05 S12）"
    )
    parser.add_argument("--n_sim", type=int, default=N_MONTE)
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # バックテスト結果読み込み
    result_csv = REPORT_DIR / "backtest_results.csv"
    logger.info(f"バックテスト結果読み込み: {result_csv}")
    df = pd.read_csv(result_csv, encoding="utf-8-sig")
    df = add_meta(df)
    logger.info(f"  {len(df):,}行  レース数: {df['race_id'].nunique():,}")

    strategies = [s for s in args.strategies if s in STRATEGIES]
    logger.info(f"検証戦略: {strategies}")

    # =========================================================
    # 1. 時系列過学習チェック
    # =========================================================
    ts_result = timeseries_validation(df, strategies)
    ts_result.to_csv(
        REPORT_DIR / "validation_timeseries.csv",
        index=False, encoding="utf-8-sig"
    )
    plot_timeseries_roi(ts_result, REPORT_DIR / "validation_timeseries.png")

    # =========================================================
    # 2. ドローダウン分析
    # =========================================================
    dd_result = drawdown_analysis(df, strategies)
    dd_result.to_csv(
        REPORT_DIR / "validation_drawdown.csv",
        index=False, encoding="utf-8-sig"
    )
    plot_drawdown(df, strategies[:4], REPORT_DIR / "validation_drawdown.png")

    # =========================================================
    # 3. モンテカルロ
    # =========================================================
    mc_result = monte_carlo(df, strategies, n_sim=args.n_sim)
    mc_result.to_csv(
        REPORT_DIR / "validation_montecarlo.csv",
        index=False, encoding="utf-8-sig"
    )
    plot_monte_carlo(
        df, strategies[:3], mc_result,
        REPORT_DIR / "validation_montecarlo.png",
        n_sim=500,
    )

    # =========================================================
    # 総合評価
    # =========================================================
    print("\n" + "=" * 80)
    print("【総合評価】")
    print("=" * 80)
    print(f"\n{'戦略':<40} {'時系列検証期回収率':>12} {'最大DD':>12} {'MC中央値':>10} {'大損確率':>10}")
    print("-" * 80)

    for sid in strategies:
        # 時系列検証期回収率
        ts_sub  = ts_result[
            (ts_result["戦略ID"] == sid) &
            (ts_result["期間"] == "検証期(2023〜)")
        ]
        ts_roi  = ts_sub["回収率"].values[0] if not ts_sub.empty else 0

        # 最大DD
        dd_sub  = dd_result[dd_result["戦略ID"] == sid]
        max_dd  = dd_sub["最大DD(円)"].values[0] if not dd_sub.empty else 0

        # MC中央値・大損確率
        mc_sub  = mc_result[mc_result["戦略ID"] == sid]
        mc_med  = mc_sub["中央値"].values[0]  if not mc_sub.empty else 0
        mc_ruin = mc_sub["大損確率(%)"].values[0] if not mc_sub.empty else 0

        # 総合判定
        ok_ts   = "✅" if ts_roi  >= 100 else "❌"
        ok_mc   = "✅" if mc_med  >= 100 else "❌"
        ok_ruin = "✅" if mc_ruin <= 10  else "⚠️"

        label = f"{sid}: {STRATEGIES[sid]}"
        print(
            f"{label:<40} {ts_roi:>10.1f}%{ok_ts} "
            f"{max_dd:>+12,}円  {mc_med:>8.1f}%{ok_mc} "
            f"{mc_ruin:>8.1f}%{ok_ruin}"
        )

    print("\n凡例: ✅=合格  ❌=不合格  ⚠️=要注意")
    print("  時系列検証期回収率: 2023〜2024年の未知データで100%超えが合格")
    print("  最大DD: ドローダウン（大きいほど資金が必要）")
    print("  MC中央値: モンテカルロ中央値が100%超えが合格")
    print("  大損確率: 投資額の50%以上を失う確率が10%以下が合格")

    print(f"\n保存先: {REPORT_DIR}")


if __name__ == "__main__":
    main()