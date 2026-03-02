"""
kelly.py
PyCaLiAI - Kelly基準による最適賭け金計算 + 実運用チェックリスト

Kelly基準:
  f* = (p * b - q) / b
  p  = 的中確率
  b  = 配当倍率 - 1（純利益倍率）
  q  = 1 - p（外れ確率）

実運用では Half-Kelly（f*/2）を推奨。
フルKellyは理論最適だが分散が大きく破産リスクが高い。

Usage:
    python kelly.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
RESULT_CSV = REPORT_DIR / "backtest_results_2024.csv"


# =========================================================
# Kelly基準計算
# =========================================================
def kelly_fraction(p: float, b: float) -> float:
    """
    Kelly基準の最適賭け比率を返す。

    Args:
        p: 的中確率（0〜1）
        b: 配当倍率 - 1（例：倍率2.0なら b=1.0）

    Returns:
        f*: 資金に対する最適賭け比率（0〜1）
    """
    q = 1 - p
    f = (p * b - q) / b
    return max(f, 0.0)


def kelly_analysis(df: pd.DataFrame, strategy: str = "S12") -> pd.DataFrame:
    """
    戦略の馬券種×クラス単位でKelly比率を計算する。

    Half-Kelly（f*/2）を実運用推奨値として併記する。
    """
    # S12フィルタ
    base = df.copy()
    base["date"] = pd.to_datetime(base["日付"].astype(str), format="%Y%m%d")
    base["曜日"] = base["date"].dt.dayofweek
    base["土日"] = base["曜日"].isin([5, 6])
    rc = (
        base.groupby(["日付", "場所"])["race_id"]
        .nunique()
        .reset_index()
        .rename(columns={"race_id": "R数"})
    )
    base = base.merge(rc, on=["日付", "場所"], how="left")
    base = base[base["土日"] & (base["R数"] >= 10)]

    filtered = base[
        (base["クラス"] == "新馬") &
        (base["馬券種"] == "馬連") &
        (base["場所"].isin(["東京", "中山", "中京", "小倉"]))
    ].copy()

    rows = []
    for (bt, cls, place), grp in filtered.groupby(["馬券種", "クラス", "場所"]):
        p       = grp["的中"].mean()
        hit_grp = grp[grp["的中"] == 1]
        if hit_grp.empty or p <= 0:
            continue
        # 平均配当倍率
        avg_b  = (hit_grp["実配当(100円)"] / 100).mean() - 1
        med_b  = (hit_grp["実配当(100円)"] / 100).median() - 1
        if avg_b <= 0:
            continue
        f_full = kelly_fraction(p, avg_b)
        f_half = f_full / 2

        n_races = grp["race_id"].nunique()
        roi     = grp["実払戻額"].sum() / grp["購入額"].sum() * 100

        rows.append({
            "場所":          place,
            "クラス":        cls,
            "馬券種":        bt,
            "レース数":      n_races,
            "的中率(%)":     round(p * 100, 1),
            "平均配当倍率":   round(avg_b + 1, 2),
            "中央配当倍率":   round(med_b + 1, 2),
            "実回収率(%)":   round(roi, 1),
            "Kelly比率(%)":  round(f_full * 100, 1),
            "HalfKelly(%)":  round(f_half * 100, 1),
            "推奨賭け金(1万資金)": round(f_half * 10000),
        })

    return pd.DataFrame(rows).sort_values("実回収率(%)", ascending=False)


def plot_kelly(df: pd.DataFrame, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左: 場所別回収率 vs Kelly比率
    ax = axes[0]
    ax.scatter(df["的中率(%)"], df["実回収率(%)"], s=80, color="steelblue")
    for _, row in df.iterrows():
        ax.annotate(
            row["場所"],
            (row["的中率(%)"], row["実回収率(%)"]),
            textcoords="offset points", xytext=(5, 5), fontsize=9,
        )
    ax.axhline(100, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("的中率（%）")
    ax.set_ylabel("実回収率（%）")
    ax.set_title("S12: 場所別 的中率 vs 回収率")

    # 右: Half-Kelly推奨賭け金
    ax = axes[1]
    colors = ["tomato" if v >= 100 else "steelblue" for v in df["実回収率(%)"] ]
    ax.barh(df["場所"], df["HalfKelly(%)"], color=colors)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Half-Kelly比率（%）")
    ax.set_title("S12: 場所別 Half-Kelly推奨賭け比率")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"グラフ保存: {save_path}")


# =========================================================
# 実運用チェックリスト
# =========================================================
def print_operation_checklist(kelly_df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("【S12 実運用チェックリスト】")
    print("=" * 70)

    print("""
■ 対象条件
  馬券種  : 馬連
  クラス  : 新馬
  会場    : 東京 / 中山 / 中京 / 小倉
  開催日  : 土日のみ（同日同会場10R以上）

■ 買い方
  ◎本命 → ◯対抗への馬連 1点
  ◯対抗 → ◎本命への馬連 1点（同上、1点で計上）
  ※ 実際は◎-◯の1点のみ購入

■ 賭け金の決め方（Half-Kelly基準）
""")

    for _, row in kelly_df.iterrows():
        print(
            f"  {row['場所']}: 資金の {row['HalfKelly(%)']:.1f}%  "
            f"（10万円資金なら {int(row['HalfKelly(%)']*1000):,}円/R）"
        )

    print("""
■ 資金管理ルール
  推奨初期資金  : 200万円（最大DD-57万の3.5倍）
  1R上限        : 資金の5%以内（Half-Kelly超過時は5%でキャップ）
  撤退ライン    : 資金が初期の50%（100万円）を下回ったら一時停止
  再開基準      : 3ヶ月間の回収率が90%以上で再開

■ 実運用フロー
  1. 毎週土曜・日曜の出走表をCSV取得（netkeiba等）
  2. python predict.py --race_id XXXX で予測実行
  3. ◎-◯の馬連を上記賭け金で購入
  4. 結果をexcel等に記録（的中/払戻/収支）
  5. 月次で回収率を確認 → 90%割れで賭け金を半減

■ 注意事項
  - 新馬戦は出走取消・除外が多い → 当日朝に出走確認必須
  - 馬連は最低100円から → 小口から始めて感覚を掴む
  - 本バックテストは2024年1年間のデータのみ
    → 2013〜2022年の検証結果が出るまでは少額運用を推奨
""")

    print("=" * 70)
    print("【賭け金シミュレーション（初期資金200万円）】")
    print("=" * 70)

    capital   = 2_000_000
    for _, row in kelly_df.iterrows():
        bet = min(
            int(capital * row["HalfKelly(%)"] / 100 / 100) * 100,
            int(capital * 0.05 / 100) * 100,
        )
        print(f"  {row['場所']}: {bet:,}円/R")


# =========================================================
# main
# =========================================================
def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"読み込み: {RESULT_CSV}")
    df = pd.read_csv(RESULT_CSV, encoding="utf-8-sig")
    logger.info(f"  {len(df):,}行")

    # Kelly分析
    logger.info("Kelly基準計算中...")
    kelly_df = kelly_analysis(df)

    print("\n" + "=" * 70)
    print("【S12 Kelly基準分析結果】")
    print("=" * 70)
    print(kelly_df[[
        "場所", "的中率(%)", "平均配当倍率", "実回収率(%)",
        "Kelly比率(%)", "HalfKelly(%)", "推奨賭け金(1万資金)"
    ]].to_string(index=False))

    # グラフ
    plot_kelly(kelly_df, REPORT_DIR / "kelly_s12.png")

    # チェックリスト
    print_operation_checklist(kelly_df)

    # CSV保存
    out = REPORT_DIR / "kelly_s12.csv"
    kelly_df.to_csv(out, index=False, encoding="utf-8-sig")
    logger.info(f"保存: {out}")


if __name__ == "__main__":
    main()