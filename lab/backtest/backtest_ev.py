"""
backtest_ev.py
PyCaLiAI - 期待値スコア閾値バックテスト

複勝確率 × 単勝オッズ = 期待値スコア（EV）を計算し、
「EVが高い馬だけ複勝を買う」戦略の閾値ごとのROIを分析する。

Usage:
    python backtest_ev.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    plt.rcParams["font.family"] = "MS Gothic"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR  = Path(r"E:\PyCaLiAI")
KEKKA_DIR = BASE_DIR / "data" / "kekka"
PRED_DIR  = BASE_DIR / "reports"
REPORT_DIR = BASE_DIR / "reports"

BUDGET = 10_000


# =========================================================
# データ読み込み
# =========================================================
def load_kekka() -> pd.DataFrame:
    files = sorted(KEKKA_DIR.glob("????????.csv"))
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f, encoding="cp932"))
        except Exception as e:
            logger.warning(f"kekkaスキップ {f.name}: {e}")
    kekka = pd.concat(dfs, ignore_index=True)
    kekka["レースキー"] = kekka["レースID(新)"].astype(str).str.zfill(18).str[:16]
    kekka["馬番_k"]     = kekka["レースID(新)"].astype(str).str.zfill(18).str[-2:].astype(int)
    # 着順列名: "確定着順" or "着順" のどちらかに対応
    order_col = "確定着順" if "確定着順" in kekka.columns else "着順"
    kekka["着順_n"]     = pd.to_numeric(kekka[order_col], errors="coerce")
    # 複勝配当: 括弧付き（単勝のみ表示）を除外
    def _parse_fuku(val):
        s = str(val).strip()
        if s.startswith("(") or s in ("nan", "", "None"):
            return None
        try: return float(s)
        except: return None
    kekka["複勝配当_n"] = kekka["複勝配当"].apply(_parse_fuku)
    return kekka


def load_pred() -> pd.DataFrame:
    files = sorted(PRED_DIR.glob("pred_????????.csv"))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding="utf-8-sig")
            dfs.append(df)
        except Exception:
            try:
                dfs.append(pd.read_csv(f, encoding="cp932"))
            except Exception as e:
                logger.warning(f"predスキップ {f.name}: {e}")
    pred = pd.concat(dfs, ignore_index=True)
    pred["レースキー"] = pred["レースID"].astype(str).str.zfill(16)
    pred["馬番_k"]     = pred["馬番"].astype(int)

    # 期待値スコアが含まれていない旧フォーマットをスキップ
    if "期待値スコア" not in pred.columns:
        logger.warning("期待値スコア列なし: predict_weekly.pyを再実行してください")
        pred["期待値スコア"] = 0.0
    if "単勝オッズ" not in pred.columns:
        pred["単勝オッズ"] = float("nan")

    pred["スコア"]     = pd.to_numeric(pred["スコア"],     errors="coerce").fillna(0)
    pred["期待値スコア"] = pd.to_numeric(pred["期待値スコア"], errors="coerce").fillna(0)
    pred["単勝オッズ"] = pd.to_numeric(pred["単勝オッズ"], errors="coerce")
    return pred


# =========================================================
# メイン
# =========================================================
def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("データ読み込み中...")
    kekka = load_kekka()
    pred  = load_pred()

    # 結合
    merged = pred.merge(
        kekka[["レースキー", "馬番_k", "着順_n", "複勝配当_n"]],
        on=["レースキー", "馬番_k"],
        how="left",
    )
    # 期待値スコアがある行のみ（旧フォーマットを除外）
    merged = merged[merged["期待値スコア"] > 0].copy()
    logger.info(f"結合後: {len(merged):,}行 / {merged['レースキー'].nunique()}レース")

    # 着内フラグ（複勝的中 = 1〜3着）
    merged["fuku_hit"] = (merged["着順_n"] <= 3).astype(int)

    # =========================================================
    # EV閾値ごとのROI分析（複勝◎）
    # =========================================================
    # ◎（最高スコア馬）のみに絞る
    hon = merged[merged["印"] == "◎"].copy()
    hon = hon[hon["LALO_戦略対象"] == "✅"].copy()
    logger.info(f"◎対象行: {len(hon):,}頭 / {hon['レースキー'].nunique()}レース")

    thresholds = np.arange(0, 25, 0.5)
    results = []
    for thr in thresholds:
        subset = hon[hon["期待値スコア"] >= thr]
        if len(subset) == 0:
            continue
        bets   = len(subset)
        invest = bets * BUDGET
        hits   = subset[subset["fuku_hit"] == 1]
        return_ = (hits["複勝配当_n"].fillna(0) / 100 * BUDGET).sum()
        profit = return_ - invest
        roi    = (return_ / invest * 100) if invest > 0 else 0.0
        hit_rate = hits["fuku_hit"].mean() if len(subset) > 0 else 0.0
        results.append({
            "EV閾値": thr,
            "買い数": bets,
            "投資額": invest,
            "回収額": return_,
            "損益": profit,
            "ROI(%)": roi,
            "複勝率(%)": hit_rate * 100,
        })

    df_res = pd.DataFrame(results)
    logger.info("\n--- EV閾値別ROI（複勝◎、LALO対象レース） ---")
    print(df_res[df_res["買い数"] >= 10].to_string(index=False, float_format="%.1f"))

    # 最適閾値（ROI最大 かつ 買い数>=20）
    valid = df_res[df_res["買い数"] >= 20]
    if not valid.empty:
        best = valid.loc[valid["ROI(%)"].idxmax()]
        logger.info(f"\n最適閾値 (買い数≥20): EV≥{best['EV閾値']:.1f}  ROI={best['ROI(%)']:.1f}%  買い数={int(best['買い数'])}")

    # =========================================================
    # 全馬対象（◎問わず）のEVバックテスト
    # =========================================================
    logger.info("\n--- 全馬 EV閾値別複勝ROI（スコア>30限定） ---")
    high_score = merged[merged["スコア"] >= 30].copy()
    results2 = []
    for thr in thresholds:
        subset = high_score[high_score["期待値スコア"] >= thr]
        if len(subset) < 5:
            continue
        bets   = len(subset)
        invest = bets * BUDGET
        hits   = subset[subset["fuku_hit"] == 1]
        return_ = (hits["複勝配当_n"].fillna(0) / 100 * BUDGET).sum()
        profit = return_ - invest
        roi    = (return_ / invest * 100) if invest > 0 else 0.0
        results2.append({
            "EV閾値": thr, "買い数": bets,
            "ROI(%)": roi, "複勝率(%)": hits["fuku_hit"].mean()*100,
        })

    df_res2 = pd.DataFrame(results2)
    if not df_res2.empty:
        print(df_res2[df_res2["買い数"] >= 5].to_string(index=False, float_format="%.1f"))

    # =========================================================
    # グラフ: ROI vs EV閾値
    # =========================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    df_plot = df_res[df_res["買い数"] >= 5]
    ax1.plot(df_plot["EV閾値"], df_plot["ROI(%)"], "b-o", markersize=4)
    ax1.axhline(80, color="gray", linestyle="--", alpha=0.5, label="80%ライン")
    ax1.axhline(100, color="red", linestyle="--", alpha=0.5, label="100%（収支均衡）")
    ax1.set_xlabel("EV閾値")
    ax1.set_ylabel("ROI (%)")
    ax1.set_title("EV閾値 vs ROI（複勝◎）")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(df_plot["EV閾値"], df_plot["買い数"], "g-o", markersize=4)
    ax2.set_xlabel("EV閾値")
    ax2.set_ylabel("買い数（レース数）")
    ax2.set_title("EV閾値 vs 買い数")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_path = REPORT_DIR / "backtest_ev_threshold.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"\nグラフ保存: {out_path}")

    # CSV保存
    df_res.to_csv(REPORT_DIR / "backtest_ev_results.csv", index=False, encoding="utf-8-sig")
    logger.info(f"結果CSV保存: {REPORT_DIR / 'backtest_ev_results.csv'}")


if __name__ == "__main__":
    main()
