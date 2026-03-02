"""
inspect_csv.py
3つのCSVの構造を診断し、アンサンブル設計の判断材料を出力する。

Usage:
    python inspect_csv.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# =========================================================
# ここだけ編集: 3つのCSVパスとニックネーム
# =========================================================
CSV_FILES = {
    "add": "data/add_20130105-20251228.csv",
}

# 共通キー候補（実CSVに合わせて追記してOK）
KEY_CANDIDATES = [
    "race_id", "horse_id", "race_date", "date",
    "レースid", "馬id", "開催日",
    "horse_number", "馬番", "枠番", "frame_number",
]

REPORT_PATH = Path(r"E:\PyCaLiAI\reports\csv_inspection.txt")
# =========================================================


def inspect_one(name: str, path: str) -> dict:
    """1つのCSVを診断してサマリを返す。"""
    logger.info(f"読み込み中: {path}")
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False, nrows=None)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp932", low_memory=False)

    n_rows, n_cols = df.shape

    # ---- 型分類 ----
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    dt_cols  = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

    # ---- 欠損 ----
    missing = df.isnull().mean().sort_values(ascending=False)
    high_missing = missing[missing > 0.1]  # 10%以上欠損

    # ---- 共通キー候補の検出 ----
    found_keys = [c for c in df.columns if c.lower() in [k.lower() for k in KEY_CANDIDATES]]

    # ---- 数値列の基礎統計（上位10列） ----
    num_stats = df[num_cols[:10]].describe().round(3) if num_cols else pd.DataFrame()

    # ---- カテゴリ列のユニーク数 ----
    cat_unique = {c: df[c].nunique() for c in cat_cols[:20]}

    return {
        "name":        name,
        "path":        path,
        "n_rows":      n_rows,
        "n_cols":      n_cols,
        "num_cols":    num_cols,
        "cat_cols":    cat_cols,
        "dt_cols":     dt_cols,
        "found_keys":  found_keys,
        "all_columns": df.columns.tolist(),
        "high_missing":high_missing,
        "num_stats":   num_stats,
        "cat_unique":  cat_unique,
        "df_head":     df.head(3),
    }


def find_common_keys(results: list[dict]) -> list[str]:
    """3つのCSVに共通して存在する列名を返す。"""
    sets = [set(r["all_columns"]) for r in results]
    common = set.intersection(*sets)
    return sorted(common)


def format_report(results: list[dict], common_keys: list[str]) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("CSV診断レポート")
    lines.append("=" * 70)

    # ---- 概要テーブル ----
    lines.append("\n【概要】")
    lines.append(f"{'名前':<12} {'行数':>10} {'列数':>6} {'数値列':>6} {'カテゴリ列':>10} {'日付列':>6}")
    lines.append("-" * 56)
    for r in results:
        lines.append(
            f"{r['name']:<12} {r['n_rows']:>10,} {r['n_cols']:>6} "
            f"{len(r['num_cols']):>6} {len(r['cat_cols']):>10} {len(r['dt_cols']):>6}"
        )

    # ---- 共通キー ----
    lines.append(f"\n【3CSV共通カラム】({len(common_keys)}個)")
    if common_keys:
        for k in common_keys:
            lines.append(f"  - {k}")
    else:
        lines.append("  ※ 共通カラムなし → JOINキーの設計が必要")

    # ---- 各CSVの詳細 ----
    for r in results:
        lines.append(f"\n{'='*70}")
        lines.append(f"【{r['name']}】  {r['path']}")
        lines.append(f"  行数: {r['n_rows']:,}  列数: {r['n_cols']}")

        lines.append(f"\n  キー候補列: {r['found_keys'] if r['found_keys'] else 'なし（要確認）'}")

        lines.append(f"\n  全カラム一覧 ({r['n_cols']}列):")
        # 10列ごとに改行
        cols = r["all_columns"]
        for i in range(0, len(cols), 5):
            lines.append("    " + " | ".join(cols[i:i+5]))

        lines.append(f"\n  数値列 ({len(r['num_cols'])}列): {r['num_cols'][:30]}")
        lines.append(f"  カテゴリ列 ({len(r['cat_cols'])}列): {r['cat_cols'][:30]}")

        if not r["high_missing"].empty:
            lines.append(f"\n  欠損10%超の列:")
            for col, rate in r["high_missing"].items():
                lines.append(f"    {col}: {rate:.1%}")
        else:
            lines.append("\n  欠損10%超の列: なし")

        if r["cat_unique"]:
            lines.append(f"\n  カテゴリ列のユニーク数 (上位20):")
            for col, n in r["cat_unique"].items():
                lines.append(f"    {col}: {n:,}")

        lines.append(f"\n  先頭3行:")
        lines.append(r["df_head"].to_string())

        if not r["num_stats"].empty:
            lines.append(f"\n  数値列基礎統計 (上位10列):")
            lines.append(r["num_stats"].to_string())

    # ---- アンサンブル設計の判断材料まとめ ----
    lines.append(f"\n{'='*70}")
    lines.append("【アンサンブル設計チェックリスト】")
    lines.append(f"  □ 共通JOINキーの確認: {'OK' if common_keys else '要設計'}")

    row_counts = [r["n_rows"] for r in results]
    same_rows = len(set(row_counts)) == 1
    lines.append(f"  □ 行数一致: {'OK' if same_rows else f'不一致 {row_counts} → 要確認'}")

    lines.append("  □ ターゲット列（着順/複勝フラグ）の有無: 目視確認してください")
    lines.append("  □ 日付列の有無（時系列分割用）: 目視確認してください")
    lines.append("=" * 70)

    return "\n".join(lines)


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for name, path in CSV_FILES.items():
        if not Path(path).exists():
            logger.warning(f"ファイルが見つかりません: {path} → スキップ")
            continue
        r = inspect_one(name, path)
        results.append(r)

    if not results:
        logger.error("CSVが1つも読み込めませんでした。CSV_FILESのパスを確認してください。")
        return

    common_keys = find_common_keys(results) if len(results) == 3 else []
    report = format_report(results, common_keys)

    print(report)
    REPORT_PATH.write_text(report, encoding="utf-8")
    logger.info(f"レポート保存完了: {REPORT_PATH}")


if __name__ == "__main__":
    main()