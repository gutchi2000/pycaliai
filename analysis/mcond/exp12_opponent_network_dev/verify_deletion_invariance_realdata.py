# -*- coding: utf-8 -*-
"""
verify_deletion_invariance_realdata.py
========================================
EXP12 Stage1。2025年データを削除しても、2024年以前の全O3特徴(O3b/EXP02方式)
が一致することを実データで確認する。合成テストに加えた実データでの直接
検証(ユーザー指定)。

2024年の性能・ROIは計算しない(特徴値の一致確認のみ、正解ラベルや評価指標は
一切見ない)。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from opponent_graph import build_opponent_features

FEATURE_COLS = [
    "unique_opponent_count", "opponent_current_strength_mean",
    "opponent_current_strength_max", "opponent_current_strength_top3_mean",
    "beaten_opponent_strength_mean", "lost_to_opponent_strength_mean",
    "strongest_beaten_opponent", "recency_weighted_opponent_strength",
    "opponent_strength_dispersion", "opponent_later_proved_strong_count",
    "external_history_gap",
]


def main():
    print("[verify_deletion_invariance] with 2025 (max_build_year=2025, output=2024)...")
    with_2025 = build_opponent_features(mode="o3b_exp02", output_years=[2024], max_build_year=2025)
    print(f"  {len(with_2025):,}行")

    print("[verify_deletion_invariance] without 2025 (max_build_year=2024, output=2024)...")
    without_2025 = build_opponent_features(mode="o3b_exp02", output_years=[2024], max_build_year=2024)
    print(f"  {len(without_2025):,}行")

    merged = with_2025.merge(without_2025, on=["rid16", "hid", "ban"], suffixes=("_with2025", "_without2025"))
    print(f"  結合後 {len(merged):,}行(両方に存在する行のみ)")
    assert len(merged) == len(with_2025) == len(without_2025), \
        "行数不一致: 2025年削除で2024年の行自体が変わってしまっている(バグの疑い)"

    all_match = True
    mismatches = {}
    for col in FEATURE_COLS:
        a = merged[f"{col}_with2025"]
        b = merged[f"{col}_without2025"]
        both_nan = a.isna() & b.isna()
        diff = (~both_nan) & (a != b)
        # 浮動小数点はapprox比較
        if a.dtype.kind == "f":
            diff = (~both_nan) & (~np.isclose(a.fillna(-999999), b.fillna(-999999), equal_nan=False))
            diff = diff & ~both_nan
        n_mismatch = int(diff.sum())
        mismatches[col] = n_mismatch
        if n_mismatch > 0:
            all_match = False

    print("\n[結果] 列ごとの不一致件数:")
    for col, n in mismatches.items():
        print(f"  {col}: {n}")
    print(f"\n全列一致: {all_match}")
    if not all_match:
        raise AssertionError("実データでの削除不変性検証に失敗。opponent_graph.pyにバグの疑い。")

    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(exist_ok=True)
    import json
    with open(out_dir / "deletion_invariance_realdata.json", "w", encoding="utf-8") as f:
        json.dump(dict(n_rows=len(merged), mismatches=mismatches, all_match=all_match), f,
                   ensure_ascii=False, indent=2)
    print(f"[saved] {out_dir / 'deletion_invariance_realdata.json'}")


if __name__ == "__main__":
    main()
