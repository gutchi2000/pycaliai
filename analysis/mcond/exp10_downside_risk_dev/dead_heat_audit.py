# -*- coding: utf-8 -*-
"""
dead_heat_audit.py
===================
EXP10 Stage 1修正3。同着・除外件数の内訳を、EXP09と比較可能な形で分解する。

カテゴリ(ユーザー指定):
  1. 1着同着(win dead heat)         — EXP09の定義(fin==1が複数)と同一
  2. 2着以下を含む任意順位の同着     — EXP10の主解析除外基準(PLは全順序前提)
  3. 着順番号の欠番                 — 例: 1,2,3,4,6,7(5が欠番)
  4. 中止・失格等の非数値着順        — master_v2は既にdropna済みのはず(Stage0確認済み)
  5. 同一race_id内の重複行           — (レースID,馬番)キーの重複
  6. その他(全馬の厳密な順位が作れない理由)

母集団: eligible_races.build_target_rows([2023])のrace_horches(=master_v2由来、
既にdropna済みの生存馬のみ)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eligible_races import build_target_rows  # noqa: E402


def audit_dead_heats(years: list[int]) -> tuple[pd.DataFrame, dict]:
    targets, race_horses, meta = build_target_rows(years)

    cat_win_dead_heat = 0
    cat_any_dead_heat = 0
    cat_missing_finish_number = 0
    cat_non_numeric_finish = 0
    cat_duplicate_row = 0
    cat_other = 0
    n_races = 0
    detail_rows = []

    for rid, g in race_horches_iter(race_horses):
        n_races += 1
        finish = g["finish"].values.astype(float)
        ban = g["ban"].values

        non_numeric = np.isnan(finish)
        if non_numeric.any():
            cat_non_numeric_finish += 1

        dup_ban = pd.Series(ban).duplicated().any()
        if dup_ban:
            cat_duplicate_row += 1

        vals = finish[~non_numeric]
        is_win_dead_heat = False
        is_any_dead_heat = False
        is_missing_number = False
        if len(vals) > 0:
            u, c = np.unique(vals, return_counts=True)
            if (c > 1).any():
                is_any_dead_heat = True
                cat_any_dead_heat += 1
                if (vals == 1.0).sum() > 1:
                    is_win_dead_heat = True
                    cat_win_dead_heat += 1
            # 欠番検査: 観測されたユニーク着順が 1..len(unique) の連番になっているか
            expected = set(range(1, len(u) + 1))
            observed = set(int(x) for x in u)
            if observed != expected and not is_any_dead_heat:
                is_missing_number = True
                cat_missing_finish_number += 1

        blocked = is_any_dead_heat or non_numeric.any() or dup_ban or is_missing_number
        if not blocked:
            pass  # 正常(主解析対象)
        detail_rows.append(dict(
            rid=rid, n=len(g), is_win_dead_heat=is_win_dead_heat,
            is_any_dead_heat=is_any_dead_heat, is_missing_number=is_missing_number,
            has_non_numeric=bool(non_numeric.any()), has_dup_row=bool(dup_ban),
        ))

    detail = pd.DataFrame(detail_rows)
    summary = dict(
        n_races_in_population=n_races,
        cat_1_win_dead_heat=cat_win_dead_heat,
        cat_2_any_position_dead_heat=cat_any_dead_heat,
        cat_3_missing_finish_number=cat_missing_finish_number,
        cat_4_non_numeric_finish=cat_non_numeric_finish,
        cat_5_duplicate_row=cat_duplicate_row,
    )
    return detail, summary


def race_horches_iter(race_horses: pd.DataFrame):
    for rid, g in race_horses.groupby("rid", sort=False):
        yield rid, g.sort_values("pos_in_group").reset_index(drop=True)


if __name__ == "__main__":
    detail, summary = audit_dead_heats([2023])
    print("[2023 同着・除外内訳、EXP09との比較用カテゴリ分解]")
    for k, v in summary.items():
        print(f"  {k}: {v:,}")
    print(f"\n  参考: EXP10主解析の除外基準(2. 任意順位の同着)={summary['cat_2_any_position_dead_heat']:,}件")
    print(f"  参考: EXP09の除外基準(1. 1着のみの同着)={summary['cat_1_win_dead_heat']:,}件")
    detail.to_parquet(Path(__file__).parent / "out" / "dead_heat_detail_2023.parquet")
