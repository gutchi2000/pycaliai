# -*- coding: utf-8 -*-
"""
region_unit_correction.py
===========================
EXP11 Stage1修正1。Stage0で報告した8領域の件数を、unique races /
horse-race rows / positive wins / unique jockeys / unique trainers /
unique jockey×trainer pairsに分解して再報告する。
低頻度騎手×調教師ペアはas-of版(pair_prior_count_asof<5)で再定義する
(Stage0の全期間集計版は使わない)。他7領域はStage0のsparse_region_audit.py
が生成したフラグをそのまま使う(これらはas-of設計の対象外、範囲は
career_start_idx等の既にas-ofな定義)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT_DIR = Path(__file__).parent / "out"
COL_RID = "レースID(新/馬番無)"
COL_JOCKEY = "騎手コード"
COL_TRAINER = "調教師コード"
COL_FINISH = "着順"


def unit_report(df: pd.DataFrame, mask: pd.Series, name: str, year_filter=None) -> dict:
    sub = df[mask]
    if year_filter is not None:
        sub = sub[sub["year"].isin(year_filter)]
    finish_num = pd.to_numeric(sub[COL_FINISH], errors="coerce")
    return dict(
        region=name,
        horse_race_rows=int(len(sub)),
        unique_races=int(sub[COL_RID].nunique()),
        positive_wins=int((finish_num == 1).sum()),
        unique_jockeys=int(sub[COL_JOCKEY].nunique()) if COL_JOCKEY in sub.columns else None,
        unique_trainers=int(sub[COL_TRAINER].nunique()) if COL_TRAINER in sub.columns else None,
        unique_jt_pairs=int(sub.groupby([COL_JOCKEY, COL_TRAINER]).ngroups)
        if COL_JOCKEY in sub.columns and COL_TRAINER in sub.columns else None,
    )


def main():
    print("[region_unit_correction] Stage0の7領域(as-of対象外)を単位訂正...")
    df0 = pd.read_parquet(OUT_DIR / "sparse_region_flags.parquet")
    # 列名がmojibakeで保存されている可能性があるため実際の列名を確認して使う
    cols = list(df0.columns)
    print("  columns:", [c for c in cols if "career" in c or "first_time" in c
                          or "class_up" in c or "stable_transfer" in c or "sire" in c])

    regions_static = {
        "career_0_2": (df0["career_start_idx"] <= 2),
        "career_3_5": (df0["career_start_idx"].between(3, 5)),
        "first_time_distance": df0["is_first_time_distance"],
        "first_time_course": df0["is_first_time_course"],
        "class_up_first_race": df0["is_class_up"].fillna(False),
        "stable_transfer_first_race": df0["is_stable_transfer"].fillna(False),
        "low_sample_sire_condition": df0["is_low_sample_sire_cond"],
    }
    results = []
    for name, mask in regions_static.items():
        r_all = unit_report(df0, mask, name)
        r_2023 = unit_report(df0, mask, name + "_2023valid", year_filter=[2023])
        # 2023valid splitだけに絞る(念のためsplit列も併用)
        sub2023 = df0[mask & (df0["split"] == "valid")]
        r_2023["horse_race_rows"] = int(len(sub2023))
        r_2023["unique_races"] = int(sub2023[COL_RID].nunique())
        finish_num = pd.to_numeric(sub2023[COL_FINISH], errors="coerce")
        r_2023["positive_wins"] = int((finish_num == 1).sum())
        r_2023["unique_jockeys"] = int(sub2023[COL_JOCKEY].nunique())
        r_2023["unique_trainers"] = int(sub2023[COL_TRAINER].nunique())
        r_2023["unique_jt_pairs"] = int(sub2023.groupby([COL_JOCKEY, COL_TRAINER]).ngroups)
        results.append((r_all, r_2023))
        print(f"\n[{name}] 全期間: unique_races={r_all['unique_races']:,} "
              f"horse_race_rows={r_all['horse_race_rows']:,} wins={r_all['positive_wins']:,} "
              f"jockeys={r_all['unique_jockeys']} trainers={r_all['unique_trainers']} "
              f"pairs={r_all['unique_jt_pairs']}")
        print(f"           2023valid: unique_races={r_2023['unique_races']:,} "
              f"horse_race_rows={r_2023['horse_race_rows']:,} wins={r_2023['positive_wins']:,} "
              f"jockeys={r_2023['unique_jockeys']} trainers={r_2023['unique_trainers']} "
              f"pairs={r_2023['unique_jt_pairs']}")

    print("\n[region_unit_correction] 低頻度騎手×調教師ペア(as-of版)を単位訂正...")
    df1 = pd.read_parquet(OUT_DIR / "asof_pair_data.parquet")
    mask_lf = df1["pair_prior_count_asof"] < 5
    r_all = unit_report(df1, mask_lf, "low_freq_jt_pair_asof")
    sub2023 = df1[mask_lf & (df1["split"] == "valid")]
    print(f"[low_freq_jt_pair_asof] 全期間: unique_races={r_all['unique_races']:,} "
          f"horse_race_rows={r_all['horse_race_rows']:,} wins={r_all['positive_wins']:,} "
          f"jockeys={r_all['unique_jockeys']} trainers={r_all['unique_trainers']} "
          f"pairs={r_all['unique_jt_pairs']}")
    print(f"                      2023valid: unique_races={sub2023[COL_RID].nunique():,} "
          f"horse_race_rows={len(sub2023):,} "
          f"jockeys={sub2023[COL_JOCKEY].nunique()} trainers={sub2023[COL_TRAINER].nunique()} "
          f"pairs={sub2023.groupby([COL_JOCKEY,COL_TRAINER]).ngroups:,}")

    import json
    all_results = {name: {"all_years": r_all, "2023valid": r_2023} for (r_all, r_2023), name in
                   zip(results, regions_static.keys())}
    with open(OUT_DIR / "region_unit_correction.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n[saved] {OUT_DIR / 'region_unit_correction.json'}")


if __name__ == "__main__":
    main()
