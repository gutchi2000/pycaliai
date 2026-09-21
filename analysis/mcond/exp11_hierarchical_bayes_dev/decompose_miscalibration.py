# -*- coding: utf-8 -*-
"""
decompose_miscalibration.py
=============================
EXP11 Stage1修正3。低頻度騎手×調教師ペア(as-of, pair_prior_count_asof<5)に
おける「v6予測0.059 vs 実現0.036」を、2023年developmentのみで分解する。

並べる値:
  - raw v6 probability (v6_p_win_raw)
  - calibrated v6 probability (v6_p_win_calibrated, pl_calibrators_v6由来)
  - market probability (market_p_win)
  - v6+market baseline probability (v6_market_blend_raw)
  - actual win rate (着順==1)

層別化: 人気帯(market_p_winの五分位)・頭数帯・新馬/未勝利フラグ・
出走回数帯(career_start_idx)・休養日数帯(間隔)。

市場確率も同程度に過大なら「騎手×調教師固有の問題」と呼ばない。
単純なpair_prior_count<5だけで補正できるなら階層ベイズ固有の価値とは
しない(この判定はStage1のモデル比較(M1 vs M4)で行うため、本スクリプトは
記述統計のみ提供する)。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).parent / "out"
COL_FINISH = "着順"


def summarize(df: pd.DataFrame, label: str) -> dict:
    y = (pd.to_numeric(df[COL_FINISH], errors="coerce") == 1).astype(float)
    out = dict(
        label=label, n=len(df),
        raw_v6=round(float(df["v6_p_win_raw"].mean()), 4),
        calibrated_v6=round(float(df["v6_p_win_calibrated"].mean()), 4),
        market=round(float(df["market_p_win"].mean()), 4),
        v6_market_blend=round(float(df["v6_market_blend_raw"].mean()), 4),
        actual_win_rate=round(float(y.mean()), 4),
    )
    out["raw_v6_over_actual_pct"] = round((out["raw_v6"] / out["actual_win_rate"] - 1) * 100, 1) \
        if out["actual_win_rate"] > 0 else None
    out["market_over_actual_pct"] = round((out["market"] / out["actual_win_rate"] - 1) * 100, 1) \
        if out["actual_win_rate"] > 0 else None
    return out


def main():
    df = pd.read_parquet(OUT_DIR / "asof_pair_data.parquet")
    df2023 = df[(df["split"] == "valid") & df["v6_p_win_raw"].notna()].copy()

    lf = df2023[df2023["pair_prior_count_asof"] < 5].copy()
    hf = df2023[df2023["pair_prior_count_asof"] >= 5].copy()

    print("[decompose_miscalibration] 2023年development、as-of低頻度ペア vs 高頻度ペア")
    print(summarize(lf, "low_freq(pair_prior_count_asof<5)"))
    print(summarize(hf, "high_freq(pair_prior_count_asof>=5)"))
    print(summarize(df2023, "全体(2023valid)"))

    print("\n[層別化] low_freqのみ、人気帯(市場確率五分位)別")
    lf = lf.dropna(subset=["market_p_win"]).copy()
    lf["pop_band"] = pd.qcut(lf["market_p_win"], 5, labels=["Q1低人気", "Q2", "Q3", "Q4", "Q5高人気"],
                              duplicates="drop")
    for band, g in lf.groupby("pop_band", observed=True):
        print(f"  {band}: {summarize(g, str(band))}")

    print("\n[層別化] low_freqのみ、頭数帯別")
    lf["field_band"] = pd.cut(lf["出走頭数"], [0, 9, 12, 14, 16, 18],
                               labels=["5-9", "10-12", "13-14", "15-16", "17-18"])
    for band, g in lf.groupby("field_band", observed=True):
        print(f"  {band}: {summarize(g, str(band))}")

    print("\n[層別化] low_freqのみ、新馬/未勝利 vs それ以外")
    lf["is_maiden"] = lf["クラス名"].isin(["新馬", "未勝利"])
    for is_m, g in lf.groupby("is_maiden"):
        print(f"  maiden={is_m}: {summarize(g, f'maiden={is_m}')}")

    print("\n[層別化] low_freqのみ、出走回数帯別(kako5_race_count、0-5走の粗い代理)")
    lf["career_band"] = pd.cut(lf["kako5_race_count"].fillna(-1), [-1.5, -0.5, 1, 2, 3, 4, 5],
                                labels=["欠損/新馬", "0-1", "2", "3", "4", "5"])
    for band, g in lf.groupby("career_band", observed=True):
        print(f"  {band}: {summarize(g, str(band))}")

    print("\n[層別化] low_freqのみ、休養日数帯別(間隔)")
    lf["rest_band"] = pd.cut(lf["間隔"], [-1, 14, 30, 60, 120, 9999],
                              labels=["~2w", "2-4w", "1-2m", "2-4m", "4m+"])
    for band, g in lf.groupby("rest_band", observed=True):
        print(f"  {band}: {summarize(g, str(band))}")

    # 保存
    import json
    results = {
        "low_freq_vs_high_freq": [summarize(lf_raw, n) for lf_raw, n in
                                    [(df2023[df2023["pair_prior_count_asof"] < 5], "low_freq"),
                                     (df2023[df2023["pair_prior_count_asof"] >= 5], "high_freq")]],
    }
    with open(OUT_DIR / "decompose_miscalibration.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[saved] {OUT_DIR / 'decompose_miscalibration.json'}")


if __name__ == "__main__":
    main()
