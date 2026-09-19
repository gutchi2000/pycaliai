# -*- coding: utf-8 -*-
"""
stage_b_sample.py — 感度分析用200レースの固定抽出 (結果ラベルを見ない)
================================================================================
2026-09-20、ユーザー指定の実行順5番目。層化変数: 年・芝ダート・人気帯・
support quintile・競馬場。結果ラベル(top3/win/fin/fpay)は一切参照しない
(historical_state.build_race_state_vectorsの戻り値から層化変数だけを取り出して使う)。

比例配分の層化抽出(5変数の全組み合わせをストラタムとし、母集団比に応じて200件を
配分、最大剰余法で端数調整、各ストラタム内はシード固定の乱数で非復元抽出)。
一度選んだら`out/STAGE_B_SENSITIVITY_SAMPLE.json`へ保存し、以後は再抽出しない
(このファイルが存在すればそれを正とする)。

実行: python -m analysis.mcond.exp06_jev_decision_dev.stage_b_sample
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.exp06_jev_decision_dev import historical_state as HS  # noqa: E402
from analysis.mcond.exp06_jev_decision_dev.ood_support import SupportModel, CONTINUOUS_COLS, CATEGORICAL_COLS  # noqa: E402

HERE = Path(__file__).resolve().parent
# out/配下ではなくexp06直下に置く: これはspec.json同様「一度固定したら再導出しない」
# 監査記録であり、out/の他の生成物(キャッシュ・応答ログ)と違って再現可能な中間生成物
# ではないため、gitで追跡してコミットする(2026-09-20)。
SAMPLE_PATH = HERE / "STAGE_B_SENSITIVITY_SAMPLE.json"
N_SAMPLE = 200
N_QUINTILES = 5
RANDOM_SEED = 20260920  # 固定シード。この日に決定した、以後変更しない


def _support_quintile(support: pd.Series) -> pd.Series:
    """2024-2025年query集合自体の分布に基づく層化用のquintile(support値そのものの
    定義=ood_support.SupportModelは2023年で凍結済み、ここではその出力を層化変数として
    5分位に区切るだけであり、support算出ロジック自体を変更・再fitするものではない)。"""
    return pd.qcut(support, N_QUINTILES, labels=[f"q{i+1}" for i in range(N_QUINTILES)],
                   duplicates="drop")


def proportional_stratified_sample(df: pd.DataFrame, strata_cols: list[str], n: int, seed: int) -> pd.Index:
    """比例配分の層化抽出。各ストラタムの母集団比に応じて配分数を決め、最大剰余法で
    端数調整、ストラタム内は固定シードで非復元抽出する。"""
    rng = np.random.default_rng(seed)
    key = df[strata_cols].astype(str).agg("|".join, axis=1)
    counts = key.value_counts()
    shares = counts / counts.sum() * n
    alloc = np.floor(shares).astype(int)
    remainder = n - alloc.sum()
    if remainder > 0:
        frac = (shares - alloc).sort_values(ascending=False)
        for stratum in frac.index[:remainder]:
            alloc[stratum] += 1
    selected = []
    for stratum, k in alloc.items():
        if k <= 0:
            continue
        idx_in_stratum = df.index[key == stratum]
        k = min(k, len(idx_in_stratum))
        chosen = rng.choice(idx_in_stratum.to_numpy(), size=k, replace=False)
        selected.extend(chosen.tolist())
    return pd.Index(selected)


def build_sample() -> dict:
    df, feature_lists = HS.load_design_and_features()
    preds = HS.fit_oos_safe_predictions(df, feature_lists)
    state = HS.build_race_state_vectors(df, preds)

    fit_cols = CONTINUOUS_COLS + CATEGORICAL_COLS
    d2023 = state[state["year"] == 2023]
    sm = SupportModel().fit(d2023[fit_cols])

    d2425 = state[state["year"].isin([2024, 2025])].copy()
    support = sm.score(d2425[fit_cols])["in_distribution_support"]
    d2425["support_quintile"] = _support_quintile(support).astype(str)

    strata_cols = ["year", "surface", "popularity_band", "support_quintile", "venue"]
    selected_idx = proportional_stratified_sample(d2425, strata_cols, N_SAMPLE, RANDOM_SEED)

    sample_df = d2425.loc[selected_idx]
    return {
        "created_at": "2026-09-20",
        "random_seed": RANDOM_SEED,
        "n_sample": len(sample_df),
        "strata_cols": strata_cols,
        "race_ids": sample_df.index.tolist(),  # rid16 のリスト(結果ラベルは含めない)
        "stratum_composition": {
            "by_year": sample_df["year"].value_counts().to_dict(),
            "by_surface": sample_df["surface"].value_counts().to_dict(),
            "by_popularity_band": sample_df["popularity_band"].value_counts().to_dict(),
            "by_support_quintile": sample_df["support_quintile"].value_counts().to_dict(),
            "by_venue": sample_df["venue"].value_counts().to_dict(),
        },
        "no_result_labels_used": True,
        "note": "結果ラベル(top3/win/fin/fpay)は層化にも抽出にも一切使用していない。"
               "一度抽出したらこのファイルをそのまま使い、再抽出しない。",
    }


def load_or_build_sample() -> dict:
    if SAMPLE_PATH.exists():
        return json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    result = build_sample()
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


if __name__ == "__main__":
    r = load_or_build_sample()
    print(json.dumps({k: v for k, v in r.items() if k != "race_ids"}, ensure_ascii=False, indent=1))
    print(f"n_race_ids={len(r['race_ids'])}")
