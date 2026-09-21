# -*- coding: utf-8 -*-
"""
Gate 0D再評価（EXP13再開） — full-starter(historical_pre_snapshot結合済み)
母集団でのtrain/selection/development陽性数・EPV・検出力（読み取り専用）
==========================================================================
2026-09-22、ユーザー指示によるEXP13 Gate 0B/0D再開作業の一部。
train=2013-2021, selection=2022, development=2023 の各期間で、
market snapshot(historical_pre_snapshot)が結合できる complete-case 母集団の
陽性数を再集計する。S4の特徴数(13、事前固定)は結果を見て削らない。

実行: venv311\\Scripts\\python.exe -m analysis.mcond.exp13_nonfinish_risk_dev.gate0d_full_starter_power
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(BASE))
from analysis.market_provenance_audit import (  # noqa: E402
    load_kekka_labels, load_tanpuk_interim, load_post_times, _to_ts,
    bucket_minutes,
)

OUT_DIR = Path(__file__).resolve().parent / "out"

S4_FEATURE_COUNT = 13  # GATE0_REPORT.md記載のS4想定特徴数、事前固定・変更しない


def per_period_stats(year_min: int, year_max: int) -> dict:
    kk = load_kekka_labels(year_min, year_max)
    kk = kk[kk["started"]].copy()

    interim = load_tanpuk_interim(year_min, year_max).reset_index(drop=True)
    interim["snap_ts"] = _to_ts(interim["snap_mmddhhmm"].to_numpy(),
                                 interim["year"].to_numpy()).to_numpy()
    post = load_post_times()
    interim = interim.merge(post, on="rid16", how="left")
    interim["minutes_to_post"] = (interim["post_ts"] - interim["snap_ts"]).dt.total_seconds() / 60.0

    full = kk.merge(
        interim[["rid16", "ban", "minutes_to_post"]],
        left_on=["rid16", "ban_i"], right_on=["rid16", "ban"], how="left",
    )
    full["bucket"] = full["minutes_to_post"].apply(bucket_minutes)
    full["has_snapshot"] = full["bucket"] == "w26_30_historical_pre_snapshot"

    n_started = len(full)
    n_positive = int(full["is_dnf"].sum())
    n_started_cc = int(full["has_snapshot"].sum())
    n_positive_cc = int((full["is_dnf"] & full["has_snapshot"]).sum())

    return {
        "year_range": f"{year_min}-{year_max}",
        "n_started_all_starter": n_started,
        "n_positive_all_starter": n_positive,
        "positive_rate_all_starter_pct": round(n_positive / n_started * 100, 4) if n_started else None,
        "n_started_complete_case": n_started_cc,
        "n_positive_complete_case": n_positive_cc,
        "positive_rate_complete_case_pct": round(n_positive_cc / n_started_cc * 100, 4) if n_started_cc else None,
        "n_positive_lost_to_missingness": n_positive - n_positive_cc,
        "epv_s4_all_starter": round(n_positive / S4_FEATURE_COUNT, 2) if n_positive else None,
        "epv_s4_complete_case": round(n_positive_cc / S4_FEATURE_COUNT, 2) if n_positive_cc else None,
    }


def main() -> int:
    train = per_period_stats(2013, 2021)
    selection = per_period_stats(2022, 2022)
    development = per_period_stats(2023, 2023)

    result = {
        "s4_feature_count_fixed": S4_FEATURE_COUNT,
        "train_2013_2021": train,
        "selection_2022": selection,
        "development_2023": development,
        "power_note": (
            "S4対S2のlogloss/Brier差の効果量に関する事前情報(パイロットデータ)が"
            "存在しないため、厳密な検出力計算はできない(GATE0_REPORT.md Gate0Dの"
            "既存結論のまま、恣意的な仮効果量を捏造して精度を偽装しない)。"
            "EPVヒューリスティック(Peduzzi et al. 1996)のみを判定材料として使う。"
        ),
        "gate0d_verdict_input": {
            "epv_s4_complete_case_2023": development["epv_s4_complete_case"],
            "epv_note": "目安下限(10)を超えるかどうかがGate0D判定の主要指標。理想は20。",
            "missingness_caused_positive_loss_2023": development["n_positive_lost_to_missingness"],
        },
    }

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "gate0d_full_starter_power.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[saved] {out_path}")
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
