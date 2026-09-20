# -*- coding: utf-8 -*-
"""
stage2a_dry_run.py — Stage 2Aドライラン(仕様書追加指示4、2026-09-20夜)

結果を見る前にStage 2Aの全パラメータを固定する。**2024・2025年の結果列
(fin/top3/win/fpay等)は一切読まない** — exp05_design.parquetから読む列を
rid16/year/ban/n_field/dateのみに限定することでこれを構造的に保証する
(読み込み時点でusecols/columns引数により結果列自体をメモリへロードしない)。

対象券種: 単勝・複勝(Gate J1で確定、CALIBRATION_AUDIT.md参照)。

実行: python -m analysis.mcond.exp07_robust_portfolio_dev.stage2a_dry_run
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))

HERE = Path(__file__).resolve().parent

# 結果列を一切含まない列リスト(構造的な安全策、意図的に fin/top3/win/fpay を含めない)
_NON_RESULT_COLS = ["rid16", "ban", "year", "date", "n_field"]


def count_races_and_candidates_per_year() -> dict:
    """2024・2025年のレース数・頭数分布のみを求める。結果列は読み込み列リストに
    含めていないため、この関数はそもそも結果へアクセスできない(usecolsで制限)。"""
    df = pd.read_parquet(
        BASE / "data" / "_research" / "mcond" / "exp05_design.parquet",
        columns=_NON_RESULT_COLS,
    )
    assert set(df.columns) == set(_NON_RESULT_COLS), (
        "結果列が誤って読み込まれていないことの構造的確認"
    )
    out = {}
    for year in (2024, 2025):
        d = df[df["year"] == year]
        races = d.drop_duplicates("rid16")
        out[str(year)] = {
            "n_races": int(races["rid16"].nunique()),
            "n_field_mean": float(races["n_field"].mean()),
            "n_field_min": int(races["n_field"].min()),
            "n_field_max": int(races["n_field"].max()),
            "n_field_lt_3_count": int((races["n_field"] < 3).sum()),
        }
    return out


# =============================================================================
# Stage 2A固定パラメータ(結果を見る前に確定、2026-09-20夜)
# =============================================================================
STAGE2A_FIXED_PARAMS = {
    "ticket_types": ["tansho", "fukusho"],
    "ticket_types_excluded_note": "馬連はGate J1(時系列安全版)でFAILのため除外(CALIBRATION_AUDIT.md§4)。ワイド・馬単はhistorical_pre_snapshotデータ無しのため対象外。",

    "candidate_generation": {
        "method": "各レースで較正済みtansho確率(pl_calibrators_v6.calibrators['tansho'])"
                 "上位3頭を選び、その3頭それぞれについて単勝候補1枚・複勝候補1枚を"
                 "生成する(最大6候補/レース)。候補選定自体は確率のみに基づき、"
                 "2023 development設計時点で決定・結果を見て変更しない。",
        "max_candidates_per_race": 6,
        "top_n_horses": 3,
        "candidate_naming": "tansho:{ban}, fukusho:{ban}",
    },

    "budget": {
        "budget_yen_per_race": 1000,
        "unit_yen": 100,
        "bankroll_yen": 100_000,
        "note": "本番の1R=1万円目安(CLAUDE.md)とは別に、EXP07の配分方式比較専用の"
               "小規模予算を採用(全政策で同額、Stage2A主比較の要件)。予算額自体が"
               "結論(どちらの配分方式が優れるか)を左右しないことは別途頑健性確認する。",
    },

    "policy_params": {
        "P0_NO_BET": {},
        "P1_FLAT": {"note": "候補間均等配分、full_spend_search不要(単純平均)"},
        "P2_CURRENT": {"note": "現行topdown配分を参照のみ(生成しない、既存reports/から読む)"},
        "P3_PROB": {"note": "確率比例配分"},
        "P4_EV": {"note": "点推定EV最大の1候補へ全額"},
        "P5_CVAR": {
            "spec_cvar_alpha": 0.10, "cvar_penalty": 0.0,
            "state_probability_scenarios": "base_probabilities のみ(不確実性シナリオなし)",
        },
        "P6_ROBUST_CVAR": {
            "spec_cvar_alpha": 0.10,
            "cvar_penalty_lambda": 1.0,
            "cvar_penalty_lambda_note": "リスク回避度λ=1.0を採用。0(CVaR制約のみ考慮せず"
                                        "期待値最大化と同義)と過大値(1レース予算のごく"
                                        "一部しか賭けなくなる)の中間的な値として、2023 "
                                        "developmentでのポートフォリオが極端な倍率に"
                                        "張らないことを確認した上で固定。2024/2025年の"
                                        "結果を見て調整しない。",
            "uncertainty_draws": 200,
            "uncertainty_score_sigma": 0.2512877264129069,
            "uncertainty_odds_shrinkage_p10": 0.7743229547222288,
            "random_seed": 20260920,
        },
    },

    "same_horse_exposure_cap": {
        "cap_fraction_of_budget": 0.6,
        "cap_yen_per_race": 600,
        "note": "1000円予算の60%(600円)を1頭あたりの上限とする。単勝・複勝が同一馬を"
               "指すケースが多いため、全額が1頭に集中することを防ぐ目的。",
    },

    "solver_limits": {
        "full_spend_search_max_portfolios": 200_000,
        "expected_grid_size_note": "6候補・budget_units=10のとき、非負整数解の理論上限は"
                                   "comb(6+10-1,5)=3,003通り。200,000の上限に対し十分小さく"
                                   "組合せ爆発の懸念はない。",
        "robust_cvar_portfolio_max_exposure_rebalance_rounds": 20,
        "wall_clock_timeout_note": "既存ソルバー(optimise_portfolio/full_spend_search)は"
                                   "壁時計タイムアウトを持たない設計(全列挙+max_portfolios"
                                   "による組合せ数上限のみ)。候補数6・グリッド最大3,003件は"
                                   "既存ハードウェアで実測ミリ秒〜数十ミリ秒/レース程度と"
                                   "見込まれるため、追加の時間制限は設けない。Stage 2A実行時に"
                                   "1レースあたりの実測時間をログし、異常な遅延(>5秒/レース)が"
                                   "あれば個別調査する。",
    },

    "anomaly_fail_conditions": [
        "Stage2ABudgetAnomaly(候補なし/state_payoffs欠如/グリッド上限超過/full-spend解なし)",
        "robust_cvar_portfolioがValueError/RuntimeErrorを吸収してno-betを返した場合"
        "(ソルバー異常の安全動作として許容するが、そのレースは異常件数に計上しFAIL判定へ含める。"
        "黙って母数から除外しない)",
        "Gate J0再検証(不確実性scenario構築時)が1件でも失敗した場合",
        "候補生成時にcalibrated probabilityがNaN/範囲外になった場合",
    ],
}


def main() -> dict:
    race_counts = count_races_and_candidates_per_year()
    report = {
        "dry_run_at": "2026-09-20夜",
        "result_columns_read": False,
        "result_columns_read_note": "exp05_design.parquetの読み込み列をrid16/ban/year/date/"
                                    "n_fieldのみに限定(pandas read_parquet columns引数)。"
                                    "fin/top3/win/fpay等の結果列はメモリへ一切ロードしていない"
                                    "(コード上構造的に不可能、assertで確認)。",
        "race_counts_2024_2025": race_counts,
        "fixed_params": STAGE2A_FIXED_PARAMS,
    }
    return report


if __name__ == "__main__":
    result = main()
    out_path = HERE / "out" / "STAGE2A_DRY_RUN.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[stage2a_dry_run] wrote {out_path}")
    print(f"[stage2a_dry_run] result_columns_read={result['result_columns_read']}")
    print(f"[stage2a_dry_run] race_counts={result['race_counts_2024_2025']}")
