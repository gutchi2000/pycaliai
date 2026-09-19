# -*- coding: utf-8 -*-
"""
stage_b_dry_run.py — Stage B主評価の対象レース数・API呼び出し数・費用見積り (API非実行)
================================================================================
2026-09-20、ユーザー指定の実行順4番目。実際のAPIコールは一切行わない
(win32com/requestsへのアクセスなし、既存のexp05_design.parquetを読むだけ)。

安全上限(ユーザー指定):
  最大新規API call数 = 全適格レース数 + 反復感度分析600回
  最大累積input tokens = 20,000,000
20M input tokens到達時は新規呼び出しを停止し、途中結果を保存する
(実行スクリプト側=stage_b_run.pyで強制する、このdry-runは見積りのみ)。

トークン単価はStage A実測(6問1レースでinput_tokens=1281)を実測ベースラインとして
使う(公式価格は実行時に別途確認・記録すること、このスクリプトはコストの見積り根拠を
明示するだけで確定額を主張しない)。

実行: python -m analysis.mcond.exp06_jev_decision_dev.stage_b_dry_run
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.exp06_jev_decision_dev import historical_state as HS  # noqa: E402
from analysis.mcond.exp06_jev_decision_dev import jev_client as JC  # noqa: E402

HERE = Path(__file__).resolve().parent

# Stage A実測(2026-09-20、6問1レース分、jev-1.13.0): input_tokens=1281, output_tokens=226
STAGE_A_INPUT_TOKENS_PER_RACE = 1281
STAGE_A_OUTPUT_TOKENS_PER_RACE = 226
SENSITIVITY_FIXED_RACES = 200
SENSITIVITY_REPEATS = 3
MAX_INPUT_TOKENS_CAP = 20_000_000
ASSUMED_PRICE_PER_M_INPUT_TOKENS_USD = 0.042  # ユーザー提示の現行価格、実行時に公式値を再確認すること


def count_eligible_races() -> dict:
    df, _ = HS.load_design_and_features()
    out = {}
    for year in (2023, 2024, 2025):
        d = df[df["year"] == year]
        out[str(year)] = int(d["rid16"].nunique())
    return out


def already_cached_count(prompt_schema_hash: str) -> int:
    """out/jev_cache/ に既に存在するキャッシュ件数(dry-run時点、実行前の既存キャッシュ)。"""
    cache_dir = JC.CACHE_DIR
    if not cache_dir.exists():
        return 0
    return sum(1 for _ in cache_dir.glob("*.json"))


def main() -> dict:
    race_counts = count_eligible_races()
    n_2024_25 = race_counts["2024"] + race_counts["2025"]
    n_primary_calls = n_2024_25  # 2023年はsupport reference/development用、Jevへは問い合わせない
    n_sensitivity_calls = SENSITIVITY_FIXED_RACES * SENSITIVITY_REPEATS
    n_total_new_calls = n_primary_calls + n_sensitivity_calls

    est_input_tokens = n_total_new_calls * STAGE_A_INPUT_TOKENS_PER_RACE
    est_output_tokens = n_total_new_calls * STAGE_A_OUTPUT_TOKENS_PER_RACE
    est_cost_usd = est_input_tokens / 1_000_000 * ASSUMED_PRICE_PER_M_INPUT_TOKENS_USD

    already_cached = already_cached_count("exp06_v2_20260920")

    report = {
        "race_counts_by_year": race_counts,
        "primary_evaluation_target_races_2024_2025": n_2024_25,
        "sensitivity_analysis_calls": n_sensitivity_calls,
        "total_new_api_calls_planned": n_total_new_calls,
        "safety_cap_max_new_calls": n_2024_25 + n_sensitivity_calls,  # ユーザー指定の上限式
        "already_cached_entries_in_out_dir": already_cached,
        "estimated_new_calls_after_cache": max(n_total_new_calls - already_cached, 0),
        "estimated_input_tokens": est_input_tokens,
        "estimated_output_tokens": est_output_tokens,
        "safety_cap_max_input_tokens": MAX_INPUT_TOKENS_CAP,
        "within_token_cap": est_input_tokens <= MAX_INPUT_TOKENS_CAP,
        "estimated_cost_usd_at_assumed_price": round(est_cost_usd, 4),
        "assumed_price_per_million_input_tokens_usd": ASSUMED_PRICE_PER_M_INPUT_TOKENS_USD,
        "price_note": "ユーザー提示の現行価格を仮定として使用。実行前に公式価格ページで再確認し、"
                      "stage_b_run.py実行ログに実測値を記録すること。",
        "token_baseline_note": f"Stage A実測({STAGE_A_INPUT_TOKENS_PER_RACE} input / "
                               f"{STAGE_A_OUTPUT_TOKENS_PER_RACE} output tokens、6問1レース、"
                               "jev-1.13.0)をレース単位の単価として外挿。実際のstateは"
                               "レースごとに長さが変わるため上下しうる。",
    }
    return report


if __name__ == "__main__":
    result = main()
    (HERE / "out").mkdir(parents=True, exist_ok=True)
    (HERE / "out" / "STAGE_B_DRY_RUN.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))
