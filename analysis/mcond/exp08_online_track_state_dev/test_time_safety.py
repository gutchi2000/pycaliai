# -*- coding: utf-8 -*-
"""
test_time_safety.py — 時点安全性の直接検証。
2026-09-20夜のユーザー訂正指示に基づく:
  1. 期待値モデルの削除不変性(未来年を削除しても過去年の期待値・残差が変わらない)
  2. 利用可能時刻(prior_result_available_ts_primary/_sensitivity)の計算が正しいこと
  3. 「先行レース利用可能時刻 <= 対象レース判断時刻」という行単位の不変条件を
     検査するユーティリティ(online_state.pyがStage3で使う、ここでは違反検出の
     正しさ自体をテストする)
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.exp08_online_track_state_dev.expected_time_model import (  # noqa: E402
    fit_expanding_lookups, predict_expected_time_expanding, FROZEN_CUTOFF_YEAR,
)


def _synthetic_df(years, n_per_year=50, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for y in years:
        for i in range(n_per_year):
            rows.append({
                "year": y, "dist_bucket": 1600, "場所": "06", "芝・ダ": "芝",
                "コース区分": "A", "馬場状態": "良", "クラス名": "1勝",
                "time_sec": 95.0 + 0.1 * (y - years[0]) + rng.normal(0, 1.0),
            })
    return pd.DataFrame(rows)


def test_deletion_invariance_past_years_unaffected_by_removing_future_year():
    """未来年(2025)を削除しても、過去年(2020-2024)の期待値・残差が変わらないこと。
    expanding-window設計(年Yの行はYより前のデータだけでfit)なら自動的に成立する
    はずだが、実装の取り違え(例: 全期間で一度だけfitする旧設計への先祖返り)を
    検出するための直接テスト。"""
    years_full = list(range(2018, 2026))  # 2018-2025
    df_full = _synthetic_df(years_full, n_per_year=40, seed=1)
    lookups_full = fit_expanding_lookups(df_full, value_col="time_sec")
    pred_full, _ = predict_expected_time_expanding(lookups_full, df_full, value_col="time_sec")

    years_no2025 = list(range(2018, 2025))  # 2018-2024のみ(2025を削除)
    df_trunc = _synthetic_df(years_no2025, n_per_year=40, seed=1)  # 同じseed→同じ行(2025以外)
    lookups_trunc = fit_expanding_lookups(df_trunc, value_col="time_sec")
    pred_trunc, _ = predict_expected_time_expanding(lookups_trunc, df_trunc, value_col="time_sec")

    mask_full_pre2025 = df_full["year"] < 2025
    np.testing.assert_allclose(
        pred_full[mask_full_pre2025].to_numpy(), pred_trunc.to_numpy(), rtol=0, atol=1e-9,
        err_msg="2025年を削除すると2018-2024年の期待値が変化した(delete-invariance違反)",
    )


def test_deletion_invariance_removing_2024_does_not_change_2023_or_earlier():
    """development年(2023)より後の年(2024)を消しても2023以前は不変。"""
    years_full = list(range(2020, 2025))  # 2020-2024
    df_full = _synthetic_df(years_full, n_per_year=40, seed=2)
    lookups_full = fit_expanding_lookups(df_full, value_col="time_sec")
    pred_full, _ = predict_expected_time_expanding(lookups_full, df_full, value_col="time_sec")

    years_trunc = list(range(2020, 2024))  # 2020-2023
    df_trunc = _synthetic_df(years_trunc, n_per_year=40, seed=2)
    lookups_trunc = fit_expanding_lookups(df_trunc, value_col="time_sec")
    pred_trunc, _ = predict_expected_time_expanding(lookups_trunc, df_trunc, value_col="time_sec")

    mask = df_full["year"] < 2024
    np.testing.assert_allclose(
        pred_full[mask].to_numpy(), pred_trunc.to_numpy(), rtol=0, atol=1e-9,
    )


def test_2023_2024_2025_share_identical_frozen_model():
    """2023・2024・2025年の観測は同一の(2022年末まででfitした)モデルを使うこと。
    年ごとに別モデルを使っていないかの直接確認(FROZEN_CUTOFF_YEAR以降の設計仕様)。"""
    years = list(range(2020, 2026))
    df = _synthetic_df(years, n_per_year=30, seed=3)
    lookups = fit_expanding_lookups(df, value_col="time_sec")
    assert FROZEN_CUTOFF_YEAR == 2023
    assert lookups[2023] is lookups[2024] is lookups[2025], \
        "2023/2024/2025が同一オブジェクト(同一fit)ではない"
    assert lookups[2023] is not lookups[2022], "2022以前は別モデルであるべき"


def test_no_refit_when_2023_development_rows_change():
    """2023年以降の行の値(仮に2023年の観測値そのもの)を変えても、2022年末までの
    lookupモデルの中身(2022以前のtime_secから作られたセル中央値)は変化しないこと。
    (=2023 developmentの結果を見て期待値モデルを再fitしていないことの直接証拠)"""
    years = list(range(2020, 2025))
    df = _synthetic_df(years, n_per_year=30, seed=4)
    lookups_a = fit_expanding_lookups(df, value_col="time_sec")

    df_perturbed = df.copy()
    mask_2023 = df_perturbed["year"] == 2023
    df_perturbed.loc[mask_2023, "time_sec"] += 1000.0  # 2023年の値を極端に変える
    lookups_b = fit_expanding_lookups(df_perturbed, value_col="time_sec")

    pd.testing.assert_series_equal(
        lookups_a[2023]["lvl_full"].sort_index(), lookups_b[2023]["lvl_full"].sort_index(),
        check_names=False,
    )


# --- 利用可能時刻ルール ---

PRIMARY_DELAY_MIN = 20
SENSITIVITY_DELAY_MIN = 30


def assert_no_availability_violations(
    pairs: pd.DataFrame, avail_col: str, decision_col: str,
) -> None:
    """先行レース利用可能時刻 <= 対象レース判断時刻、という行単位の不変条件を検査する。
    online_state.pyがStage3の状態更新ループで使う想定のユーティリティ。1件でも
    違反があればAssertionErrorを送出する(Gate 0 FAIL相当)。"""
    violations = pairs[pairs[avail_col] > pairs[decision_col]]
    assert len(violations) == 0, (
        f"利用可能時刻の不変条件違反: {len(violations)}件 "
        f"(先行レースの利用可能時刻が対象レースの判断時刻より後)"
    )


def test_availability_checker_detects_violation():
    """チェッカー自体が違反を正しく検出できることの確認(偽陰性がないか)。"""
    pairs = pd.DataFrame({
        "avail": pd.to_datetime(["2024-01-06 10:30", "2024-01-06 11:05"]),
        "decision": pd.to_datetime(["2024-01-06 11:00", "2024-01-06 11:00"]),
    })
    with pytest.raises(AssertionError):
        assert_no_availability_violations(pairs, "avail", "decision")


def test_availability_checker_passes_when_all_valid():
    pairs = pd.DataFrame({
        "avail": pd.to_datetime(["2024-01-06 10:30", "2024-01-06 10:55"]),
        "decision": pd.to_datetime(["2024-01-06 11:00", "2024-01-06 11:00"]),
    })
    assert_no_availability_violations(pairs, "avail", "decision")  # 例外が出なければOK


def test_primary_and_sensitivity_delay_are_20_and_30_minutes():
    """build_observations.pyの定数が仕様書通り20分/30分であることの固定。"""
    from analysis.mcond.exp08_online_track_state_dev.build_observations import (
        PRIMARY_DELAY_MIN as P, SENSITIVITY_DELAY_MIN as S,
    )
    assert P == 20
    assert S == 30
