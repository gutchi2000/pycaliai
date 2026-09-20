# -*- coding: utf-8 -*-
"""
test_online_state.py — online_state.pyのKalman状態エンジンの単体テスト。
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.exp08_online_track_state_dev.online_state import (  # noqa: E402
    ScalarKalmanState, init_state, run_unit_timeline, run_all_units,
    one_step_ahead_loglik, STATE_DIMS,
)


def _mk_race(rid, date, venue, surface, avail_ts, decision_ts, speed=None, agari=None, pace=None):
    return {
        "rid16": rid, "date": date, "venue": venue, "surface": surface,
        "avail_primary": pd.Timestamp(avail_ts) if avail_ts else pd.NaT,
        "decision_timestamp": pd.Timestamp(decision_ts) if decision_ts else pd.NaT,
        "speed_signal": speed, "agari_signal": agari, "pace_signal": pace,
    }


def test_kalman_update_matches_manual_calculation():
    """Kalman更新式が教科書通りであることを直接確認(単純な1次元例)。"""
    s = ScalarKalmanState(mean=0.0, var=1.0, n_obs=0, last_update_ts=pd.Timestamp("2024-01-06 10:00"))
    s2 = s.update(obs_value=2.0, r=1.0)
    # K = var/(var+r) = 1/(1+1) = 0.5 -> mean = 0 + 0.5*(2-0) = 1.0, var = 0.5*1.0 = 0.5
    assert s2.mean == pytest.approx(1.0)
    assert s2.var == pytest.approx(0.5)
    assert s2.n_obs == 1


def test_predict_grows_variance_with_elapsed_time_and_keeps_mean():
    s = ScalarKalmanState(mean=3.0, var=1.0, n_obs=2, last_update_ts=pd.Timestamp("2024-01-06 10:00"))
    s2 = s.predict(pd.Timestamp("2024-01-06 12:00"), q_per_hour=0.1)  # 2時間経過
    assert s2.mean == pytest.approx(3.0)  # 平均は時間減衰で変わらない
    assert s2.var == pytest.approx(1.0 + 0.1 * 2)


def test_predict_zero_elapsed_time_unchanged():
    s = ScalarKalmanState(mean=1.0, var=2.0, n_obs=0, last_update_ts=pd.Timestamp("2024-01-06 10:00"))
    s2 = s.predict(pd.Timestamp("2024-01-06 10:00"), q_per_hour=0.5)
    assert s2.var == pytest.approx(2.0)


Q = {"speed_signal": 0.05, "agari_signal": 0.05, "pace_signal": 0.05}
R = {"speed_signal": 1.0, "agari_signal": 1.0, "pace_signal": 1.0}
PRIOR_VAR = {"speed_signal": 2.0, "agari_signal": 2.0, "pace_signal": 2.0}


def test_day_reset_new_unit_starts_at_prior():
    """unit(開催日×競馬場×芝ダ)が変わると状態が事前分布(mean=0)にリセットされる
    (日をまたいで状態を繰り越さない)ことの確認。"""
    races = pd.DataFrame([
        _mk_race("r1", "20240106", "05", "芝", "2024-01-06 10:00", "2024-01-06 09:40", speed=3.0),
        _mk_race("r2", "20240106", "05", "芝", "2024-01-06 10:30", "2024-01-06 10:10", speed=3.0),
        # 別日(unit変化) -> 状態は0からやり直すはず
        _mk_race("r3", "20240107", "05", "芝", None, "2024-01-07 09:40"),
    ])
    out = run_all_units(races, "avail_primary", Q, R, PRIOR_VAR)
    out = out.set_index("rid16")
    # r1は先行観測が何もない(自分より前に何もない)ので事前分布のまま
    assert out.loc["r1", "speed_signal_pre_mean"] == pytest.approx(0.0)
    assert out.loc["r1", "speed_signal_pre_var"] == pytest.approx(2.0)
    # r2はr1のspeed=3.0を反映して平均が正へ動くはず
    assert out.loc["r2", "speed_signal_pre_mean"] > 0.0
    # r3は別日unitなので、r1/r2の情報を一切引き継がず事前分布のまま
    assert out.loc["r3", "speed_signal_pre_mean"] == pytest.approx(0.0)
    assert out.loc["r3", "speed_signal_pre_var"] == pytest.approx(2.0)


def test_missing_observation_does_not_move_mean_but_variance_still_evolves():
    """欠損観測(speed=None)のレースは平均を動かさない。時間経過による分散増加は
    そのレースの利用可能時刻を跨いでも(次の実イベントまで)一貫して適用される。
    r1自身には判断イベントを与えず(None)、r1のobsイベントが状態タイムライン上
    最初のイベントになるようにして、期待値の手計算を単純に保つ。"""
    races = pd.DataFrame([
        _mk_race("r1", "20240106", "05", "芝", "2024-01-06 10:00", None, speed=3.0),
        # r_missing: 有効観測なし(speed=None) -> イベントを起こさない(観測としては)
        _mk_race("r_missing", "20240106", "05", "芝", "2024-01-06 10:20", "2024-01-06 10:05"),
        _mk_race("r2", "20240106", "05", "芝", "2024-01-06 10:40", "2024-01-06 10:30", speed=3.0),
    ])
    out = run_all_units(races, "avail_primary", Q, R, PRIOR_VAR).set_index("rid16")
    mean_after_r1 = out.loc["r_missing", "speed_signal_pre_mean"]
    mean_after_r2_pre = out.loc["r2", "speed_signal_pre_mean"]
    # r1のobsイベントが最初のイベント(last_update_ts=None)なのでdt=0、
    # r_missing/r2の判断イベントはpredictのみ(平均不変)でmeanを動かさないはず
    single_obs_state = init_state(2.0).predict(pd.Timestamp("2024-01-06 10:00"), 0.05).update(3.0, 1.0)
    assert mean_after_r2_pre == pytest.approx(single_obs_state.mean, abs=1e-9)
    assert mean_after_r1 == pytest.approx(single_obs_state.mean, abs=1e-9)


def test_no_future_leakage_later_observation_does_not_affect_earlier_snapshot():
    """対象レースの判断時刻より後に利用可能になった観測が、その対象レースの
    スナップショットに混入しないこと(時系列リーク境界の直接確認)。"""
    races = pd.DataFrame([
        _mk_race("r_target", "20240106", "05", "芝", None, "2024-01-06 10:00"),
        # r_lateはr_targetの判断時刻より後に利用可能になる
        _mk_race("r_late", "20240106", "05", "芝", "2024-01-06 10:05", "2024-01-06 10:20", speed=5.0),
    ])
    out = run_all_units(races, "avail_primary", Q, R, PRIOR_VAR).set_index("rid16")
    assert out.loc["r_target", "speed_signal_pre_mean"] == pytest.approx(0.0)
    assert out.loc["r_target", "speed_signal_pre_n_obs"] == 0


def test_deletion_invariance_removing_later_event_does_not_change_earlier_snapshot():
    """未来のイベント(観測 or 判断)を削除しても、それより前のレースのスナップショットが
    変わらないこと。"""
    races_full = pd.DataFrame([
        _mk_race("r1", "20240106", "05", "芝", "2024-01-06 10:00", "2024-01-06 09:40", speed=2.0),
        _mk_race("r2", "20240106", "05", "芝", "2024-01-06 10:30", "2024-01-06 10:10", speed=-1.0),
        _mk_race("r3", "20240106", "05", "芝", "2024-01-06 11:00", "2024-01-06 10:40", speed=4.0),
    ])
    races_trunc = races_full.iloc[:2]  # r3を削除

    out_full = run_all_units(races_full, "avail_primary", Q, R, PRIOR_VAR).set_index("rid16")
    out_trunc = run_all_units(races_trunc, "avail_primary", Q, R, PRIOR_VAR).set_index("rid16")

    for rid in ["r1", "r2"]:
        for col in ["speed_signal_pre_mean", "speed_signal_pre_var", "speed_signal_pre_n_obs"]:
            assert out_full.loc[rid, col] == pytest.approx(out_trunc.loc[rid, col], abs=1e-9), \
                f"{rid}.{col} が未来イベント削除で変化した"


def test_different_venue_same_day_are_independent_units():
    """同日でも競馬場が違えば別unit(状態が独立)であること。"""
    races = pd.DataFrame([
        _mk_race("r1", "20240106", "05", "芝", "2024-01-06 10:00", "2024-01-06 09:40", speed=3.0),
        _mk_race("r2", "20240106", "06", "芝", None, "2024-01-06 10:10"),  # 別会場
    ])
    out = run_all_units(races, "avail_primary", Q, R, PRIOR_VAR).set_index("rid16")
    assert out.loc["r2", "speed_signal_pre_mean"] == pytest.approx(0.0)


def test_different_surface_same_venue_day_are_independent_units():
    """同日同会場でも芝/ダートが違えば別unitであること。"""
    races = pd.DataFrame([
        _mk_race("r1", "20240106", "05", "芝", "2024-01-06 10:00", "2024-01-06 09:40", speed=3.0),
        _mk_race("r2", "20240106", "05", "ダ", None, "2024-01-06 10:10"),
    ])
    out = run_all_units(races, "avail_primary", Q, R, PRIOR_VAR).set_index("rid16")
    assert out.loc["r2", "speed_signal_pre_mean"] == pytest.approx(0.0)


def test_primary_and_sensitivity_availability_produce_different_series():
    """avail_col を変えるだけで完全に別系列になること(混合しない)。"""
    races = pd.DataFrame([
        {"rid16": "r1", "date": "20240106", "venue": "05", "surface": "芝",
         "avail_primary": pd.Timestamp("2024-01-06 10:00"),
         "avail_sensitivity": pd.Timestamp("2024-01-06 10:10"),
         "decision_timestamp": pd.Timestamp("2024-01-06 09:40"),
         "speed_signal": 3.0, "agari_signal": None, "pace_signal": None},
        {"rid16": "r2", "date": "20240106", "venue": "05", "surface": "芝",
         "avail_primary": pd.NaT, "avail_sensitivity": pd.NaT,
         "decision_timestamp": pd.Timestamp("2024-01-06 10:05"),
         "speed_signal": None, "agari_signal": None, "pace_signal": None},
    ])
    out_primary = run_all_units(races, "avail_primary", Q, R, PRIOR_VAR).set_index("rid16")
    out_sensitivity = run_all_units(races, "avail_sensitivity", Q, R, PRIOR_VAR).set_index("rid16")
    # r2の判断時刻(10:05)はr1のprimary可用時刻(10:00)より後だがsensitivity(10:10)より前
    assert out_primary.loc["r2", "speed_signal_pre_n_obs"] == 1
    assert out_sensitivity.loc["r2", "speed_signal_pre_n_obs"] == 0


def test_raw_baseline_uses_most_recent_observation_only():
    from analysis.mcond.exp08_online_track_state_dev.online_state import run_unit_timeline_raw
    races = pd.DataFrame([
        _mk_race("r1", "20240106", "05", "芝", "2024-01-06 10:00", None, speed=1.0),
        _mk_race("r2", "20240106", "05", "芝", "2024-01-06 10:30", None, speed=5.0),
        _mk_race("r3", "20240106", "05", "芝", None, "2024-01-06 10:40"),
    ])
    out = run_unit_timeline_raw(races, "avail_primary").set_index("rid16")
    assert out.loc["r3", "speed_signal_raw"] == pytest.approx(5.0)  # r2(直近)の値


def test_ewma_baseline_day_resets_and_decays():
    """run_unit_timeline_ewma自体は1unit分の処理関数であり、日をまたぐ分割は
    呼び出し側(run_all_units_generic、date/venue/surfaceでgroupby)の責務。
    ここではrun_all_units_generic経由で正しく呼ぶ(単体関数への直接の複数日
    データ投入は誤用であり、day-resetは保証されない)。"""
    from analysis.mcond.exp08_online_track_state_dev.online_state import (
        run_unit_timeline_ewma, run_all_units_generic,
    )
    races = pd.DataFrame([
        _mk_race("r1", "20240106", "05", "芝", "2024-01-06 10:00", None, speed=1.0),
        _mk_race("r2", "20240106", "05", "芝", "2024-01-06 10:30", None, speed=5.0),
        _mk_race("r3", "20240106", "05", "芝", None, "2024-01-06 10:40"),
        _mk_race("r4", "20240107", "05", "芝", None, "2024-01-07 09:40"),  # 別日
    ])
    out = run_all_units_generic(
        races, "avail_primary", run_unit_timeline_ewma, halflife_hours=2.0,
    ).set_index("rid16")
    val_r3 = out.loc["r3", "speed_signal_ewma"]
    assert 1.0 < val_r3 < 5.0  # r1とr2の間、直近寄り
    assert out.loc["r4", "speed_signal_ewma"] == pytest.approx(0.0)  # 別日でリセット


def test_one_step_ahead_loglik_prefers_correct_noise_level():
    """合成データ(既知の観測ノイズR_true)で、one_step_ahead_loglikがR_trueに近い
    候補を、明らかに違う候補より高く評価すること。"""
    rng = np.random.default_rng(0)
    r_true = 1.0
    rows = []
    t0 = pd.Timestamp("2023-01-06 10:00")
    for i in range(200):
        rows.append({
            "date": "20230106", "venue": "05", "surface": "芝",
            "avail_primary": t0 + pd.Timedelta(minutes=30 * i),
            "speed_signal": rng.normal(0, np.sqrt(r_true)),
        })
    df = pd.DataFrame(rows)
    ll_correct = one_step_ahead_loglik(df, "avail_primary", "speed_signal", q=0.01, r=r_true, prior_var=1.0)
    ll_wrong_small = one_step_ahead_loglik(df, "avail_primary", "speed_signal", q=0.01, r=0.01, prior_var=1.0)
    ll_wrong_big = one_step_ahead_loglik(df, "avail_primary", "speed_signal", q=0.01, r=100.0, prior_var=1.0)
    assert ll_correct > ll_wrong_small
    assert ll_correct > ll_wrong_big
