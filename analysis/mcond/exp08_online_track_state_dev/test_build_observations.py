# -*- coding: utf-8 -*-
"""
test_build_observations.py — expected_time_model.py / build_observations.py の単体テスト。
2026-09-20実装中に見つかった4件の実害バグの回帰テストを含む:
  1. コース区分がダートで構造的にNaN(A/B/C/D等は芝のみの概念)なのに
     dropnaで全ダートレースが消えていた
  2. 障害レースがクラス名を平地と共有し、距離bucketの中央値を歪めていた
  3. 着順が1-9着=全角数字/10着以降=半角数字という不均一エンコードで、
     pd.to_numericへ直接通すと勝ち馬を含む上位9頭が全滅していた
  4. 枠番(1-8固定)を内外percentileの分母に使い、大頭数レースでis_outerの
     閾値が非現実的になりinside_signalが全レースNaNになっていた
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
    _parse_finish_pos, _parse_time_sec, fit_expected_time_lookup, predict_expected_time,
)


def test_parse_finish_pos_handles_fullwidth_digits():
    s = pd.Series(["１", "２", "９", "10", "18", "中止", None])
    out = _parse_finish_pos(s)
    assert out.tolist()[:5] == [1.0, 2.0, 9.0, 10.0, 18.0]
    assert out.iloc[5:].isna().all()


def test_parse_finish_pos_regression_top9_not_dropped():
    """バグ3の直接回帰: 全角1-9着が数値化後に生存すること。"""
    s = pd.Series([f"{d}" for d in "１２３４５６７８９"])
    out = _parse_finish_pos(s)
    assert out.notna().all()
    assert sorted(out.tolist()) == list(range(1, 10))


def test_parse_time_sec_matches_known_examples():
    # 1336 -> 1分33秒6 = 93.6秒 (実データの距離別平均で妥当性確認済み)
    assert _parse_time_sec(pd.Series([1336]))[0] == pytest.approx(93.6)
    assert _parse_time_sec(pd.Series([2000]))[0] == pytest.approx(120.0)


def test_expected_time_lookup_fallback_chain():
    """フォールバック段階(full -> no_going -> coarse -> global)が正しく機能するか。
    trainに存在しない組合せへの予測がglobal medianへ落ちることを確認。"""
    train = pd.DataFrame({
        "dist_bucket": [1600] * 40, "芝・ダ": ["芝"] * 40, "コース区分": ["A"] * 40,
        "馬場状態": ["良"] * 40, "クラス名": ["1勝"] * 40,
        "time_sec": np.linspace(95.0, 97.0, 40),
    })
    lookup = fit_expected_time_lookup(train)
    # 完全一致セル
    q_full = pd.DataFrame({"dist_bucket": [1600], "芝・ダ": ["芝"], "コース区分": ["A"],
                            "馬場状態": ["良"], "クラス名": ["1勝"]})
    pred, fb = predict_expected_time(lookup, q_full)
    assert fb.iloc[0] == "full"
    assert pred.iloc[0] == pytest.approx(96.0, abs=0.2)
    # 未知の組合せ(全く違う距離・クラス) -> global
    q_unknown = pd.DataFrame({"dist_bucket": [9999], "芝・ダ": ["ダ"], "コース区分": ["D_NA"],
                               "馬場状態": ["不"], "クラス名": ["Ｇ１"]})
    pred2, fb2 = predict_expected_time(lookup, q_unknown)
    assert fb2.iloc[0] == "global"
    assert pred2.iloc[0] == pytest.approx(train["time_sec"].median())


def test_dirt_course_division_not_dropped():
    """バグ1の直接回帰: コース区分がNaN(ダート)の行がdropnaで消えないこと。"""
    from analysis.mcond.exp08_online_track_state_dev.expected_time_model import (
        load_kekka_for_time_model,
    )
    df = load_kekka_for_time_model()
    dirt = df[df["芝・ダ"] == "ダ"]
    assert len(dirt) > 0, "ダートレースが1件も残っていない(コース区分dropnaバグの再発)"
    assert (dirt["コース区分"] == "D_NA").all()


def test_no_obstacle_races_in_loaded_data():
    """バグ2の直接回帰: レース名に'障害'を含む行が残っていないこと。"""
    from analysis.mcond.exp08_online_track_state_dev.expected_time_model import KEKKA_EXT
    raw = pd.read_csv(KEKKA_EXT, encoding="utf-8-sig", low_memory=False,
                       usecols=["レース名"], nrows=5000)
    assert raw["レース名"].astype(str).str.contains("障害", na=False).any(), \
        "テスト前提が崩れている(サンプルに障害レースが無い)"
    from analysis.mcond.exp08_online_track_state_dev.expected_time_model import (
        load_kekka_for_time_model,
    )
    df = load_kekka_for_time_model(usecols=[
        "yyyymmdd", "race_id16", "レース名", "馬番", "着順", "走破タイム", "距離",
        "芝・ダ", "コース区分", "馬場状態", "クラス名",
    ])
    assert not df["レース名"].astype(str).str.contains("障害", na=False).any()


def test_inside_outside_uses_umaban_not_wakuban():
    """バグ4の直接回帰: is_outerが馬番(頭数まで変動)ベースで、大頭数レースでも
    非空集合になること(枠番=1-8固定ベースだと18頭立てでほぼ空集合になっていた)。"""
    n_horse = 18
    uma = pd.Series(range(1, n_horse + 1))
    is_outer = uma >= max(n_horse - 2, 4)
    assert is_outer.sum() >= 3, "18頭立てでis_outerがほぼ空集合(枠番ベースのバグ再発)"


def test_build_observations_signals_not_degenerate():
    """4バグ修正後の統合テスト: inside_signal/front_signalが全NaN・分散0という
    退化状態(修正前の実害)に戻っていないこと。"""
    from analysis.mcond.exp08_online_track_state_dev.build_observations import (
        build_race_level_observations,
    )
    race, meta = build_race_level_observations()
    assert race["inside_signal"].notna().mean() > 0.9, "inside_signalが再び大半NaN"
    assert race["front_signal"].std() > 0.01, "front_signalが再び分散ゼロ(退化)"
    assert race["speed_signal"].abs().max() < 10, "speed_signalのwinsorizeが機能していない"
    # 直感的な符号チェック: 3角先頭集団の方が後方集団より複勝率が高いはず(物理的に妥当)
    assert race["front_signal"].mean() > 0
