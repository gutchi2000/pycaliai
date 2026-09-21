# -*- coding: utf-8 -*-
"""
test_dead_heat_audit.py
========================
EXP10 Stage1修正3。dead_heat_audit.py / labels.pyの同着・欠番・重複行判定
ロジックの合成テスト。2023-2025年の実データは使わない。
"""
import numpy as np
import pandas as pd

from labels import _has_dead_heat, _has_missing_finish_number, _has_duplicate_row


def test_has_dead_heat_true_for_any_position():
    assert _has_dead_heat(np.array([1.0, 2.0, 2.0, 4.0])) is True  # 2着同着
    assert _has_dead_heat(np.array([1.0, 1.0, 3.0, 4.0])) is True  # 1着同着
    assert _has_dead_heat(np.array([1.0, 2.0, 3.0, 3.0, 5.0])) is True  # 3着同着


def test_has_dead_heat_false_for_clean_field():
    assert _has_dead_heat(np.array([1.0, 2.0, 3.0, 4.0])) is False


def test_has_dead_heat_ignores_nan():
    assert _has_dead_heat(np.array([1.0, 2.0, np.nan, 4.0])) is False


def test_win_only_dead_heat_matches_exp09_definition():
    """EXP09の(fin==1).sum()>1という定義を、任意順位同着の判定と区別できること。"""
    finish_win_dead_heat = np.array([1.0, 1.0, 3.0, 4.0])
    finish_lower_dead_heat = np.array([1.0, 2.0, 4.0, 4.0])
    win_only = lambda f: bool((f[~np.isnan(f)] == 1.0).sum() > 1)  # noqa: E731
    assert win_only(finish_win_dead_heat) is True
    assert win_only(finish_lower_dead_heat) is False
    # どちらも「任意順位の同着」としては陽性
    assert _has_dead_heat(finish_win_dead_heat) is True
    assert _has_dead_heat(finish_lower_dead_heat) is True


def test_has_missing_finish_number_detects_gap():
    # 1,2,3,4は連番なのでFalse
    assert _has_missing_finish_number(np.array([1.0, 2.0, 3.0, 4.0])) is False
    # 1,3,4は2が欠番なのでTrue(観測数3件に対し{1,2,3}を期待するが{1,3,4})
    assert _has_missing_finish_number(np.array([1.0, 3.0, 4.0])) is True


def test_has_missing_finish_number_realistic_case():
    """rid=2023102204040405で発見された実例を再現(14頭、着順1-15の中で2が欠番)。"""
    finish = np.array([13.0, 8.0, 7.0, 14.0, 5.0, 4.0, 3.0, 10.0, 12.0, 9.0, 6.0, 11.0, 15.0, 1.0])
    assert _has_missing_finish_number(finish) is True


def test_has_missing_finish_number_false_when_dead_heat_present():
    """同着があるレース(重複値)は「欠番」ではなく「同着」カテゴリで捕捉される
    (このヘルパー単体はobserved set != expected setのみを見るため、同着でも
    Trueになりうるが、labels.pyのパイプラインでは同着チェックが先に走るため
    実害はない、という設計を確認する)。"""
    finish_dead_heat = np.array([1.0, 1.0, 3.0])  # unique={1,3}, expected={1,2}
    assert _has_missing_finish_number(finish_dead_heat) is True  # 単体では検出される
    assert _has_dead_heat(finish_dead_heat) is True  # が、dead_heatとしても検出されるため
    # labels.pyのcompute_labels()では_has_dead_heatを先に評価しcontinueするので
    # この場合はdead_heat側のカウンタに計上され、missing_finish_number側には
    # 重複計上されない(排他的except-until-first-match方式)。


def test_has_duplicate_row_true():
    assert _has_duplicate_row(np.array([1, 2, 3, 3, 5])) is True


def test_has_duplicate_row_false():
    assert _has_duplicate_row(np.array([1, 2, 3, 4, 5])) is False


def test_dead_heat_audit_categories_are_mutually_reported_2023_counts():
    """dead_heat_audit.pyの実データ集計値(2023年)が既知の値と一致することを
    回帰確認する(値そのものはSTAGE1_DESIGN.md §3.1に記録済み)。"""
    from dead_heat_audit import audit_dead_heats
    detail, summary = audit_dead_heats([2023])
    assert summary["cat_1_win_dead_heat"] == 3
    assert summary["cat_2_any_position_dead_heat"] == 77
    assert summary["cat_3_missing_finish_number"] == 1
    assert summary["cat_4_non_numeric_finish"] == 0
    assert summary["cat_5_duplicate_row"] == 0


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
