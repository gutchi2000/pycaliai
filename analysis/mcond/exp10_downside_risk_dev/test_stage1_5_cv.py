# -*- coding: utf-8 -*-
"""
test_stage1_5_cv.py
====================
EXP10 Stage1.5。forward_chaining_folds() / meeting_day_paired_bootstrap()の
合成テスト。同一開催日がtrain/testに分割されないこと、時系列順序が守られる
ことを確認する。2023-2025年の実データは使わない。
"""
import numpy as np

from stage1_5_cv import forward_chaining_folds, meeting_day_paired_bootstrap


def test_forward_chaining_folds_no_same_day_split():
    meeting_days = [f"2023{str(i).zfill(4)}01" for i in range(1, 61)]  # 60日分、昇順
    folds = forward_chaining_folds(meeting_days, n_blocks=6)
    assert len(folds) == 5
    for train_md, test_md in folds:
        assert train_md.isdisjoint(test_md), "同一開催日がtrain/testに同時に入ってはいけない"


def test_forward_chaining_folds_preserves_time_order():
    meeting_days = [f"2023{str(i).zfill(4)}01" for i in range(1, 61)]
    folds = forward_chaining_folds(meeting_days, n_blocks=6)
    for train_md, test_md in folds:
        assert max(train_md) < min(test_md), "trainは常にtestより過去でなければならない"


def test_forward_chaining_folds_train_grows_each_fold():
    meeting_days = [f"2023{str(i).zfill(4)}01" for i in range(1, 61)]
    folds = forward_chaining_folds(meeting_days, n_blocks=6)
    sizes = [len(train_md) for train_md, _ in folds]
    assert sizes == sorted(sizes), "foldが進むほどtrain集合は単調に拡大するはず"
    assert sizes[-1] > sizes[0]


def test_forward_chaining_folds_covers_all_but_first_block():
    meeting_days = [f"2023{str(i).zfill(4)}01" for i in range(1, 61)]
    folds = forward_chaining_folds(meeting_days, n_blocks=6)
    all_test = set()
    for _, test_md in folds:
        all_test |= test_md
    all_train_final = folds[-1][0] | folds[-1][1]
    assert all_train_final == set(meeting_days), "最終foldのtrain+testが全体を覆うはず"


def test_meeting_day_paired_bootstrap_basic():
    diffs = np.array([0.1, 0.2, 0.15, -0.05, 0.3, 0.1, 0.05])  # 平均的に正
    result = meeting_day_paired_bootstrap(diffs, n_boot=2000, seed=1)
    assert result["mean"] == np.mean(diffs)
    assert result["ci_lo"] < result["mean"] < result["ci_hi"]
    assert 0.0 <= result["p_negative"] <= 1.0


def test_meeting_day_paired_bootstrap_deterministic_given_seed():
    diffs = np.array([0.1, -0.2, 0.3, -0.1, 0.05])
    a = meeting_day_paired_bootstrap(diffs, n_boot=1000, seed=42)
    b = meeting_day_paired_bootstrap(diffs, n_boot=1000, seed=42)
    assert a == b


def test_meeting_day_paired_bootstrap_all_positive_gives_ci_above_zero():
    diffs = np.array([0.5, 0.6, 0.55, 0.7, 0.45, 0.62, 0.58])
    result = meeting_day_paired_bootstrap(diffs, n_boot=3000, seed=7)
    assert result["ci_lo"] > 0
    assert result["p_negative"] < 0.05


def test_meeting_day_paired_bootstrap_straddles_zero_when_mixed():
    diffs = np.array([0.001, -0.002, 0.0015, -0.001, 0.0005, -0.0018, 0.0009])
    result = meeting_day_paired_bootstrap(diffs, n_boot=3000, seed=7)
    assert result["ci_lo"] < 0 < result["ci_hi"]


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
