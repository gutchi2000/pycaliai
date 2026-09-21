# -*- coding: utf-8 -*-
"""
test_asof_features.py
=======================
EXP11 Stage1修正2。asof_features.pyの合成テスト+削除不変性テスト。
2013-2025年の実データは使わない(削除不変性テストは合成データで検証、
実データでの確認はverify_deletion_invariance_realdata.pyで別途行う)。
"""
import numpy as np
import pandas as pd
import pytest

from asof_features import add_asof_prior_counts, classify_unknown_category, add_unknown_category_class


def _toy_df():
    # J1×T1が2013-01-01と2013-01-02に、J1×T2が2013-01-03に、というシナリオ
    return pd.DataFrame({
        "日付": [20130101, 20130101, 20130102, 20130103, 20130104],
        "騎手コード": ["J1", "J1", "J1", "J1", "J2"],
        "調教師コード": ["T1", "T1", "T1", "T2", "T1"],
    })


def test_first_occurrence_has_zero_prior_count():
    df = add_asof_prior_counts(_toy_df())
    assert df.loc[0, "pair_prior_count_asof"] == 0
    assert df.loc[1, "pair_prior_count_asof"] == 0  # 同日内は未加算(日初時点固定)


def test_same_day_rows_share_snapshot():
    """同日内の複数行は同じ日初スナップショットを見る(互いにカウントしない)。"""
    df = add_asof_prior_counts(_toy_df())
    row0, row1 = df.iloc[0], df.iloc[1]
    assert row0["pair_prior_count_asof"] == row1["pair_prior_count_asof"] == 0


def test_next_day_reflects_previous_day_count():
    df = add_asof_prior_counts(_toy_df())
    # 2013-01-02のJ1×T1は、2013-01-01の2件(同日2行)を反映して2になるはず
    row2 = df[df["日付"] == 20130102].iloc[0]
    assert row2["pair_prior_count_asof"] == 2


def test_different_pair_starts_at_zero():
    df = add_asof_prior_counts(_toy_df())
    row3 = df[df["日付"] == 20130103].iloc[0]  # J1×T2、初出
    assert row3["pair_prior_count_asof"] == 0


def test_jockey_and_trainer_prior_counts_accumulate_independently():
    df = add_asof_prior_counts(_toy_df())
    row3 = df[df["日付"] == 20130103].iloc[0]  # J1×T2だがJ1自体は3走目
    assert row3["jockey_prior_count_asof"] == 3  # 01-01×2 + 01-02×1
    assert row3["trainer_prior_count_asof"] == 0  # T2は初登場
    row4 = df[df["日付"] == 20130104].iloc[0]  # J2×T1、J2初、T1は01-01(2件)+01-02(1件)=3回既出
    assert row4["jockey_prior_count_asof"] == 0
    assert row4["trainer_prior_count_asof"] == 3


def test_order_independent_of_input_row_order():
    """入力行の順序をシャッフルしても、日付ベースのソートで同じ結果になるはず。"""
    df = _toy_df()
    shuffled = df.sample(frac=1.0, random_state=1).reset_index(drop=True)
    a = add_asof_prior_counts(df).sort_values(["日付", "騎手コード", "調教師コード"]).reset_index(drop=True)
    b = add_asof_prior_counts(shuffled).sort_values(["日付", "騎手コード", "調教師コード"]).reset_index(drop=True)
    pd.testing.assert_series_equal(a["pair_prior_count_asof"], b["pair_prior_count_asof"])


# ============================================================
# 削除不変性テスト(合成データ)
# ============================================================
def test_deletion_invariance_synthetic():
    """未来の日付の行を削除しても、過去の行のpair_prior_count_asofは
    変化しないはず(真にas-ofであることの直接的な検証)。"""
    rng = np.random.default_rng(0)
    n = 500
    dates = np.sort(rng.integers(20230101, 20231231, size=n))
    jockeys = rng.choice([f"J{i}" for i in range(20)], size=n)
    trainers = rng.choice([f"T{i}" for i in range(15)], size=n)
    df = pd.DataFrame({"日付": dates, "騎手コード": jockeys, "調教師コード": trainers})

    full = add_asof_prior_counts(df)
    cutoff = np.median(dates)
    truncated_input = df[df["日付"] <= cutoff]
    truncated = add_asof_prior_counts(truncated_input)

    full_past = full[full["日付"] <= cutoff].sort_values(["日付", "騎手コード", "調教師コード"]).reset_index(drop=True)
    trunc_past = truncated.sort_values(["日付", "騎手コード", "調教師コード"]).reset_index(drop=True)

    pd.testing.assert_series_equal(
        full_past["pair_prior_count_asof"], trunc_past["pair_prior_count_asof"]
    )
    pd.testing.assert_series_equal(
        full_past["jockey_prior_count_asof"], trunc_past["jockey_prior_count_asof"]
    )
    pd.testing.assert_series_equal(
        full_past["trainer_prior_count_asof"], trunc_past["trainer_prior_count_asof"]
    )


def test_deletion_invariance_multiple_cutoffs():
    """複数の削除点で不変性を確認する(境界条件の頑健性)。"""
    rng = np.random.default_rng(7)
    n = 800
    dates = np.sort(rng.integers(20220101, 20241231, size=n))
    jockeys = rng.choice([f"J{i}" for i in range(30)], size=n)
    trainers = rng.choice([f"T{i}" for i in range(20)], size=n)
    df = pd.DataFrame({"日付": dates, "騎手コード": jockeys, "調教師コード": trainers})
    full = add_asof_prior_counts(df)

    for cutoff in [np.percentile(dates, 25), np.percentile(dates, 50), np.percentile(dates, 75)]:
        truncated = add_asof_prior_counts(df[df["日付"] <= cutoff])
        full_past = full[full["日付"] <= cutoff].sort_values(
            ["日付", "騎手コード", "調教師コード"]).reset_index(drop=True)
        trunc_past = truncated.sort_values(
            ["日付", "騎手コード", "調教師コード"]).reset_index(drop=True)
        pd.testing.assert_series_equal(
            full_past["pair_prior_count_asof"], trunc_past["pair_prior_count_asof"],
            check_names=False,
        )


# ============================================================
# 未知カテゴリ分類
# ============================================================
def test_classify_unknown_category_both_new():
    row = pd.Series({"jockey_prior_count_asof": 0, "trainer_prior_count_asof": 0, "pair_prior_count_asof": 0})
    assert classify_unknown_category(row) == "both_new"


def test_classify_unknown_category_new_jockey_known_trainer():
    row = pd.Series({"jockey_prior_count_asof": 0, "trainer_prior_count_asof": 10, "pair_prior_count_asof": 0})
    assert classify_unknown_category(row) == "new_jockey_known_trainer"


def test_classify_unknown_category_known_jockey_new_trainer():
    row = pd.Series({"jockey_prior_count_asof": 10, "trainer_prior_count_asof": 0, "pair_prior_count_asof": 0})
    assert classify_unknown_category(row) == "known_jockey_new_trainer"


def test_classify_unknown_category_known_both_new_pair():
    row = pd.Series({"jockey_prior_count_asof": 10, "trainer_prior_count_asof": 10, "pair_prior_count_asof": 0})
    assert classify_unknown_category(row) == "known_jockey_known_trainer_new_pair"


def test_classify_unknown_category_known_pair():
    row = pd.Series({"jockey_prior_count_asof": 10, "trainer_prior_count_asof": 10, "pair_prior_count_asof": 3})
    assert classify_unknown_category(row) == "known_pair"


def test_add_unknown_category_class_buckets():
    df = pd.DataFrame({
        "jockey_prior_count_asof": [10, 10, 10],
        "trainer_prior_count_asof": [10, 10, 10],
        "pair_prior_count_asof": [0, 2, 7],
    })
    out = add_unknown_category_class(df)
    assert list(out["pair_prior_bucket"].astype(str)) == ["0", "1-4", "5+"]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
