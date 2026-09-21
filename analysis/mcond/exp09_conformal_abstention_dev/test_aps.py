# -*- coding: utf-8 -*-
"""
test_aps.py — aps.pyの合成テスト(数式・有限標本quantile・tie-break・coverage分離)。
実データ(2024-2025)は一切使わない。
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.exp09_conformal_abstention_dev.aps import (  # noqa: E402
    nonconformity_score, compute_q_hat, prediction_set, fractional_effective_set_size,
    deterministic_hash_rank, aps_derived_abstention_score, build_calibration_scores,
    build_prediction_sets, empirical_coverage, ALPHA, NOMINAL_COVERAGE, _tansho_probs,
)
import pl_probs as PL  # noqa: E402


def test_tansho_probs_matches_pl_probs_reference():
    """独自実装の_tansho_probsがpl_probs.all_tansho(pl_probs.pl_weights(.))と
    数値的に完全一致すること。"""
    rng = np.random.default_rng(0)
    scores = rng.normal(size=10)
    ours = _tansho_probs(scores)
    ref = PL.all_tansho(PL.pl_weights(scores))
    np.testing.assert_allclose(ours, ref, atol=1e-12)
    assert ours.sum() == pytest.approx(1.0, abs=1e-9)


def test_nonconformity_score_true_winner_is_favorite():
    """真の勝ち馬が最有力(最高確率)なら、S_i=その馬の確率そのもの
    (累積和は1項のみ)。"""
    probs = np.array([0.5, 0.3, 0.2])
    s = nonconformity_score(probs, true_idx=0)
    assert s == pytest.approx(0.5)


def test_nonconformity_score_true_winner_is_longshot():
    """真の勝ち馬が最下位人気なら、S_i=全確率の和=1(全馬含むまでの累積)。"""
    probs = np.array([0.5, 0.3, 0.2])
    s = nonconformity_score(probs, true_idx=2)  # 0.2が最下位
    assert s == pytest.approx(1.0)


def test_nonconformity_score_middle_rank():
    probs = np.array([0.5, 0.3, 0.2])
    s = nonconformity_score(probs, true_idx=1)  # 2番人気(0.3)
    assert s == pytest.approx(0.5 + 0.3)  # 降順で1位+2位の累積


def test_q_hat_formula_matches_spec():
    """k=ceil((n+1)*(1-alpha))、q_hat=k番目に小さい値(1-indexed)。手計算と一致。"""
    scores = np.array([0.1, 0.9, 0.5, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.05])
    n = len(scores)
    alpha = 0.10
    k_expected = int(np.ceil((n + 1) * (1 - alpha)))  # ceil(11*0.9)=ceil(9.9)=10
    assert k_expected == 10
    res = compute_q_hat(scores, alpha=alpha)
    assert res.n == n
    assert res.k == k_expected
    assert res.fail_closed is False
    sorted_scores = np.sort(scores)
    assert res.q_hat == pytest.approx(sorted_scores[k_expected - 1])


def test_q_hat_fail_closed_when_k_exceeds_n():
    """nが極端に小さい場合、k>nとなりfail-closed(q_hat=1.0)になること。"""
    scores = np.array([0.1, 0.2])  # n=2, k=ceil(3*0.9)=ceil(2.7)=3>2
    res = compute_q_hat(scores, alpha=0.10)
    assert res.fail_closed is True
    assert res.q_hat == pytest.approx(1.0)
    assert res.k > res.n


def test_q_hat_not_general_percentile_function():
    """q_hatがnp.percentileの単純な補間結果とは異なる(k番目要素そのもの)ことを
    確認する退行テスト(将来np.percentileに置き換えられていないかの防止)。"""
    rng = np.random.default_rng(1)
    scores = rng.uniform(size=37)
    res = compute_q_hat(scores, alpha=0.10)
    naive_percentile = np.percentile(scores, 90, method="linear")
    # 一致する保証はない(補間 vs order statistic)ため、少なくとも計算式が
    # order statisticであることを直接検証する
    sorted_scores = np.sort(scores)
    assert res.q_hat == sorted_scores[res.k - 1]


def test_prediction_set_non_randomized_monotonic_in_q_hat():
    """q_hatが大きいほど予測集合が大きい(単調)、q_hat=1なら全馬を含む。"""
    probs = np.array([0.4, 0.3, 0.2, 0.1])
    small = prediction_set(probs, q_hat=0.3)
    large = prediction_set(probs, q_hat=0.9)
    full = prediction_set(probs, q_hat=1.0)
    assert len(small) <= len(large) <= len(full)
    assert len(full) == 4


def test_prediction_set_always_includes_favorite_when_q_hat_positive():
    probs = np.array([0.4, 0.3, 0.2, 0.1])
    pset = prediction_set(probs, q_hat=0.05)
    assert 0 in pset  # 最有力馬(index0)は必ず含まれる(non-randomized APSの性質)


def test_fractional_effective_set_size_range():
    probs = np.array([0.4, 0.3, 0.2, 0.1])
    frac = fractional_effective_set_size(probs, q_hat=0.5)
    assert 0.0 <= frac <= 1.0


def test_deterministic_hash_rank_is_deterministic_and_uses_no_result_label():
    """同じrid16・seedなら毎回同じ値(結果ラベルを引数に取らない設計そのものが
    「結果を使わない」ことの構造的保証)。"""
    a = deterministic_hash_rank("2024010606010101")
    b = deterministic_hash_rank("2024010606010101")
    assert a == b
    c = deterministic_hash_rank("2024010606010102")
    assert a != c  # 異なるrace_idなら(ほぼ確実に)異なる値


def test_aps_derived_abstention_score_orders_by_size_first():
    """3段階tie-breakのタプルが、まずsizeで、次にfracで、最後にhashで
    比較されること(タプルの辞書式比較)。"""
    small = aps_derived_abstention_score(2, 0.9, "r1")
    large = aps_derived_abstention_score(3, 0.1, "r2")
    assert small < large  # sizeが優先されるのでfracが逆でもsmallが小さい


def _mk_race_frame(scores, winner_idx):
    n = len(scores)
    fin = [2] * n
    fin[winner_idx] = 1
    win = [0] * n
    win[winner_idx] = 1
    return pd.DataFrame({"ban": range(1, n + 1), "v6_score": scores, "fin": fin, "win": win})


def test_build_calibration_scores_and_prediction_sets_integration():
    """小さな合成データセットでcalibration→q_hat→prediction set→coverageの
    一連の流れが矛盾なく動くこと。"""
    rng = np.random.default_rng(2)
    calib_frames = {}
    for i in range(50):
        n = rng.integers(5, 12)
        scores = rng.normal(size=n)
        winner = int(np.argmax(scores))  # 較正が完璧なら最有力が勝つケースを多く含める
        if rng.random() < 0.3:
            winner = int(rng.integers(0, n))  # 一部は番狂わせ
        calib_frames[f"calib_{i}"] = _mk_race_frame(scores, winner)

    calib_df = build_calibration_scores(calib_frames)
    assert len(calib_df) == 50
    assert (calib_df["nonconformity_score"] >= 0).all()
    assert (calib_df["nonconformity_score"] <= 1.0001).all()

    q_res = compute_q_hat(calib_df["nonconformity_score"].to_numpy(), alpha=ALPHA)
    assert not q_res.fail_closed

    eval_frames = {}
    for i in range(30):
        n = rng.integers(5, 12)
        scores = rng.normal(size=n)
        winner = int(rng.integers(0, n))
        eval_frames[f"eval_{i}"] = _mk_race_frame(scores, winner)

    pred_df = build_prediction_sets(eval_frames, q_res.q_hat)
    assert len(pred_df) == 30
    assert (pred_df["prediction_set_size"] >= 1).all()
    cov_all = empirical_coverage(pred_df)
    assert 0.0 <= cov_all <= 1.0


def test_empirical_coverage_subsetting():
    pred_df = pd.DataFrame({
        "rid16": ["r1", "r2", "r3", "r4"],
        "covered": [True, True, False, True],
    })
    assert empirical_coverage(pred_df) == pytest.approx(0.75)
    assert empirical_coverage(pred_df, {"r1", "r2"}) == pytest.approx(1.0)
    assert empirical_coverage(pred_df, {"r3"}) == pytest.approx(0.0)


def test_nominal_coverage_constant():
    assert NOMINAL_COVERAGE == pytest.approx(0.90)
    assert ALPHA == pytest.approx(0.10)
