# -*- coding: utf-8 -*-
"""
test_pl_rank_distribution.py
=============================
EXP10 Stage 1. pl_rank_distribution.py の合成oracleテスト。

3層の相互検証:
  1. brute_force (全順列列挙, n<=8) vs exact (bitmask DP)   — 厳密同士、完全一致必須
  2. exact (bitmask DP) vs mc (Gumbel-max モンテカルロ)     — 大標本誤差内で一致
  3. 分布の基本性質（合計1、非負、単調性なし確認等）

2023-2025年の実データは一切使わない。
"""
import numpy as np
import pytest

from pl_rank_distribution import (
    brute_force_rank_distribution,
    exact_rank_distribution,
    mc_rank_distribution,
    mc_standard_error,
    quantile_rank,
    expected_rank,
    rank_percentile,
    race_seed,
)

rng = np.random.default_rng(20260921)


# ============================================================
# 1. brute force vs exact DP（完全一致、n=3..8）
# ============================================================
@pytest.mark.parametrize("n", [3, 4, 5, 6, 7, 8])
def test_exact_matches_brute_force_random(n):
    for trial in range(5):
        scores = rng.normal(scale=1.5, size=n)
        focal = rng.integers(0, n)
        bf = brute_force_rank_distribution(scores, focal)
        ex = exact_rank_distribution(scores, focal)
        assert np.allclose(bf, ex, atol=1e-9), (
            f"n={n} trial={trial} focal={focal}: brute={bf} exact={ex}"
        )


def test_exact_matches_brute_force_uniform_weights():
    # 全馬同スコア → 各順位は 1/n で一様
    for n in (3, 5, 8):
        scores = np.zeros(n)
        bf = brute_force_rank_distribution(scores, 0)
        ex = exact_rank_distribution(scores, 0)
        uniform = np.full(n, 1.0 / n)
        assert np.allclose(bf, uniform, atol=1e-9)
        assert np.allclose(ex, uniform, atol=1e-9)


def test_exact_matches_brute_force_extreme_dominance():
    # focalが圧倒的に強い → ほぼ確実に1着
    n = 6
    scores = np.array([20.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    bf = brute_force_rank_distribution(scores, 0)
    ex = exact_rank_distribution(scores, 0)
    assert np.allclose(bf, ex, atol=1e-9)
    assert bf[0] > 0.999999  # ほぼ確実に1位

    # focalが圧倒的に弱い → ほぼ確実に最下位
    scores2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -20.0])
    bf2 = brute_force_rank_distribution(scores2, 5)
    ex2 = exact_rank_distribution(scores2, 5)
    assert np.allclose(bf2, ex2, atol=1e-9)
    assert bf2[-1] > 0.999999


# ============================================================
# 2. exact DP vs Monte Carlo（統計誤差内で一致）
# ============================================================
@pytest.mark.parametrize("n", [5, 10, 16, 18])
def test_mc_matches_exact_within_tolerance(n):
    scores = rng.normal(scale=1.2, size=n)
    focal = int(np.argmax(scores))  # ◎ = 最高スコア馬（実運用と同じ選び方）
    ex = exact_rank_distribution(scores, focal)
    n_draws = 200_000
    mc = mc_rank_distribution(scores, focal, n_draws, global_seed=1, race_id="TEST_RACE")
    se = mc_standard_error(mc, n_draws)
    # 5-sigma以内（多重比較を考慮し緩め、n個の同時検定でも十分保守的）
    tol = 5 * se + 1e-4
    assert np.all(np.abs(mc - ex) <= tol), (
        f"n={n}: max diff={np.max(np.abs(mc-ex)):.5f} tol={np.max(tol):.5f}"
    )


def test_mc_deterministic_given_seed_and_race_id():
    scores = rng.normal(size=12)
    focal = 3
    a = mc_rank_distribution(scores, focal, 10_000, global_seed=42, race_id="R1")
    b = mc_rank_distribution(scores, focal, 10_000, global_seed=42, race_id="R1")
    assert np.array_equal(a, b)


def test_mc_seed_independent_of_iteration_order():
    """異なるレースを異なる順序で処理しても、同じrace_idなら同じ結果になる
    （race_seedがrace_id由来で、グローバルRNGの消費順序に依存しないことの確認）。"""
    scores1 = rng.normal(size=8)
    scores2 = rng.normal(size=10)
    a1 = mc_rank_distribution(scores1, 0, 5_000, global_seed=7, race_id="A")
    a2 = mc_rank_distribution(scores2, 0, 5_000, global_seed=7, race_id="B")
    # 順序を入れ替えて再計算
    b2 = mc_rank_distribution(scores2, 0, 5_000, global_seed=7, race_id="B")
    b1 = mc_rank_distribution(scores1, 0, 5_000, global_seed=7, race_id="A")
    assert np.array_equal(a1, b1)
    assert np.array_equal(a2, b2)


def test_race_seed_differs_across_race_ids():
    s1 = race_seed("2024010106010101", 1)
    s2 = race_seed("2024010106010102", 1)
    assert s1 != s2


# ============================================================
# 3. 分布の基本性質
# ============================================================
@pytest.mark.parametrize("n", [3, 6, 12, 18])
def test_exact_distribution_sums_to_one_and_nonnegative(n):
    scores = rng.normal(size=n)
    focal = rng.integers(0, n)
    dist = exact_rank_distribution(scores, focal)
    assert abs(dist.sum() - 1.0) < 1e-9
    assert np.all(dist >= -1e-12)


def test_mc_distribution_sums_to_one():
    scores = rng.normal(size=14)
    dist = mc_rank_distribution(scores, 2, 20_000, global_seed=5, race_id="X")
    assert abs(dist.sum() - 1.0) < 1e-9


def test_all_horses_rank1_probability_matches_pl_tansho():
    """rank=1の周辺確率は標準PL単勝確率 w_i/sum(w) と一致するはず（pl_probs.pyとの整合性）。"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # E:\PyCaLiAI (BASE)
    import pl_probs as PL

    n = 10
    scores = rng.normal(size=n)
    w = PL.pl_weights(scores)
    tansho = PL.all_tansho(w)
    for focal in range(n):
        dist = exact_rank_distribution(scores, focal)
        assert abs(dist[0] - tansho[focal]) < 1e-9, f"focal={focal}"


# ============================================================
# 4. quantile / expected / percentile ヘルパー
# ============================================================
def test_quantile_rank_basic():
    dist = np.array([0.5, 0.3, 0.1, 0.1])  # ranks 1..4
    assert quantile_rank(dist, 0.5) == 1     # cdf(1)=0.5 >= 0.5
    assert quantile_rank(dist, 0.51) == 2    # cdf(1)=0.5 < 0.51 <= cdf(2)=0.8
    assert quantile_rank(dist, 0.8) == 2
    assert quantile_rank(dist, 0.81) == 3
    assert quantile_rank(dist, 1.0) == 4


def test_quantile_rank_degenerate_certain_win():
    dist = np.array([1.0, 0.0, 0.0, 0.0])
    assert quantile_rank(dist, 0.9) == 1
    assert quantile_rank(dist, 0.999999) == 1


def test_expected_rank_uniform():
    n = 5
    dist = np.full(n, 1.0 / n)
    assert abs(expected_rank(dist) - 3.0) < 1e-9  # 一様分布の期待値 = (n+1)/2


def test_rank_percentile_bounds():
    assert rank_percentile(1, 10) == 0.0
    assert rank_percentile(10, 10) == 1.0
    assert abs(rank_percentile(5, 9) - 0.5) < 1e-9


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
