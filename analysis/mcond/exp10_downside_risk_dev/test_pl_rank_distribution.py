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
    resolve_q90_label,
    wilson_ci,
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


# ============================================================
# 5. adaptive q90 解決（モンテカルロ誤差保護）
# ============================================================
def test_wilson_ci_basic_properties():
    lo, hi = wilson_ci(45000, 50000)  # phat=0.9
    assert 0.0 <= lo < 0.9 < hi <= 1.0
    # k=0 -> lo=0, hi>0
    lo0, hi0 = wilson_ci(0, 50000)
    assert lo0 == 0.0 and hi0 > 0.0
    # k=n -> hi=1
    lon, hin = wilson_ci(50000, 50000)
    assert abs(hin - 1.0) < 1e-9 and lon < 1.0


def test_wilson_ci_narrows_with_more_draws():
    lo1, hi1 = wilson_ci(45000, 50000)
    lo2, hi2 = wilson_ci(180000, 200000)  # 同じphat=0.9、4倍のK
    assert (hi2 - lo2) < (hi1 - lo1)


@pytest.mark.parametrize("n", [6, 9, 12])
def test_resolve_q90_uses_exact_dp_when_n_small(n):
    """exact_dp_max_n以下なら常にexact DPを使い、誤差ゼロで一致するはず。"""
    scores = rng.normal(size=n)
    focal = int(np.argmax(scores))
    result = resolve_q90_label(scores, focal, "TESTRACE", exact_dp_max_n=18)
    assert result["method"] == "exact_dp"
    assert result["resolved"] is True
    dist = exact_rank_distribution(scores, focal)
    assert result["q90_rank"] == quantile_rank(dist, 0.90)


@pytest.mark.parametrize("n", [6, 8, 10, 12])
def test_resolve_q90_mc_ladder_matches_exact_dp_ground_truth(n):
    """exact_dp_max_nを強制的に0にしてMC梯子を発火させ、厳密DPと一致するか確認する
    (小さいnで安価にMC梯子の正しさを検証する、実運用ではn<=18は常にexact_dpを使う)。"""
    for trial in range(8):
        scores = rng.normal(scale=1.3, size=n)
        focal = int(np.argmax(scores))
        race_id = f"LADDER_TEST_{n}_{trial}"
        result = resolve_q90_label(scores, focal, race_id, exact_dp_max_n=0, dp_infeasible_n=24)
        assert result["resolved"] is True
        dist = exact_rank_distribution(scores, focal)
        expected_q90 = quantile_rank(dist, 0.90)
        assert result["q90_rank"] == expected_q90, (
            f"n={n} trial={trial}: mc_ladder={result} exact={expected_q90}"
        )


def test_resolve_q90_falls_back_to_exact_dp_when_mc_ladder_exhausted():
    """MC梯子が(小さいK上限で)尽きても、n<=dp_infeasible_nならexact DPへ
    fallbackして必ず確定することを確認する。"""
    n = 10
    scores = rng.normal(size=n)
    focal = int(np.argmax(scores))
    result = resolve_q90_label(
        scores, focal, "FALLBACK_TEST", exact_dp_max_n=0, dp_infeasible_n=24,
        mc_ladder=(50,),  # 極端に小さいKでWilson CIが確定しないよう仕向ける
    )
    assert result["resolved"] is True
    assert result["method"] == "exact_dp_fallback"
    dist = exact_rank_distribution(scores, focal)
    assert result["q90_rank"] == quantile_rank(dist, 0.90)


def test_resolve_q90_unresolved_when_dp_also_infeasible():
    """dp_infeasible_nも下回るほど厳しい設定なら、q90_unresolvedとして
    resolved=Falseを返すことを確認する(このデータセットでは実際には発生しない
    安全弁だが、コードパスとして動作を保証する)。"""
    n = 10
    scores = rng.normal(size=n)
    focal = int(np.argmax(scores))
    result = resolve_q90_label(
        scores, focal, "UNRESOLVED_TEST", exact_dp_max_n=0, dp_infeasible_n=5,
        mc_ladder=(50,),
    )
    assert result["resolved"] is False
    assert result["method"] == "unresolved"
    assert result["q90_rank"] is None


# ============================================================
# 6. PL入力の整合性(raw score vs exp(raw score)、順序不変性)
# ============================================================
def test_gumbel_added_to_raw_score_not_exp_score():
    """Gumbel-maxのlocationはraw score(s)であり、exp(s)ではないことを回帰確認する。
    exp(s)を誤ってlocationに使うと、rank=1確率が標準PL単勝確率(pl_probs.all_tansho)
    と一致しなくなる(exp(s)を通すとsoftmaxの形が壊れるため)。"""
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    import pl_probs as PL

    n = 8
    scores = rng.normal(scale=1.5, size=n)
    w = PL.pl_weights(scores)
    tansho = PL.all_tansho(w)
    for focal in range(n):
        dist = exact_rank_distribution(scores, focal)
        assert abs(dist[0] - tansho[focal]) < 1e-9, (
            f"focal={focal}: raw scoreをlocationに使えばpl_probsと一致するはず"
        )
    # 誤ってexp(score)をlocationに使った場合は一致しなくなることを示す対照実験
    wrong_dist0 = exact_rank_distribution(np.exp(scores - scores.max()), 0)
    assert abs(wrong_dist0[0] - tansho[0]) > 1e-6, (
        "exp(score)をlocationにすると標準PL単勝確率とズレるはず(対照実験)"
    )


@pytest.mark.parametrize("n", [4, 7, 11])
def test_order_invariance_exact_dp(n):
    """馬の入力順序を入れ替えても、同一馬個体のrank分布は変わらないはず。"""
    for trial in range(5):
        scores = rng.normal(scale=1.2, size=n)
        focal = int(rng.integers(0, n))
        dist_orig = exact_rank_distribution(scores, focal)
        perm = rng.permutation(n)
        new_focal = int(np.where(perm == focal)[0][0])
        dist_shuffled = exact_rank_distribution(scores[perm], new_focal)
        assert np.allclose(dist_orig, dist_shuffled, atol=1e-9), f"n={n} trial={trial}"


@pytest.mark.parametrize("n", [6, 10])
def test_order_invariance_mc(n):
    """モンテカルロでも(seedがrace_id由来のため)入力順序を変えても同一分布になるはず
    (scores配列の並べ替えのみで、race_id/seedは固定)。"""
    scores = rng.normal(scale=1.2, size=n)
    focal = int(rng.integers(0, n))
    perm = rng.permutation(n)
    new_focal = int(np.where(perm == focal)[0][0])
    dist_orig = mc_rank_distribution(scores, focal, 100_000, 1, "ORDER_TEST")
    dist_shuffled = mc_rank_distribution(scores[perm], new_focal, 100_000, 1, "ORDER_TEST")
    se = mc_standard_error(dist_orig, 100_000)
    assert np.all(np.abs(dist_orig - dist_shuffled) <= 5 * se + 1e-4)


def test_resolve_q90_deterministic_given_seed_and_race_id():
    n = 10
    scores = rng.normal(size=n)
    focal = int(np.argmax(scores))
    a = resolve_q90_label(scores, focal, "DET_TEST", exact_dp_max_n=0, global_seed=7)
    b = resolve_q90_label(scores, focal, "DET_TEST", exact_dp_max_n=0, global_seed=7)
    assert a["q90_rank"] == b["q90_rank"]
    assert a["method"] == b["method"]
    assert a["resolved"] == b["resolved"]
    assert np.array_equal(a["dist"], b["dist"])


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
