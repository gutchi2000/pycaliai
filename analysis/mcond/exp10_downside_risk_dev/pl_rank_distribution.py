# -*- coding: utf-8 -*-
r"""
pl_rank_distribution.py
========================
EXP10 Stage 1. Plackett-Luce の下で、単一の focal 馬（◎）の
「レース内順位」の周辺分布 P(rank_focal = r), r=1..n を計算する。

PL 過程（逐次除去表現）:
    w_i = exp(s_i - max(s))
    各ステップで残存馬から w_i に比例した確率で1頭を選び除去、を n 回繰り返す。
    rank_i = i が選ばれたステップ番号(1-indexed)。

厳密な同値表現（Gumbel-max trick, Yellott 1977）:
    g_j ~ Gumbel(0,1) i.i.d. を独立に加え、(s_j + g_j) の降順で並べた順列は
    厳密に PL(w) 分布に従う。rank_i = 1 + #{j != i : s_j+g_j > s_i+g_i}。
    この事実により、モンテカルロでの厳密サンプリング（近似ではなく厳密分布からの
    サンプル、有限抽出数 K による標本誤差のみが近似要因）が可能になる。

本モジュールが提供する3つの計算法（相互検証用）:
    1. mc_rank_distribution()     : Gumbel-max モンテカルロ（本番用、高速・ベクトル化）
    2. exact_rank_distribution()  : bitmask 動的計画法（厳密、O(2^(n-1)*n)、中規模nの検証用）
    3. brute_force_rank_distribution(): 全順列列挙（厳密、O(n!)、n<=8のoracleテスト専用）

数式（bitmask DP、§3.2 DATA_AUDIT.md / spec.json aps相当の定義に対応）:
    focal 以外の n-1 頭を others、重み w_j (j in others)、focal 重み w_f とする。
    W_O = Σ_{j in others} w_j。
    dp[mask] = P(mask に含まれる others が「focalより前に」全て選ばれ、
                 focal はまだ選ばれていない状態に到達する確率)
    dp[∅] = 1
    dp[mask] = Σ_{j in mask} dp[mask\{j}] * w_j / (w_f + W_O - W(mask\{j}))
    P(rank_f = |mask|+1) の寄与 = Σ_{mask: |mask|=r-1} dp[mask] * w_f / (w_f + W_O - W(mask))
"""
from __future__ import annotations

import hashlib
from itertools import permutations

import numpy as np


# ============================================================
# 1. モンテカルロ（Gumbel-max、本番用）
# ============================================================
def race_seed(race_id: str, global_seed: int) -> int:
    """race_id と global_seed から決定論的な整数シードを作る（処理順序に非依存）。"""
    h = hashlib.blake2b(f"{global_seed}:{race_id}".encode("utf-8"), digest_size=8)
    return int.from_bytes(h.digest(), "big") % (2**31 - 1)


def mc_rank_distribution(
    scores: np.ndarray, focal_idx: int, n_draws: int, global_seed: int, race_id: str
) -> np.ndarray:
    """Gumbel-max trickによる厳密PLサンプリングのモンテカルロ推定。
    戻り値: dist[r-1] = 経験的 P(rank_focal = r) の推定値, r=1..n。 sum(dist)==1。
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    seed = race_seed(race_id, global_seed)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
    u = rng.random((n_draws, n))
    u = np.clip(u, 1e-12, 1 - 1e-12)
    gumbel = -np.log(-np.log(u))
    perturbed = scores[None, :] + gumbel
    focal_val = perturbed[:, focal_idx]
    rank = 1 + (perturbed > focal_val[:, None]).sum(axis=1)
    dist = np.bincount(rank - 1, minlength=n).astype(np.float64) / n_draws
    return dist


def mc_standard_error(dist: np.ndarray, n_draws: int) -> np.ndarray:
    """各 P(rank=r) の二項比率としての標準誤差 sqrt(p(1-p)/K)。近似誤差の上限に使う。"""
    p = dist
    return np.sqrt(np.clip(p * (1 - p), 0, None) / n_draws)


# ============================================================
# 2. 厳密 bitmask DP（中規模検証用）
# ============================================================
def exact_rank_distribution(scores: np.ndarray, focal_idx: int) -> np.ndarray:
    """bitmask動的計画法による厳密分布。O(2^(n-1) * n)。n<=18を想定(2^17*18≈2.4M)。"""
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    w = np.exp(scores - scores.max())
    w_focal = float(w[focal_idx])
    others = [i for i in range(n) if i != focal_idx]
    w_o = w[others]
    m = len(w_o)
    W_O = float(w_o.sum())

    n_masks = 1 << m
    # w_mask[mask] = そのmaskに含まれるothersの重み和 (incremental DP, O(2^m))
    w_mask = np.zeros(n_masks, dtype=np.float64)
    for mask in range(1, n_masks):
        low = mask & (-mask)
        bit = low.bit_length() - 1
        w_mask[mask] = w_mask[mask ^ low] + w_o[bit]

    dp = np.zeros(n_masks, dtype=np.float64)
    dp[0] = 1.0
    for mask in range(1, n_masks):
        total = 0.0
        rem = mask
        while rem:
            low = rem & (-rem)
            bit = low.bit_length() - 1
            prev_mask = mask ^ low
            denom = w_focal + W_O - w_mask[prev_mask]
            total += dp[prev_mask] * w_o[bit] / denom
            rem ^= low
        dp[mask] = total

    dist = np.zeros(n, dtype=np.float64)
    for mask in range(n_masks):
        r = bin(mask).count("1") + 1
        denom = w_focal + W_O - w_mask[mask]
        dist[r - 1] += dp[mask] * w_focal / denom
    return dist


# ============================================================
# 3. 全順列列挙（oracle、n<=8専用）
# ============================================================
def brute_force_rank_distribution(scores: np.ndarray, focal_idx: int) -> np.ndarray:
    """全n!順列を列挙して厳密分布を計算する。n<=8専用（検証用途のみ）。"""
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n > 8:
        raise ValueError(f"brute_force_rank_distribution: n={n} too large (n<=8のみ)")
    w = np.exp(scores - scores.max())
    total_w = w.sum()
    dist = np.zeros(n, dtype=np.float64)
    for perm in permutations(range(n)):
        # 完全な順列の確率を最後まで計算する(途中でbreakすると、同じprefixを
        # 共有する(n-r)!通りのtailに対して重複加算してしまうバグになる)。
        p = 1.0
        remaining = total_w
        for idx in perm:
            p *= w[idx] / remaining
            remaining -= w[idx]
        r = perm.index(focal_idx) + 1
        dist[r - 1] += p
    return dist


# ============================================================
# 派生量
# ============================================================
def quantile_rank(dist: np.ndarray, q: float) -> int:
    """CDF(r) >= q を満たす最小の r (1-indexed)。標準的な分位点定義。"""
    cdf = np.cumsum(dist)
    idx = np.searchsorted(cdf, q, side="left")
    return int(idx) + 1


def expected_rank(dist: np.ndarray) -> float:
    n = len(dist)
    return float(np.dot(np.arange(1, n + 1), dist))


def rank_percentile(rank: int, n: int) -> float:
    """rank(1..n) を [0,1] へ正規化。0=最良(1着), 1=最下位。n==1はNaN(定義不能)。"""
    if n <= 1:
        return float("nan")
    return (rank - 1) / (n - 1)


# ============================================================
# 4. adaptive q90 解決（モンテカルロ誤差からラベルを保護する）
# ============================================================
# JRA平地レースの出走可能最大頭数は18頭。exact DPはO(2^(n-1)*n)なので、
# n<=EXACT_DP_MAX_N は常に厳密解を使う(モンテカルロ誤差そのものを排除)。
# 2023-2025年の実データでもn<=18が確認済み(この定数を超える運用は想定されて
# いないが、コードパスとしては汎用的なフォールバック梯子を用意する)。
EXACT_DP_MAX_N = 18
# n<=DP_INFEASIBLE_NならMCで解決できなくても最終的にexact DPへfallbackする。
# 2^23 * 24 ≈ 2億(数十秒〜1分程度)を目安の上限とする。
DP_INFEASIBLE_N = 24
MC_LADDER = (50_000, 200_000, 1_000_000)


def wilson_ci(k: int, n_draws: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval（両側95%既定）。np/正規近似より小標本・極端pで安定。
    k=成功回数、n_draws=試行回数。戻り値=(lower, upper)。
    """
    if n_draws <= 0:
        return 0.0, 1.0
    phat = k / n_draws
    z2 = z * z
    denom = 1 + z2 / n_draws
    center = phat + z2 / (2 * n_draws)
    margin = z * np.sqrt(phat * (1 - phat) / n_draws + z2 / (4 * n_draws * n_draws))
    lo = (center - margin) / denom
    hi = (center + margin) / denom
    return max(0.0, lo), min(1.0, hi)


def _try_resolve_from_dist(cdf_counts: np.ndarray, n_draws: int, q: float) -> int | None:
    """cdf_counts[r-1] = rank<=rとなったdraw数(累積)。各rについて
    upper_CI(F(r-1)) < q かつ lower_CI(F(r)) >= q を満たす最小のrを返す。
    見つからなければNone。
    """
    n = len(cdf_counts)
    for r in range(1, n + 1):
        k_r_minus_1 = int(cdf_counts[r - 2]) if r >= 2 else 0
        k_r = int(cdf_counts[r - 1])
        _, hi_rm1 = wilson_ci(k_r_minus_1, n_draws)
        lo_r, _ = wilson_ci(k_r, n_draws)
        if hi_rm1 < q and lo_r >= q:
            return r
    return None


def resolve_q90_label(
    scores: np.ndarray,
    focal_idx: int,
    race_id: str,
    q: float = 0.90,
    global_seed: int = 20261010,
    exact_dp_max_n: int = EXACT_DP_MAX_N,
    dp_infeasible_n: int = DP_INFEASIBLE_N,
    mc_ladder: tuple[int, ...] = MC_LADDER,
) -> dict:
    """q90_predicted_rankをモンテカルロ誤差から保護しながら確定する。

    手順(ユーザー指定):
      1. n<=exact_dp_max_n なら常に厳密bitmask DPを使う(誤差ゼロ)。
      2. それ以外はK=50,000でモンテカルロ推定、各順位のCDFにWilson信頼区間を
         計算し、upper_CI(F(r-1))<q かつ lower_CI(F(r))>=q を満たすrが
         一意に決まる場合のみ確定する。
      3. 確定できなければKを200,000→1,000,000と増やして再試行する。
      4. それでも確定できず、かつ n<=dp_infeasible_n なら厳密DPへfallbackする
         (厳密解には信頼区間の概念が要らないので必ず確定する)。
      5. n>dp_infeasible_nで確定不能な場合のみ q90_unresolved として返す
         (このデータセットでは実際には発生しない設計上の安全弁)。

    戻り値: dict(q90_rank, method, resolved, k_draws, dist)
    dist は使用した手法で得られた順位分布(unresolvedの場合はNone、期待値等の
    派生量の再計算に再利用できる)。
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    if n <= exact_dp_max_n:
        dist = exact_rank_distribution(scores, focal_idx)
        return dict(q90_rank=quantile_rank(dist, q), method="exact_dp", resolved=True,
                    k_draws=None, dist=dist)

    for K in mc_ladder:
        dist = mc_rank_distribution(scores, focal_idx, K, global_seed, race_id)
        counts = np.round(dist * K).astype(np.int64)
        cdf_counts = np.cumsum(counts)
        r = _try_resolve_from_dist(cdf_counts, K, q)
        if r is not None:
            return dict(q90_rank=r, method=f"mc_K{K}", resolved=True, k_draws=K, dist=dist)

    if n <= dp_infeasible_n:
        dist = exact_rank_distribution(scores, focal_idx)
        return dict(q90_rank=quantile_rank(dist, q), method="exact_dp_fallback", resolved=True,
                    k_draws=None, dist=dist)

    return dict(q90_rank=None, method="unresolved", resolved=False, k_draws=mc_ladder[-1], dist=None)
