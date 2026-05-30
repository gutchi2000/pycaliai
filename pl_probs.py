"""
pl_probs.py
===========
Plackett-Luce による厳密 joint 確率計算エンジン。

unified_rank_v1 等のランキングスコア s_i から w_i = exp(s_i) を作り、
全馬券種の厳密 joint 確率を閉形式で返す。近似一切なし。

PL 定義:
    P(順列 (h1,...,hk) が上位 k 着) =
        Π_{m=1..k}  w_{hm} / ( Σ w - Σ_{j<m} w_{hj} )

単体テスト: python pl_probs.py
"""
from __future__ import annotations

from itertools import permutations

import numpy as np


# ============================================================
# 重み
# ============================================================
def pl_weights(scores: np.ndarray) -> np.ndarray:
    """スコア → PL 重み w_i = exp(s_i). オーバーフロー回避のため max を引く."""
    s = np.asarray(scores, dtype=np.float64)
    return np.exp(s - s.max())


# ============================================================
# 基本馬券 (O(1) or O(N))
# ============================================================
def p_tansho(w: np.ndarray, i: int) -> float:
    """単勝: i が 1 着."""
    return float(w[i] / w.sum())


def p_umatan(w: np.ndarray, i: int, j: int) -> float:
    """馬単: i → j."""
    if i == j:
        return 0.0
    total = w.sum()
    return float(w[i] / total * w[j] / (total - w[i]))


def p_sanrentan(w: np.ndarray, i: int, j: int, k: int) -> float:
    """三連単: i → j → k."""
    if len({i, j, k}) < 3:
        return 0.0
    total = w.sum()
    p1 = w[i] / total
    p2 = w[j] / (total - w[i])
    p3 = w[k] / (total - w[i] - w[j])
    return float(p1 * p2 * p3)


def p_umaren(w: np.ndarray, i: int, j: int) -> float:
    """馬連: {i,j} が top2."""
    return p_umatan(w, i, j) + p_umatan(w, j, i)


def p_sanrenpuku(w: np.ndarray, i: int, j: int, k: int) -> float:
    """三連複: {i,j,k} が top3. 6 順列和."""
    if len({i, j, k}) < 3:
        return 0.0
    return sum(p_sanrentan(w, a, b, c) for a, b, c in permutations((i, j, k)))


# ============================================================
# 複勝 (i が top3 のいずれか)
# ============================================================
def p_place_at(w: np.ndarray, i: int, pos: int) -> float:
    """i がちょうど pos 着 (1-indexed)."""
    total = w.sum()
    n = len(w)
    if pos == 1:
        return float(w[i] / total)
    if pos == 2:
        s = 0.0
        for j in range(n):
            if j == i:
                continue
            s += w[j] / total * w[i] / (total - w[j])
        return float(s)
    if pos == 3:
        s = 0.0
        for j in range(n):
            if j == i: continue
            pj = w[j] / total
            for k in range(n):
                if k == i or k == j: continue
                pk = w[k] / (total - w[j])
                pi = w[i] / (total - w[j] - w[k])
                s += pj * pk * pi
        return float(s)
    raise ValueError("pos must be 1, 2, or 3")


def p_fukusho(w: np.ndarray, i: int) -> float:
    """複勝: i ∈ top3."""
    return p_place_at(w, i, 1) + p_place_at(w, i, 2) + p_place_at(w, i, 3)


# ============================================================
# ワイド ({i,j} ⊂ top3)
# ============================================================
def p_wide(w: np.ndarray, i: int, j: int) -> float:
    """ワイド: {i,j} が top3 のどこか 2 席."""
    if i == j:
        return 0.0
    n = len(w)
    s = 0.0
    for k in range(n):
        if k == i or k == j:
            continue
        s += p_sanrenpuku(w, i, j, k)
    return float(s)


# ============================================================
# ベクトル版 (レース全体を一括計算)
# ============================================================
def all_tansho(w: np.ndarray) -> np.ndarray:
    """各馬の単勝確率ベクトル."""
    return w / w.sum()


def all_fukusho(w: np.ndarray) -> np.ndarray:
    """各馬の複勝確率ベクトル. O(N^3) だが N<=18 なら余裕."""
    n = len(w)
    return np.array([p_fukusho(w, i) for i in range(n)])


def all_umaren(w: np.ndarray) -> dict[tuple[int, int], float]:
    """全ペアの馬連確率. key=(min,max)."""
    n = len(w)
    out = {}
    for i in range(n):
        for j in range(i + 1, n):
            out[(i, j)] = p_umaren(w, i, j)
    return out


def all_wide(w: np.ndarray) -> dict[tuple[int, int], float]:
    n = len(w)
    out = {}
    for i in range(n):
        for j in range(i + 1, n):
            out[(i, j)] = p_wide(w, i, j)
    return out


def all_sanrenpuku(w: np.ndarray) -> dict[tuple[int, int, int], float]:
    n = len(w)
    out = {}
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                out[(i, j, k)] = p_sanrenpuku(w, i, j, k)
    return out


# ============================================================
# 単体テスト
# ============================================================
def _test():
    rng = np.random.default_rng(42)

    for n in (5, 10, 16):
        print(f"\n=== N={n} ===")
        scores = rng.normal(size=n)
        w = pl_weights(scores)

        # 1. 単勝の合計 = 1
        tansho = all_tansho(w)
        assert abs(tansho.sum() - 1.0) < 1e-10, f"Σ単勝 = {tansho.sum()}"
        print(f"  Σ単勝 = {tansho.sum():.12f}  OK")

        # 2. 各位置の確率合計 = 1
        for pos in (1, 2, 3):
            s = sum(p_place_at(w, i, pos) for i in range(n))
            assert abs(s - 1.0) < 1e-10, f"Σ P(pos={pos}) = {s}"
            print(f"  Σ P(着={pos}) = {s:.12f}  OK")

        # 3. 複勝合計 = 3 (各馬の top3 確率合計 = 3 席 × P=1)
        fuku = all_fukusho(w)
        assert abs(fuku.sum() - 3.0) < 1e-9, f"Σ複勝 = {fuku.sum()}"
        print(f"  Σ複勝 = {fuku.sum():.12f}  (= 3.0 OK)")

        # 4. 馬連合計 = 1
        umaren = all_umaren(w)
        s = sum(umaren.values())
        assert abs(s - 1.0) < 1e-9, f"Σ馬連 = {s}"
        print(f"  Σ馬連 = {s:.12f}  OK")

        # 5. 三連複合計 = 1
        s3f = all_sanrenpuku(w)
        s = sum(s3f.values())
        assert abs(s - 1.0) < 1e-9, f"Σ三連複 = {s}"
        print(f"  Σ三連複 = {s:.12f}  OK")

        # 6. ワイド合計 = 3 (top3 から 2 頭選ぶ = C(3,2) = 3 ペア × P=1)
        wide = all_wide(w)
        s = sum(wide.values())
        assert abs(s - 3.0) < 1e-9, f"Σワイド = {s}"
        print(f"  Σワイド = {s:.12f}  (= 3.0 OK)")

        # 7. 三連単合計 = 1 (全順列)
        s = 0.0
        for i, j, k in permutations(range(n), 3):
            s += p_sanrentan(w, i, j, k)
        assert abs(s - 1.0) < 1e-9, f"Σ三連単 = {s}"
        print(f"  Σ三連単 = {s:.12f}  OK")

        # 8. 整合性: 複勝(i) == Σ_j≠i ワイド(i,j) / 2 ? いや違う.
        #    各 i について: Σ_{j≠i} wide(i,j) = 2 * fukusho(i)
        #    (i が top3 のとき他 2 席との pair が 2 個)
        for i in range(n):
            pair_sum = sum(umaren.get((min(i, j), max(i, j)), 0) + 0 for j in range(n) if j != i)
            # Monte Carlo 的整合性: Σ_j wide(i,j) = 2*fuku(i)
            w_pair = sum(wide.get((min(i, j), max(i, j)), 0) for j in range(n) if j != i)
            assert abs(w_pair - 2 * fuku[i]) < 1e-9, \
                f"馬 {i}: Σwide={w_pair}, 2*fuku={2*fuku[i]}"
        print(f"  整合性 Σ_j wide(i,j) == 2*fuku(i)  OK")

        # 9. 馬単 Σ = 1
        s = 0.0
        for i in range(n):
            for j in range(n):
                if i != j:
                    s += p_umatan(w, i, j)
        assert abs(s - 1.0) < 1e-9
        print(f"  Σ馬単 = {s:.12f}  OK")

    print("\n全テスト通過. PL 厳密計算 OK.")


if __name__ == "__main__":
    _test()
