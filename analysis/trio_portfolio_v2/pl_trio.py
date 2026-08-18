# -*- coding: utf-8 -*-
"""pl_trio.py — 設計書 §3「三連複成立確率 q」

スクラッチ後の全馬について 元AI **生score** 由来の PL 強さ w_i = exp(ai_score_i) を
作り、三頭組 c={a,b,d} の成立確率を 6 順列の厳密和で出す。近似なし。

    q(c) = Σ_{(x,y,z) ∈ perm(a,b,d)}
             w_x/W * w_y/(W-w_x) * w_z/(W-w_x-w_y)         (W = Σ w)

設計書 §1 の役割分離を厳守する:
  - 確率は「元AIの生score由来 PL」だけで作る。blend 順位も市場も混ぜない。
  - §2.2「p_win の並びで AI 順位を作ることは禁止」→ 本モジュールは ai_score のみ読む。

単体テスト: python -m analysis.trio_portfolio_v2.pl_trio
"""
from __future__ import annotations

import math
from itertools import combinations, permutations

__all__ = ["pl_weights", "q_trio", "q_all_trios", "check_rank_consistency"]


def pl_weights(scores) -> list[float]:
    """生score → PL 強さ w_i = exp(s_i)。オーバーフロー回避に max を引く
    （w の定数倍は q を変えない＝順序も値も不変）。"""
    s = [float(x) for x in scores]
    if not s:
        return []
    m = max(s)
    return [math.exp(x - m) for x in s]


def q_trio(w, i: int, j: int, k: int) -> float:
    """三連複 {i,j,k} の成立確率（6 順列の厳密和）。"""
    if len({i, j, k}) < 3:
        return 0.0
    W = math.fsum(w)
    tot = 0.0
    for x, y, z in permutations((i, j, k)):
        d1 = W
        d2 = W - w[x]
        d3 = W - w[x] - w[y]
        if d2 <= 0 or d3 <= 0:
            continue
        tot += (w[x] / d1) * (w[y] / d2) * (w[z] / d3)
    return tot


def q_all_trios(w) -> dict:
    """{(i,j,k): q} を全 C(n,3) 組について返す（index は w の並び）。"""
    n = len(w)
    return {(i, j, k): q_trio(w, i, j, k)
            for i, j, k in combinations(range(n), 3)}


def check_rank_consistency(horses) -> dict:
    """§3 最終テスト: ai_rank と ai_score 降順が一致し、p_win 順位が混入していないか。

    horses: [{umaban, ai_rank, ai_score, p_win}, ...]
    返り値 {ok, n, rank_vs_score_mismatch, pwin_rank_mismatch, note}
      - rank_vs_score_mismatch > 0 なら bundle 側の ai_rank が生score順でない（要調査）
      - pwin_rank_mismatch は「p_win 順位と ai_score 順位が食い違う頭数」。
        0 でも異常ではない（較正が単調なら一致する）。混入検出は
        rank_vs_score_mismatch 側で行う。
    """
    hs = [h for h in horses
          if h.get("ai_rank") is not None and h.get("ai_score") is not None]
    by_score = sorted(hs, key=lambda h: (-float(h["ai_score"]), int(h["umaban"])))
    rank_of_score = {int(h["umaban"]): i + 1 for i, h in enumerate(by_score)}
    mism = sum(1 for h in hs if int(h["ai_rank"]) != rank_of_score[int(h["umaban"])])
    hp = [h for h in hs if h.get("p_win") is not None]
    by_p = sorted(hp, key=lambda h: (-float(h["p_win"]), int(h["umaban"])))
    rank_of_p = {int(h["umaban"]): i + 1 for i, h in enumerate(by_p)}
    pmism = sum(1 for h in hp if rank_of_p[int(h["umaban"])] != rank_of_score[int(h["umaban"])])
    return {"ok": mism == 0, "n": len(hs), "rank_vs_score_mismatch": mism,
            "pwin_rank_mismatch": pmism}


# ============================================================
# 単体テスト（設計書 §3 の必須項目）
# ============================================================
def _t_sum_to_one():
    import random
    random.seed(7)
    for n in (3, 4, 6, 10, 18):
        w = [math.exp(random.uniform(-3, 3)) for _ in range(n)]
        s = math.fsum(q_all_trios(w).values())
        assert abs(s - 1.0) < 1e-9, (n, s)
    return "全 C(n,3) の q 総和 = 1 (n=3,4,6,10,18, 誤差<1e-9)"


def _t_permutation_invariant():
    w = [3.0, 1.0, 2.0, 0.5, 1.7]
    base = q_trio(w, 0, 2, 4)
    for x, y, z in permutations((0, 2, 4)):
        assert abs(q_trio(w, x, y, z) - base) < 1e-15
    return "馬番の並べ替えで q 不変"


def _t_hand_examples():
    # 3頭: 唯一の組は必ず成立 → q = 1
    assert abs(q_trio([1.0, 1.0, 1.0], 0, 1, 2) - 1.0) < 1e-12
    assert abs(q_trio([5.0, 1.0, 0.2], 0, 1, 2) - 1.0) < 1e-12
    # 4頭均等: 対称性より各組 1/4
    w = [1.0] * 4
    for c in combinations(range(4), 3):
        assert abs(q_trio(w, *c) - 0.25) < 1e-12
    # 4頭 w=[2,1,1,1]: 0番を含まない組 = P(0番が4着)
    #   = 6 順列 × (1/5)(1/4)(1/3) = 0.1、残り3組は対称で各 0.3
    w = [2.0, 1.0, 1.0, 1.0]
    assert abs(q_trio(w, 1, 2, 3) - 0.1) < 1e-12
    for c in [(0, 1, 2), (0, 1, 3), (0, 2, 3)]:
        assert abs(q_trio(w, *c) - 0.3) < 1e-12
    return "手計算例 (3頭=1 / 4頭均等=1/4 / 4頭[2,1,1,1]=0.1,0.3) と一致"


def _t_scratch_renormalize():
    # 取消馬を除いた集合で再正規化されること（除外前の q を按分しない）
    w_full = [4.0, 2.0, 1.0, 1.0, 0.5]
    keep = [0, 1, 2, 4]                      # index3 が取消
    w_cut = [w_full[i] for i in keep]
    assert abs(math.fsum(q_all_trios(w_cut).values()) - 1.0) < 1e-12
    # 取消馬を含む組は候補集合に存在しない
    assert len(q_all_trios(w_cut)) == 4
    # 取消前の「その組」の値より必ず大きい（母数が減るため）
    before = q_trio(w_full, 0, 1, 2)
    after = q_trio(w_cut, 0, 1, 2)
    assert after > before
    return "取消後も再正規化される（総和1・組数C(4,3)・値は増加）"


def _t_weight_scale_invariance():
    w = [2.0, 1.0, 0.5, 0.25]
    a = q_all_trios(w)
    b = q_all_trios([x * 1234.5 for x in w])
    assert all(abs(a[k] - b[k]) < 1e-12 for k in a)
    return "w の定数倍で q 不変 (exp(s-max) の実装が安全)"


def _t_against_pl_probs():
    """既存 pl_probs.p_sanrenpuku（本番系の独立実装）と一致するか。"""
    try:
        import numpy as np
        import pl_probs as PL
    except Exception as e:                      # 依存が無い環境ではスキップ
        return f"pl_probs 突合 SKIP ({type(e).__name__})"
    import random
    random.seed(11)
    s = [random.uniform(-3, 3) for _ in range(9)]
    w1 = pl_weights(s)
    w2 = PL.pl_weights(np.array(s))
    for c in combinations(range(9), 3):
        assert abs(q_trio(w1, *c) - PL.p_sanrenpuku(w2, *c)) < 1e-12
    return "既存 pl_probs.p_sanrenpuku と全 C(9,3) 組で一致 (<1e-12)"


TESTS = [_t_sum_to_one, _t_permutation_invariant, _t_hand_examples,
         _t_scratch_renormalize, _t_weight_scale_invariance, _t_against_pl_probs]


def run_tests() -> list[dict]:
    out = []
    for t in TESTS:
        try:
            out.append({"test": t.__name__, "ok": True, "detail": t()})
        except Exception as e:
            out.append({"test": t.__name__, "ok": False,
                        "detail": f"{type(e).__name__}: {e}"})
    return out


if __name__ == "__main__":
    for r in run_tests():
        print(("  OK  " if r["ok"] else "  NG  ") + r["test"] + " — " + r["detail"])
