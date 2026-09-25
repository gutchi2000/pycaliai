# -*- coding: utf-8 -*-
"""
payout_formula.py — EXP18 S0-C: JRA 複勝オッズ (Lo/Hi) の式と逆算
=================================================================
式は TOMOGRAPHY_IDENTIFIABILITY_AUDIT.md §2 に**実行前に固定**したものをそのまま実装する。

  純プール      D = (1 − 0.20) · V                      (V = 複勝プール総票数, 控除率 20%)
  的中票控除    元本返還分 Σ_{j∈W} V_j を D から差し引く (W = 払戻対象の k 頭)
  分配          残りを k 頭で等分: 1 票あたり利益 = (D − Σ_{j∈W} V_j) / (k · V_i)
  オッズ        odds_i(W) = 1 + (D − Σ_{j∈W} V_j) / (k · V_i)
  表示 Lo       他の k−1 頭が票数上位 (最も人気) のとき = 最小
  表示 Hi       他の k−1 頭が票数下位 (最も不人気) のとき = 最大
  丸め          表示は 0.1 倍単位の切り捨て (払戻 10 円単位の breakage と同値)、下限 1.0 (元返し)
  k             places(n): 出走 8 頭以上 3、5〜7 頭 2 (4 頭以下は複勝発売なし)
  取消・返還    締切前取消は pool に居ない (starter 集合から除外)。締切後の返還は表示に現れない
シェア v_i = V_i / V で書けば式は票数の尺度に依存しない (0 次同次)。
"""
from __future__ import annotations

import numpy as np

TAKEOUT_PLACE = 0.20
FLOOR_ODDS = 1.0
EPS_FLOOR = 1e-9


def places(n: int) -> int:
    return 3 if n >= 8 else (2 if n >= 5 else 0)


def display_floor(raw: np.ndarray) -> np.ndarray:
    """0.1 倍単位の切り捨てと下限 1.0"""
    return np.maximum(FLOOR_ODDS, np.floor(np.asarray(raw) * 10.0 + EPS_FLOOR) / 10.0)


def raw_lo_hi(v: np.ndarray, k: int, takeout: float = TAKEOUT_PLACE):
    """シェア v (Σ=1) から丸め前の Lo / Hi を返す"""
    v = np.asarray(v, dtype=float)
    n = len(v)
    D = 1.0 - takeout
    lo = np.empty(n)
    hi = np.empty(n)
    for i in range(n):
        others = np.delete(v, i)
        srt = np.sort(others)
        top = srt[::-1][:k - 1].sum()
        bot = srt[:k - 1].sum()
        lo[i] = 1.0 + (D - v[i] - top) / (k * v[i])
        hi[i] = 1.0 + (D - v[i] - bot) / (k * v[i])
    return lo, hi


def display_lo_hi(v: np.ndarray, k: int, takeout: float = TAKEOUT_PLACE):
    lo, hi = raw_lo_hi(v, k, takeout)
    return display_floor(lo), display_floor(hi)


def invert_display(lo_disp: np.ndarray, hi_disp: np.ndarray, k: int,
                   takeout: float = TAKEOUT_PLACE):
    """表示 Lo/Hi から票シェア v を逆算する数値解法 (式そのものは上で固定)。
    人気順 (表示 Lo、同値は Hi の昇順) を固定すると、各馬の「他の上位 k−1 頭」「他の下位 k−1 頭」
    の集合が決まり、丸め前オッズ r_i = ((1−t) − C_i(v)) / v_i ... の切り捨て区間条件
        v_i·(k(L−1)+1) <= (1−t) − C_i(v) < v_i·(k(U−1)+1)     ([L, U) = [表示, 表示+0.1))
    は v について**線形不等式**になる (表示 1.0 は下側を課さない)。Σv=1・v>0・人気順の単調性を加え、
    全不等式の最小余裕 s を最大化する LP (HiGHS) を解いて内点を取る。"""
    from itertools import permutations, product
    lo = np.asarray(lo_disp, dtype=float)
    hi = np.asarray(hi_disp, dtype=float)
    n = len(lo)
    base = list(np.lexsort((hi, lo)))            # 人気順 (最も人気 = 最小オッズが先頭)
    # 表示が完全に同じ馬 (tie) の並びは表示から決まらないので、tie 群の中の順列を候補として試す
    groups, cur = [], [base[0]]
    for a, c in zip(base[:-1], base[1:]):
        if (lo[a], hi[a]) == (lo[c], hi[c]):
            cur.append(c)
        else:
            groups.append(cur)
            cur = [c]
    groups.append(cur)
    cands = product(*[list(permutations(g)) for g in groups])
    best, best_ok = None, -1
    for t_i, cand in enumerate(cands):
        if t_i >= 64:
            break
        order = np.array([x for g in cand for x in g])
        v = _lp_for_order(lo, hi, k, takeout, order)
        if v is None:
            continue
        l2, h2 = display_lo_hi(v, k, takeout)
        nok = int((roundtrip_ok(lo, l2) & roundtrip_ok(hi, h2)).sum())
        if nok > best_ok:
            best, best_ok = v, nok
        if nok == n:
            break
    if best is None:
        return _invert_gauss_seidel(lo, hi, k, takeout)
    return best


def _lp_for_order(lo, hi, k, takeout, order):
    """人気順 order を固定したときの LP (最小余裕最大化)。解けなければ None"""
    from scipy.optimize import linprog
    n = len(lo)
    t = takeout
    A, b = [], []

    def add(row_v, rhs, s_coef):
        A.append(np.concatenate([row_v, [s_coef]]))
        b.append(rhs)

    for i in range(n):
        others = [j for j in order if j != i]
        for disp, cset in ((lo[i], others[:k - 1]), (hi[i], others[::-1][:k - 1])):
            U = disp + 0.1
            coefU = k * (U - 1.0) + 1.0
            # (1−t) − Σ_{C} v − coefU·v_i <= −s   →   −Σ_C v − coefU v_i + s <= −(1−t)
            row = np.zeros(n)
            row[list(cset)] -= 1.0
            row[i] -= coefU
            add(row, -(1.0 - t), 1.0)
            if disp > FLOOR_ODDS:
                coefL = k * (disp - 1.0) + 1.0
                # coefL·v_i − (1−t) + Σ_C v <= −s   →   Σ_C v + coefL v_i + s <= (1−t)
                row = np.zeros(n)
                row[list(cset)] += 1.0
                row[i] += coefL
                add(row, (1.0 - t), 1.0)
    for a, c in zip(order[:-1], order[1:]):     # 人気順の単調性
        row = np.zeros(n)
        row[c] += 1.0
        row[a] -= 1.0
        add(row, 0.0, 0.0)
    A_eq = np.concatenate([np.ones(n), [0.0]])[None, :]
    res = linprog(c=np.concatenate([np.zeros(n), [-1.0]]), A_ub=np.array(A), b_ub=np.array(b),
                  A_eq=A_eq, b_eq=[1.0], bounds=[(1e-9, 1.0)] * n + [(None, 1.0)], method="highs")
    if res.status == 0:
        v = np.clip(res.x[:n], 1e-12, None)
        return v / v.sum()
    return None


def _invert_gauss_seidel(lo_disp: np.ndarray, hi_disp: np.ndarray, k: int,
                         takeout: float = TAKEOUT_PLACE, sweeps: int = 300):
    """表示 Lo/Hi から票 V (尺度は任意) を逆算する数値解法 (式そのものは上で固定)。
    票 V_i と他馬の票から、式は V_i について閉じた形で解ける:
        丸め前オッズ r に対し  V_i = ((1−t)·O_i − C_i) / (k·(r − 1) + t)
        (O_i = 他馬の票の和, C_i = Lo なら他馬上位 k−1 頭 / Hi なら下位 k−1 頭の票の和)
    r は切り捨て区間 [表示, 表示+0.1) (表示 1.0 は (−∞, 1.1)) にあるので、V_i の許容区間が
    Lo 側・Hi 側で一つずつ得られる。両区間の共通部分の中点 (log 尺度) へ V_i を更新する
    Gauss-Seidel を収束まで繰り返す。共通部分が空なら二区間の境界の中点を使う。"""
    lo = np.asarray(lo_disp, dtype=float)
    hi = np.asarray(hi_disp, dtype=float)
    n = len(lo)
    t = takeout
    V = 1.0 / ((lo + hi) / 2.0 + 0.05)
    V = V / V.sum()

    def interval(disp, Oi, Ci):
        num = (1.0 - t) * Oi - Ci
        if num <= 0:
            return 1e-12, 1e-12
        r_lo = disp if disp > FLOOR_ODDS else 0.9          # 表示 1.0 は下側が開いている
        r_hi = disp + 0.1
        a = num / (k * (r_hi - 1.0) + t)                   # r が大きいほど V_i は小さい
        b = num / max(k * (r_lo - 1.0) + t, 1e-12)
        return a, b

    for _ in range(sweeps):
        maxrel = 0.0
        for i in range(n):
            others = np.delete(V, i)
            srt = np.sort(others)
            Oi = others.sum()
            top, bot = srt[::-1][:k - 1].sum(), srt[:k - 1].sum()
            a1, b1 = interval(lo[i], Oi, top)
            a2, b2 = interval(hi[i], Oi, bot)
            lo_b, hi_b = max(a1, a2), min(b1, b2)
            if lo_b <= hi_b:
                new = np.sqrt(lo_b * hi_b)
            else:
                new = np.sqrt(min(b1, b2) * max(a1, a2))
            maxrel = max(maxrel, abs(new - V[i]) / V[i])
            V[i] = new
        V = V / V.sum()
        if maxrel < 1e-12:
            break
    return V


def tau_from_shares(v: np.ndarray, k: int) -> np.ndarray:
    """T2 の複勝 soft 制約の目標 τ_i = k · v_i (宣言した定義。Σ τ = k を構成的に満たす)"""
    return k * np.asarray(v, dtype=float)


def roundtrip_ok(disp: np.ndarray, regen: np.ndarray) -> np.ndarray:
    """|再生成 − 表示| <= max(0.05, 0.02 × 表示)"""
    disp = np.asarray(disp, dtype=float)
    return np.abs(np.asarray(regen) - disp) <= np.maximum(0.05, 0.02 * disp) + 1e-9
