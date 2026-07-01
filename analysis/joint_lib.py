# -*- coding: utf-8 -*-
"""
joint_lib.py — 市場内部(Harville/Stern)裁定の共有ライブラリ。
単勝marginal π から連結(馬連)公正確率を作る複数モデル + EVゲート決済ハーネス。
全て leak-safe: 決定=9時odds、決済=確定odds/払戻、λ等の選定は≤2023のみ。
"""
import numpy as np, pandas as pd
from pathlib import Path
from itertools import combinations

OUT = Path("E:/PyCaLiAI/data/_joint")
UM_TAKEOUT_FLOOR = 0.775   # 馬連 控除後ライン
FUKU_TAKEOUT_FLOOR = 0.80


def load():
    H = pd.read_parquet(OUT / "mkt_horses.parquet")
    P = pd.read_parquet(OUT / "mkt_umaren.parquet")
    return H, P


# ---------- 連結公正確率: 単勝marginal π → 馬連ペア確率 ----------
def harville_pairs(pi, bans, lam=1.0):
    """Discount-Harville: P(i 1st)=π_i; P(j 2nd|i)=π_j^λ/Σ_{k≠i}π_k^λ。
    返り値: dict {(min,max ban): P_unordered_top2}。lam=1.0=純Harville。"""
    pi = np.asarray(pi, float); n = len(pi)
    s = np.power(np.clip(pi, 1e-12, 1), lam)
    out = {}
    for a in range(n):
        for b in range(a + 1, n):
            # i=a 1st, j=b 2nd : π_a * s_b/(Σs - s_a)
            denom_a = s.sum() - s[a]
            denom_b = s.sum() - s[b]
            p_ab = pi[a] * (s[b] / denom_a if denom_a > 0 else 0)
            p_ba = pi[b] * (s[a] / denom_b if denom_b > 0 else 0)
            key = (min(bans[a], bans[b]), max(bans[a], bans[b]))
            out[key] = p_ab + p_ba
    return out


def fit_lambda(H, years, grid=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2)):
    """≤years で realized top2 の馬連pair logloss を最小化する λ を選ぶ。"""
    races = _race_pack(H, lambda y: y <= years)
    best = (None, 1e18)
    curve = []
    for lam in grid:
        ll = _pair_logloss(races, "harville", lam)
        curve.append((lam, round(ll, 5)))
        if ll < best[1]: best = (lam, ll)
    return best[0], curve


def _race_pack(H, yr_filter):
    """per-race: bans, pi, winpair(min,max)。馬連評価用。"""
    races = []
    for rid, g in H.groupby("rid", sort=False):
        y = int(g.year.iloc[0])
        if not yr_filter(y): continue
        bans = g.ban.values.astype(int); pi = g.pi.values.astype(float)
        wb = g.ban[g.is_win == 1]; pb = g.ban[g.is_plc == 1]
        if len(wb) == 0 or len(pb) == 0: continue
        w = int(wb.iloc[0]); p = int(pb.iloc[0])
        races.append({"rid": rid, "year": y, "bans": bans, "pi": pi,
                      "winpair": (min(w, p), max(w, p))})
    return races


def _pair_logloss(races, model, lam):
    tot = 0.0; n = 0
    for r in races:
        if model == "harville":
            pp = harville_pairs(r["pi"], r["bans"], lam)
        ps = np.array(list(pp.values())); ps = ps / ps.sum()
        wk = r["winpair"]
        pw = pp.get(wk, 1e-12) / max(sum(pp.values()), 1e-12)
        tot += -np.log(max(pw, 1e-12)); n += 1
    return tot / max(n, 1)


_RNG = np.random.default_rng(42)


def pl_topk_probs(pi, k, K=4000, rng=None):
    """Plackett-Luce: Gumbel-max で K本サンプル → 各馬の top-k 包含率 (Σ≈k)。
    複勝(k=3, n>=8 / k=2, n<=7)・ワイド基盤に使う。検証済み実装。"""
    r = rng or _RNG
    pi = np.asarray(pi, float); n = len(pi)
    logits = np.log(np.clip(pi, 1e-12, 1))
    g = r.gumbel(size=(K, n)) + logits
    order = np.argsort(-g, axis=1)[:, :k]
    cnt = np.zeros(n)
    for c in range(k):
        np.add.at(cnt, order[:, c], 1)
    return cnt / K


def pl_pair_both_topk(pi, bans, k=3, K=4000, rng=None):
    """ワイド公正: P(i と j が両方 top-k)。dict {(min,max ban): prob}。"""
    r = rng or _RNG
    pi = np.asarray(pi, float); n = len(pi)
    logits = np.log(np.clip(pi, 1e-12, 1))
    g = r.gumbel(size=(K, n)) + logits
    order = np.argsort(-g, axis=1)[:, :k]
    intop = np.zeros((K, n), bool)
    for c in range(k):
        intop[np.arange(K), order[:, c]] = True
    out = {}
    for a in range(n):
        for b in range(a + 1, n):
            p = float(np.mean(intop[:, a] & intop[:, b]))
            out[(min(bans[a], bans[b]), max(bans[a], bans[b]))] = p
    return out


def market_pair_devig(P_race):
    """1レースの馬連odds行 → de-vig q_ij dict。"""
    q = {}
    inv = 1.0 / P_race["o_um9"].values
    s = inv.sum()
    for (i, j), v in zip(zip(P_race.i, P_race.j), inv / s):
        q[(int(i), int(j))] = float(v)
    return q
