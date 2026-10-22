# -*- coding: utf-8 -*-
"""
evaluate.py — 方策の評価（仕様 §4 成績の数え方・§7 の指標）
============================================================
入力は「買い目帳」book: 1 行 = 1 券（rid16, date, bet_type, combo, stake[円], payout100[100円あたり払戻], p_model, lpq）
と、評価対象レースの母集団 universe（rid16, date）。見送りレースは book に行が無い。

- 回収率 = Σ払戻 / Σ賭け金（stake は 100 円単位に丸め済み）
- 95% CI = 開催日ブロック bootstrap（日付ごとに賭け金・払戻を束ね、日付を復元抽出）。差は同じ抽出で対にする
- 上位 10 払戻除外 = 払戻額（円）の大きい 10 券の払戻を 0 にした回収率
- 対数成長 = レースごとの log(1 + κ·Σ (stake/B)·(R−1)) の全レース平均（見送り = 0）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.bet_decision_ai.data import KAPPA, NSLOT, SLOT_BET_TYPE

REPS = 4000


def round_stakes(s: np.ndarray, B: int, unit: int = 100) -> np.ndarray:
    """割合 s[R,S]（Σ≤1）→ 円。各券を 100 円単位に切り捨て（端数はすべて切り捨てなので券の順序に依らない）、
    余りは見送り。仕様 §4「端数は確率の高い券から切り捨て、余りは見送り」の実装（全券切り捨て）。"""
    s = np.clip(s, 0.0, None)
    tot = s.sum(-1, keepdims=True)
    s = np.where(tot > 1.0, s / np.maximum(tot, 1e-12), s)
    return (np.floor(s * B / unit + 1e-9) * unit).astype(np.int64)


def book_from_tensor(T, stakes_yen: np.ndarray, name: str = "") -> pd.DataFrame:
    r, c = np.nonzero(stakes_yen > 0)
    return pd.DataFrame({"rid16": T.races[r], "date": T.dates[r], "bet_type": SLOT_BET_TYPE[c],
                         "combo": T.combo[r, c], "stake": stakes_yen[r, c], "payout100": T.ret[r, c] * 100.0,
                         "p_model": T.p[r, c], "lpq": T.lpq[r, c], "policy": name})


def _day_arrays(book: pd.DataFrame, universe: pd.DataFrame, B: int):
    days = np.sort(universe["date"].astype(str).unique())
    di = {d: i for i, d in enumerate(days)}
    nd = len(days)
    st = np.zeros(nd)
    pay = np.zeros(nd)
    if len(book):
        bi = book["date"].astype(str).map(di).to_numpy()
        np.add.at(st, bi, book["stake"].to_numpy(float))
        np.add.at(pay, bi, (book["stake"] * book["payout100"] / 100.0).to_numpy(float))
    # 対数成長（レース単位 → 日付で合計）
    g = race_log_growth(book, universe, B)
    gd = np.zeros(nd)
    nr = np.zeros(nd)
    ui = universe["date"].astype(str).map(di).to_numpy()
    np.add.at(gd, ui, g)
    np.add.at(nr, ui, 1.0)
    return days, st, pay, gd, nr


def race_log_growth(book: pd.DataFrame, universe: pd.DataFrame, B: int) -> np.ndarray:
    if len(book):
        x = book.assign(v=book["stake"] / B * (book["payout100"] / 100.0 - 1.0)).groupby("rid16")["v"].sum()
        v = universe["rid16"].map(x).fillna(0.0).to_numpy(float)
    else:
        v = np.zeros(len(universe))
    return np.log1p(KAPPA * v)


def boot_counts(nd: int, reps: int = REPS, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multinomial(nd, np.full(nd, 1.0 / nd), size=reps).astype(np.float64)


def metrics(book: pd.DataFrame, universe: pd.DataFrame, B: int, W: np.ndarray | None = None) -> dict:
    days, st, pay, gd, nr = _day_arrays(book, universe, B)
    if W is None:
        W = boot_counts(len(days))
    S, P = st.sum(), pay.sum()
    roi = P / S if S > 0 else np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        rb = (W @ pay) / (W @ st)
    gb = (W @ gd) / (W @ nr)
    out = {"B": B, "n_races": int(len(universe)), "n_bet_races": int(book["rid16"].nunique()) if len(book) else 0,
           "n_tickets": int(len(book)), "stake": float(S), "return": float(P),
           "roi": float(roi) if S > 0 else None,
           "roi_ci": [float(np.nanpercentile(rb, 2.5)), float(np.nanpercentile(rb, 97.5))] if S > 0 else None,
           "log_growth": float(gd.sum() / nr.sum()),
           "log_growth_ci": [float(np.percentile(gb, 2.5)), float(np.percentile(gb, 97.5))]}
    if len(book):
        ret = (book["stake"] * book["payout100"] / 100.0).to_numpy(float)
        top = np.sort(ret)[::-1][:10]
        out["roi_ex_top10"] = float((P - top.sum()) / S)
        out["top10_returns"] = [float(x) for x in top]
        hit = book.assign(h=book["payout100"] > 0).groupby("rid16")["h"].any()
        out["race_hit_rate"] = float(hit.mean())
        bt = book.assign(ret=ret).groupby("bet_type").agg(stake=("stake", "sum"), ret=("ret", "sum"),
                                                          n=("stake", "size"))
        out["by_bet_type"] = {k: {"stake_share": float(r.stake / S), "stake": float(r.stake), "n_tickets": int(r.n),
                                  "roi": float(r.ret / r.stake)} for k, r in bt.iterrows()}
        out["mean_stake_per_bet_race"] = float(S / out["n_bet_races"])
    return out


def diff(bookA: pd.DataFrame, bookB: pd.DataFrame, universe: pd.DataFrame, B: int,
         W: np.ndarray | None = None) -> dict:
    """A − B の回収率差・対数成長差（同じ日付抽出で対にする）。"""
    days, sa, pa, ga, nr = _day_arrays(bookA, universe, B)
    _, sb, pb, gb_, _ = _day_arrays(bookB, universe, B)
    if W is None:
        W = boot_counts(len(days))
    with np.errstate(invalid="ignore", divide="ignore"):
        d = (W @ pa) / (W @ sa) - (W @ pb) / (W @ sb)
    dg = (W @ (ga - gb_)) / (W @ nr)
    pt = (pa.sum() / sa.sum() if sa.sum() > 0 else np.nan) - (pb.sum() / sb.sum() if sb.sum() > 0 else np.nan)
    return {"roi_diff": float(pt), "roi_diff_ci": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))],
            "log_growth_diff": float((ga - gb_).sum() / nr.sum()),
            "log_growth_diff_ci": [float(np.percentile(dg, 2.5)), float(np.percentile(dg, 97.5))]}


def quintile_edges(cand: pd.DataFrame) -> dict:
    """券種ごとの log_p_over_q 五分位の境界（学習期間 2017-2020 の候補全体で決める）。"""
    return {bt: np.quantile(g["log_p_over_q"].to_numpy(float), [0.2, 0.4, 0.6, 0.8]).tolist()
            for bt, g in cand.groupby("bet_type")}


def quintile_table(book: pd.DataFrame, edges: dict, universe: pd.DataFrame, B: int,
                   W: np.ndarray | None = None) -> list[dict]:
    """乖離五分位（券種ごとの境界で Q1=市場よりモデルが低い … Q5=モデルが市場より高い）ごとの賭け金・回収率・CI。"""
    if not len(book):
        return []
    b = book[book["bet_type"].isin(edges)].copy()
    q = np.zeros(len(b), int)
    for bt, e in edges.items():
        m = (b["bet_type"] == bt).to_numpy()
        q[m] = np.searchsorted(np.array(e), b.loc[m, "lpq"].to_numpy(float), side="right") + 1
    b["q"] = q
    days = np.sort(universe["date"].astype(str).unique())
    if W is None:
        W = boot_counts(len(days))
    out = []
    S = b["stake"].sum()
    for k in range(1, 6):
        g = b[b["q"] == k]
        if not len(g):
            out.append({"quintile": k, "stake_share": 0.0, "n_tickets": 0})
            continue
        m = metrics(g.drop(columns="q"), universe, B, W)
        out.append({"quintile": k, "stake_share": float(g["stake"].sum() / S), "n_tickets": int(len(g)),
                    "n_bet_races": m["n_bet_races"], "roi": m["roi"], "roi_ci": m["roi_ci"],
                    "roi_ex_top10": m.get("roi_ex_top10")})
    return out


# ------------------------------------------------------------------ 期間まるごとの評価（train.py と confirm.py で共用）
def evaluate_period(T, fracs: dict, baseline_books: dict, universe: pd.DataFrame, B: int, edges: dict,
                    mains: list[str], compare_to: list[str]) -> dict:
    """fracs: {方策名: 割合[R,S]}（判断 AI・GBM・B4）、baseline_books: {名前: book（予算 B 用）}。
    mains の各方策と compare_to の各方策の回収率差を同じ日付抽出で対比較する。"""
    W = boot_counts(universe["date"].astype(str).nunique())
    books = dict(baseline_books)
    for k, s in fracs.items():
        books[k] = book_from_tensor(T, round_stakes(s, B), k)
    res = {"B": B, "policies": {}, "diff_vs": {}, "quintiles": {}}
    for k, b in books.items():
        res["policies"][k] = metrics(b, universe, B, W)
        res["quintiles"][k] = quintile_table(b, edges, universe, B, W)
    res["books"] = books
    for main_k in mains:
        res["diff_vs"][main_k] = {c: diff(books[main_k], books[c], universe, B, W) for c in compare_to
                                  if c in books and c != main_k}
    return res
