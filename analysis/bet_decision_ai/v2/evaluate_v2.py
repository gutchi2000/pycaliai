# -*- coding: utf-8 -*-
"""
evaluate_v2.py — 馬券判断 AI v2 の評価（仕様 v1.0 §5-§7）
==========================================================
入力は候補ごとの配列: 予測採算 pred（主は控えめな採算 = bootstrap 10 本の下 20% 点）、実際の払戻倍率 R、開催日 date、
券種 bet_type、購入フラグ buy（100 円固定）。
- 回収率 = Σ R[buy] / 購入数。95% CI は開催日ブロック bootstrap（開催日を復元抽出、日ごとに購入数・払戻を束ねる、4,000 回）
- 上位 10 払戻除外 = 購入分のうち払戻の大きい 10 件を除いた回収率（件数・払戻とも除外）
- チケット単位の順位相関 = Spearman(pred, R)（全候補）。95% CI は開催日ブロック bootstrap（2,000 回）
- 十分位表 = pred の十分位ごとの候補数・平均 pred・平均 R
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPS_ROI = 4000
REPS_RHO = 2000


def _day_index(date: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    days, inv = np.unique(np.asarray(date).astype(str), return_inverse=True)
    return days, inv


def boot_weights(nd: int, reps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multinomial(nd, np.full(nd, 1.0 / nd), size=reps).astype(np.float64)


def roi_stats(R: np.ndarray, buy: np.ndarray, date: np.ndarray, reps: int = REPS_ROI, seed: int = 0) -> dict:
    days, inv = _day_index(date)
    nd = len(days)
    n_buy = int(buy.sum())
    out = {"n_candidates": int(len(R)), "n_buy": n_buy, "buy_share": float(n_buy / len(R)) if len(R) else None,
           "n_buy_days": int(len(np.unique(inv[buy]))) if n_buy else 0}
    if n_buy == 0:
        out.update(roi=None, roi_ci=None, roi_ex_top10=None, hit_rate=None)
        return out
    cnt = np.bincount(inv[buy], minlength=nd).astype(float)
    pay = np.bincount(inv[buy], weights=R[buy], minlength=nd)
    W = boot_weights(nd, reps, seed)
    with np.errstate(invalid="ignore", divide="ignore"):
        rb = (W @ pay) / (W @ cnt)
    rb = rb[np.isfinite(rb)]
    r = R[buy]
    top = np.sort(r)[::-1][:10]
    out.update(roi=float(r.sum() / n_buy), roi_ci=[float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))],
               roi_boot_valid=int(len(rb)),
               roi_ex_top10=float((r.sum() - top.sum()) / (n_buy - len(top))) if n_buy > len(top) else None,
               top10_R=[float(x) for x in top], hit_rate=float((r > 0).mean()),
               refund_n=int((r == 1.0).sum()))
    return out


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = rankdata(x), rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / den) if den > 0 else np.nan


def spearman_stats(pred: np.ndarray, R: np.ndarray, date: np.ndarray, reps: int = REPS_RHO, seed: int = 1) -> dict:
    days, inv = _day_index(date)
    nd = len(days)
    rho = _spearman(pred, R)
    order = np.argsort(inv, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(inv, minlength=nd))]
    rng = np.random.default_rng(seed)
    bs = np.empty(reps)
    for i in range(reps):
        pick = rng.integers(0, nd, nd)
        idx = np.concatenate([order[bounds[k]:bounds[k + 1]] for k in pick])
        bs[i] = _spearman(pred[idx], R[idx])
    return {"rho": rho, "rho_ci": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))], "reps": reps}


def decile_table(pred: np.ndarray, R: np.ndarray, buy: np.ndarray | None = None) -> list[dict]:
    r = rankdata(pred, method="ordinal")
    dec = np.minimum((r - 1) * 10 // len(pred), 9).astype(int)
    rows = []
    for k in range(10):
        m = dec == k
        rows.append({"decile": k + 1, "n": int(m.sum()), "pred_min": float(pred[m].min()), "pred_max": float(pred[m].max()),
                     "pred_mean": float(pred[m].mean()), "R_mean": float(R[m].mean()),
                     "hit_rate": float((R[m] > 0).mean()),
                     "n_buy": int(buy[m].sum()) if buy is not None else None})
    return rows


def full_report(pred: np.ndarray, R: np.ndarray, date: np.ndarray, bet_type: np.ndarray, buy: np.ndarray,
                with_rho_ci: bool = True, pred_mean: np.ndarray | None = None) -> dict:
    out = {"all": roi_stats(R, buy, date)}
    out["all"].update(spearman_stats(pred, R, date) if with_rho_ci else {"rho": _spearman(pred, R)})
    if pred_mean is not None:
        out["all"]["rho_boot_mean_pred"] = _spearman(pred_mean, R)
    out["deciles"] = decile_table(pred, R, buy)
    out["by_bet_type"] = {}
    for b in np.unique(bet_type):
        m = bet_type == b
        d = roi_stats(R[m], buy[m], date[m])
        d.update(spearman_stats(pred[m], R[m], date[m]) if with_rho_ci else {"rho": _spearman(pred[m], R[m])})
        d["deciles"] = decile_table(pred[m], R[m], buy[m])
        out["by_bet_type"][str(b)] = d
    return out


def judge_select(rep: dict) -> dict:
    """仕様 §6-2: 順位相関の CI 下限 ≤ 0、または購入 100 件未満なら打ち切り。"""
    a = rep["all"]
    lo = a["rho_ci"][0]
    stop_rho = lo <= 0
    stop_n = a["n_buy"] < 100
    return {"rho_ci_lo": lo, "n_buy": a["n_buy"], "stop_rho_ci_le_0": bool(stop_rho), "stop_n_buy_lt_100": bool(stop_n),
            "abort": bool(stop_rho or stop_n)}


def judge_confirm(rep: dict) -> dict:
    """仕様 §7。"""
    a = rep["all"]
    rho_ok = a["rho_ci"][0] > 0
    roi_ok = a["roi_ci"] is not None and a["roi_ci"][0] > 1.0
    ex_ok = a["roi_ex_top10"] is not None and a["roi_ex_top10"] >= 1.0
    n_ok = a["n_buy"] >= 300
    roi_straddle = a["roi_ci"] is not None and a["roi_ci"][0] <= 1.0 <= a["roi_ci"][1]
    note = None
    if rho_ok and roi_ok and ex_ok and n_ok:
        v = "PASS"
    elif rho_ok and roi_straddle:
        v = "SIGNAL"
    else:
        v = "FAIL"
        if rho_ok:
            note = ("順位相関の CI 下限 > 0 だが、回収率の CI が 100 をまたがない（または購入 0）ため、§7 の文言どおり FAIL。"
                    "CI が 100 より上で PASS の他条件を欠く場合もここに入る")
    return {"verdict": v, "rho_ci_lo_gt_0": bool(rho_ok), "roi_ci_lo_gt_100": bool(roi_ok),
            "roi_ci_straddles_100": bool(roi_straddle), "roi_ex_top10_ge_100": bool(ex_ok), "n_buy_ge_300": bool(n_ok),
            "note": note}


def paired_diff(predA: np.ndarray, buyA: np.ndarray, predB: np.ndarray, buyB: np.ndarray, R: np.ndarray,
                date: np.ndarray, reps: int = REPS_RHO, seed: int = 2) -> dict:
    """A − B（同じ候補・同じ開催日抽出で対にする）: 順位相関の差と購入分回収率の差。仕様 §6-4。"""
    days, inv = _day_index(date)
    nd = len(days)
    order = np.argsort(inv, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(inv, minlength=nd))]
    rng = np.random.default_rng(seed)

    def roi(buy, idx):
        n = buy[idx].sum()
        return R[idx][buy[idx]].sum() / n if n > 0 else np.nan

    allidx = np.arange(len(R))
    d_rho = _spearman(predA, R) - _spearman(predB, R)
    d_roi = roi(buyA, allidx) - roi(buyB, allidx)
    br, bo = np.empty(reps), np.empty(reps)
    for i in range(reps):
        pick = rng.integers(0, nd, nd)
        idx = np.concatenate([order[bounds[k]:bounds[k + 1]] for k in pick])
        br[i] = _spearman(predA[idx], R[idx]) - _spearman(predB[idx], R[idx])
        bo[i] = roi(buyA, idx) - roi(buyB, idx)
    bo = bo[np.isfinite(bo)]
    return {"rho_diff": float(d_rho), "rho_diff_ci": [float(np.percentile(br, 2.5)), float(np.percentile(br, 97.5))],
            "roi_diff": float(d_roi) if np.isfinite(d_roi) else None,
            "roi_diff_ci": [float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))] if len(bo) else None,
            "roi_diff_valid_reps": int(len(bo)), "reps": reps}


# ================================================================ v1.1 §10: 超過回収
NCELL = 20


def _excess_arrays(R: np.ndarray, buy: np.ndarray, date: np.ndarray, cell: np.ndarray):
    days, inv = _day_index(date)
    nd = len(days)
    idx = inv * NCELL + cell
    sumR = np.bincount(idx, weights=R, minlength=nd * NCELL).reshape(nd, NCELL)
    cnt = np.bincount(idx, minlength=nd * NCELL).reshape(nd, NCELL).astype(float)
    bcnt = np.bincount(idx[buy], minlength=nd * NCELL).reshape(nd, NCELL).astype(float)
    bR = np.bincount(inv[buy], weights=R[buy], minlength=nd)
    return nd, sumR, cnt, bcnt, bR


def _excess_from(W, sumR, cnt, bcnt, bR):
    """W: [reps, nd]（点推定は 1 行の全 1）。超過回収 = (Σ_buy R − Σ_c n_buy,c · 平均R_c) / n_buy。"""
    cs, cc = W @ sumR, W @ cnt
    with np.errstate(invalid="ignore", divide="ignore"):
        base = np.where(cc > 0, cs / cc, 0.0)
        nb = W @ bcnt
        return ((W @ bR) - (nb * base).sum(1)) / nb.sum(1)


def excess_stats(R: np.ndarray, buy: np.ndarray, date: np.ndarray, cell: np.ndarray, reps: int = REPS_ROI,
                 seed: int = 3) -> dict:
    """購入券の R − 同じ期間・券種・オッズ十分位の全候補平均 R、の購入平均。CI は開催日ブロック bootstrap（セル平均も抽出ごとに再計算）。"""
    n_buy = int(buy.sum())
    if n_buy == 0:
        return {"excess": None, "excess_ci": None, "excess_ci_contains_0": None}
    nd, sumR, cnt, bcnt, bR = _excess_arrays(R, buy, date, cell)
    pt = float(_excess_from(np.ones((1, nd)), sumR, cnt, bcnt, bR)[0])
    W = boot_weights(nd, reps, seed)
    eb = _excess_from(W, sumR, cnt, bcnt, bR)
    eb = eb[np.isfinite(eb)]
    ci = [float(np.percentile(eb, 2.5)), float(np.percentile(eb, 97.5))]
    return {"excess": pt, "excess_ci": ci, "excess_ci_contains_0": bool(ci[0] <= 0.0 <= ci[1]),
            "excess_boot_valid": int(len(eb))}


def excess_diff(buyA: np.ndarray, buyB: np.ndarray, R: np.ndarray, date: np.ndarray, cell: np.ndarray,
                reps: int = REPS_ROI, seed: int = 4) -> dict:
    """A − B の超過回収の差と回収率の差（同じ開催日抽出で対）。§6-4。"""
    nd, sumR, cnt, bA, rA = _excess_arrays(R, buyA, date, cell)
    _, _, _, bB, rB = _excess_arrays(R, buyB, date, cell)
    W = np.vstack([np.ones((1, nd)), boot_weights(nd, reps, seed)])
    with np.errstate(invalid="ignore", divide="ignore"):
        eA = _excess_from(W, sumR, cnt, bA, rA) if buyA.any() else np.full(len(W), np.nan)
        eB = _excess_from(W, sumR, cnt, bB, rB) if buyB.any() else np.full(len(W), np.nan)
        roA = (W @ rA) / (W @ bA.sum(1)) if buyA.any() else np.full(len(W), np.nan)
        roB = (W @ rB) / (W @ bB.sum(1)) if buyB.any() else np.full(len(W), np.nan)
    de, dr = eA - eB, roA - roB

    def ci(x):
        x = x[1:][np.isfinite(x[1:])]
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) else None
    return {"excess_diff": float(de[0]) if np.isfinite(de[0]) else None, "excess_diff_ci": ci(de),
            "roi_diff": float(dr[0]) if np.isfinite(dr[0]) else None, "roi_diff_ci": ci(dr), "reps": reps}


def full_report_v11(pred: np.ndarray, R: np.ndarray, date: np.ndarray, bet_type: np.ndarray, buy: np.ndarray,
                    cell: np.ndarray, pred_mean: np.ndarray | None = None) -> dict:
    out = {"all": roi_stats(R, buy, date)}
    out["all"].update(excess_stats(R, buy, date, cell))
    out["all"].update(spearman_stats(pred, R, date))          # 記述のみ
    if pred_mean is not None:
        out["all"]["rho_boot_mean_pred"] = _spearman(pred_mean, R)
    out["deciles"] = decile_table(pred, R, buy)
    out["by_bet_type"] = {}
    for b in np.unique(bet_type):
        m = bet_type == b
        dd = roi_stats(R[m], buy[m], date[m])
        dd.update(excess_stats(R[m], buy[m], date[m], cell[m]))
        dd.update(spearman_stats(pred[m], R[m], date[m]))
        dd["deciles"] = decile_table(pred[m], R[m], buy[m])
        out["by_bet_type"][str(b)] = dd
    # 購入券のオッズ十分位別（セル別）
    cb = []
    for c in range(NCELL):
        m = cell == c
        if m.any():
            cb.append({"cell": c, "bet_type": "tan" if c < 10 else "fuku", "decile": c % 10 + 1, "n": int(m.sum()),
                       "cell_mean_R": float(R[m].mean()), "n_buy": int(buy[m].sum()),
                       "buy_mean_R": float(R[m & buy].mean()) if (m & buy).any() else None})
    out["cells"] = cb
    return out


def judge_select_v11(rep: dict) -> dict:
    """§10 §6-2（改）: 超過回収の CI 下限 ≤ 0、または購入 100 件未満なら打ち切り。"""
    a = rep["all"]
    lo = a["excess_ci"][0] if a.get("excess_ci") else None
    stop_ex = lo is None or lo <= 0
    stop_n = a["n_buy"] < 100
    return {"excess_ci_lo": lo, "n_buy": a["n_buy"], "stop_excess_ci_le_0": bool(stop_ex),
            "stop_n_buy_lt_100": bool(stop_n), "abort": bool(stop_ex or stop_n)}


def judge_confirm_v11(rep: dict) -> dict:
    """§10 §7（改）。SIGNAL は文言どおり「超過回収の CI 下限 > 0 で、回収率の CI が 100 をまたぐ」。"""
    a = rep["all"]
    ex_ok = bool(a.get("excess_ci") and a["excess_ci"][0] > 0)
    roi_ok = bool(a["roi_ci"] is not None and a["roi_ci"][0] > 1.0)
    straddle = bool(a["roi_ci"] is not None and a["roi_ci"][0] <= 1.0 <= a["roi_ci"][1])
    top_ok = bool(a["roi_ex_top10"] is not None and a["roi_ex_top10"] >= 1.0)
    n_ok = a["n_buy"] >= 300
    note = None
    if ex_ok and roi_ok and top_ok and n_ok:
        v = "PASS"
    elif ex_ok and straddle:
        v = "SIGNAL"
    else:
        v = "FAIL"
        if ex_ok:
            note = "超過回収の CI 下限 > 0 だが PASS の条件を欠き、回収率の CI が 100 をまたがないため、§7（改）の文言どおり FAIL"
    return {"verdict": v, "excess_ci_lo_gt_0": ex_ok, "roi_ci_lo_gt_100": roi_ok, "roi_ci_straddles_100": straddle,
            "roi_ex_top10_ge_100": top_ok, "n_buy_ge_300": bool(n_ok), "note": note}
