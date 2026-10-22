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
    r = rankdata(pred, method="first")
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
