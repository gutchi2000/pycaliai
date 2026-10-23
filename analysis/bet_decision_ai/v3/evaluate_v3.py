# -*- coding: utf-8 -*-
"""
evaluate_v3.py — 馬券判断 AI v3 の評価（仕様 v1.0 §4）
=====================================================
集計の部品は v2 の evaluate_v2.py をそのまま使う（v2 のファイルは変更しない）:
  回収率と開催日ブロック bootstrap CI（4,000 回）、上位 10 払戻除外、超過回収（購入券の R − 同じ期間・同じセルの全候補平均 R、
  CI は抽出ごとにセル平均も作り直す）、十分位表、Spearman（記述）、対の差。
v3 で足すもの: §4 の判定（購入 100 件以上の SIGNAL）、報告の束ね方。
"""
from __future__ import annotations

import numpy as np

from analysis.bet_decision_ai.v2.evaluate_v2 import (NCELL, _spearman, decile_table, excess_diff,  # noqa: F401
                                                     excess_stats, roi_stats, spearman_stats)


def full_report(pred: np.ndarray, R: np.ndarray, date: np.ndarray, bet_type: np.ndarray, buy: np.ndarray,
                cell: np.ndarray, pred_mean: np.ndarray | None = None, with_rho_ci: bool = True) -> dict:
    out = {"all": roi_stats(R, buy, date)}
    out["all"].update(excess_stats(R, buy, date, cell))
    out["all"].update(spearman_stats(pred, R, date) if with_rho_ci else {"rho": _spearman(pred, R)})   # 記述のみ
    if pred_mean is not None:
        out["all"]["rho_boot_mean_pred"] = _spearman(pred_mean, R)
    out["deciles"] = decile_table(pred, R, buy)
    out["by_bet_type"] = {}
    for b in ("tan", "fuku"):
        m = bet_type == b
        if not m.any():
            continue
        dd = roi_stats(R[m], buy[m], date[m])
        dd.update(excess_stats(R[m], buy[m], date[m], cell[m]))
        dd.update({"rho": _spearman(pred[m], R[m])})
        dd["deciles"] = decile_table(pred[m], R[m], buy[m])
        out["by_bet_type"][b] = dd
    cb = []
    for c in range(NCELL):
        m = cell == c
        if m.any():
            cb.append({"cell": c, "bet_type": "tan" if c < 10 else "fuku", "decile": c % 10 + 1, "n": int(m.sum()),
                       "cell_mean_R": float(R[m].mean()), "n_buy": int(buy[m].sum()),
                       "buy_mean_R": float(R[m & buy].mean()) if (m & buy).any() else None})
    out["cells"] = cb
    return out


def judge_confirm(rep: dict) -> dict:
    """仕様 v3 §4（v1.1 §6 で SIGNAL を改訂）: SIGNAL =「超過回収の CI 下限 > 0、かつ購入 100 件以上、かつ PASS に当たらない」。"""
    a = rep["all"]
    ex_ok = bool(a.get("excess_ci") and a["excess_ci"][0] > 0)
    roi_ok = bool(a.get("roi_ci") is not None and a["roi_ci"][0] > 1.0)
    straddle = bool(a.get("roi_ci") is not None and a["roi_ci"][0] <= 1.0 <= a["roi_ci"][1])
    top_ok = bool(a.get("roi_ex_top10") is not None and a["roi_ex_top10"] >= 1.0)
    n300 = a["n_buy"] >= 300
    n100 = a["n_buy"] >= 100
    note = None
    if ex_ok and roi_ok and top_ok and n300:
        v = "PASS"
    elif ex_ok and n100:
        v = "SIGNAL"
    else:
        v = "FAIL"
    return {"verdict": v, "excess_ci_lo_gt_0": ex_ok, "roi_ci_lo_gt_100": roi_ok, "roi_ci_straddles_100": straddle,
            "roi_ex_top10_ge_100": top_ok, "n_buy_ge_300": bool(n300), "n_buy_ge_100": bool(n100), "note": note}
