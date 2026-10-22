# -*- coding: utf-8 -*-
"""
power.py — 合成設計の検出力（ラベルを使わない。合成の抽選もしない）
====================================================================
事前登録の前に、部分集合 S の大きさと「神託」（S と真の確率を知る Kelly）の選択期間 2021-22 での
期待対数成長・その標準誤差を、生成モデルから解析的に計算する。学習器は一切動かさない。
出力: learner_check/out/power_analysis.json

実行: python -m analysis.bet_decision_ai.learner_check.power
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import torch

from analysis.bet_decision_ai import data as D
from analysis.bet_decision_ai.learner_check import synth as SY

OUT = Path(__file__).resolve().parent / "out"


def main():
    warnings.filterwarnings("ignore")
    OUT.mkdir(exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cand = SY.load_universe()
    edges = SY.subset_edges(cand)
    S = SY.subset_mask(cand, edges)
    lay = SY.race_layout(cand)
    yrs = cand.groupby("rid16")["year"].first().loc[lay["races"]].to_numpy()
    sel_rows = np.nonzero((yrs >= D.SELECT_YEARS[0]) & (yrs <= D.SELECT_YEARS[1]))[0]
    res = {"edges_lpq_q60_train": edges, "subset_share_all": float(S.mean()),
           "subset_share_by_year": {int(y): float(S[(cand["year"] == y).to_numpy()].mean()) for y in sorted(cand["year"].unique())},
           "subset_by_bet_type": {bt: int(S[(cand["bet_type"] == bt).to_numpy()].sum()) for bt in SY.SUBSET_BET_TYPES},
           "races_with_subset_ticket_share": float(pd_any(S, lay)),
           "n_select_races": int(len(sel_rows)), "conditions": {}}
    for name, boost in SY.CONDITIONS.items():
        pi, info = SY.true_probs(cand, S, boost)
        s6, ev = SY.oracle_stakes(cand, pi, S, lay, sel_rows, device=dev)
        # 分散（同じシナリオで）: 簡便に、最適配分での 1 レース対数成長の分散を再計算
        var = race_var(cand, pi, S, lay, sel_rows, s6)
        n = len(sel_rows)
        mean = ev.mean()
        se = np.sqrt(var.sum()) / n
        res["conditions"][name] = {
            "boost": boost, **info, "fuku_coupling_clip": SY.fuku_coupling_clip_count(cand, pi, lay),
            "oracle_expected_mean_log_growth": float(mean), "oracle_se": float(se),
            "oracle_z": float(mean / se) if se > 0 else None,
            "oracle_bet_race_share": float((s6.sum(1) > 1e-4).mean()),
            "oracle_mean_stake_frac": float(s6.sum(1).mean())}
        print(name, json.dumps(res["conditions"][name], ensure_ascii=False))
    (OUT / "power_analysis.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "conditions"}, ensure_ascii=False, indent=1))


def pd_any(S, lay):
    m = np.zeros(len(lay["races"]), bool)
    np.logical_or.at(m, lay["inv"], S)
    return m.mean()


def race_var(cand, pi, S, lay, rows, s6):
    """最適配分 s6 での 1 レース対数成長の分散（32 シナリオで厳密）。"""
    ti, fi = lay["tan"][rows], lay["fuku"][rows]
    odds = cand["odds_decision"].to_numpy(float)
    pt, pf, ot, of = pi[ti], pi[fi], odds[ti], odds[fi]
    r = np.clip((pf - pt) / np.maximum(1.0 - pt, 1e-9), 0.0, 1.0)
    rest = np.clip(1.0 - pt.sum(1), 0.0, 1.0)
    m1 = np.zeros(len(rows))
    m2 = np.zeros(len(rows))
    for w in range(4):
        pw = pt[:, w] if w < 3 else rest
        for bits in range(8):
            f = np.array([(bits >> k) & 1 for k in range(3)])
            if w < 3 and f[w] == 0:
                continue
            p = pw.copy()
            for k in range(3):
                if k != w:
                    p = p * (r[:, k] if f[k] else (1.0 - r[:, k]))
            Rt = np.zeros_like(ot)
            if w < 3:
                Rt[:, w] = ot[:, w]
            Rv = np.concatenate([Rt, of * f[None, :]], 1)
            g = np.log1p(D.KAPPA * (s6 * (Rv - 1.0)).sum(1))
            m1 += p * g
            m2 += p * g * g
    return np.maximum(m2 - m1 ** 2, 0.0)


if __name__ == "__main__":
    main()
