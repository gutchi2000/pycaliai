# -*- coding: utf-8 -*-
"""
learner_check_diag_v3.py — v3 学習器確認の不合格の診断（合成のみ、判定外、実データの res_* は読まない）
===================================================================================================
learner_check_v3 と同じ合成・同じ学習 2021／評価 2022 で、12 設定すべてを（CV で選ばずに）当てはめ、
部分集合の券の E[R]（10 本平均）・控えめな採算（下 20% 点）・購入数・部分集合の割合を比べる。
「CV が弱すぎる設定を選んだせい」か「推定器の水準そのもの」かを切り分けるため。
実行: python -m analysis.bet_decision_ai.v3.learner_check_diag_v3 [--conds control A120 B120] [--seed 11]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from analysis.bet_decision_ai.v2 import learner_check_v2 as LC2
from analysis.bet_decision_ai.v3 import evaluate_v3 as E
from analysis.bet_decision_ai.v3 import learner_check_v3 as LC
from analysis.bet_decision_ai.v3 import model_v3 as M

OUT = LC.OUT.parent / "learner_check_diag"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", nargs="+", default=["control", "A120", "B120"])
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    d = LC.load_universe()
    L = LC2.Layout(d)
    Ssets = LC.subset_masks(d, LC.subset_edges(d))
    yr = d["year"].to_numpy()
    tr, ev = yr == LC.TRAIN_YEAR, yr == LC.EVAL_YEAR
    X = LC.D.make_X(d, LC.MODE)
    Xtr, Xev = X[tr].reset_index(drop=True), X[ev].reset_index(drop=True)
    day = d["date"].to_numpy()
    out = {}
    for cond in a.conds:
        sname, boost = LC.CONDITIONS[cond]
        S = Ssets[sname] if sname else np.zeros(len(d), bool)
        pt, pf, _ = LC2.true_probs(d, L, S, boost)
        er = LC.design_er(d, L, pt, pf)
        R = LC2.draw(d, L, pt, pf, a.seed)
        Sev, er_ev, Rev, dev = S[ev], er[ev], R[ev], day[ev]
        base = M.CellBase.fit(Xtr, R[tr])
        cell_ev = base.cell(Xev)
        rows = []
        for ci, cfg in enumerate(M.GRID):
            t0 = time.time()
            ms = M.fit_bootstrap(Xtr, R[tr], day[tr], cfg, base_seed=0)
            P = M.predict_ensemble(ms, Xev)
            mean, cons = P.mean(0), M.conservative(P)
            buy = M.buy_mask(cons)
            nb = int(buy.sum())
            row = {"cfg_id": ci, **cfg, "n_buy": nb,
                   "subset_share_of_buys": float(Sev[buy].mean()) if nb and sname else None,
                   "true_er_of_buys": float(er_ev[buy].mean()) if nb else None,
                   "mean_ER_nonS": float(mean[~Sev].mean()), "cons_nonS": float(cons[~Sev].mean()),
                   "true_er_nonS": float(er_ev[~Sev].mean())}
            if sname:
                row.update(mean_ER_S=float(mean[Sev].mean()), cons_S=float(cons[Sev].mean()),
                           true_er_S=float(er_ev[Sev].mean()), share_S_cons_gt_1=float((cons[Sev] > 1).mean()),
                           share_S_mean_gt_1=float((mean[Sev] > 1).mean()))
            if nb:
                row.update({k: E.roi_stats(Rev, buy, dev)[k] for k in ("roi", "roi_ci")})
                row.update({k: E.excess_stats(Rev, buy, dev, cell_ev)[k] for k in ("excess", "excess_ci")})
            row["sec"] = round(time.time() - t0, 1)
            rows.append(row)
            print(cond, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()},
                                   ensure_ascii=False), flush=True)
        out[cond] = rows
    (OUT / f"diag_seed{a.seed}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float),
                                                 encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
