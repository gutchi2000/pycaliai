# -*- coding: utf-8 -*-
"""
learner_check_diag_v2.py — 学習器の動作確認（合成）が不合格だった原因の診断。判定には使わない。
====================================================================================================
learner_check_v2 と同じ合成（同じ種・同じ条件）を作り、results.json に記録された選択設定で bootstrap 10 本を学習し直して、
2023 の購入を (券種 × オッズ帯) に分け、真の期待回収率・推定の分解（較正後の的中確率 p と条件付き倍率 m）を比べる。
実データの結果列は読まない。
実行: python -m analysis.bet_decision_ai.v2.learner_check_diag_v2 [--cond control --seed 11]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from analysis.bet_decision_ai.v2 import data_v2 as D
from analysis.bet_decision_ai.v2 import model_v2 as M
from analysis.bet_decision_ai.v2 import learner_check_v2 as LC

BANDS = [0, 2, 5, 10, 20, 50, 100, 1e9]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="control")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--cfg-id", type=int, default=None, help="診断専用: 選択設定の代わりに GRID[i] を使う（判定には使わない）")
    a = ap.parse_args()
    res = json.loads((LC.OUT / "results.json").read_text(encoding="utf-8"))
    rr = [r for r in res["results"] if r["cond"] == a.cond and r["seed"] == a.seed][0]
    cfg = rr["two_stage"]["cfg"] if a.cfg_id is None else M.GRID[a.cfg_id]
    d = LC.load_universe()
    L = LC.Layout(d)
    S = LC.subset_mask(d, LC.subset_edges(d))
    pt, pf, _ = LC.true_probs(d, L, S, LC.CONDITIONS[a.cond])
    R = LC.draw(d, L, pt, pf, a.seed)
    pi = np.zeros(len(d))
    pi[L.tan], pi[L.fuku] = pt, pf
    o = d["o_bet"].to_numpy(float)
    true_er = pi * o
    X = D.make_X(d, LC.MODE)
    yr = d["year"].to_numpy()
    tr = (yr >= 2021) & (yr <= 2022)
    se = yr == 2023
    day = d["date"].to_numpy()
    Xtr, Xse = X[tr].reset_index(drop=True), X[se].reset_index(drop=True)
    ms = M.fit_bootstrap("two_stage", Xtr, R[tr], day[tr], cfg, base_seed=0)
    parts = [m.predict_parts(Xse) for m in ms]
    P = np.stack([p * mm for p, mm in parts])
    cons = M.conservative(P)
    buy = M.buy_mask(cons)
    p0 = np.stack([p for p, _ in parts]).mean(0)
    m0 = np.stack([mm for _, mm in parts]).mean(0)
    # 単一モデル（boot 0）の生の分類器出力と較正後
    raw0 = ms[0].clf.predict(Xse)
    t = pd.DataFrame({"bt": d["bet_type"].to_numpy()[se], "o": o[se], "pi": pi[se], "true_er": true_er[se],
                      "R": R[se], "cons": cons, "p_hat": p0, "m_hat": m0, "raw0": raw0, "buy": buy})
    t["band"] = pd.cut(t["o"], BANDS, right=False)
    g = t.groupby(["bt", "band"], observed=True)
    tab = g.agg(n=("o", "size"), n_buy=("buy", "sum"), true_pi=("pi", "mean"), p_hat=("p_hat", "mean"),
                m_over_o=("m_hat", lambda s: float((s / t.loc[s.index, "o"]).mean())),
                true_er=("true_er", "mean"), cons_mean=("cons", "mean"),
                synth_R_mean=("R", "mean")).reset_index()
    tb = t[t.buy].groupby(["bt", "band"], observed=True).agg(
        n=("o", "size"), true_pi=("pi", "mean"), p_hat=("p_hat", "mean"),
        m_over_o=("m_hat", lambda s: float((s / t.loc[s.index, "o"]).mean())),
        true_er=("true_er", "mean"), cons=("cons", "mean"), synth_roi=("R", "mean")).reset_index()
    # isotonic の段の幅: 購入券の生出力と較正後の関係
    iso = ms[0].iso
    out = {"cond": a.cond, "seed": a.seed, "cfg": cfg, "n_buy": int(buy.sum()),
           "buy_true_er_mean": float(t.loc[t.buy, "true_er"].mean()) if buy.any() else None,
           "buy_p_hat_over_true_pi": float((t.loc[t.buy, "p_hat"] / t.loc[t.buy, "pi"]).mean()) if buy.any() else None,
           "buy_m_hat_over_odds": float((t.loc[t.buy, "m_hat"] / t.loc[t.buy, "o"]).mean()) if buy.any() else None,
           "iso_n_thresholds": int(len(iso.X_thresholds_)),
           "all_by_type_band": tab.astype({"band": str}).to_dict(orient="records"),
           "buys_by_type_band": tb.astype({"band": str}).to_dict(orient="records"),
           "sigma2": [m.sigma2 for m in ms]}
    out["synth_roi_of_buys"] = float(t.loc[t.buy, "R"].mean()) if buy.any() else None
    out["cfg_override"] = a.cfg_id
    p = LC.OUT / (f"diag_{a.cond}_s{a.seed}" + (f"_cfg{a.cfg_id:02d}" if a.cfg_id is not None else "") + ".json")
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(json.dumps({k: v for k, v in out.items() if k not in ("all_by_type_band", "buys_by_type_band")},
                     ensure_ascii=False, default=float))
    print(tab.round(4).to_string())
    print(tb.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
