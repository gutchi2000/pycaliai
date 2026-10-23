# -*- coding: utf-8 -*-
"""
learner_check_v3.py — 馬券判断 AI v3 の学習器の動作確認（仕様 v1.0 §3-1、合成データ）
=====================================================================================
事前登録: docs/research/BET_DECISION_AI_V3_LEARNER_CHECK_PREREG_20261024.md（sha256 を out/learner_check/prereg_sha256.txt に記録、--run で照合）

2021-22 の実際の候補と特徴（data_v3.load_rows((2021, 2022), with_results=False) → build_features）だけを使い、払戻だけを合成する。
実データの結果列 res_* は読まない（列そのものを読み込まない）。2023 年以降は特徴も読まない。
合成の作り方は v2 と同じ（analysis.bet_decision_ai.v2.learner_check_v2 の Layout / true_probs / draw をそのまま呼ぶ）。
学習 2021・評価 2022。条件 = 対照 / A110 / A120 / B110 / B120、合成の種 11 / 22 / 33（同じ種では全条件で同じ一様乱数）。
  A（低オッズ、v2 と同じ）: log_p_over_q ≥ 2021-22 の券種別 60% 点、n_field_s1 ≥ 12、odds_tan ≤ 10
  B（中〜高オッズ、仕様 v1.1 §6）: log_p_over_q ≥ 同じ 60% 点、n_field_s1 ≥ 14、10 < odds_tan ≤ 30
合否: 対照（3 本すべて）、A110・A120・B120（3 本中 2 本以上）。B110 は記述のみ（仕様 v1.1 §6）。
R̄_cell・境界・打ち切り点は bootstrap の各回で作り直す（model_v3.fit_bootstrap）。評価の超過回収のセルは学習 2021 全体の十分位境界、
セル平均は評価 2022 の全候補。
実行:
  python -m analysis.bet_decision_ai.v3.learner_check_v3 --design          # 設計の数値（抽選も学習もしない）
  python -m analysis.bet_decision_ai.v3.learner_check_v3 --record-prereg   # 事前登録の sha256 を記録
  python -m analysis.bet_decision_ai.v3.learner_check_v3 --run             # 本番（事前登録の sha256 を照合してから）
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.bet_decision_ai.v2 import learner_check_v2 as LC2     # 合成の作り方（v2 と同じ）。v2 のファイルは変更しない
from analysis.bet_decision_ai.v3 import data_v3 as D
from analysis.bet_decision_ai.v3 import evaluate_v3 as E
from analysis.bet_decision_ai.v3 import model_v3 as M

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "learner_check"
PREREG = D.BASE / "docs" / "research" / "BET_DECISION_AI_V3_LEARNER_CHECK_PREREG_20261024.md"
PREREG_SHA = OUT / "prereg_sha256.txt"

SUBSET_LPQ_Q = 0.60
SUBSET_MIN_FIELD = 12           # A
B_MIN_FIELD = 14                # B（仕様 v1.1 §6）
A_MAX_ODDS = 10.0
B_ODDS = (10.0, 30.0)          # 10 < odds_tan ≤ 30（仕様 v1.1 §6）
B_SIZE_RANGE = (0.02, 0.10)    # B の大きさ（2021-22 全候補に対する割合）の事前確認範囲
CONDITIONS = {"control": (None, None), "A110": ("A", 1.10), "A120": ("A", 1.20), "B110": ("B", 1.10),
              "B120": ("B", 1.20)}
GATED = ["control", "A110", "A120", "B120"]     # B110 は記述のみ
SYNTH_SEEDS = [11, 22, 33]
TRAIN_YEAR, EVAL_YEAR = 2021, 2022
MODE = "model"
FORCED_TOP = (0.01, 0.05)
_LOGF = None


def log(msg: str) -> None:
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    if _LOGF:
        with open(_LOGF, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def load_universe() -> pd.DataFrame:
    df = D.load_rows(D.TRAIN_YEARS, with_results=False)
    assert not any(c.startswith("res_") for c in df.columns)
    assert set(df["year"].unique()) <= set(D.TRAIN_YEARS)
    d, _ = D.build_features(df)
    return d.sort_values(["rid16", "umaban", "bet_type"]).reset_index(drop=True)


def subset_edges(d: pd.DataFrame) -> dict:
    return {bt: float(d.loc[d["bet_type"] == bt, "log_p_over_q"].quantile(SUBSET_LPQ_Q)) for bt in ("tan", "fuku")}


def subset_masks(d: pd.DataFrame, edges: dict) -> dict:
    e = d["bet_type"].map(edges).to_numpy(float)
    hi = d["log_p_over_q"].to_numpy(float) >= e
    nf = d["n_field_s1"].to_numpy(float)
    ot = d["odds_tan"].to_numpy(float)
    return {"A": hi & (nf >= SUBSET_MIN_FIELD) & (ot <= A_MAX_ODDS),
            "B": hi & (nf >= B_MIN_FIELD) & (ot > B_ODDS[0]) & (ot <= B_ODDS[1])}


def design_er(d: pd.DataFrame, L, pt: np.ndarray, pf: np.ndarray) -> np.ndarray:
    er = np.zeros(len(d))
    er[L.tan] = pt * d["odds_tan"].to_numpy(float)[L.tan]
    er[L.fuku] = pf * d["odds_fuku_mid"].to_numpy(float)[L.fuku]
    return er


def er_summary(d, er, S) -> dict:
    out = {}
    yr = d["year"].to_numpy()
    bt = d["bet_type"].to_numpy()
    for y in D.TRAIN_YEARS:
        m = yr == y
        o = {"nonS_mean": float(er[m & ~S].mean()), "nonS_max": float(er[m & ~S].max()),
             "nonS_share_gt_1": float((er[m & ~S] > 1.0).mean())}
        if S is not None and (m & S).any():
            o["S_mean"] = float(er[m & S].mean())
            o["S_mean_by_type"] = {b: float(er[m & S & (bt == b)].mean()) for b in ("tan", "fuku") if (m & S & (bt == b)).any()}
        out[str(y)] = o
    return out


def oracle_power(d, er, S, ev) -> dict:
    """評価年で S を全部買ったときの期待回収率と、独立近似の標準誤差・z。"""
    o = np.where(d["bet_type"].to_numpy() == "tan", d["odds_tan"], d["odds_fuku_mid"]).astype(float)
    m = ev & S
    pi = er[m] / o[m]
    v = (pi * o[m] ** 2 - (pi * o[m]) ** 2).sum()
    n = int(m.sum())
    se = float(np.sqrt(v) / n)
    return {"n": n, "expected_roi": float(er[m].mean()), "se_indep": se, "z_vs_100": float((er[m].mean() - 1.0) / se)}


def run_one(d, L, X, tr, ev, Ssets, cell_fn, cond, seed) -> dict:
    sname, boost = CONDITIONS[cond]
    S = Ssets[sname] if sname else np.zeros(len(d), bool)
    pt, pf, info = LC2.true_probs(d, L, S, boost)
    er = design_er(d, L, pt, pf)
    R = LC2.draw(d, L, pt, pf, seed)
    day = d["date"].to_numpy()
    Xtr, Xev = X[tr].reset_index(drop=True), X[ev].reset_index(drop=True)
    Rtr, Rev = R[tr], R[ev]
    dev = day[ev]
    Sev = S[ev]
    bt = d["bet_type"].to_numpy()[ev]
    er_ev = er[ev]
    pref = f"[{cond} s{seed}]"
    out = {"cond": cond, "subset": sname, "boost": boost, "seed": seed, "pi_info": info}
    t0 = time.time()
    cfg, grid = M.select_cfg_cv(Xtr, Rtr, day[tr], seed=0)
    best = min((g for g in grid if g["cfg_id"] >= 0), key=lambda g: (g["cv_mse"], g["cfg_id"]))
    base = M.CellBase.fit(Xtr, Rtr)                 # 評価の超過回収のセル境界（学習 2021 全体）。購入判断には使わない
    models = M.fit_bootstrap(Xtr, Rtr, day[tr], cfg, base_seed=0)
    P = M.predict_ensemble(models, Xev)
    cons = M.conservative(P)
    buy = M.buy_mask(cons)
    nb = int(buy.sum())
    cell_ev = base.cell(Xev)
    cm_true = np.bincount(cell_ev, weights=er_ev, minlength=E.NCELL) / np.maximum(np.bincount(cell_ev, minlength=E.NCELL), 1)
    true_ex = er_ev - cm_true[cell_ev]
    res = {"cfg": cfg, "cfg_id": best["cfg_id"], "grid": grid, "train_base": base.to_state(),
           "boot_rbar_range": [[float(min(m.base.rbar[c] for m in models)), float(max(m.base.rbar[c] for m in models))]
                               for c in range(E.NCELL)],
           **E.roi_stats(Rev, buy, dev), **E.excess_stats(Rev, buy, dev, cell_ev),
           "subset_share_of_buys": float(Sev[buy].mean()) if nb else None,
           "subset_recall": float(buy[Sev].mean()) if Sev.any() else None,
           "buy_by_bet_type": {b: int(buy[bt == b].sum()) for b in ("tan", "fuku")},
           "buy_by_cell": np.bincount(cell_ev[buy], minlength=E.NCELL).tolist(),
           "true_er_mean_of_buys": float(er_ev[buy].mean()) if nb else None,
           "true_excess_of_buys": float(true_ex[buy].mean()) if nb else None,
           "cons_q": [float(x) for x in np.quantile(cons, [0, .5, .9, .99, .999, 1])],
           "cons_mean_S": float(cons[Sev].mean()) if Sev.any() else None,
           "cons_mean_nonS": float(cons[~Sev].mean()),
           "boot_spread_median": float(np.median(P.max(0) - P.min(0)))}
    if sname:
        o = E.roi_stats(Rev, Sev, dev)
        o.update(E.excess_stats(Rev, Sev, dev, cell_ev))
        o["true_excess"] = float(true_ex[Sev].mean())
        out["oracle_buy_S"] = {k: o[k] for k in ("n_buy", "roi", "roi_ci", "excess", "excess_ci", "true_excess")}
    forced = {}
    order = np.argsort(-cons, kind="stable")
    for k in FORCED_TOP:
        fb = np.zeros(len(cons), bool)
        fb[order[: int(round(k * len(cons)))]] = True
        f = {**E.roi_stats(Rev, fb, dev), **E.excess_stats(Rev, fb, dev, cell_ev)}
        f["true_excess"] = float(true_ex[fb].mean())
        f["subset_share"] = float(Sev[fb].mean())
        forced[f"top{int(k * 100)}pct"] = {kk: f[kk] for kk in ("n_buy", "roi", "roi_ci", "excess", "excess_ci",
                                                                 "excess_ci_contains_0", "true_excess", "subset_share")}
    res["forced_top"] = forced
    if cond == "control":
        res["pass"] = bool(res["buy_share"] <= 0.01 and (nb == 0 or res["excess_ci_contains_0"]))
    else:
        res["pass"] = bool(nb > 0 and res["roi_ci"][0] > 1.0 and res["subset_share_of_buys"] >= 0.5
                           and res["excess_ci"][0] > 0)
    out["sq_excess"] = res
    out["sec"] = round(time.time() - t0, 1)
    log(f"{pref} cfg{best['cfg_id']:02d} {cfg} buys={nb} share={res['buy_share']:.4f} tan/fuku={res['buy_by_bet_type']} "
        f"roi={res['roi']} ci={res['roi_ci']} excess={res['excess']} ci={res['excess_ci']} "
        f"S_share={res['subset_share_of_buys']} recall={res['subset_recall']} pass={res['pass']} | forced "
        + " ".join(f"{k}: ex={v['excess']:.4f} ci={[round(x, 4) for x in v['excess_ci']]} true={v['true_excess']:.4f}"
                   for k, v in forced.items()) + f" ({out['sec']}s)")
    return out


def judge(results: list) -> dict:
    jj = {}
    for cond in CONDITIONS:
        ps = [r["sq_excess"]["pass"] for r in results if r["cond"] == cond]
        need = len(ps) if cond == "control" else 2
        jj[cond] = {"passes": ps, "n_pass": int(sum(ps)), "need": need, "ok": bool(len(ps) == 3 and sum(ps) >= need),
                    "gated": cond in GATED}
    jj["overall"] = bool(all(jj[c]["ok"] for c in GATED))
    ctrl = [r for r in results if r["cond"] == "control"]
    jj["metric_check_control_forced_top1_contains_0"] = [r["sq_excess"]["forced_top"]["top1pct"]["excess_ci_contains_0"]
                                                         for r in ctrl]
    jj["metric_check_ok"] = bool(len(ctrl) == 3 and all(jj["metric_check_control_forced_top1_contains_0"]))
    jj["proceed_to_real_data"] = bool(jj["overall"] and jj["metric_check_ok"])
    return jj


def main() -> int:
    global _LOGF
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--design", action="store_true")
    g.add_argument("--record-prereg", action="store_true")
    g.add_argument("--run", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    spec_sha = D.check_spec()
    if a.record_prereg:
        if (OUT / "results.json").exists():
            raise SystemExit("結果が既にある（事前登録の記録し直しは不可）")
        s = D.sha256(PREREG)
        PREREG_SHA.write_text(f"{s}  {PREREG.name}  recorded {datetime.now():%Y-%m-%d %H:%M:%S}\n", encoding="utf-8")
        print(s)
        return 0
    d = load_universe()
    L = LC2.Layout(d)
    edges = subset_edges(d)
    Ssets = subset_masks(d, edges)
    yr = d["year"].to_numpy()
    tr = yr == TRAIN_YEAR
    ev = yr == EVAL_YEAR
    if a.design:
        bt = d["bet_type"].to_numpy()
        des = {"spec_sha256": spec_sha, "n_candidates": int(len(d)),
               "n_by_year": {str(y): int((yr == y).sum()) for y in D.TRAIN_YEARS},
               "n_races_by_year": {str(y): int(d.loc[yr == y, "rid16"].nunique()) for y in D.TRAIN_YEARS},
               "n_days_by_year": {str(y): int(d.loc[yr == y, "date"].nunique()) for y in D.TRAIN_YEARS},
               "lpq_edges_2021_22": edges, "grid": M.GRID}
        for s, S in Ssets.items():
            des[f"subset_{s}"] = {"share_all": float(S.mean()), "n": int(S.sum()),
                                  "share_by_year": {str(y): float(S[yr == y].mean()) for y in D.TRAIN_YEARS},
                                  "n_by_year_type": {f"{y}_{b}": int((S & (yr == y) & (bt == b)).sum())
                                                     for y in D.TRAIN_YEARS for b in ("tan", "fuku")},
                                  "races_share": float(pd.Series(S).groupby(d["rid16"].to_numpy()).any().mean())}
        des["B_size_ok"] = bool(B_SIZE_RANGE[0] <= des["subset_B"]["share_all"] <= B_SIZE_RANGE[1])
        base_tmp_edges = D.odds_decile_edges(d[tr])
        cell_ev = D.odds_cell(d[ev], base_tmp_edges)
        des["odds_decile_edges_2021"] = base_tmp_edges
        des["cell_counts_2022"] = np.bincount(cell_ev, minlength=E.NCELL).tolist()
        for s, S in Ssets.items():
            des[f"subset_{s}"]["share_by_cell_2022"] = [float(S[ev][cell_ev == c].mean()) for c in range(E.NCELL)]
        for cond, (sname, boost) in CONDITIONS.items():
            S = Ssets[sname] if sname else np.zeros(len(d), bool)
            pt, pf, info = LC2.true_probs(d, L, S, boost)
            er = design_er(d, L, pt, pf)
            e2 = er[ev]
            cm = np.bincount(cell_ev, weights=e2, minlength=E.NCELL) / np.bincount(cell_ev, minlength=E.NCELL)
            dd = {"pi_info": info, "er_summary": er_summary(d, er, S if sname else np.zeros(len(d), bool)),
                  "true_cell_mean_er_2022": [float(x) for x in cm]}
            if sname:
                dd["oracle_2022"] = oracle_power(d, er, S, ev)
                dd["true_excess_of_S_2022"] = float((e2[S[ev]] - cm[cell_ev[S[ev]]]).mean())
            des[cond] = dd
        (OUT / "design_v1_1.json").write_text(json.dumps(des, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps(des, ensure_ascii=False, indent=1))
        return 0
    # --run
    if (OUT / "results.json").exists():
        raise SystemExit("結果が既にある（再実行しない）")
    rec = PREREG_SHA.read_text(encoding="utf-8").split()[0]
    cur = D.sha256(PREREG)
    if rec != cur:
        raise SystemExit(f"事前登録が記録後に変わった: {rec} != {cur}")
    bshare = float(Ssets["B"].mean())
    if not (B_SIZE_RANGE[0] <= bshare <= B_SIZE_RANGE[1]):
        raise SystemExit(f"B の大きさが範囲外: {bshare:.4f}")
    _LOGF = OUT / "log.txt"
    log(f"prereg sha256 {cur} OK; spec {spec_sha[:16]}; A share {Ssets['A'].mean():.4f} B share {bshare:.4f}; "
        f"lpq edges {edges}; train {TRAIN_YEAR} n={int(tr.sum())} eval {EVAL_YEAR} n={int(ev.sum())}")
    X = D.make_X(d, MODE)
    results = []
    tmp = OUT / "results_partial.json"
    for seed in SYNTH_SEEDS:
        for cond in CONDITIONS:
            results.append(run_one(d, L, X, tr, ev, Ssets, None, cond, seed))
            tmp.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    j = judge(results)
    (OUT / "results.json").write_text(json.dumps({"prereg_sha256": cur, "spec_sha256": spec_sha, "lpq_edges": edges,
                                                  "subset_share": {k: float(v.mean()) for k, v in Ssets.items()},
                                                  "judgement": j, "results": results},
                                                 ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    tmp.unlink(missing_ok=True)
    log(f"JUDGEMENT {json.dumps(j, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
