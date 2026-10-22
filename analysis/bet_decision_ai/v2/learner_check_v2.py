# -*- coding: utf-8 -*-
"""
learner_check_v2.py — 馬券判断 AI v2 の学習器の動作確認（仕様 v1.0 §6-1、合成データ）
=====================================================================================
事前登録: docs/research/BET_DECISION_AI_V2_LEARNER_CHECK_PREREG_20261023.md（sha256 を out/learner_check/prereg_sha256.txt に記録）

実際の 2021-23 の候補と特徴（data_v2.load_rows(with_results=False) → build_features）をそのまま使い、払戻だけを合成する。
実データの結果列 res_* は一切読まない（with_results=False で読み込むので列そのものが無い）。

合成（v1 learner_check/synth.py の作り方を全出走馬に広げたもの）:
  単勝: レース内の全出走馬から 1 頭だけ勝つカテゴリ抽選。基本の勝率 π_単 = q_tan_mkt（市場が正しい世界、期待回収率 = 1/Σ(1/odds)）。
        埋め込み: 部分集合 S の単勝券の馬は π_単 = boost/odds_tan、残りの馬は q_tan_mkt をレース内で縮めて合計 1 に保つ。
  複勝: 基本 π_複 = 0.8/odds_fuku_mid（期待回収率 80%）、S の複勝券は boost/odds_fuku_mid。π_複 < π_単 なら π_単 に持ち上げる。
        単勝の勝ち馬の複勝は必ず的中、他の馬は (π_複 − π_単)/(1 − π_単) で独立に的中（周辺確率 = π_複 を保つ。的中頭数は固定しない）。
  払戻倍率: 単勝 = odds_tan、複勝 = odds_fuku_mid（判断時点の値で確定）、外れ 0。π の上限 0.97。
  共通乱数: 同じ合成の種では 3 条件で同じ一様乱数。
実行:
  python -m analysis.bet_decision_ai.v2.learner_check_v2 --design      # 設計の数値（抽選も学習もしない）
  python -m analysis.bet_decision_ai.v2.learner_check_v2 --record-prereg # 事前登録の sha256 を記録
  python -m analysis.bet_decision_ai.v2.learner_check_v2 --run          # 本番（事前登録の sha256 を照合してから）
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.bet_decision_ai.v2 import data_v2 as D
from analysis.bet_decision_ai.v2 import evaluate_v2 as E
from analysis.bet_decision_ai.v2 import model_v2 as M

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "learner_check"
PREREG = D.BASE / "docs" / "research" / "BET_DECISION_AI_V2_LEARNER_CHECK_PREREG_20261023.md"
PREREG_SHA = OUT / "prereg_sha256.txt"

# ---- 部分集合 S（事前登録で固定）
SUBSET_LPQ_Q = 0.60          # 学習期間 2021-22 の券種別 log_p_over_q の 60% 点以上
SUBSET_MIN_FIELD = 12        # n_field_s1 ≥ 12
SUBSET_MAX_ODDS_TAN = 10.0   # odds_tan ≤ 10（単勝・複勝とも、その馬の判断時点の単勝オッズ）
FUKU_BASE_ER = 0.80
PI_CAP = 0.97
TAN_S_SUM_CAP = 0.995
CONDITIONS = {"control": None, "e110": 1.10, "e120": 1.20}
SYNTH_SEEDS = [11, 22, 33]
MODE = "model"                # 確認する学習器 = 主の推定器（2 段）、PyCaLiAI 版の入力
_LOGF = None


def log(msg: str) -> None:
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    if _LOGF:
        with open(_LOGF, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ================================================================ 合成
def load_universe() -> pd.DataFrame:
    df = D.load_rows((2021, 2023), with_results=False)
    assert not any(c.startswith("res_") for c in df.columns)
    d, _ = D.build_features(df)
    d = d.sort_values(["rid16", "umaban", "bet_type"]).reset_index(drop=True)
    d["o_bet"] = np.where(d["bet_type"] == "tan", d["odds_tan"], d["odds_fuku_mid"])
    return d


def subset_edges(d: pd.DataFrame) -> dict:
    tr = d[d["year"].between(*D.TRAIN_YEARS)]
    return {bt: float(tr.loc[tr["bet_type"] == bt, "log_p_over_q"].quantile(SUBSET_LPQ_Q)) for bt in ("tan", "fuku")}


def subset_mask(d: pd.DataFrame, edges: dict) -> np.ndarray:
    e = d["bet_type"].map(edges).to_numpy(float)
    return ((d["log_p_over_q"].to_numpy(float) >= e) & (d["n_field_s1"].to_numpy(float) >= SUBSET_MIN_FIELD)
            & (d["odds_tan"].to_numpy(float) <= SUBSET_MAX_ODDS_TAN))


class Layout:
    """単勝行と複勝行を馬で対応づけ、レース境界を持つ。"""

    def __init__(self, d: pd.DataFrame):
        tan = np.flatnonzero(d["bet_type"].to_numpy() == "tan")
        fuku = np.flatnonzero(d["bet_type"].to_numpy() == "fuku")
        assert len(tan) == len(fuku)
        assert (d["rid16"].to_numpy()[tan] == d["rid16"].to_numpy()[fuku]).all()
        assert (d["umaban"].to_numpy()[tan] == d["umaban"].to_numpy()[fuku]).all()
        self.tan, self.fuku = tan, fuku
        rid = d["rid16"].to_numpy()[tan]
        self.races, self.inv = np.unique(rid, return_inverse=True)
        self.starts = np.flatnonzero(np.r_[True, rid[1:] != rid[:-1]])
        assert (np.diff(self.inv) >= 0).all()


def true_probs(d: pd.DataFrame, L: Layout, S: np.ndarray, boost: float | None) -> tuple[np.ndarray, np.ndarray, dict]:
    """馬ごとの π_単、π_複（L.tan の順）。"""
    q = d["q_tan_mkt"].to_numpy(float)[L.tan]
    ot = d["odds_tan"].to_numpy(float)[L.tan]
    om = d["odds_fuku_mid"].to_numpy(float)[L.fuku]
    St, Sf = S[L.tan], S[L.fuku]
    info = {}
    if boost is None:
        pt = q.copy()
        pf = FUKU_BASE_ER / om
    else:
        ps = np.minimum(boost / ot, PI_CAP)
        sum_s = np.bincount(L.inv, weights=np.where(St, ps, 0.0), minlength=len(L.races))
        scale_s = np.where(sum_s > TAN_S_SUM_CAP, TAN_S_SUM_CAP / np.maximum(sum_s, 1e-12), 1.0)
        info["n_races_tan_S_sum_scaled"] = int((sum_s > TAN_S_SUM_CAP).sum())
        ps = ps * scale_s[L.inv]
        sum_s = np.bincount(L.inv, weights=np.where(St, ps, 0.0), minlength=len(L.races))
        sum_q_ns = np.bincount(L.inv, weights=np.where(St, 0.0, q), minlength=len(L.races))
        f_ns = (1.0 - sum_s) / np.maximum(sum_q_ns, 1e-12)
        pt = np.where(St, ps, q * f_ns[L.inv])
        pf = np.where(Sf, boost / om, FUKU_BASE_ER / om)
    info["n_pi_tan_capped"] = int((pt > PI_CAP).sum())
    pt = np.minimum(pt, PI_CAP)
    info["n_pi_fuku_capped"] = int((pf > PI_CAP).sum())
    pf = np.minimum(pf, PI_CAP)
    lift = pf < pt
    info["n_fuku_lifted_to_tan"] = int(lift.sum())
    info["n_fuku_lifted_er_gt_1"] = int((lift & (pt * om > 1.0)).sum())
    pf = np.maximum(pf, pt)
    s_tan_sum = np.bincount(L.inv, weights=pt, minlength=len(L.races))
    info["tan_pi_race_sum"] = [float(s_tan_sum.min()), float(s_tan_sum.max())]
    return pt, pf, info


def design_er(d: pd.DataFrame, L: Layout, S: np.ndarray, pt: np.ndarray, pf: np.ndarray) -> dict:
    er = np.zeros(len(d))
    er[L.tan] = pt * d["odds_tan"].to_numpy(float)[L.tan]
    er[L.fuku] = pf * d["odds_fuku_mid"].to_numpy(float)[L.fuku]
    out = {}
    for y in (2021, 2022, 2023):
        m = d["year"].to_numpy() == y
        out[str(y)] = {"S_mean": float(er[m & S].mean()), "nonS_mean": float(er[m & ~S].mean()),
                       "nonS_max": float(er[m & ~S].max()), "nonS_share_gt_1": float((er[m & ~S] > 1.0).mean())}
    return {"by_year": out, "er": er}


def draw(d: pd.DataFrame, L: Layout, pt: np.ndarray, pf: np.ndarray, seed: int) -> np.ndarray:
    """合成の払戻倍率 R（d の行順）。"""
    rng = np.random.default_rng(seed)
    U = rng.random(len(L.races))
    V = rng.random(len(L.tan))
    cum = np.cumsum(pt)
    base = np.r_[0.0, cum][L.starts][L.inv]                     # レース先頭までの累積
    within = cum - base                                          # レース内の累積
    rs = np.bincount(L.inv, weights=pt, minlength=len(L.races))
    u = U[L.inv] * rs[L.inv]
    prev = within - pt
    last = np.r_[L.starts[1:], len(L.tan)] - 1
    within = within.copy()
    within[last] = np.inf
    win = (u >= prev) & (u < within)
    # 各レースでちょうど 1 頭（浮動小数の端は最後の馬に寄せる）
    nwin = np.bincount(L.inv, weights=win.astype(float), minlength=len(L.races))
    assert (nwin == 1).all(), (nwin.min(), nwin.max())
    r = np.clip((pf - pt) / np.maximum(1.0 - pt, 1e-12), 0.0, 1.0)
    fh = win | (V < r)
    R = np.zeros(len(d))
    R[L.tan] = np.where(win, d["odds_tan"].to_numpy(float)[L.tan], 0.0)
    R[L.fuku] = np.where(fh, d["odds_fuku_mid"].to_numpy(float)[L.fuku], 0.0)
    return R


# ================================================================ 学習器の手順（実データと同じ）
def select_cfg(kind: str, Xtr, Rtr, daytr, Xse, Rse, log_prefix="") -> tuple[dict, list]:
    rows = []
    for ci, cfg in enumerate(M.GRID):
        t0 = time.time()
        m = M.ESTIMATORS[kind]().fit(Xtr, Rtr, daytr, cfg, seed=0)
        rho = E._spearman(m.predict(Xse), Rse)
        rows.append({"cfg_id": ci, **cfg, "rho_select": rho, "sec": round(time.time() - t0, 1)})
        log(f"{log_prefix}  grid {kind} cfg{ci:02d} {cfg} rho={rho:.5f} ({time.time() - t0:.0f}s)")
    best = max(rows, key=lambda r: (r["rho_select"], -r["cfg_id"]))
    return M.GRID[best["cfg_id"]], rows


def run_one(d, L, X, tr, se, S, cond, boost, seed, with_huber=True) -> dict:
    pt, pf, info = true_probs(d, L, S, boost)
    de = design_er(d, L, S, pt, pf)
    R = draw(d, L, pt, pf, seed)
    day = d["date"].to_numpy()
    Xtr, Xse = X[tr].reset_index(drop=True), X[se].reset_index(drop=True)
    Rtr, Rse = R[tr], R[se]
    out = {"cond": cond, "boost": boost, "seed": seed, "pi_info": info, "design_er": de["by_year"]}
    Sse = S[se]
    bt = d["bet_type"].to_numpy()[se]
    dse = day[se]
    # 神託（S を知っていて S をすべて買う）
    out["oracle_buy_S"] = E.roi_stats(Rse, Sse, dse) if boost else None
    for kind in (["two_stage", "huber"] if with_huber else ["two_stage"]):
        pref = f"[{cond} s{seed}]"
        cfg, grid = select_cfg(kind, Xtr, Rtr, day[tr], Xse, Rse, pref)
        log(f"{pref} {kind} chosen {cfg}")
        models = M.fit_bootstrap(kind, Xtr, Rtr, day[tr], cfg, base_seed=0)
        P = M.predict_ensemble(models, Xse)
        cons = M.conservative(P)
        buy = M.buy_mask(cons)
        rep = E.roi_stats(Rse, buy, dse)
        rho = E.spearman_stats(cons, Rse, dse, reps=500)
        nb = int(buy.sum())
        res = {"cfg": cfg, "grid": grid, **rep, **rho,
               "subset_share_of_buys": float(Sse[buy].mean()) if nb else None,
               "subset_recall": float(buy[Sse].mean()) if Sse.any() else None,
               "buy_by_bet_type": {b: int(buy[bt == b].sum()) for b in ("tan", "fuku")},
               "cons_mean_S": float(cons[Sse].mean()), "cons_mean_nonS": float(cons[~Sse].mean()),
               "true_er_mean_of_buys": float(de["er"][se][buy].mean()) if nb else None}
        if cond == "control":
            res["pass"] = bool(rep["buy_share"] <= 0.01 and (nb == 0 or rep["roi_ci"][0] <= 1.0))
        else:
            res["pass"] = bool(nb > 0 and rep["roi_ci"][0] > 1.0 and res["subset_share_of_buys"] >= 0.5)
        log(f"{pref} {kind} buys={nb} share={rep['buy_share']:.4f} roi={rep['roi']} ci={rep['roi_ci']} "
            f"S_share={res['subset_share_of_buys']} rho={rho['rho']:.4f} pass={res['pass']}")
        out[kind] = res
    return out


def judge(results: list) -> dict:
    j = {}
    for kind in ("two_stage", "huber"):
        if not all(kind in r for r in results):
            continue
        jj = {}
        for cond in CONDITIONS:
            ps = [r[kind]["pass"] for r in results if r["cond"] == cond]
            need = len(ps) if cond == "control" else 2
            jj[cond] = {"passes": ps, "n_pass": int(sum(ps)), "need": need, "ok": bool(sum(ps) >= need)}
        jj["overall"] = bool(all(v["ok"] for v in jj.values()))
        j[kind] = jj
    return j


def main() -> int:
    global _LOGF
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", action="store_true")
    ap.add_argument("--record-prereg", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--no-huber", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    D.check_spec()
    if a.record_prereg:
        s = D.sha256(PREREG)
        PREREG_SHA.write_text(f"{s}  {PREREG.name}  recorded {datetime.now():%Y-%m-%d %H:%M:%S}\n", encoding="utf-8")
        print(s)
        return 0
    d = load_universe()
    L = Layout(d)
    edges = subset_edges(d)
    S = subset_mask(d, edges)
    if a.design:
        des = {"n_candidates": int(len(d)), "edges": edges, "subset_share": float(S.mean()),
               "subset_share_by_year": {str(k): float(v) for k, v in pd.Series(S).groupby(d["year"].to_numpy()).mean().items()},
               "subset_n_by_type": {b: int(S[d["bet_type"].to_numpy() == b].sum()) for b in ("tan", "fuku")},
               "subset_races_share": float(pd.Series(S).groupby(d["rid16"].to_numpy()).any().mean()),
               "n_2023": int((d["year"] == 2023).sum()), "n_S_2023": int(S[d["year"].to_numpy() == 2023].sum())}
        se = d["year"].to_numpy() == 2023
        for cond, boost in CONDITIONS.items():
            pt, pf, info = true_probs(d, L, S, boost)
            de = design_er(d, L, S, pt, pf)
            o = d["o_bet"].to_numpy(float)
            er = de["er"]
            pi = er / o
            v = (pi * o * o - (pi * o) ** 2)[se & S].sum()
            nS = int((se & S).sum())
            des[cond] = {"pi_info": info, "design_er": de["by_year"],
                         "oracle_buy_S_2023_expected_roi": float(er[se & S].mean()),
                         "oracle_roi_se_indep": float(np.sqrt(v) / nS),
                         "oracle_z": float((er[se & S].mean() - 1.0) / (np.sqrt(v) / nS))}
        (OUT / "design.json").write_text(json.dumps(des, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps(des, ensure_ascii=False, indent=1))
        return 0
    if a.run:
        rec = PREREG_SHA.read_text(encoding="utf-8").split()[0]
        cur = D.sha256(PREREG)
        if rec != cur:
            raise SystemExit(f"事前登録が記録後に変わった: {rec} != {cur}")
        _LOGF = OUT / "log.txt"
        log(f"prereg sha256 {cur} OK; subset share {S.mean():.4f} edges {edges}")
        X = D.make_X(d, MODE)
        yr = d["year"].to_numpy()
        tr = (yr >= D.TRAIN_YEARS[0]) & (yr <= D.TRAIN_YEARS[1])
        se = yr == D.SELECT_YEAR
        results = []
        for seed in SYNTH_SEEDS:
            for cond, boost in CONDITIONS.items():
                t0 = time.time()
                r = run_one(d, L, X, tr, se, S, cond, boost, seed, with_huber=not a.no_huber)
                r["sec"] = round(time.time() - t0, 1)
                results.append(r)
                (OUT / "results.json").write_text(json.dumps({"results": results}, ensure_ascii=False, indent=1,
                                                             default=float), encoding="utf-8")
        j = judge(results)
        (OUT / "results.json").write_text(json.dumps({"prereg_sha256": cur, "edges": edges,
                                                      "subset_share": float(S.mean()), "judgement": j,
                                                      "results": results}, ensure_ascii=False, indent=1, default=float),
                                          encoding="utf-8")
        log(f"JUDGEMENT {json.dumps(j, ensure_ascii=False)}")
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
