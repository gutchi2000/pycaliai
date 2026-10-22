# -*- coding: utf-8 -*-
"""
learner_check_v21.py — 馬券判断 AI v2 仕様 v1.1（§10）の学習器の動作確認（合成データ）
=====================================================================================
事前登録: docs/research/BET_DECISION_AI_V2_1_LEARNER_CHECK_PREREG_20261023.md（sha256 を out/learner_check_v1_1/prereg_sha256.txt）
合成データ・部分集合 S・条件・種は v1.0 と同じ（learner_check_v2 の関数をそのまま使う）。v1.0 の出力（out/learner_check/）は触らない。
推定器は Huber 直接回帰のみ（2 段は廃止）。ハイパーパラメータは学習期間 2021-22 内の開催日 3 分割 CV の Huber 損失で選ぶ。
判定量に超過回収（購入券の R − 同じ期間・券種・判断時点オッズ十分位の全候補平均 R）を加える。
実データの結果列 res_* は読まない。
実行:
  python -m analysis.bet_decision_ai.v2.learner_check_v21 --design
  python -m analysis.bet_decision_ai.v2.learner_check_v21 --record-prereg
  python -m analysis.bet_decision_ai.v2.learner_check_v21 --run
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from analysis.bet_decision_ai.v2 import data_v2 as D
from analysis.bet_decision_ai.v2 import evaluate_v2 as E
from analysis.bet_decision_ai.v2 import learner_check_v2 as LC
from analysis.bet_decision_ai.v2 import model_v2 as M

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "learner_check_v1_1"
PREREG = D.BASE / "docs" / "research" / "BET_DECISION_AI_V2_1_LEARNER_CHECK_PREREG_20261023.md"
PREREG_SHA = OUT / "prereg_sha256.txt"
FORCED_TOP = (0.01, 0.05)       # 指標の確認: 控えめな採算の上位 1%・5% を強制的に「買った」ときの超過回収
_LOGF = None


def log(msg: str) -> None:
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    if _LOGF:
        with open(_LOGF, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def run_one(d, L, X, tr, se, S, cell_se, cond, boost, seed) -> dict:
    pt, pf, info = LC.true_probs(d, L, S, boost)
    de = LC.design_er(d, L, S, pt, pf)
    R = LC.draw(d, L, pt, pf, seed)
    day = d["date"].to_numpy()
    Xtr, Xse = X[tr].reset_index(drop=True), X[se].reset_index(drop=True)
    Rtr, Rse = R[tr], R[se]
    dse = day[se]
    Sse = S[se]
    bt = d["bet_type"].to_numpy()[se]
    er_se = de["er"][se]
    pref = f"[{cond} s{seed}]"
    out = {"cond": cond, "boost": boost, "seed": seed, "pi_info": info, "design_er": de["by_year"]}
    if boost:
        o = E.roi_stats(Rse, Sse, dse)
        o.update(E.excess_stats(Rse, Sse, dse, cell_se))
        out["oracle_buy_S"] = o
    cfg, grid = M.select_cfg_cv(Xtr, Rtr, day[tr], seed=0)
    log(f"{pref} chosen {cfg} (cv loss {min(g['cv_huber_loss'] for g in grid):.6f})")
    models = M.fit_bootstrap("huber", Xtr, Rtr, day[tr], cfg, base_seed=0)
    P = M.predict_ensemble(models, Xse)
    cons = M.conservative(P)
    buy = M.buy_mask(cons)
    nb = int(buy.sum())
    res = {"cfg": cfg, "grid": grid, **E.roi_stats(Rse, buy, dse), **E.excess_stats(Rse, buy, dse, cell_se),
           **E.spearman_stats(cons, Rse, dse, reps=500),
           "subset_share_of_buys": float(Sse[buy].mean()) if nb else None,
           "subset_recall": float(buy[Sse].mean()),
           "buy_by_bet_type": {b: int(buy[bt == b].sum()) for b in ("tan", "fuku")},
           "true_er_mean_of_buys": float(er_se[buy].mean()) if nb else None}
    # 指標の確認（判定外の記述＋停止条件）: 上位 k を強制購入
    forced = {}
    order = np.argsort(-cons, kind="stable")
    for k in FORCED_TOP:
        fb = np.zeros(len(cons), bool)
        fb[order[: int(round(k * len(cons)))]] = True
        f = {**E.roi_stats(Rse, fb, dse), **E.excess_stats(Rse, fb, dse, cell_se)}
        cm = np.bincount(cell_se, weights=er_se, minlength=E.NCELL) / np.maximum(np.bincount(cell_se, minlength=E.NCELL), 1)
        f["true_excess"] = float((er_se[fb] - cm[cell_se[fb]]).mean())
        f["subset_share"] = float(Sse[fb].mean())
        forced[f"top{int(k * 100)}pct"] = {kk: f[kk] for kk in ("n_buy", "roi", "roi_ci", "excess", "excess_ci",
                                                                 "excess_ci_contains_0", "true_excess", "subset_share")}
    res["forced_top"] = forced
    if cond == "control":
        res["pass"] = bool(res["buy_share"] <= 0.01 and (nb == 0 or res["excess_ci_contains_0"]))
    else:
        res["pass"] = bool(nb > 0 and res["roi_ci"][0] > 1.0 and res["subset_share_of_buys"] >= 0.5
                           and res["excess_ci"][0] > 0)
    log(f"{pref} buys={nb} share={res['buy_share']:.4f} roi={res['roi']} ci={res['roi_ci']} excess={res['excess']} "
        f"ci={res['excess_ci']} S_share={res['subset_share_of_buys']} pass={res['pass']} | forced "
        + " ".join(f"{k}: ex={v['excess']:.4f} ci={[round(x, 4) for x in v['excess_ci']]} true={v['true_excess']:.4f}"
                   for k, v in forced.items()))
    out["huber"] = res
    return out


def judge(results: list) -> dict:
    jj = {}
    for cond in LC.CONDITIONS:
        ps = [r["huber"]["pass"] for r in results if r["cond"] == cond]
        need = len(ps) if cond == "control" else 2
        jj[cond] = {"passes": ps, "n_pass": int(sum(ps)), "need": need, "ok": bool(sum(ps) >= need)}
    jj["overall"] = bool(all(v["ok"] for v in jj.values()))
    ctrl = [r for r in results if r["cond"] == "control"]
    jj["metric_check_control_forced_top1_contains_0"] = [r["huber"]["forced_top"]["top1pct"]["excess_ci_contains_0"]
                                                         for r in ctrl]
    jj["metric_check_ok"] = bool(all(jj["metric_check_control_forced_top1_contains_0"]))
    return {"huber": jj}


def main() -> int:
    global _LOGF
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", action="store_true")
    ap.add_argument("--record-prereg", action="store_true")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    D.check_spec()
    if a.record_prereg:
        s = D.sha256(PREREG)
        PREREG_SHA.write_text(f"{s}  {PREREG.name}  recorded {datetime.now():%Y-%m-%d %H:%M:%S}\n", encoding="utf-8")
        print(s)
        return 0
    d = LC.load_universe()
    L = LC.Layout(d)
    edges = LC.subset_edges(d)
    S = LC.subset_mask(d, edges)
    yr = d["year"].to_numpy()
    tr = (yr >= D.TRAIN_YEARS[0]) & (yr <= D.TRAIN_YEARS[1])
    se = yr == D.SELECT_YEAR
    oedges = D.odds_decile_edges(d[tr])
    cell_se = D.odds_cell(d[se], oedges)
    if a.design:
        des = {"subset_share": float(S.mean()), "subset_edges": edges, "odds_decile_edges_2021_22": oedges,
               "cell_counts_2023": np.bincount(cell_se, minlength=E.NCELL).tolist(),
               "S_share_by_cell_2023": [float(S[se][cell_se == c].mean()) for c in range(E.NCELL)]}
        for cond, boost in LC.CONDITIONS.items():
            pt, pf, _ = LC.true_probs(d, L, S, boost)
            er = LC.design_er(d, L, S, pt, pf)["er"][se]
            cm = np.bincount(cell_se, weights=er, minlength=E.NCELL) / np.bincount(cell_se, minlength=E.NCELL)
            des[cond] = {"true_cell_mean_er": [float(x) for x in cm],
                         "true_excess_of_S": float((er[S[se]] - cm[cell_se[S[se]]]).mean()) if boost else None}
        (OUT / "design.json").write_text(json.dumps(des, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps(des, ensure_ascii=False, indent=1))
        return 0
    if a.run:
        rec = PREREG_SHA.read_text(encoding="utf-8").split()[0]
        cur = D.sha256(PREREG)
        if rec != cur:
            raise SystemExit(f"事前登録が記録後に変わった: {rec} != {cur}")
        _LOGF = OUT / "log.txt"
        log(f"prereg sha256 {cur} OK; subset share {S.mean():.4f}; odds edges {oedges}")
        X = D.make_X(d, LC.MODE)
        results = []
        for seed in LC.SYNTH_SEEDS:
            for cond, boost in LC.CONDITIONS.items():
                t0 = time.time()
                r = run_one(d, L, X, tr, se, S, cell_se, cond, boost, seed)
                r["sec"] = round(time.time() - t0, 1)
                results.append(r)
                (OUT / "results.json").write_text(json.dumps({"results": results}, ensure_ascii=False, indent=1,
                                                             default=float), encoding="utf-8")
        j = judge(results)
        (OUT / "results.json").write_text(json.dumps({"prereg_sha256": cur, "spec_sha256": D.sha256(D.SPEC),
                                                      "subset_edges": edges, "odds_decile_edges": oedges,
                                                      "judgement": j, "results": results},
                                                     ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        log(f"JUDGEMENT {json.dumps(j, ensure_ascii=False)}")
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
