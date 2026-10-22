# -*- coding: utf-8 -*-
"""
train_v2.py — 馬券判断 AI v2 の学習と選択（仕様 v1.0 §5・§6-2・§6-4）
=====================================================================
学習 2021-22、選択 2023（実データ）。4 通り = {主の推定器 two_stage, 比較の推定器 huber} × {PyCaLiAI 版 model, モデル列を外した版 nomodel}。
- ハイパーパラメータは 12 候補から、学習 2021-22 で 1 本ずつ学習した点推定 E[R] と 2023 の実 R のチケット単位 Spearman が最大のもの。
- 選んだ設定で開催日 bootstrap 10 本 → analysis/bet_decision_ai/v2/models/{kind}_{mode}.pkl、manifest.json に sha256。
- 2023 の報告: 購入数・回収率と CI・上位 10 払戻除外・順位相関と CI・十分位・単複別。§6-2 打ち切り判定（主 = two_stage × model）。
前提: 学習器の動作確認（learner_check_v2）の判定が合格であること（out/learner_check/results.json で確認、不合格なら拒否）。
2024 年以降の結果列は読まない（data_v2 のロック）。
実行: python -m analysis.bet_decision_ai.v2.train_v2
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
MODELS = HERE / "models"
MANIFEST = MODELS / "manifest.json"
OUT = HERE / "out" / "train_v2"
LC_RESULTS = HERE / "out" / "learner_check" / "results.json"
VERSIONS = [("two_stage", "model"), ("two_stage", "nomodel"), ("huber", "model"), ("huber", "nomodel")]
PRIMARY = ("two_stage", "model")
CODE_FILES = ["data_v2.py", "model_v2.py", "evaluate_v2.py", "train_v2.py", "confirm_v2.py"]
_LOGF = None


def log(msg: str) -> None:
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    if _LOGF:
        with open(_LOGF, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def vname(kind: str, mode: str) -> str:
    return f"{kind}_{mode}"


def report_period(models: dict, d: pd.DataFrame, R: np.ndarray, label: str) -> dict:
    """models: {vname: [10 本]}。d: build_features 済み、R: 実 R。4 通りの報告と PyCaLiAI 版 − 外した版の差。"""
    date = d["date"].to_numpy()
    bt = d["bet_type"].to_numpy()
    preds = {}
    out = {"label": label, "n_candidates": int(len(d)), "n_races": int(d["rid16"].nunique()),
           "n_days": int(d["date"].nunique()), "versions": {}}
    for kind, mode in VERSIONS:
        v = vname(kind, mode)
        X = D.make_X(d, mode)
        P = M.predict_ensemble(models[v], X)
        cons = M.conservative(P)
        pmean = P.mean(0)
        buy = M.buy_mask(cons)
        rep = E.full_report(cons, R, date, bt, buy, with_rho_ci=True, pred_mean=pmean)
        rep["pred_summary"] = {"cons_q": [float(x) for x in np.quantile(cons, [0, .01, .5, .99, .999, 1])],
                               "boot_spread_median": float(np.median(P.max(0) - P.min(0)))}
        out["versions"][v] = rep
        preds[v] = (cons, buy)
        a = rep["all"]
        log(f"  [{label}] {v}: buy={a['n_buy']} roi={a['roi']} ci={a['roi_ci']} ex10={a['roi_ex_top10']} "
            f"rho={a['rho']:.4f} ci={a['rho_ci']}")
    out["model_minus_nomodel"] = {}
    for kind in ("two_stage", "huber"):
        A, B = preds[vname(kind, "model")], preds[vname(kind, "nomodel")]
        out["model_minus_nomodel"][kind] = E.paired_diff(A[0], A[1], B[0], B[1], R, date)
    return out


def learner_check_passed() -> tuple[bool, dict]:
    if not LC_RESULTS.exists():
        return False, {"reason": "learner check results missing"}
    j = json.loads(LC_RESULTS.read_text(encoding="utf-8")).get("judgement", {})
    return bool(j.get("two_stage", {}).get("overall")), j


def main() -> int:
    global _LOGF
    ap = argparse.ArgumentParser()
    ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    _LOGF = OUT / "log.txt"
    spec_sha = D.check_spec()
    ok, j = learner_check_passed()
    if not ok:
        raise SystemExit(f"学習器の動作確認が合格していないので実データに進まない: {j}")
    log("load 2021-2023 with results (2024+ locked)")
    df = D.load_rows((2021, 2023), with_results=True)
    d, fstats = D.build_features(df)
    R = D.payout_R(d)
    yr = d["year"].to_numpy()
    tr = (yr >= D.TRAIN_YEARS[0]) & (yr <= D.TRAIN_YEARS[1])
    se = yr == D.SELECT_YEAR
    day = d["date"].to_numpy()
    dse = d[se].reset_index(drop=True)
    Rse = R[se]
    models, chosen, grids = {}, {}, {}
    man = {"created": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "spec_sha256": spec_sha,
           "substrate_sha256": D.sha256(D.SUB), "base_parquet_sha256": D.sha256(D.BASE_PQ),
           "lambda_sha256": D.sha256(D.LAMBDA),
           "learner_check": {"results_sha256": D.sha256(LC_RESULTS), "judgement_two_stage_overall": True},
           "code_sha256": {f: D.sha256(HERE / f) for f in CODE_FILES if (HERE / f).exists()},
           "train_years": list(D.TRAIN_YEARS), "select_year": D.SELECT_YEAR, "n_boot": M.N_BOOT, "boot_q": M.BOOT_Q,
           "buy_threshold": M.BUY_THRESHOLD, "calibration_windows": None, "versions": {}}
    for kind, mode in VERSIONS:
        v = vname(kind, mode)
        X = D.make_X(d, mode)
        Xtr, Xse = X[tr].reset_index(drop=True), X[se].reset_index(drop=True)
        log(f"== {v}: features {list(X.columns)}")
        rows = []
        for ci, cfg in enumerate(M.GRID):
            t0 = time.time()
            m = M.ESTIMATORS[kind]().fit(Xtr, R[tr], day[tr], cfg, seed=0)
            rho = E._spearman(m.predict(Xse), Rse)
            rows.append({"cfg_id": ci, **cfg, "rho_select": rho, "sec": round(time.time() - t0, 1)})
            log(f"  grid cfg{ci:02d} {cfg} rho2023={rho:.5f} ({time.time() - t0:.0f}s)")
        best = max(rows, key=lambda r: (r["rho_select"], -r["cfg_id"]))
        cfg = M.GRID[best["cfg_id"]]
        chosen[v], grids[v] = cfg, rows
        log(f"  chosen cfg{best['cfg_id']:02d} {cfg}")
        ms = M.fit_bootstrap(kind, Xtr, R[tr], day[tr], cfg, base_seed=0, log=log)
        blob = M.dump_models(ms)
        p = MODELS / f"{v}.pkl"
        p.write_bytes(blob)
        models[v] = M.load_models(blob)
        man["versions"][v] = {"kind": kind, "mode": mode, "features": list(X.columns), "cfg": cfg,
                              "cfg_id": best["cfg_id"], "file": p.name, "sha256": M.sha256_bytes(blob),
                              "member_info": [m.info for m in ms]}
    man["calibration_windows"] = D.calibrator_info()
    man["feature_stats_2021_23"] = fstats
    MANIFEST.write_text(json.dumps(man, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    log(f"manifest written: {MANIFEST}")
    rep = report_period(models, dse, Rse, "select_2023")
    rep["grids"] = grids
    rep["chosen"] = chosen
    rep["select_judgement"] = {v: E.judge_select(rep["versions"][v]) for v in rep["versions"]}
    rep["abort"] = rep["select_judgement"][vname(*PRIMARY)]["abort"]
    rep["primary"] = vname(*PRIMARY)
    rep["manifest_sha256"] = D.sha256(MANIFEST)
    (OUT / "select_2023.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    log(f"select judgement: {json.dumps(rep['select_judgement'], ensure_ascii=False)}")
    log(f"ABORT (primary {rep['primary']}): {rep['abort']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
