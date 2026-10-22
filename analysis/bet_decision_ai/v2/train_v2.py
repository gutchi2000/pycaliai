# -*- coding: utf-8 -*-
"""
train_v2.py — 馬券判断 AI v2 仕様 v1.1（§10）の学習と選択
==========================================================
学習 2021-22、選択 2023（実データ）。2 通り = 主の推定器 Huber 直接回帰 × {PyCaLiAI 版 model, モデル列を外した版 nomodel}。
（v1.0 の 2 段推定器は廃止。v1.0 用に書いた本スクリプトの版は一度も実行していない。）
- ハイパーパラメータ: 学習期間 2021-22 内の開催日 3 分割 CV の Huber 損失が最小の設定（model_v2.select_cfg_cv）。2023 の結果は使わない。
- 選んだ設定で開催日 bootstrap 10 本 → models/huber_{mode}_v1_1.pkl、models/manifest_v1_1.json に sha256。
- 超過回収のセル境界（券種 × 判断時点オッズ十分位）は学習期間 2021-22 の候補で決め、manifest に記録。
- 2023 の報告: 超過回収と CI・回収率と CI・購入数・上位 10 払戻除外・十分位・単複別・セル別・順位相関（記述のみ）、
  PyCaLiAI 版 − 外した版の差。§10 §6-2 打ち切り判定（主 = huber × model）。
前提: v1.1 の学習器確認（learner_check_v21）が合格し、指標の確認も満たすこと（不合格なら拒否）。
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
MANIFEST = MODELS / "manifest_v1_1.json"
OUT = HERE / "out" / "train_v2_1"
SELECT_JSON = OUT / "select_2023_v1_1.json"
LC_RESULTS = HERE / "out" / "learner_check_v1_1" / "results.json"
KIND = "huber"
VERSIONS = [(KIND, "model"), (KIND, "nomodel")]
PRIMARY = (KIND, "model")
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


def report_period(models: dict, d: pd.DataFrame, R: np.ndarray, odds_edges: dict, label: str,
                  with_diff: bool = True) -> dict:
    """models: {vname: [10 本]}。d: build_features 済み、R: 実 R。2 通りの報告と PyCaLiAI 版 − 外した版の差。"""
    date = d["date"].to_numpy()
    bt = d["bet_type"].to_numpy()
    cell = D.odds_cell(d, odds_edges)
    preds = {}
    out = {"label": label, "n_candidates": int(len(d)), "n_races": int(d["rid16"].nunique()),
           "n_days": int(d["date"].nunique()), "versions": {}}
    for kind, mode in VERSIONS:
        v = vname(kind, mode)
        X = D.make_X(d, mode)
        P = M.predict_ensemble(models[v], X)
        cons = M.conservative(P)
        buy = M.buy_mask(cons)
        rep = E.full_report_v11(cons, R, date, bt, buy, cell, pred_mean=P.mean(0))
        rep["pred_summary"] = {"cons_q": [float(x) for x in np.quantile(cons, [0, .01, .5, .99, .999, 1])],
                               "boot_spread_median": float(np.median(P.max(0) - P.min(0)))}
        out["versions"][v] = rep
        preds[v] = (cons, buy)
        a = rep["all"]
        log(f"  [{label}] {v}: buy={a['n_buy']} roi={a['roi']} ci={a['roi_ci']} ex10={a['roi_ex_top10']} "
            f"excess={a.get('excess')} ci={a.get('excess_ci')} rho={a['rho']:.4f}")
    if with_diff:
        A, B = preds[vname(KIND, "model")], preds[vname(KIND, "nomodel")]
        dd = E.excess_diff(A[1], B[1], R, date, cell)
        dd["rho"] = E.paired_diff(A[0], A[1], B[0], B[1], R, date)
        out["model_minus_nomodel"] = dd
    return out


def learner_check_passed() -> tuple[bool, dict]:
    if not LC_RESULTS.exists():
        return False, {"reason": "learner check v1.1 results missing"}
    j = json.loads(LC_RESULTS.read_text(encoding="utf-8")).get("judgement", {}).get("huber", {})
    return bool(j.get("overall") and j.get("metric_check_ok")), j


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
        raise SystemExit(f"v1.1 の学習器確認が合格していないので実データに進まない: {j}")
    if SELECT_JSON.exists():
        raise SystemExit("v1.1 の選択結果が既にある（上書きしない）")
    log(f"spec {spec_sha[:16]} OK; load 2021-2023 with results (2024+ locked)")
    df = D.load_rows((2021, 2023), with_results=True)
    d, fstats = D.build_features(df)
    R = D.payout_R(d)
    yr = d["year"].to_numpy()
    tr = (yr >= D.TRAIN_YEARS[0]) & (yr <= D.TRAIN_YEARS[1])
    se = yr == D.SELECT_YEAR
    day = d["date"].to_numpy()
    odds_edges = D.odds_decile_edges(d[tr])
    models, chosen, grids = {}, {}, {}
    man = {"spec_version": "v1.1", "created": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "spec_sha256": spec_sha,
           "substrate_sha256": D.sha256(D.SUB), "base_parquet_sha256": D.sha256(D.BASE_PQ),
           "lambda_sha256": D.sha256(D.LAMBDA),
           "learner_check_v1_1": {"results_sha256": D.sha256(LC_RESULTS), "overall": True, "metric_check_ok": True},
           "code_sha256": {f: D.sha256(HERE / f) for f in CODE_FILES if (HERE / f).exists()},
           "train_years": list(D.TRAIN_YEARS), "select_year": D.SELECT_YEAR, "n_boot": M.N_BOOT, "boot_q": M.BOOT_Q,
           "buy_threshold": M.BUY_THRESHOLD, "huber_delta": M.HUBER_DELTA, "clip_q": list(M.CLIP_Q),
           "odds_decile_edges": odds_edges, "versions": {}}
    for kind, mode in VERSIONS:
        v = vname(kind, mode)
        X = D.make_X(d, mode)
        Xtr = X[tr].reset_index(drop=True)
        log(f"== {v}: {len(X.columns)} features {list(X.columns)}")
        t0 = time.time()
        cfg, rows = M.select_cfg_cv(Xtr, R[tr], day[tr], seed=0, log=log)
        best = [r for r in rows if {k: r[k] for k in cfg} == cfg][0]
        chosen[v], grids[v] = cfg, rows
        log(f"  chosen cfg{best['cfg_id']:02d} {cfg} cv_loss={best['cv_huber_loss']:.6f} ({time.time() - t0:.0f}s)")
        ms = M.fit_bootstrap(kind, Xtr, R[tr], day[tr], cfg, base_seed=0, log=log)
        blob = M.dump_models(ms)
        p = MODELS / f"{v}_v1_1.pkl"
        p.write_bytes(blob)
        models[v] = M.load_models(blob)
        man["versions"][v] = {"kind": kind, "mode": mode, "features": list(X.columns), "cfg": cfg,
                              "cfg_id": best["cfg_id"], "cv_grid": rows, "file": p.name,
                              "sha256": M.sha256_bytes(blob), "member_info": [m.info for m in ms]}
    man["calibration_windows"] = D.calibrator_info()
    man["feature_stats_2021_23"] = fstats
    MANIFEST.write_text(json.dumps(man, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    log(f"manifest written: {MANIFEST} sha256 {D.sha256(MANIFEST)}")
    dse = d[se].reset_index(drop=True)
    rep = report_period(models, dse, R[se], odds_edges, "select_2023")
    rep["chosen"] = chosen
    rep["select_judgement"] = {v: E.judge_select_v11(rep["versions"][v]) for v in rep["versions"]}
    rep["primary"] = vname(*PRIMARY)
    rep["abort"] = rep["select_judgement"][rep["primary"]]["abort"]
    rep["manifest_sha256"] = D.sha256(MANIFEST)
    SELECT_JSON.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    log(f"select judgement: {json.dumps(rep['select_judgement'], ensure_ascii=False)}")
    log(f"ABORT (primary {rep['primary']}): {rep['abort']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
