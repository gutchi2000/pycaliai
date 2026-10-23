# -*- coding: utf-8 -*-
"""
train_v3.py — 馬券判断 AI v3 の実データ学習（仕様 v1.0 §3-2、2021-22 のみ）
==========================================================================
- 前提: 学習器確認（learner_check_v3）が合格し、対照の指標確認も満たすこと（out/learner_check/results.json の
  judgement.proceed_to_real_data）。満たさなければ拒否。
- 2021-22 の実データ（結果列あり）。2023 年以降の結果列は data_v3 のロックで読めない。
- 2 通り: 主（PyCaLiAI 版 = model）と、モデル列を外した版（nomodel）。どちらも
  ハイパーパラメータ = 2021-22 内の開催日 3 分割 CV の y の二乗誤差最小（model_v3.select_cfg_cv）、
  開催日 bootstrap 10 本（各回で R̄_cell・境界・打ち切り点を作り直す、仕様 v1.1 §6）→ models/sq_excess_{mode}_v3.pkl。
  manifest の base（2021-22 全体の境界）は超過回収のセル定義（評価期間でセル平均を計算）にだけ使う。
- 記述（判定ではない）: 同じ 3 分割の交差予測（各保持側を残り 2 つで作り直した基準・10 本で予測）での購入数・回収率・
  超過回収・十分位・単複別・年別。ハイパーパラメータを同じ分割で選んでいるので in-sample に近い。
- manifest.json に仕様・データ・コード・モデル・交差予測ファイルの sha256。
実行: python -m analysis.bet_decision_ai.v3.train_v3
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.bet_decision_ai.v3 import data_v3 as D
from analysis.bet_decision_ai.v3 import evaluate_v3 as E
from analysis.bet_decision_ai.v3 import model_v3 as M

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
MANIFEST = MODELS / "manifest.json"
OUT = HERE / "out" / "train"
OOF_PQ = OUT / "oof_2021_22.parquet"
OOF_JSON = OUT / "oof_report_2021_22.json"
LC_RESULTS = HERE / "out" / "learner_check" / "results.json"
MODES = ["model", "nomodel"]
PRIMARY = "model"
CODE_FILES = ["data_v3.py", "model_v3.py", "evaluate_v3.py", "learner_check_v3.py", "train_v3.py", "confirm_v3.py"]
V2_DEPS = {"evaluate_v2.py": D.BASE / "analysis" / "bet_decision_ai" / "v2" / "evaluate_v2.py",
           "learner_check_v2.py": D.BASE / "analysis" / "bet_decision_ai" / "v2" / "learner_check_v2.py",
           "build_substrate.py(v1)": D.BASE / "analysis" / "bet_decision_ai" / "build_substrate.py"}
_LOGF = None


def log(msg: str) -> None:
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    if _LOGF:
        with open(_LOGF, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def vname(mode: str) -> str:
    return f"sq_excess_{mode}"


def report_preds(preds: dict, d: pd.DataFrame, R: np.ndarray, edges: dict, label: str, with_diff: bool = True,
                 with_rho_ci: bool = True) -> dict:
    """preds: {mode: (cons, mean)}。d: 特徴量表、R: 実 R。超過回収のセルは学習期間の境界（edges）。"""
    date = d["date"].astype(str).to_numpy()
    bt = d["bet_type"].astype(str).to_numpy()
    cell = D.odds_cell(d, edges)
    out = {"label": label, "n_candidates": int(len(d)), "n_races": int(d["rid16"].nunique()),
           "n_days": int(d["date"].nunique()), "versions": {}}
    buys = {}
    for mode, (cons, mean) in preds.items():
        buy = M.buy_mask(cons)
        rep = E.full_report(cons, R, date, bt, buy, cell, pred_mean=mean, with_rho_ci=with_rho_ci)
        rep["pred_summary"] = {"cons_q": [float(x) for x in np.quantile(cons, [0, .01, .5, .99, .999, 1])]}
        out["versions"][vname(mode)] = rep
        buys[mode] = buy
        a = rep["all"]
        log(f"  [{label}] {vname(mode)}: buy={a['n_buy']} roi={a['roi']} ci={a['roi_ci']} ex10={a['roi_ex_top10']} "
            f"excess={a.get('excess')} ci={a.get('excess_ci')} rho={a['rho']:.4f}")
    if with_diff and set(MODES) <= set(preds):
        out["model_minus_nomodel"] = E.excess_diff(buys["model"], buys["nomodel"], R, date, cell)
    return out


def learner_check_passed() -> tuple[bool, dict]:
    if not LC_RESULTS.exists():
        return False, {"reason": "learner check v3 results missing"}
    j = json.loads(LC_RESULTS.read_text(encoding="utf-8")).get("judgement", {})
    return bool(j.get("proceed_to_real_data")), j


def main() -> int:
    global _LOGF
    ap = argparse.ArgumentParser()
    ap.parse_args()
    spec_sha = D.check_spec()
    ok, j = learner_check_passed()
    if not ok:
        raise SystemExit(f"v3 の学習器確認が合格していないので実データに進まない: {j}")
    if MANIFEST.exists() or OOF_JSON.exists():
        raise SystemExit("v3 の学習結果が既にある（上書きしない）")
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    _LOGF = OUT / "log.txt"
    log(f"spec {spec_sha[:16]} OK; load {D.TRAIN_YEARS} with results (2023+ locked)")
    df = D.load_rows(D.TRAIN_YEARS, with_results=True)
    d, fstats = D.build_features(df)
    R = D.payout_R(d)
    day = d["date"].astype(str).to_numpy()
    base = M.CellBase.fit(D.make_X(d, "nomodel"), R)
    man = {"spec_version": "v3 v1.1", "created": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "spec_sha256": spec_sha,
           "substrate_sha256": D.sha256(D.SUB), "base_parquet_sha256": D.sha256(D.BASE_PQ),
           "lambda_sha256": D.sha256(D.LAMBDA), "learner_check_results_sha256": D.sha256(LC_RESULTS),
           "code_sha256": {f: D.sha256(HERE / f) for f in CODE_FILES if (HERE / f).exists()},
           "dep_sha256": {k: D.sha256(p) for k, p in V2_DEPS.items()},
           "train_years": list(D.TRAIN_YEARS), "n_boot": M.N_BOOT, "boot_q": M.BOOT_Q, "buy_threshold": M.BUY_THRESHOLD,
           "clip_hi_q": M.CLIP_HI_Q, "grid": M.GRID, "base": base.to_state(), "versions": {}}
    oof_cols = {"rid16": d["rid16"].to_numpy(), "umaban": d["umaban"].to_numpy(), "bet_type": d["bet_type"].to_numpy(),
                "year": d["year"].to_numpy(), "date": day}
    preds = {}
    for mode in MODES:
        v = vname(mode)
        X = D.make_X(d, mode)
        log(f"== {v}: {len(X.columns)} features {list(X.columns)}")
        t0 = time.time()
        cfg, rows = M.select_cfg_cv(X, R, day, seed=0, log=log)
        best = min((r for r in rows if r["cfg_id"] >= 0), key=lambda r: (r["cv_mse"], r["cfg_id"]))
        log(f"  chosen cfg{best['cfg_id']:02d} {cfg} cv_mse={best['cv_mse']:.6f} "
            f"(constant {rows[-1]['cv_mse']:.6f}) ({time.time() - t0:.0f}s)")
        ms = M.fit_bootstrap(X, R, day, cfg, base_seed=0, log=log)
        blob = M.dump_models(ms)
        p = MODELS / f"{v}_v3.pkl"
        p.write_bytes(blob)
        oof = M.oof_conservative(X, R, day, cfg, seed=0, log=log)
        preds[mode] = (oof["cons"], oof["mean"])
        oof_cols[f"cons_{mode}"] = oof["cons"]
        oof_cols[f"mean_{mode}"] = oof["mean"]
        oof_cols[f"fold_{mode}"] = oof["fold"]
        man["versions"][v] = {"mode": mode, "features": list(X.columns), "cfg": cfg, "cfg_id": best["cfg_id"],
                              "cv_grid": rows, "file": p.name, "sha256": M.sha256_bytes(blob),
                              "member_info": [m.info for m in ms]}
    oof_df = pd.DataFrame(oof_cols)
    oof_df.to_parquet(OOF_PQ, index=False)
    man["oof_file"] = str(OOF_PQ.relative_to(HERE))
    man["oof_sha256"] = D.sha256(OOF_PQ)
    man["calibration_windows"] = D.calibrator_info()
    man["feature_stats_2021_22"] = fstats
    MANIFEST.write_text(json.dumps(man, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    log(f"manifest written: {MANIFEST} sha256 {D.sha256(MANIFEST)}")
    rep = {"note": "交差予測（開催日 3 分割、ハイパーパラメータも同じ分割で選択）による記述。in-sample に近い。判定には使わない。",
           "all_2021_22": report_preds(preds, d, R, base.edges, "oof_2021_22"), "by_year": {}}
    for y in D.TRAIN_YEARS:
        m = d["year"].to_numpy() == y
        rep["by_year"][str(y)] = report_preds({k: (c[m], mm[m]) for k, (c, mm) in preds.items()},
                                              d[m].reset_index(drop=True), R[m], base.edges, f"oof_{y}",
                                              with_rho_ci=False)
    rep["manifest_sha256"] = D.sha256(MANIFEST)
    OOF_JSON.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    log(f"oof report written: {OOF_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
