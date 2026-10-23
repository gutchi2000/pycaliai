# -*- coding: utf-8 -*-
"""
confirm_v3.py — 馬券判断 AI v3 の確認（仕様 v1.0 §3-3・§4）。**確認期間 2024-25 を 1 回だけ開く。**
=================================================================================================
使い方:
  python -m analysis.bet_decision_ai.v3.confirm_v3 --dry-run-dev    # 2022 の交差予測で同じ処理を最後まで通す（ロックは解除しない）
  python -m analysis.bet_decision_ai.v3.confirm_v3 --confirm-once   # 本番。1 回限り
本番（--confirm-once）の手順:
  1. 結果ファイル out/confirm/confirm_result.json か開始印 out/confirm/STARTED があれば拒否。
  2. 照合: 仕様の sha256（v1.1 凍結値の先頭 10a4c627af6a449f、manifest の記録とも一致）、manifest の各モデル sha256、
     data_v3.py / model_v3.py / evaluate_v3.py と v2 依存ファイルの sha256（学習時と同じか）、交差予測ファイルの sha256、
     dry-run の記録があり、その時の confirm_v3.py の sha256 が今と同じか。
  3. 開始印を書いてから data_v3.confirm_unlock() で 2024-25 の結果列を読み、凍結モデルで評価、§4 で判定（主 = PyCaLiAI 版）。
     超過回収のセル境界は学習期間 2021-22 で決めたもの（manifest）、セル平均は確認期間の全候補。
     外した版との差（超過回収・回収率、CI 付き、判定外）。年別。再学習・閾値変更は一切しない。
  4. 判定を書き出した後、2023 を同じ凍結モデルで記述として評価（判定外）。
  5. 結果を out/confirm/confirm_result.json に書く（開始印は残す）。
2024-25 の確率は、2021-23 の OOF で fit した較正写像を本番スコアに当てたもの（v2 仕様 §4 の注意書き）。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.bet_decision_ai.v3 import data_v3 as D
from analysis.bet_decision_ai.v3 import evaluate_v3 as E
from analysis.bet_decision_ai.v3 import model_v3 as M
from analysis.bet_decision_ai.v3 import train_v3 as T

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "confirm"
RESULT = OUT / "confirm_result.json"
STARTED = OUT / "STARTED"
DRY = OUT / "dry_run_dev_2022.json"
STRICT_CODE = ["data_v3.py", "model_v3.py", "evaluate_v3.py"]


def verify() -> tuple[dict, dict, dict]:
    spec_sha = D.check_spec()
    man = json.loads(T.MANIFEST.read_text(encoding="utf-8"))
    if man["spec_sha256"] != spec_sha:
        raise SystemExit(f"仕様の sha256 が学習時と違う: {man['spec_sha256']} != {spec_sha}")
    for f in STRICT_CODE:
        cur = D.sha256(HERE / f)
        if man["code_sha256"].get(f) != cur:
            raise SystemExit(f"{f} が学習時から変わった: {man['code_sha256'].get(f)} != {cur}")
    for k, p in T.V2_DEPS.items():
        if man["dep_sha256"].get(k) != D.sha256(p):
            raise SystemExit(f"依存ファイル {k} が学習時から変わった")
    for k, p in (("substrate_sha256", D.SUB), ("base_parquet_sha256", D.BASE_PQ), ("lambda_sha256", D.LAMBDA)):
        if man[k] != D.sha256(p):
            raise SystemExit(f"データ {p.name} の sha256 が学習時と違う")
    if man["oof_sha256"] != D.sha256(HERE / man["oof_file"]):
        raise SystemExit("交差予測ファイルの sha256 が manifest と違う")
    models = {}
    for v, info in man["versions"].items():
        blob = (T.MODELS / info["file"]).read_bytes()
        if M.sha256_bytes(blob) != info["sha256"]:
            raise SystemExit(f"モデル {info['file']} の sha256 が manifest と違う")
        models[v] = M.load_models(blob)
        assert len(models[v]) == man["n_boot"]
    code_now = {f: D.sha256(HERE / f) for f in T.CODE_FILES if (HERE / f).exists()}
    check = {"spec_sha256": spec_sha, "manifest_sha256": D.sha256(T.MANIFEST),
             "models": {v: i["sha256"] for v, i in man["versions"].items()}, "code_sha256_now": code_now,
             "code_changed_since_train": [f for f in code_now if man["code_sha256"].get(f) != code_now[f]]}
    return models, man, check


def frozen_preds(models: dict, d: pd.DataFrame) -> dict:
    out = {}
    for mode in T.MODES:
        X = D.make_X(d, mode)
        P = M.predict_ensemble(models[T.vname(mode)], X)
        out[mode] = (M.conservative(P), P.mean(0))
    return out


def judged(rep: dict) -> dict:
    rep["verdict"] = {v: E.judge_confirm(rep["versions"][v]) for v in rep["versions"]}
    rep["primary"] = T.vname(T.PRIMARY)
    rep["primary_verdict"] = rep["verdict"][rep["primary"]]
    return rep


def oof_preds_for(d: pd.DataFrame, oof: pd.DataFrame) -> dict:
    key = ["rid16", "umaban", "bet_type"]
    m = d[key].merge(oof, on=key, how="left", validate="one_to_one")
    assert len(m) == len(d) and m["cons_model"].notna().all(), "交差予測が候補と対応しない"
    return {mode: (m[f"cons_{mode}"].to_numpy(float), m[f"mean_{mode}"].to_numpy(float)) for mode in T.MODES}


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--confirm-once", action="store_true")
    g.add_argument("--dry-run-dev", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.confirm_once and (RESULT.exists() or STARTED.exists()):
        raise SystemExit(f"確認は 1 回限り: 既に {'結果' if RESULT.exists() else '開始印'} がある")
    models, man, check = verify()
    edges = man["base"]["edges"]
    me_sha = D.sha256(Path(__file__).resolve())
    T._LOGF = OUT / ("confirm_log.txt" if a.confirm_once else "dry_run_log.txt")

    if a.dry_run_dev:
        T.log("dry-run on 2022 out-of-fold predictions (dev period). result lock NOT released")
        oof = pd.read_parquet(HERE / man["oof_file"])
        rep_all = {"mode": "dry_run_dev", "verify": check, "confirm_v3_sha256": me_sha}
        for y, slot in ((2022, "judged_slot_as_confirm"), (2021, "descriptive_slot_as_2023")):
            df = D.load_rows((y, y), with_results=True)
            d, fstats = D.build_features(df)
            R = D.payout_R(d)
            rep = T.report_preds(oof_preds_for(d, oof), d, R, edges, f"dry_run_oof_{y}")
            if slot == "judged_slot_as_confirm":
                rep = judged(rep)
                fp = frozen_preds(models, d)          # 凍結モデルの予測経路も通す（2022 は学習データなので in-sample、記述のみ）
                rep["frozen_models_in_sample"] = {T.vname(k): {"n_buy": int(M.buy_mask(c).sum()),
                                                               "cons_q": [float(x) for x in np.quantile(c, [.5, .99, 1])]}
                                                  for k, (c, _) in fp.items()}
            rep["feature_stats"] = fstats
            rep_all[slot] = rep
        oofrep = json.loads(T.OOF_JSON.read_text(encoding="utf-8"))
        rep_all["matches_train_oof_2022"] = {
            v: {k: (rep_all["judged_slot_as_confirm"]["versions"][v]["all"][k],
                    oofrep["by_year"]["2022"]["versions"][v]["all"][k]) for k in ("n_buy", "roi", "excess")}
            for v in rep_all["judged_slot_as_confirm"]["versions"]}
        DRY.write_text(json.dumps(rep_all, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        T.log(f"dry-run written: {DRY}; verdict-if-confirm {rep_all['judged_slot_as_confirm']['primary_verdict']}")
        return 0

    # ---------------- 本番（1 回限り）
    if not DRY.exists():
        raise SystemExit("dry-run（--dry-run-dev）の記録が無い")
    dry = json.loads(DRY.read_text(encoding="utf-8"))
    if dry.get("confirm_v3_sha256") != me_sha:
        raise SystemExit("confirm_v3.py が dry-run 時から変わった")
    STARTED.write_text(f"started {datetime.now():%Y-%m-%d %H:%M:%S} argv={sys.argv}\n", encoding="utf-8")
    unlock = D.confirm_unlock()
    T.log("CONFIRM: opening 2024-2025 results (one time)")
    df = D.load_rows(D.CONFIRM_YEARS, with_results=True, unlock=unlock)
    d, fstats = D.build_features(df, unlock)
    R = D.payout_R(d)
    preds = frozen_preds(models, d)
    rep = judged(T.report_preds(preds, d, R, edges, "confirm_2024_25"))
    rep["by_year"] = {}
    for y in D.CONFIRM_YEARS:
        m = d["year"].to_numpy() == y
        rep["by_year"][str(y)] = T.report_preds({k: (c[m], mm[m]) for k, (c, mm) in preds.items()},
                                                d[m].reset_index(drop=True), R[m], edges, f"confirm_{y}",
                                                with_rho_ci=False)["versions"]
    rep["verify"] = check
    rep["confirm_v3_sha256"] = me_sha
    rep["feature_stats"] = fstats
    rep["calibration_note"] = "2024-25 の本番スコアに 2021-23 の OOF で fit した較正写像を当てた（出所が違う。v2 仕様 §4 の注意書き）"
    RESULT.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    T.log(f"CONFIRM verdict primary {rep['primary']}: {rep['primary_verdict']}")
    # 2023 は記述（判定外）。判定を書き出した後に読む
    df3 = D.load_rows((D.DESCRIBE_YEAR, D.DESCRIBE_YEAR), with_results=True, unlock=unlock)
    d3, fs3 = D.build_features(df3, unlock)
    R3 = D.payout_R(d3)
    rep["describe_2023_not_judged"] = T.report_preds(frozen_preds(models, d3), d3, R3, edges, "describe_2023")
    rep["describe_2023_not_judged"]["feature_stats"] = fs3
    rep["calibration_windows"] = D.calibrator_info()
    RESULT.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    T.log("2023 descriptive appended (not judged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
