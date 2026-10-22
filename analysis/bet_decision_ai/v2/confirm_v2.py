# -*- coding: utf-8 -*-
"""
confirm_v2.py — 馬券判断 AI v2 の確認（仕様 v1.0 §6-3・§6-4・§7）。**確認期間 2024-25 を 1 回だけ開く。**
=========================================================================================================
使い方:
  python -m analysis.bet_decision_ai.v2.confirm_v2 --dry-run-select   # 2023（選択期間）で最後まで通す。ロックは解除しない
  python -m analysis.bet_decision_ai.v2.confirm_v2 --confirm-once     # 本番。1 回限り
本番（--confirm-once）の手順:
  1. 結果ファイル out/confirm/confirm_result.json か開始印 out/confirm/STARTED があれば拒否（2 回目は無い）。
  2. 仕様の sha256（凍結値の先頭 d41323d6db13e48f）、manifest.json の各モデルファイル sha256、
     data_v2.py / model_v2.py の sha256（学習時と同じか）、select_2023.json に記録した manifest の sha256 を照合。
  3. 選択段の打ち切り（§6-2、主 = two_stage × model）に当たっていたら拒否（確認期間を開けない）。
  4. 開始印を書いてから、data_v2.confirm_unlock() で 2024-25 の結果列を読み、4 通りを評価。§7 で判定（主 = two_stage × model）。
     PyCaLiAI 版 − モデル列を外した版の差（§6-4）を出力。再学習・閾値変更は一切しない。
  5. 結果を out/confirm/confirm_result.json に書く（開始印は残す）。
2024-25 の確率は、2021-23 の OOF で fit した較正写像を本番スコアに当てたもの（仕様 §4 の注意書き）。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from analysis.bet_decision_ai.v2 import data_v2 as D
from analysis.bet_decision_ai.v2 import evaluate_v2 as E
from analysis.bet_decision_ai.v2 import model_v2 as M
from analysis.bet_decision_ai.v2 import train_v2 as T

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "confirm"
RESULT = OUT / "confirm_result.json"
STARTED = OUT / "STARTED"
DRY = OUT / "dry_run_select_2023.json"
STRICT_CODE = ["data_v2.py", "model_v2.py"]


def verify() -> tuple[dict, dict, dict]:
    spec_sha = D.check_spec()
    man = json.loads(T.MANIFEST.read_text(encoding="utf-8"))
    if man["spec_sha256"] != spec_sha:
        raise SystemExit(f"仕様の sha256 が学習時と違う: {man['spec_sha256']} != {spec_sha}")
    for f in STRICT_CODE:
        cur = D.sha256(HERE / f)
        if man["code_sha256"].get(f) != cur:
            raise SystemExit(f"{f} が学習時から変わった: {man['code_sha256'].get(f)} != {cur}")
    sel = json.loads((T.OUT / "select_2023.json").read_text(encoding="utf-8"))
    if sel.get("manifest_sha256") != D.sha256(T.MANIFEST):
        raise SystemExit("manifest.json が選択段の記録と違う")
    models = {}
    for v, info in man["versions"].items():
        blob = (T.MODELS / info["file"]).read_bytes()
        if M.sha256_bytes(blob) != info["sha256"]:
            raise SystemExit(f"モデル {info['file']} の sha256 が manifest と違う")
        models[v] = M.load_models(blob)
        assert len(models[v]) == man["n_boot"]
    code_now = {f: D.sha256(HERE / f) for f in T.CODE_FILES if (HERE / f).exists()}
    check = {"spec_sha256": spec_sha, "manifest_sha256": D.sha256(T.MANIFEST),
             "models": {v: i["sha256"] for v, i in man["versions"].items()},
             "code_sha256_now": code_now,
             "code_changed_since_train": [f for f in code_now if man["code_sha256"].get(f) != code_now[f]]}
    return models, sel, check


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--confirm-once", action="store_true")
    g.add_argument("--dry-run-select", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.confirm_once and (RESULT.exists() or STARTED.exists()):
        raise SystemExit(f"確認は 1 回限り: 既に {'結果' if RESULT.exists() else '開始印'} がある")
    models, sel, check = verify()
    T._LOGF = OUT / ("confirm_log.txt" if a.confirm_once else "dry_run_log.txt")
    if a.dry_run_select:
        T.log("dry-run on 2023 (select period). result lock NOT released")
        df = D.load_rows((D.SELECT_YEAR, D.SELECT_YEAR), with_results=True)
        d, fstats = D.build_features(df)
        R = D.payout_R(d)
        rep = T.report_period(models, d, R, "dry_run_select_2023")
        rep["verdict_if_this_were_confirm"] = {v: E.judge_confirm(rep["versions"][v]) for v in rep["versions"]}
        rep["matches_train_select"] = {
            v: {"n_buy": (rep["versions"][v]["all"]["n_buy"], sel["versions"][v]["all"]["n_buy"]),
                "rho": (rep["versions"][v]["all"]["rho"], sel["versions"][v]["all"]["rho"]),
                "roi": (rep["versions"][v]["all"]["roi"], sel["versions"][v]["all"]["roi"])}
            for v in rep["versions"]}
        rep["verify"] = check
        rep["feature_stats"] = fstats
        DRY.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        T.log(f"dry-run written: {DRY}")
        return 0
    # ---------------- 本番（1 回限り）
    if sel.get("abort"):
        raise SystemExit("選択段（§6-2）で打ち切り判定のため、確認期間は開けない")
    STARTED.write_text(f"started {datetime.now():%Y-%m-%d %H:%M:%S} argv={sys.argv}\n", encoding="utf-8")
    unlock = D.confirm_unlock()
    T.log("CONFIRM: opening 2024-2025 results (one time)")
    df = D.load_rows(D.CONFIRM_YEARS, with_results=True, unlock=unlock)
    d, fstats = D.build_features(df)
    R = D.payout_R(d)
    rep = T.report_period(models, d, R, "confirm_2024_25")
    rep["by_year"] = {}
    for y in D.CONFIRM_YEARS:
        m = d["year"].to_numpy() == y
        rep["by_year"][str(y)] = T.report_period(models, d[m].reset_index(drop=True), R[m], f"confirm_{y}")["versions"]
    rep["verdict"] = {v: E.judge_confirm(rep["versions"][v]) for v in rep["versions"]}
    rep["primary"] = T.vname(*T.PRIMARY)
    rep["primary_verdict"] = rep["verdict"][rep["primary"]]
    rep["verify"] = check
    rep["feature_stats"] = fstats
    rep["calibration_note"] = ("2024-25 の本番スコアに 2021-23 の OOF で fit した較正写像を当てた（出所が違う。仕様 §4 の注意書き）")
    rep["calibration_windows"] = D.calibrator_info()
    RESULT.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    T.log(f"CONFIRM verdict primary {rep['primary']}: {rep['primary_verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
