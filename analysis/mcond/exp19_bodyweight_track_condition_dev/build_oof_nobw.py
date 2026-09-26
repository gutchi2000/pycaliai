# -*- coding: utf-8 -*-
"""
build_oof_nobw.py — EXP19 Stage 0: R0-clean-nobw (110 列) の rolling OOF 基盤 (40 fit)
====================================================================================
spec v0.2-frozen baseline_contract: EXP15 R0-clean 111 列から `斤量体重比` (今走馬体重由来) だけを除く。
学習レシピ・年窓・seed・τ の取り方は EXP16A build_oof.py と同一 (関数を import して再利用し、別経路を作らない):
    train <= Y−2 / ES = Y−1 / predict = Y、Y = 2016..2023、5 seed = 40 fit
    encoder は train (<= Y−2) の語彙だけ、τ は Y−1 の正式 race set 行だけ
**結果評価はしない** (2019-2023 の logloss・AUC 等を一切計算しない)。確認するのは再現性・未来行違反・manifest だけ。
2024/2025 は読まない (EXP16A の行 cache は 2013-2023 だけで、読み込み時に assert)。

出力:
  data/_research/mcond/exp19/scores/N1_Y{year}_s{seed}.npz   (gitignore)
  out/oof_nobw_manifest.json / out/oof_nobw_checks.json
実行: python -m analysis.mcond.exp19_bodyweight_track_condition_dev.build_oof_nobw
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

from ..exp16a_close_market_residual_dev import build_oof as B16
from ..exp16a_close_market_residual_dev.provenance import MASTER, sha256
from ..exp16a_close_market_residual_dev.stage0_dry_run import EXP15_OUT, sha_text

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
OUT = HERE / "out"
RESEARCH = BASE / "data" / "_research" / "mcond" / "exp19"
SCORES = RESEARCH / "scores"
REMOVE = ["斤量体重比"]
RETAIN_CHECK = ["前走馬体重", "前走馬体重増減", "斤量", "馬齢斤量差"]
YEARS, SEEDS = B16.YEARS, B16.SEEDS
REPRO = (2019, SEEDS[0])            # 再現性検査: 同じ fit をもう一度回して予測が bit 一致すること


def file_sha(p: Path) -> str:
    return sha256(p)


def fit_one(X, y_rel, rid, tr_rows, es_rows, pr_rows, seed, dtr):
    params = {**B16.FIXED, **B16.CFG, "seed": int(seed)}
    dsl = lgb.Dataset(X[es_rows], label=y_rel[es_rows], group=B16.groups(rid[es_rows]), reference=dtr)
    m = lgb.train(params, dtr, num_boost_round=B16.MAX_ROUNDS, valid_sets=[dsl],
                  callbacks=[lgb.early_stopping(B16.ES_ROUNDS, verbose=False)])
    return m, m.predict(X[pr_rows], num_iteration=m.best_iteration).astype(np.float64)


def main():
    t00 = time.time()
    OUT.mkdir(exist_ok=True)
    SCORES.mkdir(parents=True, exist_ok=True)
    contract = json.loads((EXP15_OUT / "feature_contract.json").read_text(encoding="utf-8"))
    clean = contract["features"]["clean"]
    assert len(clean) == 111 and all(c in clean for c in REMOVE)
    cols = [c for c in clean if c not in REMOVE]
    assert len(cols) == 110 and all(c in cols for c in RETAIN_CHECK)
    assert not any(c in cols for c in ("馬体重", "馬体重増減", "斤量体重比")), "今走馬体重由来の列が残っている"
    cats = [c for c in cols if c in contract["cat_cols"]]
    nums = [c for c in cols if c not in contract["cat_cols"]]
    official = json.loads((B16.RESEARCH / "official_rids_by_year.json").read_text(encoding="utf-8"))

    df = B16.load_rows(contract)                    # EXP16A と同じ行 cache (2013-2023、着順が数値の行)
    assert int(df["date"].max()) <= B16.D_MAX
    num = B16.numeric_block(df, nums)
    rid = df["rid16"].to_numpy()
    y_rel = df["rel"].to_numpy()
    win = df["win"].to_numpy()
    year = df["year"].to_numpy()
    key_all = (df["rid16"] + "_" + df["ban"].astype(str)).to_numpy()
    assert not np.any(np.char.startswith(df["rid16"].to_numpy().astype(str), "2024"))

    manifest = {
        "spec_version": "EXP19 v0.2-frozen (commit 4a4c4903)",
        "role": "N1_CLEAN_NOBW の rolling OOF artifact 契約。学習・予測のみ。結果指標は含まない",
        "created": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "baseline_contract": {"name": "R0-clean-nobw", "n_features": len(cols), "removed": REMOVE,
                              "retained_checked": RETAIN_CHECK, "n_cat": len(cats)},
        "feature_list_sha256": sha_text("\n".join(cols)),
        "feature_list_sorted_sha256": sha_text("\n".join(sorted(cols))),
        "features": cols,
        "input_sha256": {"master_v2": sha256(MASTER), "rows_cache": file_sha(B16.ROWS_CACHE),
                         "feature_contract_exp15": file_sha(EXP15_OUT / "feature_contract.json"),
                         "official_rids_by_year": file_sha(B16.RESEARCH / "official_rids_by_year.json")},
        "loader_sha256": {"build_oof_nobw.py": file_sha(Path(__file__)),
                          "exp16a_build_oof.py (reused functions)": file_sha(Path(B16.__file__))},
        "row_hashes": {"all_rows": sha_text("\n".join(sorted(key_all))),
                       "rows_2016_2023": sha_text("\n".join(sorted(key_all[(year >= 2016) & (year <= 2023)])))},
        "recipe": {"family": "v6_lr_half (EXP15 2022 選択済み、EXP16A と同一)", "params": B16.CFG, "fixed": B16.FIXED,
                   "max_rounds": B16.MAX_ROUNDS, "early_stopping_rounds": B16.ES_ROUNDS,
                   "windows": "train <= Y-2 / ES = Y-1 / predict = Y", "years": YEARS, "seeds": SEEDS,
                   "tau_population": "Y-1 の正式 race set (EXP16A official) 行。Y=2016 のみ 2015 全レース"},
        "fits": {},
    }
    checks = {"n_rows": int(len(df)), "n_races": int(df["rid16"].nunique()),
              "date_range": [int(df["date"].min()), int(df["date"].max())],
              "duplicate_rid_ban_rows": int(len(key_all) - len(set(key_all))), "future_row_checks": {}}
    assert checks["duplicate_rid_ban_rows"] == 0

    repro = {}
    for Y in YEARS:
        tr, es, pr = year <= Y - 2, year == Y - 1, year == Y
        cat_codes, vocab = B16.cat_block(df, cats, tr)
        X = B16.assemble(cols, cats, nums, cat_codes, num)
        tr_rows, es_rows, pr_rows = np.flatnonzero(tr), np.flatnonzero(es), np.flatnonzero(pr)
        fc = {"train_max_year": int(year[tr_rows].max()), "es_years": sorted(set(year[es_rows].tolist())),
              "predict_years": sorted(set(year[pr_rows].tolist())),
              "train_max_date": int(df["date"].to_numpy()[tr_rows].max())}
        fc["ok"] = fc["train_max_year"] <= Y - 2 and fc["es_years"] == [Y - 1] and fc["predict_years"] == [Y]
        checks["future_row_checks"][str(Y)] = fc
        assert fc["ok"], (Y, fc)
        off_prev = set(official.get(str(Y - 1), []))
        es_off_rows = es_rows[np.isin(rid[es_rows], list(off_prev))] if off_prev else es_rows
        races_es_off = B16.race_index(rid, es_off_rows)
        dtr = lgb.Dataset(X[tr_rows], label=y_rel[tr_rows], group=B16.groups(rid[tr_rows]), free_raw_data=False)
        for seed in SEEDS:
            t0 = time.time()
            m, s_pred = fit_one(X, y_rel, rid, tr_rows, es_rows, pr_rows, seed, dtr)
            s_es = m.predict(X[es_off_rows], num_iteration=m.best_iteration).astype(np.float64)
            se = np.full(len(df), np.nan)
            se[es_off_rows] = s_es
            tau, tau_nll = B16.fit_tau(se, win, races_es_off)
            f = SCORES / f"N1_Y{Y}_s{seed}.npz"
            np.savez(f, rows=pr_rows.astype(np.int64), score=s_pred, key=key_all[pr_rows], year=np.int32(Y),
                     seed=np.int64(seed), tau=np.float64(tau), best_iter=np.int64(m.best_iteration))
            mdl = m.model_to_string(num_iteration=m.best_iteration)
            manifest["fits"][f"Y{Y}_s{seed}"] = {
                "year": Y, "seed": seed, "n_train_rows": int(len(tr_rows)), "n_es_rows": int(len(es_rows)),
                "n_predict_rows": int(len(pr_rows)), "best_iter": int(m.best_iteration), "tau": tau,
                "encoder_hash": sha_text(json.dumps(vocab, ensure_ascii=False, sort_keys=True)),
                "model_sha256": sha_text(mdl), "prediction_sha256": sha256(f),
                "prediction_row_hash": sha_text("\n".join(key_all[pr_rows])),
                "seconds": round(time.time() - t0, 1)}
            if (Y, seed) == REPRO:
                m2, s2 = fit_one(X, y_rel, rid, tr_rows, es_rows, pr_rows, seed, dtr)
                repro = {"fit": f"Y{Y}_s{seed}", "predictions_bitwise_equal": bool(np.array_equal(s_pred, s2)),
                         "model_string_equal": mdl == m2.model_to_string(num_iteration=m2.best_iteration)}
            print(f"  Y{Y} s{seed}: iter={m.best_iteration} tau={tau:.4f} ({time.time()-t0:.0f}s)", flush=True)
        del dtr, X
    checks["reproducibility"] = repro
    checks["fits"] = len(manifest["fits"])
    checks["elapsed_sec"] = round(time.time() - t00, 1)
    (OUT / "oof_nobw_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1, default=float),
                                                encoding="utf-8")
    (OUT / "oof_nobw_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[saved] 40 fits repro={repro} ({checks['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
