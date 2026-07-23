# -*- coding: utf-8 -*-
"""
exp_oof_scores.py — expanding-window OOF スコア生成 (deepvalue 基質の毒抜き)
============================================================================
v6 は ≤2022 学習なので train 期間の f は in-sample (◎単勝フラット ROI 170%)。
意思決定ネットの学習には as-of スコアが必須 ([[project_v8_affinity_insample_leak]] の教訓)。

各年 Y in 2015..2023 を:
  train ≤Y-2 / early-stop valid = Y-1 / predict Y
の LGBM lambdarank (v6 optuna params, label=clip(6-着順,0,5), v6と同じ120特徴+encoder) で採点。

出力: data/oof_scores_v6params.parquet  (rid, ban, year, score)
実行: python exp_oof_scores.py   (~30-60分, 9モデル)
"""
from __future__ import annotations
from itertools import groupby
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from joint_calibration_v6 import apply_encoders, COL_RID, COL_JYUN, COL_BAN, MASTER_CSV

BASE = Path(__file__).parent
OUT = BASE / "data" / "oof_scores_v6params.parquet"
SEED = 42
YEARS = list(range(2015, 2024))


def lgb_params(p):
    return {
        "objective": "lambdarank", "lambdarank_truncation_level": 5,
        "metric": "ndcg", "eval_at": [1, 3, 5],
        "learning_rate": p["lr"], "num_leaves": p["num_leaves"],
        "max_depth": p["max_depth"], "min_data_in_leaf": p["min_data_in_leaf"],
        "feature_fraction": p["ff"], "bagging_fraction": p["bf"], "bagging_freq": 5,
        "lambda_l1": p["l1"], "lambda_l2": p["l2"],
        "verbose": -1, "n_jobs": -1, "seed": SEED,
        "deterministic": True, "force_col_wise": True,
    }


def make_ds(df, feats):
    df = df.sort_values(COL_RID, kind="stable").reset_index(drop=True)
    X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    y = df["label"].values.astype(int)
    groups = np.array([len(list(g)) for _, g in groupby(df[COL_RID])])
    return lgb.Dataset(X, label=y, group=groups, free_raw_data=False), df


def main():
    b = joblib.load(BASE / "models/unified_rank_v6.pkl")
    feats, encs, params = b["feature_cols"], b["encoders"], b["optuna_best_params"]
    print(f"[load] master ... feats={len(feats)}")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID]).copy()
    df["label"] = np.clip(6 - df[COL_JYUN].astype(int), 0, 5).astype(int)
    df["year"] = df[COL_RID].astype(str).str[:4].astype(int)
    enc = apply_encoders(df, encs)
    print(f"  rows={len(enc):,}  years {enc['year'].min()}-{enc['year'].max()}")

    frames = []
    for Y in YEARS:
        tr = enc[enc["year"] <= Y - 2]
        vl = enc[enc["year"] == Y - 1]
        te = enc[enc["year"] == Y]
        if len(te) == 0:
            continue
        ds_tr, _ = make_ds(tr, feats)
        ds_vl, _ = make_ds(vl, feats)
        model = lgb.train(lgb_params(params), ds_tr, num_boost_round=3000,
                          valid_sets=[ds_vl],
                          callbacks=[lgb.early_stopping(150, verbose=False)])
        te_s = te.sort_values(COL_RID, kind="stable").reset_index(drop=True)
        X = te_s[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
        sc = model.predict(X, num_iteration=model.best_iteration)
        frames.append(pd.DataFrame({
            "rid": te_s[COL_RID].astype(str), "ban": te_s[COL_BAN].astype(int),
            "year": Y, "score": sc.astype(np.float32),
        }))
        print(f"  Y={Y}: train={len(tr):,} rows  best_iter={model.best_iteration}  scored={len(te_s):,}", flush=True)

    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(OUT, index=False)
    print(f"[saved] {OUT}  rows={len(out):,}")


if __name__ == "__main__":
    main()
