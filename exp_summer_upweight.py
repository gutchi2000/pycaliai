# -*- coding: utf-8 -*-
"""
exp_summer_upweight.py — 夏重み付け学習 (学生大会=夏競馬向け, 未検定の中間解)
==============================================================================
既決着: 夏専用学習(M_sum)は夏testで全敗 (60.2% vs プール62.6%) = 夏only は死。
未検定: 全データ学習 + 夏レース重み k 倍 (データ量を捨てずに夏へ寄せる)。

k ∈ {1(=v6相当), 2, 4} で LGBM lambdarank (v6 optuna params) を学習し、
test(2024-25) の **夏レース(6-9月)** で ◎top3 / ◎top1 / NDCG を比較。
選定は valid2023夏。夏で k>1 が valid/test 両方で上回らなければ v6 のまま。

実行: python exp_summer_upweight.py  (3モデル, ~20-30分)
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
SEED = 42
SUMMER = (6, 7, 8, 9)


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


def make_ds(df, feats, w=None):
    df = df.sort_values(COL_RID, kind="stable").reset_index(drop=True)
    X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    y = df["label"].values.astype(int)
    groups = np.array([len(list(g)) for _, g in groupby(df[COL_RID])])
    ww = df["_w"].values if w else None
    return lgb.Dataset(X, label=y, group=groups, weight=ww, free_raw_data=False), df


def eval_marks(df, model, feats, mask_label):
    df = df.sort_values(COL_RID, kind="stable").reset_index(drop=True)
    X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    df = df.copy()
    df["_s"] = model.predict(X, num_iteration=model.best_iteration)
    top1 = top3 = n = 0
    for rid, g in df.groupby(COL_RID, sort=False):
        if len(g) < 6:
            continue
        i = g["_s"].idxmax()
        fin = g.loc[i, COL_JYUN]
        n += 1
        top1 += fin == 1
        top3 += fin <= 3
    return {"label": mask_label, "n": n,
            "top1": round(top1 / n * 100, 2), "top3": round(top3 / n * 100, 2)}


def main():
    b = joblib.load(BASE / "models/unified_rank_v6.pkl")
    feats, encs, params = b["feature_cols"], b["encoders"], b["optuna_best_params"]
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID]).copy()
    df["label"] = np.clip(6 - df[COL_JYUN].astype(int), 0, 5).astype(int)
    rid = df[COL_RID].astype(str)
    df["year"] = rid.str[:4].astype(int)
    df["month"] = rid.str[4:6].astype(int)
    df["is_summer"] = df["month"].isin(SUMMER)
    enc = apply_encoders(df, encs)
    tr = enc[enc["year"] <= 2022]
    vl = enc[enc["year"] == 2023]
    te = enc[enc["year"] >= 2024]
    print(f"train={len(tr):,} valid={len(vl):,} test={len(te):,} "
          f"(夏率 train={tr['is_summer'].mean()*100:.0f}%)")

    results = []
    for k in [1.0, 2.0, 4.0]:
        trk = tr.copy()
        trk["_w"] = np.where(trk["is_summer"], k, 1.0)
        ds_tr, _ = make_ds(trk, feats, w=True)
        ds_vl, _ = make_ds(vl.assign(_w=1.0), feats)
        model = lgb.train(lgb_params(params), ds_tr, num_boost_round=3000,
                          valid_sets=[ds_vl],
                          callbacks=[lgb.early_stopping(150, verbose=False)])
        row = {"k": k, "best_iter": model.best_iteration}
        for name, sub in [("valid夏", vl[vl["is_summer"]]), ("valid全", vl),
                          ("test夏", te[te["is_summer"]]), ("test全", te)]:
            row[name] = eval_marks(sub, model, feats, name)
        results.append(row)
        print(f"k={k}: iter={model.best_iteration}  "
              + "  ".join(f"{nm} top3={row[nm]['top3']}%(top1 {row[nm]['top1']}%)"
                          for nm in ("valid夏", "test夏", "test全")), flush=True)
        if k > 1.0:
            joblib.dump({"model": model, "feature_cols": feats, "encoders": encs,
                         "summer_weight": k, "note": "夏重み付けプール学習 (学生大会用実験)"},
                        BASE / f"models/summer_upweight_k{int(k)}.pkl")

    import json
    out = BASE / "reports" / "summer_upweight.json"
    json.dump([{kk: (vv if not isinstance(vv, dict) else vv) for kk, vv in r.items()}
               for r in results], open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
