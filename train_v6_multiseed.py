"""
train_v6_multiseed.py
=====================
v6 best params + best_iter を流用して、 異なる seed で 4 個追加学習する。
seed=42 は既存 unified_rank_v6.pkl を使うので、 新規は 4 個のみ。

合計 5 model の score 平均化で ensemble effect を狙う (案 1)。

実行: python train_v6_multiseed.py
出力:
  models/unified_rank_v6_s123.pkl
  models/unified_rank_v6_s456.pkl
  models/unified_rank_v6_s789.pkl
  models/unified_rank_v6_s1234.pkl
"""
from __future__ import annotations
import io
import sys
import warnings
from itertools import groupby
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE       = Path(__file__).parent
V6_PKL     = BASE / "models/unified_rank_v6.pkl"
MASTER_CSV = BASE / "data/master_v2_20130105-20251228.csv"
KEKKA_CSV  = BASE / "data/kekka_20130105-20251228.csv"

SEEDS = [123, 456, 789, 1234]

COL_RID  = "レースID(新/馬番無)"
COL_JYUN = "着順"

LEAK_COLS = {
    "着順", "fukusho_flag", "roi_target",
    "レースID(新)", "レースID(新/馬番無)",
    "馬名", "レース名", "発走時刻", "date_dt", "日付",
    "血統登録番号", "split",
}


def load_winner_tansho_pay():
    df = pd.read_csv(KEKKA_CSV, encoding="cp932", low_memory=False)
    df.columns = ["rid_horse", "ban", "ped", "jyun",
                  "tansho", "fukusho", "wakuren", "umaren",
                  "umatan", "sanrenpuku", "sanrentan"]
    df["rid_s"] = df["rid_horse"].astype(str).str[:16]
    df["jyun"] = pd.to_numeric(df["jyun"], errors="coerce")
    df["tansho"] = pd.to_numeric(df["tansho"], errors="coerce")
    win = df[df["jyun"] == 1].drop_duplicates("rid_s")
    return dict(zip(win["rid_s"].values, win["tansho"].values))


def apply_encoders(d, encs):
    d = d.copy()
    for c, le in encs.items():
        if c not in d.columns: continue
        v = d[c].astype(str).fillna("__NaN__")
        known = set(le.classes_)
        v = v.where(v.isin(known), "__NaN__")
        d[c] = le.transform(v)
    return d


def make_dataset(d, feats, alpha=0.0):
    d = d.sort_values(COL_RID).reset_index(drop=True)
    X = d[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    y = d["label"].values.astype(int)
    g = np.array([len(list(gr)) for _, gr in groupby(d[COL_RID])])
    if alpha > 0 and "winner_tansho" in d.columns:
        w = (1.0 + alpha * np.log1p(d["winner_tansho"].values / 100.0)).astype(float)
    else:
        w = np.ones(len(d), dtype=float)
    return lgb.Dataset(X, label=y, group=g, weight=w, free_raw_data=False)


def main():
    print("=" * 70)
    print("train_v6_multiseed.py")
    print("=" * 70)
    print(f"[load v6] {V6_PKL}")
    v6 = joblib.load(V6_PKL)
    bp     = v6["optuna_best_params"]
    encs   = v6["encoders"]
    feats  = v6["feature_cols"]
    # v6 retrain 時に early_stopping なしで学習しているため best_iteration=0
    # → num_trees() で実際の木の本数を取得 (= 元 best_iter * 1.1 程度)
    best_iter = v6["model"].best_iteration
    if best_iter == 0:
        best_iter = v6["model"].num_trees()
        print(f"  (best_iteration=0 → num_trees={best_iter} を使用)")
    alpha     = bp.get("alpha", v6.get("sample_weight_alpha", 0.0))
    rr_mode = v6.get("race_relative_mode")
    ca_mode = v6.get("course_affinity_mode")
    print(f"  best_iter = {best_iter}")
    print(f"  alpha     = {alpha:.4f}")
    print(f"  feats     = {len(feats)}")
    print(f"  race_relative_mode = {rr_mode}")
    print(f"  course_affinity_mode = {ca_mode}")
    print(f"  best_params: {bp}")

    print(f"\n[load] {MASTER_CSV}")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]).copy()
    df["label"] = np.clip(6 - df[COL_JYUN].astype(int), 0, 5).astype(int)
    if rr_mode:
        from race_relative_feats import add_race_relative_feats
        df = add_race_relative_feats(df, mode=rr_mode)
        print(f"  + race_relative_feats(mode={rr_mode})")
    if ca_mode:
        from course_affinity_feats import add_course_affinity_feats
        df = add_course_affinity_feats(df, mode=ca_mode)
        print(f"  + course_affinity_feats(mode={ca_mode})")
    df = apply_encoders(df, encs)
    tr = df[df["split"] == "train"].copy()
    vl = df[df["split"] == "valid"].copy()
    print(f"  train={len(tr):,}  valid={len(vl):,}")

    print("\n[load winner_tansho_pay]")
    win_pay = load_winner_tansho_pay()
    tr["rid_s"] = tr[COL_RID].astype(str).str[:16]
    tr["winner_tansho"] = tr["rid_s"].map(win_pay).fillna(100.0)

    ds_vl = make_dataset(vl, feats, alpha=0.0)

    for seed in SEEDS:
        out = BASE / f"models/unified_rank_v6_s{seed}.pkl"
        if out.exists():
            print(f"\n[skip] {out.name} already exists")
            continue
        print(f"\n[seed={seed}] training")
        params = {
            "objective":                  "lambdarank",
            "lambdarank_truncation_level": 5,
            "metric":                     "ndcg",
            "eval_at":                    [1, 3, 5],
            "learning_rate":              bp["lr"],
            "num_leaves":                 bp["num_leaves"],
            "max_depth":                  bp["max_depth"],
            "min_data_in_leaf":           bp["min_data_in_leaf"],
            "feature_fraction":           bp["ff"],
            "bagging_fraction":           bp["bf"],
            "bagging_freq":               5,
            "lambda_l1":                  bp["l1"],
            "lambda_l2":                  bp["l2"],
            "verbose": -1, "n_jobs": -1, "seed": seed,
            "deterministic": True, "force_col_wise": True, "feature_pre_filter": False,
        }
        ds_tr = make_dataset(tr, feats, alpha=alpha)
        # best_iter は既に num_trees() (= 元 best_iter * 1.1) なのでそのまま使う
        n_round = best_iter
        model = lgb.train(
            params, ds_tr,
            num_boost_round=n_round,
            valid_sets=[ds_vl],
            callbacks=[lgb.log_evaluation(period=500)],
        )
        joblib.dump({
            "model": model, "feature_cols": feats, "encoders": encs,
            "cat_cols": v6.get("cat_cols"), "seed": seed,
            "master_csv": MASTER_CSV.name,
            "optuna_best_params": bp,
            "sample_weight_alpha": alpha,
            "label_scheme": "clip(6 - 着順, 0, 5)",
            "race_relative_mode": rr_mode,
            "course_affinity_mode": ca_mode,
            "parent_model": "unified_rank_v6.pkl",
            "description": f"v6 multiseed (seed={seed}). "
                           f"v6 best params + 異なる seed で再学習。 ensemble 用。",
        }, out)
        print(f"  [saved] {out}  ({out.stat().st_size/1024/1024:.1f} MB)")

    print("\nDONE")


if __name__ == "__main__":
    main()
