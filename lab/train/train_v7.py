"""
train_v7.py
===========
v7 = drift対策(train≤2023/valid2024で前進) + serve skew対策(serve不安定12特徴を除外)。
v6 は一切触らない。新規 unified_rank_v7.pkl を保存。

- 特徴: v6 の feature_cols から「serve時に死ぬ12特徴」を除外
        (hist_same_*×4 / course_*×3 / jockey_*×3 / 騎手コード/調教師コード)
        → 使わない特徴は死にようがない = train/serve 一致 = serve skew 構造的に消滅
- 分割: train year<=2023 / valid(早期停止) year==2024  (2025はOOS評価用に温存)
- params: v6 Optuna ベストを再利用 (再探索なし)
- encoders: v6 のを流用 (騎手/調教師コードは除外)

実行: PYTHONUTF8=1 python train_v7.py
"""
from __future__ import annotations
import io, sys
from itertools import groupby
from pathlib import Path
import joblib, numpy as np, pandas as pd
import lightgbm as lgb

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

BASE = Path(__file__).parent
MASTER = BASE / "data/master_v2_20130105-20251228.csv"
KEKKA = BASE / "data/kekka_20130105-20251228.csv"
V6_PKL = BASE / "models/unified_rank_v6.pkl"
OUT = BASE / "models/unified_rank_v7.pkl"
COL_RID = "レースID(新/馬番無)"; COL_JYUN = "着順"; COL_BAN = "馬番"

DROP_EXACT = {"course_n_prev", "course_win_rate", "course_top3_rate",
              "jockey_n_prev", "jockey_win_rate", "jockey_top3_rate",
              "騎手コード", "調教師コード"}

PARAMS = dict(objective="lambdarank", lambdarank_truncation_level=5,
              learning_rate=0.05083, num_leaves=59, max_depth=12,
              min_data_in_leaf=197, feature_fraction=0.8761, bagging_fraction=0.7032,
              bagging_freq=1, lambda_l1=0.00111, lambda_l2=7.538,
              seed=42, verbosity=-1, metric="ndcg", ndcg_eval_at=[5])
ALPHA = 0.03084


def winner_pay():
    df = pd.read_csv(KEKKA, encoding="cp932", low_memory=False)
    df.columns = ["rid_horse", "ban", "ped", "jyun", "tansho", "fukusho",
                  "wakuren", "umaren", "umatan", "sanrenpuku", "sanrentan"]
    df["rid_s"] = df["rid_horse"].astype(str).str[:16]
    df["jyun"] = pd.to_numeric(df["jyun"], errors="coerce")
    df["tansho"] = pd.to_numeric(df["tansho"], errors="coerce")
    w = df[df["jyun"] == 1].drop_duplicates("rid_s")
    return dict(zip(w["rid_s"].values, w["tansho"].values))


def apply_enc(d, encs):
    d = d.copy()
    for c, le in encs.items():
        if c not in d.columns:
            continue
        v = d[c].astype(str).fillna("__NaN__")
        v = v.where(v.isin(set(le.classes_)), "__NaN__")
        d[c] = le.transform(v)
    return d


def make_ds(d, feats):
    d = d.sort_values(COL_RID).reset_index(drop=True)
    X = d[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    y = np.clip(6 - d[COL_JYUN].astype(int), 0, 5).astype(int).values
    g = np.array([len(list(gr)) for _, gr in groupby(d[COL_RID])])
    if "winner_tansho" in d.columns:
        w = (1.0 + ALPHA * np.log1p(d["winner_tansho"].values / 100.0)).astype(float)
    else:
        w = np.ones(len(d))
    return lgb.Dataset(X, label=y, group=g, weight=w, free_raw_data=False)


def main():
    v6 = joblib.load(V6_PKL)
    feats = [c for c in v6["feature_cols"] if c not in DROP_EXACT and not c.startswith("hist_same_")]
    encs = {k: v for k, v in v6["encoders"].items() if k not in ("騎手コード", "調教師コード")}
    n_drop = len(v6["feature_cols"]) - len(feats)
    print(f"[feats] v6={len(v6['feature_cols'])} -> v7={len(feats)}  (除外 {n_drop})")

    print(f"[load] {MASTER.name}")
    df = pd.read_csv(MASTER, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID]).copy()
    df["year"] = df[COL_RID].astype(str).str[:4].astype(int)
    tr = df[df["year"] <= 2023].copy()
    vl = df[df["year"] == 2024].copy()
    print(f"[split] train(<=2023)={len(tr):,} races={tr[COL_RID].nunique():,} / "
          f"valid(2024)={len(vl):,} races={vl[COL_RID].nunique():,}")

    wp = winner_pay()
    for d in (tr, vl):
        d["rid_s"] = d[COL_RID].astype(str).str[:16]
        d["winner_tansho"] = d["rid_s"].map(wp).fillna(100.0)

    tr = apply_enc(tr, encs); vl = apply_enc(vl, encs)
    dtr = make_ds(tr, feats); dvl = make_ds(vl, feats)

    print("[train] LGBM LambdaRank ...")
    model = lgb.train(PARAMS, dtr, num_boost_round=3000, valid_sets=[dvl],
                      callbacks=[lgb.early_stopping(120), lgb.log_evaluation(200)])
    print(f"[train] best_iter={model.best_iteration}  ndcg@5(valid2024)={model.best_score['valid_0']['ndcg@5']:.5f}")

    joblib.dump({"model": model, "feature_cols": feats, "encoders": encs,
                 "note": "v7: train<=2023/valid2024, 12 serve-dead feats dropped"}, OUT)
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
