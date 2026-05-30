"""
train_unified_rank.py
=======================
master_v2 で LightGBM LambdaRank を学習し unified_rank_v1 を作る。

Split: train ≤ 2022, valid = 2023, test = 2024-2025  (master_v2 の split 列に従う)
Target: 着順 (1=最上位) → label = max(0, 11 - 着順)   (LambdaRank は大きいほど上位として扱う)

特徴量:
  - master_v2 の数値列全部 (リーク列 = 着順, fukusho_flag, roi_target 等 除く)
  - カテゴリ列は label-encode (train で fit)

再現性: seed=42, deterministic=True. 入力 master_v2 が同じなら同じ出力。

実行: python train_unified_rank.py

出力: models/unified_rank_v1.pkl
       reports/unified_rank_v1_eval.json
"""
from __future__ import annotations

import io
import json
import sys
import warnings
from itertools import groupby
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE       = Path(__file__).parent
MASTER_CSV = BASE / "data/master_v2_20130105-20251228.csv"
OUT_MODEL  = BASE / "models/unified_rank_v2.pkl"
OUT_EVAL   = BASE / "reports/unified_rank_v2_eval.json"
SEED       = 42

from race_relative_feats import add_race_relative_feats

COL_RID    = "レースID(新/馬番無)"
COL_JYUN   = "着順"
COL_BAN    = "馬番"

# 目的変数直系 → 学習から除く
LEAK_COLS = {
    "着順", "fukusho_flag", "roi_target",
    "レースID(新)", "レースID(新/馬番無)",
    "馬名", "レース名", "発走時刻", "date_dt", "日付",
    "血統登録番号", "split",
}

# カテゴリ列
CAT_COLS = [
    "場所", "芝・ダ", "コース区分", "芝(内・外)", "馬場状態", "天気",
    "クラス名", "種牡馬", "父タイプ名", "母父馬", "母父タイプ名", "毛色",
    "馬主(最新/仮想)", "生産者", "騎手コード", "調教師コード",
    "年齢限定", "限定", "性別限定", "指定条件", "重量種別", "性別",
    "ブリンカー", "前走場所", "前芝・ダ", "前走馬場状態", "前走競走種別", "前好走",
]


def load_data():
    print(f"[load] {MASTER_CSV}")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    print(f"  rows={len(df):,}  cols={len(df.columns)}")

    # 着順あり行のみ
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]).copy()
    df["label"] = np.clip(11 - df[COL_JYUN].astype(int), 0, 10).astype(int)
    print(f"  着順あり: {len(df):,}")

    print("  [race-relative feats 付加]")
    df = add_race_relative_feats(df)
    print(f"  cols after: {len(df.columns)}")
    return df


def split_df(df):
    tr = df[df["split"] == "train"].copy()
    vl = df[df["split"] == "valid"].copy()
    te = df[df["split"] == "test"].copy()
    print(f"  train={len(tr):,}  valid={len(vl):,}  test={len(te):,}")
    return tr, vl, te


def fit_encoders(tr: pd.DataFrame) -> dict:
    encs = {}
    for c in CAT_COLS:
        if c not in tr.columns:
            continue
        le = LabelEncoder()
        vals = tr[c].astype(str).fillna("__NaN__")
        le.fit(pd.concat([vals, pd.Series(["__NaN__"])], ignore_index=True))
        encs[c] = le
    return encs


def apply_encoders(df: pd.DataFrame, encs: dict) -> pd.DataFrame:
    df = df.copy()
    for c, le in encs.items():
        if c not in df.columns: continue
        v = df[c].astype(str).fillna("__NaN__")
        known = set(le.classes_)
        v = v.where(v.isin(known), "__NaN__")
        df[c] = le.transform(v)
    return df


def select_features(df: pd.DataFrame) -> list[str]:
    feats = []
    for c in df.columns:
        if c in LEAK_COLS or c == "label":
            continue
        feats.append(c)
    return feats


def make_lgb_dataset(df: pd.DataFrame, feats: list[str]):
    df = df.sort_values(COL_RID).reset_index(drop=True)
    X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    y = df["label"].values.astype(int)
    groups = np.array([len(list(g)) for _, g in groupby(df[COL_RID])])
    return lgb.Dataset(X, label=y, group=groups, free_raw_data=False), X, y, groups, df


def train(tr, vl, feats):
    print(f"\n[train] features={len(feats)}")
    ds_tr, *_ = make_lgb_dataset(tr, feats)
    ds_vl, *_ = make_lgb_dataset(vl, feats)

    params = {
        "objective":                  "lambdarank",
        "lambdarank_truncation_level": 5,
        "metric":                     "ndcg",
        "eval_at":                    [1, 3, 5],
        "learning_rate":              0.05,
        "num_leaves":                 63,
        "max_depth":                  -1,
        "min_data_in_leaf":           50,
        "feature_fraction":           0.8,
        "bagging_fraction":           0.8,
        "bagging_freq":               5,
        "lambda_l1":                  0.1,
        "lambda_l2":                  0.1,
        "verbose":                   -1,
        "n_jobs":                    -1,
        "seed":                      SEED,
        "deterministic":             True,
        "force_col_wise":            True,
    }
    model = lgb.train(
        params, ds_tr, num_boost_round=3000, valid_sets=[ds_vl],
        callbacks=[lgb.early_stopping(150, verbose=False),
                   lgb.log_evaluation(period=200)],
    )
    print(f"  best iter: {model.best_iteration}")
    return model


# ============================================================
# 評価 (着順ベース各指標 + 三連単)
# ============================================================
def softmax_race(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def evaluate(df: pd.DataFrame, model, feats: list[str], label: str):
    """レース単位で score → softmax → p1 = 単勝予想, top3 = 複勝予想, perm = 三連単.
    勝ち馬、着順データのみで成績を集計."""
    X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    df = df.copy()
    df["_score"] = model.predict(X)

    races = df.groupby(COL_RID, sort=False)
    n_race = 0
    top1_hit = 0   # 単勝 pred top1 が 1 着
    top3_hit = 0   # 複勝 pred top1 が 3 着以内
    order_hit = 0  # 三連単 pred top-1 perm が実際の 1-2-3 着
    for rid, g in races:
        if len(g) < 3:
            continue
        jyun = g[COL_JYUN].astype(int).values
        scores = g["_score"].values
        order = np.argsort(-scores)  # スコア降順
        ban_by_order = g[COL_BAN].values[order]
        actual_sorted = g.sort_values(COL_JYUN)[COL_BAN].astype(int).values[:3]
        n_race += 1
        # 単勝
        if int(ban_by_order[0]) == int(actual_sorted[0]):
            top1_hit += 1
        # 複勝 top1
        if int(ban_by_order[0]) in set(int(x) for x in actual_sorted[:3]):
            top3_hit += 1
        # 三連単 pred 上位 3 頭
        if len(ban_by_order) >= 3 and \
           int(ban_by_order[0]) == int(actual_sorted[0]) and \
           int(ban_by_order[1]) == int(actual_sorted[1]) and \
           int(ban_by_order[2]) == int(actual_sorted[2]):
            order_hit += 1

    res = {
        "label":        label,
        "n_races":      n_race,
        "tansho_hit":   top1_hit,
        "tansho_rate":  top1_hit / n_race if n_race else 0.0,
        "fukusho_hit":  top3_hit,
        "fukusho_rate": top3_hit / n_race if n_race else 0.0,
        "santan_hit":   order_hit,
        "santan_rate":  order_hit / n_race if n_race else 0.0,
    }
    print(f"  [{label}] n={n_race:5,} 単勝={res['tansho_rate']*100:5.2f}% "
          f"複勝={res['fukusho_rate']*100:5.2f}% 三連単={res['santan_rate']*100:5.3f}%")
    return res


def main():
    print("=" * 60)
    print("train_unified_rank.py  (seed=42, deterministic=True)")
    print("=" * 60)

    df = load_data()
    tr, vl, te = split_df(df)

    encs = fit_encoders(tr)
    tr = apply_encoders(tr, encs)
    vl = apply_encoders(vl, encs)
    te = apply_encoders(te, encs)

    feats = select_features(tr)
    print(f"\n特徴量 ({len(feats)}):  {feats[:10]}... (+{len(feats)-10})")

    model = train(tr, vl, feats)

    OUT_MODEL.parent.mkdir(exist_ok=True)
    joblib.dump({
        "model":        model,
        "feature_cols": feats,
        "encoders":     encs,
        "cat_cols":     CAT_COLS,
        "split":        {"train_max": 2022, "valid": 2023, "test": [2024, 2025]},
        "seed":         SEED,
        "master_csv":   MASTER_CSV.name,
        "description":  "unified_rank_v2: LambdaRank + race-relative feats (z, rank_in_race)",
    }, OUT_MODEL)
    print(f"\n[saved] {OUT_MODEL}")

    # 分割ごとに評価 + test は 2024/2025 分解
    print("\n=== evaluation ===")
    te["year"] = te[COL_RID].astype(str).str[:4].astype(int)
    results = [
        evaluate(tr, model, feats, "train ≤2022"),
        evaluate(vl, model, feats, "valid 2023"),
        evaluate(te[te["year"]==2024], model, feats, "test 2024"),
        evaluate(te[te["year"]==2025], model, feats, "test 2025"),
        evaluate(te, model, feats, "test 2024+2025"),
    ]

    OUT_EVAL.parent.mkdir(exist_ok=True)
    with open(OUT_EVAL, "w", encoding="utf-8") as f:
        json.dump({
            "seed": SEED, "best_iter": model.best_iteration,
            "n_features": len(feats),
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {OUT_EVAL}")


if __name__ == "__main__":
    main()
