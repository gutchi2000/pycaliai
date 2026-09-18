# -*- coding: utf-8 -*-
"""
build_bp.py — 行動予測方式の逸脱 (surprisal = -log P(実際の行動 | 陣営履歴, 馬の文脈))
====================================================================================
各行動 (6次元) について多クラス LightGBM で「陣営が今回どれを選ぶか」を予測する。
予測対象は陣営の行動のみ。レース結果は学習しない。

時点安全: 対象年 Y の行の確率は、date < Y-01-01 の行だけで学習したモデルで出す
          (expanding window, 年単位)。コード内 assert で強制。
入力の文脈 ctx_* は build_features.py が前日までの情報で作ったもの。
パラメータは事前固定 (探索しない): num_leaves 31, learning_rate 0.05, 300 木, min_child 200。

出力: data/_research/mcond/exp01_features_bp.parquet
      (exp01_features.parquet に bp_*_s と bp_total を付けたもの)
実行: python -m analysis.mcond.exp01_choice_dev.build_bp
"""
from __future__ import annotations
from pathlib import Path

import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
IN = BASE / "data/_research/mcond/exp01_features.parquet"
OUT = BASE / "data/_research/mcond/exp01_features_bp.parquet"
ACTIONS = {"a_interval": 5, "a_dist": 3, "a_venue": 2, "a_surface": 2, "a_cls": 3, "a_jockey": 2}
PRED_YEARS = list(range(2016, 2026))
FIRST_TRAIN_YEAR = 2014
PARAMS = dict(num_leaves=31, learning_rate=0.05, n_estimators=300, min_child_samples=200,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=42)
EPS = 1e-4


# 厳格版 (仕様 §7 を文字どおり読む): 過去のレース結果に由来する文脈を行動モデルから外す
STRICT_DROP = ["ctx_prev_fin", "ctx_prev_margin", "ctx_prev_jq"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="前走着順・前走着差・前走騎手の過去成績を行動モデルの入力から外す")
    args = ap.parse_args()
    out = OUT.with_name("exp01_features_bp_strict.parquet") if args.strict else OUT
    f = pd.read_parquet(IN)
    f["year"] = f["date"].dt.year
    ctx = [c for c in f.columns if c.startswith("ctx_")]
    if args.strict:
        ctx = [c for c in ctx if c not in STRICT_DROP]
    print(f"rows={len(f):,}  文脈特徴 {len(ctx)} 本")
    for a, k in ACTIONS.items():
        f[f"bp_{a[2:]}_s"] = np.nan

    for y in PRED_YEARS:
        tr = (f["year"] >= FIRST_TRAIN_YEAR) & (f["year"] <= y - 1)
        te = f["year"] == y
        assert f.loc[tr, "date"].max() < pd.Timestamp(f"{y}-01-01"), "学習に対象年以降が混入"
        for a, k in ACTIONS.items():
            clf = lgb.LGBMClassifier(objective="multiclass" if k > 2 else "binary", **PARAMS)
            clf.fit(f.loc[tr, ctx], f.loc[tr, a].astype(int))
            p = clf.predict_proba(f.loc[te, ctx])
            obs = f.loc[te, a].astype(int).to_numpy()
            classes = list(clf.classes_)
            idx = np.array([classes.index(o) if o in classes else -1 for o in obs])
            po = np.where(idx >= 0, p[np.arange(len(obs)), np.clip(idx, 0, None)], EPS)
            f.loc[te, f"bp_{a[2:]}_s"] = -np.log(np.clip(po, EPS, 1.0))
        print(f"  {y}: train {int(tr.sum()):,} 行 (≤{y-1}) → 予測 {int(te.sum()):,} 行", flush=True)

    bcols = [f"bp_{a[2:]}_s" for a in ACTIONS]
    f["bp_total"] = f[bcols].sum(axis=1, min_count=len(bcols))
    f.to_parquet(out, index=False)
    print(f"saved -> {out}  (文脈 {len(ctx)} 本{'、厳格版' if args.strict else ''})")
    print(f[f.year >= 2016].groupby("year")[bcols + ["bp_total"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
