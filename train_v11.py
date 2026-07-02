"""
train_v11.py — unified_rank_v11: v6 (120特徴) + grade_feats v1 (格文脈15特徴)
=============================================================================
Optuna 再探索なし。v6 pkl の optuna_best_params を固定し、
  1. valid(2023) early stopping で best_iter を決定
  2. v6 と同じ流儀で train のみ num_boost_round = best_iter×1.1 の最終学習
の2段で学習する (optuna_v6_marks.py:389-429 の retrain ブロックと同一手順)。

出自: 格13特徴ゲート (analysis/_tmp_grade_feats_gate.py, ΔAUC(複勝)+0.0130) の
servable 再定義 (grade_feats.py, 直近5走窓 = kako5 serve parity)。

実行: venv311/Scripts/python.exe train_v11.py
評価: build_pl_calibrators (tag上書き) → audit_marks.py --model v11
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

from optuna_v6_marks import (  # noqa: E402
    CAT_COLS, COL_JYUN, COL_RID, LEAK_COLS, MASTER_CSV, SEED, W_ECE_PENALTY,
    load_winner_tansho_pay, make_dataset,
)
from grade_feats import GRADE_FEATS_V1, add_grade_feats  # noqa: E402

V6_PKL = BASE / "models/unified_rank_v6.pkl"
OUT_MODEL = BASE / "models/unified_rank_v11.pkl"


def main() -> None:
    if OUT_MODEL.exists():
        print(f"[warn] {OUT_MODEL.name} は既存 → 上書きします")

    v6 = joblib.load(V6_PKL)
    bp = v6["optuna_best_params"]
    print(f"[params] v6 固定: {bp}")

    print(f"[load] {MASTER_CSV}")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]).copy()

    print("[feats] add_grade_feats(v1) ...")
    df = add_grade_feats(df, mode="v1")
    viol = df.attrs.get("grade_leak_violations", -1)
    print(f"  leak violations (グループ内日付逆行) = {viol}")
    if viol != 0:
        print("[ERROR] leak violation 検出 → 中止", file=sys.stderr)
        sys.exit(2)
    cov = {c: float(df[c].notna().mean()) for c in GRADE_FEATS_V1}
    print("  coverage: " + "  ".join(f"{k}={v:.2f}" for k, v in cov.items()))

    df["label"] = np.clip(6 - df[COL_JYUN].astype(int), 0, 5).astype(int)
    tr = df[df["split"] == "train"].copy()
    vl = df[df["split"] == "valid"].copy()
    print(f"  train={len(tr):,}  valid={len(vl):,}")

    # --- encoders (optuna_v6_marks.prep と同一手順) ---
    encs = {}
    for c in CAT_COLS:
        if c not in tr.columns:
            continue
        le = LabelEncoder()
        vals = tr[c].astype(str).fillna("__NaN__")
        le.fit(pd.concat([vals, pd.Series(["__NaN__"])], ignore_index=True))
        encs[c] = le

    def _apply(d):
        d = d.copy()
        for c, le in encs.items():
            if c not in d.columns:
                continue
            v = d[c].astype(str).fillna("__NaN__")
            known = set(le.classes_)
            v = v.where(v.isin(known), "__NaN__")
            d[c] = le.transform(v)
        return d

    tr, vl = _apply(tr), _apply(vl)
    feats = [c for c in tr.columns if c not in LEAK_COLS and c != "label"]

    # --- 特徴セット検証: v6 の120特徴 + gf 15特徴のみの増分であること ---
    extra = set(feats) - set(v6["feature_cols"])
    assert extra == set(GRADE_FEATS_V1), f"想定外の特徴差分: {extra ^ set(GRADE_FEATS_V1)}"
    missing = set(v6["feature_cols"]) - set(feats)
    assert not missing, f"v6特徴の欠落: {missing}"
    print(f"  feats = {len(feats)} (v6 {len(v6['feature_cols'])} + gf {len(GRADE_FEATS_V1)})")

    win_pay = load_winner_tansho_pay()
    tr["rid_s"] = tr[COL_RID].astype(str).str[:16] if tr[COL_RID].dtype == object \
        else tr[COL_RID].astype(str)
    tr["winner_tansho"] = tr["rid_s"].map(win_pay).fillna(100.0)

    params = {
        "objective": "lambdarank",
        "lambdarank_truncation_level": 5,
        "metric": "ndcg", "eval_at": [1, 3, 5],
        "learning_rate": bp["lr"],
        "num_leaves": bp["num_leaves"],
        "max_depth": bp["max_depth"],
        "min_data_in_leaf": bp["min_data_in_leaf"],
        "feature_fraction": bp["ff"],
        "bagging_fraction": bp["bf"],
        "bagging_freq": 5,
        "lambda_l1": bp["l1"], "lambda_l2": bp["l2"],
        "verbose": -1, "n_jobs": -1, "seed": SEED,
        "deterministic": True, "force_col_wise": True, "feature_pre_filter": False,
    }

    ds_tr, _, _ = make_dataset(tr, feats, alpha=bp["alpha"])
    ds_vl, _, _ = make_dataset(vl, feats, alpha=0.0)

    print("[train 1/2] early stopping で best_iter 決定 ...")
    m1 = lgb.train(params, ds_tr, num_boost_round=3000, valid_sets=[ds_vl],
                   callbacks=[lgb.early_stopping(100), lgb.log_evaluation(period=200)])
    best_iter = m1.best_iteration
    print(f"  best_iter = {best_iter}")

    print(f"[train 2/2] final (num_boost_round = {int(best_iter * 1.1)}) ...")
    ds_tr2, _, _ = make_dataset(tr, feats, alpha=bp["alpha"])
    model = lgb.train(params, ds_tr2, num_boost_round=int(best_iter * 1.1),
                      valid_sets=[ds_vl], callbacks=[lgb.log_evaluation(period=200)])

    joblib.dump({
        "model": model, "feature_cols": feats, "encoders": encs,
        "cat_cols": CAT_COLS, "seed": SEED,
        "master_csv": MASTER_CSV.name,
        "optuna_best_params": bp,
        "optuna_best_composite": v6.get("optuna_best_composite"),
        "n_folds": v6.get("n_folds"),
        "label_scheme": "clip(6 - 着順, 0, 5)",
        "sample_weight_alpha": bp["alpha"],
        "ece_penalty_weight": W_ECE_PENALTY,
        "grade_feats_mode": "v1",
        "description": "unified_rank_v11: v6の120特徴 + grade_feats v1 (格文脈15特徴, "
                       "直近5走窓=kako5 serve parity)。Optuna再探索なし (v6 best params 固定, "
                       "valid ES で best_iter 決定 → train 再学習 ×1.1)。"
                       "出自: 格13ゲート ΔAUC+0.0130 (analysis/_tmp_grade_feats_gate.py)。",
    }, OUT_MODEL)
    print(f"[saved] {OUT_MODEL}")

    # gf 特徴の gain 寄与を即時確認
    imp = dict(zip(feats, model.feature_importance("gain")))
    tot = sum(imp.values()) or 1.0
    gf_gain = sum(imp.get(c, 0.0) for c in GRADE_FEATS_V1)
    print(f"[gain] gf計 {gf_gain / tot * 100:.2f}%  内訳:")
    for c in sorted(GRADE_FEATS_V1, key=lambda c: -imp.get(c, 0.0)):
        print(f"  {c:24s} {imp.get(c, 0.0) / tot * 100:6.3f}%")


if __name__ == "__main__":
    main()
