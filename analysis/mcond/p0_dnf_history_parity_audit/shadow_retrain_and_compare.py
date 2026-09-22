# -*- coding: utf-8 -*-
"""
item6: モデル影響確認（読み取り専用の比較実験。本番modelは一切変更しない）。

比較対象:
  A) current v6 model × current master(現行master_v2の120特徴のまま)
  B) current v6 model × corrected features(19特徴のみ修正版に差し替え)
  C) shadow retrained v6 × corrected features(同一ハイパーパラメータ・
     同一alpha・同一num_boost_round・同一seedで再学習、Optuna再探索なし)

評価はvalid(2023年development)のみで行う。2024/2025年(test)は本プロジェクト
全体の既存規律([[feedback_asof_population_definition]]等)に倣い開封しない。

出力: models/unified_rank_v6_shadow_corrected.pkl(本番modelとは別artifact、
本番modelファイルは上書きしない) + out/model_impact_comparison.json

実行: venv311\\Scripts\\python.exe -m analysis.mcond.p0_dnf_history_parity_audit.shadow_retrain_and_compare
"""
from __future__ import annotations
import json
import sys
import time
from itertools import groupby
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

BASE = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE))
import pl_probs as PL  # noqa: E402
from optuna_v6_marks import (  # noqa: E402
    evaluate_marks_with_ece, composite_score, ndcg_at_k,
    load_winner_tansho_pay, LEAK_COLS, CAT_COLS,
)

OUT_DIR = Path(__file__).resolve().parent / "out"
MASTER_V2 = BASE / "data" / "master_v2_20130105-20251228.csv"
MODEL_PKL = BASE / "models" / "unified_rank_v6.pkl"
SHADOW_19 = Path(__file__).resolve().parent / "shadow" / "corrected_19features.parquet"
SHADOW_MODEL_OUT = BASE / "models" / "unified_rank_v6_shadow_corrected.pkl"

COL_RID, COL_JYUN, COL_BAN = "レースID(新/馬番無)", "着順", "馬番"
ALL19 = ["course_n_prev", "course_win_rate", "course_top3_rate", "jockey_n_prev",
          "jockey_win_rate", "jockey_top3_rate", "kako5_avg_pos", "kako5_std_pos",
          "kako5_best_pos", "kako5_avg_agari3f", "kako5_best_agari3f",
          "kako5_same_td_ratio", "kako5_same_dist_ratio", "kako5_same_place_ratio",
          "kako5_pos_trend", "kako5_race_count", "kako5_expected_good_count",
          "kako5_hidden_good_count", "kako5_same_cond_best_pos"]


def apply_encoders(df, encs):
    df = df.copy()
    for c, le in encs.items():
        if c not in df.columns:
            continue
        v = df[c].astype(str).fillna("__NaN__")
        known = set(le.classes_)
        v = v.where(v.isin(known), "__NaN__")
        df[c] = le.transform(v)
    return df


def make_dataset(d, feats, alpha=0.0):
    d = d.sort_values(COL_RID).reset_index(drop=True)
    X = d[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    y = d["label"].values.astype(int)
    g = np.array([len(list(gr)) for _, gr in groupby(d[COL_RID])])
    if alpha > 0 and "winner_tansho" in d.columns:
        w = (1.0 + alpha * np.log1p(d["winner_tansho"].values / 100.0)).astype(float)
    else:
        w = np.ones(len(d), dtype=float)
    return lgb.Dataset(X, label=y, group=g, weight=w, free_raw_data=False), X, d


def compute_composite_full(model, vl_df, X_vl, seed):
    vl_scored = vl_df.copy()
    vl_scored["_score"] = model.predict(X_vl)
    unique_rids = vl_scored[COL_RID].drop_duplicates().values
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    scores, metrics_all = [], None
    for _, eval_idx in kf.split(unique_rids):
        rids_eval = unique_rids[eval_idx]
        sub = vl_scored[vl_scored[COL_RID].isin(rids_eval)]
        m = evaluate_marks_with_ece(sub)
        if m is None:
            continue
        scores.append(composite_score(m))
        if metrics_all is None:
            metrics_all = {k: [v] for k, v in m.items()}
        else:
            for k, v in m.items():
                metrics_all[k].append(v)
    return float(np.mean(scores)), {k: float(np.mean(v)) for k, v in metrics_all.items()}


def main():
    t0 = time.time()
    print("[1] bundle(現行v6)読み込み...")
    bundle = joblib.load(MODEL_PKL)
    model_A, feats, encs = bundle["model"], bundle["feature_cols"], bundle["encoders"]
    bp = bundle["optuna_best_params"]
    alpha = bundle["sample_weight_alpha"]
    seed = bundle["seed"]
    n_trees_production = model_A.num_trees()
    print(f"    feats={len(feats)}  alpha={alpha:.4f}  seed={seed}  "
          f"production_n_trees={n_trees_production}")

    print("[2] master_v2 + shadow(corrected 19特徴)読み込み、corrected masterを構築...")
    v2 = pd.read_csv(MASTER_V2, encoding="utf-8-sig", low_memory=False)
    shadow19 = pd.read_parquet(SHADOW_19)
    key = [COL_RID, COL_BAN]
    corrected_master = v2.copy()
    shadow_idx = shadow19.set_index(key)
    v2_idx = v2.set_index(key)
    for c in ALL19:
        corrected_master[c] = shadow_idx.loc[v2_idx.index, c].values

    def prep_split(df_master):
        df = df_master.copy()
        df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
        df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]).copy()
        df["label"] = np.clip(6 - df[COL_JYUN].astype(int), 0, 5).astype(int)
        tr = df[df["split"] == "train"].copy()
        vl = df[df["split"] == "valid"].copy()
        tr = apply_encoders(tr, encs)
        vl = apply_encoders(vl, encs)
        win_pay = load_winner_tansho_pay()
        tr["rid_s"] = tr[COL_RID].astype(str)
        tr["winner_tansho"] = tr["rid_s"].map(win_pay).fillna(100.0)
        return tr, vl

    print("[3] A) current v6 × current master(基準)...")
    tr_orig, vl_orig = prep_split(v2)
    ds_vl_orig, X_vl_orig, vl_df_orig = make_dataset(vl_orig, feats, alpha=0.0)
    composite_A, metrics_A = compute_composite_full(model_A, vl_df_orig, X_vl_orig, seed)
    print(f"    composite={composite_A*100:.3f}  ◎top3={metrics_A['hon_top3_rate']*100:.2f}%  "
          f"ECE={metrics_A['ece_high_p']:.4f}  ({time.time()-t0:.0f}s)")

    print("[4] B) current v6(再学習なし) × corrected features(19特徴のみ差し替え)...")
    tr_corr, vl_corr = prep_split(corrected_master)
    ds_vl_corr, X_vl_corr, vl_df_corr = make_dataset(vl_corr, feats, alpha=0.0)
    composite_B, metrics_B = compute_composite_full(model_A, vl_df_corr, X_vl_corr, seed)
    print(f"    composite={composite_B*100:.3f}  ◎top3={metrics_B['hon_top3_rate']*100:.2f}%  "
          f"ECE={metrics_B['ece_high_p']:.4f}  ({time.time()-t0:.0f}s)")

    print("[5] C) shadow retrained v6(同一ハイパーパラメータ) × corrected features...")
    ds_tr_corr, _, _ = make_dataset(tr_corr, feats, alpha=alpha)
    params = {
        "objective": "lambdarank", "lambdarank_truncation_level": 5,
        "metric": "ndcg", "eval_at": [1, 3, 5],
        "learning_rate": bp["lr"], "num_leaves": bp["num_leaves"],
        "max_depth": bp["max_depth"], "min_data_in_leaf": bp["min_data_in_leaf"],
        "feature_fraction": bp["ff"], "bagging_fraction": bp["bf"], "bagging_freq": 5,
        "lambda_l1": bp["l1"], "lambda_l2": bp["l2"],
        "verbose": -1, "n_jobs": -1, "seed": seed,
        "deterministic": True, "force_col_wise": True, "feature_pre_filter": False,
    }
    model_C = lgb.train(params, ds_tr_corr, num_boost_round=n_trees_production,
                         valid_sets=[ds_vl_corr], callbacks=[lgb.log_evaluation(period=500)])
    print(f"    n_trees={model_C.num_trees()} ({time.time()-t0:.0f}s)")
    composite_C, metrics_C = compute_composite_full(model_C, vl_df_corr, X_vl_corr, seed)
    print(f"    composite={composite_C*100:.3f}  ◎top3={metrics_C['hon_top3_rate']*100:.2f}%  "
          f"ECE={metrics_C['ece_high_p']:.4f}  ({time.time()-t0:.0f}s)")

    print("[6] shadow model保存(本番modelとは別artifact、本番は変更しない)...")
    joblib.dump({
        "model": model_C, "feature_cols": feats, "encoders": encs,
        "cat_cols": CAT_COLS, "seed": seed,
        "master_csv": "corrected_master(19features_from_p0_dnf_audit)",
        "optuna_best_params": bp, "sample_weight_alpha": alpha,
        "note": "P0 DNF history parity audit用shadow model。ハイパーパラメータ・alpha・"
                "num_boost_round(=本番production_n_trees)は本番unified_rank_v6.pklと完全同一。"
                "Optuna再探索・特徴追加・閾値調整は一切行っていない。学習データの19特徴のみ"
                "DNF_SEMANTIC_SPEC.md準拠の修正版に差し替え。",
    }, SHADOW_MODEL_OUT)

    result = {
        "production_n_trees": n_trees_production,
        "optuna_best_params": bp, "sample_weight_alpha": alpha, "seed": seed,
        "A_current_model_current_master": {"composite_pct": composite_A * 100, **metrics_A},
        "B_current_model_corrected_features": {"composite_pct": composite_B * 100, **metrics_B},
        "C_shadow_retrained_corrected_features": {"composite_pct": composite_C * 100, **metrics_C},
        "delta_B_minus_A": {
            "composite_pct": (composite_B - composite_A) * 100,
            "hon_top3_rate_pt": (metrics_B["hon_top3_rate"] - metrics_A["hon_top3_rate"]) * 100,
            "ece_high_p": metrics_B["ece_high_p"] - metrics_A["ece_high_p"],
            "ndcg5": metrics_B["ndcg5"] - metrics_A["ndcg5"],
        },
        "delta_C_minus_A": {
            "composite_pct": (composite_C - composite_A) * 100,
            "hon_top3_rate_pt": (metrics_C["hon_top3_rate"] - metrics_A["hon_top3_rate"]) * 100,
            "ece_high_p": metrics_C["ece_high_p"] - metrics_A["ece_high_p"],
            "ndcg5": metrics_C["ndcg5"] - metrics_A["ndcg5"],
        },
        "evaluation_scope": "valid(2023年development)のみ。test(2024-2025年)は開封していない",
        "purpose_note": "新モデル探索が目的ではなく、19特徴のデータ定義修正がモデル出力に"
                        "与える影響の確認のみ。Optuna再探索・特徴追加・閾値調整は行っていない。",
    }
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "model_impact_comparison.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1, default=str))
    print(f"\n[saved] {out_path}")
    print(f"[saved] {SHADOW_MODEL_OUT}")
    print(f"TOTAL TIME: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
