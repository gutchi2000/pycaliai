# -*- coding: utf-8 -*-
"""shadow parquet(既に構築済み)を再利用し、符号付きraw score差の分布を
確認する(読み取り専用、追加の高コスト計算なし)。"""
from __future__ import annotations
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(BASE))
from analysis.mcond.p0_dnf_history_parity_audit.build_corrected_universe import COL_RID16, COL_BAN  # noqa: E402

MASTER_V2 = BASE / "data" / "master_v2_20130105-20251228.csv"
MODEL_PKL = BASE / "models" / "unified_rank_v6.pkl"
SHADOW = Path(__file__).resolve().parent / "shadow" / "corrected_19features.parquet"
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


def main():
    v2 = pd.read_csv(MASTER_V2, encoding="utf-8-sig", low_memory=False)
    shadow = pd.read_parquet(SHADOW)
    key = [COL_RID16, COL_BAN]
    merged = v2[key + ALL19].merge(shadow[key + ALL19], on=key, suffixes=("_stored", "_corrected"))
    diff_mask = pd.Series(False, index=merged.index)
    for c in ALL19:
        a = pd.to_numeric(merged[f"{c}_stored"], errors="coerce")
        b = pd.to_numeric(merged[f"{c}_corrected"], errors="coerce")
        diff_mask |= ((a - b).abs() > 1e-9) | (a.isna() != b.isna())
    affected = merged.loc[diff_mask, key]

    bundle = joblib.load(MODEL_PKL)
    model, feats, encs = bundle["model"], bundle["feature_cols"], bundle["encoders"]
    v2a = v2.merge(affected, on=key, how="inner").copy()
    sha = shadow.merge(affected, on=key, how="inner")[key + ALL19].copy()

    X_stored_df = v2a[feats].copy()
    X_true_df = v2a[feats].copy()
    sha_idx = sha.set_index(key)
    v2a_idx = v2a.set_index(key)
    for c in ALL19:
        X_true_df[c] = sha_idx.loc[v2a_idx.index, c].values

    Xs = apply_encoders(X_stored_df, encs)[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    Xt = apply_encoders(X_true_df, encs)[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    score_stored = model.predict(Xs)
    score_corrected = model.predict(Xt)
    delta = score_corrected - score_stored

    n = len(delta)
    n_neg = int((delta < -1e-9).sum())
    n_pos = int((delta > 1e-9).sum())
    n_zero = n - n_neg - n_pos
    print(f"n_affected={n}")
    print(f"negative(corrected<stored, つまり修正で評価が下がる)={n_neg} ({n_neg/n*100:.1f}%)")
    print(f"positive(corrected>stored, つまり修正で評価が上がる)={n_pos} ({n_pos/n*100:.1f}%)")
    print(f"zero={n_zero}")
    print(f"mean(signed)={delta.mean():.5f}  median(signed)={np.median(delta):.5f}")


if __name__ == "__main__":
    main()
