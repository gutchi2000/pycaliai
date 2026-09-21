# -*- coding: utf-8 -*-
"""
Gate 0B（EXP13再開） — raw v6 score parity（読み取り専用、学習は行わない）
==========================================================================
2026-09-22。`gate0b_course_jockey_history_parity.py` で course_n_prev/win_rate/
top3_rate・jockey_n_prev/win_rate/top3_rateの6特徴に実測diffが見つかった
(3,401〜3,854/626,774行)。この特徴diffが実際にunified_rank_v6のraw score(model.predict)
にどれだけ影響するかを、**既存の学習済みモデルに対してのみ**score before/after比較で
確認する。モデルの再学習・修正は一切行わない。

比較対象:
  X_stored: master_v2に格納されている現行120特徴そのまま
  X_true  : 上記のうちcourse_*/jockey_n_prev系6列だけをtrue-universe(DNF込み母集団)
            再計算値に差し替え、残り114列は現行のまま(既にstructurally_immune/
            likely_immuneと判定済みのため)

実行: venv311\\Scripts\\python.exe -m analysis.mcond.exp13_nonfinish_risk_dev.gate0b_raw_score_parity
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(BASE))
from analysis.p0_5_verification.reconstruct_true_pipeline_universe import (  # noqa: E402
    build_full_pre_dropna_universe,
)

OUT_DIR = Path(__file__).resolve().parent / "out"
MASTER_V2 = BASE / "data" / "master_v2_20130105-20251228.csv"
MODEL_PKL = BASE / "models" / "unified_rank_v6.pkl"

COL_RID16, COL_BAN, COL_DATE = "レースID(新/馬番無)", "馬番", "日付"
COL_PEDIGREE, COL_PLACE, COL_SURFACE, COL_DIST = "血統登録番号", "場所", "芝・ダ", "距離"
COL_JOCKEY, COL_JYUN = "騎手コード", "着順"
HIST6 = ["course_n_prev", "course_win_rate", "course_top3_rate",
          "jockey_n_prev", "jockey_win_rate", "jockey_top3_rate"]


def compute_history_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[COL_DATE] = pd.to_numeric(df[COL_DATE], errors="coerce").astype("Int64")
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df["_is_win"] = (df[COL_JYUN] == 1).astype("Int8")
    df["_is_top3"] = (df[COL_JYUN] <= 3).astype("Int8")

    def dist_band(d):
        if pd.isna(d):
            return "?"
        d = int(d)
        if d <= 1400:
            return "短"
        if d <= 1700:
            return "マ"
        if d <= 2200:
            return "中"
        return "長"

    df["_dist_b"] = df[COL_DIST].apply(dist_band)
    df["_course_key"] = (df[COL_PLACE].astype(str) + "|" +
                          df[COL_SURFACE].astype(str) + "|" + df["_dist_b"])
    df = df.sort_values([COL_PEDIGREE, COL_DATE]).reset_index(drop=True)
    g = df.groupby([COL_PEDIGREE, "_course_key"])
    df["course_n_prev"] = g.cumcount()
    df["course_wins_prev"] = g["_is_win"].cumsum().astype("Int64") - df["_is_win"].astype("Int64")
    df["course_top3_prev"] = g["_is_top3"].cumsum().astype("Int64") - df["_is_top3"].astype("Int64")
    df["course_win_rate"] = np.where(df["course_n_prev"] > 0,
                                      df["course_wins_prev"] / df["course_n_prev"], np.nan)
    df["course_top3_rate"] = np.where(df["course_n_prev"] > 0,
                                       df["course_top3_prev"] / df["course_n_prev"], np.nan)
    gj = df.groupby([COL_PEDIGREE, COL_JOCKEY])
    df["jockey_n_prev"] = gj.cumcount()
    df["jockey_wins_prev"] = gj["_is_win"].cumsum().astype("Int64") - df["_is_win"].astype("Int64")
    df["jockey_top3_prev"] = gj["_is_top3"].cumsum().astype("Int64") - df["_is_top3"].astype("Int64")
    df["jockey_win_rate"] = np.where(df["jockey_n_prev"] > 0,
                                      df["jockey_wins_prev"] / df["jockey_n_prev"], np.nan)
    df["jockey_top3_rate"] = np.where(df["jockey_n_prev"] > 0,
                                       df["jockey_top3_prev"] / df["jockey_n_prev"], np.nan)
    return df


def apply_encoders(df: pd.DataFrame, encs: dict) -> pd.DataFrame:
    df = df.copy()
    for c, le in encs.items():
        if c not in df.columns:
            continue
        v = df[c].astype(str).fillna("__NaN__")
        known = set(le.classes_)
        v = v.where(v.isin(known), "__NaN__")
        df[c] = le.transform(v)
    return df


def main() -> int:
    print("[1] true-universe版6特徴を再計算...")
    full = build_full_pre_dropna_universe()
    true_universe = compute_history_features(full.copy())
    survivors_true = true_universe[true_universe[COL_JYUN].notna()].copy()

    print("[2] master_v2 (120特徴 + キー) を読み込み...")
    bundle = joblib.load(MODEL_PKL)
    model, feats, encs = bundle["model"], bundle["feature_cols"], bundle["encoders"]
    v2 = pd.read_csv(MASTER_V2, encoding="utf-8-sig", low_memory=False)

    key = [COL_RID16, COL_BAN]
    v2_key = v2[key + HIST6].merge(
        survivors_true[key + HIST6], on=key, suffixes=("_stored", "_true"))
    diff_mask = pd.Series(False, index=v2_key.index)
    for c in HIST6:
        a = pd.to_numeric(v2_key[f"{c}_stored"], errors="coerce")
        b = pd.to_numeric(v2_key[f"{c}_true"], errors="coerce")
        diff_mask |= ((a - b).abs() > 1e-9) | (a.isna() != b.isna())
    affected_keys = v2_key.loc[diff_mask, key]
    print(f"    affected rows (union, course/jockey いずれかで不一致) = {len(affected_keys):,}")

    print("[3] X_stored / X_true 特徴行列を構築し、既存モデルでスコアリング...")
    v2_affected = v2.merge(affected_keys, on=key, how="inner").copy()
    true_affected = survivors_true.merge(affected_keys, on=key, how="inner")[key + HIST6].copy()

    X_stored_df = v2_affected[feats].copy()
    X_true_df = v2_affected[feats].copy()
    true_indexed = true_affected.set_index(key)
    v2_affected_idx = v2_affected.set_index(key)
    for c in HIST6:
        X_true_df[c] = true_indexed.loc[v2_affected_idx.index, c].values

    X_stored_enc = apply_encoders(X_stored_df, encs)
    X_true_enc = apply_encoders(X_true_df, encs)
    X_stored = X_stored_enc[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    X_true = X_true_enc[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values

    score_stored = model.predict(X_stored)
    score_true = model.predict(X_true)
    delta = score_true - score_stored
    abs_delta = np.abs(delta)

    tolerances = [1e-6, 1e-4, 1e-3, 1e-2]
    tol_report = {}
    for tol in tolerances:
        n_exceed = int((abs_delta > tol).sum())
        tol_report[str(tol)] = {"n_exceed": n_exceed,
                                 "pct_of_affected": round(n_exceed / len(abs_delta) * 100, 3) if len(abs_delta) else None}

    result = {
        "n_total_finisher_rows_2013_2025": int(len(survivors_true)),
        "n_affected_rows_course_or_jockey_diff": int(len(affected_keys)),
        "pct_affected_of_total": round(len(affected_keys) / len(survivors_true) * 100, 4),
        "raw_score_delta_stats": {
            "mean_abs_delta": float(abs_delta.mean()) if len(abs_delta) else None,
            "median_abs_delta": float(np.median(abs_delta)) if len(abs_delta) else None,
            "p95_abs_delta": float(np.percentile(abs_delta, 95)) if len(abs_delta) else None,
            "max_abs_delta": float(abs_delta.max()) if len(abs_delta) else None,
        },
        "n_rows_exceeding_tolerance": tol_report,
        "note": (
            "この比較は他114特徴(既にstructurally_immune/likely_immuneと判定済み)を"
            "現行値のまま固定し、course_*/jockey_n_prev系6列のみをtrue-universe"
            "(DNF込み母集団)再計算値へ差し替えた場合のraw score差である。"
            "モデル自体の再学習・修正は一切行っていない(既存model.predict()を"
            "そのまま使用)。"
        ),
    }

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "gate0b_raw_score_parity.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"\n[saved] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
