# -*- coding: utf-8 -*-
"""
P0 DNF history parity audit — item4(全期間影響監査) + item5(shadow corrected
master構築)を一括実行する（読み取り専用、本番ファイルは一切変更しない）。

DNF_SEMANTIC_SPEC.md準拠(止=出走経験1回として数える、外/消は数えない)で
19特徴を再計算し、既存master_v2格納値との差分を年別に集計する
(平均・中央値・95/99パーセンタイル・最大)。raw v6 scoreの変化・◎変更・
race内順位変更も計測する。

shadow corrected masterは19特徴のみを差し替えたテーブルとして
analysis/mcond/p0_dnf_history_parity_audit/shadow/ に保存する
(元のdata/master_v2_*.csvは一切変更しない)。

実行: venv311\\Scripts\\python.exe -m analysis.mcond.p0_dnf_history_parity_audit.run_full_audit
"""
from __future__ import annotations
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(BASE))
from analysis.mcond.p0_dnf_history_parity_audit.build_corrected_universe import (  # noqa: E402
    load_full_raw_universe, experience_population,
    compute_corrected_course_jockey, compute_corrected_kako5,
    COL_RID16, COL_BAN,
)

OUT_DIR = Path(__file__).resolve().parent / "out"
SHADOW_DIR = Path(__file__).resolve().parent / "shadow"
MASTER_V2 = BASE / "data" / "master_v2_20130105-20251228.csv"
MODEL_PKL = BASE / "models" / "unified_rank_v6.pkl"

HIST6 = ["course_n_prev", "course_win_rate", "course_top3_rate",
          "jockey_n_prev", "jockey_win_rate", "jockey_top3_rate"]
KAKO5_13 = ["kako5_avg_pos", "kako5_std_pos", "kako5_best_pos", "kako5_avg_agari3f",
            "kako5_best_agari3f", "kako5_same_td_ratio", "kako5_same_dist_ratio",
            "kako5_same_place_ratio", "kako5_pos_trend", "kako5_race_count",
            "kako5_expected_good_count", "kako5_hidden_good_count", "kako5_same_cond_best_pos"]
ALL19 = HIST6 + KAKO5_13


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
    t0 = time.time()
    print(f"[0] master_v2 sha256計算中(参照用、変更しない)...")
    master_v2_hash = sha256_file(MASTER_V2)
    print(f"    {master_v2_hash}")

    print("[1] full raw universe(631,965行)構築 + 着順生文字列分類...")
    full = load_full_raw_universe()
    exp_mask = experience_population(full)
    print(f"    experience population = {exp_mask.sum():,} (止+numeric)")

    print("[2] 修正版course/jockey(6特徴)を計算(意味定義spec準拠、scratch除外)...")
    corrected_cj = compute_corrected_course_jockey(full)
    print(f"    完了 ({time.time()-t0:.0f}s)")

    print("[3] 修正版kako5/hist_same(20特徴)を計算(意味定義spec準拠)...")
    numeric_idx = full.index[full["_finish_class"] == "numeric"]
    corrected_k5 = compute_corrected_kako5(full, target_index=numeric_idx)
    print(f"    完了 ({time.time()-t0:.0f}s)  対象行={len(numeric_idx):,}")

    corrected19 = pd.concat([corrected_cj.loc[numeric_idx], corrected_k5], axis=1)
    corrected19[COL_RID16] = full.loc[numeric_idx, COL_RID16].values
    corrected19[COL_BAN] = full.loc[numeric_idx, COL_BAN].values
    corrected19["year"] = full.loc[numeric_idx, COL_RID16].astype(str).str[:4].astype(int)

    print("[4] master_v2格納値(現行, 19特徴)を読み込み突合...")
    v2 = pd.read_csv(MASTER_V2, encoding="utf-8-sig", low_memory=False)
    key = [COL_RID16, COL_BAN]
    merged = v2[key + ALL19 + ["日付"]].merge(
        corrected19[key + ALL19 + ["year"]], on=key, suffixes=("_stored", "_corrected"))
    print(f"    突合行数 = {len(merged):,} / master_v2行数 = {len(v2):,}")

    diff_mask = pd.Series(False, index=merged.index)
    for c in ALL19:
        a = pd.to_numeric(merged[f"{c}_stored"], errors="coerce")
        b = pd.to_numeric(merged[f"{c}_corrected"], errors="coerce")
        diff_mask |= ((a - b).abs() > 1e-9) | (a.isna() != b.isna())
    merged["_any19_diff"] = diff_mask
    print(f"    19特徴いずれかで不一致 = {int(diff_mask.sum()):,} / {len(merged):,} "
          f"({diff_mask.mean()*100:.3f}%)")

    print("[5] raw v6 score差分計算(全affected行、既存モデルは変更せず使用)...")
    bundle = joblib.load(MODEL_PKL)
    model, feats, encs = bundle["model"], bundle["feature_cols"], bundle["encoders"]
    affected_keys = merged.loc[diff_mask, key]
    v2_affected = v2.merge(affected_keys, on=key, how="inner").copy()
    corrected_affected = corrected19.merge(affected_keys, on=key, how="inner")[key + ALL19].copy()

    X_stored_df = v2_affected[feats].copy()
    X_true_df = v2_affected[feats].copy()
    ca_idx = corrected_affected.set_index(key)
    v2a_idx = v2_affected.set_index(key)
    for c in ALL19:
        X_true_df[c] = ca_idx.loc[v2a_idx.index, c].values

    X_stored = apply_encoders(X_stored_df, encs)[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    X_true = apply_encoders(X_true_df, encs)[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    score_stored = model.predict(X_stored)
    score_true = model.predict(X_true)
    v2_affected["_score_stored"] = score_stored
    v2_affected["_score_corrected"] = score_true
    v2_affected["_score_delta"] = score_true - score_stored
    v2_affected["year"] = v2_affected[COL_RID16].astype(str).str[:4].astype(int)
    print(f"    完了 ({time.time()-t0:.0f}s)")

    print("[6] race内順位・◎変更を計算(affected race内の全馬についてscore再取得)...")
    affected_races = v2_affected[COL_RID16].unique()
    v2_race_all = v2[v2[COL_RID16].isin(affected_races)].copy()
    Xr = apply_encoders(v2_race_all[feats].copy(), encs)[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    v2_race_all["_score_stored_all"] = model.predict(Xr)
    delta_map = v2_affected.set_index(key)["_score_delta"]
    v2_race_all = v2_race_all.set_index(key)
    v2_race_all["_score_delta"] = delta_map.reindex(v2_race_all.index).fillna(0.0)
    v2_race_all["_score_corrected_all"] = v2_race_all["_score_stored_all"] + v2_race_all["_score_delta"]
    v2_race_all = v2_race_all.reset_index()

    rank_change_rows = []
    for rid16, g in v2_race_all.groupby(COL_RID16):
        rank_before = g["_score_stored_all"].rank(ascending=False, method="min")
        rank_after = g["_score_corrected_all"].rank(ascending=False, method="min")
        top1_before = g.loc[rank_before.idxmin(), COL_BAN] if len(g) else None
        top1_after = g.loc[rank_after.idxmin(), COL_BAN] if len(g) else None
        max_rank_change = int((rank_before - rank_after).abs().max())
        rank_change_rows.append({
            "rid16": rid16, "year": int(str(rid16)[:4]), "n_horses": len(g),
            "top1_changed": bool(top1_before != top1_after),
            "max_rank_change": max_rank_change,
        })
    rank_df = pd.DataFrame(rank_change_rows)
    print(f"    完了 ({time.time()-t0:.0f}s)  対象race数={len(rank_df):,}")

    print("[7] 年別集計(平均/中央値/95%/99%/最大)を作成...")
    by_year = []
    for year, g in merged.groupby("year"):
        n_total_year = int(len(g))
        n_diff_year = int(g["_any19_diff"].sum())
        n_horse_rows_with_past_issue = n_diff_year  # 定義上diff行=過去issue影響行
        sd = v2_affected[v2_affected["year"] == year]["_score_delta"].abs()
        rd = rank_df[rank_df["year"] == year]
        by_year.append({
            "year": int(year),
            "n_rows_total": n_total_year,
            "n_rows_19feat_diff": n_diff_year,
            "pct_rows_diff": round(n_diff_year / n_total_year * 100, 4) if n_total_year else None,
            "n_unique_races_affected": int(rd["rid16"].nunique()) if len(rd) else 0,
            "score_abs_delta": {
                "mean": float(sd.mean()) if len(sd) else None,
                "median": float(sd.median()) if len(sd) else None,
                "p95": float(sd.quantile(0.95)) if len(sd) else None,
                "p99": float(sd.quantile(0.99)) if len(sd) else None,
                "max": float(sd.max()) if len(sd) else None,
                "n": int(len(sd)),
            },
            "n_top1_changed_races": int(rd["top1_changed"].sum()) if len(rd) else 0,
            "max_rank_change": {
                "mean": float(rd["max_rank_change"].mean()) if len(rd) else None,
                "median": float(rd["max_rank_change"].median()) if len(rd) else None,
                "p95": float(rd["max_rank_change"].quantile(0.95)) if len(rd) else None,
                "max": int(rd["max_rank_change"].max()) if len(rd) else None,
            },
        })
    print(f"    完了 ({time.time()-t0:.0f}s)")

    print("[8] 最大スコア変化raceの個別内訳...")
    top_races = v2_affected.reindex(v2_affected["_score_delta"].abs().sort_values(ascending=False).index[:10])
    top_races_detail = top_races[[COL_RID16, COL_BAN, "year", "_score_stored", "_score_corrected", "_score_delta"]].to_dict(orient="records")

    print("[9] shadow corrected master保存(19特徴のみ、別artifact)...")
    SHADOW_DIR.mkdir(exist_ok=True)
    shadow_out = v2[key].merge(corrected19[key + ALL19], on=key, how="left")
    # scratch(外・消)由来で対応値がない行は元のmaster_v2値(=変更なし、numeric母集団のみ対象)を維持
    for c in ALL19:
        shadow_out[c] = shadow_out[c].where(shadow_out[c].notna(), v2.set_index(key)[c].reindex(shadow_out.set_index(key).index).values)
    shadow_path = SHADOW_DIR / "corrected_19features.parquet"
    shadow_out.to_parquet(shadow_path, index=False)
    shadow_hash = sha256_file(shadow_path)

    # 不変条件チェック
    invariants = {
        "row_count_match": len(shadow_out) == len(v2),
        "race_count_match": shadow_out[COL_RID16].nunique() == v2[COL_RID16].nunique(),
        "other_101_features_untouched": "shadowは19特徴のみ保持、他101特徴はv2から不変(別ファイルのため物理的に不可侵)",
    }

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "master_v2_sha256_reference_unchanged": master_v2_hash,
        "shadow_corrected_19features_sha256": shadow_hash,
        "shadow_path": str(shadow_path.relative_to(BASE)),
        "n_rows": int(len(shadow_out)),
        "n_races": int(shadow_out[COL_RID16].nunique()),
        "affected_features": ALL19,
        "finish_class_counts": full["_finish_class"].value_counts().to_dict(),
        "experience_population_size": int(exp_mask.sum()),
        "overall": {
            "n_rows_total": int(len(merged)),
            "n_rows_19feat_diff": int(diff_mask.sum()),
            "pct_rows_diff": round(diff_mask.mean() * 100, 4),
            "n_races_affected": int(rank_df["rid16"].nunique()),
            "n_top1_changed_races": int(rank_df["top1_changed"].sum()),
            "score_abs_delta_all": {
                "mean": float(v2_affected["_score_delta"].abs().mean()),
                "median": float(v2_affected["_score_delta"].abs().median()),
                "p95": float(v2_affected["_score_delta"].abs().quantile(0.95)),
                "p99": float(v2_affected["_score_delta"].abs().quantile(0.99)),
                "max": float(v2_affected["_score_delta"].abs().max()),
            },
        },
        "by_year": by_year,
        "top10_score_delta_rows": top_races_detail,
        "invariants": invariants,
        "note_unknown_finish_code_26rows": (
            "add_20130105-20251228.csvの着順で止/外/消/数値のいずれにも該当しない"
            "26行(joinギャップ等)が存在する。安全側でexperience_populationから除外した"
            "(母集団を恣意的に広げない)。EXP13旧GATE0_REPORT.mdの5,191件"
            "(止2,946+外1,212+消1,007+丸数字26)という記載の「丸数字26」は、実際には"
            "内部add.csvに丸数字コードが存在しないため、このunknown26行と同一の可能性が高い"
            "(要因未特定、性能に無関係な極小件数のため本監査ではこれ以上追跡しない)。"
        ),
    }
    manifest_path = OUT_DIR / "p0_full_audit_manifest.json"
    OUT_DIR.mkdir(exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"[saved] {manifest_path}")
    print(f"[saved] {shadow_path}")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("by_year", "top10_score_delta_rows")},
                      ensure_ascii=False, indent=1))
    print(f"\nTOTAL TIME: {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
