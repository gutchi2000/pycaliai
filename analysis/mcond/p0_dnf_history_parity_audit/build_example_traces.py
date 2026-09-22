# -*- coding: utf-8 -*-
"""item3: DNF後再出走馬の実例追跡（読み取り専用）。
既存のshadow parquet(corrected_19features.parquet)とmaster_v2を再利用し、
追加の高コスト計算(kako5の全馬ループ等)は行わない。"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(BASE))
from analysis.mcond.p0_dnf_history_parity_audit.build_corrected_universe import (  # noqa: E402
    load_full_raw_universe, COL_RID16, COL_BAN, COL_PEDIGREE, COL_PLACE, COL_SURFACE,
)

OUT_DIR = Path(__file__).resolve().parent / "out"
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


def race_probs(scores: np.ndarray) -> np.ndarray:
    w = np.exp(scores - scores.max())
    return w / w.sum()


def main():
    print("[1] full raw universe再構築(過去走履歴の抽出用)...")
    full = load_full_raw_universe()
    full = full.sort_values([COL_PEDIGREE, "date_dt"]).reset_index(drop=True)

    print("[2] shadow/master_v2読み込み、affected行を特定...")
    v2 = pd.read_csv(MASTER_V2, encoding="utf-8-sig", low_memory=False)
    shadow = pd.read_parquet(SHADOW)
    key = [COL_RID16, COL_BAN]
    merged = v2[key + ALL19 + [COL_PEDIGREE, COL_PLACE, COL_SURFACE, "騎手コード"]].merge(
        shadow[key + ALL19], on=key, suffixes=("_stored", "_corrected"))
    diff_mask = pd.Series(False, index=merged.index)
    for c in ALL19:
        a = pd.to_numeric(merged[f"{c}_stored"], errors="coerce")
        b = pd.to_numeric(merged[f"{c}_corrected"], errors="coerce")
        diff_mask |= ((a - b).abs() > 1e-9) | (a.isna() != b.isna())
    affected = merged.loc[diff_mask].copy()

    # 多様性を確保: 芝/ダ・場所違い・を手動で選定(スコア差が大きい行から候補抽出)
    candidates_pool = affected.copy()
    picks = []
    seen_combo = set()
    for _, row in candidates_pool.sort_values(COL_RID16).iterrows():
        combo = (row[COL_SURFACE], row[COL_PLACE])
        if combo in seen_combo:
            continue
        seen_combo.add(combo)
        picks.append((row[COL_RID16], row[COL_BAN]))
        if len(picks) >= 8:
            break

    print(f"    選定した例: {picks}")

    bundle = joblib.load(MODEL_PKL)
    model, feats, encs = bundle["model"], bundle["feature_cols"], bundle["encoders"]

    traces = []
    for rid16, ban in picks:
        row_v2 = v2[(v2[COL_RID16] == rid16) & (v2[COL_BAN] == ban)]
        if row_v2.empty:
            continue
        row_v2 = row_v2.iloc[0]
        hid = row_v2[COL_PEDIGREE]

        hist = full[full[COL_PEDIGREE] == hid].copy()
        # レースID(新/馬番無)は整数、日付順ソート済みのfullを使い、対象行より前の行を厳密に抽出
        target_row_full = full[(full[COL_RID16] == rid16) & (full[COL_BAN] == ban)]
        if target_row_full.empty:
            continue
        target_pos = target_row_full.index[0]
        horse_positions = hist.index.tolist()
        seq_i = horse_positions.index(target_pos) if target_pos in horse_positions else None
        if seq_i is None:
            continue
        past5 = hist.loc[horse_positions[max(0, seq_i - 5):seq_i]]
        raw_history = []
        for _, pr in past5.iterrows():
            raw_history.append({
                "date": str(pr.get("日付")),
                "race_id": str(pr[COL_RID16]), "finish_class": pr["_finish_class"],
                "着順": None if pr["_finish_class"] != "numeric" else float(pr["着順"]),
                "surface": pr[COL_SURFACE], "place": pr[COL_PLACE], "distance": pr.get("距離"),
                "jockey": pr.get("騎手コード"),
            })
        dnf_races = [r for r in raw_history if r["finish_class"] == "dnf"]

        stored19 = {c: row_v2[c] if c in row_v2 else None for c in ALL19}
        corrected_row = shadow[(shadow[COL_RID16] == rid16) & (shadow[COL_BAN] == ban)]
        corrected19 = corrected_row.iloc[0][ALL19].to_dict() if not corrected_row.empty else {}

        # race内スコア(修正前後)・順位・確率
        race_rows = v2[v2[COL_RID16] == rid16].copy()
        Xs = apply_encoders(race_rows[feats].copy(), encs)[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
        score_stored_all = model.predict(Xs)
        race_rows["_score_stored"] = score_stored_all

        race_rows_corr = race_rows.copy()
        for c in ALL19:
            corr_map = shadow[shadow[COL_RID16] == rid16].set_index(COL_BAN)[c]
            race_rows_corr[c] = race_rows_corr[COL_BAN].map(corr_map).fillna(race_rows_corr[c])
        Xt = apply_encoders(race_rows_corr[feats].copy(), encs)[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
        score_corrected_all = model.predict(Xt)
        race_rows["_score_corrected"] = score_corrected_all

        rank_stored = race_rows["_score_stored"].rank(ascending=False, method="min")
        rank_corrected = race_rows["_score_corrected"].rank(ascending=False, method="min")
        p_stored = race_probs(race_rows["_score_stored"].values)
        p_corrected = race_probs(race_rows["_score_corrected"].values)
        target_pos_in_race = race_rows[COL_BAN].values.tolist().index(ban)

        traces.append({
            "race_id16": str(rid16), "馬番": int(ban), "血統登録番号": str(hid),
            "surface": row_v2[COL_SURFACE], "place": row_v2[COL_PLACE],
            "raw_history_last5_before_this_race": raw_history,
            "dnf_races_in_window": dnf_races,
            "training_master_stored_19feat": {k: (None if pd.isna(v) else float(v)) for k, v in stored19.items()},
            "corrected_shadow_19feat": {k: (None if pd.isna(v) else float(v)) for k, v in corrected19.items()},
            "live_history_builder_note": (
                "TARGETの週次kako5 CSV/jockey_stats.csv等は過去日付分を遡及保存していないため、"
                "この履歴実例についてlive serve相当値を再現することはできない。"
                "LIVE_SERVE_DNF_TRACE.mdの定性的結論(course/jockey 2013-2025分は"
                "training同様のバグを継承、kako5系はbuild_from_kako5()の別実装で"
                "window slot自体を欠落させる異なるバグ)を参照。"
            ),
            "v6_raw_score": {
                "stored": float(race_rows.loc[race_rows[COL_BAN] == ban, "_score_stored"].iloc[0]),
                "corrected": float(race_rows.loc[race_rows[COL_BAN] == ban, "_score_corrected"].iloc[0]),
                "delta": float(race_rows.loc[race_rows[COL_BAN] == ban, "_score_corrected"].iloc[0] -
                               race_rows.loc[race_rows[COL_BAN] == ban, "_score_stored"].iloc[0]),
            },
            "race_rank": {
                "stored": int(rank_stored.iloc[target_pos_in_race]),
                "corrected": int(rank_corrected.iloc[target_pos_in_race]),
            },
            "win_probability": {
                "stored": float(p_stored[target_pos_in_race]),
                "corrected": float(p_corrected[target_pos_in_race]),
            },
            "n_horses_in_race": int(len(race_rows)),
        })

    out_path = OUT_DIR / "example_traces.json"
    out_path.write_text(json.dumps(traces, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"[saved] {out_path}  n_traces={len(traces)}")


if __name__ == "__main__":
    main()
