# -*- coding: utf-8 -*-
"""
DNF-inclusive baseline 再構築（算術矛盾の是正版）。

前回(dnf_inclusive_baseline.py)の誤りを修正する:
  1. 年別(2023/2024/2025)を完全分離し、DNFあり/なしレース別も**同じ年の中で**
     集計する(全期間ブレンドと年別を混在させない)。
  2. 加重平均恒等式 overall = dnf_share*dnf_top3 + (1-dnf_share)*non_dnf_top3
     を許容誤差1e-12でhard gate検証する(不一致ならAssertionErrorで停止)。
  3. Effect A(denominator-only、DNF馬は現行の未修正パイプライン定義でスコア)と
     Effect B(feature-semantic correction、全員corrected特徴で再スコア)を
     明確に分離する(前回はB相当の構築をAと称して混同していた)。
  4. score provenanceを年ごとに明示する。
  5. 2013-2022(train)は診断値、2023(valid)/2024-2025(test)のみ正式指標として扱う。

実行: venv311\\Scripts\\python.exe -m analysis.mcond.p0_dnf_history_parity_audit.reconciled_dnf_baseline
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
from build_dataset import add_rolling_stats, add_horse_rolling_stats, add_pace_features  # noqa: E402
from build_master_v2 import merge_hosei, merge_training  # noqa: E402
from analysis.mcond.exp13_nonfinish_risk_dev.gate0b_dnf_full_starter_score import (  # noqa: E402
    build_full_pre_dropna_universe_all_cols, compute_history_features_expanding,
    kako5_for_target_rows,
)
from analysis.mcond.p0_dnf_history_parity_audit.build_corrected_universe import (  # noqa: E402
    load_full_raw_universe,
)
from analysis.mcond.p0_dnf_history_parity_audit.dnf_buggy_definition_features import (  # noqa: E402
    compute_buggy_course_jockey_for_dnf, compute_buggy_kako5_for_dnf,
)

OUT_DIR = Path(__file__).resolve().parent / "out"
MASTER_V2 = BASE / "data" / "master_v2_20130105-20251228.csv"
MODEL_PKL = BASE / "models" / "unified_rank_v6.pkl"
SHADOW_MODEL_PKL = BASE / "models" / "unified_rank_v6_shadow_corrected.pkl"
SHADOW_19 = Path(__file__).resolve().parent / "shadow" / "corrected_19features.parquet"

COL_RID16, COL_BAN, COL_JYUN = "レースID(新/馬番無)", "馬番", "着順"
HIST6 = ["course_n_prev", "course_win_rate", "course_top3_rate",
          "jockey_n_prev", "jockey_win_rate", "jockey_top3_rate"]
ALL19 = HIST6 + ["kako5_avg_pos", "kako5_std_pos", "kako5_best_pos", "kako5_avg_agari3f",
                  "kako5_best_agari3f", "kako5_same_td_ratio", "kako5_same_dist_ratio",
                  "kako5_same_place_ratio", "kako5_pos_trend", "kako5_race_count",
                  "kako5_expected_good_count", "kako5_hidden_good_count", "kako5_same_cond_best_pos"]

TRAIN_END, VALID_END = 20221231, 20231231


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_of(date_int: int) -> str:
    if date_int <= TRAIN_END:
        return "train"
    if date_int <= VALID_END:
        return "valid"
    return "test"


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


def race_metrics(scores: np.ndarray, jyun: np.ndarray, is_dnf: np.ndarray) -> dict:
    w = np.exp(scores - scores.max())
    p_win = w / w.sum()
    actual_win = np.where(is_dnf, 0.0, (jyun == 1).astype(float))
    hon = int(np.argmax(scores))
    hon_top3 = bool((not is_dnf[hon]) and jyun[hon] <= 3)
    eps = 1e-9
    p_clip = np.clip(p_win, eps, 1 - eps)
    logloss = float(-(actual_win * np.log(p_clip) + (1 - actual_win) * np.log(1 - p_clip)).mean())
    brier = float(np.mean((p_win - actual_win) ** 2))
    return {"hon_top3": hon_top3, "logloss": logloss, "brier": brier, "has_dnf": bool(is_dnf.any())}


def build_race_table(v2: pd.DataFrame, v2_score_col: str, v2_jyun_col: str,
                      dnf_scored: pd.DataFrame, dnf_score_col: str) -> pd.DataFrame:
    """finisher(v2, スコア列=v2_score_col, 着順列=v2_jyun_col)にdnf_scored
    (スコア列=dnf_score_col)を合流させ、レース単位でold(finisher-only、
    DNFなしの場合と同一)/new(dnf-inclusive)両方の指標を計算する。"""
    dnf_by_race = dnf_scored.groupby(COL_RID16) if len(dnf_scored) else {}
    rows = []
    for rid16, g in v2.groupby(COL_RID16):
        if len(g) < 3:
            continue
        scores_fin = g[v2_score_col].values
        jyun_fin = g[v2_jyun_col].values
        has_dnf = len(dnf_scored) and rid16 in dnf_by_race.groups
        m_old = race_metrics(scores_fin, jyun_fin, np.zeros(len(g), dtype=bool))
        if has_dnf:
            dg = dnf_by_race.get_group(rid16)
            scores_all = np.concatenate([scores_fin, dg[dnf_score_col].values])
            jyun_all = np.concatenate([jyun_fin, np.full(len(dg), np.nan)])
            is_dnf_all = np.concatenate([np.zeros(len(g), dtype=bool), np.ones(len(dg), dtype=bool)])
        else:
            scores_all, jyun_all, is_dnf_all = scores_fin, jyun_fin, np.zeros(len(g), dtype=bool)
        m_new = race_metrics(scores_all, jyun_all, is_dnf_all)
        rows.append({
            "rid16": rid16, "year": int(str(rid16)[:4]), "has_dnf": bool(has_dnf),
            "old_hon_top3": m_old["hon_top3"], "old_logloss": m_old["logloss"], "old_brier": m_old["brier"],
            "new_hon_top3": m_new["hon_top3"], "new_logloss": m_new["logloss"], "new_brier": m_new["brier"],
        })
    return pd.DataFrame(rows)


def verify_identity_hard_gate(race_df: pd.DataFrame, value_col: str, label: str, tol=1e-12):
    """overall = dnf_share*dnf_val + (1-dnf_share)*non_dnf_val を年ごとにhard gate検証する。
    不一致ならAssertionErrorで停止する(レポート生成をFAILさせる)。"""
    failures = []
    for year, g in race_df.groupby("year"):
        n = len(g)
        n_dnf = int(g["has_dnf"].sum())
        n_non = n - n_dnf
        overall = float(g[value_col].mean())
        dnf_share = n_dnf / n
        dnf_val = float(g.loc[g["has_dnf"], value_col].mean()) if n_dnf > 0 else 0.0
        non_val = float(g.loc[~g["has_dnf"], value_col].mean()) if n_non > 0 else 0.0
        reconstructed = dnf_share * dnf_val + (1 - dnf_share) * non_val
        diff = abs(overall - reconstructed)
        if diff > tol:
            failures.append({"year": int(year), "col": value_col, "label": label,
                              "overall": overall, "reconstructed": reconstructed, "diff": diff})
    if failures:
        raise AssertionError(
            f"[HARD GATE FAIL] weighted-average identity violated for {label}/{value_col}: {failures}")
    return True


def main():
    t0 = time.time()
    print("[1] pre-dropna full universe構築(全120特徴の原材料込み)...")
    full = build_full_pre_dropna_universe_all_cols()
    raw = load_full_raw_universe()
    dnf_keys = raw[raw["_finish_class"] == "dnf"][[COL_RID16, COL_BAN]].copy()
    dnf_keys[COL_RID16] = dnf_keys[COL_RID16].astype(np.int64)
    dnf_keys[COL_BAN] = dnf_keys[COL_BAN].astype(np.int64)
    print(f"    全期間DNF = {len(dnf_keys):,}  ({time.time()-t0:.0f}s)")

    print("[2] 共通101特徴(raw passthrough + pre-dropna rolling + asof)を構築...")
    full2 = add_rolling_stats(full.copy())
    full2 = add_horse_rolling_stats(full2)
    full2 = add_pace_features(full2)
    dnf_common = full2.merge(dnf_keys, on=[COL_RID16, COL_BAN], how="inner")
    dnf_common = merge_hosei(dnf_common)
    dnf_common = merge_training(dnf_common)
    print(f"    DNF共通特徴行 = {len(dnf_common):,}  ({time.time()-t0:.0f}s)")

    print("[3] Effect A用(buggy definition)19特徴を構築...")
    v2 = pd.read_csv(MASTER_V2, encoding="utf-8-sig", low_memory=False)
    # dnf_common(full2由来)は既に場所/芝・ダ/距離/騎手コード/血統登録番号/日付を
    # 生列として持つため、追加のmergeは不要。
    dnf_cj_buggy = compute_buggy_course_jockey_for_dnf(v2, dnf_common)
    dnf_k5_buggy = compute_buggy_kako5_for_dnf(v2, dnf_common)
    dnf_A = dnf_common.copy()
    for c in HIST6:
        dnf_A[c] = dnf_A.merge(dnf_cj_buggy[[COL_RID16, COL_BAN, c]], on=[COL_RID16, COL_BAN], how="left")[c].values
    # kako5_*(13、ALL19対象) + kako5_avg_ninki等(常時NaN, N/A) + hist_same_*(4、confirmed_immune)
    # の全kako5由来列をdnf_k5_buggyから転記する(モデルのfeature_colsに含まれるため欠落させない)。
    k5_extra_cols = [c for c in dnf_k5_buggy.columns if c not in (COL_RID16, COL_BAN)]
    for c in k5_extra_cols:
        dnf_A[c] = dnf_A.merge(dnf_k5_buggy[[COL_RID16, COL_BAN, c]], on=[COL_RID16, COL_BAN], how="left")[c].values
    print(f"    完了 ({time.time()-t0:.0f}s)")

    print("[4] Effect B用(corrected definition)19特徴を構築...")
    full3 = compute_history_features_expanding(full2)
    dnf_rows_corr = full3.merge(dnf_keys, on=[COL_RID16, COL_BAN], how="inner")
    dnf_rows_corr = merge_hosei(dnf_rows_corr)
    dnf_rows_corr = merge_training(dnf_rows_corr)
    kako5_feats_corr = kako5_for_target_rows(full, dnf_keys)
    dnf_B = dnf_rows_corr.merge(kako5_feats_corr, on=[COL_RID16, COL_BAN], how="left")
    print(f"    完了 ({time.time()-t0:.0f}s)")

    print("[5] モデルロード & スコアリング(A/B、finisher old/corrected)...")
    bundle = joblib.load(MODEL_PKL)
    model, feats, encs = bundle["model"], bundle["feature_cols"], bundle["encoders"]

    def score(df):
        X = apply_encoders(df.reindex(columns=feats), encs)[feats].apply(
            pd.to_numeric, errors="coerce").fillna(-9999).values
        return model.predict(X)

    v2 = v2.copy()
    v2["_score_old"] = score(v2)
    v2[COL_JYUN] = pd.to_numeric(v2[COL_JYUN], errors="coerce")

    shadow19 = pd.read_parquet(SHADOW_19)
    v2_corr = v2.copy()
    key = [COL_RID16, COL_BAN]
    shadow_idx = shadow19.set_index(key)
    v2_idx = v2.set_index(key)
    for c in ALL19:
        v2_corr[c] = shadow_idx.loc[v2_idx.index, c].values
    v2_corr["_score_B"] = score(v2_corr)

    # 「score不能」の定義: raw CSV(lgbm/cat/torch/add)のJOINキーに実在しない
    # DNF行のみ(=dnf_keysに対しdnf_common/dnf_Bへ落ちる過程でinner joinから
    # 漏れた行)。個々の特徴が一部NaNであること自体は本番同様fillna(-9999)で
    # 扱う(モデルはNaNを許容する設計であり、行を落とす理由にはならない)。
    dnf_A_valid = dnf_A.copy()
    dnf_A_valid["_score_A"] = score(dnf_A_valid)
    incomplete_A = len(dnf_keys) - len(dnf_A_valid)

    dnf_B_valid = dnf_B.copy()
    dnf_B_valid["_score_B"] = score(dnf_B_valid)
    incomplete_B = len(dnf_keys) - len(dnf_B_valid)
    print(f"    完了 ({time.time()-t0:.0f}s)  DNF行: A構築可={len(dnf_A_valid)}/{len(dnf_keys)}"
          f"(raw JOIN不能{incomplete_A})  B構築可={len(dnf_B_valid)}/{len(dnf_keys)}(raw JOIN不能{incomplete_B})")

    print("[6] Effect C(shadow retrained model)でB特徴をスコアリング...")
    shadow_bundle = joblib.load(SHADOW_MODEL_PKL)
    shadow_model = shadow_bundle["model"]

    def score_shadow(df):
        X = apply_encoders(df.reindex(columns=feats), encs)[feats].apply(
            pd.to_numeric, errors="coerce").fillna(-9999).values
        return shadow_model.predict(X)

    v2_corr["_score_C"] = score_shadow(v2_corr)
    dnf_B_valid["_score_C"] = score_shadow(dnf_B_valid)
    print(f"    完了 ({time.time()-t0:.0f}s)")

    print("[7] レース単位テーブル構築(Effect A / Effect B / Effect C)...")
    # Effect A: finisherは現行の既存スコア(_score_old)のまま、DNFだけbuggy定義でスコア追加
    race_A = build_race_table(v2, "_score_old", COL_JYUN, dnf_A_valid, "_score_A")
    # Effect B: finisher・DNFとも corrected特徴 + 現行モデルでスコア
    race_B = build_race_table(v2_corr, "_score_B", COL_JYUN, dnf_B_valid, "_score_B")
    # Effect C: finisher・DNFとも corrected特徴 + shadow再学習モデルでスコア
    race_C = build_race_table(v2_corr, "_score_C", COL_JYUN, dnf_B_valid, "_score_C")
    print(f"    完了 ({time.time()-t0:.0f}s)")

    print("[8] hard gate検証(1e-12)...")
    for label, df in [("A", race_A), ("B", race_B), ("C", race_C)]:
        verify_identity_hard_gate(df, "new_hon_top3", label)
        verify_identity_hard_gate(df, "old_hon_top3", label)
    print("    PASS: 全ての年・全ての効果でoverall = dnf_share*dnf_val + (1-dnf_share)*non_dnf_val が誤差1e-12以内で成立")

    print("[9] 年別・DNFあり/なし別テーブルを構築...")
    result = {"hard_gate": "PASS (tolerance=1e-12)", "by_year": {}}
    master_v2_hash = sha256_file(MASTER_V2)
    model_hash = sha256_file(MODEL_PKL)
    shadow_model_hash = sha256_file(SHADOW_MODEL_PKL)

    for year in sorted(race_A["year"].unique()):
        yA = race_A[race_A["year"] == year]
        yB = race_B[race_B["year"] == year]
        yC = race_C[race_C["year"] == year]

        def block(df, old_col="old_hon_top3", new_col="new_hon_top3"):
            n = len(df)
            n_dnf = int(df["has_dnf"].sum())
            n_non = n - n_dnf
            old_overall = float(df[old_col].mean())
            new_overall = float(df[new_col].mean())
            old_dnf = float(df.loc[df["has_dnf"], old_col].mean()) if n_dnf else None
            new_dnf = float(df.loc[df["has_dnf"], new_col].mean()) if n_dnf else None
            old_non = float(df.loc[~df["has_dnf"], old_col].mean()) if n_non else None
            new_non = float(df.loc[~df["has_dnf"], new_col].mean()) if n_non else None
            share = n_dnf / n if n else 0.0
            reconstructed_new = share * (new_dnf or 0) + (1 - share) * (new_non or 0)
            return {
                "n_race_total": n, "n_dnf_race": n_dnf, "n_non_dnf_race": n_non,
                "dnf_share": round(share, 6),
                "old_overall_hon_top3": old_overall, "new_overall_hon_top3": new_overall,
                "old_dnf_race_hon_top3": old_dnf, "new_dnf_race_hon_top3": new_dnf,
                "old_non_dnf_race_hon_top3": old_non, "new_non_dnf_race_hon_top3": new_non,
                "reconstructed_new_overall_from_weighted_avg": reconstructed_new,
                "diff_reconstructed_vs_direct": abs(reconstructed_new - new_overall),
            }

        result["by_year"][str(year)] = {
            "split": split_of(int(f"{year}0101")),
            "effect_A_denominator_only": block(yA),
            "effect_B_feature_correction": block(yB),
            "effect_C_retrained_model": block(yC),
        }

    result["score_provenance"] = {
        "model_pkl_sha256": model_hash,
        "shadow_model_pkl_sha256": shadow_model_hash,
        "training_master_sha256": master_v2_hash,
        "note_per_year": "train(<=2022-12-31)=in-sample(モデルが学習時に直接見た行、"
                          "遡及適用は診断値でありOOS性能とは呼ばない)。"
                          "valid(2023-01-01〜2023-12-31)=selection/development。"
                          "test(2024-01-01〜)=OOS。全ての年で__score_old__は"
                          "master_v2に保存済みの値ではなく本監査で再計算したもの"
                          "(現行model.predict()を変更なしで使用)。",
        "hon_column_selection": "race内_score最大の馬(argmax、tie-breakなし="
                                 "本データでの完全な数値タイは未観測)",
        "n_dnf_total": int(len(dnf_keys)),
        "n_dnf_effect_A_scoreable": int(len(dnf_A_valid)),
        "n_dnf_effect_A_unscoreable": int(incomplete_A),
        "n_dnf_effect_B_scoreable": int(len(dnf_B_valid)),
        "n_dnf_effect_B_unscoreable": int(incomplete_B),
        "unscoreable_handling": "スコア不能なDNF行はその特定レースのDNF追加対象から除外し、"
                                 "'full starter'とは呼ばない。除外件数を上記に明示する。",
    }

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "dnf_baseline_reconciled.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"hard_gate": result["hard_gate"], "score_provenance": result["score_provenance"]},
                      ensure_ascii=False, indent=1))
    for year, v in result["by_year"].items():
        print(f"\n=== {year} ({v['split']}) ===")
        for eff in ("effect_A_denominator_only", "effect_B_feature_correction", "effect_C_retrained_model"):
            b = v[eff]
            print(f"  {eff}: n={b['n_race_total']} dnf_share={b['dnf_share']:.4f} "
                  f"old_overall={b['old_overall_hon_top3']:.4f} new_overall={b['new_overall_hon_top3']:.4f} "
                  f"recon_diff={b['diff_reconstructed_vs_direct']:.2e}")

    print(f"\n[saved] {out_path}")
    print(f"TOTAL TIME: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
