# -*- coding: utf-8 -*-
"""
stage0_dry_run.py — EXP16A Stage 0 の成果物 STAGE0_DRY_RUN.json を組み立てる
============================================================================
学習しない。2023 の結果指標は計算しない。ROI 評価も候補生成もしない。
入力 (すべて既存の成果物を読むだけ):
  out/race_population.json     (race_population.py)
  out/power_audit.json         (power_audit.py)
  out/market_provenance.json   (provenance.py)
  out/derivation_checks.json   (verify_growth.py)
  analysis/mcond/exp15_race_as_set_dev/out/feature_contract.json
ここで新たに実測するもの (artifact 契約):
  master_v2 の絶対パス・サイズ・mtime・sha256・row hash
  feature schema hash と、現在の master 実ヘッダに 111 列すべてが在るかの再検証
  race population hash (json の hash と official rid 一覧の hash)
  P0-5 (C1) 修正後 artifact かどうかの **実測判定**
  jockey_fuku90 / prev_hosei の provenance
出力: STAGE0_DRY_RUN.json
実行: python -m analysis.mcond.exp16a_close_market_residual_dev.stage0_dry_run
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .provenance import BASE, HERE, OUT, MASTER, TANPUK_DIR, sha256, load_master

EXP15_OUT = BASE / "analysis" / "mcond" / "exp15_race_as_set_dev" / "out"
RESEARCH = BASE / "data" / "_research" / "mcond" / "exp16a"
CROSSFIT_YEARS = [2019, 2020, 2021, 2022, 2023]
THRESHOLDS = {"absolute_practical_floor_nats_per_race": 0.005, "recovery_ratio_floor": 0.10,
              "ci_condition": "年層化 meeting-day bootstrap CI95 上限 < 0",
              "fixed_before": "検出力監査より先に固定した。監査結果で変更しない"}


def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def p05_c1_check(m: pd.DataFrame) -> dict:
    """同一 (調教師, レース) ブロック内で trainer_fuku30 が一定か = C1 修正後の署名"""
    cols = ["レースID(新/馬番無)", "調教師コード", "trainer_fuku30"]
    parts = []
    for ch in pd.read_csv(MASTER, encoding="utf-8-sig", dtype=str,
                          usecols=["日付"] + cols, chunksize=300_000):
        d = pd.to_numeric(ch["日付"], errors="coerce")
        parts.append(ch[(d >= 20160101) & (d <= 20231231)])
    x = pd.concat(parts, ignore_index=True)
    x["rid"] = x["レースID(新/馬番無)"].astype(str).str[:16]
    g = x.groupby(["rid", "調教師コード"])["trainer_fuku30"]
    nun, sz = g.nunique(dropna=False), g.size()
    multi = sz[sz >= 2]
    bad = int((nun[multi.index] > 1).sum())
    return {"blocks_with_2plus_horses": int(len(multi)), "blocks_with_multiple_values": bad,
            "is_post_p0_5_c1_artifact": bool(len(multi) > 0 and bad == 0),
            "method": "C1 (最小修正) は同一 (調教師, レース) ブロックへ先頭行の値を配る。"
                      "ブロック内に複数値が残っていれば修正前 artifact",
            "reference": "build_dataset.py add_rolling_stats / Vol.III 欠陥台帳 P0-5"}


def main():
    t0 = time.time()
    pop = json.loads((OUT / "race_population.json").read_text(encoding="utf-8"))
    pw = json.loads((OUT / "power_audit.json").read_text(encoding="utf-8"))
    prov = json.loads((OUT / "market_provenance.json").read_text(encoding="utf-8"))
    der = json.loads((OUT / "derivation_checks.json").read_text(encoding="utf-8"))
    contract = json.loads((EXP15_OUT / "feature_contract.json").read_text(encoding="utf-8"))

    # ---- artifact 契約: master の実測
    st = os.stat(MASTER)
    m = load_master()
    row_hash_now = sha_text("\n".join(sorted(m["rid16"] + "_" + m["ban"].astype(str))))
    header = pd.read_csv(MASTER, encoding="utf-8-sig", nrows=0).columns.tolist()
    clean = contract["features"]["clean"]
    missing = [c for c in clean if c not in header]
    npz = np.load(RESEARCH / "official_races_le2022.npz", allow_pickle=True)
    rid_le2022 = sorted(npz["rid16"].astype(str).tolist())

    artifact = {
        "master_v2": {
            "absolute_path": str(MASTER), "size_bytes": int(st.st_size),
            "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
            "sha256": sha256(MASTER),
            "row_hash_2016_2023": row_hash_now,
            "row_hash_matches_exp15": bool(row_hash_now == contract.get("row_hash_2016_2023")),
            "exp15_recorded_row_hash": contract.get("row_hash_2016_2023"),
            "rehash_policy": "EXP15 の hash を流用せず、今回読み込んだ実ファイルから再計算して一致を検証した",
        },
        "feature_schema": {
            "source": str(EXP15_OUT / "feature_contract.json"),
            "file_sha256": sha256(EXP15_OUT / "feature_contract.json"),
            "n_clean": contract["n_clean"],
            "schema_hash": sha_text("\n".join(sorted(clean))),
            "columns_missing_from_current_master_header": missing,
            "verified": bool(not missing),
        },
        "race_population": {
            "json_sha256": sha256(OUT / "race_population.json"),
            "official_rid_list_sha256_le2022": sha_text("\n".join(rid_le2022)),
            "n_official_races_le2022": len(rid_le2022),
        },
        "p0_5_status": p05_c1_check(m),
        "feature_provenance": {
            "jockey_fuku90": {
                "train_side": "build_dataset.py add_rolling_stats。直近90『馬走』の複勝率を "
                              "shift(1) で as-of 計算し、同一 (騎手, レース) ブロックへ先頭値を配る (C1, 2026-09-11)",
                "serve_side": "data/jockey_stats.csv のスナップショットを weekly へ join "
                              "(predict_weekly.py:523 / app.py:858)。**as-of ではなく現在値**",
                "risk": "train は as-of・serve はスナップショットという provenance 差がある。"
                        "EXP16A は master_v2 (train 側) のみを使うが、将来 production へ渡す場合はこの差が残る",
                "measured_non_null_share_in_official_set":
                    pw["directions"]["prefixed_residual_linear"]["features_and_signs"]
                      .get("jockey_fuku90", {}).get("non_null_share"),
            },
            "prev_hosei": {
                "train_side": "build_master_v2.py:70 で data/hosei の 『前走補正』を rename。"
                              "TARGET 由来の補正タイム (前走)",
                "serve_side": "make_weekly_hosei.py。旧実装は前々走を入れる off-by-one があり "
                              "(make_weekly_hosei.py:166-176)、proxy 方式で修正済み",
                "risk": "train 側は TARGET 実測値・serve 側は proxy。EXP16A は train 側のみを使う",
                "measured_non_null_share_in_official_set":
                    pw["directions"]["prefixed_residual_linear"]["features_and_signs"]
                      .get("prev_hosei", {}).get("non_null_share"),
            },
        },
        "pending_stage1": {
            "encoder_hash": "Stage 1 で category encoder を作った時点で oof_manifest.json に記録する",
            "model_hash_per_seed": "各 (年, seed) の LightGBM model と score npz の sha256",
            "q1_direction_power": "rolling OOF 作成直後、結果開封前に power_audit を log(Q1/π) 方向で再実行",
        },
    }

    # ---- crossfit 設計 (年ごとの期間と実数)
    py = pop["per_year"]
    crossfit = {}
    for y in CROSSFIT_YEARS:
        crossfit[str(y)] = {
            "q1_train_years": f"2013..{y-2}", "q1_early_stopping_year": y - 1, "predict_year": y,
            "q2_q3_coef_fit_oof_years": f"2016..{y-1}",
            "calibration_tau_subset_placebo_cells": f"<= {y-1} のみ",
            "official_races_in_predict_year": py[str(y)]["official_eligible"],
            "meeting_days_in_predict_year": py[str(y)]["meeting_days_official"],
        }

    ref = pop["reference_values_by_year_le2022"]
    t2 = pw["tier2_empirical_cluster_power"]
    dry = {
        "purpose": "Stage 0 の実行可能性確認。学習なし・2019-2023 の結果は評価しない・ROI なし",
        "naming": {"development_design": "retrospective_rolling_crossfit_development",
                    "note": "2019-2023 は過去研究で接触済みの期間を含むため、独立した未使用 holdout ではない。"
                            "仮説探索・内部再現性の評価である。2024/2025 は封印を維持する"},
        "official_race_set": {
            "rules": pop["official_race_set_rule"],
            "funnel_order": pop["funnel_order"],
            "per_year": py,
            "totals": pop["totals"],
            "provisional_population_note": pop["provisional_population_note"],
        },
        "retrospective_rolling_crossfit_development": crossfit,
        "reference_values_by_year_le2022": ref,
        "reference_values_2022_official_set": {
            "n_races": ref["2022"]["n_races"],
            "terminal_close_market": {
                "race_categorical_logloss": ref["2022"]["terminal_close_market_logloss"],
                "favorite_top1_rate": ref["2022"]["favorite_top1_close"]},
            "historical_pre_snapshot": {
                "race_categorical_logloss": ref["2022"]["historical_pre_snapshot_logloss"],
                "favorite_top1_rate": ref["2022"]["favorite_top1_pre"]},
            "pre_minus_close_gap_nats": ref["2022"]["pre_minus_close_gap_nats"],
            "R0_clean_table_only": "Stage 1 で rolling OOF (train<=2020 / ES 2021 / predict 2022) を"
                                    "作ってから同一 race set で再計算する。EXP15 の 2022 スコアは "
                                    "ES に 2022 を使っているため OOS ではなく、ここでは使わない",
            "note": "障害除外・DNF 除外後の正式 set 上の値。改訂前 (障害を含む) の値とは一致しない",
        },
        "dnf_sensitivity": pop["dnf_sensitivity"],
        "learned_subset_boundaries_by_eval_year": pop["learned_subset_boundaries_by_eval_year"],
        "fixed_domain_partitions": pop["fixed_domain_partitions"],
        "thresholds_fixed_before_power_audit": THRESHOLDS,
        "power_audit": {
            "tier1_ideal_local_approximation": pw["tier1_ideal_local_approximation"],
            "tier2_empirical_cluster_power": {
                "pass_at_floor": t2["pass_at_floor"],
                "power_curve_primary": t2["power_curve_primary"],
                "decision_rule": t2["decision_rule"],
            },
            "tier3_secondary_reference_check": pw["tier3_secondary_reference_check"],
            "directions": pw["directions"],
        },
        "growth_checks": {k: (v if not isinstance(v, dict) else
                             {k2: v2 for k2, v2 in v.items() if k2 != "by_target"})
                          for k, v in der.items() if k.startswith("C")},
        "market_provenance_summary": {
            "files": {k: v["sha256"][:16] for k, v in prov["files"].items()},
            "pre_snapshot_minutes_before_post_2022": prov["per_year"]["2022"]["pre_minutes_before_post"],
            "final_minutes_after_post_2022": prov["per_year"]["2022"]["final_minutes_after_post"],
        },
        "artifact_contract": artifact,
        "stage0_prohibitions_observed": ["production 変更なし", "候補馬券生成なし", "資金配分変更なし",
                                          "2024/2025 開封なし", "2019-2023 の結果評価なし",
                                          "ROI 本評価なし", "学習 (rolling OOF 作成) なし"],
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (HERE / "STAGE0_DRY_RUN.json").write_text(
        json.dumps(dry, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(dry["official_race_set"]["totals"], ensure_ascii=False))
    print(json.dumps(dry["power_audit"]["tier2_empirical_cluster_power"]["power_curve_primary"],
                     ensure_ascii=False))
    print(json.dumps(artifact["master_v2"], ensure_ascii=False))
    print(json.dumps(artifact["feature_schema"], ensure_ascii=False))
    print(json.dumps(artifact["p0_5_status"], ensure_ascii=False))
    print(f"[saved] STAGE0_DRY_RUN.json ({dry['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
