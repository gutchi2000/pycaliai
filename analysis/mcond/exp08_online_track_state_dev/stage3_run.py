# -*- coding: utf-8 -*-
"""
stage3_run.py — EXP08 Stage3 Gate評価ドライバ。

処理順序(結果情報の混入を防ぐ):
  1. 2023年developmentでQ/R/prior_var選択(one-step-ahead尤度、既にコミット前に
     実施済みの値をspec.jsonから読む)
  2. 2023年developmentでM0-M3/RAW/EWMA/ZEROをfit(beta凍結)
  3. 2024年・2025年でgenuinely OOS評価(betaを再fitしない)
  4. meeting-day paired bootstrap(M1-M0, M3-M1)
  5. permutation placebo test(>=1000回、競馬場×日×芝ダート内シャッフル)
  6. Gate判定

実行: PYTHONUTF8=1 ./venv311/Scripts/python.exe -m analysis.mcond.exp08_online_track_state_dev.stage3_run
出力: analysis/mcond/exp08_online_track_state_dev/out/STAGE3_GATE_REPORT.json
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
HERE = Path(__file__).resolve().parent

from analysis.mcond.exp08_online_track_state_dev.online_state import (  # noqa: E402
    STATE_DIMS, NEGATIVE_CONTROL_DIMS, attach_decision_timestamps,
    run_all_units, run_all_units_generic, run_unit_timeline_raw, run_unit_timeline_ewma,
    EWMA_HALFLIFE_HOURS,
)
from analysis.mcond.exp08_online_track_state_dev.build_features import (  # noqa: E402
    build_feature_table, load_v6_market_baseline, load_horse_aptitude,
)
from analysis.mcond.exp08_online_track_state_dev.evaluate import (  # noqa: E402
    MODEL_FEATURES, fit_and_eval_model, paired_bootstrap_delta_logloss,
    shuffle_observations_within_unit, logloss,
)

N_PLACEBO = 1000
PLACEBO_SEED_BASE = 20260920

# 2023 developmentで選択済み(one-step-ahead尤度、primary/sensitivityで同一、
# 理由: 固定遅延の一様シフトは観測間の相対間隔を変えないため。README.md参照)
Q = {"speed_signal": 0.0014934032768609586, "agari_signal": 0.003588564669598636,
     "pace_signal": 0.010535367137079392}
R = {"speed_signal": 0.896041966116575, "agari_signal": 0.43062776035183625,
     "pace_signal": 10.535367137079392}
PRIOR_VAR = {"speed_signal": 0.7467016384304792, "agari_signal": 0.35885646695986356,
             "pace_signal": 5.267683568539696}


def compute_state_features_from_obs(obs: pd.DataFrame, avail_col: str) -> pd.DataFrame:
    kalman = run_all_units(obs, avail_col, Q, R, PRIOR_VAR)
    raw = run_all_units_generic(obs, avail_col, run_unit_timeline_raw)
    ewma = run_all_units_generic(
        obs, avail_col, run_unit_timeline_ewma, halflife_hours=EWMA_HALFLIFE_HOURS)
    return kalman.merge(raw, on="rid16", how="outer").merge(ewma, on="rid16", how="outer")


def build_full_table(obs: pd.DataFrame, avail_col: str, baseline: pd.DataFrame,
                     aptitude: pd.DataFrame) -> pd.DataFrame:
    state = compute_state_features_from_obs(obs, avail_col)
    race_meta = obs[["rid16", "date", "venue", "surface"] + STATE_DIMS + NEGATIVE_CONTROL_DIMS]
    race_feat = race_meta.merge(state, on="rid16", how="left")
    tbl = baseline.merge(race_feat, on="rid16", how="inner")
    tbl = tbl.merge(aptitude, on=["rid16", "ban"], how="left")
    tbl["horse_aptitude"] = tbl["horse_aptitude"].fillna(0.0)
    for d in STATE_DIMS:
        tbl[f"{d}_pre_mean"] = tbl[f"{d}_pre_mean"].fillna(0.0)
        tbl[f"{d}_pre_var"] = tbl[f"{d}_pre_var"].fillna(PRIOR_VAR[d])
        tbl[f"{d}_pre_n_obs"] = tbl[f"{d}_pre_n_obs"].fillna(0)
        tbl[f"{d}_raw"] = tbl[f"{d}_raw"].fillna(0.0)
        tbl[f"{d}_ewma"] = tbl[f"{d}_ewma"].fillna(0.0)
        tbl[f"interaction_{d}"] = tbl["horse_aptitude"] * tbl[f"{d}_pre_mean"]
    return tbl


def load_year_obs(years: list[int]) -> pd.DataFrame:
    obs = pd.read_parquet(HERE / "out" / "observations.parquet")
    obs = obs[obs["date"].str[:4].astype(int).isin(years)].copy()
    obs = attach_decision_timestamps(obs)
    obs = obs.dropna(subset=["actual_post_datetime", "decision_timestamp"])
    return obs


def main():
    t_start = time.time()
    print("[stage3] loading 2023 development, fitting M0-M3/RAW/EWMA/ZERO (beta frozen)...")
    obs_2023 = load_year_obs([2023])
    baseline_2023 = load_v6_market_baseline([2023])
    aptitude = load_horse_aptitude()

    train_primary = build_full_table(obs_2023, "prior_result_available_ts_primary",
                                      baseline_2023, aptitude)

    print("[stage3] loading 2024/2025 (結果ラベルを初めて評価に使う段階)...")
    obs_eval = load_year_obs([2024, 2025])
    baseline_eval = load_v6_market_baseline([2024, 2025])

    eval_tables = {}
    for label, avail_col in [("primary", "prior_result_available_ts_primary"),
                             ("sensitivity", "prior_result_available_ts_sensitivity")]:
        tbl = build_full_table(obs_eval, avail_col, baseline_eval, aptitude)
        tbl["year"] = tbl["date"].str[:4].astype(int)
        eval_tables[label] = tbl

    results = {"n_train_2023": len(train_primary), "availability": {}}

    for label in ["primary", "sensitivity"]:
        eval_df = eval_tables[label]
        eval_2024 = eval_df[eval_df["year"] == 2024]
        eval_2025 = eval_df[eval_df["year"] == 2025]
        eval_all = eval_df

        model_results = {}
        for model_name in ["M0", "RAW", "EWMA", "ZERO", "M1", "M2", "M3"]:
            res = fit_and_eval_model(
                model_name, train_primary,
                {"2024": eval_2024, "2025": eval_2025, "all": eval_all}, l2=1.0,
            )
            model_results[model_name] = {
                "beta": res["beta"], "features": res["features"],
                "logloss_2024": res["2024"]["logloss"], "logloss_2025": res["2025"]["logloss"],
                "logloss_all": res["all"]["logloss"],
            }
        results["availability"][label] = {"models": model_results, "_full": None}
        # 予測値・実測値・rid16をbootstrap用に保存(このavailabilityラベルのみ)
        results["availability"][label]["_full"] = {
            m: fit_and_eval_model(m, train_primary, {"2024": eval_2024, "2025": eval_2025,
                                                       "all": eval_all}, l2=1.0)
            for m in ["M0", "M1", "M3"]
        }

    out_dir = HERE / "out"
    out_dir.mkdir(exist_ok=True)
    # bootstrap(M1 vs M0, M3 vs M1)をprimary/sensitivityそれぞれで計算
    bootstrap_results = {}
    for label in ["primary", "sensitivity"]:
        full = results["availability"][label]["_full"]
        boot_for_label = {}
        for comp_name, (a, b) in [("M1_vs_M0", ("M1", "M0")), ("M3_vs_M1", ("M3", "M1"))]:
            boot_for_label[comp_name] = {}
            for period in ["2024", "2025", "all"]:
                pa = full[a][period]["p"]
                pb = full[b][period]["p"]
                y = full[a][period]["y"]
                rid = full[a][period]["rid16"]
                boot_for_label[comp_name][period] = paired_bootstrap_delta_logloss(
                    y, pa, pb, rid, n_boot=2000)
        bootstrap_results[label] = boot_for_label

    results["bootstrap"] = bootstrap_results
    for label in results["availability"]:
        del results["availability"][label]["_full"]

    elapsed_pre_placebo = time.time() - t_start
    print(f"[stage3] Gate統計(bootstrap込み)完了 {elapsed_pre_placebo:.0f}s、placebo({N_PLACEBO}回)開始...")

    # --- permutation placebo(primaryのみ、M1-M0とM3-M1、対象は時計・上がり・ペース) ---
    # 2023でfit済みの凍結betaを再利用する(shuffleされたデータで毎回refitしない、
    # 「2023developmentでfitしたモデルが、無意味にシャッフルされた同日情報に対して
    # 見かけ上の改善を示すか」を検定するのが目的のため)。
    from analysis.mcond.exp08_online_track_state_dev.evaluate import predict_offset_logistic
    frozen_beta = {m: np.array(results["availability"]["primary"]["models"][m]["beta"])
                  for m in ["M0", "M1", "M3"]}

    def _predict_with_frozen(model_name, tbl):
        feats = MODEL_FEATURES[model_name]
        X = tbl[feats].to_numpy(dtype=float) if feats else np.zeros((len(tbl), 0))
        off = tbl["baseline_logit_top3"].to_numpy(dtype=float)
        return predict_offset_logistic(X, off, frozen_beta[model_name])

    placebo_real = {}
    placebo_dist = {"M1_vs_M0": [], "M3_vs_M1": []}
    for comp_name in ["M1_vs_M0", "M3_vs_M1"]:
        placebo_real[comp_name] = bootstrap_results["primary"][comp_name]["all"]["point"]

    t0 = time.time()
    for i in range(N_PLACEBO):
        obs_shuffled = shuffle_observations_within_unit(obs_eval, seed=PLACEBO_SEED_BASE + i)
        tbl_shuf = build_full_table(
            obs_shuffled, "prior_result_available_ts_primary", baseline_eval, aptitude)
        y = tbl_shuf["top3"].to_numpy(dtype=float)
        p0 = _predict_with_frozen("M0", tbl_shuf)
        p1 = _predict_with_frozen("M1", tbl_shuf)
        p3 = _predict_with_frozen("M3", tbl_shuf)
        placebo_dist["M1_vs_M0"].append(logloss(y, p1) - logloss(y, p0))
        placebo_dist["M3_vs_M1"].append(logloss(y, p3) - logloss(y, p1))
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"[stage3] placebo {i+1}/{N_PLACEBO} ({elapsed:.0f}s, "
                 f"~{elapsed/(i+1)*N_PLACEBO:.0f}s total est.)")

    placebo_summary = {}
    for comp_name in ["M1_vs_M0", "M3_vs_M1"]:
        dist = np.array(placebo_dist[comp_name])
        real = placebo_real[comp_name]
        pct_97_5 = float(np.percentile(dist, 97.5))
        # 改善=負なので「realがplacebo分布の下側(より改善側)97.5%点を下回るか」で判定
        pct_2_5_as_improvement_bound = float(np.percentile(dist, 2.5))
        passed = bool(real < pct_2_5_as_improvement_bound)
        placebo_summary[comp_name] = {
            "real_delta": real, "placebo_mean": float(dist.mean()), "placebo_std": float(dist.std()),
            "placebo_2_5pct": pct_2_5_as_improvement_bound, "placebo_97_5pct": pct_97_5,
            "n_placebo": len(dist), "passed": passed,
        }

    results["permutation_placebo"] = placebo_summary
    results["elapsed_seconds"] = time.time() - t_start

    out_path = out_dir / "STAGE3_GATE_REPORT.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str),
                        encoding="utf-8")
    print(f"[stage3] wrote {out_path}")
    print(f"[stage3] total elapsed: {results['elapsed_seconds']:.0f}s")


if __name__ == "__main__":
    main()
