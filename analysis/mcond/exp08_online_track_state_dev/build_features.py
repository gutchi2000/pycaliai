# -*- coding: utf-8 -*-
"""
build_features.py — M0-M3/RAW/EWMA/ZERO比較用の馬単位特徴量テーブルを構築する。
=====================================================================================
設計(2026-09-20夜、ユーザーStage3指示):

M0: v6+市場(exp05_design.parquetのcalibrated_v6_probability[top3較正済み]と
    mkt_p3_pre[市場pre top3確率]を単純平均しlogit化した固定baseline_logit)。
M1: M0 + 利用可能な過去レースの生集計(RAW: 直近1レースの観測値)。
M2: M0 + Kalman状態平均・状態分散・観測数(次元ごと)。
M3: M1 + Kalman状態 + 対象馬との事前固定interaction。
    interaction = horse_aptitude(closing_power、master_v2の既存serve-safe特徴量、
    対象レースの実績を使わず対象日より前の履歴のみから構築済み) × Kalman状態平均。
    履歴不足馬(closing_power欠損)は0(母集団平均に近い値)へ縮約する、行を除外しない。
RAW: 直近1レースの観測値のみ(状態情報なし)。
EWMA: 事前固定した半減期(2時間)の指数移動平均のみ。
ZERO: 状態情報なし(0固定)。

対象は各次元(speed/agari/pace)の状態。内外/脚質はnegative control専用のため
主要比較には含めない(permutation placebo用に別途同じ機構を通す)。

baseline_logitはoffsetとして扱う(固定、fitしない)。追加する状態由来の特徴量
(RAW/EWMA/Kalman平均・分散・観測数/interaction)だけを正則化ロジスティック回帰で
fitする。

実行: 単体実行しない。evaluate.py から import して使う。
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
HERE = Path(__file__).resolve().parent

from analysis.mcond.exp08_online_track_state_dev.online_state import (  # noqa: E402
    STATE_DIMS, NEGATIVE_CONTROL_DIMS, ALL_SIGNAL_DIMS,
    attach_decision_timestamps, run_all_units, run_all_units_generic,
    run_unit_timeline_raw, run_unit_timeline_ewma, EWMA_HALFLIFE_HOURS,
)

EPS = 1e-6


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def load_race_level_state_features(
    obs: pd.DataFrame, avail_col: str, q: dict, r: dict, prior_var: dict,
) -> pd.DataFrame:
    """observations(decision_timestamp/avail_col付与済み)から、指定avail_col
    (primaryまたはsensitivity)についてKalman/RAW/EWMAの状態スナップショットを
    全レース分計算し、rid16単位で結合して返す(すべて対象レース自身の結果は
    一切使わない、判断時刻直前の値のみ)。"""
    kalman = run_all_units(obs, avail_col, q, r, prior_var)
    raw = run_all_units_generic(obs, avail_col, run_unit_timeline_raw)
    ewma = run_all_units_generic(
        obs, avail_col, run_unit_timeline_ewma, halflife_hours=EWMA_HALFLIFE_HOURS)

    merged = kalman.merge(raw, on="rid16", how="outer").merge(ewma, on="rid16", how="outer")
    return merged


def load_v6_market_baseline(years: list[int]) -> pd.DataFrame:
    """exp05_design.parquetからv6+市場のtop3較正済み確率・実績ラベルを読む
    (2016-2025を収録、年で絞り込む)。EXP07で確立済みのソース、この呼び出しは
    読み込みのみで新規学習は一切しない。"""
    cols = ["rid16", "ban", "year", "v6_pwin", "v6_p3", "mkt_pi_pre", "mkt_p3_pre",
            "fin", "top3", "win", "calibrated_v6_probability", "calibrated_v6_probability_win"]
    df = pd.read_parquet(BASE / "data/_research/mcond/exp05_design.parquet", columns=cols)
    df = df[df["year"].isin(years)].copy()
    p_blend = (df["calibrated_v6_probability"] + df["mkt_p3_pre"]) / 2.0
    df["baseline_logit_top3"] = _logit(p_blend)
    p_blend_win = (df["calibrated_v6_probability_win"] + df["mkt_pi_pre"]) / 2.0
    df["baseline_logit_win"] = _logit(p_blend_win)
    return df


def load_horse_aptitude() -> pd.DataFrame:
    """馬ごとのinteraction用aptitude(closing_power、master_v2の既存serve-safe特徴量、
    対象レースの実績を使わず対象日より前の履歴のみで構築済み)。欠損(履歴不足馬)は
    0(母集団中央値相当)へ縮約、行は落とさない。"""
    m = pd.read_csv(
        BASE / "data/master_v2_20130105-20251228.csv", encoding="utf-8-sig", low_memory=False,
        usecols=["レースID(新/馬番無)", "馬番", "closing_power"],
    )
    m["rid16"] = pd.to_numeric(m["レースID(新/馬番無)"], errors="coerce").astype("Int64").astype(str)
    m["ban"] = pd.to_numeric(m["馬番"], errors="coerce")
    m["horse_aptitude"] = m["closing_power"].fillna(0.0)
    return m[["rid16", "ban", "horse_aptitude"]].drop_duplicates(["rid16", "ban"])


def build_feature_table(years: list[int], q: dict, r: dict, prior_var: dict) -> dict:
    """primary/sensitivity両方のavailability仮定について、馬単位の特徴量テーブルを
    構築して返す({"primary": df, "sensitivity": df})。dfにはM0のbaseline_logit、
    RAW/EWMA/Kalman(平均・分散・観測数)、horse_aptitude、interaction列、outcome
    (top3/win)、rid16/ban/date/venue/surfaceが含まれる。"""
    obs = pd.read_parquet(HERE / "out" / "observations.parquet")
    obs = obs[obs["date"].str[:4].astype(int).isin(years)].copy()
    obs = attach_decision_timestamps(obs)
    obs = obs.dropna(subset=["actual_post_datetime", "decision_timestamp"])

    baseline = load_v6_market_baseline(years)
    aptitude = load_horse_aptitude()

    out = {}
    for label, avail_col in [
        ("primary", "prior_result_available_ts_primary"),
        ("sensitivity", "prior_result_available_ts_sensitivity"),
    ]:
        state = load_race_level_state_features(obs, avail_col, q, r, prior_var)
        race_meta = obs[["rid16", "date", "venue", "surface"] + ALL_SIGNAL_DIMS]
        race_feat = race_meta.merge(state, on="rid16", how="left")

        tbl = baseline.merge(race_feat, on="rid16", how="inner")
        tbl = tbl.merge(aptitude, on=["rid16", "ban"], how="left")
        tbl["horse_aptitude"] = tbl["horse_aptitude"].fillna(0.0)

        for d in STATE_DIMS:
            tbl[f"{d}_pre_mean"] = tbl[f"{d}_pre_mean"].fillna(0.0)
            tbl[f"{d}_pre_var"] = tbl[f"{d}_pre_var"].fillna(prior_var[d])
            tbl[f"{d}_pre_n_obs"] = tbl[f"{d}_pre_n_obs"].fillna(0)
            tbl[f"{d}_raw"] = tbl[f"{d}_raw"].fillna(0.0)
            tbl[f"{d}_ewma"] = tbl[f"{d}_ewma"].fillna(0.0)
            tbl[f"interaction_{d}"] = tbl["horse_aptitude"] * tbl[f"{d}_pre_mean"]

        out[label] = tbl
    return out
