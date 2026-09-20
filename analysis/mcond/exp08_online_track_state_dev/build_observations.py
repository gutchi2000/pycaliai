# -*- coding: utf-8 -*-
"""
build_observations.py — レース単位の speed_signal / agari_signal / pace_signal 観測を
構築する。
=====================================================================================
スコープ(README.md参照、2026-09-20夜ユーザー確定): 主対象は時計(speed_signal)・
上がり(agari_signal)・ペース(pace_signal)の3つを**独立した状態**として扱う
(2026-09-20夜Stage3指示により、上がりを時計・ペースと別次元の独立状態として
明示的に追加)。内外(inside_signal)・前残り(front_signal)はnegative control専用
として同時に構築するが、online_state.pyでの採用候補には含めない。

【観測信号定義(2026-09-20夜、ユーザー指摘によりStage3着手前に確定・凍結。
2024・2025年の結果を見る前にこの定義へ固定する)】

- **集約対象**: そのレースの公式着順が付いた馬全員(中止・取消・除外・失格は
  fin非数値として`expected_time_model.load_kekka_for_time_model`が既に除外済み)。
  同着(デッドヒート)は着順が同値の複数行として扱われ、特別処理はしない
  (集約統計量への寄与は他の馬と同じ1票)。
- **集約統計量**: 中央値(median)。単純平均は少数の大穴・出遅れ・単一馬の
  タイム異常に弱いため、レース単位の期待値ルックアップ自体が中央値である
  ことと整合させ、レース内集約も中央値にする。
- **距離・競馬場・芝ダ・コース区分・公表馬場状態・クラスの補正**:
  `expected_time_model.py`の段階的フォールバック付き中央値ルックアップ
  (full→no_going→no_class→coarse→global)。競馬場を条件付けキーに含めるのは、
  競馬場固有の恒常的な時計の速さ/遅さ(トラック形状・標高等)を「日次で
  変動する状態」と混同しないため。
- **時系列安全性**: 年Yの観測はYより前のデータだけでfitしたexpanding-window
  モデルから作る(2013-2022年は年ごとに拡大窓、2023年以降は2022年末までの
  単一モデルに固定・再fitしない)。詳細は`expected_time_model.py`docstring。
- **信号の符号**:
  speed_signal(race) = -(そのレースの (actual_time - expected_time) の中央値)
    正 = 期待より速いタイム = 馬場が速い方向に振れている。
  agari_signal(race) = -(そのレースの (actual_agari3f - expected_agari3f) の中央値)
    正 = 期待より上がり3Fが速い = 上がり性能が効きやすい方向に振れている。
    speed_signalと同じexpanding-window期待値ルックアップ機構を上り3F列に適用する
    (距離・競馬場・芝ダ・コース区分・馬場状態・クラスで条件付け)。
  pace_signal(race) = -(そのレースの RPCI - 期待RPCI)
    RPCIは値が小さいほど前傾(ハイペース)、大きいほど後傾(スローペース)。
    正 = 期待よりペースが緩んだ(上がり勝負寄り)。RPCIはレース単位で
    既に1値(全馬共通)のため中央値集約は不要、そのまま残差化する。
- **外れ値のclip範囲**: train(年<2023)のみで決めたIQR×3の境界
  (`q1-3*iqr` 〜 `q3+3*iqr`)。2023-2025を見て変えない。
- **芝/ダート・競馬場は別状態**: online_state.py側でz[開催日,競馬場,芝ダ,時刻]と
  して個別に管理する(この時点では観測をレース単位で構築するのみ)。

negative control(内外/前残り、Gate評価専用、採用しない):
  inside_signal(race) = 内馬番(馬番<=3)複勝率 − 外馬番(馬番>=頭数-2)複勝率。
  front_signal(race) = 3角通過順位が上位1/3の馬の複勝率 − 下位1/3の馬の複勝率。
  （馬番は1〜頭数の個別番号。枠番(1-8固定)は使わない、理由は下記バグ4参照）

時系列リーク境界: この関数はレース**単体**の観測値を計算するだけで、日内の
順序付け・時点フィルタリングはonline_state.py側の責務(このファイルはリーク
境界を持たない、素材だけを作る)。ただしexpected_time_model側のexpanding-window
は年単位の粗い安全境界であり、日内の判断時刻ちょうどの境界はonline_state.pyが
別途保証する。

実行: PYTHONUTF8=1 ./venv311/Scripts/python.exe -m analysis.mcond.exp08_online_track_state_dev.build_observations
出力: analysis/mcond/exp08_online_track_state_dev/out/observations.parquet
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
HERE = Path(__file__).resolve().parent
from analysis.mcond.exp08_online_track_state_dev.expected_time_model import (  # noqa: E402
    load_kekka_for_time_model, fit_expanding_lookups, predict_expected_time_expanding,
)


def load_raw() -> pd.DataFrame:
    """走破タイム・RPCI用の列に加え、negative control用の列も読む。"""
    cols = [
        "yyyymmdd", "race_id", "レース名", "場所", "馬番", "着順", "走破タイム", "距離",
        "芝・ダ", "コース区分", "馬場状態", "天気", "クラス名", "枠番", "3角", "頭数",
        "出走頭数", "上り3F", "RPCI", "PCI",
    ]
    df = load_kekka_for_time_model(usecols=cols)
    return df


# 【2026-09-20夜、ユーザー指摘により訂正】
# 旧版は「TANPUK確定(区分4)オッズのタイムスタンプ」を「競走結果(着順・走破タイム・
# 上がり・通過順)が利用可能になった時刻」の代理として使い、これを2023年の実測と
# 称していた。これは誤り: TANPUK確定はオッズ・払戻が確定した時刻であり、着順等の
# 結果レコード自体がいつ取得・公開されたかを直接示すものではない
# (オッズ確定には結果確定が前提だが、両者の時刻が一致する保証はない)。
#
# JV-Link・TARGET・保存済みデータのいずれにも、競走結果レコード自体の取得/公開/
# 更新時刻を示す信頼できるタイムスタンプは存在しないことを別途調査で確認した
# (JV-LinkのRACE(RA)レコードにある「データ作成年月日」はレース**前**の番組表発表日で
# 別物、日付のみで時刻情報なし)。したがって「実測」ではなく、保守的な仮定として
# 固定する: 主解析は発走時刻+20分、時点安全の感度分析は発走時刻+30分
# (両者で効果の方向が一致しない場合はFAILとする、spec.json参照)。
PRIMARY_DELAY_MIN = 20
SENSITIVITY_DELAY_MIN = 30


def _attach_availability_timestamps(race: pd.DataFrame) -> pd.DataFrame:
    """master_v2の発走時刻(実測値)と結合し、各レースの
    actual_post_datetime / prior_result_available_ts_primary(+20分) /
    prior_result_available_ts_sensitivity(+30分) を付与する。
    これらは「このレースの結果が後続レースの状態更新に使って良い最早時刻」を表す
    (このレース自身が後続レースにとっての"先行レース"になる場合の可用時刻)。"""
    m = pd.read_csv(
        BASE / "data/master_v2_20130105-20251228.csv", encoding="utf-8-sig",
        low_memory=False, usecols=["レースID(新/馬番無)", "発走時刻"],
    )
    m["rid16"] = pd.to_numeric(m["レースID(新/馬番無)"], errors="coerce").astype("Int64").astype(str)
    m = m.drop_duplicates("rid16")

    race = race.copy()
    race["rid16"] = race["rid16"].astype(str)
    race = race.merge(m[["rid16", "発走時刻"]], on="rid16", how="left")
    post_str = race["date"] + " " + race["発走時刻"].astype(str).str.replace(":", "", regex=False).str.zfill(4)
    race["actual_post_datetime"] = pd.to_datetime(post_str, format="%Y%m%d %H%M", errors="coerce")
    race["prior_result_available_ts_primary"] = race["actual_post_datetime"] + pd.Timedelta(
        minutes=PRIMARY_DELAY_MIN)
    race["prior_result_available_ts_sensitivity"] = race["actual_post_datetime"] + pd.Timedelta(
        minutes=SENSITIVITY_DELAY_MIN)
    n_missing_post = race["actual_post_datetime"].isna().sum()
    if n_missing_post:
        print(f"[build_observations] WARNING: 発走時刻が結合できなかったレース {n_missing_post}件"
              f"(online_state.py側でこれらは先行レース候補から自動除外されるべき)")
    return race.drop(columns=["発走時刻"])


def build_race_level_observations() -> pd.DataFrame:
    df = load_raw()

    # --- 時計: expanding-window(年単位leave-year-out、2023+は2022年末で固定) ---
    time_lookups = fit_expanding_lookups(df, value_col="time_sec")
    exp_time, time_fb = predict_expected_time_expanding(time_lookups, df, value_col="time_sec")
    df["time_resid"] = df["time_sec"] - exp_time

    # --- 上がり(上り3F): 同じexpanding-window機構を上り3F列へ適用 ---
    agari_df = df.copy()
    agari_df["time_sec"] = pd.to_numeric(agari_df["上り3F"], errors="coerce")
    agari_lookups = fit_expanding_lookups(agari_df.dropna(subset=["time_sec"]), value_col="time_sec")
    exp_agari, agari_fb = predict_expected_time_expanding(agari_lookups, agari_df, value_col="time_sec")
    df["agari_resid"] = pd.to_numeric(df["上り3F"], errors="coerce") - exp_agari

    # --- ペース(RPCI): 同じexpanding-window機構をRPCI列へ適用 ---
    rpci_df = df.copy()
    rpci_df["time_sec"] = rpci_df["RPCI"]  # fit_expected_time_lookupの値列名を使い回す
    rpci_lookups = fit_expanding_lookups(rpci_df.dropna(subset=["time_sec"]), value_col="time_sec")
    exp_rpci, rpci_fb = predict_expected_time_expanding(rpci_lookups, rpci_df, value_col="time_sec")
    df["rpci_resid"] = df["RPCI"] - exp_rpci

    # --- negative control素材(内外・脚質) ---
    # 枠番(1-8固定、複数馬が同枠を共有)ではなく馬番(1〜頭数、個別)を使う
    # (2026-09-20実装中に発覚: 枠番で内外percentileを取ると大頭数レースで
    # is_outerが枠番>=14等の非現実的閾値になりほぼ常にFalse、内外信号が
    # 全レースNaNになる実害バグだった)。
    uma = pd.to_numeric(df["馬番"], errors="coerce")
    n_horse = pd.to_numeric(df["頭数"], errors="coerce")
    df["is_inner"] = uma <= 3
    df["is_outer"] = uma >= (n_horse - 2).clip(lower=4)
    df["top3"] = df["fin"] <= 3
    corner3 = pd.to_numeric(df["3角"], errors="coerce")
    df["corner3_rel"] = (corner3 - 1) / (n_horse - 1).replace(0, np.nan)
    df["is_front3"] = df["corner3_rel"] <= (1 / 3)
    df["is_back3"] = df["corner3_rel"] >= (2 / 3)

    g = df.groupby("rid16")
    race = g.agg(
        date=("date", "first"), venue=("場所", "first"),
        surface=("芝・ダ", "first"), dist_bucket=("dist_bucket", "first"),
        going=("馬場状態", "first"), n_field=("頭数", "first"),
        time_resid_median=("time_resid", "median"),
        agari_resid_median=("agari_resid", "median"),
        rpci_resid=("rpci_resid", "first"),
        n_time_obs=("time_resid", "count"),
        n_agari_obs=("agari_resid", "count"),
    ).reset_index()

    def rate_diff(mask_a_col, mask_b_col):
        a = df[df[mask_a_col] == True].groupby("rid16")["top3"].mean()  # noqa: E712
        b = df[df[mask_b_col] == True].groupby("rid16")["top3"].mean()  # noqa: E712
        return (a - b)

    inside_diff = rate_diff("is_inner", "is_outer").rename("inside_signal_raw")
    front_diff = rate_diff("is_front3", "is_back3").rename("front_signal_raw")
    race = race.merge(inside_diff, on="rid16", how="left")
    race = race.merge(front_diff, on="rid16", how="left")

    race["speed_signal"] = -race["time_resid_median"]
    race["agari_signal"] = -race["agari_resid_median"]
    race["pace_signal"] = -race["rpci_resid"]
    race["inside_signal"] = race["inside_signal_raw"]  # negative control、符号調整なし
    race["front_signal"] = race["front_signal_raw"]    # negative control、符号調整なし

    # 極端値のwinsorize(train=年<2023のみでIQR境界を決め、2023-2025を見て変えない)。
    # 実データ確認: 希少長距離条件でlookupセルのサンプルが薄く、単一レースの
    # 残差が大きく暴れる実例を発見(データ破損ではなく希少条件の統計的な分散)。
    # IQRベースの頑健な境界で抑える([[project_exp07]]で確立したσ推定と同じ規律:
    # 素のstd/minmaxではなくIQRを使う)。
    train_mask = race["date"].str[:4].astype(int) < 2023
    clip_bounds = {}
    for col in ["speed_signal", "agari_signal", "pace_signal"]:
        q1, q3 = race.loc[train_mask, col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
        clip_bounds[col] = (float(lo), float(hi))
        race[col] = race[col].clip(lo, hi)

    race = _attach_availability_timestamps(race)
    meta = {
        "time_fallback_dist": time_fb.value_counts(normalize=True).to_dict(),
        "agari_fallback_dist": agari_fb.value_counts(normalize=True).to_dict(),
        "rpci_fallback_dist": rpci_fb.value_counts(normalize=True).to_dict(),
        "n_races": race["rid16"].nunique(),
        "clip_bounds": clip_bounds,
    }
    return race, meta


def main():
    race, meta = build_race_level_observations()
    out_dir = HERE / "out"
    out_dir.mkdir(exist_ok=True)
    race.to_parquet(out_dir / "observations.parquet", index=False)
    print(f"[build_observations] n_races={meta['n_races']}")
    print(f"[build_observations] time fallback dist: {meta['time_fallback_dist']}")
    print(f"[build_observations] agari fallback dist: {meta['agari_fallback_dist']}")
    print(f"[build_observations] rpci fallback dist: {meta['rpci_fallback_dist']}")
    print(f"[build_observations] clip bounds: {meta['clip_bounds']}")
    print(f"[build_observations] wrote {out_dir / 'observations.parquet'}")
    print(race[["speed_signal", "agari_signal", "pace_signal",
               "inside_signal", "front_signal"]].describe())


if __name__ == "__main__":
    main()
