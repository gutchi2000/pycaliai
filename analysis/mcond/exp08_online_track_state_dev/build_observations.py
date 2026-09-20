# -*- coding: utf-8 -*-
"""
build_observations.py — レース単位の speed_signal / pace_signal 観測を構築する。
=====================================================================================
スコープ(README.md参照、2026-09-20夜ユーザー確定): 主対象は時計(speed_signal)・
上がり/ペース(pace_signal)のみ。内外(inside_signal)・前残り(front_signal)は
negative control専用として同時に構築するが、online_state.pyでの採用候補には含めない。

speed_signal(race) = -(そのレースの全公式着順馬の (actual_time - expected_time) 平均)
  符号反転により「正 = 期待より速いタイム = 馬場が速い方向に振れている」に統一。
  expected_timeはexpected_time_model.py(train<=2022のみでfit、距離50m単位×芝ダ×
  コース区分×公表馬場状態×クラス名の中央値ルックアップ、段階的フォールバック付き)。

pace_signal(race) = -(そのレースのRPCI - 期待RPCI)
  RPCIは前半-後半ペースの指数で、値が小さいほど前傾(ハイペース、前が止まりやすい)、
  大きいほど後傾(スローペース、上がり勝負)。ここでは「期待より数値が大きい
  (=上がり勝負寄りにペースが緩んだ)」を正の値とする(符号の解釈はonline_state.py
  側では使わず単なる観測値として扱うため、実質的な向きの取り決めはRPCI残差の生値でも良いが、
  「正=有利/不利」の統一感のためspeed_signalと合わせてマイナスを掛けている
  ―― RPCI自体は距離が長い/短いで系統的に水準が違うため、時計と全く同じ
  ルックアップ手法(距離×芝ダ×コース区分×公表馬場状態×クラス名の中央値)で
  期待RPCIを作り残差化する)。

negative control(内外/前残り、Gate評価専用、採用しない):
  inside_signal(race) = 内枠(枠番<=3)複勝率 - 外枠(枠番>=6相当)複勝率 の
    レース内粗指標(v6残差化なし、day_state_counting.py Stage1と同型の簡易指標。
    厳密なnegative controlとして「本当に死んでいる次元がGate2Aで再度死ぬか」
    確認する目的のみに使う)。
  front_signal(race) = 前3角通過順位が上位1/3の馬の複勝率 - 下位1/3の馬の複勝率。

時系列リーク境界: この関数はレース**単体**の観測値を計算するだけで、日内の
順序付け・時点フィルタリングはonline_state.py側の責務(このファイルはリーク
境界を持たない、素材だけを作る)。

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
    load_kekka_for_time_model, fit_expected_time_lookup, predict_expected_time,
)

VENUE_FROM_RID16 = None  # rid16[8:10]から場コードを取る(EXP07と同じ規約)


def load_raw() -> pd.DataFrame:
    """走破タイム・RPCI用の列に加え、negative control用の列も読む。"""
    cols = [
        "yyyymmdd", "race_id16", "レース名", "馬番", "着順", "走破タイム", "距離", "芝・ダ",
        "コース区分", "馬場状態", "天気", "クラス名", "枠番", "3角", "頭数", "出走頭数",
        "上り3F", "RPCI", "PCI",
    ]
    df = load_kekka_for_time_model(usecols=cols)
    return df


def fit_expected_rpci_lookup(train_df: pd.DataFrame):
    """RPCIは時計と同じ条件付けキーで中央値ルックアップを作る(距離×芝ダ×
    コース区分×公表馬場状態×クラス名、train<=2022のみ)。時計モデルと同じ
    段階的フォールバック機構を再利用するため、一時的にtime_sec列をRPCIへ
    差し替えたコピーで既存関数を呼ぶ(実装の重複を避ける)。"""
    tmp = train_df.copy()
    tmp["time_sec"] = tmp["RPCI"]
    tmp = tmp.dropna(subset=["time_sec"])
    return fit_expected_time_lookup(tmp)


def build_race_level_observations() -> pd.DataFrame:
    df = load_raw()
    df["year"] = df["date"].str[:4].astype(int)
    train = df[df["year"] <= 2022]

    time_lookup = fit_expected_time_lookup(train)
    rpci_lookup = fit_expected_rpci_lookup(train)

    exp_time, time_fb = predict_expected_time(time_lookup, df)
    df["time_resid"] = df["time_sec"] - exp_time

    rpci_df = df.copy()
    rpci_df["time_sec"] = rpci_df["RPCI"]
    exp_rpci, rpci_fb = predict_expected_time(rpci_lookup, rpci_df)
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

    g = df.groupby("race_id16")
    race = g.agg(
        date=("date", "first"), venue=("race_id16", lambda s: str(s.iloc[0])[8:10]),
        surface=("芝・ダ", "first"), dist_bucket=("dist_bucket", "first"),
        going=("馬場状態", "first"), n_field=("頭数", "first"),
        time_resid_mean=("time_resid", "mean"), rpci_resid=("rpci_resid", "first"),
        n_time_obs=("time_resid", "count"),
    ).reset_index()

    def rate_diff(mask_a_col, mask_b_col):
        a = df[df[mask_a_col] == True].groupby("race_id16")["top3"].mean()  # noqa: E712
        b = df[df[mask_b_col] == True].groupby("race_id16")["top3"].mean()  # noqa: E712
        return (a - b)

    inside_diff = rate_diff("is_inner", "is_outer").rename("inside_signal_raw")
    front_diff = rate_diff("is_front3", "is_back3").rename("front_signal_raw")
    race = race.merge(inside_diff, on="race_id16", how="left")
    race = race.merge(front_diff, on="race_id16", how="left")

    race["speed_signal"] = -race["time_resid_mean"]
    race["pace_signal"] = -race["rpci_resid"]
    race["inside_signal"] = race["inside_signal_raw"]  # negative control、符号調整なし
    race["front_signal"] = race["front_signal_raw"]    # negative control、符号調整なし

    # 極端値のwinsorize(train<=2022のIQRのみで境界を決め、2023-2025を見て変えない)。
    # 実データ確認: 3300m/3900m等の希少長距離条件でlookupセルのサンプルが薄く、
    # 単一レースの残差が±100秒級に暴れる実例を発見(データ破損ではなく、希少条件の
    # 統計的な分散の大きさそのもの)。IQRベースの頑健な境界で抑える([[project_exp07]]
    # で確立したσ推定と同じ規律: 素のstd/minmaxではなくIQRを使う)。
    train_mask = race["date"].str[:4].astype(int) <= 2022
    for col in ["speed_signal", "pace_signal"]:
        q1, q3 = race.loc[train_mask, col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
        race[col] = race[col].clip(lo, hi)

    race["post_datetime"] = race["date"]  # 発走時刻の突合はonline_state.py側でmaster_v2と結合
    return race, {"time_fallback_dist": time_fb.value_counts(normalize=True).to_dict(),
                  "rpci_fallback_dist": rpci_fb.value_counts(normalize=True).to_dict(),
                  "n_races": race["race_id16"].nunique()}


def main():
    race, meta = build_race_level_observations()
    out_dir = HERE / "out"
    out_dir.mkdir(exist_ok=True)
    race.to_parquet(out_dir / "observations.parquet", index=False)
    print(f"[build_observations] n_races={meta['n_races']}")
    print(f"[build_observations] time fallback dist: {meta['time_fallback_dist']}")
    print(f"[build_observations] rpci fallback dist: {meta['rpci_fallback_dist']}")
    print(f"[build_observations] wrote {out_dir / 'observations.parquet'}")
    print(race[["speed_signal", "pace_signal", "inside_signal", "front_signal"]].describe())


if __name__ == "__main__":
    main()
