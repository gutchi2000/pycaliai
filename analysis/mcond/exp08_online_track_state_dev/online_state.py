# -*- coding: utf-8 -*-
"""
online_state.py — Kalman状態空間モデル本体(時計・上がり・ペースの3独立状態)。
=====================================================================================
状態単位: 開催日×競馬場×芝ダート(rid16の日付・場所・surfaceで一意に決まるunit)。
各unit内で、時計(speed)・上がり(agari)・ペース(pace)をそれぞれ独立なスカラー
Kalman状態として持つ(内外・脚質はnegative control専用、ここでは状態化しない、
build_observations.pyのinside_signal/front_signalをpermutation placebo比較のみに使う)。

設計(2026-09-20夜、ユーザーStage3指示):
  - 状態更新は「利用可能時刻を通過した過去レースのみ」使う。primary(+20分)/
    sensitivity(+30分)を混合せず完全に別系列として生成する(run_all_units()を
    availability列を変えて2回呼ぶ)。
  - 同日最初の状態は平均0・分散は事前固定値(prior_var、2023developmentのみで選択)。
  - 日をまたいで状態を繰り越さない(unit = 開催日単位で独立、そもそもunit自体が
    日を含むキーなので日をまたぐ処理はしない設計)。
  - 欠損観測(speed_signal等がNaN)では平均を更新せず、不確実性(分散の時間減衰)
    だけ適切に扱う: 欠損レースは「観測イベント」を発生させず、時間経過による
    分散増加は次の実イベント(観測 or 判断時刻の問い合わせ)発生時にまとめて
    適用される(連続時間の時間減衰なので、間に欠損があっても数学的に等価)。

イベント駆動アルゴリズム: 各unit内のレースを時系列に並べ、
  (a) 観測利用可能イベント(有効な観測を持つレースのprior_result_available_ts)
  (b) 判断(スナップショット)イベント(全レースのdecision_timestamp、状態を読むだけ
      で更新しない)
を時刻順にマージして処理する。これにより、あるレースの判断時刻までに実際に
「利用可能になっている」先行レースだけが反映された状態を、レース番号ではなく
実時刻で正しく再現する。
"""
from __future__ import annotations
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
HERE = Path(__file__).resolve().parent

STATE_DIMS = ["speed_signal", "agari_signal", "pace_signal"]
NEGATIVE_CONTROL_DIMS = ["inside_signal", "front_signal"]
ALL_SIGNAL_DIMS = STATE_DIMS + NEGATIVE_CONTROL_DIMS


# =============================================================================
# 判断時刻(historical_pre_snapshot、EXP07で確立、TANPUK区分1最終スナップショット)
# =============================================================================

def load_decision_timestamps(years: list[int]) -> pd.DataFrame:
    """rid16 -> decision_timestamp(TANPUK区分1最終スナップショット時刻)。
    EXP07のhistorical_pre_snapshot(発走26-30分前・中央値28分前)と同じ規約を再利用。"""
    t = pd.read_csv(
        BASE / "data/Time _series_odds/TANPUK_20210105-20251228.csv",
        encoding="cp932", low_memory=False,
    )
    t.columns = [c.strip() for c in t.columns]
    rid_col, kbn_col, tcol = t.columns[0], t.columns[1], t.columns[2]
    t["rid16"] = t[rid_col].astype(str).str.replace(r"\D", "", regex=True).str[:16]
    t["_year"] = t["rid16"].str[:4].astype(int)
    t = t[t["_year"].isin(years) & (t[kbn_col] == 1)]
    t = t.sort_values(tcol).groupby("rid16").last().reset_index()
    t["mmddhhmm"] = pd.to_numeric(t[tcol], errors="coerce")
    return t[["rid16", "mmddhhmm"]]


def attach_decision_timestamps(obs: pd.DataFrame) -> pd.DataFrame:
    """observations.parquetへdecision_timestamp列を付与する(TANPUKが無い年
    (2013-2020、TANPUKは2021年以降のみ収録)は欠損のままとなり、対象レース
    としては評価から自動除外される想定)。"""
    years = sorted(obs["date"].str[:4].astype(int).unique().tolist())
    ts = load_decision_timestamps(years)
    obs = obs.merge(ts, on="rid16", how="left")
    date_str = obs["date"]
    mmddhhmm_str = obs["mmddhhmm"].astype("Int64").astype(str).str.zfill(8)
    year_str = date_str.str[:4]
    combined = year_str + mmddhhmm_str.str[0:4] + " " + mmddhhmm_str.str[4:8]
    obs["decision_timestamp"] = pd.to_datetime(combined, format="%Y%m%d %H%M", errors="coerce")
    return obs.drop(columns=["mmddhhmm"])


# =============================================================================
# Kalman状態(1次元)
# =============================================================================

@dataclass
class ScalarKalmanState:
    mean: float
    var: float
    n_obs: int = 0
    last_update_ts: pd.Timestamp | None = None

    def predict(self, now_ts: pd.Timestamp, q_per_hour: float) -> "ScalarKalmanState":
        """時間減衰のみ適用(平均は不変、分散だけ経過時間に応じて増える)。
        last_update_tsが無ければ(まだ何も起きていない初期状態)、そのまま返す
        (unitの最初のイベント時に呼ばれると経過時間ゼロとして扱われる)。"""
        if self.last_update_ts is None:
            return replace(self, last_update_ts=now_ts)
        dt_hours = (now_ts - self.last_update_ts).total_seconds() / 3600.0
        dt_hours = max(dt_hours, 0.0)
        return replace(self, var=self.var + q_per_hour * dt_hours, last_update_ts=now_ts)

    def update(self, obs_value: float, r: float) -> "ScalarKalmanState":
        """観測1件でKalman更新(平均・分散とも更新、n_obsを+1)。"""
        k = self.var / (self.var + r)
        new_mean = self.mean + k * (obs_value - self.mean)
        new_var = (1 - k) * self.var
        return replace(self, mean=new_mean, var=new_var, n_obs=self.n_obs + 1)


def init_state(prior_var: float) -> ScalarKalmanState:
    return ScalarKalmanState(mean=0.0, var=prior_var, n_obs=0, last_update_ts=None)


# =============================================================================
# unit単位のイベント駆動タイムライン処理
# =============================================================================

def _unit_key(row: pd.Series) -> tuple:
    return (row["date"], row["venue"], row["surface"])


def run_unit_timeline(
    unit_races: pd.DataFrame, avail_col: str, q: dict, r: dict, prior_var: dict,
) -> pd.DataFrame:
    """1つの(開催日,競馬場,芝ダ)unit内のレース群に対し、各レースの**判断時刻直前**の
    状態(mean/var/n_obs、次元ごと)を計算して返す。unit_racesは同一unit内の行のみ
    (呼び出し側でgroupby済み)、decision_timestamp/avail_colが両方揃っている行のみ
    使う(NaTの行はイベントを起こさない=先行レースとして使われず、対象レースとしても
    スナップショットされない)。

    戻り値: rid16ごとに1行、{dim}_pre_mean / {dim}_pre_var / {dim}_pre_n_obs 列。
    """
    states = {d: init_state(prior_var[d]) for d in STATE_DIMS}

    events = []
    for _, row in unit_races.iterrows():
        avail_ts = row.get(avail_col)
        if pd.notna(avail_ts) and any(pd.notna(row.get(d)) for d in STATE_DIMS):
            events.append((avail_ts, "obs", row["rid16"]))
        dec_ts = row.get("decision_timestamp")
        if pd.notna(dec_ts):
            events.append((dec_ts, "decision", row["rid16"]))
    events.sort(key=lambda e: (e[0], 0 if e[1] == "obs" else 1))
    # 同時刻ならobs(観測)をdecision(判断)より先に処理する(「同じ時間帯に複数の
    # 結果が利用可能になる場合は時刻順」+観測が判断に先行できるようにする安全側の順序)

    rows_by_rid = unit_races.set_index("rid16")
    snapshots: dict[str, dict] = {}

    for ts, kind, rid in events:
        for d in STATE_DIMS:
            states[d] = states[d].predict(ts, q[d])
        if kind == "obs":
            row = rows_by_rid.loc[rid]
            for d in STATE_DIMS:
                val = row.get(d)
                if pd.notna(val):
                    states[d] = states[d].update(float(val), r[d])
        else:  # decision: 読むだけ、更新しない
            snap = {}
            for d in STATE_DIMS:
                snap[f"{d}_pre_mean"] = states[d].mean
                snap[f"{d}_pre_var"] = states[d].var
                snap[f"{d}_pre_n_obs"] = states[d].n_obs
            snapshots[rid] = snap

    if not snapshots:
        return pd.DataFrame(columns=["rid16"] + [
            f"{d}_pre_{s}" for d in STATE_DIMS for s in ("mean", "var", "n_obs")])
    out = pd.DataFrame.from_dict(snapshots, orient="index")
    out.index.name = "rid16"
    return out.reset_index()


def run_all_units(
    obs: pd.DataFrame, avail_col: str, q: dict, r: dict, prior_var: dict,
) -> pd.DataFrame:
    """observations(decision_timestamp付与済み)全体をunitごとに分割してrun_unit_timeline
    を適用し、結合する。"""
    results = []
    for _, unit_races in obs.groupby(["date", "venue", "surface"], sort=False):
        snap = run_unit_timeline(unit_races, avail_col, q, r, prior_var)
        if len(snap):
            results.append(snap)
    if not results:
        return pd.DataFrame(columns=["rid16"])
    return pd.concat(results, ignore_index=True)


# =============================================================================
# Q/R/prior_var のグリッド選択(2023developmentのみ、one-step-ahead尤度)
# =============================================================================

def one_step_ahead_loglik(
    obs: pd.DataFrame, avail_col: str, dim: str, q: float, r: float, prior_var: float,
) -> float:
    """指定(q,r,prior_var)で、指定次元1つだけについてunit単位のタイムラインを流し、
    各観測イベント直前の予測分布N(mean, var+r)の下での対数尤度を積算する
    (Q/R選択専用、decisionイベントは無視して良い=状態のみ更新)。"""
    total_ll = 0.0
    n = 0
    for _, unit_races in obs.groupby(["date", "venue", "surface"], sort=False):
        sub = unit_races[unit_races[dim].notna() & unit_races[avail_col].notna()]
        if len(sub) == 0:
            continue
        sub = sub.sort_values(avail_col)
        state = init_state(prior_var)
        for _, row in sub.iterrows():
            state = state.predict(row[avail_col], q)
            y = float(row[dim])
            pred_var = state.var + r
            ll = -0.5 * np.log(2 * np.pi * pred_var) - 0.5 * (y - state.mean) ** 2 / pred_var
            total_ll += ll
            n += 1
            state = state.update(y, r)
    return total_ll / max(n, 1)


# =============================================================================
# 単純対照(RAW/EWMA/ZERO): §9「状態空間モデル固有の価値」を判定する比較対象
# =============================================================================

EWMA_HALFLIFE_HOURS = 2.0  # 事前固定(2023developmentの結果を見て変えない、docstring参照)


def run_unit_timeline_raw(unit_races: pd.DataFrame, avail_col: str) -> pd.DataFrame:
    """RAW対照: 直近1レースの観測値をそのまま状態として使う(フィルタリングなし)。
    日初期・欠損時の扱いはKalman版と揃える(同日最初はNone/0相当、欠損観測は状態を
    動かさない)。"""
    events = []
    for _, row in unit_races.iterrows():
        avail_ts = row.get(avail_col)
        if pd.notna(avail_ts) and any(pd.notna(row.get(d)) for d in STATE_DIMS):
            events.append((avail_ts, "obs", row["rid16"]))
        dec_ts = row.get("decision_timestamp")
        if pd.notna(dec_ts):
            events.append((dec_ts, "decision", row["rid16"]))
    events.sort(key=lambda e: (e[0], 0 if e[1] == "obs" else 1))

    rows_by_rid = unit_races.set_index("rid16")
    last_val = {d: 0.0 for d in STATE_DIMS}
    has_obs = {d: False for d in STATE_DIMS}
    n_obs = {d: 0 for d in STATE_DIMS}
    snapshots: dict[str, dict] = {}

    for ts, kind, rid in events:
        if kind == "obs":
            row = rows_by_rid.loc[rid]
            for d in STATE_DIMS:
                val = row.get(d)
                if pd.notna(val):
                    last_val[d] = float(val)
                    has_obs[d] = True
                    n_obs[d] += 1
        else:
            snap = {}
            for d in STATE_DIMS:
                snap[f"{d}_raw"] = last_val[d] if has_obs[d] else 0.0
                snap[f"{d}_raw_n_obs"] = n_obs[d]
            snapshots[rid] = snap
    if not snapshots:
        return pd.DataFrame(columns=["rid16"] + [f"{d}_raw" for d in STATE_DIMS])
    out = pd.DataFrame.from_dict(snapshots, orient="index")
    out.index.name = "rid16"
    return out.reset_index()


def run_unit_timeline_ewma(
    unit_races: pd.DataFrame, avail_col: str, halflife_hours: float = EWMA_HALFLIFE_HOURS,
) -> pd.DataFrame:
    """EWMA対照: 時間減衰付き指数移動平均。同日最初は0、欠損観測は動かさない。
    減衰率は事前固定(EWMA_HALFLIFE_HOURS)、2023developmentの結果を見て変えない。"""
    events = []
    for _, row in unit_races.iterrows():
        avail_ts = row.get(avail_col)
        if pd.notna(avail_ts) and any(pd.notna(row.get(d)) for d in STATE_DIMS):
            events.append((avail_ts, "obs", row["rid16"]))
        dec_ts = row.get("decision_timestamp")
        if pd.notna(dec_ts):
            events.append((dec_ts, "decision", row["rid16"]))
    events.sort(key=lambda e: (e[0], 0 if e[1] == "obs" else 1))

    rows_by_rid = unit_races.set_index("rid16")
    ewma = {d: 0.0 for d in STATE_DIMS}
    n_obs = {d: 0 for d in STATE_DIMS}
    last_ts = {d: None for d in STATE_DIMS}
    snapshots: dict[str, dict] = {}
    ln2 = np.log(2.0)

    for ts, kind, rid in events:
        if kind == "obs":
            row = rows_by_rid.loc[rid]
            for d in STATE_DIMS:
                val = row.get(d)
                if pd.notna(val):
                    if last_ts[d] is None:
                        ewma[d] = float(val)
                    else:
                        dt_hours = max((ts - last_ts[d]).total_seconds() / 3600.0, 0.0)
                        decay = np.exp(-ln2 * dt_hours / halflife_hours)
                        ewma[d] = decay * ewma[d] + (1 - decay) * float(val)
                    last_ts[d] = ts
                    n_obs[d] += 1
        else:
            snap = {}
            for d in STATE_DIMS:
                snap[f"{d}_ewma"] = ewma[d]
                snap[f"{d}_ewma_n_obs"] = n_obs[d]
            snapshots[rid] = snap
    if not snapshots:
        return pd.DataFrame(columns=["rid16"] + [f"{d}_ewma" for d in STATE_DIMS])
    out = pd.DataFrame.from_dict(snapshots, orient="index")
    out.index.name = "rid16"
    return out.reset_index()


def run_all_units_generic(obs: pd.DataFrame, avail_col: str, fn, **kwargs) -> pd.DataFrame:
    results = []
    for _, unit_races in obs.groupby(["date", "venue", "surface"], sort=False):
        snap = fn(unit_races, avail_col, **kwargs)
        if len(snap):
            results.append(snap)
    if not results:
        return pd.DataFrame(columns=["rid16"])
    return pd.concat(results, ignore_index=True)


def select_qr_grid(obs_2023: pd.DataFrame, avail_col: str) -> dict:
    """次元ごとに小さな事前固定格子でQ/R/prior_varをone-step-ahead尤度最大化で選ぶ。
    格子は各次元のtrain(年<2023)スケール(spec.jsonのclip_bounds等から導出される
    frozenな分散)に対する比率で定義する(2023developmentの結果を見て格子自体を
    広げない、格子外再探索はしない)。"""
    # 各次元の frozen スケール(年<2023のwinsorize後標準偏差、build_observations.py
    # 実行時にすでに固定されている値を再計算するのではなく、ここでも同じ年<2023の
    # データから独立に求める、2024-2025は使わない)
    train = obs_2023  # 呼び出し側で年=2023のみに絞って渡す想定(2023 development)
    selected = {}
    for dim in STATE_DIMS:
        scale_var = float(train[dim].var(skipna=True))
        if not np.isfinite(scale_var) or scale_var <= 0:
            scale_var = 1.0
        q_grid = [scale_var * f for f in (0.001, 0.005, 0.02, 0.08, 0.3)]
        r_grid = [scale_var * f for f in (0.3, 0.6, 1.0, 1.6, 2.5)]
        prior_var_grid = [scale_var * f for f in (0.5, 1.0, 2.0)]
        best = None
        for pv in prior_var_grid:
            for q in q_grid:
                for r in r_grid:
                    ll = one_step_ahead_loglik(train, avail_col, dim, q, r, pv)
                    if best is None or ll > best[0]:
                        best = (ll, q, r, pv)
        selected[dim] = {"q": best[1], "r": best[2], "prior_var": best[3], "loglik": best[0]}
    return selected
