# -*- coding: utf-8 -*-
"""
features.py — EXP19 Stage 0: W (当日馬体重状態)・P (公式物理馬場)・WP (事前固定 interaction)
=============================================================================================
すべて race day-start までの過去値と、当日発表済みの馬体重・公式馬場値だけで作る (結果列を読まない)。

W (SPEC §4.1、主 block)
  bw_log_kg            log(current kg)。measured のみ
  bw_sex_age_z         性別×年齢×四半期の基準 (対象年より前の年の measured だけ) による z
  bw_robust_z5         直近最大 5 走の measured 体重の median / MAD (×1.4826、MAD floor) による z。履歴 < 2 は欠損。winsorize
  bw_abs_robust_z5     上の絶対値
  bw_change_x_layoff   (current − previous measured) / previous × log1p(休養日数)。previous が無ければ欠損
  bw_status            歴史で識別可能な one-hot: measured / not_measured (歴史 torch は 999 と原票欠損を区別できない)
  bw_history_n         使えた過去 measured 体重数 (0..5)
  説明用 (主 model に入れない): bw_change_kg / bw_change_pct / bw_dev_med5_pct
P (SPEC §4.2、race 内定数)
  cushion_z_place_asof / moist_gp_z / moist_4c_z (place×surface) / moist_gradient_z (GP−4C を同様に標準化)
  標準化は対象日の**前日まで**の venue-day だけの expanding window。履歴 venue-day < P_MIN_HIST は欠損
WP (SPEC §4.3、6 本固定)
  #1 bw_robust_z5×cushion_z (芝) / #2 |z5|×|cushion_z| (芝) / #3 z5×moist_gp_z (surface 別) /
  #4 |z5|×|moist_gp_z| (surface 別) / #5 z5×moist_gradient_z / #6 bw_change_x_layoff×track_extreme_z

欠損の扱い (Stage 0 固定): 欠損値は「0kg・増減 0」とみなさない。設計行列では欠損指示子を必ず併置し、値の位置には
定数 FILL を置く。指示子の係数が定数を吸収するので、FILL の値は fit した尤度に影響しない (test_invariants で検査)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAD_SCALE = 1.4826
MIN_HIST = 2                 # SPEC: 履歴 2 走未満は missing
MAX_HIST = 5
MEASURED_LO, MEASURED_HI = 200.0, 800.0
SEX_AGE_MIN_N = 30           # 性別×年齢×四半期の基準セルの最小件数 (未満は欠損)
P_MIN_HIST = 10              # 馬場の as-of 標準化に必要な過去 venue-day 数
FILL = 0.0
# Stage 0 で分布だけを見て固定した値 (fix_w_params.py → out/w_param_fixing.json、2013-2018 の measured・履歴>=2 行、
# 結果列不使用)。MAD floor = max(2 kg 記録刻み, 生 MAD の p10=0)、winsorize = ceil(|z| の p99.5=7.42 / 0.5)*0.5
MAD_FLOOR_KG = 2.0
WINSOR_Z = 7.5

W_MAIN = ["bw_log_kg", "bw_sex_age_z", "bw_robust_z5", "bw_abs_robust_z5", "bw_change_x_layoff",
          "bw_status_not_measured", "bw_history_n"]
# bw_log_kg の欠損 = bw_status_not_measured と同一なので指示子は持たない (重複列を作らない)
W_MISS = ["miss_bw_sex_age_z", "miss_bw_robust_z5", "miss_bw_change_x_layoff"]
W_DESCRIPTIVE = ["bw_change_kg", "bw_change_pct", "bw_dev_med5_pct"]
WP_COLS = ["wp1_z5_x_cushion_turf", "wp2_absz5_x_abscushion_turf", "wp3_z5_x_moistgp_turf",
           "wp3_z5_x_moistgp_dirt", "wp4_absz5_x_absmoistgp_turf", "wp4_absz5_x_absmoistgp_dirt",
           "wp5_z5_x_moistgrad", "wp6_chglay_x_trackextreme"]


def set_params(mad_floor_kg: float, winsor_z: float):
    global MAD_FLOOR_KG, WINSOR_Z
    MAD_FLOOR_KG, WINSOR_Z = float(mad_floor_kg), float(winsor_z)


# ---------------------------------------------------------------- W
def _roll_mad(x: np.ndarray) -> float:
    m = np.median(x)
    return float(np.median(np.abs(x - m)))


def history_table(t: pd.DataFrame) -> pd.DataFrame:
    """馬ごとの measured 系列の包含 rolling 統計 (直近 5 measured、自身を含む)。
    各行は後で『対象日より前の最後の measured 行』へ as-of 結合するので、自身は対象 race に入らない"""
    m = t[t["measured"]].sort_values(["pid", "date", "rid16"])[["pid", "date", "kg"]].copy()
    g = m.groupby("pid", sort=False)["kg"]
    m["h_med5"] = g.transform(lambda s: s.rolling(MAX_HIST, min_periods=1).median())
    m["h_mad5"] = g.transform(lambda s: s.rolling(MAX_HIST, min_periods=1).apply(_roll_mad, raw=True))
    m["h_n"] = g.transform(lambda s: s.rolling(MAX_HIST, min_periods=1).count())
    m = m.rename(columns={"date": "h_date", "kg": "h_kg"})
    return m


def asof_history(t: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    """各行に『同じ馬の、対象日より前の最後の measured 行』の包含統計を付ける (同日は含めない)"""
    left = t[["pid", "date"]].reset_index().rename(columns={"index": "_row"}).sort_values("date")
    right = hist.sort_values("h_date")
    j = pd.merge_asof(left, right, left_on="date", right_on="h_date", by="pid",
                      allow_exact_matches=False, direction="backward")
    return j.set_index("_row").sort_index()


def sex_age_baseline(t: pd.DataFrame) -> dict:
    """年 Y ごとに、Y より前の年の measured 行だけで (性別, 年齢, 四半期) の平均・SD を作る"""
    m = t[t["measured"]]
    out = {}
    for Y in sorted(t["year"].unique()):
        past = m[m["year"] < Y]
        if not len(past):
            continue
        g = past.groupby(["性別", "年齢", "quarter"])["kg"].agg(["mean", "std", "count"])
        out[int(Y)] = g[g["count"] >= SEX_AGE_MIN_N]
    return out


def starter_rows(t: pd.DataFrame, term_starters: dict) -> pd.DataFrame:
    """体重履歴の slot は starter 行だけ (取消・除外 = 市場の terminal starter でない行を落とす)。
    DNF 馬は starter なので残る。市場データの無い race の行は判定できないので残す (件数は population で報告)"""
    keep = np.array([(b in term_starters[r]) if r in term_starters else True for r, b in zip(t["rid16"], t["ban"])])
    return t[keep].copy()


def build_w(t: pd.DataFrame) -> pd.DataFrame:
    """torch 構造 (loaders.load_torch_struct の出力) → W 特徴。行順は入力と同じ"""
    assert MAD_FLOOR_KG is not None and WINSOR_Z is not None, "MAD floor / winsorize を先に固定すること"
    orig_index = t.index
    # 行順に依存しないよう正準順 (馬・日付・race・馬番) で計算し、最後に入力の行順へ戻す (浮動小数の集計順も固定)
    t = t.sort_values(["pid", "date", "rid16", "ban"], kind="mergesort").copy()
    t["measured"] = t["kg"].between(MEASURED_LO, MEASURED_HI)
    t["quarter"] = ((t["date"] // 100 % 100) - 1) // 3 + 1
    hist = history_table(t)
    h = asof_history(t, hist).reindex(t.index)          # 位置演算 (np.select) でもずれないよう t の行順へ揃える
    # 直前の出走 (measured か否かを問わない) の日付 → 休養日数
    s = t.sort_values(["pid", "date", "rid16"])
    prev_run = s.groupby("pid", sort=False)["date"].shift(1)
    t["prev_run_date"] = prev_run.reindex(t.index)
    d_cur = pd.to_datetime(t["date"].astype(str), format="%Y%m%d")
    d_prev = pd.to_datetime(t["prev_run_date"].dropna().astype("int64").astype(str), format="%Y%m%d").reindex(t.index)
    t["layoff_days"] = (d_cur - d_prev).dt.days

    kg = t["kg"].where(t["measured"])
    n = h["h_n"].fillna(0).astype(int)
    med5, mad5, prev_kg = h["h_med5"], h["h_mad5"], h["h_kg"]
    denom = MAD_SCALE * np.maximum(mad5, MAD_FLOOR_KG)
    z5 = ((kg - med5) / denom).where(n >= MIN_HIST).clip(-WINSOR_Z, WINSOR_Z)
    w = pd.DataFrame(index=t.index)
    w["bw_log_kg"] = np.log(kg)
    base = sex_age_baseline(t)
    sa = pd.Series(np.nan, index=t.index)
    for Y, tab in base.items():
        rows = t.index[(t["year"] == Y) & t["measured"]]
        if not len(rows):
            continue
        key = pd.MultiIndex.from_frame(t.loc[rows, ["性別", "年齢", "quarter"]])
        mu = tab["mean"].reindex(key).to_numpy()
        sd = tab["std"].reindex(key).to_numpy()
        sa.loc[rows] = (t.loc[rows, "kg"].to_numpy() - mu) / sd
    w["bw_sex_age_z"] = sa.clip(-WINSOR_Z, WINSOR_Z)
    w["bw_robust_z5"] = z5
    w["bw_abs_robust_z5"] = z5.abs()
    chg = (kg - prev_kg) / prev_kg
    w["bw_change_x_layoff"] = chg * np.log1p(t["layoff_days"])
    w["bw_status_not_measured"] = (~t["measured"]).astype(float)
    w["bw_history_n"] = n.clip(upper=MAX_HIST).astype(float)
    # 説明用
    w["bw_change_kg"] = kg - prev_kg
    w["bw_change_pct"] = chg
    w["bw_dev_med5_pct"] = (kg - med5) / med5
    # 原票増減の検算 (fail-closed で記録するだけ。値は書き換えない)
    src = t["chg_src"]
    calc = kg - prev_kg
    w["chg_check"] = np.select(
        [~t["measured"], prev_kg.isna(), src.isna(), (src - calc).abs() < 1e-9],
        ["current_not_measured", "no_asof_previous", "source_missing", "match"], default="mismatch")
    w["prev_measured_date"] = h["h_date"]
    w["layoff_days"] = t["layoff_days"]
    return w.reindex(orig_index)


# ---------------------------------------------------------------- P
def build_p(baba: pd.DataFrame) -> pd.DataFrame:
    """venue-day ごとの公式物理値 → 対象日の前日までの expanding as-of z (place / place×surface)"""
    b = baba.rename(columns={"場所": "venue", "日付": "date"}).sort_values(["venue", "date"]).copy()
    b["shiba_grad"] = b["shiba_gp"] - b["shiba_4c"]
    b["dirt_grad"] = b["dirt_gp"] - b["dirt_4c"]
    out = b[["venue", "date"]].copy()
    for col in ["cushion", "shiba_gp", "shiba_4c", "dirt_gp", "dirt_4c", "shiba_grad", "dirt_grad"]:
        g = b.groupby("venue", sort=False)[col]
        # 前日まで: shift(1) した expanding (venue-day は 1 日 1 行)
        mu = g.transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
        sd = g.transform(lambda s: s.shift(1).expanding(min_periods=2).std())
        cnt = g.transform(lambda s: s.shift(1).expanding(min_periods=1).count())
        z = (b[col] - mu) / sd
        out[f"{col}_z"] = z.where((cnt >= P_MIN_HIST) & (sd > 0))
        out[f"{col}_raw"] = b[col]
    return out.reset_index(drop=True)


def race_p(races: pd.DataFrame, pz: pd.DataFrame) -> pd.DataFrame:
    """race (rid16, date, 場所, 芝・ダ) に P を付ける。芝は cushion + 芝含水率、ダートは ダ含水率"""
    r = races.merge(pz, left_on=["場所", "date"], right_on=["venue", "date"], how="left")
    turf = r["芝・ダ"].astype(str).str.startswith("芝")
    r["surface"] = np.where(turf, "turf", "dirt")
    r["cushion_z"] = r["cushion_z"].where(turf)
    r["moist_gp_z"] = np.where(turf, r["shiba_gp_z"], r["dirt_gp_z"])
    r["moist_4c_z"] = np.where(turf, r["shiba_4c_z"], r["dirt_4c_z"])
    r["moist_gradient_z"] = np.where(turf, r["shiba_grad_z"], r["dirt_grad_z"])
    # track_extreme_z (Stage 0 で結果を使わず一意に定義): 芝 = max(|cushion_z|, |moist_gp_z|)、ダ = |moist_gp_z|
    te_turf = np.fmax(np.abs(r["cushion_z"]), np.abs(r["moist_gp_z"]))
    te_turf = np.where(r["cushion_z"].isna() | pd.isna(r["moist_gp_z"]), np.nan, te_turf)
    r["track_extreme_z"] = np.where(turf, te_turf, np.abs(r["moist_gp_z"]))
    need = np.where(turf, r["cushion_z"].notna() & pd.notna(r["moist_gp_z"]) & pd.notna(r["moist_gradient_z"]),
                    pd.notna(r["moist_gp_z"]) & pd.notna(r["moist_gradient_z"]))
    r["p_available"] = need
    return r


def build_wp(w: pd.DataFrame, rp: pd.DataFrame) -> pd.DataFrame:
    """WP 6 本 (surface 別列を含め 8 列)。W の欠損は欠損指示子側で扱い、積の値は FILL (race 側 P 欠損の race は
    WP 不可 = p_available False で母集団から外す)"""
    z5 = w["bw_robust_z5"].fillna(FILL)
    az5 = w["bw_abs_robust_z5"].fillna(FILL)
    cl = w["bw_change_x_layoff"].fillna(FILL)
    turf = (rp["surface"] == "turf").to_numpy()
    cz = rp["cushion_z"].to_numpy(dtype=float)
    mg = rp["moist_gp_z"].to_numpy(dtype=float)
    gr = rp["moist_gradient_z"].to_numpy(dtype=float)
    te = rp["track_extreme_z"].to_numpy(dtype=float)
    o = pd.DataFrame(index=w.index)
    o["wp1_z5_x_cushion_turf"] = np.where(turf, z5 * np.nan_to_num(cz), 0.0)
    o["wp2_absz5_x_abscushion_turf"] = np.where(turf, az5 * np.abs(np.nan_to_num(cz)), 0.0)
    o["wp3_z5_x_moistgp_turf"] = np.where(turf, z5 * np.nan_to_num(mg), 0.0)
    o["wp3_z5_x_moistgp_dirt"] = np.where(~turf, z5 * np.nan_to_num(mg), 0.0)
    o["wp4_absz5_x_absmoistgp_turf"] = np.where(turf, az5 * np.abs(np.nan_to_num(mg)), 0.0)
    o["wp4_absz5_x_absmoistgp_dirt"] = np.where(~turf, az5 * np.abs(np.nan_to_num(mg)), 0.0)
    o["wp5_z5_x_moistgrad"] = z5 * np.nan_to_num(gr)
    o["wp6_chglay_x_trackextreme"] = cl * np.nan_to_num(te)
    return o


def design_w(w: pd.DataFrame) -> pd.DataFrame:
    """W の設計行列 (主 7 項 + 欠損指示子)。欠損値の位置は FILL"""
    d = pd.DataFrame(index=w.index)
    for c in W_MAIN:
        d[c] = w[c].fillna(FILL)
    for c in ["bw_sex_age_z", "bw_robust_z5", "bw_change_x_layoff"]:
        d[f"miss_{c}"] = w[c].isna().astype(float)
    # bw_abs_robust_z5 の欠損は bw_robust_z5 と同一なので指示子を共有する
    return d
