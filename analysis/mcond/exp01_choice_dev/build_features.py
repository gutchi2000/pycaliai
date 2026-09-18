# -*- coding: utf-8 -*-
"""
build_features.py — 陣営選択の生特徴・逸脱特徴 (時点安全)
==========================================================
時点安全の約束 (各行について):
  - 対象レース自身・同日の他レース・将来レースの情報を一切使わない
      調教師/騎手/組み合わせの統計は「対象日より前の日」だけで集計 (history_end < target_date)
  - 前走は同一 血統登録番号 の直前行。master の 前走日付 と一致した行だけを有効とする
      (障害・地方を挟むと直前行が真の前走でないため)
  - 対象レースの結果 (着順・払戻・オッズ) は使わない。
    前走の着順・着差は対象レース前に確定した公知情報なので「通常行動」の文脈にのみ使う。
  - 馬主列は時点整合性が未確認のため使わない (audit.py で検査のみ)

選択の次元 (6):
  interval   前走からの日数               5区分: ≤14 / 15-35 / 36-63 / 64-139 / ≥140
  dist       距離変更                     3区分: ≤-200 / ±200未満 / ≥+200
  venue      競馬場変更                   2区分
  surface    芝ダ変更                     2区分
  cls        クラス移動 (序数差の符号)     3区分: 降級 / 同 / 昇級
  jockey     騎手継続                     2区分

出力列:
  生の選択 (M2): raw_log_int, raw_dist_chg, raw_venue_chg, raw_surface_chg, raw_cls_chg,
                 raw_jockey_same, raw_jq_delta, raw_jt_pair
  単純統計の逸脱 (8.1): ss_int_z, ss_int_horse_z, ss_dist_z, ss_venue_s, ss_surface_s,
                 ss_cls_s, ss_jockey_s, ss_total
  行動予測の逸脱 (8.2): bp_interval_s, bp_dist_s, bp_venue_s, bp_surface_s, bp_cls_s,
                 bp_jockey_s, bp_total   (build_bp.py が付与)
  文脈 (行動予測の入力): ctx_*
出力: data/_research/mcond/exp01_features.parquet
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from grade_feats import class_name_to_ord  # noqa: E402

MASTER = BASE / "data/master_v2_20130105-20251228.csv"
OUT = BASE / "data/_research/mcond/exp01_features.parquet"
K_TRAINER = 30.0   # 調教師統計の縮約の強さ (事前固定)
K_JOCKEY = 50.0
K_HORSE = 3.0
EPS = 1e-4

COLS = ["日付", "発走時刻", "レースID(新)", "血統登録番号", "馬番", "距離", "場所", "芝・ダ",
        "クラス名", "騎手コード", "調教師コード", "性別", "年齢", "前走日付", "前走着差タイム",
        "着順", "馬主(最新/仮想)"]

INT_BINS = [-np.inf, 14, 35, 63, 139, np.inf]
# カテゴリは固定の対応表で数値化する (全期間から語彙を作ると未来の情報が混ざり、削除不変性も崩れる)
SEX_MAP = {"牡": 0, "牝": 1, "セ": 2}
PLACE_MAP = {"札幌": 0, "函館": 1, "福島": 2, "新潟": 3, "東京": 4, "中山": 5,
             "中京": 6, "京都": 7, "阪神": 8, "小倉": 9}


def load_master(path=MASTER, max_date: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False, usecols=COLS)
    df["date"] = pd.to_datetime(df["日付"].astype(str), format="%Y%m%d")
    if max_date is not None:
        df = df[df["date"] <= pd.Timestamp(max_date)]
    df["rid16"] = df["レースID(新)"].astype(str).str[:16]
    df["ban"] = pd.to_numeric(df["馬番"], errors="coerce")
    df["fin"] = pd.to_numeric(df["着順"], errors="coerce")
    df = df.dropna(subset=["ban", "血統登録番号"])
    df = df[df["fin"].notna() & (df["fin"] >= 1)]          # 取消・除外は出走していない
    df["ban"] = df["ban"].astype(int)
    df["hid"] = df["血統登録番号"].astype(str)
    df["cls_ord"] = df["クラス名"].map(class_name_to_ord)
    df["dist"] = pd.to_numeric(df["距離"], errors="coerce")
    df["trainer"] = df["調教師コード"].astype(str)
    df["jockey"] = df["騎手コード"].astype(str)
    df["top3"] = (df["fin"] <= 3).astype(int)                # 過去日の集計 (騎手の格) にのみ使う
    return df.sort_values(["hid", "date", "発走時刻"]).reset_index(drop=True)


def asof_by_key(df: pd.DataFrame, key: str, cols: list[str]) -> pd.DataFrame:
    """key ごとに、対象日より前の日の cols 合計 (同日は含めない)。"""
    daily = df.groupby([key, "date"])[cols].sum().sort_index()
    prior = daily.groupby(level=0).cumsum() - daily
    return prior.add_prefix("prior_").reset_index()


def asof_global(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    daily = df.groupby("date")[cols].sum().sort_index()
    prior = daily.cumsum() - daily
    return prior.add_prefix("gprior_").reset_index()


def build(df: pd.DataFrame) -> pd.DataFrame:
    """df: load_master の出力。時点安全な特徴を返す (行順は df と同じ)。"""
    df = df.copy()
    g = df.groupby("hid", sort=False)
    for c in ["date", "dist", "場所", "芝・ダ", "cls_ord", "jockey", "trainer", "fin", "前走着差タイム"]:
        df[f"prev_{c}"] = g[c].shift(1)
    df["prev2_date"] = g["date"].shift(2)
    df["n_prev_runs"] = g.cumcount()

    # 前走連結の検証: master の 前走日付 (YYMMDD) と直前行の日付が一致するか
    zd = pd.to_numeric(df["前走日付"], errors="coerce")
    zd_ts = pd.to_datetime((zd + 20000000).astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    df["chain_ok"] = (df["prev_date"].notna() & zd_ts.notna() & (df["prev_date"] == zd_ts))

    # ---- 生の選択 ----
    df["int_days"] = (df["date"] - df["prev_date"]).dt.days
    df["raw_log_int"] = np.log(df["int_days"].clip(lower=1))
    df["raw_dist_chg"] = df["dist"] - df["prev_dist"]
    df["raw_venue_chg"] = (df["場所"] != df["prev_場所"]).astype(float)
    df["raw_surface_chg"] = (df["芝・ダ"] != df["prev_芝・ダ"]).astype(float)
    d = df["cls_ord"] - df["prev_cls_ord"]
    df["raw_cls_chg"] = np.sign(d).fillna(0.0)
    df["raw_jockey_same"] = (df["jockey"] == df["prev_jockey"]).astype(float)

    # 行動の区分 (単純統計・行動予測で共通)
    df["a_interval"] = pd.cut(df["int_days"], INT_BINS, labels=False)
    df["a_dist"] = np.select([df["raw_dist_chg"] <= -200, df["raw_dist_chg"] >= 200], [0, 2], 1)
    df["a_venue"] = df["raw_venue_chg"].astype(int)
    df["a_surface"] = df["raw_surface_chg"].astype(int)
    df["a_cls"] = (df["raw_cls_chg"] + 1).astype(int)
    df["a_jockey"] = df["raw_jockey_same"].astype(int)

    valid = df["chain_ok"]
    v = df[valid].copy()

    # ---- 騎手の格 (対象日より前の日の3着内率, 縮約) ----
    df["one"] = 1.0
    jq = asof_by_key(df, "jockey", ["top3", "one"])
    gq = asof_global(df, ["top3", "one"])
    jq = jq.merge(gq, on="date")
    jq["jq"] = (jq["prior_top3"] + K_JOCKEY * jq["gprior_top3"] / jq["gprior_one"].clip(lower=1)) / \
               (jq["prior_one"] + K_JOCKEY)
    jqd = jq.set_index(["jockey", "date"])["jq"]
    v["jq_cur"] = jqd.reindex(pd.MultiIndex.from_arrays([v["jockey"], v["date"]])).to_numpy()
    v["jq_prev"] = jqd.reindex(pd.MultiIndex.from_arrays([v["prev_jockey"], v["date"]])).to_numpy()
    gmean = (gq.set_index("date")["gprior_top3"] / gq.set_index("date")["gprior_one"].clip(lower=1))
    v["jq_prev"] = v["jq_prev"].fillna(v["date"].map(gmean))   # 前走騎手が当日時点で未登場
    v["raw_jq_delta"] = v["jq_cur"] - v["jq_prev"]

    # ---- 騎手×調教師の組み合わせ頻度 (前日まで) ----
    v["one"] = 1.0
    df["pair"] = df["jockey"] + "|" + df["trainer"]
    pr = asof_by_key(df.assign(one=1.0), "pair", ["one"]).set_index(["pair", "date"])["prior_one"]
    tr_n = asof_by_key(df.assign(one=1.0), "trainer", ["one"]).set_index(["trainer", "date"])["prior_one"]
    v["pair"] = v["jockey"] + "|" + v["trainer"]
    pn = pr.reindex(pd.MultiIndex.from_arrays([v["pair"], v["date"]])).fillna(0).to_numpy()
    tn = tr_n.reindex(pd.MultiIndex.from_arrays([v["trainer"], v["date"]])).fillna(0).to_numpy()
    v["raw_jt_pair"] = (pn + 1.0) / (tn + 10.0)
    v["trainer_n_prior"] = tn

    # ---- 単純統計の逸脱 (調教師の前日までの分布, 全体へ縮約) ----
    v["x_li"] = v["raw_log_int"]
    v["x_li2"] = v["raw_log_int"] ** 2
    v["x_dc"] = v["raw_dist_chg"]
    v["x_dc2"] = v["raw_dist_chg"] ** 2
    cat_cols = []
    for a, k in [("a_venue", 2), ("a_surface", 2), ("a_jockey", 2), ("a_cls", 3), ("a_interval", 5), ("a_dist", 3)]:
        for c in range(k):
            col = f"{a}_{c}"
            v[col] = (v[a] == c).astype(float)
            cat_cols.append(col)
    stat_cols = ["one", "x_li", "x_li2", "x_dc", "x_dc2"] + cat_cols
    ts = asof_by_key(v, "trainer", stat_cols)
    gs = asof_global(v, stat_cols)
    v = v.merge(ts, on=["trainer", "date"], how="left").merge(gs, on="date", how="left")

    def shrunk_mean(col):
        n = v["prior_one"]
        gm = v[f"gprior_{col}"] / v["gprior_one"].clip(lower=1)
        return (v[f"prior_{col}"] + K_TRAINER * gm) / (n + K_TRAINER)

    for x, out in [("li", "ss_int_z"), ("dc", "ss_dist_z")]:
        m1 = shrunk_mean(f"x_{x}")
        m2 = shrunk_mean(f"x_{x}2")
        sd = np.sqrt((m2 - m1 ** 2).clip(lower=EPS))
        base = v["raw_log_int"] if x == "li" else v["raw_dist_chg"]
        v[out] = (base - m1) / sd
    for a, k, out in [("a_venue", 2, "ss_venue_s"), ("a_surface", 2, "ss_surface_s"),
                      ("a_jockey", 2, "ss_jockey_s"), ("a_cls", 3, "ss_cls_s")]:
        p = np.zeros(len(v))
        for c in range(k):
            p = np.where(v[a] == c, shrunk_mean(f"{a}_{c}"), p)
        v[out] = -np.log(np.clip(p, EPS, 1.0))

    # 馬自身の通常ローテーションとの差 (前走までの間隔の平均, 調教師平均へ縮約)
    hv = v.groupby("hid", sort=False)
    cs = hv["raw_log_int"].cumsum() - v["raw_log_int"]
    cs2 = hv["x_li2"].cumsum() - v["x_li2"]
    cn = hv.cumcount()
    tm = shrunk_mean("x_li")
    hm = (cs + K_HORSE * tm) / (cn + K_HORSE)
    hsd = np.sqrt(((cs2 + K_HORSE * shrunk_mean("x_li2")) / (cn + K_HORSE) - hm ** 2).clip(lower=EPS))
    v["ss_int_horse_z"] = (v["raw_log_int"] - hm) / hsd
    v["ss_total"] = (0.5 * (v["ss_int_z"] ** 2 + v["ss_dist_z"] ** 2 + v["ss_int_horse_z"] ** 2)
                     + v["ss_venue_s"] + v["ss_surface_s"] + v["ss_jockey_s"] + v["ss_cls_s"])

    # ---- 行動予測モデルの文脈 (対象レースの選択そのものは含めない) ----
    v["ctx_age"] = pd.to_numeric(v["年齢"], errors="coerce")
    v["ctx_sex"] = v["性別"].map(SEX_MAP)
    v["ctx_month"] = v["date"].dt.month
    v["ctx_n_prev"] = v["n_prev_runs"]
    v["ctx_prev_fin"] = pd.to_numeric(v["prev_fin"], errors="coerce")
    v["ctx_prev_margin"] = pd.to_numeric(v["prev_前走着差タイム"], errors="coerce")
    v["ctx_prev_cls"] = v["prev_cls_ord"]
    v["ctx_prev_dist"] = v["prev_dist"]
    v["ctx_prev_surface"] = (v["prev_芝・ダ"].astype(str) == "芝").astype(int)
    v["ctx_prev_place"] = v["prev_場所"].map(PLACE_MAP)
    v["ctx_prev_int"] = (v["prev_date"] - v["prev2_date"]).dt.days
    v["ctx_prev_jq"] = v["jq_prev"]
    v["ctx_horse_mean_li"] = hm
    v["ctx_tr_n"] = v["prior_one"]
    for col in ["x_li", "x_dc"] + cat_cols:
        v[f"ctx_tr_{col}"] = shrunk_mean(col)

    keep = (["rid16", "ban", "hid", "date", "trainer", "jockey", "trainer_n_prior", "n_prev_runs"]
            + [c for c in v.columns if c.startswith(("raw_", "ss_", "ctx_", "a_"))
               and not c.startswith(tuple(f"{a}_" for a in ["a_venue", "a_surface", "a_jockey",
                                                              "a_cls", "a_interval", "a_dist"]))])
    return v[keep].reset_index(drop=True)


def main() -> None:
    df = load_master()
    print(f"master rows (出走のみ) = {len(df):,}")
    f = build(df)
    ok = len(f) / len(df)
    print(f"前走連結が 前走日付 と一致した行 = {len(f):,} ({100*ok:.1f}%)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    f.to_parquet(OUT, index=False)
    print(f"saved -> {OUT}  cols={len(f.columns)}")
    print(f.groupby(f.date.dt.year).size().to_string())


if __name__ == "__main__":
    main()
