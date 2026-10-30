# -*- coding: utf-8 -*-
"""
analysis/shusse/build_career_v2.py — 出世予測 AI v2: 外部走破タイム結合・馬属性
==============================================================================
v1（build_career.py）の平地出走明細に、外部 kekka（E:\\競馬過去走データ\\raw_data\\
kekka_2010_2025_fix_raceid_v2.csv、読み取り専用）の **自走の走破タイム・上り3F・
上り3F順・馬体重・生年月日** を結合し、レース内の勝ち時計との差 `margin`（秒）を
自走行から as-of で付ける（v1 の「次走行から持ち上げる」方式は引退と相関する leak
のため廃止）。

実行（root から）:
    venv311\\Scripts\\python.exe -m analysis.shusse.build_career_v2

出力（analysis/shusse/out_v2/）:
    starts_v2.parquet   平地出走明細 + time_sec / win_time / margin / agari3f / agari_rank / weight
    horses_v2.parquet   馬 1 行（デビュー情報・true_debut・生年月日・デビュー時日齢）
    join_report.json    結合率（年別）、キーの診断、タイム欠損率

結合キー:
  外部 kekka の `race_id` 列は master の `レースID(新/馬番無)` と 95.6% しか一致しない
  （約 4.4% のレースで場コード/開催部分が壊れている: 例 2014053100040211 vs
  2014053105021111）。一方 (日付, 場所, R 番号, 血統登録番号) は master の平地行と
  100% 一致し着順も全件一致したので、これを主キーにする（race_id は診断のみ）。
  master は中止/除外/取消馬を含まないため、結合後に走破タイム "----" は残らない。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.shusse.build_career import (
    COL_HORSE, COL_RACE_ID, DATA_END, build_starts, load_master,
)

OUT = Path(__file__).resolve().parent / "out_v2"
OUT.mkdir(parents=True, exist_ok=True)
KEKKA_EXT = Path(r"E:\競馬過去走データ\raw_data\kekka_2010_2025_fix_raceid_v2.csv")
KEKKA_USECOLS = ["日付", "場所", "rr", "race_id", "血統登録番号", "着順", "走破タイム",
                 "上り3F", "上り3F順", "馬体重", "生年月日"]
_FW2HW = str.maketrans("０１２３４５６７８９", "0123456789")
_CIRCLED = {c: i + 1 for i, c in enumerate("①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱")}


def parse_pos(x) -> float:
    """着順: 1〜9 着は全角、10 着以降は半角、降着は丸数字。中止/除外/取消は NaN。"""
    s = str(x).strip().translate(_FW2HW)
    if s in _CIRCLED:
        return float(_CIRCLED[s])
    try:
        return float(int(s))
    except ValueError:
        return np.nan


def parse_time_mssd(x) -> float:
    """走破タイム: 半角 4 桁 'MSSd'（例 1336 = 1 分 33 秒 6）→ 秒。'----' は NaN。"""
    s = str(x).strip()
    if not s.isdigit() or len(s) != 4:
        return np.nan
    m, ss, d = int(s[0]), int(s[1:3]), int(s[3])
    return m * 60 + ss + d / 10.0


def parse_birth(x) -> pd.Timestamp:
    """生年月日: '2022. 5. 2' 形式。"""
    s = str(x).replace(" ", "")
    try:
        return pd.Timestamp(s.replace(".", "-"))
    except Exception:
        return pd.NaT


def load_kekka_ext(path: Path = KEKKA_EXT) -> pd.DataFrame:
    t0 = time.time()
    k = pd.read_csv(path, usecols=KEKKA_USECOLS, encoding="cp932", low_memory=False,
                    dtype={"着順": str, "走破タイム": str, "生年月日": str})
    k["ymd"] = 20000000 + k["日付"].astype(int)             # 6 桁 yymmdd → 8 桁
    k = k[k["ymd"] >= 20130101].copy()
    k["pos_ext"] = k["着順"].map(parse_pos)
    k["time_sec"] = k["走破タイム"].map(parse_time_mssd)
    k["agari3f"] = pd.to_numeric(k["上り3F"], errors="coerce")
    k["agari_rank"] = pd.to_numeric(k["上り3F順"], errors="coerce")
    k["weight"] = pd.to_numeric(k["馬体重"], errors="coerce")
    k["birth_date"] = k["生年月日"].map(parse_birth)
    k = k.rename(columns={"rr": "R", "race_id": "race_id_ext"})
    k["場所"] = k["場所"].astype(str).str.strip()
    # レース内の勝ち時計: 1 着馬のタイム（同着は min）。無ければ完走馬の最速。
    g = k.groupby(["ymd", "場所", "R"])
    win_t = k[k["pos_ext"] == 1].groupby(["ymd", "場所", "R"])["time_sec"].min().rename("win_time")
    fast_t = g["time_sec"].min().rename("fast_time")
    k = k.merge(win_t, left_on=["ymd", "場所", "R"], right_index=True, how="left")
    k = k.merge(fast_t, left_on=["ymd", "場所", "R"], right_index=True, how="left")
    k["win_time"] = k["win_time"].fillna(k["fast_time"])
    k["margin"] = k["time_sec"] - k["win_time"]
    print(f"kekka_ext rows(2013+)={len(k):,} loaded in {time.time() - t0:.1f}s")
    return k[["ymd", "場所", "R", COL_HORSE, "race_id_ext", "pos_ext", "time_sec", "win_time", "margin",
              "agari3f", "agari_rank", "weight", "birth_date"]]


def join_times(flat: pd.DataFrame, kex: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    flat = flat.copy()
    flat["R"] = (flat[COL_RACE_ID] % 100).astype(int)
    flat["場所"] = flat["場所"].astype(str).str.strip()
    key = ["日付", "場所", "R", COL_HORSE]
    kx = kex.rename(columns={"ymd": "日付"})
    assert not kx.duplicated(key).any(), "kekka_ext key duplicated"
    m = flat.merge(kx, on=key, how="left", indicator=True)
    assert len(m) == len(flat)
    matched = m["_merge"] == "both"
    rep: dict = {
        "rows_flat": int(len(flat)),
        "rows_matched_key_date_place_R_horse": int(matched.sum()),
        "join_rate_total": round(float(matched.mean()), 5),
        "join_rate_by_year": {int(y): round(float(v), 5) for y, v in m.groupby(m["日付"] // 10000)["_merge"].apply(lambda s: (s == "both").mean()).items()},
        "race_id_ext_equals_master_rate_among_matched": round(float((m.loc[matched, "race_id_ext"] == m.loc[matched, COL_RACE_ID]).mean()), 5),
        "pos_agreement_among_matched": round(float((m.loc[matched, "pos_ext"] == m.loc[matched, "pos"]).mean()), 5),
        "time_sec_missing_among_matched": round(float(m.loc[matched, "time_sec"].isna().mean()), 5),
        "margin_missing_total": round(float(m["margin"].isna().mean()), 5),
        "agari_rank_missing_total": round(float(m["agari_rank"].isna().mean()), 5),
        "weight_missing_total": round(float(m["weight"].isna().mean()), 5),
        "margin_describe": {k: round(float(v), 3) for k, v in m["margin"].describe().items()},
        "margin_negative_rows": int((m["margin"] < -1e-9).sum()),
    }
    # pos が一致しない行はタイム系を信用しない
    bad = matched & (m["pos_ext"] != m["pos"])
    rep["rows_pos_mismatch_nulled"] = int(bad.sum())
    m.loc[bad, ["time_sec", "win_time", "margin", "agari3f", "agari_rank", "weight"]] = np.nan
    m = m.drop(columns=["_merge", "pos_ext", "race_id_ext"])
    return m, rep


def add_jockey_asof(flat: pd.DataFrame, window_days: int = 730, m: float = 50.0) -> pd.DataFrame:
    """騎手の as-of 3 着内率（直近 window_days、当日を含めない）。
    master の jockey_top3_rate は 馬×騎手 のペア率（初コンビで NaN、k=1 で 100% 欠損）なので、
    騎手一般の強さは平地明細から自前で as-of 集計する。縮約 m=50 で全体率へ。"""
    flat = flat.copy()
    jd = flat.groupby(["騎手コード", "date"]).agg(n=("top3", "size"), t3=("top3", "sum")).reset_index().sort_values(["騎手コード", "date"])
    jd["cum_n"] = jd.groupby("騎手コード")["n"].cumsum()
    jd["cum_t3"] = jd.groupby("騎手コード")["t3"].cumsum()
    ev = jd[["騎手コード", "date", "cum_n", "cum_t3"]].rename(columns={"date": "t"}).sort_values("t")
    q = flat[["騎手コード", "date"]].copy()
    q["row"] = np.arange(len(q))
    q["t"] = q["date"]
    q = q.sort_values("t")
    a = pd.merge_asof(q, ev, on="t", by="騎手コード", allow_exact_matches=False, direction="backward")
    q2 = q.copy()
    q2["t"] = q2["date"] - pd.Timedelta(days=window_days)
    q2 = q2.sort_values("t")
    b = pd.merge_asof(q2, ev, on="t", by="騎手コード", allow_exact_matches=True, direction="backward")
    a = a.sort_values("row")
    b = b.sort_values("row")
    n = a["cum_n"].fillna(0).values - b["cum_n"].fillna(0).values
    t3 = a["cum_t3"].fillna(0).values - b["cum_t3"].fillna(0).values
    g = float(flat["top3"].mean())
    flat["jky_n_2y"] = n
    flat["jky_top3_2y"] = (t3 + m * g) / (n + m)
    return flat


def build_horses(flat: pd.DataFrame) -> pd.DataFrame:
    """馬 1 行: デビュー情報（v1 の true_debut 判定を踏襲）+ 生年月日・デビュー時日齢。"""
    first = flat.drop_duplicates(COL_HORSE, keep="first").set_index(COL_HORSE)
    g = flat.groupby(COL_HORSE)
    birth = g["birth_date"].first()
    # 父タイプ名は新しい種牡馬で欠損が増える（2022 行の 60%）。種牡馬の血統タイプは固定属性なので
    # 同じ種牡馬の他行の最頻値で埋める（時間情報を含まない）。
    st = flat.dropna(subset=["父タイプ名"]).groupby("種牡馬")["父タイプ名"].agg(lambda x: x.value_counts().index[0])
    bt = flat.dropna(subset=["母父タイプ名"]).groupby("母父馬")["母父タイプ名"].agg(lambda x: x.value_counts().index[0])
    h = pd.DataFrame({
        "馬名": first["馬名"],
        "debut_date": first["date"],
        "debut_year": first["year"],
        "debut_age": first["年齢"],
        "true_debut": first["前走確定着順"].isna() & first["年齢"].isin([2, 3]),
        "sex": first["性別"],
        "sire": first["種牡馬"],
        "sire_type": first["父タイプ名"].fillna(first["種牡馬"].map(st)),
        "bms_type": first["母父タイプ名"].fillna(first["母父馬"].map(bt)),
        "sire_type_raw_missing": first["父タイプ名"].isna(),
        "breeder": first["生産者"],
        "trainer_debut": first["調教師コード"],
        "n_starts": g.size(),
        "last_date": g["date"].max(),
        "birth_date": birth,
    })
    h["debut_dayage"] = (h["debut_date"] - h["birth_date"]).dt.days
    # 妥当性: デビュー時日齢は 2 歳 6 月〜3 歳末なので概ね 600〜1,400 日
    bad = h["debut_dayage"].notna() & ~h["debut_dayage"].between(500, 1500)
    h.loc[bad, "debut_dayage"] = np.nan
    return h.reset_index()


def main() -> None:
    t0 = time.time()
    df = load_master()
    flat = build_starts(df)
    flat = flat.drop(columns=["前走着差タイム", "前走日付", "margin_prev", "margin_this"])
    kex = load_kekka_ext()
    flat, rep = join_times(flat, kex)
    flat = add_jockey_asof(flat)
    horses = build_horses(flat)
    rep["jky_top3_2y_missing"] = round(float(flat["jky_top3_2y"].isna().mean()), 5)
    rep["jky_n_2y_zero_rate_by_year"] = {int(y): round(float(v), 4) for y, v in flat.groupby("year")["jky_n_2y"].apply(lambda x: (x == 0).mean()).items()}
    rep["sire_type_missing_true_debut_raw"] = round(float(horses.loc[horses["true_debut"], "sire_type_raw_missing"].mean()), 5)
    rep["sire_type_missing_true_debut_filled"] = round(float(horses.loc[horses["true_debut"], "sire_type"].isna().mean()), 5)
    rep["horses_total"] = int(len(horses))
    rep["horses_true_debut"] = int(horses["true_debut"].sum())
    rep["birth_date_missing_true_debut"] = round(float(horses.loc[horses["true_debut"], "birth_date"].isna().mean()), 5)
    rep["debut_dayage_describe_true_debut"] = {k: round(float(v), 1) for k, v in horses.loc[horses["true_debut"], "debut_dayage"].describe().items()}
    rep["data_end"] = str(DATA_END.date())
    rep["source_kekka_ext"] = str(KEKKA_EXT)
    rep["elapsed_sec"] = round(time.time() - t0, 1)
    flat.to_parquet(OUT / "starts_v2.parquet", index=False)
    horses.to_parquet(OUT / "horses_v2.parquet", index=False)
    (OUT / "join_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
