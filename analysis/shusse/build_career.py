# -*- coding: utf-8 -*-
"""
analysis/shusse/build_career.py — 出世予測 AI: データ理解・キャリア集計
=====================================================================
master_v2 から馬ごとのキャリア（平地のみ）を組み立て、クラス名を正規化し、
デビュー年コホート別の馬数・2 歳デビュー数・重賞到達率などを集計する。

実行（root から）:
    venv311\\Scripts\\python.exe -m analysis.shusse.build_career

出力（analysis/shusse/out/）:
    starts.parquet         馬×出走の正規化済み明細（平地のみ、障害は除外）
    career.parquet         馬 1 行のキャリア要約
    cohort_summary.csv     デビュー年コホート別の集計
    class_transition.csv   k 走目クラス → k+1 走目クラスの遷移表
    data_summary.json      上記の要点

設計メモ:
  - 障害は race_eligibility.py と同じ基準 `トラックコード(JV) ∈ 51..59` で除外
    （クラス名 ＪＧ１〜３ も同基準に含まれる）。
  - 馬主(最新/仮想) は future overwrite 列のため一切使わない。
  - 2013 年の初出走行の 60% は 2012 以前に走った馬（左打ち切り）。
    「真のデビュー」= 初出走行で 前走確定着順 が欠損、かつ 年齢 ∈ {2,3}。
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
MASTER = BASE / "data" / "master_v2_20130105-20251228.csv"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)

COL_RACE_ID = "レースID(新/馬番無)"
COL_HORSE = "血統登録番号"

USECOLS = [
    "日付", COL_RACE_ID, COL_HORSE, "馬名", "クラス名", "トラックコード(JV)", "着順",
    "出走頭数", "年齢", "性別", "芝・ダ", "距離", "場所", "レース名", "年齢限定", "指定条件",
    "父タイプ名", "母父タイプ名", "種牡馬", "母父馬", "生産者", "調教師コード", "騎手コード",
    "騎手年齢", "調教師年齢", "前走確定着順", "前走着差タイム", "前走日付", "jockey_top3_rate",
]

# ---- クラス正規化 -----------------------------------------------------------
# race_level: レースの格（細分）。2019 の名称変更（500万→1勝 等）をそろえる。
RACE_LEVEL = {
    "新馬": 0, "未勝利": 1,
    "500万": 2, "1勝": 2,
    "1000万": 3, "2勝": 3,
    "1600万": 4, "3勝": 4,
    "オープン": 5, "OP(L)": 5,         # オープン特別 / リステッド（半角ｵｰﾌﾟﾝは NFKC で全角化して照合）
    "重賞": 6, "Ｇ３": 6,               # 「重賞」= 格付け前の新設重賞（G3 相当として扱う）
    "Ｇ２": 7, "Ｇ１": 8,
    "ＪＧ３": 6, "ＪＧ２": 7, "ＪＧ１": 8,  # 障害重賞（トラックコードで除外されるが念のため）
}
RACE_LEVEL_NFKC = {unicodedata.normalize("NFKC", k): v for k, v in RACE_LEVEL.items()}
RACE_LEVEL_NAME = {0: "新馬", 1: "未勝利", 2: "1勝", 3: "2勝", 4: "3勝", 5: "OP/L", 6: "G3", 7: "G2", 8: "G1"}

# target_level: 出世予測の順序カテゴリ（5 段）
#   L0 未勝利圏（新馬/未勝利のみ、または出走なし） L1 1勝  L2 2勝  L3 3勝/OP/L  L4 重賞(G3+)
TARGET_LEVEL_OF_RACE = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 3, 6: 4, 7: 4, 8: 4}
TARGET_LEVEL_NAME = {0: "未勝利圏", 1: "1勝クラス", 2: "2勝クラス", 3: "3勝/OP", 4: "重賞"}


def to_target_level(race_level: pd.Series) -> pd.Series:
    return race_level.map(TARGET_LEVEL_OF_RACE).astype(int)


JUMP_MIN, JUMP_MAX = 51, 59
DATA_END = pd.Timestamp("2025-12-28")


def load_master() -> pd.DataFrame:
    df = pd.read_csv(MASTER, usecols=USECOLS, encoding="utf-8-sig", low_memory=False)
    df["date"] = pd.to_datetime(df["日付"].astype(str), format="%Y%m%d")
    df["year"] = df["日付"] // 10000
    df["class_norm"] = df["クラス名"].astype(str).map(lambda x: unicodedata.normalize("NFKC", x))
    df["race_level"] = df["class_norm"].map(RACE_LEVEL_NFKC)
    assert df["race_level"].notna().all(), df.loc[df["race_level"].isna(), "クラス名"].unique()
    df["race_level"] = df["race_level"].astype(int)
    df["is_jump"] = df["トラックコード(JV)"].between(JUMP_MIN, JUMP_MAX)
    df["margin_prev"] = pd.to_numeric(df["前走着差タイム"].astype(str).str.strip(), errors="coerce")
    df["pos"] = df["着順"].astype(int)
    df["field"] = df["出走頭数"].astype(int)
    df["win"] = (df["pos"] == 1).astype(int)
    df["top3"] = (df["pos"] <= 3).astype(int)
    df["rel_pos"] = (df["pos"] - 1) / (df["field"] - 1).clip(lower=1)
    return df


def build_starts(df: pd.DataFrame) -> pd.DataFrame:
    """平地のみの出走明細。k = 平地通算何走目か。"""
    n_jump = df.groupby(COL_HORSE)["is_jump"].sum().rename("n_jump_starts")
    flat = df[~df["is_jump"]].sort_values([COL_HORSE, "date", COL_RACE_ID]).copy()
    flat["k"] = flat.groupby(COL_HORSE).cumcount() + 1
    g = flat.groupby(COL_HORSE)
    # 自分のこの走の着差: 次行の 前走着差タイム を持ち上げる
    # ★ k+1 走目が存在する馬しか埋まらない（＝引退と相関）。学習では基準点より前の走にのみ使う
    flat["margin_this"] = g["margin_prev"].shift(-1)
    flat["days_since_prev"] = (flat["date"] - g["date"].shift(1)).dt.days
    flat["target_level"] = to_target_level(flat["race_level"])
    flat = flat.merge(n_jump, left_on=COL_HORSE, right_index=True, how="left")
    return flat


def build_career(flat: pd.DataFrame) -> pd.DataFrame:
    g = flat.groupby(COL_HORSE)
    first = flat.drop_duplicates(COL_HORSE, keep="first").set_index(COL_HORSE)  # 初出走行そのもの（groupby.first は非 NaN 先取りなので不可）
    career = pd.DataFrame({
        "馬名": first["馬名"],
        "debut_date": first["date"],
        "debut_year": first["year"],
        "debut_age": first["年齢"],
        "debut_class": first["クラス名"],
        "debut_place": first["場所"],
        "true_debut": first["前走確定着順"].isna() & first["年齢"].isin([2, 3]),
        "sex": first["性別"],
        "sire": first["種牡馬"],
        "sire_type": first["父タイプ名"],
        "bms_type": first["母父タイプ名"],
        "breeder": first["生産者"],
        "trainer_debut": first["調教師コード"],
        "n_starts": g.size(),
        "n_wins": g["win"].sum(),
        "n_top3": g["top3"].sum(),
        "last_date": g["date"].max(),
        "max_race_level": g["race_level"].max(),
        "n_jump_starts": first["n_jump_starts"],
    })
    career["career_years"] = (career["last_date"] - career["debut_date"]).dt.days / 365.25
    # 右打ち切り: データ末尾 6 か月以内に走っていれば現役扱い
    career["retired_flag"] = career["last_date"] < (DATA_END - pd.Timedelta(days=183))
    graded = flat[flat["race_level"] >= 6].groupby(COL_HORSE)["date"].min().rename("first_graded_date")
    g1 = flat[flat["race_level"] >= 8].groupby(COL_HORSE)["date"].min().rename("first_g1_date")
    first_win = flat[flat["win"] == 1].groupby(COL_HORSE)["date"].min().rename("first_win_date")
    career = career.join(graded).join(g1).join(first_win)
    career["graded_ever"] = career["first_graded_date"].notna()
    career["graded_within_24m"] = ((career["first_graded_date"] - career["debut_date"]).dt.days <= 730).fillna(False)
    career["g1_ever"] = career["first_g1_date"].notna()
    career["won_ever"] = career["first_win_date"].notna()
    # 3 走目基準点
    third = flat[flat["k"] == 3].set_index(COL_HORSE)
    career["third_start_date"] = third["date"]
    career["third_start_age"] = third["年齢"]
    tmp = flat.merge(third[["date"]].rename(columns={"date": "base_date"}), left_on=COL_HORSE, right_index=True, how="inner")
    after = tmp[tmp["date"] > tmp["base_date"]]
    win12 = after[after["date"] <= after["base_date"] + pd.Timedelta(days=365)]
    career = career.join(win12.groupby(COL_HORSE)["target_level"].max().rename("level_12m_after_3rd"))
    career = career.join(win12.groupby(COL_HORSE).size().rename("n_starts_12m_after_3rd"))
    has3 = career["third_start_date"].notna()
    career.loc[has3 & career["level_12m_after_3rd"].isna(), "level_12m_after_3rd"] = 0
    career.loc[has3 & career["n_starts_12m_after_3rd"].isna(), "n_starts_12m_after_3rd"] = 0
    career["horizon12_complete"] = has3 & (career["third_start_date"] + pd.Timedelta(days=365) <= DATA_END)
    win24 = after[(after["date"] <= after["base_date"] + pd.Timedelta(days=730)) & (after["race_level"] >= 6)]
    career["graded_24m_after_3rd"] = career.index.isin(win24[COL_HORSE].unique()) & has3
    career["horizon24_complete"] = has3 & (career["third_start_date"] + pd.Timedelta(days=730) <= DATA_END)
    return career.reset_index()


def cohort_summary(career: pd.DataFrame) -> pd.DataFrame:
    c = career[career["true_debut"]]
    rows = []
    for y, d in c.groupby("debut_year"):
        h3 = d[d["horizon12_complete"]]
        dist = h3["level_12m_after_3rd"].value_counts(normalize=True).sort_index() if len(h3) else pd.Series(dtype=float)
        h24 = d[d["horizon24_complete"]]
        rows.append({
            "debut_year": int(y),
            "n_horses": int(len(d)),
            "n_debut_age2": int((d["debut_age"] == 2).sum()),
            "share_debut_age2": round(float((d["debut_age"] == 2).mean()), 3),
            "mean_starts": round(float(d["n_starts"].mean()), 2),
            "share_won_ever": round(float(d["won_ever"].mean()), 3),
            "share_3plus_starts": round(float((d["n_starts"] >= 3).mean()), 3),
            "graded_ever_rate": round(float(d["graded_ever"].mean()), 4),
            "graded_within_24m_rate": round(float(d["graded_within_24m"].mean()), 4),
            "g1_ever_rate": round(float(d["g1_ever"].mean()), 4),
            "n_third_start_h12": int(len(h3)),
            **{f"L{k}_12m_after_3rd": round(float(dist.get(k, 0.0)), 4) for k in range(5)},
            "n_third_start_h24": int(len(h24)),
            "graded_24m_after_3rd_rate": round(float(h24["graded_24m_after_3rd"].mean()), 4) if len(h24) else np.nan,
            "retired_share": round(float(d["retired_flag"].mean()), 3),
        })
    return pd.DataFrame(rows)


def class_transition(flat: pd.DataFrame) -> pd.DataFrame:
    nxt = flat.groupby(COL_HORSE)["race_level"].shift(-1)
    t = pd.crosstab(flat["race_level"].map(RACE_LEVEL_NAME), nxt.map(RACE_LEVEL_NAME).fillna("(引退/未出走)"), normalize="index")
    order = [RACE_LEVEL_NAME[i] for i in range(9)]
    t = t.reindex(index=order, columns=order + ["(引退/未出走)"]).fillna(0)
    return t.round(3)


def main() -> None:
    df = load_master()
    flat = build_starts(df)
    career = build_career(flat)
    summ = cohort_summary(career)
    trans = class_transition(flat)

    flat.drop(columns=["前走着差タイム", "前走日付"]).to_parquet(OUT / "starts.parquet", index=False)
    career.to_parquet(OUT / "career.parquet", index=False)
    summ.to_csv(OUT / "cohort_summary.csv", index=False, encoding="utf-8-sig")
    trans.to_csv(OUT / "class_transition.csv", encoding="utf-8-sig")

    td = career[career["true_debut"]]
    summary = {
        "rows_master": int(len(df)),
        "rows_jump_excluded": int(df["is_jump"].sum()),
        "rows_flat": int(len(flat)),
        "horses_total": int(career.shape[0]),
        "horses_true_debut": int(td.shape[0]),
        "horses_left_censored_or_transfer": int((~career["true_debut"]).sum()),
        "class_name_counts": {k: int(v) for k, v in df["クラス名"].value_counts().items()},
        "race_level_map": RACE_LEVEL,
        "target_level_name": TARGET_LEVEL_NAME,
        "cohort_summary": summ.to_dict(orient="records"),
        "data_end": str(DATA_END.date()),
    }
    (OUT / "data_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(f"master rows={len(df):,} jump rows={df['is_jump'].sum():,} flat rows={len(flat):,}")
    print(f"horses={career.shape[0]:,} true_debut={td.shape[0]:,}")
    print(summ.to_string(index=False))
    print(trans.to_string())


if __name__ == "__main__":
    main()
