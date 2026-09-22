# -*- coding: utf-8 -*-
"""
Effect A(denominator-only)用: DNF馬の19特徴を「現行の未修正パイプラインを
単純にDNF行にも適用したら何を返すか」という定義で構築する（読み取り専用）。

これはGate0B/第2ラウンドで使った「corrected(DNF_SEMANTIC_SPEC.md準拠)」定義
とは異なる。現行production(`build_master_v2.compute_history_features()`・
`parse_kako5.build_from_master()`)は、post-dropna(626,774行、DNFを含まない
"生存者"のみ)を履歴母集団として使う。本モジュールは、DNF行を新たに
「照会点」として追加した場合に、同じpost-dropna母集団を参照して何が返る
はずかを計算する——他のどの行の既存値も変更しない(生存者側のcumcountは
一切再計算しない、新規のクエリ結果を追加するだけ)。

これにより:
  Effect A = 同じ既存スコア(finisher)+この定義でスコアしたDNF馬 をsoftmax分母へ追加
  Effect B = 全員をDNF_SEMANTIC_SPEC.md準拠のcorrected特徴で再スコア
の2つを明確に分離できる。
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(BASE))
from parse_kako5 import _compute_features, _safe_float, _safe_int, KAKO5_COLS, HIST_COLS  # noqa: E402

COL_RID16, COL_BAN, COL_DATE = "レースID(新/馬番無)", "馬番", "日付"
COL_PEDIGREE, COL_PLACE, COL_SURFACE, COL_DIST, COL_JOCKEY = (
    "血統登録番号", "場所", "芝・ダ", "距離", "騎手コード")
COL_JYUN = "着順"

HIST6 = ["course_n_prev", "course_win_rate", "course_top3_rate",
          "jockey_n_prev", "jockey_win_rate", "jockey_top3_rate"]


def _dist_band(d):
    if pd.isna(d):
        return "?"
    d = int(d)
    if d <= 1400:
        return "短"
    if d <= 1700:
        return "マ"
    if d <= 2200:
        return "中"
    return "長"


def compute_buggy_course_jockey_for_dnf(survivors: pd.DataFrame, dnf_rows: pd.DataFrame) -> pd.DataFrame:
    """survivors(post-dropna, 現行productionの実際の履歴母集団)を対象に、
    dnf_rows各行を照会点として course_n_prev系6特徴を計算する。
    survivors自身の値は一切変更しない(現行master_v2格納値と同一のはず)。
    """
    s = survivors.copy()
    s[COL_DATE] = pd.to_numeric(s[COL_DATE], errors="coerce").astype("Int64")
    s[COL_JYUN] = pd.to_numeric(s[COL_JYUN], errors="coerce")
    s["_is_win"] = (s[COL_JYUN] == 1)
    s["_is_top3"] = (s[COL_JYUN] <= 3)
    s["_dist_b"] = pd.to_numeric(s[COL_DIST], errors="coerce").apply(_dist_band)
    s["_course_key"] = s[COL_PLACE].astype(str) + "|" + s[COL_SURFACE].astype(str) + "|" + s["_dist_b"]

    results = []
    for _, row in dnf_rows.iterrows():
        hid = row[COL_PEDIGREE]
        race_date = int(row[COL_DATE])
        hist = s[(s[COL_PEDIGREE] == hid) & (s[COL_DATE] < race_date)]

        band = _dist_band(row[COL_DIST])
        ck = f"{row[COL_PLACE]}|{row[COL_SURFACE]}|{band}"
        sel_course = hist["_course_key"] == ck
        n_c = int(sel_course.sum())
        course_win_rate = float(hist.loc[sel_course, "_is_win"].mean()) if n_c > 0 else np.nan
        course_top3_rate = float(hist.loc[sel_course, "_is_top3"].mean()) if n_c > 0 else np.nan

        jc = row[COL_JOCKEY]
        sel_j = hist[COL_JOCKEY] == jc
        n_j = int(sel_j.sum())
        jockey_win_rate = float(hist.loc[sel_j, "_is_win"].mean()) if n_j > 0 else np.nan
        jockey_top3_rate = float(hist.loc[sel_j, "_is_top3"].mean()) if n_j > 0 else np.nan

        results.append({
            COL_RID16: row[COL_RID16], COL_BAN: row[COL_BAN],
            "course_n_prev": float(n_c), "course_win_rate": course_win_rate,
            "course_top3_rate": course_top3_rate,
            "jockey_n_prev": float(n_j), "jockey_win_rate": jockey_win_rate,
            "jockey_top3_rate": jockey_top3_rate,
        })
    return pd.DataFrame(results)


def compute_buggy_kako5_for_dnf(survivors: pd.DataFrame, dnf_rows: pd.DataFrame) -> pd.DataFrame:
    """survivors(post-dropna)を対象に、dnf_rows各行を照会点として
    parse_kako5.build_from_master()と同一ロジック(直近5"survivor行")で
    kako5_*/hist_same_*を計算する。"""
    s = survivors.copy()
    s["date_dt"] = pd.to_datetime(s[COL_DATE].astype(str), format="%Y%m%d", errors="coerce")
    td_map = {"芝": "T", "ダ": "D", "ダート": "D", "T": "T", "D": "D"}
    s["_td_code"] = s[COL_SURFACE].map(td_map).fillna("")
    s["_dist"] = pd.to_numeric(s[COL_DIST], errors="coerce")
    s["_place"] = s[COL_PLACE].astype(str)
    s = s.sort_values([COL_PEDIGREE, "date_dt"])

    results = []
    for _, row in dnf_rows.iterrows():
        hid = row[COL_PEDIGREE]
        race_date = pd.to_datetime(str(int(row[COL_DATE])), format="%Y%m%d")
        group = s[s[COL_PEDIGREE] == hid]
        past5 = group[group["date_dt"] < race_date].tail(5)

        past_races = []
        for _, pr in past5.iloc[::-1].iterrows():
            past_races.append({
                "着順": _safe_int(pr.get(COL_JYUN)),
                "人気": None,
                "上り3F": _safe_float(pr.get("前走上り3F")),
                "TD": td_map.get(str(pr.get(COL_SURFACE, "")), ""),
                "距離": _safe_float(pr.get(COL_DIST)),
                "場所": str(pr.get(COL_PLACE, "")),
            })
        cur_td = td_map.get(str(row[COL_SURFACE]), "")
        cur_dist = _safe_float(row[COL_DIST])
        cur_place = str(row[COL_PLACE])
        feats = _compute_features(past_races, current_td=cur_td, current_dist=cur_dist,
                                   current_place=cur_place)
        for hcol in HIST_COLS:
            feats[hcol] = np.nan  # 19特徴の対象外(参考、本関数では計算しない)
        feats[COL_RID16] = row[COL_RID16]
        feats[COL_BAN] = row[COL_BAN]
        results.append(feats)
    return pd.DataFrame(results)
