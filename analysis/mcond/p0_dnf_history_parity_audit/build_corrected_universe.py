# -*- coding: utf-8 -*-
"""
P0 DNF history parity audit — 修正版(意味定義spec準拠)のfull-starter母集団
構築ユーティリティ（読み取り専用、本番ファイルは一切変更しない）。

`DNF_SEMANTIC_SPEC.md`で固定した定義を実装する:
  - 止(DNF)は「出走経験」に数える(course/jockey分母+kako5 window slot)
  - 外・消は「出走経験」に数えない(scratchとして完全除外)
  - kako5のDNF走は着順・上り3Fを欠損(None)、TD/距離/場所は実際の値を保持

EXP13 Gate0Bの`gate0b_course_jockey_history_parity.py`等は、pre-dropna全体
(631,965行)に対して素朴にcumcountしていたため、**外・消(2,219行)も
「出走経験」として誤って数えていた**(止2,946行を正しく数える一方で)。
本モジュールはこの点を修正し、着順の生文字列(convert_finish適用前)で
止/外/消/数値を正しく分類してから集計する。
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(BASE))
from build_dataset import (  # noqa: E402
    load_csv, normalize_date,
    CSV_LGBM, CSV_CAT, CSV_TORCH, CSV_ADD, JOIN_KEY, COL_FINISH,
    COL_DATE as BD_COL_DATE, TARGET,
)

COL_PEDIGREE, COL_PLACE, COL_SURFACE, COL_DIST, COL_JOCKEY = (
    "血統登録番号", "場所", "芝・ダ", "距離", "騎手コード")
COL_RID16, COL_BAN = "レースID(新/馬番無)", "馬番"

_ZEN2HAN = str.maketrans("０１２３４５６７８９", "0123456789")


def classify_finish_raw(raw: str) -> str:
    """着順の生文字列(convert_finish適用前)を分類する。
    内部add CSVの実測(2026-09-22)では非数値コードは 止/外/消 の3種のみ
    (丸数字は本ファイルには出現しない、631,965行全数チェック済み)。"""
    s = str(raw).translate(_ZEN2HAN)
    if s == "止":
        return "dnf"
    if s == "外":
        return "scratch_jogai"
    if s == "消":
        return "scratch_torikeshi"
    try:
        float(s)
        return "numeric"
    except (TypeError, ValueError):
        return "unknown"


def load_full_raw_universe() -> pd.DataFrame:
    """build_dataset.py:build_master()のJOIN部分(dropna適用前、全120特徴の
    原材料となるlgbm+cat+torch+add全列)を再現し、着順の生文字列分類列
    (_finish_class)を追加して返す。631,965行。"""
    df_lgbm = load_csv(CSV_LGBM, "lgbm")
    df_cat = load_csv(CSV_CAT, "cat")
    df_torch = load_csv(CSV_TORCH, "torch")
    df_add = load_csv(CSV_ADD, "add")

    df_add["_finish_raw_str"] = df_add[COL_FINISH].astype(str)
    df_add["_finish_class"] = df_add["_finish_raw_str"].apply(classify_finish_raw)

    master = df_lgbm.merge(df_cat, on=JOIN_KEY, how="left", suffixes=("", "_cat"))
    master = master.merge(df_torch, on=JOIN_KEY, how="left", suffixes=("", "_torch"))
    master = master.merge(df_add, on=JOIN_KEY, how="left", suffixes=("", "_add"))
    dup_cols = [c for c in master.columns if c.endswith("_cat") or c.endswith("_torch") or c.endswith("_add")]
    master = master.drop(columns=dup_cols)
    master[BD_COL_DATE] = normalize_date(master[BD_COL_DATE])
    master["date_dt"] = pd.to_datetime(master[BD_COL_DATE].astype(str), format="%Y%m%d")

    # 数値着順(NaN=止/外/消/join gap)
    zen2han_num = master[COL_FINISH].astype(str).str.translate(_ZEN2HAN)
    master[COL_FINISH] = pd.to_numeric(zen2han_num, errors="coerce")
    master[TARGET] = (master[COL_FINISH] <= 3).astype("Int8")

    n_join_gap = int((master["_finish_class"].isna()).sum())
    assert len(master) == 631_965, f"行数異常: {len(master):,}"
    return master


def experience_population(df: pd.DataFrame) -> pd.Series:
    """「出走経験」として数えるべき行のbool mask。
    止(dnf)+numeric(通常完走)を含む、外・消(scratch)は除外。
    _finish_classがNaN(join gap)の行も、そもそも実在する出走記録か不明な
    ため安全側で除外する(母集団を恣意的に広げない)。"""
    cls = df["_finish_class"]
    return cls.isin(["dnf", "numeric"])


def compute_corrected_course_jockey(df: pd.DataFrame) -> pd.DataFrame:
    """course_n_prev系/jockey_n_prev系6特徴を、意味定義spec通りに計算する。
    scratchは母集団から除外してからcumcount(=scratchはwindow/分母を一切
    消費しない)、DNFは残るため分母には数えるがnumeratorには寄与しない
    (着順NaNのため_is_win/_is_top3が自動的に0になる、ロジック自体は
    build_master_v2.py:compute_history_features()と同一)。"""
    df = df.copy()
    exp_mask = experience_population(df)
    d = df.loc[exp_mask].copy()

    d["_is_win"] = (d[COL_FINISH] == 1).astype("Int8")
    d["_is_top3"] = (d[COL_FINISH] <= 3).astype("Int8")

    def dist_band(dd):
        if pd.isna(dd):
            return "?"
        dd = int(dd)
        if dd <= 1400:
            return "短"
        if dd <= 1700:
            return "マ"
        if dd <= 2200:
            return "中"
        return "長"

    d["_dist_b"] = pd.to_numeric(d[COL_DIST], errors="coerce").apply(dist_band)
    d["_course_key"] = d[COL_PLACE].astype(str) + "|" + d[COL_SURFACE].astype(str) + "|" + d["_dist_b"]
    d = d.sort_values([COL_PEDIGREE, BD_COL_DATE]).reset_index(drop=False)  # keep original index

    g = d.groupby([COL_PEDIGREE, "_course_key"])
    d["course_n_prev"] = g.cumcount()
    d["course_wins_prev"] = g["_is_win"].cumsum().astype("Int64") - d["_is_win"].astype("Int64")
    d["course_top3_prev"] = g["_is_top3"].cumsum().astype("Int64") - d["_is_top3"].astype("Int64")
    d["course_win_rate"] = np.where(d["course_n_prev"] > 0, d["course_wins_prev"] / d["course_n_prev"], np.nan)
    d["course_top3_rate"] = np.where(d["course_n_prev"] > 0, d["course_top3_prev"] / d["course_n_prev"], np.nan)

    gj = d.groupby([COL_PEDIGREE, COL_JOCKEY])
    d["jockey_n_prev"] = gj.cumcount()
    d["jockey_wins_prev"] = gj["_is_win"].cumsum().astype("Int64") - d["_is_win"].astype("Int64")
    d["jockey_top3_prev"] = gj["_is_top3"].cumsum().astype("Int64") - d["_is_top3"].astype("Int64")
    d["jockey_win_rate"] = np.where(d["jockey_n_prev"] > 0, d["jockey_wins_prev"] / d["jockey_n_prev"], np.nan)
    d["jockey_top3_rate"] = np.where(d["jockey_n_prev"] > 0, d["jockey_top3_prev"] / d["jockey_n_prev"], np.nan)

    out_cols = ["index", "course_n_prev", "course_win_rate", "course_top3_rate",
                "jockey_n_prev", "jockey_win_rate", "jockey_top3_rate"]
    result = d[out_cols].set_index("index")
    # scratch行はそもそも母集団に存在しない(course/jockey特徴は定義されない=NaN)。
    full_result = result.reindex(df.index)
    return full_result


def compute_corrected_kako5(df: pd.DataFrame, target_index=None) -> pd.DataFrame:
    """kako5_*(16)+hist_same_cond/place_*(4)を、意味定義spec通りに計算する。
    scratchはwindow slotを一切消費しない(母集団除外)。DNFは直近1走として
    window slotへ含めるが、着順・上り3Fは欠損(None)、TD/距離/場所は実値を
    保持する(既存parse_kako5.py:_compute_features()のロジックは変更せず
    そのままimportして使う)。
    target_index: 計算対象行のindexを絞る場合に指定(高速化用)。Noneなら全行。
    """
    sys.path.insert(0, str(BASE))
    from parse_kako5 import _compute_features, _safe_float, _safe_int, KAKO5_COLS, HIST_COLS  # noqa: E402

    df = df.copy()
    exp_mask = experience_population(df)
    d = df.loc[exp_mask].copy()
    td_map = {"芝": "T", "ダ": "D", "ダート": "D", "T": "T", "D": "D"}
    d["_td_code"] = d[COL_SURFACE].map(td_map).fillna("")
    d["_dist"] = pd.to_numeric(d[COL_DIST], errors="coerce")
    d["_place"] = d[COL_PLACE].astype(str)
    d = d.sort_values([COL_PEDIGREE, "date_dt"])

    targets = df.index if target_index is None else target_index
    target_set = set(targets)

    rows_out = []
    for horse_id, group in d.groupby(COL_PEDIGREE, sort=False):
        idxs = group.index.tolist()
        if target_index is not None and not (target_set & set(idxs)):
            continue
        for seq_i, idx in enumerate(idxs):
            if idx not in target_set:
                continue
            row = group.loc[idx]
            past_indices = idxs[max(0, seq_i - 5): seq_i]
            past_races = []
            for pi in reversed(past_indices):
                pr = group.loc[pi]
                is_dnf = pr["_finish_class"] == "dnf"
                past_races.append({
                    "着順": None if is_dnf else _safe_int(pr.get(COL_FINISH)),
                    "人気": None,
                    "上り3F": None if is_dnf else _safe_float(pr.get("前走上り3F")),
                    "TD": td_map.get(str(pr.get(COL_SURFACE, "")), ""),
                    "距離": _safe_float(pr.get(COL_DIST)),
                    "場所": str(pr.get(COL_PLACE, "")),
                })
            feats = _compute_features(
                past_races, current_td=row.get("_td_code"),
                current_dist=_safe_float(row.get("_dist")), current_place=row.get("_place"))
            cur_td, cur_dist, cur_place = row.get("_td_code", ""), _safe_float(row.get("_dist")), row.get("_place", "")
            for hcol in HIST_COLS:
                feats[hcol] = np.nan
            all_idxs = group.index.tolist()
            if seq_i > 0 and cur_td and cur_dist is not None:
                same_cond_pos = []
                for pi_seq in range(seq_i):
                    pr = group.loc[all_idxs[pi_seq]]
                    if pr["_finish_class"] != "numeric" and pr["_finish_class"] != "dnf":
                        continue
                    pos = _safe_int(pr.get(COL_FINISH))
                    if pos is None:
                        continue
                    td_i = td_map.get(str(pr.get(COL_SURFACE, "")), "")
                    dist_i = _safe_float(pr.get(COL_DIST))
                    if td_i == cur_td and dist_i is not None and abs(dist_i - cur_dist) <= 200:
                        same_cond_pos.append(pos)
                if same_cond_pos:
                    feats["hist_same_cond_best_pos"] = min(same_cond_pos)
                    feats["hist_same_cond_top3_rate"] = sum(1 for p in same_cond_pos if p <= 3) / len(same_cond_pos)
                    feats["hist_same_cond_count"] = len(same_cond_pos)
            if seq_i > 0 and cur_place:
                same_place_pos = []
                for pi_seq in range(seq_i):
                    pr = group.loc[all_idxs[pi_seq]]
                    pos = _safe_int(pr.get(COL_FINISH))
                    if pos is None:
                        continue
                    if str(pr.get(COL_PLACE, "")) == cur_place:
                        same_place_pos.append(pos)
                if same_place_pos:
                    feats["hist_same_place_best_pos"] = min(same_place_pos)
            feats["_idx"] = idx
            rows_out.append(feats)

    out = pd.DataFrame(rows_out).set_index("_idx") if rows_out else pd.DataFrame(
        columns=KAKO5_COLS + HIST_COLS)
    return out.reindex(df.index if target_index is None else list(target_index))
