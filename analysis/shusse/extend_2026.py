# -*- coding: utf-8 -*-
"""
analysis/shusse/extend_2026.py — 出世予測 AI v2: 2026 年の走を starts_v2 に連結する
=====================================================================================
master は 2025-12-28 で終わる。`data/kekka/2026*.csv`（確定着順・全馬）と
`data/weekly/2026*.csv`（レース行: クラス名・芝ダ・距離・場所、馬行: 性別・年齢・騎手・
調教師・父・母父・タイプ・前走着差・前走上り3F・前走上り3F順位・馬体重[49 列形式のみ]）
から `starts_v2.parquet` と同じ列の 2026 行を作って連結し、現役馬の推論に使う。

実行（root から）:
    venv311\\Scripts\\python.exe -m analysis.shusse.extend_2026

出力（analysis/shusse/out_v2/live/）:
    starts_ext.parquet        starts_v2 + 2026 平地行（k / days_since_prev / jky as-of を 2026 行で再計算）
    horses_ext.parquet        horses_v2 + 2026 デビュー馬（既存馬は n_starts / last_date のみ更新）
    extend_2026_report.json   取り込み件数・結合率・欠損率・除外理由

設計上の決定（docs/research/SHUSSE_AI_V2_RESULTS_20261031.md §11 に記載）:
  - 血統登録番号は data/_horse_history.parquet の 2026 行（馬名+日付+race_id）から引く。
    無い行は馬名+父で master を一意に引ければその番号、無ければ build_horse_history.synthetic_id
    と同じ合成 ID（負値）。
  - 障害: 週次の 芝・ダート に「障」、クラス名 ＪＧ、レース名「ジャンプ/障害」で除外。
    3 月以降の週次は障害レースを出力していないため「kekka にあって週次に無いレース」も障害として除外
    （2026 の該当 50 レースは全て土曜 R1/R4/R8 等の障害枠、頭数 7〜14）。
  - 出走頭数 = kekka の 確定着順 > 0 の頭数（取消・除外・中止は 0）。
  - margin / agari3f / agari_rank は自走行に無いので、その馬の **次の出走の週次馬行** の
    前走着差・前走上り3F・前走上り3F順位 で埋める（前走日付と着順が一致する行のみ）。
    週次の 前走着差 は 2025 年末 3,966 行で master の margin と一致（上り3F・順位も 100%）。
    ただし勝ち馬は 2 着との差が負で入る（master は 0）ので 0 で下限クリップする。
  - 馬体重は 49 列形式（1〜2 月）の週次にしか無い。46 列形式は NaN。
  - 調教師年齢・騎手年齢・jockey_top3_rate（馬×騎手ペア率）は NaN。
  - 騎手 as-of 3 着内率は master+2026 の明細で 2026 行だけ再計算（縮約の事前分布 g は master の値）。
    master 行は一切変更しない（tests/test_shusse_live.py で同一性を検査）。
  - 真のデビュー: 初出走行の 前走 が空欄 & 年齢 2〜3（学習と同じ定義）。地方・海外転入馬は週次の
    前走 列に 園田/川崎/笠松 等が入る（2026 で 3,845 行）ので、前走空欄の年齢 3 未勝利デビューも真の
    デビューに含める（学習側でも年齢 3 の真のデビューの半数は未勝利戦デビュー）。
"""
from __future__ import annotations

import json
import re
import sys
import time
import unicodedata
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.shusse.build_career import COL_HORSE, COL_RACE_ID, RACE_LEVEL_NFKC, TARGET_LEVEL_OF_RACE
from predict_weekly import (HORSE_COLS_33, HORSE_COLS_44, HORSE_COLS_46, HORSE_COLS_48, HORSE_COLS_49,
                            HORSE_COLS_99, RACE_COLS)

BASE = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "out_v2"
OUT_LIVE = OUT / "live"
OUT_LIVE.mkdir(parents=True, exist_ok=True)
WEEKLY_DIR = BASE / "data" / "weekly"
KEKKA_DIR = BASE / "data" / "kekka"
HORSE_HISTORY = BASE / "data" / "_horse_history.parquet"
CODE_MAPS = BASE / "data" / "serve_code_maps.json"

_HORSE_COLS = {33: HORSE_COLS_33, 44: HORSE_COLS_44, 46: HORSE_COLS_46, 48: HORSE_COLS_48,
               49: HORSE_COLS_49, 99: HORSE_COLS_99}
_SURF = {"芝": "芝", "ダ": "ダ", "ダート": "ダ"}
JKY_WINDOW_DAYS, JKY_M = 730, 50.0
UNKNOWN_CODE = -1


def _log(msg: str) -> None:
    print(msg, flush=True)


def _clean_name(s) -> str:
    """build_horse_history._clean_name と同一（合成 ID の整合のため）。"""
    return str(s).replace("　", "").strip()


def synthetic_id(name: str, sire: str) -> int:
    """build_horse_history.synthetic_id と同一。"""
    return -int(zlib.crc32(f"{name}|{sire}".encode("utf-8")))


def _clean_sire(s) -> str | None:
    """週次の 父/母父 は外国馬名が '"Gun Runner"' と引用符付き。master は引用符無し。"""
    s = _clean_name(s).strip('"').strip()
    return s or None


def _num(x) -> float:
    try:
        return float(str(x).strip())
    except ValueError:
        return np.nan


def _nfkc(s) -> str:
    return unicodedata.normalize("NFKC", str(s)).strip()


# --------------------------------------------------------------------------- weekly
def weekly_files() -> list[Path]:
    return sorted(p for p in WEEKLY_DIR.glob("2026*.csv") if re.fullmatch(r"\d{8}\.csv", p.name))


def kekka_files() -> list[Path]:
    return sorted(p for p in KEKKA_DIR.glob("2026*.csv") if re.fullmatch(r"\d{8}\.csv", p.name))


def parse_weekly(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """レース行（19 列）と馬行（33/44/46/48/49/99 列）を分けて返す。"""
    text = None
    for enc in ("cp932", "shift_jis", "utf-8"):
        try:
            text = path.read_bytes().decode(enc)
            break
        except Exception:
            continue
    if text is None:
        return pd.DataFrame(), pd.DataFrame(), {"unreadable": True}
    ymd = int(path.stem)
    races, horses, widths = [], [], {}
    cur = None
    for line in text.splitlines():
        cols = line.split(",")
        if cols[0] in ("レースID(新)", "枠番", "番", ""):
            continue
        if len(cols) == 19:
            cur = dict(zip(RACE_COLS, cols))
            cur["rid16"] = cur["レースID(新)"].strip()[:16]
            cur["ymd"] = ymd
            races.append(cur)
        elif cur is not None and len(cols) in _HORSE_COLS:
            h = dict(zip(_HORSE_COLS[len(cols)], cols))
            h["rid16"] = cur["rid16"]
            h["ymd"] = ymd
            h["ncols"] = len(cols)
            horses.append(h)
        elif cur is not None and len(cols) > 1:
            widths[len(cols)] = widths.get(len(cols), 0) + 1
    return pd.DataFrame(races), pd.DataFrame(horses), {"unmatched_widths": widths}


def load_weekly_all() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    R, H, rep = [], [], {"files": 0, "unmatched_widths": {}}
    for p in weekly_files():
        r, h, info = parse_weekly(p)
        rep["files"] += 1
        for w, n in info.get("unmatched_widths", {}).items():
            rep["unmatched_widths"][str(w)] = rep["unmatched_widths"].get(str(w), 0) + n
        if len(r):
            R.append(r)
        if len(h):
            H.append(h)
    races = pd.concat(R, ignore_index=True)
    horses = pd.concat(H, ignore_index=True)
    horses["name"] = horses["馬名S"].map(_clean_name)
    horses["umaban"] = pd.to_numeric(horses["馬番"], errors="coerce")
    for c in ("馬体重", "前走着差", "前走上り3F", "前走上り3F順位", "前走月", "前走日", "年齢"):
        if c in horses.columns:
            horses[c] = pd.to_numeric(horses[c].astype(str).str.strip(), errors="coerce")
        else:
            horses[c] = np.nan
    horses["prev_pos_num"] = pd.to_numeric(horses["前走着順"].astype(str).str.strip(), errors="coerce")
    horses["prev_blank"] = (horses["前走月"].isna() & horses["前走レース名"].astype(str).str.strip().eq("")
                            & horses["前走着順"].astype(str).str.strip().eq(""))
    # 前走の日付（年は、前走月 > 今走月 なら前年）
    m = horses["ymd"] // 100 % 100
    y = horses["ymd"] // 10000
    py = np.where(horses["前走月"] > m, y - 1, y)
    horses["prev_date"] = pd.to_datetime(
        pd.DataFrame({"year": py, "month": horses["前走月"], "day": horses["前走日"]}), errors="coerce")
    rep["race_rows"] = int(len(races))
    rep["horse_rows"] = int(len(horses))
    rep["horse_rows_by_width"] = {int(k): int(v) for k, v in horses["ncols"].value_counts().items()}
    return races, horses, rep


# --------------------------------------------------------------------------- kekka
def load_kekka_2026() -> tuple[pd.DataFrame, dict]:
    K = []
    for p in kekka_files():
        d = pd.read_csv(p, encoding="cp932", dtype=str)
        if "レースID(新)" not in d.columns or "確定着順" not in d.columns:
            _log(f"  skip {p.name}: 列不足")
            continue
        d["ymd"] = int(p.stem)
        K.append(d)
    k = pd.concat(K, ignore_index=True)
    k["rid16"] = k["レースID(新)"].astype(str).str.strip().str[:16]
    k["umaban"] = pd.to_numeric(k["馬番"], errors="coerce")
    k["pos"] = pd.to_numeric(k["確定着順"], errors="coerce")
    k["name"] = k["馬名"].map(_clean_name)
    k["場所"] = k["場所"].astype(str).str.strip()
    k["R"] = pd.to_numeric(k["Ｒ"], errors="coerce").astype(int)
    rep = {"files": len(K), "rows": int(len(k)), "rows_pos0_or_nan(取消/除外/中止)": int((~(k["pos"] > 0)).sum()),
           "date_from_filename_equals_日付": bool(((k["日付"].astype(int) + 20000000) == k["ymd"]).all())}
    k = k[k["pos"] > 0].copy()
    k["field"] = k.groupby("rid16")["pos"].transform("size").astype(int)
    rep["rows_ran"] = int(len(k))
    rep["races"] = int(k["rid16"].nunique())
    rep["dup_rid16_umaban"] = int(k.duplicated(["rid16", "umaban"]).sum())
    return k, rep


# --------------------------------------------------------------------------- jump
def jump_reason(surface: str, class_name: str, race_name: str) -> str | None:
    if "障" in str(surface):
        return "surface_障"
    if _nfkc(class_name).upper().startswith("JG"):
        return "class_JG"
    if "ジャンプ" in str(race_name) or "障害" in str(race_name):
        return "racename_jump"
    return None


# --------------------------------------------------------------------------- ped ids
def resolve_ped_ids(df: pd.DataFrame, horses_master: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """1) _horse_history 2026 行 (name, date, race_id) 2) 馬名+父で master 一意 3) 合成 ID。"""
    hh = pd.read_parquet(HORSE_HISTORY)
    hh = hh[hh["src"] == "kekka2026"][["name", "date", "race_id", "ped_id"]].copy()
    hh["name"] = hh["name"].map(_clean_name)
    hh["date"] = hh["date"].astype(int)
    hh = hh.drop_duplicates(["name", "date", "race_id"])
    m = df.merge(hh, left_on=["name", "ymd", "rid16"], right_on=["name", "date", "race_id"], how="left")
    assert len(m) == len(df)
    src = np.where(m["ped_id"].notna(), "horse_history", "")
    # 2) 馬名 + 父 で master を一意に引く（2024 年以降に走った馬のみ候補）
    recent = horses_master[horses_master["last_date"] >= "2023-01-01"]
    by_name_sire = recent.groupby(["馬名", "sire"])[COL_HORSE].agg(list)
    need = m["ped_id"].isna()
    for i in np.where(need)[0]:
        key = (m.at[i, "name"], m.at[i, "sire_clean"])
        cands = by_name_sire.get(key, [])
        if len(cands) == 1:
            m.at[i, "ped_id"] = cands[0]
            src[i] = "master_name_sire"
        else:
            m.at[i, "ped_id"] = synthetic_id(m.at[i, "name"], _clean_name(m.at[i, "父"]))
            src[i] = "synthetic"
    m["ped_id"] = m["ped_id"].astype("int64")
    m["ped_src"] = src
    # 同じ馬は 2026 の全行で同一 ID であるべき。馬名ごとに複数 ID が出たら多数決で統一
    multi = m.groupby("name")["ped_id"].nunique()
    fix_names = multi[multi > 1].index
    n_fixed = 0
    for nm in fix_names:
        sub = m[m["name"] == nm]
        pos_ids = sub.loc[sub["ped_id"] > 0, "ped_id"]
        target = int(pos_ids.mode().iloc[0]) if len(pos_ids) else int(sub["ped_id"].mode().iloc[0])
        n_fixed += int((sub["ped_id"] != target).sum())
        m.loc[m["name"] == nm, "ped_id"] = target
    rep = {"ped_src_counts": {k: int(v) for k, v in pd.Series(src).value_counts().items()},
           "rows_unified_within_name": n_fixed,
           "horses_total": int(m["ped_id"].nunique()),
           "horses_in_master": int(m.loc[m["ped_id"] > 0, "ped_id"].nunique()),
           "horses_synthetic": int(m.loc[m["ped_id"] < 0, "ped_id"].nunique())}
    return m.drop(columns=["date", "race_id"]), rep


# --------------------------------------------------------------------------- build 2026 rows
def build_2026_rows(starts_master: pd.DataFrame, horses_master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    t0 = time.time()
    races, wh, rep_w = load_weekly_all()
    kekka, rep_k = load_kekka_2026()
    rep: dict = {"weekly": rep_w, "kekka": rep_k}
    maps = json.loads(CODE_MAPS.read_text(encoding="utf-8"))
    jky_map, trn_map = maps["jockey"], maps["trainer"]

    races = races.drop_duplicates("rid16").set_index("rid16")
    races["jump_reason"] = [jump_reason(s, c, n) for s, c, n in zip(races["芝・ダート"], races["クラス名"], races["レース名"])]
    rep["weekly_races"] = int(len(races))
    rep["weekly_jump_races"] = {k: int(v) for k, v in races["jump_reason"].value_counts().items()}

    # kekka × weekly 馬行（rid16, 馬番）
    wh_k = wh[wh["rid16"].isin(kekka["rid16"])].drop_duplicates(["rid16", "umaban"])
    m = kekka.merge(wh_k.drop(columns=["ymd"]), on=["rid16", "umaban"], how="left", suffixes=("", "_w"))
    assert len(m) == len(kekka)
    in_weekly_race = m["rid16"].isin(races.index)
    has_horse_row = m["name_w"].notna()
    name_ok = has_horse_row & (m["name"] == m["name_w"])
    rep["join"] = {
        "kekka_ran_rows": int(len(m)),
        "race_in_weekly": int(in_weekly_race.sum()),
        "races_kekka_only(assumed_jump)": int((~kekka["rid16"].isin(races.index)).groupby(kekka["rid16"]).first().sum()),
        "horse_row_matched": int(has_horse_row.sum()),
        "horse_row_name_agree": int(name_ok.sum()),
        "horse_row_name_disagree": int((has_horse_row & ~name_ok).sum()),
        "in_weekly_race_but_no_horse_row": int((in_weekly_race & ~has_horse_row).sum()),
    }
    # 除外: 週次に無いレース（障害）, 週次で障害判定, 馬行が無い/馬名不一致
    m = m[in_weekly_race].copy()
    jr = m["rid16"].map(races["jump_reason"])
    rep["excluded_jump_rows_by_reason"] = {k: int(v) for k, v in jr.value_counts().items()}
    m = m[jr.isna()].copy()
    ok = (m["name"] == m["name_w"])
    rep["excluded_flat_rows_without_horse_row_or_name_mismatch"] = int((~ok).sum())
    m = m[ok].copy()

    # レース属性
    rc = races.reindex(m["rid16"])
    m["場所"] = rc["場所"].astype(str).str.strip().values
    m["レース名"] = rc["レース名"].astype(str).str.strip().values
    m["クラス名"] = rc["クラス名"].astype(str).str.strip().values
    m["芝・ダ"] = rc["芝・ダート"].astype(str).str.strip().map(_SURF).values
    m["距離"] = pd.to_numeric(rc["距離"], errors="coerce").values
    m["class_norm"] = m["クラス名"].map(_nfkc)
    m["race_level"] = m["class_norm"].map(RACE_LEVEL_NFKC)
    bad_cls = m["race_level"].isna()
    rep["unmapped_class_names"] = {k: int(v) for k, v in m.loc[bad_cls, "クラス名"].value_counts().items()}
    m = m[~bad_cls].copy()
    rep["surface_values_after_norm"] = {k: int(v) for k, v in m["芝・ダ"].value_counts(dropna=False).items()}
    rep["place_values"] = sorted(m["場所"].unique().tolist())
    assert set(m["場所"].unique()) <= set(starts_master["場所"].unique()), m["場所"].unique()
    assert set(m["芝・ダ"].dropna().unique()) <= {"芝", "ダ"}

    # 馬属性
    m["sire_clean"] = m["父"].map(_clean_sire)
    m["bms_clean"] = m["母父"].map(_clean_sire)
    m["sire_type"] = m["父タイプ"].map(_clean_name).replace("", np.nan)
    m["bms_type"] = m["母父タイプ"].map(_clean_name).replace("", np.nan)
    m["sex"] = m["性別"].map(_clean_name)
    assert set(m["sex"].unique()) <= set(starts_master["性別"].unique()), m["sex"].unique()
    m["jky_code"] = m["騎手"].map(_clean_name).map(jky_map)
    m["trn_code"] = m["調教師"].map(_clean_name).map(trn_map)
    rep["jockey_code_mapped_rate"] = round(float(m["jky_code"].notna().mean()), 5)
    rep["trainer_code_mapped_rate"] = round(float(m["trn_code"].notna().mean()), 5)
    rep["trainer_unmapped_top"] = {k: int(v) for k, v in m.loc[m["trn_code"].isna(), "調教師"].value_counts().head(10).items()}
    master_sires = set(starts_master["種牡馬"].dropna().unique())
    rep["sire_name_in_master_rate"] = round(float(m["sire_clean"].isin(master_sires).mean()), 5)
    rep["sire_type_missing_rate_rows"] = round(float(m["sire_type"].isna().mean()), 5)
    rep["bms_type_missing_rate_rows"] = round(float(m["bms_type"].isna().mean()), 5)
    rep["weight_available_rate_rows"] = round(float(m["馬体重"].notna().mean()), 5)
    # 前走確定着順: 前走が空欄なら NaN（デビュー）。前走があるが着順が数値でない（取消等）は 0
    m["前走確定着順"] = np.where(m["prev_blank"], np.nan, m["prev_pos_num"].fillna(0))

    m, rep_p = resolve_ped_ids(m, horses_master)
    rep["ped"] = rep_p

    # master に居る馬は 生産者 / birth_date を master から
    hm = horses_master.set_index(COL_HORSE)
    m["breeder"] = hm["breeder"].reindex(m["ped_id"]).values
    m["birth_date"] = hm["birth_date"].reindex(m["ped_id"]).values
    nj = starts_master.groupby(COL_HORSE)["n_jump_starts"].last()
    m["n_jump_starts_master"] = nj.reindex(m["ped_id"]).values

    # 次走の週次馬行から 着差・上り3F・上り3F順位 を埋める（全週次ファイル、kekka 未配置の先の週も含む）
    nxt = wh.dropna(subset=["prev_date"])[["name", "prev_date", "前走着差", "前走上り3F", "前走上り3F順位", "prev_pos_num"]].copy()
    nxt = nxt.drop_duplicates(["name", "prev_date"], keep="first")
    nxt.columns = ["name", "nx_date", "nx_margin", "nx_agari", "nx_agari_rank", "nx_pos"]
    m["date"] = pd.to_datetime(m["ymd"].astype(str), format="%Y%m%d")
    m = m.merge(nxt, left_on=["name", "date"], right_on=["name", "nx_date"], how="left")
    found = m["nx_date"].notna()
    agree = found & (m["nx_pos"] == m["pos"])
    rep["margin_fill"] = {"rows": int(len(m)), "next_start_row_found": int(found.sum()),
                          "pos_agree": int(agree.sum()), "pos_disagree_nulled": int((found & ~agree).sum()),
                          "fill_rate": round(float(agree.mean()), 5)}
    m["margin"] = np.where(agree, m["nx_margin"].clip(lower=0), np.nan)
    m["agari3f"] = np.where(agree, m["nx_agari"], np.nan)
    m["agari_rank"] = np.where(agree, m["nx_agari_rank"], np.nan)
    rep["margin_fill"]["margin_describe"] = {k: round(float(v), 3) for k, v in pd.Series(m["margin"]).describe().items()}

    out = pd.DataFrame({
        "日付": m["ymd"].astype(int),
        "場所": m["場所"],
        "レース名": m["レース名"],
        COL_RACE_ID: m["rid16"].astype("int64"),
        COL_HORSE: m["ped_id"].astype("int64"),
        "馬名": m["name"],
        "芝・ダ": m["芝・ダ"],
        "距離": m["距離"].astype(int),
        "クラス名": m["クラス名"],
        "トラックコード(JV)": np.nan,
        "前走確定着順": m["前走確定着順"].astype(float),
        "種牡馬": m["sire_clean"],
        "父タイプ名": m["sire_type"],
        "母父馬": m["bms_clean"],
        "母父タイプ名": m["bms_type"],
        "騎手コード": m["jky_code"].fillna(UNKNOWN_CODE).astype(int),
        "調教師コード": m["trn_code"].fillna(UNKNOWN_CODE).astype(int),
        "生産者": m["breeder"],
        "年齢限定": np.nan,
        "指定条件": np.nan,
        "性別": m["sex"],
        "年齢": m["年齢"].astype(int),
        "着順": m["pos"].astype(float),
        "出走頭数": m["field"].astype(int),
        "騎手年齢": np.nan,
        "調教師年齢": np.nan,
        "jockey_top3_rate": np.nan,
        "date": m["date"],
        "year": (m["ymd"] // 10000).astype(int),
        "class_norm": m["class_norm"],
        "race_level": m["race_level"].astype(int),
        "is_jump": False,
        "pos": m["pos"].astype(int),
        "field": m["field"].astype(int),
        "R": m["R"].astype(int),
        "time_sec": np.nan,
        "win_time": np.nan,
        "margin": m["margin"].astype(float),
        "agari3f": m["agari3f"].astype(float),
        "agari_rank": m["agari_rank"].astype(float),
        "weight": m["馬体重"].astype(float),
        "birth_date": pd.to_datetime(m["birth_date"]),
    })
    out["win"] = (out["pos"] == 1).astype(int)
    out["top3"] = (out["pos"] <= 3).astype(int)
    out["rel_pos"] = (out["pos"] - 1) / (out["field"] - 1).clip(lower=1)
    out["target_level"] = out["race_level"].map(TARGET_LEVEL_OF_RACE).astype(int)
    out["n_jump_starts"] = pd.to_numeric(m["n_jump_starts_master"], errors="coerce").fillna(0).astype(int).values
    out["src"] = "weekly2026"
    out = out.sort_values([COL_HORSE, "date", COL_RACE_ID]).reset_index(drop=True)
    dup = out.duplicated([COL_HORSE, "date"]).sum()
    rep["rows_2026_flat"] = int(len(out))
    rep["dup_horse_date_rows"] = int(dup)
    if dup:
        out = out.drop_duplicates([COL_HORSE, "date"], keep="first")
    rep["missing_rate_2026_rows"] = {c: round(float(out[c].isna().mean()), 5) for c in
                                    ("margin", "agari3f", "agari_rank", "weight", "父タイプ名", "母父タイプ名", "生産者", "birth_date", "前走確定着順")}
    rep["unknown_code_rate"] = {"騎手コード": round(float((out["騎手コード"] == UNKNOWN_CODE).mean()), 5),
                                "調教師コード": round(float((out["調教師コード"] == UNKNOWN_CODE).mean()), 5)}
    rep["elapsed_build_sec"] = round(time.time() - t0, 1)
    return out, m, rep


# --------------------------------------------------------------------------- extend
def jockey_asof_new_rows(flat_all: pd.DataFrame, new_mask: np.ndarray, g: float) -> tuple[np.ndarray, np.ndarray]:
    """build_career_v2.add_jockey_asof と同じ定義（直近 730 日、当日除く、縮約 m=50）を
    new_mask 行だけに計算する。g は master の全体 3 着内率（学習時と同じ事前分布）。"""
    jd = flat_all.groupby(["騎手コード", "date"]).agg(n=("top3", "size"), t3=("top3", "sum")).reset_index().sort_values(["騎手コード", "date"])
    jd["cum_n"] = jd.groupby("騎手コード")["n"].cumsum()
    jd["cum_t3"] = jd.groupby("騎手コード")["t3"].cumsum()
    ev = jd[["騎手コード", "date", "cum_n", "cum_t3"]].rename(columns={"date": "t"}).sort_values("t")
    q = flat_all.loc[new_mask, ["騎手コード", "date"]].copy()
    q["row"] = np.arange(len(q))
    q["t"] = q["date"]
    a = pd.merge_asof(q.sort_values("t"), ev, on="t", by="騎手コード", allow_exact_matches=False, direction="backward").sort_values("row")
    q2 = q.copy()
    q2["t"] = q2["date"] - pd.Timedelta(days=JKY_WINDOW_DAYS)
    b = pd.merge_asof(q2.sort_values("t"), ev, on="t", by="騎手コード", allow_exact_matches=True, direction="backward").sort_values("row")
    n = a["cum_n"].fillna(0).values - b["cum_n"].fillna(0).values
    t3 = a["cum_t3"].fillna(0).values - b["cum_t3"].fillna(0).values
    return n, (t3 + JKY_M * g) / (n + JKY_M)


def extend(starts_master: pd.DataFrame, horses_master: pd.DataFrame, new_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    sm = starts_master.copy()
    sm["src"] = "master"
    cols = sm.columns.tolist()
    nr = new_rows.reindex(columns=cols)
    allf = pd.concat([sm, nr], ignore_index=True).sort_values([COL_HORSE, "date", COL_RACE_ID], kind="stable").reset_index(drop=True)
    new = (allf["src"] == "weekly2026").values
    g = allf.groupby(COL_HORSE)
    k_new = (g.cumcount() + 1).values
    dsp_new = (allf["date"] - g["date"].shift(1)).dt.days.values
    # master 行は変更しない（k は構成上同じになるはず。検査）
    assert (allf.loc[~new, "k"].values == k_new[~new]).all(), "master k changed"
    allf.loc[new, "k"] = k_new[new]
    allf.loc[new, "days_since_prev"] = dsp_new[new]
    allf["k"] = allf["k"].astype(int)
    g_prior = float(starts_master["top3"].mean())
    n2y, r2y = jockey_asof_new_rows(allf, new, g_prior)
    allf.loc[new, "jky_n_2y"] = n2y
    allf.loc[new, "jky_top3_2y"] = r2y
    # 不明コード (-1) の騎手は事前分布に落とす
    unk = new & (allf["騎手コード"].values == UNKNOWN_CODE)
    allf.loc[unk, "jky_n_2y"] = 0.0
    allf.loc[unk, "jky_top3_2y"] = g_prior

    # horses: 既存馬は n_starts / last_date 更新、新馬は build_horses 相当
    gh = allf.groupby(COL_HORSE)
    h_old = horses_master.set_index(COL_HORSE).copy()
    h_old["n_starts"] = gh.size().reindex(h_old.index).values
    h_old["last_date"] = gh["date"].max().reindex(h_old.index).values
    new_ids = sorted(set(allf.loc[new, COL_HORSE]) - set(h_old.index))
    fn = allf[allf[COL_HORSE].isin(new_ids)]
    first = fn.drop_duplicates(COL_HORSE, keep="first").set_index(COL_HORSE)
    st = allf.dropna(subset=["父タイプ名"]).groupby("種牡馬")["父タイプ名"].agg(lambda x: x.value_counts().index[0])
    bt = allf.dropna(subset=["母父タイプ名"]).groupby("母父馬")["母父タイプ名"].agg(lambda x: x.value_counts().index[0])
    gn = fn.groupby(COL_HORSE)
    h_new = pd.DataFrame({
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
        "n_starts": gn.size(),
        "last_date": gn["date"].max(),
        "birth_date": gn["birth_date"].first(),
    })
    # 合成 ID × 年齢 3 × 初出走が新馬戦でない馬（未勝利デビュー）。地方・海外転入なら週次の 前走 列に
    # 園田/川崎/笠松 等が入る（2026 で 3,845 行）ので、前走空欄は真の未出走と判断できる。学習側でも
    # 年齢 3 の真のデビューの半数（2020 年以降 4,318/8,603 頭）は未勝利戦デビューなので除外しない（件数のみ記録）。
    amb = h_new["true_debut"] & (h_new.index.to_series() < 0) & (h_new["debut_age"] == 3) & (first["class_norm"] != "新馬")
    h_new["debut_dayage"] = (h_new["debut_date"] - h_new["birth_date"]).dt.days
    h_new["debut_dayage"] = h_new["debut_dayage"].where(h_new["debut_dayage"].between(500, 1500))
    horses_ext = pd.concat([h_old, h_new]).reset_index().rename(columns={"index": COL_HORSE})
    horses_ext["debut_year"] = horses_ext["debut_year"].astype(int)
    horses_ext["trainer_debut"] = horses_ext["trainer_debut"].astype(int)
    rep = {
        "rows_total": int(len(allf)), "rows_master": int((~new).sum()), "rows_2026": int(new.sum()),
        "data_end_ext": str(allf["date"].max().date()),
        "horses_total": int(len(horses_ext)), "horses_new_2026": int(len(h_new)),
        "horses_new_true_debut": int(h_new["true_debut"].sum()),
        "horses_new_age3_first_race_not_shinba(kept)": int(amb.sum()),
        "horses_new_debut_age_counts": {int(k): int(v) for k, v in h_new["debut_age"].value_counts().items()},
        "horses_new_sire_type_missing": round(float(h_new["sire_type"].isna().mean()), 5),
        "jky_prior_g": g_prior,
        "jky_unknown_code_rows_2026": int(unk.sum()),
    }
    return allf, horses_ext, rep


def main() -> None:
    T0 = time.time()
    starts_master = pd.read_parquet(OUT / "starts_v2.parquet")
    horses_master = pd.read_parquet(OUT / "horses_v2.parquet")
    new_rows, _raw, rep = build_2026_rows(starts_master, horses_master)
    allf, horses_ext, rep_e = extend(starts_master, horses_master, new_rows)
    rep["extend"] = rep_e
    rep["elapsed_sec"] = round(time.time() - T0, 1)
    allf.to_parquet(OUT_LIVE / "starts_ext.parquet", index=False)
    horses_ext.to_parquet(OUT_LIVE / "horses_ext.parquet", index=False)
    (OUT_LIVE / "extend_2026_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _log(json.dumps(rep, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
