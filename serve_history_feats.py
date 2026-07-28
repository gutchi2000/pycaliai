# -*- coding: utf-8 -*-
"""
serve_history_feats.py
======================
serve skew 残差回収: 週次出走表 (馬名のみ・血統登録番号なし) に対し、
data/_horse_history.parquet (build_horse_history.py 生成) を馬名で JOIN し、
レース日より厳密に前の走のみ (leak-safe as-of) で学習と同一定義の
以下 10 特徴 + 騎手/調教師コードを再計算して埋める。

学習側の定義 (同一実装であること):
  hist_same_cond_*        parse_kako5.build_from_master: 同TD(芝/ダ) かつ
                          距離±200m の全キャリア着順から best/top3率/回数。
                          過去走ゼロ or 着順全NaN → NaN
  hist_same_place_best_pos 同場所の全キャリア最高着順。なければ NaN
  course_*                build_master_v2.compute_history_features:
                          course_key = 場所|芝ダ|距離帯(短≤1400/マ≤1700/中≤2200/長)。
                          n_prev = 過去同 key 走数 (着順NaN含む, 初出走=0)、
                          win/top3 rate = n_prev>0 のときのみ (else NaN)
  jockey_*                馬×騎手コード ペアの累積 (騎手単独ではない)。
                          n_prev = 過去同騎手騎乗数 (初コンビ=0)、rate 同上
  騎手コード/調教師コード   カテゴリ特徴。serve_code_maps.json の名前→コードで復元

同名馬の曖昧性解消: 父名 (種牡馬) 一致 or 生年 (レース年-年齢) ±1 一致。
解消不能 (複数候補が残る) → 従来通り NaN (安全側)。
未知の馬名 (新馬等) → 学習と同じく n_prev=0 / rate,hist は NaN。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE = Path(__file__).parent
HIST_PARQUET = BASE / "data" / "_horse_history.parquet"
CODE_MAPS = BASE / "data" / "serve_code_maps.json"

# 埋める特徴 (v6 feature_cols 名)
NUM_FEATS = [
    "hist_same_cond_best_pos", "hist_same_cond_top3_rate",
    "hist_same_cond_count", "hist_same_place_best_pos",
    "course_n_prev", "course_win_rate", "course_top3_rate",
    "jockey_n_prev", "jockey_win_rate", "jockey_top3_rate",
]
CAT_FEATS = ["騎手コード", "調教師コード"]

_SURF = {"芝": "芝", "ダ": "ダ", "ダート": "ダ"}


def _dist_band(d) -> str:
    # build_master_v2.compute_history_features と同一
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


def _clean_name(s) -> str:
    return str(s).replace("　", "").strip()


class _HistoryIndex:
    """馬名 → [entity] の索引。entity = 1頭分の全走 (numpy 配列)。"""

    def __init__(self, hist: pd.DataFrame):
        self.by_name: dict[str, list[dict]] = {}
        hist = hist.sort_values(["ped_id", "date"])
        for ped_id, g in hist.groupby("ped_id", sort=False):
            name = str(g["name"].iloc[-1])
            ent = {
                "ped_id": int(ped_id),
                "sire": str(g["sire"].iloc[-1]) if pd.notna(g["sire"].iloc[-1]) else "",
                "birth_year": (int(g["birth_year"].iloc[-1])
                               if pd.notna(g["birth_year"].iloc[-1]) else None),
                "date": g["date"].to_numpy(np.int32),
                "place": g["place"].astype(str).to_numpy(),
                "surface": g["surface"].astype(str).to_numpy(),
                "dist": pd.to_numeric(g["dist"], errors="coerce").to_numpy(np.float64),
                "pos": pd.to_numeric(g["pos"], errors="coerce").to_numpy(np.float64),
                "jockey": pd.to_numeric(g["jockey_code"], errors="coerce").to_numpy(np.float64),
            }
            self.by_name.setdefault(name, []).append(ent)

    def resolve(self, name: str, sire: str, birth_year) -> tuple[dict | None, str]:
        """→ (entity or None, status)  status: hit/new/ambiguous"""
        cands = self.by_name.get(name)
        if not cands:
            return None, "new"
        if len(cands) == 1:
            c = cands[0]
            # 同名の別馬 (名前再利用) ガード: 父・生年が両方分かって両方不一致なら別馬
            sire_known = bool(sire) and bool(c["sire"])
            by_known = birth_year is not None and c["birth_year"] is not None
            sire_ok = (not sire_known) or (c["sire"] == sire)
            by_ok = (not by_known) or abs(c["birth_year"] - int(birth_year)) <= 1
            if (sire_known and not sire_ok) and (by_known and not by_ok):
                return None, "new"
            if (sire_known and not sire_ok) or (by_known and not by_ok):
                # 片方だけ不一致 → 判定不能。安全側で曖昧扱い
                return None, "ambiguous"
            return c, "hit"
        hit = [c for c in cands
               if (sire and c["sire"] and c["sire"] == sire)
               or (birth_year is not None and c["birth_year"] is not None
                   and abs(c["birth_year"] - int(birth_year)) <= 1)]
        if len(hit) > 1:
            hit = [c for c in hit
                   if sire and c["sire"] == sire
                   and birth_year is not None and c["birth_year"] is not None
                   and abs(c["birth_year"] - int(birth_year)) <= 1]
        if len(hit) == 1:
            return hit[0], "hit"
        return None, "ambiguous"


_CACHE: dict = {}


def _load(base: Path):
    key = str(base)
    if key in _CACHE:
        return _CACHE[key]
    hist_path = base / "data" / "_horse_history.parquet"
    maps_path = base / "data" / "serve_code_maps.json"
    if not hist_path.exists():
        raise FileNotFoundError(
            f"{hist_path} 未生成 (python build_horse_history.py を先に実行)")
    hist = pd.read_parquet(hist_path)
    idx = _HistoryIndex(hist)
    maps = {"jockey": {}, "trainer": {}}
    if maps_path.exists():
        with open(maps_path, encoding="utf-8") as f:
            raw = json.load(f)
        maps["jockey"] = raw.get("jockey", {})
        maps["trainer"] = raw.get("trainer", {})
    meta = {"n_rows": len(hist), "max_date": int(hist["date"].max())}
    _CACHE[key] = (idx, maps, meta)
    return _CACHE[key]


def compute_row_feats(ent: dict, race_date: int, place: str, surface: str,
                      dist, jockey_code) -> dict:
    """1頭分: entity の as-of (date < race_date) 走から 10 特徴を計算。"""
    out: dict[str, float] = {}
    if ent is None:
        past = None
    else:
        m = ent["date"] < race_date  # 厳密に前 (同日は除く = leak-safe)
        past = {k: ent[k][m] for k in ("date", "place", "surface", "dist", "pos", "jockey")}

    surface = _SURF.get(str(surface).strip(), "")
    dist_v = float(dist) if pd.notna(dist) else np.nan

    # --- hist_same_cond_* (parse_kako5: 同TD + 距離±200, 着順ありのみ) ---
    if past is not None and len(past["date"]) and surface and not np.isnan(dist_v):
        sel = ((past["surface"] == surface)
               & ~np.isnan(past["dist"]) & (np.abs(past["dist"] - dist_v) <= 200)
               & ~np.isnan(past["pos"]))
        pos = past["pos"][sel]
        if len(pos):
            out["hist_same_cond_best_pos"] = float(pos.min())
            out["hist_same_cond_top3_rate"] = float((pos <= 3).mean())
            out["hist_same_cond_count"] = float(len(pos))
        else:
            out["hist_same_cond_best_pos"] = np.nan
            out["hist_same_cond_top3_rate"] = np.nan
            out["hist_same_cond_count"] = np.nan
    else:
        out["hist_same_cond_best_pos"] = np.nan
        out["hist_same_cond_top3_rate"] = np.nan
        out["hist_same_cond_count"] = np.nan

    # --- hist_same_place_best_pos (同場所, 着順ありのみ) ---
    if past is not None and len(past["date"]) and place:
        sel = (past["place"] == place) & ~np.isnan(past["pos"])
        pos = past["pos"][sel]
        out["hist_same_place_best_pos"] = float(pos.min()) if len(pos) else np.nan
    else:
        out["hist_same_place_best_pos"] = np.nan

    # --- course_* (build_master_v2: 場所|芝ダ|距離帯, NaN着順も走数に含む) ---
    band = _dist_band(dist_v)
    if past is not None:
        if len(past["date"]):
            bands = np.array([_dist_band(d) for d in past["dist"]])
            sel = (past["place"] == place) & (past["surface"] == surface) & (bands == band)
        else:
            sel = np.zeros(0, dtype=bool)
        n = int(sel.sum())
        out["course_n_prev"] = float(n)
        if n > 0:
            pos = past["pos"][sel]
            wins = float(np.nansum(pos == 1))
            top3 = float(np.nansum(pos <= 3))
            out["course_win_rate"] = wins / n
            out["course_top3_rate"] = top3 / n
        else:
            out["course_win_rate"] = np.nan
            out["course_top3_rate"] = np.nan
    else:
        # 未知馬 (新馬): 学習では cumcount=0
        out["course_n_prev"] = 0.0
        out["course_win_rate"] = np.nan
        out["course_top3_rate"] = np.nan

    # --- jockey_* (馬×騎手コード ペア累積) ---
    jc = float(jockey_code) if pd.notna(jockey_code) else np.nan
    if past is not None:
        if not np.isnan(jc) and len(past["date"]):
            sel = past["jockey"] == jc
            n = int(sel.sum())
            out["jockey_n_prev"] = float(n)
            if n > 0:
                pos = past["pos"][sel]
                out["jockey_win_rate"] = float(np.nansum(pos == 1)) / n
                out["jockey_top3_rate"] = float(np.nansum(pos <= 3)) / n
            else:
                out["jockey_win_rate"] = np.nan
                out["jockey_top3_rate"] = np.nan
        elif not np.isnan(jc):
            out["jockey_n_prev"] = 0.0
            out["jockey_win_rate"] = np.nan
            out["jockey_top3_rate"] = np.nan
        else:
            # 騎手コード不明 → ペア集計不能 (従来通り NaN)
            out["jockey_n_prev"] = np.nan
            out["jockey_win_rate"] = np.nan
            out["jockey_top3_rate"] = np.nan
    else:
        jc_ok = not np.isnan(jc)
        out["jockey_n_prev"] = 0.0 if jc_ok else np.nan
        out["jockey_win_rate"] = np.nan
        out["jockey_top3_rate"] = np.nan

    return out


def fill_history_features(df: pd.DataFrame, base: Path = BASE,
                          jockey_code_override: pd.Series | None = None,
                          trainer_code_override: pd.Series | None = None) -> dict:
    """df (parse_csv 済み weekly frame) に NUM_FEATS + CAT_FEATS を in-place で埋める。

    必要列: 馬名, 日付(YYYYMMDD str), 場所, 芝・ダ, 距離。
    任意列: 種牡馬, 年齢, 騎手, 調教師 (あるほど解決精度が上がる)。
    jockey_code_override: 検証用 — 騎手コードを名前マップでなく直接与える。
    戻り値: 統計 dict (coverage 等)。
    """
    idx, maps, meta = _load(base)
    jmap, tmap = maps["jockey"], maps["trainer"]

    n = len(df)
    num_out = {c: np.full(n, np.nan) for c in NUM_FEATS}
    jockey_code_col = np.array(["__NaN__"] * n, dtype=object)
    trainer_code_col = np.array(["__NaN__"] * n, dtype=object)
    stats = {"hit": 0, "new": 0, "ambiguous": 0, "bad_date": 0,
             "jockey_code": 0, "trainer_code": 0, "hist_max_date": meta["max_date"]}

    names = df["馬名"].map(_clean_name) if "馬名" in df.columns else pd.Series([""] * n)
    sires = (df["種牡馬"].map(_clean_name) if "種牡馬" in df.columns
             else pd.Series([""] * n, index=df.index))
    ages = (pd.to_numeric(df["年齢"], errors="coerce") if "年齢" in df.columns
            else pd.Series([np.nan] * n, index=df.index))
    dates = pd.to_numeric(df.get("日付"), errors="coerce")
    places = (df["場所"].astype(str).str.strip() if "場所" in df.columns
              else pd.Series([""] * n, index=df.index))
    surfs = df.get("芝・ダ", pd.Series([""] * n, index=df.index))
    dists = pd.to_numeric(df.get("距離"), errors="coerce")
    jnames = (df["騎手"].map(_clean_name) if "騎手" in df.columns
              else pd.Series([""] * n, index=df.index))
    tnames = (df["調教師"].map(_clean_name) if "調教師" in df.columns
              else pd.Series([""] * n, index=df.index))

    for i, (ix, name) in enumerate(names.items()):
        d = dates.loc[ix] if ix in dates.index else np.nan
        if pd.isna(d):
            stats["bad_date"] += 1
            continue
        race_date = int(d)

        # 騎手・調教師コード
        if jockey_code_override is not None:
            jc = pd.to_numeric(pd.Series([jockey_code_override.loc[ix]]),
                               errors="coerce").iloc[0]
        else:
            jc = jmap.get(jnames.loc[ix], np.nan)
        if trainer_code_override is not None:
            tc = pd.to_numeric(pd.Series([trainer_code_override.loc[ix]]),
                               errors="coerce").iloc[0]
        else:
            tc = tmap.get(tnames.loc[ix], np.nan)
        if pd.notna(jc):
            jockey_code_col[i] = str(int(jc))
            stats["jockey_code"] += 1
        if pd.notna(tc):
            trainer_code_col[i] = str(int(tc))
            stats["trainer_code"] += 1

        age = ages.loc[ix]
        birth_year = int(race_date // 10000 - age) if pd.notna(age) else None
        ent, status = idx.resolve(name, sires.loc[ix], birth_year)
        stats[status] += 1
        if status == "ambiguous":
            continue  # 従来通り NaN (安全側)

        feats = compute_row_feats(ent, race_date, places.loc[ix],
                                  surfs.loc[ix], dists.loc[ix], jc)
        for c in NUM_FEATS:
            num_out[c][i] = feats[c]

    for c in NUM_FEATS:
        df[c] = num_out[c]
    df["騎手コード"] = jockey_code_col
    df["調教師コード"] = trainer_code_col

    stats["coverage"] = {c: round(float(pd.Series(num_out[c]).notna().mean()), 3)
                         for c in NUM_FEATS}
    return stats
