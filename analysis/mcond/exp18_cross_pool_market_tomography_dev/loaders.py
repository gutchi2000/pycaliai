# -*- coding: utf-8 -*-
"""
loaders.py — EXP18 Stage 0: 二つに分離した loader
==================================================
spec v0.4 (凍結 commit 67b874ec) の stage0 規約どおり、**結果 loader と市場構造 loader を分離**する。

  load_structure(years)  市場構造 loader
      TANPUK / UMAREN のオッズ・区分・記録時刻・頭数・券種全体票数と、master_v2 の
      **結果を含まない** レース情報列 (日付・場所・芝ダ・トラックコード(JV)・発走時刻・出走頭数) だけ。
      2019-2023 を読んでよい。**着順・払戻・realized top2 列が存在しないことを hard assert**。
  load_outcomes()        結果 loader
      着順・払戻・realized top2。**max(year) <= 2018 を hard assert**。

両 loader とも 2024/2025 の行は**読み込み直後に破棄**し、残っていないことを assert する。
T2・offset 残差の性能は Stage 0 では一切計算しない。
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
OUT = HERE / "out"
ODIR = BASE / "data" / "Time _series_odds"
MASTER = BASE / "data" / "master_v2_20130105-20251228.csv"
KEKKA = BASE / "data" / "kekka_20130105-20251228.csv"
RESEARCH = BASE / "data" / "_research" / "mcond" / "exp18"

SEALED_FROM_YEAR = 2024          # 2024/2025 は読み込み直後に破棄
RESULT_MAX_YEAR = 2018           # 結果 loader の上限 (hard assert)
STRUCT_MAX_YEAR = 2023           # 構造 loader の上限
JUMP_MIN, JUMP_MAX = 51, 59      # 障害 (race_eligibility.py と同一定義)
MIN_GAP_PRE = 15                 # historical_pre_snapshot: 確定記録の 15 分以上前 (EXP16A と同契約)
NMAX = 18

# 構造 loader が読んでよい master 列 (結果を含まない)
MASTER_STRUCT_COLS = ["日付", "レースID(新/馬番無)", "馬番", "場所", "芝・ダ", "トラックコード(JV)",
                      "発走時刻", "出走頭数"]
# 構造 loader の出力に**存在してはならない**列名パターン
FORBIDDEN_OUTCOME_PATTERNS = [r"着順", r"着差", r"払戻", r"配当", r"pay", r"realized", r"winner",
                              r"is_win", r"is_plc", r"is_sho", r"top2", r"top3", r"jyun", r"走破",
                              r"上り", r"fpay", r"winpair"]

UMAREN_PAIRS = [(i, j) for i in range(1, NMAX + 1) for j in range(i + 1, NMAX + 1)]   # 153 組


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 22), b""):
            h.update(ch)
    return h.hexdigest()


def rid16_of(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\D", "", regex=True).str[:16]


def assert_no_outcome_columns(cols, where: str):
    bad = [c for c in cols for p in FORBIDDEN_OUTCOME_PATTERNS if re.search(p, str(c), re.I)]
    assert not bad, f"[{where}] 結果列が構造 loader に混入: {sorted(set(bad))}"


def _drop_sealed(df: pd.DataFrame, year: pd.Series, where: str) -> pd.DataFrame:
    keep = year < SEALED_FROM_YEAR
    out = df[keep.to_numpy()].copy()
    assert int(year[keep].max()) < SEALED_FROM_YEAR, f"[{where}] 2024/2025 が残っている"
    return out


# ---------------------------------------------------------------- 構造 loader
def _read_pool(kind: str, years) -> pd.DataFrame:
    parts = []
    for f in sorted(ODIR.glob(f"{kind}_*.csv")):
        df = pd.read_csv(f, encoding="cp932", low_memory=False)
        df["rid16"] = rid16_of(df.iloc[:, 0])
        yr = df["rid16"].str[:4].astype(int)
        df = _drop_sealed(df, yr, kind)                      # 読み込み直後に 2024/2025 を破棄
        df["year"] = df["rid16"].str[:4].astype(int)
        df = df[df["year"].isin(years)]
        if len(df):
            parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    assert df["year"].max() <= STRUCT_MAX_YEAR and df["year"].max() < SEALED_FROM_YEAR
    return df


def load_race_info(years) -> pd.DataFrame:
    """master_v2 から結果を含まないレース情報だけを読む (1 レース 1 行)"""
    parts = []
    for ch in pd.read_csv(MASTER, encoding="utf-8-sig", dtype=str, usecols=MASTER_STRUCT_COLS,
                          chunksize=200_000):
        assert_no_outcome_columns(ch.columns, "master_struct")
        d = pd.to_numeric(ch["日付"], errors="coerce")
        ch = ch[(d // 10000 < SEALED_FROM_YEAR) & (d // 10000).isin(years)]
        if len(ch):
            parts.append(ch)
    m = pd.concat(parts, ignore_index=True)
    m["rid16"] = rid16_of(m["レースID(新/馬番無)"])
    m["date"] = pd.to_numeric(m["日付"], errors="coerce").astype(int)
    m["year"] = m["date"] // 10000
    tc = pd.to_numeric(m["トラックコード(JV)"], errors="coerce")
    m["track_code"] = tc
    m["is_jump"] = ((tc >= JUMP_MIN) & (tc <= JUMP_MAX)).fillna(False)
    hm = m["発走時刻"].astype(str).str.extract(r"(\d{1,2}):(\d{2})")
    m["post_min"] = pd.to_numeric(hm[0], errors="coerce") * 60 + pd.to_numeric(hm[1], errors="coerce")
    m["entrants_master"] = pd.to_numeric(m["出走頭数"], errors="coerce")
    info = (m.groupby("rid16")
              .agg(date=("date", "first"), year=("year", "first"), venue=("場所", "first"),
                   surface=("芝・ダ", "first"), track_code=("track_code", "first"),
                   is_jump=("is_jump", "first"), post_min=("post_min", "first"),
                   entrants_master=("entrants_master", "first"),
                   master_rows=("馬番", "size")))
    assert_no_outcome_columns(info.columns, "race_info")
    assert int(info["year"].max()) < SEALED_FROM_YEAR
    return info


def _snap_time(md: pd.Series) -> tuple[pd.Series, pd.Series]:
    s = md.astype(str).str.zfill(8)
    mmdd = s.str[:4]
    minutes = pd.to_numeric(s.str[4:6], errors="coerce") * 60 + pd.to_numeric(s.str[6:8], errors="coerce")
    return mmdd, minutes


def load_structure(years=range(2013, 2024)) -> dict:
    """市場構造 loader。戻り値: {"tan": DataFrame, "um": DataFrame, "info": DataFrame}
    tan/um は snapshot 行 (区分・時刻・オッズ行列・票数)。結果列は含まない。"""
    years = [y for y in years if y <= STRUCT_MAX_YEAR]
    tan = _read_pool("TANPUK", years)
    um = _read_pool("UMAREN", years)
    for df, kind in ((tan, "TANPUK"), (um, "UMAREN")):
        df.rename(columns={df.columns[1]: "kubun", df.columns[2]: "mdhm", df.columns[3]: "tou"},
                  inplace=True)
        df["kubun"] = pd.to_numeric(df["kubun"], errors="coerce").astype(int)
        df["mmdd"], df["snap_min"] = _snap_time(df["mdhm"])
        assert_no_outcome_columns(df.columns, kind)
    tan.rename(columns={"単勝票数": "votes_win_total", "複勝票数": "votes_place_total"}, inplace=True)
    um.rename(columns={"馬連票数": "votes_umaren_total"}, inplace=True)
    info = load_race_info(years)
    return {"tan": tan, "um": um, "info": info}


def tan_arrays(row) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """TANPUK 1 行 → win, place_lo, place_hi (index 0..17 = 馬番 1..18, 欠損は nan)"""
    w = np.array([pd.to_numeric(row.get(f"{b}単"), errors="coerce") for b in range(1, NMAX + 1)],
                 dtype=float)
    lo = np.array([pd.to_numeric(row.get(f"{b}複Lo"), errors="coerce") for b in range(1, NMAX + 1)],
                  dtype=float)
    hi = np.array([pd.to_numeric(row.get(f"{b}複Hi"), errors="coerce") for b in range(1, NMAX + 1)],
                  dtype=float)
    return w, lo, hi


def umaren_cols(df: pd.DataFrame) -> list[str]:
    cols = [f"馬{i:02d}-{j:02d}" for i, j in UMAREN_PAIRS]
    missing = [c for c in cols if c not in df.columns]
    assert not missing, f"UMAREN 列不足: {missing[:3]}"
    return cols


# ---------------------------------------------------------------- 結果 loader
def load_outcomes(max_year: int = RESULT_MAX_YEAR) -> pd.DataFrame:
    """結果 loader。着順・realized top2・馬連払戻。**max(year) <= 2018 を hard assert**。"""
    assert max_year <= RESULT_MAX_YEAR, f"結果 loader は {RESULT_MAX_YEAR} 年以前だけ (要求 {max_year})"
    parts = []
    for ch in pd.read_csv(MASTER, encoding="utf-8-sig", dtype=str,
                          usecols=["日付", "レースID(新/馬番無)", "馬番", "着順"], chunksize=200_000):
        d = pd.to_numeric(ch["日付"], errors="coerce")
        ch = ch[(d // 10000 < SEALED_FROM_YEAR) & (d // 10000 <= max_year)]
        if len(ch):
            parts.append(ch)
    m = pd.concat(parts, ignore_index=True)
    m["rid16"] = rid16_of(m["レースID(新/馬番無)"])
    m["year"] = pd.to_numeric(m["日付"], errors="coerce").astype(int) // 10000
    assert int(m["year"].max()) <= RESULT_MAX_YEAR, "結果 loader に 2019 年以降が混入"
    m["ban"] = pd.to_numeric(m["馬番"], errors="coerce").astype(int)
    m["jyun"] = pd.to_numeric(m["着順"], errors="coerce")
    return m[["rid16", "year", "ban", "jyun"]]


def dnf_horses(starters, finishers) -> list[int]:
    """DNF = terminal starter (単勝 > 1.0) のうち finisher 行が無い馬 (EXP16A と同定義)"""
    return sorted(set(int(x) for x in starters) - set(int(x) for x in finishers))


def realized_top2(outc: pd.DataFrame) -> pd.DataFrame:
    """race ごとの realized 順不同 top2・finisher 集合・同着フラグ。<= 2018 のみ"""
    assert int(outc["year"].max()) <= RESULT_MAX_YEAR
    rows = []
    for rid, g in outc.groupby("rid16", sort=False):
        j = g["jyun"]
        first = g.loc[j == 1, "ban"].tolist()
        second = g.loc[j == 2, "ban"].tolist()
        dnf = int(j.isna().sum())
        # 1着・2着のどちらかに同着がある race は複数の馬連的中組 (または組の曖昧さ) を生むので
        # 主評価から除外する (保守的に「1着1頭・2着1頭」の race だけ top2 を持つ)
        single = len(first) == 1 and len(second) == 1
        # 注意: master_v2 は dropna 後で DNF 馬の行を持たない。DNF は「terminal starter なのに
        # finisher 行が無い馬」として呼び出し側で判定する (EXP16A と同定義)。nan_jyun_rows は参考値
        rows.append({"rid16": rid, "year": int(g["year"].iloc[0]), "n_rows": len(g), "nan_jyun_rows": dnf,
                     "finishers": tuple(sorted(g.loc[j.notna(), "ban"].tolist())),
                     "n_first": len(first), "n_second": len(second),
                     "dead_heat_top2": not single,
                     "top2": (tuple(sorted(first + second)) if single else None),
                     "top3": tuple(sorted(g.loc[j <= 3, "ban"].tolist()))})
    return pd.DataFrame(rows)
