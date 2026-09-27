# -*- coding: utf-8 -*-
"""
loaders.py — EXP21 Stage 0: 価格 loader と払戻 loader を分離する
===============================================================
価格 (結果を含まない):
  TANPUK / UMAREN 時系列 (2013-2023。2024/2025 は EXP18 の _read_pool が読み込み直後に破棄)
  TARGET オッズ export (1 行 1 頭):
    2023 91 列  E:/競馬過去走データ/test_v1.csv
    2026 227 列 data/odds/OD{yymmdd}.CSV (91 列 + 三連複 136 列)
  TARGET 列配置 (仮説。G0 で公式払戻から逆引き検証するまで採用しない):
    0 race key (場2 年2 回1 日1(16 進) R2 馬番2) / 1 頭数 / 2 区分コード / 3 空 / 4 馬番 / 5 枠番 / 6 馬名
    7 単勝 / 8-9 複勝 Lo,Hi / 10-27 馬連 slot 1..18 / 28-35 枠連 slot 枠 1..8 / 36 export 時刻 HHMM /
    37-72 ワイド (Lo,Hi) × slot 1..18 / 73-90 馬単 (自馬 1 着 → slot 2 着) 1..18 /
    91-226 三連複 (自馬 + slot 対 {j<k}、自馬を除く 17 slot の C(17,2)=136 を辞書順)
払戻 (結果。G0 の配置逆引きと取消・同着・返還の分類にだけ使う。ROI は計算しない):
  kekka master (2013-2023 だけ。2024/2025 は破棄) / wide_payouts parquet / 2026 週次 kekka / wide_kekka.csv
"""
from __future__ import annotations

import hashlib
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
OUT = HERE / "out"
RESEARCH = BASE / "data" / "_research" / "mcond" / "exp21"
ODIR = BASE / "data" / "Time _series_odds"
RAW2023 = Path("E:/競馬過去走データ/test_v1.csv")
OD_DIR = BASE / "data" / "odds"
KEKKA_MASTER = BASE / "data" / "kekka_20130105-20251228.csv"
WIDE_PARQUET = BASE / "data" / "wide_payouts_2016-2025.parquet"
KEKKA_2026_DIR = BASE / "data" / "kekka"
WIDE_2026 = BASE / "data" / "kekka" / "wide_kekka.csv"
SEALED_FROM = 20240101
NMAX = 18
TRIO_PAIRS = {}          # 自馬 i → 136 slot の (j,k)
for i in range(1, NMAX + 1):
    others = [x for x in range(1, NMAX + 1) if x != i]
    TRIO_PAIRS[i] = list(combinations(others, 2))


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 22), b""):
            h.update(ch)
    return h.hexdigest()


def num(x) -> float:
    try:
        return float(str(x).strip())
    except ValueError:
        return np.nan


def parse_key(k: str):
    """TARGET race key → (venue, yy, kai, nichi, R, ban)。日は 16 進 1 桁 (A=10, B=11, C=12)"""
    k = k.strip()
    return k[0:2], k[2:4], int(k[4], 16), int(k[5], 16), int(k[6:8]), int(k[8:10])


def read_target_odds(path: Path, ncols_expected: int) -> pd.DataFrame:
    """TARGET オッズ export を読む (1 行 1 頭)。値は float、0.0 はそのまま保持 (欠損と区別して後で分類)"""
    recs = []
    with open(path, encoding="cp932", errors="strict") as fh:
        for line in fh:
            p = line.rstrip("\r\n").split(",")
            assert len(p) == ncols_expected, (path, len(p))
            venue, yy, kai, nichi, R, ban_key = parse_key(p[0])
            recs.append({"venue": venue, "yy": yy, "kai": kai, "nichi": nichi, "R": R, "ban_key": ban_key,
                         "n": int(p[1]), "code": p[2].strip(), "ban": int(p[4]), "waku": int(p[5]),
                         "export_hhmm": p[36].strip(), "vals": np.array([num(x) for x in p[7:36] + p[37:]])})
    df = pd.DataFrame(recs)
    return df


def split_vals(v: np.ndarray, with_trio: bool) -> dict:
    """read_target_odds の vals (列 7..35, 37..) を券種ブロックへ分ける"""
    out = {"tan": v[0], "fuku_lo": v[1], "fuku_hi": v[2], "umaren": v[3:21], "wakuren": v[21:29]}
    w = v[29:65]
    out["wide_lo"], out["wide_hi"] = w[0::2], w[1::2]
    out["umatan"] = v[65:83]
    if with_trio:
        out["trio"] = v[83:219]
    return out


def race_key_to_rid16(df: pd.DataFrame, date_of) -> pd.Series:
    """(venue, kai, nichi, R) + 日付 → rid16 = YYYYMMDD + 場 + 回 2 + 日 2 + R 2"""
    return df.apply(lambda r: f"{date_of(r)}{r['venue']}{r['kai']:02d}{r['nichi']:02d}{r['R']:02d}", axis=1)


# ---------------------------------------------------------------- 払戻 (G0 専用)
def load_kekka_master(max_date: int = 20231231) -> pd.DataFrame:
    assert max_date < SEALED_FROM
    df = pd.read_csv(KEKKA_MASTER, encoding="cp932", dtype=str)
    df["rid16"] = df["レースID(新)"].astype(str).str[:16]
    df["date"] = df["rid16"].str[:8].astype(int)
    df = df[df["date"] <= max_date].copy()
    assert int(df["date"].max()) < SEALED_FROM
    df["ban"] = pd.to_numeric(df["馬番"], errors="coerce")
    return df


def load_wide_payouts(max_date: int = 20231231) -> pd.DataFrame:
    w = pd.read_parquet(WIDE_PARQUET)
    w = w[w["date"] <= max_date].copy()
    assert int(w["date"].max()) < SEALED_FROM
    return w


def load_kekka_2026(dates) -> pd.DataFrame:
    parts = []
    for d in dates:
        f = KEKKA_2026_DIR / f"{d}.csv"
        if f.exists():
            k = pd.read_csv(f, encoding="cp932", dtype=str)
            k["rid16"] = k["レースID(新)"].astype(str).str[:16]
            parts.append(k)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def load_wide_2026() -> pd.DataFrame:
    """wide_kekka.csv: 年,月,日,場,R,クラス,芝ダ,距離,頭数,"i-j \\pay (人気)/ ..." """
    recs = []
    with open(WIDE_2026, encoding="cp932", errors="replace") as fh:
        for line in fh:
            m = re.match(r"(\d{4}),(\d{2}),(\d{2}),([^,]+),(\d+),.*?\"(.*)\"", line.strip())
            if not m:
                continue
            y, mo, d, venue, R, body = m.groups()
            for a, b, pay in re.findall(r"(\d{2})-(\d{2})\s*\\\s*([\d,]+)", body):
                recs.append({"date": int(f"{y}{mo}{d}"), "venue_name": venue, "R": int(R), "i": int(a), "j": int(b),
                             "pay": int(pay.replace(",", ""))})
    return pd.DataFrame(recs)
