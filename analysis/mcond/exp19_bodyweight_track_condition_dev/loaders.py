# -*- coding: utf-8 -*-
"""
loaders.py — EXP19 Stage 0: 3 つに分離した loader (歴史 torch 構造 / 市場 / 結果)
================================================================================
spec v0.2-frozen loader_contract / SPEC §6 S0-B:

  load_torch_struct()   歴史 torch (data/torch_20130105-20251228.csv) を**明示 usecols whitelist** で読む。
                        読み込み後に禁止列 (具体名・prefix・結果/払戻) が存在しないことを assert。
                        ファイルには `人気`・`単勝オッズ`・`指時系*`・`複上N`・`複人気N`・`補正` 等が実在するが、
                        whitelist の外なので一切読まない。2024/2025 行は読み込み直後に破棄。
  load_market()         TANPUK から historical_pre_snapshot / terminal_close_market の単勝オッズ (EXP16A/18 と同契約)。
  load_finishers()      結果 loader。着順は finisher 判定と勝馬一意性 (母集団定義) にだけ使う。
                        Stage 0 では性能を計算しない。2024/2025 は破棄。

loader_sha256() は本ファイルの sha256 を返し、manifest に保存する。
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
RESEARCH = BASE / "data" / "_research" / "mcond" / "exp19"
TORCH = BASE / "data" / "torch_20130105-20251228.csv"
BABA = BASE / "data" / "baba_feats.parquet"
BABA_TODAY = BASE / "data" / "baba_today.json"
MASTER = BASE / "data" / "master_v2_20130105-20251228.csv"
FORWARD = BASE / "data" / "forward_bodyweight"
SEALED_FROM = 20240101

TORCH_WHITELIST = ["日付", "開催", "場所", "Ｒ", "発走時刻", "レースID(新/馬番無)", "血統登録番号", "馬番",
                   "芝・ダ", "距離", "トラックコード(JV)", "性別", "年齢", "馬体重", "馬体重増減"]
FORBIDDEN_EXACT = (["人気", "単勝オッズ", "複勝オッズ下限", "複勝オッズ上限", "複勝シェア", "補正"]
                   + [f"複上{i}" for i in range(1, 5)] + [f"複人気{i}" for i in range(1, 5)])
FORBIDDEN_PREFIX = ["指時系"]
FORBIDDEN_RESULT = [r"着順", r"着差", r"タイム", r"走破", r"上り", r"通過", r"払戻", r"配当", r"確定", r"結果",
                    r"入線", r"コーナー", r"賞金"]
JUMP_MIN, JUMP_MAX = 51, 59


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 22), b""):
            h.update(ch)
    return h.hexdigest()


def loader_sha256() -> str:
    return sha256_file(Path(__file__))


def assert_torch_frame_clean(cols) -> None:
    cols = list(cols)
    bad = [c for c in cols if c in FORBIDDEN_EXACT]
    bad += [c for c in cols for p in FORBIDDEN_PREFIX if str(c).startswith(p)]
    bad += [c for c in cols for p in FORBIDDEN_RESULT if re.search(p, str(c))]
    assert not bad, f"禁止列が torch 構造 loader に混入: {sorted(set(bad))}"
    assert set(cols) <= set(TORCH_WHITELIST) | {"date", "year", "rid16", "ban", "pid", "kg", "chg_src"}, \
        f"whitelist 外の列: {sorted(set(cols) - set(TORCH_WHITELIST))}"


def torch_header() -> list[str]:
    return pd.read_csv(TORCH, encoding="cp932", nrows=0).columns.tolist()


def load_torch_struct() -> pd.DataFrame:
    """歴史 torch を whitelist だけで読む。2024/2025 は破棄。結果・人気・オッズ列は存在し得ない"""
    df = pd.read_csv(TORCH, encoding="cp932", usecols=TORCH_WHITELIST, dtype=str)
    assert list(sorted(df.columns)) == sorted(TORCH_WHITELIST)
    assert_torch_frame_clean(df.columns)
    d = pd.to_numeric(df["日付"], errors="coerce").astype("int64")
    df["date"] = np.where(d < 1_000_000, 20_000_000 + d, d)            # yymmdd → yyyymmdd
    df = df[df["date"] < SEALED_FROM].copy()
    assert int(df["date"].max()) < SEALED_FROM, "2024/2025 が残っている"
    df["year"] = df["date"] // 10000
    df["rid16"] = df["レースID(新/馬番無)"].astype(str).str.replace(r"\D", "", regex=True).str[:16]
    df["ban"] = pd.to_numeric(df["馬番"], errors="coerce").astype("int64")
    df["pid"] = df["血統登録番号"].astype(str).str.strip()
    df["kg"] = pd.to_numeric(df["馬体重"], errors="coerce")
    df["chg_src"] = pd.to_numeric(df["馬体重増減"].astype(str).str.replace("+", "", regex=False), errors="coerce")
    assert_torch_frame_clean(df.columns)
    return df.reset_index(drop=True)


def is_jump(track_code) -> np.ndarray:
    tc = pd.to_numeric(pd.Series(track_code), errors="coerce")
    return ((tc >= JUMP_MIN) & (tc <= JUMP_MAX)).fillna(False).to_numpy()


def load_market(years) -> pd.DataFrame:
    """TANPUK の単勝オッズ (historical_pre_snapshot / terminal_close_market)。結果列を持たない。
    pre = 区分1 のうちレース当日で確定記録の 15 分以上前の最後の 1 本 (EXP16A/18 と同契約)"""
    from ..exp18_cross_pool_market_tomography_dev.loaders import MIN_GAP_PRE, NMAX, _read_pool, _snap_time
    tan = _read_pool("TANPUK", [y for y in years if y < SEALED_FROM // 10000])
    tan = tan.rename(columns={tan.columns[1]: "kubun", tan.columns[2]: "mdhm", tan.columns[3]: "tou"})
    tan["kubun"] = pd.to_numeric(tan["kubun"], errors="coerce").astype(int)
    tan["mmdd"], tan["snap_min"] = _snap_time(tan["mdhm"])
    W = np.column_stack([pd.to_numeric(tan[f"{b}単"], errors="coerce") for b in range(1, NMAX + 1)])
    rows = []
    tan = tan.assign(_i=np.arange(len(tan)))
    for rid, g in tan.groupby("rid16", sort=False):
        g4 = g[g["kubun"] == 4]
        if not len(g4):
            continue
        t = g4.iloc[-1]
        mmdd = rid[4:8]
        g1 = g[(g["kubun"] == 1) & (g["mmdd"] == mmdd) & (t["snap_min"] - g["snap_min"] >= MIN_GAP_PRE)]
        p = g1.iloc[-1] if len(g1) else None
        rows.append({"rid16": rid, "term_i": int(t["_i"]), "pre_i": int(p["_i"]) if p is not None else -1,
                     "term_min": float(t["snap_min"]), "pre_min": float(p["snap_min"]) if p is not None else np.nan})
    idx = pd.DataFrame(rows).set_index("rid16")
    return idx, W


def load_finishers(max_date: int = 20231231) -> pd.DataFrame:
    """結果 loader (母集団定義専用)。(rid16, 馬番, 着順) だけ。2024/2025 は読まない"""
    assert max_date < SEALED_FROM
    parts = []
    for ch in pd.read_csv(MASTER, encoding="utf-8-sig", dtype=str,
                          usecols=["日付", "レースID(新/馬番無)", "馬番", "着順"], chunksize=200_000):
        d = pd.to_numeric(ch["日付"], errors="coerce")
        ch = ch[(d <= max_date) & (d >= 20130101)]
        if len(ch):
            parts.append(ch)
    m = pd.concat(parts, ignore_index=True)
    m["rid16"] = m["レースID(新/馬番無)"].astype(str).str.replace(r"\D", "", regex=True).str[:16]
    m["ban"] = pd.to_numeric(m["馬番"], errors="coerce").astype("int64")
    m["jyun"] = pd.to_numeric(m["着順"], errors="coerce")
    assert int(pd.to_numeric(m["日付"]).max()) < SEALED_FROM
    return m[["rid16", "ban", "jyun"]]
