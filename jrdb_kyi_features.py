# -*- coding: utf-8 -*-
"""
jrdb_kyi_features.py — JRDB の前日データ (KYI) と外厩データから「今回のレースについての発走前情報」の特徴を作る (v8 候補、2026-11-15)
=====================================================================================================
v7 (jrdb_features.py の j1_*) は JRDB の「過去走の中身」だけを使う。v8 はそれに次を足す (接頭辞 k8_):
  KYI (前日配布、racekey+馬番で 1 行):
    IDM, 騎手指数, 情報指数, 総合指数, 調教指数, 厩舎指数, 激走指数, 脚質, 距離適性, 上昇度, ローテーション,
    厩舎評価コード, 重適正コード, ブリンカー, 展開予想 (テン/ペース/上がり/位置指数, ペース予想, 道中・後3F・ゴールの順位/差/内外, 展開記号)
    + レース内の相対値 (IDM・総合指数・騎手指数・情報指数のレース内 z と順位、展開の順位は頭数で割った相対位置)
  外厩 (JRDB 会員ページの TARGET 取り込みデータ「外厩コメント」、2016〜。出走の約 55% に記録):
    外厩あり, 施設 (2016-2022 の出現数順のコード、上位 60 以外は 99), 施設の日付からレース日までの日数, 「中N週」の N

使わないもの: KYI の 基準オッズ・基準人気・人気指数 (市場の代わりになる値。v6/v7 と同じく、オッズ系は学習に入れない)。
TARGET 取り込みデータの「展開」「番手」「IDM 印」は KYI から作られたもの (2024 年で 100% / 99.3% / 上位 6 頭の印と一致を確認)、
「成績 IDM」は SED の IDM と同じなので、ここでは足さない。「厩舎ランク」「騎手ランク」は過去分が無いので学習に使えない。

先読み: KYI と外厩は発走前日に配布される。アーカイブは後から直されていないか未確認 (CW-12 が発走前保存と照合中)。
規約: JRDB のデータ・指数は加工しても第三者が見られる場所に出さない (k8_ で始まる列は公開物から除外する)。data/jrdb/ は git 管理外。
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
JRDB = BASE / "data" / "jrdb"
KYI_RAW = JRDB / "raw" / "Kyi"
TM_RAW = JRDB / "raw" / "targetmark"
KYI_PQ = JRDB / "kyi.parquet"
GK_PQ = JRDB / "gaikyu.parquet"
PREFIX = "k8_"
MODE = "k8_v1"
_PACE = {"H": 1.0, "M": 0.0, "S": -1.0}

# (列名, 開始位置 1 始まり, バイト数, 型)  仕様 http://www.jrdb.com/program/Kyi/kyi_doc.txt (第 3 版以降、1022 バイト + CRLF)
KYI_FIELDS = [("racekey", 1, 8, "s"), ("ban", 9, 2, "n"), ("idm", 55, 5, "n"), ("jk_idx", 60, 5, "n"), ("info_idx", 65, 5, "n"),
              ("sogo_idx", 85, 5, "n"), ("style", 90, 1, "n"), ("kyori", 91, 1, "n"), ("joushou", 92, 1, "n"), ("rot", 93, 3, "n"),
              ("chokyo_idx", 145, 5, "n"), ("kyusha_idx", 150, 5, "n"), ("kyusha_hyoka", 156, 1, "n"), ("gekiso", 161, 3, "n"),
              ("omo", 166, 1, "n"), ("blinker", 171, 1, "n"),
              ("ten", 359, 5, "n"), ("pidx", 364, 5, "n"), ("agari", 369, 5, "n"), ("posidx", 374, 5, "n"), ("pace", 379, 1, "s"),
              ("d_jun", 380, 2, "n"), ("d_sa", 382, 2, "n"), ("d_io", 384, 1, "n"), ("a_jun", 385, 2, "n"), ("a_sa", 387, 2, "n"),
              ("a_io", 389, 1, "n"), ("g_jun", 390, 2, "n"), ("g_sa", 392, 2, "n"), ("g_io", 394, 1, "n"), ("tenkai", 395, 1, "n")]
KYI_NUM = [c for c, _, _, t in KYI_FIELDS if t == "n" and c != "ban"] + ["pace_q"]
REL = ["idm", "sogo_idx", "jk_idx", "info_idx"]
N_FAC = 60
FEATURES = ([f"k8_{c}" for c in KYI_NUM if c not in ("d_jun", "a_jun", "g_jun")]
            + ["k8_d_jun_rel", "k8_a_jun_rel", "k8_g_jun_rel"]
            + [f"k8_{c}_z" for c in REL] + [f"k8_{c}_rk" for c in REL]
            + ["k8_gk_has", "k8_gk_fac", "k8_gk_days", "k8_gk_weeks"])


# ------------------------------------------------------------------ 読み込み
def _parse_kyi_bytes(raw: bytes) -> list[tuple]:
    out = []
    for line in raw.split(b"\r\n"):
        if len(line) >= 395:
            out.append(tuple(line[s - 1:s - 1 + n] for _, s, n, _ in KYI_FIELDS))
    return out


def kyi_frame(rows: list[tuple]) -> pd.DataFrame:
    k = pd.DataFrame(rows, columns=[c for c, *_ in KYI_FIELDS])
    for c, _, _, t in KYI_FIELDS:
        v = k[c].str.decode("cp932", errors="replace").str.strip()
        k[c] = pd.to_numeric(v, errors="coerce") if t == "n" else v
    k["pace_q"] = k["pace"].map(_PACE)
    return k.drop(columns=["pace"]).drop_duplicates(["racekey", "ban"], keep="last")


def build_kyi(raw_dir: Path = KYI_RAW, out: Path = KYI_PQ) -> pd.DataFrame:
    rows = []
    for zp in sorted(raw_dir.glob("*.zip")):
        with zipfile.ZipFile(zp) as z:
            for nm in z.namelist():
                if re.search(r"KYI\d{6}", nm, re.I):
                    rows += _parse_kyi_bytes(z.read(nm))
    k = kyi_frame(rows)
    k.to_parquet(out, index=False)
    return k


def _gk_rows(name: str, text: str) -> list[tuple]:
    rows = []
    for l in text.splitlines():
        if "," not in l:
            continue
        key, v = l.split(",", 1)
        key = key.strip()
        if len(key) != 10:
            continue
        parts = v.split("　")
        fac = parts[0].strip()
        mmdd = parts[1].strip() if len(parts) > 1 else ""
        m = re.search(r"中(\d*)週", v)
        weeks = float(m.group(1)) if (m and m.group(1)) else np.nan
        rows.append((key[:8], int(key[8:10]), fac, mmdd, weeks))
    return rows


def build_gaikyu(raw_dir: Path = TM_RAW, out: Path = GK_PQ) -> pd.DataFrame:
    """外厩コメント (YC{場2}{年2}{回1}{日1}.DAT、1 行 = 'racekey8+馬番2,施設　MM/DD　中N週') を表にする。"""
    rows = []
    for zp in sorted(raw_dir.glob("*gaikyucomment_*.zip")):
        with zipfile.ZipFile(zp) as z:
            for nm in z.namelist():
                if nm.endswith("/"):
                    continue
                rows += _gk_rows(nm, z.read(nm).decode("cp932", errors="replace"))
    g = pd.DataFrame(rows, columns=["racekey", "ban", "gk_fac", "gk_mmdd", "gk_weeks"]).drop_duplicates(["racekey", "ban"], keep="last")
    g.to_parquet(out, index=False)
    return g


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    k = pd.read_parquet(KYI_PQ) if KYI_PQ.exists() else build_kyi()
    g = pd.read_parquet(GK_PQ) if GK_PQ.exists() else build_gaikyu()
    return k, g


# ------------------------------------------------------------------ キー
def rid16_to_racekey(rid16: pd.Series) -> pd.Series:
    """16 桁のレース ID (YYYYMMDD 場2 回2 日2 R2) → JRDB racekey (場2 年2 回1 日1(16進) R2)。"""
    r = rid16.astype(str).str[:16]
    kai = pd.to_numeric(r.str[10:12], errors="coerce").fillna(0).astype(int)
    day = pd.to_numeric(r.str[12:14], errors="coerce").fillna(0).astype(int)
    return r.str[8:10] + r.str[2:4] + kai.map(lambda x: "%x" % x) + day.map(lambda x: "%x" % x) + r.str[14:16]


def facility_codes(g: pd.DataFrame, racekeys_train: set | None = None) -> dict:
    """施設 → コード (2016-2022 の出現数順に 1..N_FAC、それ以外は 99)。racekey の年 2 桁で期間を切る。"""
    yy = pd.to_numeric(g["racekey"].str[2:4], errors="coerce")
    tr = g[(yy >= 16) & (yy <= 22)]
    top = tr["gk_fac"].value_counts().head(N_FAC).index
    return {f: i + 1 for i, f in enumerate(top)}


# ------------------------------------------------------------------ 特徴
def make_features(keys: pd.DataFrame, ymd: pd.Series, k: pd.DataFrame, g: pd.DataFrame, fac_map: dict,
                  n_start: pd.Series | None = None, race_col: pd.Series | None = None) -> pd.DataFrame:
    """keys: racekey, ban (index は呼び出し側の行)。ymd: レース日 YYYYMMDD。戻り値は FEATURES 列 (index は keys と同じ)。"""
    x = keys[["racekey", "ban"]].copy()
    x["ban"] = pd.to_numeric(x["ban"], errors="coerce")
    x["_row"] = np.arange(len(x))
    x = x.merge(k, on=["racekey", "ban"], how="left").merge(g, on=["racekey", "ban"], how="left").sort_values("_row")
    x.index = keys.index
    out = pd.DataFrame(index=keys.index)
    for c in KYI_NUM:
        if c in ("d_jun", "a_jun", "g_jun"):
            continue
        out[f"k8_{c}"] = x[c].astype(float)
    race = race_col if race_col is not None else x["racekey"]
    ns = n_start if n_start is not None else x.groupby(race)["ban"].transform("size")
    ns = pd.to_numeric(ns, errors="coerce").astype(float)
    for p in ("d", "a", "g"):
        out[f"k8_{p}_jun_rel"] = (x[f"{p}_jun"] - 1) / (ns - 1).clip(lower=1)
    for c in REL:
        v = x[c].astype(float)
        grp = v.groupby(race)
        out[f"k8_{c}_z"] = (v - grp.transform("mean")) / grp.transform("std").replace(0, np.nan)
        out[f"k8_{c}_rk"] = grp.rank(ascending=False, method="min")
    has = x["gk_fac"].notna() & (x["gk_fac"].astype(str) != "")
    out["k8_gk_has"] = has.astype(float)
    out["k8_gk_fac"] = np.where(has, x["gk_fac"].map(fac_map).fillna(99), np.nan)
    out["k8_gk_weeks"] = x["gk_weeks"].astype(float)
    yv = pd.to_numeric(ymd, errors="coerce").values
    md = x["gk_mmdd"].fillna("").astype(str).values
    days = np.full(len(x), np.nan)
    for i, (s, y) in enumerate(zip(md, yv)):
        m = re.fullmatch(r"(\d{2})/(\d{2})", s)
        if not m or not np.isfinite(y):
            continue
        y = int(y); ry, rm, rd = y // 10000, y // 100 % 100, y % 100
        mm, dd = int(m.group(1)), int(m.group(2))
        yr = ry if (mm, dd) <= (rm, rd) else ry - 1
        try:
            days[i] = (pd.Timestamp(ry, rm, rd) - pd.Timestamp(yr, mm, dd)).days
        except ValueError:
            pass
    out["k8_gk_days"] = days
    return out[FEATURES]


def add_master_feats(df: pd.DataFrame, mode: str = MODE, rid_col: str = "レースID(新/馬番無)", ban_col: str = "馬番") -> pd.DataFrame:
    """学習用 (master_v2 の各行) に k8_* を足して返す。"""
    if mode != MODE:
        raise ValueError(f"未知の kyi_mode: {mode}")
    k, g = load_tables()
    fac = load_fac_map()
    keys = pd.DataFrame({"racekey": rid16_to_racekey(df[rid_col]), "ban": df[ban_col]}, index=df.index)
    ymd = pd.to_numeric(df[rid_col].astype(str).str[:8], errors="coerce")
    f = make_features(keys, ymd, k, g, fac, race_col=df[rid_col].astype(str).str[:16])
    df = df.drop(columns=[c for c in FEATURES if c in df.columns])
    return pd.concat([df, f], axis=1)


SNAP_KYI = JRDB / "snapshots" / "Kyi"
SNAP_TM = JRDB / "snapshots" / "targetmark"
FAC_JSON = JRDB / "gaikyu_fac_map.json"
KYI_COVERAGE_MIN = 0.90


def load_fac_map() -> dict:
    """施設コード表。学習時に作ったものを data/jrdb/gaikyu_fac_map.json に固定し、配信も同じ表を引く。"""
    import json
    if FAC_JSON.exists():
        return json.loads(FAC_JSON.read_text(encoding="utf-8"))
    _, g = load_tables()
    fac = facility_codes(g)
    FAC_JSON.write_text(json.dumps(fac, ensure_ascii=False), encoding="utf-8")
    return fac


def _latest(paths: list[Path]) -> Path | None:
    return sorted(paths)[-1] if paths else None


def serve_sources(date_str: str | int) -> tuple[Path | None, Path | None]:
    """その開催日の KYI と外厩コメントの zip。発走前保存 (snapshots、最新の取得) を優先し、無ければアーカイブ。"""
    ymd = str(int(date_str))
    kyi = _latest(list(SNAP_KYI.glob(f"KYI{ymd[2:]}__*.zip"))) or (KYI_RAW / f"KYI{ymd[2:]}.zip" if (KYI_RAW / f"KYI{ymd[2:]}.zip").exists() else None)
    gk = _latest(list(SNAP_TM.glob(f"gaikyucomment_{ymd}__*.zip"))) or \
        (TM_RAW / f"gaikyucomment_{ymd}.zip" if (TM_RAW / f"gaikyucomment_{ymd}.zip").exists() else None)
    return kyi, gk


def add_serve_feats(df: pd.DataFrame, date_str: str | int, rid_col: str = "レースID(新/馬番無)", ban_col: str = "馬番") -> dict:
    """配信経路 (出走表の frame) に k8_* 列を in-place で足す。学習と同じ make_features を通す。

    戻り値: {"coverage", "gaikyu", "kyi_file", "gaikyu_file", "error"}。error が None でなければその日の v8 特徴は信用できない。
    外厩コメントが無い日は外厩なし (2013-15 の学習行と同じ扱い) として続け、error にはしない。
    """
    kyi_p, gk_p = serve_sources(date_str)
    if kyi_p is None:
        for c in FEATURES:
            df[c] = np.nan
        return {"coverage": 0.0, "gaikyu": 0.0, "kyi_file": None, "gaikyu_file": None,
                "error": f"JRDB 前日ファイル KYI{str(int(date_str))[2:]} が無い (python scripts/jrdb_snapshot.py capture)"}
    with zipfile.ZipFile(kyi_p) as z:
        rows = []
        for nm in z.namelist():
            rows += _parse_kyi_bytes(z.read(nm))
    k = kyi_frame(rows)
    g_rows = []
    if gk_p is not None:
        with zipfile.ZipFile(gk_p) as z:
            for nm in z.namelist():
                if not nm.endswith("/"):
                    g_rows += _gk_rows(nm, z.read(nm).decode("cp932", errors="replace"))
    g = pd.DataFrame(g_rows, columns=["racekey", "ban", "gk_fac", "gk_mmdd", "gk_weeks"]).drop_duplicates(["racekey", "ban"], keep="last")
    rid = df[rid_col].astype(str).str[:16]
    keys = pd.DataFrame({"racekey": rid16_to_racekey(rid), "ban": df[ban_col]}, index=df.index)
    gk_name = gk_p.name if gk_p else None
    if gk_p is None and GK_PQ.exists():          # 過去日の再生: 年単位アーカイブ (gaikyu.parquet) から引く
        ga = pd.read_parquet(GK_PQ)
        ga = ga[ga["racekey"].isin(set(keys["racekey"]))]
        if len(ga):
            g, gk_name = ga, "archive:gaikyu.parquet"
    f = make_features(keys, pd.Series(int(date_str), index=df.index), k, g, load_fac_map(), race_col=rid)
    for c in FEATURES:
        df[c] = f[c].values
    cov = coverage(df)
    err = None
    if cov["kyi_idm"] < KYI_COVERAGE_MIN:
        err = f"JRDB 前日 KYI の埋まりが {cov['kyi_idm'] * 100:.0f}% ({kyi_p.name})。レースキーの対応か取得日を確認"
    return {"coverage": cov["kyi_idm"], "gaikyu": cov["gaikyu"], "kyi_file": kyi_p.name,
            "gaikyu_file": gk_name, "error": err}


def coverage(df: pd.DataFrame) -> dict:
    return {"kyi_idm": float(df["k8_idm"].notna().mean()), "gaikyu": float(df["k8_gk_has"].mean())}
