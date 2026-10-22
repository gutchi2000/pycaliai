# -*- coding: utf-8 -*-
"""
build_substrate.py — 馬券判断 AI の学習用データ（substrate）
==========================================================
仕様: docs/research/BET_DECISION_AI_SPEC.md (v0.2) §2・§3・§4・§9・§10 手順 1
棚卸し: docs/research/BET_DECISION_AI_DATA_INVENTORY_20261022.md

行 = レース × 候補券。期間 2015-2025、平地のみ（トラックコード(JV) 51-59 = 障害を除外。
race_eligibility.py / EXP18 loaders と同一定義）。
  arm = "main"        : 単勝・複勝・馬連・ワイド・馬単・三連複・三連単（2015-2025）
  arm = "sub_wakuren" : 枠連（2021-2025 のみ。判断時点オッズが O1 原本にしか無いため）

**このスクリプトは方策の成績（回収率・的中率・ROI）を一切計算しない。** 検査は label-free のみ。

入力（すべて読み取りのみ）
  data/_research/mcond/base.parquet          v6 C1 の v6_score / v6_pwin（2015-2023 OOF、2024-25 本番 C1）
  data/master_v2_20130105-20251228.csv        レース情報（場・芝ダ・距離・クラス・トラックコード・発走時刻・枠番）
  data/Time _series_odds/TANPUK_*.csv         単勝・複勝 Lo/Hi の時系列（判断時点 = T−28 前売り断面）
  data/Time _series_odds/UMAREN_*.csv         馬連 全ペアの時系列（同上）
  data/Time _series_odds/{2011..,2016..,2021..}.csv  ヘッダ無し 227 列 OD 形式の確定オッズ（決済照合・縮み推定用）
  data/kekka_20130105-20251228.csv            公式払戻（1-3 着行。単勝・複勝・枠連・馬連・馬単・三連複・三連単）
  data/wide_payouts_2016-2025.parquet         ワイド公式払戻（3 組/レース、2016-2025）
  E:/競馬過去走データ/TAN_PUKU_WAKU_20210105-20260922.txt  JV O1 原本（枠連の判断時点オッズ、T−11 単複枠）
  data/harville_lambda.json                   λ 補正 PL（Lo-Bacon-Shone）
  data/chaos_quantiles.json                   本番の混戦度参照分布（参考列 chaos_pct_prodref 用）
使わない: data/oof_scores_v6params.parquet, data/deepvalue_races_oof.pkl（P0-5 リーク入り）

出力
  data/_research/bet_decision_ai/substrate.parquet        候補券行（race 列を結合済み）
  data/_research/bet_decision_ai/substrate_races.parquet  レース単位の文脈・検査列
  analysis/bet_decision_ai/out/substrate_checks.json      label-free 検査の数値
  analysis/bet_decision_ai/SUBSTRATE_REPORT.md            検査レポート（自動生成）

主な定義（詳細は SUBSTRATE_REPORT.md §1）
  p_model    : v6_pwin（レース内で和 1 に再正規化）→ λ 補正 PL
               P(a,b,c) = p_a · p_b^λ1/(Σ_{m≠a} p_m^λ1) · p_c^λ2/(Σ_{m≠a,b} p_m^λ2)
               （compute_bets.pl_pair_probs と同一式）。複勝 = 払戻対象着内確率（判断時点の頭数 ≥8 で 3 着内、
               5-7 頭で 2 着内）、ワイド = 2 頭とも 3 着内、枠連 = 馬連確率を枠の組で合算。
  q_market   : 単勝・馬連・枠連 = 1/odds をプール内で正規化（控除除去）。複勝 = 1/mid を正規化し和 = 払戻着数。
               ワイド・馬単・三連複・三連単 = 推定（q_src='est_tan_umaren'）:
                 馬単(i→j) = q馬連{i,j} · H(i→j)/(H(i→j)+H(j→i))   H = 市場単勝 π の λ 補正 PL
                 三連単(i→j→k) = 馬単(i→j) · π_k^λ2/Σ_{m≠i,j} π_m^λ2
                 ワイド・三連複は三連単の推定同時分布を足し上げ。
               馬連断面が無いレースは H だけで代用（q_src='est_tan_only'）。
  odds_decision : 実オッズ（単勝・馬連・枠連、複勝は mid と lo/hi）。推定券種は (1−控除率)/q_market。
  close_*    : 確定オッズ（発走後）。**判断の入力に使ってはいけない**。決済照合と縮み推定の専用列。
  payout     : 100 円あたりの確定払戻。外れ = 0。取消・除外を含む組 = 100（返還）。
               候補は「v6 スコアがある完走馬」だけから作るため、本 substrate では返還は発生しない（0 件）。
実行: python -m analysis.bet_decision_ai.build_substrate
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timedelta
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
sys.path.insert(0, str(BASE))

OUT_DIR = BASE / "data" / "_research" / "bet_decision_ai"
OUT = OUT_DIR / "substrate.parquet"
RACES_OUT = OUT_DIR / "substrate_races.parquet"
CHECKS_OUT = HERE / "out" / "substrate_checks.json"
REPORT = HERE / "SUBSTRATE_REPORT.md"

BASE_PQ = BASE / "data" / "_research" / "mcond" / "base.parquet"
MASTER = BASE / "data" / "master_v2_20130105-20251228.csv"
ODIR = BASE / "data" / "Time _series_odds"
OD_FILES = ["20110105-20151227.csv", "20160105-20201227.csv", "20210105-20251228.csv"]
KEKKA = BASE / "data" / "kekka_20130105-20251228.csv"
WIDE = BASE / "data" / "wide_payouts_2016-2025.parquet"
O1_RAW = Path(r"E:\競馬過去走データ\TAN_PUKU_WAKU_20210105-20260922.txt")
LAMBDA = BASE / "data" / "harville_lambda.json"
CHAOS_Q = BASE / "data" / "chaos_quantiles.json"
FORBIDDEN_INPUTS = [BASE / "data" / "oof_scores_v6params.parquet", BASE / "data" / "deepvalue_races_oof.pkl"]

YEARS = list(range(2015, 2026))
WAKU_YEARS = list(range(2021, 2026))
JUMP_MIN, JUMP_MAX = 51, 59
MIN_GAP_PRE = 15            # historical_pre_snapshot: 確定記録の 15 分以上前（analysis/mcond/market.py と同契約）
NMAX = 18
BET_TYPES = ["tan", "fuku", "umaren", "wide", "umatan", "sanrenpuku", "sanrentan", "wakuren"]
K = {"tan": 3, "fuku": 3, "umaren": 6, "wide": 6, "umatan": 6, "sanrenpuku": 10, "sanrentan": 10, "wakuren": 6}
TAKEOUT = {"tan": 0.20, "fuku": 0.20, "umaren": 0.225, "wakuren": 0.225, "wide": 0.225,
           "umatan": 0.25, "sanrenpuku": 0.25, "sanrentan": 0.275}
CLASS_MAP = {"500万": "1勝", "1000万": "2勝", "1600万": "3勝", "ｵｰﾌﾟﾝ": "OP", "OP(L)": "OP(L)",
             "Ｇ３": "G3", "Ｇ２": "G2", "Ｇ１": "G1", "重賞": "重賞(格付無)"}
# OD 227 列 → vals 配列（列 7..35, 37..226 を連結）内の位置（EXP21 loaders.split_vals と同一）
OD_TAN, OD_FLO, OD_FHI, OD_UMAREN, OD_WAKU, OD_WIDE, OD_UMATAN, OD_TRIO = 0, 1, 2, 3, 21, 29, 65, 83
TRIO_IDX = {i: {pr: n for n, pr in enumerate(combinations([x for x in range(1, NMAX + 1) if x != i], 2))}
            for i in range(1, NMAX + 1)}


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 22), b""):
            h.update(ch)
    return h.hexdigest()


# =============================================================== loaders
def load_race_meta() -> tuple[pd.DataFrame, dict]:
    cols = ["日付", "場所", "レースID(新/馬番無)", "発走時刻", "芝・ダ", "距離", "クラス名",
            "トラックコード(JV)", "出走頭数", "枠番", "馬番"]
    m = pd.read_csv(MASTER, encoding="utf-8-sig", dtype=str, usecols=cols)
    m["rid16"] = m["レースID(新/馬番無)"].str.replace(r"\D", "", regex=True).str[:16]
    m = m[m["rid16"].str.len() == 16]
    m = m[m["rid16"].str[:4].astype(int).isin(YEARS)].copy()
    m["ban"] = pd.to_numeric(m["馬番"], errors="coerce")
    m["waku"] = pd.to_numeric(m["枠番"], errors="coerce")
    wk = m.dropna(subset=["ban", "waku"])
    waku_map = {rid: dict(zip(g["ban"].astype(int), g["waku"].astype(int)))
                for rid, g in wk.groupby("rid16", sort=False)}
    r = m.groupby("rid16").first()
    out = pd.DataFrame(index=r.index)
    out["venue_code"] = out.index.str[8:10]
    out["venue"] = r["場所"].str.strip()
    out["surface"] = r["芝・ダ"].str.strip()
    out["distance"] = pd.to_numeric(r["距離"], errors="coerce").astype("Int64")
    out["class_raw"] = r["クラス名"].str.strip()
    out["class_norm"] = out["class_raw"].map(lambda x: CLASS_MAP.get(x, x))
    out["track_code"] = pd.to_numeric(r["トラックコード(JV)"], errors="coerce").astype("Int64")
    out["is_jump"] = ((out["track_code"] >= JUMP_MIN) & (out["track_code"] <= JUMP_MAX)).fillna(False).astype(bool)
    out["post_time"] = r["発走時刻"].str.strip()
    out["n_entries_master"] = pd.to_numeric(r["出走頭数"], errors="coerce").astype("Int64")
    return out, waku_map


def _ts_read(kind: str) -> pd.DataFrame:
    parts = []
    for f in sorted(ODIR.glob(f"{kind}_*.csv")):
        df = pd.read_csv(f, encoding="cp932", low_memory=False, skipinitialspace=True)
        c = list(df.columns)
        df = df.rename(columns={c[0]: "rid", c[1]: "kb", c[2]: "mdhm", c[3]: "tou", c[4]: "votes"})
        df["rid16"] = df["rid"].astype(str).str.replace(r"\D", "", regex=True).str[:16]
        df = df[df["rid16"].str.len() == 16]
        df = df[df["rid16"].str[:4].astype(int).isin(YEARS)]
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df["kb"] = pd.to_numeric(df["kb"], errors="coerce")
    df["mdhm"] = pd.to_numeric(df["mdhm"], errors="coerce")
    df = df.dropna(subset=["kb", "mdhm"]).copy()
    df["mdhm"] = df["mdhm"].astype(np.int64)
    df["ts"] = pd.to_datetime(df["rid16"].str[:4] + df["mdhm"].astype(str).str.zfill(8),
                              format="%Y%m%d%H%M", errors="coerce")
    return df


def select_snapshots(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """pre = 当日の区分1のうち確定(区分4)記録より 15 分以上前で最も遅いもの / final = 区分4 の最後"""
    fin = df[df["kb"] == 4].sort_values("ts").groupby("rid16").tail(1).set_index("rid16")
    mid = df[df["kb"] == 1].merge(fin[["ts"]].rename(columns={"ts": "final_ts"}),
                                  left_on="rid16", right_index=True, how="inner")
    gap = (mid["final_ts"] - mid["ts"]).dt.total_seconds() / 60.0
    same = (mid["mdhm"] // 10000) == mid["rid16"].str[4:8].astype(int)
    pre = mid[same & (gap >= MIN_GAP_PRE)].sort_values("ts").groupby("rid16").tail(1).set_index("rid16")
    return pre, fin


def load_tanpuk():
    df = _ts_read("TANPUK")
    cols = list(df.columns)
    tan_c, lo_c, hi_c = {}, {}, {}
    for x in cols:
        s = str(x).strip()
        if (mm := re.fullmatch(r"(\d+)単", s)):
            tan_c[int(mm.group(1))] = x
        elif (mm := re.fullmatch(r"(\d+)複Lo", s)):
            lo_c[int(mm.group(1))] = x
        elif (mm := re.fullmatch(r"(\d+)複Hi", s)):
            hi_c[int(mm.group(1))] = x
    assert len(tan_c) == NMAX and len(lo_c) == NMAX and len(hi_c) == NMAX, (len(tan_c), len(lo_c), len(hi_c))
    pre, fin = select_snapshots(df)

    def mats(d: pd.DataFrame) -> dict:
        out = {}
        for key, cmap in (("tan", tan_c), ("lo", lo_c), ("hi", hi_c)):
            a = np.zeros((len(d), NMAX))
            for b, c in cmap.items():
                a[:, b - 1] = pd.to_numeric(d[c], errors="coerce").fillna(0.0).to_numpy()
            out[key] = a
        out["idx"] = {rid: i for i, rid in enumerate(d.index)}
        out["ts"] = d["ts"].to_numpy()
        out["mdhm"] = d["mdhm"].to_numpy()
        return out
    n_snap = df[df["kb"] == 1].groupby("rid16").size()
    return mats(pre), mats(fin), n_snap


def load_umaren():
    df = _ts_read("UMAREN")
    pc = {}
    for x in df.columns:
        mm = re.fullmatch(r"馬(\d{2})-(\d{2})", str(x).strip())
        if mm:
            pc[(int(mm.group(1)), int(mm.group(2)))] = x
    assert len(pc) == 153, len(pc)
    pre, _fin = select_snapshots(df)
    a = np.zeros((len(pre), NMAX, NMAX))
    for (i, j), c in pc.items():
        v = pd.to_numeric(pre[c], errors="coerce").fillna(0.0).to_numpy()
        a[:, i - 1, j - 1] = v
        a[:, j - 1, i - 1] = v
    return {"odds": a, "idx": {rid: i for i, rid in enumerate(pre.index)}, "ts": pre["ts"].to_numpy()}


def load_od_close(race_ids: set) -> dict:
    """OD 227 列（区分4 = 確定）→ {rid16: (bans int[], waku int[], vals float32[n,219])}"""
    key2rid = {}
    for rid in race_ids:
        key2rid[f"{rid[8:10]}{rid[2:4]}{int(rid[10:12]):X}{int(rid[12:14]):X}{rid[14:16]}"] = rid
    usecols = [0, 1, 2, 4, 5] + list(range(7, 36)) + list(range(37, 227))
    dt = {0: str}
    for c in usecols[1:]:
        dt[c] = np.float32
    out = {}
    stats = {"rows": 0, "rows_used": 0, "kubun_non4": 0}
    for fn in OD_FILES:
        df = pd.read_csv(ODIR / fn, header=None, usecols=usecols, dtype=dt, encoding="cp932",
                         skipinitialspace=True)
        stats["rows"] += len(df)
        stats["kubun_non4"] += int((df[2] != 4).sum())
        k = df[0].astype(str).str.strip()
        df["rid16"] = k.str[:8].str.upper().map(key2rid)
        df = df[df["rid16"].notna()].copy()
        stats["rows_used"] += len(df)
        df["ban"] = df[4].astype(int)
        df = df.sort_values(["rid16", "ban"])
        vals = df[[c for c in usecols if c >= 7]].to_numpy(np.float32)
        bans = df["ban"].to_numpy(int)
        wakus = df[5].to_numpy(int)
        rids = df["rid16"].to_numpy()
        u, st = np.unique(rids, return_index=True)
        ends = list(st[1:]) + [len(rids)]
        for rid, a, b in zip(u, st, ends):
            out[rid] = (bans[a:b], wakus[a:b], vals[a:b])
        log(f"  OD {fn}: {len(df):,} rows / {len(u):,} races")
        del df, vals
    return out, stats


def load_kekka(race_ids: set) -> dict:
    k = pd.read_csv(KEKKA, encoding="cp932", dtype=str)
    k.columns = ["rid", "ban", "hid", "fin", "tan", "fuku", "waku", "umaren", "umatan", "sanpuku", "sanrentan"]
    k["rid16"] = k["rid"].astype(str).str[:16]
    k = k[k["rid16"].isin(race_ids)]
    for c in ["ban", "fin", "tan", "fuku", "waku", "umaren", "umatan", "sanpuku", "sanrentan"]:
        k[c] = pd.to_numeric(k[c], errors="coerce")      # 単勝配当の非勝ち馬 "(odds)" は NaN になる
    cols = ["ban", "fin", "tan", "fuku", "waku", "umaren", "umatan", "sanpuku", "sanrentan"]
    return {rid: g[cols].to_numpy(float) for rid, g in k.groupby("rid16", sort=False)}


def load_wide() -> dict:
    w = pd.read_parquet(WIDE)
    out = {}
    for r in w.itertuples(index=False):
        d = {}
        for s in (1, 2, 3):
            i, j, p = getattr(r, f"w{s}_i"), getattr(r, f"w{s}_j"), getattr(r, f"w{s}_pay")
            d[(min(i, j), max(i, j))] = float(p)
        out[str(r.race_id)] = d
    return out


# ---------------------------------------------------------- JV O1（枠連、T−11 単複枠）
# レイアウトは EXP22 o1_history.py（Codex/PyCaLiAI-exp24/analysis/mcond/exp22_synchronized_cross_pool_waku_dev）
# と同一。数値以外のトークン（取消 *、非発売 -、空白）は「オッズ無し」として 0 にする（候補は出走馬のみ）。
O1_LEN = 960
TAN_ODDS0, TAN_STRIDE = 45, 8
FUK_ODDS0, FUK_STRIDE = 269, 12
WAKU0, WAKU_STRIDE = 603, 9
WAKU_PAIRS = [(i, j) for i in range(1, 9) for j in range(i, 9)]


def _int(tok: str):
    t = tok.strip()
    return int(t) if t.isdigit() else None


def load_o1(race_ids: set) -> tuple[dict, dict]:
    out: dict = {}
    stats = {"records": 0, "records_used": 0, "bad_len": 0}
    with open(O1_RAW, "r", encoding="cp932", errors="strict", newline="") as fh:
        for line in fh:
            rec = line.rstrip("\r\n")
            if not rec.strip():
                continue
            stats["records"] += 1
            if len(rec) != O1_LEN or rec[:2] != "O1":
                stats["bad_len"] += 1
                continue
            rid = rec[11:27]
            if rid not in race_ids:
                continue
            stats["records_used"] += 1
            div = _int(rec[2:3])
            ts = datetime.strptime(rid[:4] + rec[27:35], "%Y%m%d%H%M")
            tan = np.zeros(NMAX); lo = np.zeros(NMAX); hi = np.zeros(NMAX)
            for h in range(NMAX):
                v = _int(rec[TAN_ODDS0 + h * TAN_STRIDE: TAN_ODDS0 + h * TAN_STRIDE + 4])
                if v:
                    tan[h] = v / 10.0
                s = FUK_ODDS0 + h * FUK_STRIDE
                a, b = _int(rec[s:s + 4]), _int(rec[s + 4:s + 8])
                if a and b and 0 < a <= b:
                    lo[h], hi[h] = a / 10.0, b / 10.0
            wk = {}
            for n, pr in enumerate(WAKU_PAIRS):
                s = WAKU0 + n * WAKU_STRIDE
                raw = rec[s:s + 2]
                if raw.strip() and raw != f"{pr[0]}{pr[1]}":
                    raise ValueError(f"waku slot mismatch {rid} {raw} {pr}")
                v = _int(rec[s + 2:s + 7])
                if v:
                    wk[pr] = v / 10.0
            out.setdefault(rid, []).append((div, ts, tan, lo, hi, wk))
    return out, stats


# =============================================================== PL
def pl_tensors(p: np.ndarray, l1: float, l2: float):
    """λ 補正 PL。A[i,j] = P(1着 i, 2着 j)、P3[i,j,k] = P(1着 i, 2着 j, 3着 k)。"""
    n = len(p)
    q1, q2 = p ** l1, p ** l2
    S1, S2 = q1.sum(), q2.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        A = p[:, None] * q1[None, :] / (S1 - q1)[:, None]
        np.fill_diagonal(A, 0.0)
        den = S2 - q2[:, None] - q2[None, :]
        P3 = A[:, :, None] * q2[None, None, :] / den[:, :, None]
    ii = np.arange(n)
    P3[ii, :, ii] = 0.0
    P3[:, ii, ii] = 0.0
    P3[ii, ii, :] = 0.0
    P3 = np.nan_to_num(P3, nan=0.0, posinf=0.0, neginf=0.0)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    return A, P3


def derived(A: np.ndarray, P3: np.ndarray) -> dict:
    top2 = A.sum(1) + A.sum(0)
    top3 = P3.sum((1, 2)) + P3.sum((0, 2)) + P3.sum((0, 1))
    UM = A + A.T
    S = P3.sum(2) + P3.sum(1) + P3.sum(0)
    W = S + S.T
    TRI = (P3 + P3.transpose(0, 2, 1) + P3.transpose(1, 0, 2) + P3.transpose(1, 2, 0)
           + P3.transpose(2, 0, 1) + P3.transpose(2, 1, 0))
    return {"top2": top2, "top3": top3, "UM": UM, "W": W, "TRI": TRI}


_IDX_CACHE: dict = {}


def idx_sets(n: int):
    if n not in _IDX_CACHE:
        pairs = np.array(list(combinations(range(n), 2)), int).reshape(-1, 2)
        opairs = np.array(list(permutations(range(n), 2)), int).reshape(-1, 2)
        trip = np.array(list(combinations(range(n), 3)), int).reshape(-1, 3)
        otrip = np.array(list(permutations(range(n), 3)), int).reshape(-1, 3)
        _IDX_CACHE[n] = (pairs, opairs, trip, otrip)
    return _IDX_CACHE[n]


def topk(values: np.ndarray, combos: np.ndarray, k: int) -> np.ndarray:
    """値の降順、同値は組の辞書順（馬番の小さい方）で決定的に上位 k 本の行番号"""
    keys = [combos[:, c] for c in range(combos.shape[1] - 1, -1, -1)] + [-values]
    order = np.lexsort(keys)
    return order[:k]


def eq100(odds: float, official: float):
    """round(確定オッズ×100) == 公式払戻。公式が無ければ NaN（照合対象外）、オッズが無ければ False。"""
    if official is None or np.isnan(official):
        return np.nan
    if odds is None or np.isnan(odds):
        return False
    return bool(round(odds * 100) == round(official))


# =============================================================== 決済（勝ち組の判定）
def winning_sets(fin: dict) -> dict:
    """確定着順（完走馬のみ）から JRA 規則どおりの的中組を列挙（同着対応）。"""
    top = {b: f for b, f in fin.items() if 1 <= f <= 3}
    tb = sorted(top)

    def better(f, excl):
        return sum(1 for b, fb in top.items() if b not in excl and fb < f)
    umatan = set()
    for x, y in permutations(tb, 2):
        fx, fy = top[x], top[y]
        if fx == 1 and fy <= 2 and fx <= fy and better(fy, {x, y}) == 0:
            umatan.add((x, y))
    sanrentan = set()
    for x, y, z in permutations(tb, 3):
        fx, fy, fz = top[x], top[y], top[z]
        if fx == 1 and fx <= fy <= fz and fy <= 2 and better(fy, {x, y, z}) == 0 and better(fz, {x, y, z}) == 0:
            sanrentan.add((x, y, z))
    wide = set()
    for x, y in combinations(tb, 2):
        if better(max(top[x], top[y]), {x, y}) <= 1:
            wide.add((x, y))
    return {"tan": {b for b, f in top.items() if f == 1},
            "umatan": umatan, "umaren": {tuple(sorted(t)) for t in umatan},
            "sanrentan": sanrentan, "sanrenpuku": {tuple(sorted(t)) for t in sanrentan},
            "wide": wide}


# =============================================================== main build
def build(smoke: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    for f in FORBIDDEN_INPUTS:
        assert f.name not in str(BASE_PQ)
    t0 = time.time()
    lam = json.loads(LAMBDA.read_text(encoding="utf-8"))
    l1, l2 = float(lam["lambda1"]), float(lam["lambda2"])
    cq = json.loads(CHAOS_Q.read_text(encoding="utf-8"))
    chaos_ref = np.array(cq["quantiles"]["field_chaos_score"], float)

    log("race meta (master_v2)")
    meta, waku_map = load_race_meta()
    log("base.parquet")
    b = pd.read_parquet(BASE_PQ, columns=["rid16", "ban", "year", "v6_src", "v6_score", "tau", "v6_pwin", "fin"])
    b = b[b["year"].isin(YEARS)]
    races_all = set(b["rid16"].unique())
    jump = set(meta.index[meta["is_jump"]])
    no_meta = races_all - set(meta.index)
    race_ids = sorted(races_all - jump - no_meta)
    if smoke:   # 動作確認用: 各年の先頭 smoke レースだけ
        race_ids = sorted(pd.Series(race_ids).groupby(pd.Series(race_ids).str[:4]).head(smoke))
    rset = set(race_ids)
    log(f"  base races={len(races_all):,}  jump={len(races_all & jump):,}  no_meta={len(no_meta)}  flat={len(race_ids):,}")
    bg = {rid: g.sort_values("ban") for rid, g in b[b["rid16"].isin(rset)].groupby("rid16", sort=False)}

    log("TANPUK / UMAREN time series")
    tp_pre, tp_fin, n_snap = load_tanpuk()
    um_pre = load_umaren()
    log("OD close odds (227 col)")
    od, od_stats = load_od_close(rset)
    log("kekka / wide payouts")
    kek = load_kekka(rset)
    wide_pay = load_wide()
    log("JV O1 raw (2021-2025)")
    o1, o1_stats = load_o1({r for r in race_ids if int(r[:4]) in WAKU_YEARS})
    log(f"  O1 records={o1_stats['records']:,} used={o1_stats['records_used']:,} races={len(o1):,}")

    rows = {c: [] for c in ["rid16", "arm", "bet_type", "combo", "h1", "h2", "h3", "cand_rank", "p_model",
                            "q_market", "q_src", "odds_decision", "odds_decision_lo", "odds_decision_hi",
                            "close_odds", "close_odds_lo", "close_odds_hi",
                            "odds_t11", "odds_t11_lo", "odds_t11_hi",
                            "payout", "payout_src", "settle_status"]}
    rrec = []
    log(f"race loop ({len(race_ids):,} races)")
    for ri, rid in enumerate(race_ids):
        if ri and ri % 5000 == 0:
            log(f"  {ri:,} races")
        g = bg[rid]
        bans = g["ban"].to_numpy(int)
        pw = g["v6_pwin"].to_numpy(float)
        p = pw / pw.sum()
        n = len(bans)
        fin = dict(zip(bans, g["fin"].to_numpy(int)))
        pos = {bn: i for i, bn in enumerate(bans)}
        year = int(rid[:4])
        mrow = meta.loc[rid]
        rr = {"rid16": rid, "year": year, "date": rid[:8], "v6_src": g["v6_src"].iloc[0],
              "tau": float(g["tau"].iloc[0]), "n_runners_model": n, "pwin_raw_sum": float(pw.sum())}
        if n < 3:
            rr["skip_reason"] = "n<3"
            rrec.append(rr)
            continue

        # ---------- post time
        post_dt = None
        mm = re.match(r"(\d{1,2}):(\d{2})", str(mrow["post_time"]))
        if mm:
            post_dt = datetime(year, int(rid[4:6]), int(rid[6:8]), int(mm.group(1)), int(mm.group(2)))

        def min_before_post(ts):
            if ts is None or post_dt is None or pd.isna(ts):
                return np.nan
            return (post_dt - pd.Timestamp(ts).to_pydatetime()).total_seconds() / 60.0

        # ---------- market: T−28 tan/fuku
        ti = tp_pre["idx"].get(rid)
        tan_pre = tp_pre["tan"][ti] if ti is not None else None
        lo_pre = tp_pre["lo"][ti] if ti is not None else None
        hi_pre = tp_pre["hi"][ti] if ti is not None else None
        tan_ts = tp_pre["ts"][ti] if ti is not None else None
        rr["snap_tan_mdhm"] = int(tp_pre["mdhm"][ti]) if ti is not None else np.nan
        rr["snap_tan_min_before_post"] = min_before_post(tan_ts)
        fi = tp_fin["idx"].get(rid)
        starters = set(int(x) + 1 for x in np.flatnonzero(tp_fin["tan"][fi] > 1.0)) if fi is not None else None
        rr["close_min_after_post"] = -min_before_post(tp_fin["ts"][fi]) if fi is not None else np.nan
        mkt_bans = [int(x) + 1 for x in np.flatnonzero(tan_pre > 1.0)] if tan_pre is not None else []
        rr["n_field_decision"] = len(mkt_bans) if mkt_bans else np.nan
        rr["n_starters_close"] = len(starters) if starters is not None else np.nan
        rr["n_dnf"] = len(starters - set(bans)) if starters is not None else np.nan
        rr["n_model_not_starter"] = len(set(bans) - starters) if starters is not None else np.nan
        rr["n_scratch_after_snap"] = len(set(mkt_bans) - starters) if (starters is not None and mkt_bans) else np.nan
        rr["n_model_unpriced_at_snap"] = len(set(bans) - set(mkt_bans)) if mkt_bans else n
        n_dec = len(mkt_bans) if mkt_bans else n
        fuku_places = 3 if n_dec >= 8 else 2
        fuku_sold = n_dec >= 5
        rr["fuku_places"] = fuku_places

        pi = {}
        if mkt_bans:
            inv = np.array([1.0 / tan_pre[x - 1] for x in mkt_bans])
            rr["overround_tan"] = float(inv.sum())
            pi = dict(zip(mkt_bans, inv / inv.sum()))
        fq, fmid = {}, {}
        if lo_pre is not None:
            fb = [x for x in mkt_bans if lo_pre[x - 1] > 0 and hi_pre[x - 1] > 0]
            if fb:
                mid = np.array([(lo_pre[x - 1] + hi_pre[x - 1]) / 2.0 for x in fb])
                inv = 1.0 / mid
                rr["overround_fuku"] = float(inv.sum() / fuku_places)
                fq = dict(zip(fb, inv / inv.sum() * fuku_places))
                fmid = dict(zip(fb, mid))
        # ---------- market: T−28 umaren
        ui = um_pre["idx"].get(rid)
        um_odds = um_pre["odds"][ui] if ui is not None else None
        rr["snap_um_min_before_post"] = min_before_post(um_pre["ts"][ui]) if ui is not None else np.nan
        rr["snap_um_minus_tan_min"] = ((pd.Timestamp(um_pre["ts"][ui]) - pd.Timestamp(tan_ts)).total_seconds() / 60.0
                                       if (ui is not None and tan_ts is not None) else np.nan)
        qum = {}
        if um_odds is not None:
            iu = np.triu_indices(NMAX, 1)
            ov = um_odds[iu]
            ok = ov > 0
            if ok.any():
                inv = 1.0 / ov[ok]
                rr["overround_umaren"] = float(inv.sum())
                for a, c, v in zip(iu[0][ok] + 1, iu[1][ok] + 1, inv / inv.sum()):
                    qum[(int(a), int(c))] = float(v)
        # ---------- market: 推定同時分布（単勝 π + 馬連 q）
        mk_pos, UTm, P3m = {}, None, None
        est_src = "missing"
        if len(pi) >= 3:
            mb = sorted(pi)
            mk_pos = {x: i for i, x in enumerate(mb)}
            piv = np.array([pi[x] for x in mb])
            Hm, _ = pl_tensors(piv, l1, l2)
            q2m = piv ** l2
            S2m = q2m.sum()
            ratio = np.divide(Hm, Hm + Hm.T, out=np.zeros_like(Hm), where=(Hm + Hm.T) > 0)
            nm = len(mb)
            UTm = np.zeros((nm, nm))
            nfb = 0
            for a in range(nm):
                for c in range(a + 1, nm):
                    qq = qum.get((mb[a], mb[c]))
                    if qq is None:
                        qq = Hm[a, c] + Hm[c, a]
                        nfb += 1
                    UTm[a, c] = qq * ratio[a, c]
                    UTm[c, a] = qq * ratio[c, a]
            est_src = "est_tan_umaren" if qum else "est_tan_only"
            rr["n_umaren_pair_fallback"] = nfb if qum else np.nan
            with np.errstate(divide="ignore", invalid="ignore"):
                den = S2m - q2m[:, None] - q2m[None, :]
                P3m = UTm[:, :, None] * q2m[None, None, :] / den[:, :, None]
            ii = np.arange(nm)
            P3m[ii, :, ii] = 0.0
            P3m[:, ii, ii] = 0.0
            P3m[ii, ii, :] = 0.0
            P3m = np.nan_to_num(P3m, nan=0.0, posinf=0.0, neginf=0.0)
            Dm = derived(UTm, P3m)
            rr["qsum_umatan_est"] = float(UTm.sum())
            rr["qsum_sanrentan_est"] = float(P3m.sum())
            rr["qsum_wide_est"] = float(np.triu(Dm["W"], 1).sum())
        rr["est_src"] = est_src

        # ---------- model PL
        A, P3 = pl_tensors(p, l1, l2)
        D = derived(A, P3)
        pairs, opairs, trip, otrip = idx_sets(n)
        rr["psum_win"] = float(p.sum())
        rr["psum_top3"] = float(D["top3"].sum())
        rr["psum_umaren"] = float(D["UM"][pairs[:, 0], pairs[:, 1]].sum())
        rr["psum_wide"] = float(D["W"][pairs[:, 0], pairs[:, 1]].sum())
        rr["psum_umatan"] = float(A.sum())
        rr["psum_sanrentan"] = float(P3.sum())
        rr["psum_sanrenpuku"] = float(D["TRI"][trip[:, 0], trip[:, 1], trip[:, 2]].sum())
        ent = -np.sum(p[p > 1e-12] * np.log(p[p > 1e-12]))
        rr["field_chaos_score"] = float(min(1.0, max(0.0, ent / np.log(n))))
        ps = np.sort(p)[::-1]
        rr["top1_p"], rr["top2_p"] = float(ps[0]), float(ps[1])
        rr["chaos_pct_prodref"] = float(np.interp(rr["field_chaos_score"], chaos_ref, np.arange(101) / 100.0))

        # ---------- results / payouts
        kr = kek.get(rid)
        ws = winning_sets(fin)
        topfins = sorted(f for f in fin.values() if 1 <= f <= 3)
        deadheat = topfins != [1, 2, 3]
        rr["deadheat_top3"] = deadheat
        krow = {}
        if kr is not None:
            for row in kr:
                krow[int(row[0])] = row
            k_top = sorted((int(r_[0]), int(r_[1])) for r_ in kr if 1 <= r_[1] <= 3)
            b_top = sorted((bn, f) for bn, f in fin.items() if 1 <= f <= 3)
            # kekka の 1 着行に単勝配当（数値）が無いレースは kekka 自体が壊れている（例: 2020110705050104）
            bad_tan = any(1 == r_[1] and np.isnan(r_[2]) for r_ in kr)
            rr["result_mismatch"] = (k_top != b_top) or bad_tan
        else:
            rr["result_mismatch"] = True
        odr = od.get(rid)
        odpos = {int(x): i for i, x in enumerate(odr[0])} if odr is not None else {}
        odv = odr[2] if odr is not None else None
        rr["od_close_found"] = odr is not None
        wmap = waku_map.get(rid, {})
        wide_src = wide_pay.get(rid)

        def od_val(h, off):
            if odv is None or h not in odpos:
                return np.nan
            v = float(odv[odpos[h], off])
            return v if v > 0 else np.nan

        def od_waku(wa, wb):
            if odv is None:
                return np.nan
            for h, wk_ in wmap.items():
                if wk_ == wa and h in odpos:
                    v = float(odv[odpos[h], OD_WAKU + wb - 1])
                    return v if v > 0 else np.nan
            return np.nan

        def od_trio(i, j, k):
            a, bb, c = sorted((i, j, k))
            if odv is None or a not in odpos:
                return np.nan
            v = float(odv[odpos[a], OD_TRIO + TRIO_IDX[a][(bb, c)]])
            return v if v > 0 else np.nan

        # 確定オッズと公式払戻の一致（同着なしレース・的中組のみ。label-free の照合）
        if not deadheat and not rr["result_mismatch"] and odv is not None:
            w1 = next(x for x, f in fin.items() if f == 1)
            w2 = next(x for x, f in fin.items() if f == 2)
            w3 = next(x for x, f in fin.items() if f == 3)
            r1, r3 = krow.get(w1), krow.get(w3)
            rr["chk_tan"] = eq100(od_val(w1, OD_TAN), r1[2])
            rr["chk_umaren"] = eq100(od_val(w1, OD_UMAREN + w2 - 1), r1[5])
            rr["chk_umatan"] = eq100(od_val(w1, OD_UMATAN + w2 - 1), r1[6])
            rr["chk_sanrenpuku"] = eq100(od_trio(w1, w2, w3), r3[7])
            if not np.isnan(r1[4]) and w1 in wmap and w2 in wmap:
                wa, wb = sorted((wmap[w1], wmap[w2]))
                rr["chk_wakuren"] = eq100(od_waku(wa, wb), r1[4])
            ins = []
            for x in (w1, w2, w3):
                fp = krow[x][3] if x in krow else np.nan
                if not np.isnan(fp):
                    lo_, hi_ = od_val(x, OD_FLO), od_val(x, OD_FHI)
                    ins.append(bool(lo_ * 100 - 0.5 <= fp <= hi_ * 100 + 0.5))
            rr["chk_fuku_in_range"] = float(np.mean(ins)) if ins else np.nan
            if wide_src:
                ins = []
                for (x, y), pay in wide_src.items():
                    lo_, hi_ = od_val(x, OD_WIDE + 2 * (y - 1)), od_val(x, OD_WIDE + 2 * (y - 1) + 1)
                    ins.append(bool(lo_ * 100 - 0.5 <= pay <= hi_ * 100 + 0.5))
                rr["chk_wide_in_range"] = float(np.mean(ins))
        if wide_src:
            rr["chk_wide_rule_eq_source"] = set(wide_src) == ws["wide"]
        if deadheat and odv is not None and not rr["result_mismatch"]:
            # 行対応の妥当性: 2 着同着の馬連 / 3 着同着の三連複は「その馬の行の払戻 ÷ 確定オッズ×100」が
            # 払戻分割（的中組 2 つ）でほぼ 1/2 になるはず。比が [0.3, 0.7] に入る率を記録
            f1 = [x for x, f in fin.items() if f == 1]
            f2 = [x for x, f in fin.items() if f == 2]
            f3 = [x for x, f in fin.items() if f == 3]
            ratios = []
            if len(f1) == 1 and len(f2) == 2:
                for y in f2:
                    o = od_val(f1[0], OD_UMAREN + y - 1)
                    if y in krow and not np.isnan(o):
                        ratios.append(krow[y][5] / (o * 100))
            if len(f1) == 1 and len(f2) == 1 and len(f3) == 2:
                for z in f3:
                    o = od_trio(f1[0], f2[0], z)
                    if z in krow and not np.isnan(o):
                        ratios.append(krow[z][7] / (o * 100))
            if ratios:
                rr["chk_dh_rowmap_half"] = float(np.mean([0.3 <= x <= 0.7 for x in ratios]))

        def settle(bt, combo):
            """(payout, src, status)"""
            if rr["result_mismatch"]:
                return np.nan, "result_mismatch", "unsettleable"
            if bt == "tan":
                h = combo[0]
                if h in ws["tan"]:
                    v = krow[h][2] if h in krow else np.nan
                    return (v, "kekka", "win") if not np.isnan(v) else (np.nan, "missing", "win_amount_missing")
                return 0.0, "lose", "lose"
            if bt == "fuku":
                h = combo[0]
                v = krow[h][3] if h in krow else np.nan
                if not np.isnan(v) and v > 0:
                    return v, "kekka", "win"
                return 0.0, "lose", "lose"
            if bt == "wide":
                key = tuple(sorted(combo))
                if wide_src is None:
                    return np.nan, "no_source", "no_payout_source"
                if key in ws["wide"]:
                    v = wide_src.get(key)
                    return (v, "wide_parquet", "win") if v is not None else (np.nan, "missing", "win_amount_missing")
                return 0.0, "lose", "lose"
            col = {"umaren": 5, "umatan": 6, "wakuren": 4, "sanrenpuku": 7, "sanrentan": 8}[bt]
            if bt == "wakuren":
                wpairs = {}
                for x, y in ws["umaren"]:
                    wpairs.setdefault(tuple(sorted((wmap.get(x, -1), wmap.get(y, -1)))), []).append((x, y))
                key = tuple(combo)
                if key not in wpairs:
                    return 0.0, "lose", "lose"
                wins = list(wpairs)
                members = wpairs[key]
            else:
                key = {"umaren": tuple(sorted(combo)), "umatan": tuple(combo),
                       "sanrenpuku": tuple(sorted(combo)), "sanrentan": tuple(combo)}[bt]
                if key not in ws[bt]:
                    return 0.0, "lose", "lose"
                wins = list(ws[bt])
                members = [key]
            h = payout_row(bt, key, wins, members)
            if h is None:
                return np.nan, "deadheat_ambiguous", "win_amount_missing"
            v = krow[h][col] if h in krow else np.nan
            src = "kekka" if not deadheat else "kekka_rowmap_deadheat"
            return (v, src, "win") if not np.isnan(v) else (np.nan, "missing", "win_amount_missing")

        def payout_row(bt, key, wins, members):
            """公式払戻が載っている kekka の行（馬番）。同着で 1 行に 1 つの値しか無く組を特定できない場合は None。
            kekka は 3 着以内の各行に「その馬を含む的中組」の配当を 1 つだけ持つ:
              馬連・馬単・枠連 = 1 着行（2 着同着ならその 2 着馬の行）、三連複・三連単 = 3 着行（3 着同着ならその 3 着馬の行）。
            1 着同着の馬単・三連単、2 着同着の三連単のように的中組が複数で行が組を区別しない場合は特定不能。"""
            if len(wins) == 1:
                k0 = members[0]
                if bt in ("umaren", "umatan", "wakuren"):
                    return max(k0, key=lambda x: (fin[x], x))          # 2 着側（同順なら任意、値は同じ）
                return max(k0, key=lambda x: (fin[x], x))              # 3 着側
            # 的中組が複数（同着）
            if bt == "wakuren":
                # 2 着同着で 2 着馬の枠が異なる: その 2 着馬の行に、その枠連の配当。1 着が複数なら特定不能
                f1 = [b_ for b_, f in fin.items() if f == 1]
                if len(members) != 1 or len(f1) != 1:
                    return None
                x, y = members[0]
                return y if fin[y] >= fin[x] else x
            if bt in ("umatan", "sanrentan"):
                prefix = {w[:-1] for w in wins}
                if len(prefix) == 1:
                    return key[-1]
                return None
            # umaren / sanrenpuku（順不同）: 共通部分を除いた残りが 1 頭なら、その馬の行
            common = set.intersection(*[set(w) for w in wins])
            rest = set(key) - common
            if len(rest) == 1 and len(set(w for w in wins)) == len(wins):
                return rest.pop()
            return None

        # ---------- O1 (2021-2025): 枠連の判断時点（T−28 単勝断面以前の最終 区分1）、T−11
        o1_dec = o1_t11 = None
        if year in WAKU_YEARS and rid in o1:
            snaps = sorted([s for s in o1[rid] if s[0] == 1], key=lambda s: s[1])
            if tan_ts is not None:
                tts = pd.Timestamp(tan_ts).to_pydatetime()
                c = [s for s in snaps if s[1] <= tts]
                o1_dec = c[-1] if c else None
            if post_dt is not None:
                c = [s for s in snaps if s[1] <= post_dt - timedelta(minutes=5)]
                o1_t11 = c[-1] if c else None
            rr["snap_waku_min_before_post"] = min_before_post(o1_dec[1]) if o1_dec else np.nan
            rr["snap_t11_min_before_post"] = min_before_post(o1_t11[1]) if o1_t11 else np.nan
            rr["snap_waku_minus_tan_min"] = ((o1_dec[1] - pd.Timestamp(tan_ts).to_pydatetime()).total_seconds() / 60.0
                                             if (o1_dec and tan_ts is not None) else np.nan)

        # ---------- emit helpers
        def emit(arm, bt, combos_list, pvals, qvals, qsrc, od_dec, od_lo, od_hi, cl, cl_lo, cl_hi, t11, t11lo, t11hi):
            for r_, (cmb, pv, qv, qs, o, olo, ohi, c, clo, chi, t, tlo, thi) in enumerate(
                    zip(combos_list, pvals, qvals, qsrc, od_dec, od_lo, od_hi, cl, cl_lo, cl_hi, t11, t11lo, t11hi), 1):
                pay, psrc, st = settle(bt, cmb)
                sep = ">" if bt in ("umatan", "sanrentan") else "-"
                rows["rid16"].append(rid); rows["arm"].append(arm); rows["bet_type"].append(bt)
                rows["combo"].append(sep.join(f"{x:02d}" for x in cmb))
                rows["h1"].append(cmb[0]); rows["h2"].append(cmb[1] if len(cmb) > 1 else -1)
                rows["h3"].append(cmb[2] if len(cmb) > 2 else -1)
                rows["cand_rank"].append(r_); rows["p_model"].append(pv); rows["q_market"].append(qv)
                rows["q_src"].append(qs); rows["odds_decision"].append(o)
                rows["odds_decision_lo"].append(olo); rows["odds_decision_hi"].append(ohi)
                rows["close_odds"].append(c); rows["close_odds_lo"].append(clo); rows["close_odds_hi"].append(chi)
                rows["odds_t11"].append(t); rows["odds_t11_lo"].append(tlo); rows["odds_t11_hi"].append(thi)
                rows["payout"].append(pay); rows["payout_src"].append(psrc); rows["settle_status"].append(st)

        NaN = np.nan

        def t11_tan(h):
            if o1_t11 is None:
                return NaN, NaN, NaN
            return (o1_t11[2][h - 1] or NaN), NaN, NaN

        def t11_fuku(h):
            if o1_t11 is None or not (o1_t11[3][h - 1] > 0):
                return NaN, NaN, NaN
            lo_, hi_ = o1_t11[3][h - 1], o1_t11[4][h - 1]
            return (lo_ + hi_) / 2.0, lo_, hi_

        def est_q(kind, cmb):
            if P3m is None or any(x not in mk_pos for x in cmb):
                return NaN
            ix = [mk_pos[x] for x in cmb]
            if kind == "umatan":
                return float(UTm[ix[0], ix[1]])
            if kind == "sanrentan":
                return float(P3m[ix[0], ix[1], ix[2]])
            if kind == "wide":
                return float(Dm["W"][ix[0], ix[1]])
            return float(Dm["TRI"][ix[0], ix[1], ix[2]])

        # 単勝
        sel = topk(p, np.arange(n)[:, None], K["tan"])
        cmb = [(int(bans[i]),) for i in sel]
        qv = [pi.get(c[0], NaN) for c in cmb]
        od_ = [tan_pre[c[0] - 1] if (tan_pre is not None and tan_pre[c[0] - 1] > 1.0) else NaN for c in cmb]
        t = [t11_tan(c[0]) for c in cmb]
        emit("main", "tan", cmb, p[sel], qv, ["real_odds" if not np.isnan(x) else "missing" for x in qv],
             od_, [NaN] * len(cmb), [NaN] * len(cmb),
             [od_val(c[0], OD_TAN) for c in cmb], [NaN] * len(cmb), [NaN] * len(cmb),
             [x[0] for x in t], [x[1] for x in t], [x[2] for x in t])
        # 複勝
        if fuku_sold:
            pf = D["top3"] if fuku_places == 3 else D["top2"]
            sel = topk(pf, np.arange(n)[:, None], K["fuku"])
            cmb = [(int(bans[i]),) for i in sel]
            qv = [fq.get(c[0], NaN) for c in cmb]
            t = [t11_fuku(c[0]) for c in cmb]
            emit("main", "fuku", cmb, pf[sel], qv, ["real_odds" if not np.isnan(x) else "missing" for x in qv],
                 [fmid.get(c[0], NaN) for c in cmb],
                 [lo_pre[c[0] - 1] if (lo_pre is not None and lo_pre[c[0] - 1] > 0) else NaN for c in cmb],
                 [hi_pre[c[0] - 1] if (hi_pre is not None and hi_pre[c[0] - 1] > 0) else NaN for c in cmb],
                 [(od_val(c[0], OD_FLO) + od_val(c[0], OD_FHI)) / 2.0 for c in cmb],
                 [od_val(c[0], OD_FLO) for c in cmb], [od_val(c[0], OD_FHI) for c in cmb],
                 [x[0] for x in t], [x[1] for x in t], [x[2] for x in t])
        # 馬連
        pv = D["UM"][pairs[:, 0], pairs[:, 1]]
        bp = bans[pairs]
        sel = topk(pv, bp, K["umaren"])
        cmb = [tuple(int(x) for x in bp[i]) for i in sel]
        qv = [qum.get(c, NaN) for c in cmb]
        od_ = [um_odds[c[0] - 1, c[1] - 1] if (um_odds is not None and um_odds[c[0] - 1, c[1] - 1] > 0) else NaN for c in cmb]
        nn = [NaN] * len(cmb)
        emit("main", "umaren", cmb, pv[sel], qv, ["real_odds" if not np.isnan(x) else "missing" for x in qv],
             od_, nn, nn, [od_val(c[0], OD_UMAREN + c[1] - 1) for c in cmb], nn, nn, nn, nn, nn)
        # ワイド
        pv = D["W"][pairs[:, 0], pairs[:, 1]]
        sel = topk(pv, bp, K["wide"])
        cmb = [tuple(int(x) for x in bp[i]) for i in sel]
        qv = [est_q("wide", c) for c in cmb]
        nn = [NaN] * len(cmb)
        emit("main", "wide", cmb, pv[sel], qv, [est_src if not np.isnan(x) else "missing" for x in qv],
             [(1 - TAKEOUT["wide"]) / x if x and not np.isnan(x) and x > 0 else NaN for x in qv], nn, nn,
             [(od_val(c[0], OD_WIDE + 2 * (c[1] - 1)) + od_val(c[0], OD_WIDE + 2 * (c[1] - 1) + 1)) / 2.0 for c in cmb],
             [od_val(c[0], OD_WIDE + 2 * (c[1] - 1)) for c in cmb],
             [od_val(c[0], OD_WIDE + 2 * (c[1] - 1) + 1) for c in cmb], nn, nn, nn)
        # 馬単
        pv = A[opairs[:, 0], opairs[:, 1]]
        bo = bans[opairs]
        sel = topk(pv, bo, K["umatan"])
        cmb = [tuple(int(x) for x in bo[i]) for i in sel]
        qv = [est_q("umatan", c) for c in cmb]
        nn = [NaN] * len(cmb)
        emit("main", "umatan", cmb, pv[sel], qv, [est_src if not np.isnan(x) else "missing" for x in qv],
             [(1 - TAKEOUT["umatan"]) / x if x and not np.isnan(x) and x > 0 else NaN for x in qv], nn, nn,
             [od_val(c[0], OD_UMATAN + c[1] - 1) for c in cmb], nn, nn, nn, nn, nn)
        # 三連複
        pv = D["TRI"][trip[:, 0], trip[:, 1], trip[:, 2]]
        bt_ = bans[trip]
        sel = topk(pv, bt_, K["sanrenpuku"])
        cmb = [tuple(int(x) for x in bt_[i]) for i in sel]
        qv = [est_q("sanrenpuku", c) for c in cmb]
        nn = [NaN] * len(cmb)
        emit("main", "sanrenpuku", cmb, pv[sel], qv, [est_src if not np.isnan(x) else "missing" for x in qv],
             [(1 - TAKEOUT["sanrenpuku"]) / x if x and not np.isnan(x) and x > 0 else NaN for x in qv], nn, nn,
             [od_trio(*c) for c in cmb], nn, nn, nn, nn, nn)
        # 三連単
        pv = P3[otrip[:, 0], otrip[:, 1], otrip[:, 2]]
        bt_ = bans[otrip]
        sel = topk(pv, bt_, K["sanrentan"])
        cmb = [tuple(int(x) for x in bt_[i]) for i in sel]
        qv = [est_q("sanrentan", c) for c in cmb]
        nn = [NaN] * len(cmb)
        emit("main", "sanrentan", cmb, pv[sel], qv, [est_src if not np.isnan(x) else "missing" for x in qv],
             [(1 - TAKEOUT["sanrentan"]) / x if x and not np.isnan(x) and x > 0 else NaN for x in qv], nn, nn,
             nn, nn, nn, nn, nn, nn)
        # 枠連（2021-2025、判断時点で発売・数値オッズがあるレースだけ）
        if o1_dec is not None and o1_dec[5] and all(x in wmap for x in bans):
            wk_odds = o1_dec[5]
            inv = {pr: 1.0 / o for pr, o in wk_odds.items()}
            sinv = sum(inv.values())
            rr["overround_waku"] = float(sinv)
            wp: dict = {}
            UMv = D["UM"]
            for a, c in pairs:
                key = tuple(sorted((wmap[int(bans[a])], wmap[int(bans[c])])))
                wp[key] = wp.get(key, 0.0) + float(UMv[a, c])
            rr["psum_wakuren"] = float(sum(wp.values()))
            keys = np.array(sorted(wp), int).reshape(-1, 2)
            vals = np.array([wp[tuple(k_)] for k_ in keys])
            sel = topk(vals, keys, K["wakuren"])
            cmb = [tuple(int(x) for x in keys[i]) for i in sel]
            qv = [inv[c] / sinv if c in inv else NaN for c in cmb]
            nn = [NaN] * len(cmb)
            t11w = [o1_t11[5].get(c, NaN) if o1_t11 is not None else NaN for c in cmb]
            emit("sub_wakuren", "wakuren", cmb, vals[sel], qv,
                 ["real_odds" if not np.isnan(x) else "missing" for x in qv],
                 [wk_odds.get(c, NaN) for c in cmb], nn, nn, [od_waku(*c) for c in cmb], nn, nn, t11w, nn, nn)
            rr["wakuren_emitted"] = True
        else:
            rr["wakuren_emitted"] = False if year in WAKU_YEARS else np.nan
        rrec.append(rr)

    log("assemble")
    races = pd.DataFrame(rrec)
    races = races.merge(meta.drop(columns=["is_jump"]), left_on="rid16", right_index=True, how="left")
    # 混戦度パーセンタイル（as-of: 前年までの substrate レースの分布。2015 は NaN）
    races["chaos_pct_asof"] = np.nan
    for y in YEARS:
        ref = np.sort(races.loc[races["year"] < y, "field_chaos_score"].dropna().to_numpy())
        if len(ref) == 0:
            continue
        m_ = races["year"] == y
        races.loc[m_, "chaos_pct_asof"] = np.searchsorted(ref, races.loc[m_, "field_chaos_score"], side="right") / len(ref)
    sub = pd.DataFrame(rows)
    for c in ["h1", "h2", "h3", "cand_rank"]:
        sub[c] = sub[c].astype(np.int16)
    for c in ["arm", "bet_type", "q_src", "payout_src", "settle_status"]:
        sub[c] = sub[c].astype("category")
    sub["log_p_over_q"] = np.log(sub["p_model"] / sub["q_market"])
    race_cols = ["year", "date", "venue_code", "venue", "surface", "distance", "class_raw", "class_norm",
                 "track_code", "post_time", "n_entries_master", "n_runners_model", "n_field_decision",
                 "n_starters_close", "n_dnf", "n_scratch_after_snap", "fuku_places", "v6_src",
                 "field_chaos_score", "chaos_pct_asof", "chaos_pct_prodref", "top1_p", "top2_p",
                 "overround_tan", "overround_fuku", "overround_umaren", "overround_waku",
                 "snap_tan_min_before_post", "snap_um_min_before_post", "snap_waku_min_before_post",
                 "snap_t11_min_before_post", "est_src", "deadheat_top3"]
    sub = sub.merge(races[["rid16"] + race_cols], on="rid16", how="left")
    order = (["rid16", "year", "date", "arm", "bet_type", "cand_rank", "combo", "h1", "h2", "h3",
              "p_model", "q_market", "q_src", "log_p_over_q", "odds_decision", "odds_decision_lo", "odds_decision_hi"]
             + [c for c in race_cols if c not in ("year", "date")]
             + ["odds_t11", "odds_t11_lo", "odds_t11_hi", "close_odds", "close_odds_lo", "close_odds_hi",
                "payout", "payout_src", "settle_status"])
    sub = sub[order]
    info = {"elapsed_sec": round(time.time() - t0, 1), "lambda": [l1, l2], "od_stats": od_stats, "o1_stats": o1_stats,
            "n_base_races": len(races_all), "n_jump_excluded": len(races_all & jump), "n_no_meta": len(no_meta),
            "n_flat_races": len(race_ids), "n_tanpuk_pre": len(tp_pre["idx"]), "n_umaren_pre": len(um_pre["idx"]),
            "tanpuk_snapshots_per_race_median": float(n_snap.median())}
    return sub, races, info


# =============================================================== label-free checks & report
def _q(s: pd.Series, qs=(0.01, 0.05, 0.5, 0.95, 0.99)) -> dict:
    s = s.dropna()
    if len(s) == 0:
        return {}
    return {f"p{int(q * 100):02d}": round(float(s.quantile(q)), 4) for q in qs} | {"n": int(len(s))}


def checks(sub: pd.DataFrame, races: pd.DataFrame, info: dict) -> dict:
    """label-free 検査。方策の成績（回収率・的中率・ROI）は計算しない。"""
    c: dict = {"build_info": info}
    r = races[races.get("skip_reason").isna()] if "skip_reason" in races else races
    c["n_rows"] = int(len(sub))
    c["n_races"] = int(sub["rid16"].nunique())
    c["races_skipped"] = int(len(races) - len(r))
    c["rows_by_type_year"] = sub.pivot_table(index="year", columns="bet_type", values="rid16",
                                             aggfunc="count", observed=True).fillna(0).astype(int).to_dict()
    c["races_by_type_year"] = sub.groupby(["year", "bet_type"], observed=True)["rid16"].nunique().unstack().fillna(0).astype(int).to_dict()
    c["races_by_year"] = r.groupby("year").size().to_dict()
    c["missing_rate_by_type"] = {bt: {col: round(float(g[col].isna().mean()), 5) for col in
                                      ["p_model", "q_market", "odds_decision", "close_odds", "payout", "odds_t11"]}
                                 for bt, g in sub.groupby("bet_type", observed=True)}
    c["q_src_by_type"] = {bt: g["q_src"].value_counts().loc[lambda s: s > 0].to_dict()
                          for bt, g in sub.groupby("bet_type", observed=True)}
    c["q_missing_by_type_year"] = sub.assign(m=sub["q_market"].isna()).pivot_table(
        index="year", columns="bet_type", values="m", aggfunc="mean", observed=True).round(4).to_dict()
    # 和の検査
    c["model_prob_sums"] = {col: _q(r[col]) for col in
                            ["pwin_raw_sum", "psum_win", "psum_top3", "psum_umaren", "psum_wide", "psum_umatan",
                             "psum_sanrentan", "psum_sanrenpuku", "psum_wakuren"] if col in r}
    c["market_overround"] = {col: {int(y): _q(g[col], (0.05, 0.5, 0.95)) for y, g in r.groupby("year")}
                             for col in ["overround_tan", "overround_fuku", "overround_umaren", "overround_waku"]
                             if col in r}
    c["market_est_sums"] = {col: _q(r[col]) for col in ["qsum_umatan_est", "qsum_sanrentan_est", "qsum_wide_est"]
                            if col in r}
    c["candidate_mass"] = {bt: _q(g.groupby("rid16")["p_model"].sum(), (0.05, 0.5, 0.95))
                           for bt, g in sub.groupby("bet_type", observed=True)}
    c["candidate_qmass"] = {bt: _q(g.groupby("rid16")["q_market"].sum(min_count=1), (0.05, 0.5, 0.95))
                            for bt, g in sub.groupby("bet_type", observed=True)}
    # p_model が候補順位で単調か
    s2 = sub.sort_values(["rid16", "bet_type", "cand_rank"])
    d = s2.groupby(["rid16", "bet_type"], observed=True)["p_model"].diff()
    c["p_model_rank_monotone_violations"] = int((d > 1e-12).sum())
    # 時点の一貫性
    c["timing_min_before_post"] = {col: {int(y): _q(g[col], (0.01, 0.1, 0.5, 0.9, 0.99)) for y, g in r.groupby("year")}
                                   for col in ["snap_tan_min_before_post", "snap_um_min_before_post",
                                               "snap_waku_min_before_post", "snap_t11_min_before_post",
                                               "close_min_after_post"] if col in r}
    c["timing_um_minus_tan_min"] = _q(r["snap_um_minus_tan_min"], (0.01, 0.05, 0.5, 0.95, 0.99))
    if "snap_waku_minus_tan_min" in r:
        c["timing_waku_minus_tan_min"] = _q(r["snap_waku_minus_tan_min"], (0.01, 0.05, 0.5, 0.95, 0.99))
    for col in ["snap_tan_min_before_post", "snap_um_min_before_post", "snap_waku_min_before_post"]:
        if col in r:
            c[f"{col}_le0_count"] = int((r[col] <= 0).sum())
            c[f"{col}_lt15_count"] = int((r[col] < 15).sum())
    # 確定オッズと公式払戻の一致（同着なし、的中組）
    chk = {}
    for col in ["chk_tan", "chk_umaren", "chk_umatan", "chk_sanrenpuku", "chk_wakuren",
                "chk_fuku_in_range", "chk_wide_in_range", "chk_wide_rule_eq_source", "chk_dh_rowmap_half"]:
        if col in r:
            chk[col] = {int(y): {"n": int(g[col].notna().sum()),
                                 "rate": (round(float(g[col].dropna().astype(float).mean()), 5)
                                          if g[col].notna().any() else None)} for y, g in r.groupby("year")}
    c["close_vs_official"] = chk
    # レース文脈・欠落
    c["race_flags"] = {
        "deadheat_top3": int(r["deadheat_top3"].sum()),
        "result_mismatch": int(r["result_mismatch"].sum()),
        "od_close_missing": int((~r["od_close_found"]).sum()),
        "n_dnf_races": int((r["n_dnf"] > 0).sum()), "n_dnf_horses": int(r["n_dnf"].sum()),
        "n_scratch_after_snap_races": int((r["n_scratch_after_snap"] > 0).sum()),
        "n_scratch_after_snap_horses": int(r["n_scratch_after_snap"].sum()),
        "n_model_unpriced_at_snap_races": int((r["n_model_unpriced_at_snap"] > 0).sum()),
        "no_tan_snapshot": int(r["snap_tan_min_before_post"].isna().sum()),
        "no_umaren_snapshot": int(r["snap_um_min_before_post"].isna().sum()),
        "fuku_places_2": int((r["fuku_places"] == 2).sum()),
        "est_src": r["est_src"].value_counts().to_dict(),
        "umaren_pair_fallback_races": int((r["n_umaren_pair_fallback"] > 0).sum()),
        "model_horse_not_close_starter": int(r["n_model_not_starter"].fillna(0).sum()),
    }
    lo = r.assign(low=r["overround_tan"] < 1.23)
    c["low_tan_overround_share"] = {"by_year": lo.groupby("year")["low"].mean().round(4).to_dict(),
                                    "by_class": lo.groupby("class_norm")["low"].mean().round(4).to_dict()}
    mism = {}
    for col in ["chk_umaren", "chk_umatan", "chk_sanrenpuku", "chk_wakuren", "chk_tan"]:
        x = r[r[col].notna() & (r[col].astype(float) == 0)]
        mism[col] = x["date"].value_counts().to_dict()
    for col in ["chk_fuku_in_range", "chk_wide_in_range"]:
        x = r[r[col].notna() & (r[col] < 1)]
        mism[col] = x["date"].value_counts().to_dict()
    c["close_vs_official_mismatch_dates"] = mism
    c["result_mismatch_rids"] = sorted(r.loc[r["result_mismatch"], "rid16"])
    w = sub[(sub["bet_type"] == "wide") & (sub["payout_src"] == "missing")]
    c["wide_amount_missing_deadheat"] = {"rows": int(len(w)), "in_deadheat_races": int(w["deadheat_top3"].sum())}
    # 判断時点断面の再現: base.parquet の mkt_odds_pre（analysis/mcond/market.py）と単勝 odds_decision の一致
    bm = pd.read_parquet(BASE_PQ, columns=["rid16", "ban", "mkt_odds_pre"])
    t = sub[sub["bet_type"] == "tan"][["rid16", "h1", "odds_decision"]].merge(
        bm, left_on=["rid16", "h1"], right_on=["rid16", "ban"], how="left")
    ok = t["mkt_odds_pre"].notna()
    c["tan_odds_vs_base_mkt_pre"] = {"n": int(ok.sum()),
                                     "equal_rate": round(float((np.isclose(t.loc[ok, "odds_decision"], t.loc[ok, "mkt_odds_pre"])).mean()), 6)}
    # PL 実装が本番 compute_bets.pl_pair_probs と一致するか（読み取り import のみ。200 レース標本）
    try:
        import compute_bets as cb
        lam = json.loads(LAMBDA.read_text(encoding="utf-8"))
        l1, l2 = float(lam["lambda1"]), float(lam["lambda2"])
        bb = pd.read_parquet(BASE_PQ, columns=["rid16", "ban", "v6_pwin"])
        mx = 0.0
        for rid in bb["rid16"].drop_duplicates().sample(200, random_state=0):
            g = bb[bb["rid16"] == rid].sort_values("ban")
            pp = g["v6_pwin"].to_numpy() / g["v6_pwin"].sum()
            um, wd = cb.pl_pair_probs([{"umaban": int(x), "p_win": float(y)} for x, y in zip(g["ban"], pp)])
            A_, P3_ = pl_tensors(pp, l1, l2)
            D_ = derived(A_, P3_)
            pos_ = {int(x): i for i, x in enumerate(g["ban"])}
            for (i, j), v in um.items():
                mx = max(mx, abs(D_["UM"][pos_[i], pos_[j]] - v), abs(D_["W"][pos_[i], pos_[j]] - wd[(i, j)]))
        c["pl_vs_compute_bets_max_abs_diff"] = float(mx)
    except Exception as e:  # noqa: BLE001
        c["pl_vs_compute_bets_max_abs_diff"] = f"skipped: {e!r}"
    c["wakuren_emitted_by_year"] = (r[r["year"] >= 2021].assign(w=lambda d: d["wakuren_emitted"].astype(float))
                                   .groupby("year")["w"].mean().round(4).to_dict())
    c["context_missing"] = {col: round(float(r[col].isna().mean()), 5) for col in
                            ["venue", "surface", "distance", "class_norm", "n_field_decision",
                             "field_chaos_score", "chaos_pct_asof", "chaos_pct_prodref"]}
    c["class_norm_counts"] = r["class_norm"].value_counts().to_dict()
    c["surface_counts"] = r["surface"].value_counts().to_dict()
    c["chaos"] = {"field_chaos_score": {int(y): _q(g["field_chaos_score"], (0.05, 0.5, 0.95)) for y, g in r.groupby("year")},
                  "chaos_pct_prodref": _q(r["chaos_pct_prodref"], (0.05, 0.25, 0.5, 0.75, 0.95))}
    # 決済列の状態（回収率は出さない。非ゼロ率と状態の件数だけ）
    c["settle_status_by_type"] = {bt: g["settle_status"].value_counts().loc[lambda s: s > 0].to_dict()
                                  for bt, g in sub.groupby("bet_type", observed=True)}
    c["payout_src_by_type"] = {bt: g["payout_src"].value_counts().loc[lambda s: s > 0].to_dict()
                               for bt, g in sub.groupby("bet_type", observed=True)}
    c["payout_nonzero_rate_by_type"] = {bt: round(float((g["payout"] > 0).mean()), 4)
                                        for bt, g in sub.groupby("bet_type", observed=True)}
    c["log_p_over_q"] = {bt: _q(g["log_p_over_q"].replace([np.inf, -np.inf], np.nan), (0.05, 0.5, 0.95))
                         for bt, g in sub.groupby("bet_type", observed=True)}
    return c


def _tbl(header: list, rows: list) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
    return "\n".join(out)


def write_report(c: dict, sub_cols: list, inputs: dict) -> None:
    bts = BET_TYPES
    years = sorted(int(y) for y in c["races_by_year"])
    L = []
    L.append("# 馬券判断 AI substrate 検査レポート（label-free）")
    L.append("")
    L.append(f"> 自動生成: `python -m analysis.bet_decision_ai.build_substrate`（{datetime.now():%Y-%m-%d %H:%M}）。"
             "仕様 `docs/research/BET_DECISION_AI_SPEC.md` v0.2 §10 手順 1。")
    L.append("> **方策の成績（回収率・的中率・ROI）は一切計算していない。** 決済列 payout は作ったが集計していない"
             "（非ゼロ率と決済状態の件数だけ記録）。")
    L.append("")
    L.append("## 0. 出力")
    L.append("")
    L.append(f"- `data/_research/bet_decision_ai/substrate.parquet`: **{c['n_rows']:,} 行 / {c['n_races']:,} レース**"
             "（`data/_research/` は .gitignore 済み＝commit 対象外。このスクリプトで再生成する）")
    L.append("- `data/_research/bet_decision_ai/substrate_races.parquet`: レース単位の文脈・検査列")
    L.append("- `analysis/bet_decision_ai/out/substrate_checks.json`: 本レポートの数値の正本")
    bi = c["build_info"]
    L.append(f"- 母集団: base.parquet 2015-2025 の {bi['n_base_races']:,}R − 障害 {bi['n_jump_excluded']:,}R"
             f"（トラックコード(JV) 51-59）− master 欠落 {bi['n_no_meta']}R = 平地 {bi['n_flat_races']:,}R"
             f"（うち頭数<3 で除外 {c['races_skipped']}R）")
    L.append("")
    L.append("## 1. 定義と判断したこと（仕様の曖昧点）")
    L.append("")
    L.append(DECISIONS)
    L.append("")
    L.append("## 2. 列一覧")
    L.append("")
    L.append(COLUMN_DOC)
    L.append("")
    L.append("## 3. 券種 × 年の被覆（レース数）")
    L.append("")
    rbt = c["races_by_type_year"]
    rows = []
    for y in years:
        rows.append([y, c["races_by_year"].get(y, c["races_by_year"].get(str(y), ""))] +
                    [rbt.get(bt, {}).get(y, 0) for bt in bts])
    L.append(_tbl(["年", "平地R"] + bts, rows))
    L.append("")
    L.append("候補数/レース: 単勝・複勝 3、馬連・ワイド・馬単・枠連 6、三連複・三連単 10。"
             "複勝は判断時点頭数 <5 のレースで出さない。枠連は 2021-2025 で判断時点の数値オッズがあるレースだけ。")
    L.append("")
    L.append("### 3a. q_market の出所（実オッズ / 推定）")
    L.append("")
    rows = [[bt, ", ".join(f"{k}={v:,}" for k, v in c["q_src_by_type"].get(bt, {}).items())] for bt in bts]
    L.append(_tbl(["券種", "q_src 件数"], rows))
    L.append("")
    L.append("### 3b. 欠損率（行単位）")
    L.append("")
    cols = ["p_model", "q_market", "odds_decision", "close_odds", "odds_t11", "payout"]
    rows = [[bt] + [f"{c['missing_rate_by_type'].get(bt, {}).get(col, 0):.4f}" for col in cols] for bt in bts]
    L.append(_tbl(["券種"] + cols, rows))
    L.append("")
    L.append("- close_odds: 三連単は確定オッズが無いので 100% 欠損（仕様どおり）。odds_t11: 単勝・複勝・枠連の 2021-2025 のみ。")
    L.append("- payout 欠損: ワイド 2015（払戻データ無し）と、払戻額が取れなかった的中組（下の決済状態を参照）。")
    L.append("")
    L.append("q_market 欠損率（年 × 券種）:")
    L.append("")
    qm = c["q_missing_by_type_year"]
    rows = [[y] + [f"{qm.get(bt, {}).get(y, float('nan')):.4f}" if y in qm.get(bt, {}) else "—" for bt in bts] for y in years]
    L.append(_tbl(["年"] + bts, rows))
    L.append("")
    L.append("## 4. 確率の和の検査")
    L.append("")
    L.append("### 4a. p_model（モデル、λ 補正 PL）— レース単位の和の分位")
    L.append("")
    exp = {"pwin_raw_sum": "≈1（base の v6_pwin 生の和）", "psum_win": "1", "psum_top3": "3", "psum_umaren": "1",
           "psum_wide": "3", "psum_umatan": "1", "psum_sanrentan": "1", "psum_sanrenpuku": "1", "psum_wakuren": "1"}
    rows = [[k, exp.get(k, ""), v.get("p01"), v.get("p50"), v.get("p99"), v.get("n")] for k, v in c["model_prob_sums"].items()]
    L.append(_tbl(["量", "期待", "p01", "p50", "p99", "n"], rows))
    L.append("")
    L.append(f"候補順位と p_model の単調性違反: **{c['p_model_rank_monotone_violations']} 件**。")
    L.append("")
    L.append("### 4b. 市場（判断時点 T−28）の控除込み和 Σ1/odds（年別の中央値 [p05, p95]）")
    L.append("")
    L.append("期待値: 単勝 ≈ 1/(1−0.20)=1.25、複勝（Σ1/mid ÷ 払戻着数）≈ 1.25 前後（Lo/Hi の幅のため粗い）、"
             "馬連・枠連 ≈ 1/(1−0.225)=1.29。")
    L.append("")
    mo = c["market_overround"]
    rows = []
    for y in years:
        row = [y]
        for col in ["overround_tan", "overround_fuku", "overround_umaren", "overround_waku"]:
            v = mo.get(col, {}).get(y) or mo.get(col, {}).get(str(y))
            row.append(f"{v['p50']:.3f} [{v['p05']:.3f}, {v['p95']:.3f}]" if v else "—")
        rows.append(row)
    L.append(_tbl(["年", "単勝", "複勝", "馬連", "枠連"], rows))
    L.append("")
    L.append("### 4c. 推定市場確率（ワイド・馬単・三連複・三連単）の和")
    L.append("")
    rows = [[k, v.get("p01"), v.get("p50"), v.get("p99"), v.get("n")] for k, v in c["market_est_sums"].items()]
    L.append(_tbl(["量（期待: 馬単・三連単 1、ワイド 3）", "p01", "p50", "p99", "n"], rows))
    L.append("")
    L.append("### 4d. 候補 K 本が占める確率質量（レースごとの和の中央値 [p05, p95]）")
    L.append("")
    rows = []
    for bt in bts:
        a, b_ = c["candidate_mass"].get(bt, {}), c["candidate_qmass"].get(bt, {})
        rows.append([bt, f"{a.get('p50', '')} [{a.get('p05', '')}, {a.get('p95', '')}]" if a else "—",
                     f"{b_.get('p50', '')} [{b_.get('p05', '')}, {b_.get('p95', '')}]" if b_ else "—"])
    L.append(_tbl(["券種", "Σp_model（候補）", "Σq_market（候補）"], rows))
    L.append("")
    L.append("log(p/q) の分位（券種別、p05 / p50 / p95）: " + "; ".join(
        f"{bt} {v.get('p05')}/{v.get('p50')}/{v.get('p95')}" for bt, v in c["log_p_over_q"].items() if v))
    L.append("")
    L.append("## 5. 確定オッズ（OD 227 列）と公式払戻の一致（同着なしレースの的中組、label-free の照合）")
    L.append("")
    L.append("単勝・馬連・馬単・三連複・枠連は `round(確定オッズ×100) == 公式払戻` の一致率。複勝・ワイドは公式払戻が"
             "確定オッズの [Lo, Hi]×100 に入る率。ワイドの「規則で列挙した的中組 == 払戻データの組」も併記。")
    L.append("")
    chk = c["close_vs_official"]
    names = ["chk_tan", "chk_umaren", "chk_umatan", "chk_sanrenpuku", "chk_wakuren", "chk_fuku_in_range",
             "chk_wide_in_range", "chk_wide_rule_eq_source"]
    rows = []
    for y in years:
        row = [y]
        for nm in names:
            v = chk.get(nm, {}).get(y) or chk.get(nm, {}).get(str(y))
            row.append(f"{v['rate']:.4f} ({v['n']})" if v and v["rate"] is not None else "—")
        rows.append(row)
    L.append(_tbl(["年", "単勝", "馬連", "馬単", "三連複", "枠連", "複勝∈[Lo,Hi]", "ワイド∈[Lo,Hi]", "ワイド組一致"], rows))
    L.append("")
    dh = chk.get("chk_dh_rowmap_half", {})
    n_dh = sum(v["n"] for v in dh.values())
    ok_dh = sum((v["rate"] or 0) * v["n"] for v in dh.values())
    L.append(f"同着の行対応の妥当性（2 着同着の馬連・3 着同着の三連複で、行の払戻 ÷ 確定オッズ×100 が [0.3, 0.7]＝"
             f"払戻 2 分割に整合するレースの割合の平均）: {ok_dh / n_dh if n_dh else float('nan'):.3f}（{n_dh} レース）。")
    L.append("同着レースでは確定オッズ（同着前の値）≠ 払戻なので、この照合は同着なしレースだけで行い、同着の決済は kekka の行対応で行った（§1-9）。")
    L.append("")
    L.append("## 6. 時点の一貫性")
    L.append("")
    L.append("発走（master の発走時刻）から見た断面時刻（分前、年別の p01 / p10 / p50 / p90 / p99）:")
    L.append("")
    tm = c["timing_min_before_post"]
    for col, lab in [("snap_tan_min_before_post", "単勝・複勝 T−28 断面"), ("snap_um_min_before_post", "馬連 T−28 断面"),
                     ("snap_waku_min_before_post", "枠連 判断断面（O1）"), ("snap_t11_min_before_post", "O1 T−11 断面"),
                     ("close_min_after_post", "確定記録（発走後、分）")]:
        if col not in tm:
            continue
        rows = []
        for y in years:
            v = tm[col].get(y) or tm[col].get(str(y))
            if v:
                rows.append([y, v["p01"], v["p10"], v["p50"], v["p90"], v["p99"], v["n"]])
        L.append(f"**{lab}**")
        L.append("")
        L.append(_tbl(["年", "p01", "p10", "p50", "p90", "p99", "n"], rows))
        L.append("")
    L.append(f"- 馬連断面 − 単勝断面（分）: {c['timing_um_minus_tan_min']}")
    if "timing_waku_minus_tan_min" in c:
        L.append(f"- 枠連判断断面 − 単勝断面（分）: {c['timing_waku_minus_tan_min']}")
    for col in ["snap_tan_min_before_post", "snap_um_min_before_post", "snap_waku_min_before_post"]:
        if f"{col}_le0_count" in c:
            L.append(f"- {col}: 発走時刻以降 {c[f'{col}_le0_count']} 件 / 15 分前より後 {c[f'{col}_lt15_count']} 件")
    L.append("")
    L.append("## 7. レース単位のフラグ")
    L.append("")
    for k, v in c["race_flags"].items():
        L.append(f"- {k}: {v}")
    L.append(f"- 枠連の行が出たレースの割合（年別）: {c['wakuren_emitted_by_year']}")
    L.append(f"- 文脈列の欠損率: {c['context_missing']}")
    L.append(f"- クラス（正規化後）: {c['class_norm_counts']}")
    L.append(f"- 芝ダ: {c['surface_counts']}")
    L.append(f"- 混戦度 field_chaos_score（年別 p05/p50/p95）: "
             + "; ".join(f"{y}: {v.get('p05')}/{v.get('p50')}/{v.get('p95')}" for y, v in c["chaos"]["field_chaos_score"].items()))
    L.append(f"- chaos_pct_prodref の分布: {c['chaos']['chaos_pct_prodref']}")
    L.append("")
    L.append("## 8. 決済列の状態（回収率は計算しない）")
    L.append("")
    rows = [[bt, ", ".join(f"{k}={v:,}" for k, v in c["settle_status_by_type"].get(bt, {}).items()),
             ", ".join(f"{k}={v:,}" for k, v in c["payout_src_by_type"].get(bt, {}).items()),
             c["payout_nonzero_rate_by_type"].get(bt)] for bt in bts]
    L.append(_tbl(["券種", "settle_status", "payout_src", "payout>0 の行割合"], rows))
    L.append("")
    L.append("## 9. 異常・注意")
    L.append("")
    L.append(ANOMALY_NOTES)
    L.append("")
    L.append("## 10. 入力ファイル sha256")
    L.append("")
    for k, v in inputs.items():
        L.append(f"- `{k}`: `{v}`")
    L.append("")
    REPORT.write_text("\n".join(L), encoding="utf-8")


DECISIONS = """\
1. **母集団**: `base.parquet`（v6 C1 の OOF 2015-2023 + 本番 C1 2024-2025）のレース。障害は master の `トラックコード(JV)` 51-59
   で除外（`race_eligibility.py`・EXP18 と同一定義。master の `芝・ダ` には障害の値が無いのでトラックコードを使う）。
2. **p_model**: base の `v6_pwin`（レース内で温度 τ のソフトマックス。OOF τ=0.845、本番 τ=0.865）をレース内で和 1 に
   再正規化し、`data/harville_lambda.json` の λ1=0.8405・λ2=0.7542 で PL（`compute_bets.pl_pair_probs` と同一式）。
   λ は 2026 本番の較正済み p_win で fit した値で、OOF の尺度とは厳密には合わない（仕様 §2 の指定どおり流用）。
   **年をまたぐ確率の尺度合わせ（過去だけで再較正）はしていない**（棚卸し §3 注意 3。学習段階で行う）。生の `v6_score` は
   substrate に入れていない（必要なら base と rid16×馬番で結合できる）。
3. **複勝の p_model**: 判断時点の頭数（T−28 断面で単勝オッズがある馬の数）が 8 以上なら 3 着内、5-7 なら 2 着内の確率。
   4 頭以下は複勝の行を出さない。**ワイド**は常に 3 着内の 2 頭。**枠連**は馬連確率を枠の組（同枠のゾロ目を含む）で合算。
4. **候補の作り方**: 券種ごとに p_model の降順、同値は馬番の辞書順で上位 K（単勝・複勝 3、馬連・ワイド・馬単・枠連 6、
   三連複・三連単 10）。枠連の K は仕様に無いので馬連と同じ 6 にした。
5. **判断時点**: `TANPUK` / `UMAREN` の区分 1 のうち、当日かつ確定記録の 15 分以上前で最も遅い断面（`analysis/mcond/market.py`
   と同契約＝棚卸しの T−28 断面）。単勝と馬連は別ファイルなので、断面時刻の差を §6 に記録。
6. **q_market（実オッズ）**: 単勝・馬連 = 1/odds をプール全体（判断時点でオッズのある全馬・全組）で正規化。出走取消前の
   馬も判断時点で見えているので母集団に含める。複勝 = 1/mid（mid=(Lo+Hi)/2）を正規化し和を払戻着数（3 か 2）にした。
   **枠連** = O1 原本の区分 1 のうち「単勝 T−28 断面の時刻以前で最も遅い」断面（= S0、発走約 30 分前）の 1/odds を正規化。
   派生 CSV `WAKU_20210105-20260922.csv` には T−11 と T−1 しか無いので原本を読んだ（パーサは EXP22 `o1_history.py` と同じ
   レイアウトを本スクリプト内に複製）。
7. **q_market（推定、q_src=est_tan_umaren）**: 本番 T−10 でも同じ情報（単勝と馬連のオッズ）で計算できる作り方。
   市場単勝確率 π に同じ λ の PL を当て H とし、馬単(i→j) = q馬連{i,j} × H(i→j)/(H(i→j)+H(j→i))、
   三連単(i→j→k) = 馬単(i→j) × π_k^λ2/Σ_{m∉{i,j}} π_m^λ2、ワイド・三連複はこの同時分布の足し上げ。
   馬連断面が無いレースは H だけ（q_src=est_tan_only）、組単位で馬連オッズが 0 の組は H で補完（件数を記録）。
   推定券種の odds_decision = (1−控除率)/q_market（控除率: 単複 20%、馬連・枠連・ワイド 22.5%、馬単・三連複 25%、三連単 27.5%）。
   **これは市場の価格ではなく単勝・馬連からの合成値**。
8. **混戦度**: 本番と同じ式の `field_chaos_score` = p_win の正規化エントロピー（`export_marks_json.race_confidence`）。
   パーセンタイルは 2 種: `chaos_pct_asof` = その年より前の substrate レースの分布での位置（2015 は NaN、先読み無し）、
   `chaos_pct_prodref` = 本番の凍結参照分布 `data/chaos_quantiles.json`（2026 serve34、較正済み p_win 由来）での位置。
   後者は確率の尺度が違うので参考値（OOF の softmax p_win は本番より平らで、chaos が高めに出る）。
9. **payout（100 円あたり）**: 的中組は確定着順から JRA の同着規則どおり列挙（`winning_sets`）。金額は
   - 単勝・複勝: `kekka_20130105-20251228.csv` の各馬の行（公式）。外れ・払戻対象外の 3 着（5-7 頭立て）= 0。
   - 馬連・馬単・枠連・三連複・三連単（同着なし）: 公式払戻（1 着行 / 3 着行）。
   - 同着レース: kekka は 3 着以内の各行に「その馬を含む的中組」の配当を 1 つだけ持つので、行対応で取る
     （2 着同着の馬連・馬単・枠連 = その 2 着馬の行、3 着同着の三連複・三連単 = その 3 着馬の行、的中組が 1 つなら任意の行）。
     **1 着同着の馬単・三連単、2 着同着の三連単のように的中組が複数あるのに値が 1 つしか載っていない組は特定不能で
     payout=NaN（payout_src=deadheat_ambiguous、settle_status=win_amount_missing）**。確定オッズは同着で払戻が分割される前の
     値なので、同着レースの決済には使わない（実データで確認: 1 着同着の馬単は確定オッズのほぼ半分が払戻）。
   - kekka の 1 着行に数値の単勝配当が無い、または base と kekka の 3 着以内が食い違うレースは result_mismatch として
     全候補を payout=NaN（unsettleable）にした。
   - ワイド: `wide_payouts_2016-2025.parquet`（2015 は払戻データが無いので payout=NaN、settle_status=no_payout_source）。
     3 着同着でワイドの的中組が 5 組になるレースは parquet が 3 組しか持たないので、欠けた組は payout=NaN（win_amount_missing）。
   - **取消・除外を含む組 = 100（返還）**が規則だが、候補は v6 スコアのある完走馬だけから作るので本 substrate では 0 件。
     ただし判断時点に見えていて後で取消・除外になった馬（n_scratch_after_snap）と競走中止馬（n_dnf）は候補から落ちている
     （結果を見た選別＝ごく小さな先読み。仕様 §9 の DNF 注記と同じ種類）。
10. **確定オッズ（close_*）**: OD 227 列ファイル（区分 4）。単勝・複勝 Lo/Hi・馬連・ワイド Lo/Hi・馬単・三連複・枠連。三連単は無し。
    **判断の入力に使ってはいけない**（決済照合と §4 の「T−28→確定の縮み」を学習期間内で推定する用途だけ）。
11. **odds_t11**: O1 原本の「発走 5 分前以前で最も遅い区分 1 断面」（= S1、発走約 11 分前）。単勝・複勝・枠連の 2021-2025 のみ。
    本番 T−10 との時点差を後で確認するための参考列で、主の判断時点（T−28）ではない。
12. 枠連は同じ parquet に `arm='sub_wakuren'` として入れた（列は共通）。"""

COLUMN_DOC = """\
| 列 | 意味 |
|---|---|
| rid16, year, date | レースキー（YYYYMMDD+場+回+日+R）、年、日付 |
| arm | `main`（2015-2025 の 7 券種）/ `sub_wakuren`（枠連 2021-2025） |
| bet_type | tan, fuku, umaren, wide, umatan, sanrenpuku, sanrentan, wakuren |
| cand_rank | 券種内のモデル確率順位（1 = 最大） |
| combo, h1, h2, h3 | 組（順序付きは `>`、順不同は `-`）。h は馬番（枠連は枠番）、無い位置は -1 |
| p_model | モデル確率（§1-2〜4） |
| q_market | 判断時点の市場確率（控除除去済み。§1-6, 7） |
| q_src | `real_odds` / `est_tan_umaren` / `est_tan_only` / `missing` |
| log_p_over_q | log(p_model / q_market) |
| odds_decision, odds_decision_lo, odds_decision_hi | 判断時点オッズ。複勝は mid と Lo/Hi。推定券種は (1−控除率)/q |
| venue_code, venue, surface, distance, class_raw, class_norm, track_code, post_time | レース情報（master_v2）。class_norm は 500万→1勝 等に統一 |
| n_entries_master | master の出走頭数 |
| n_runners_model | v6 スコアのある馬の数（完走馬） |
| n_field_decision | 判断時点（T−28）で単勝オッズがある馬の数 |
| n_starters_close / n_dnf / n_scratch_after_snap | 確定時の出走馬数（確定単勝オッズ >1.0）/ 出走したが v6 スコアが無い馬（競走中止＋master 欠落）/ 判断時点後の取消・除外 |
| fuku_places | 複勝の払戻着数（判断時点頭数 ≥8 で 3、5-7 で 2） |
| v6_src | `oof`（2015-2023）/ `prod`（2024-2025） |
| field_chaos_score, chaos_pct_asof, chaos_pct_prodref | 混戦度（p_win の正規化エントロピー）とそのパーセンタイル 2 種 |
| top1_p, top2_p | p_win の 1 位・2 位 |
| overround_tan / _fuku / _umaren / _waku | 判断時点の Σ1/odds（複勝は ÷払戻着数） |
| snap_*_min_before_post | 各断面が発走の何分前か |
| est_src | そのレースの推定市場確率の出所 |
| deadheat_top3 | 3 着以内に同着があるか（結果列） |
| odds_t11, odds_t11_lo, odds_t11_hi | O1 の T−11 断面（単勝・複勝・枠連、2021-2025）。参考 |
| close_odds, close_odds_lo, close_odds_hi | 確定オッズ（発走後）。**入力禁止**。複勝・ワイドは mid と Lo/Hi |
| payout | 100 円あたり確定払戻（外れ 0、返還 100、取れない場合 NaN） |
| payout_src | `kekka` / `wide_parquet` / `kekka_rowmap_deadheat` / `deadheat_ambiguous` / `lose` / `no_source` / `missing` / `result_mismatch` |
| settle_status | `win` / `lose` / `no_payout_source` / `win_amount_missing` / `unsettleable` |

結果に依存する列（**学習の入力にしてはいけない**）: payout, payout_src, settle_status, close_odds*, deadheat_top3,
n_starters_close, n_dnf, n_scratch_after_snap。"""

ANOMALY_NOTES = """数値は上の節と `out/substrate_checks.json` が正本。ここは読み方の注意（2026-10-23 の build で確認）。

1. **確率の尺度が年で揺れる**: field_chaos_score の中央値が 2015 だけ 0.886（他年 0.81-0.85）。2015 の OOF は 2013 だけで
   学習したモデルなので p_win が平らになっている。`chaos_pct_asof` の 2016 は 2015 だけを参照分布にしているので偏る。
   学習段階で「各年 Y を ≤Y−1 だけで較正し直す」（棚卸し §3 注意 3）までは、年をまたいで p_model・混戦度の水準を比べない。
2. **単勝の Σ1/odds が 1.23 未満のレース**が 2017-2023 で年 15-17%、2015-16 で約 5%、2024-25 で 1-2%。新馬 52%・未勝利 16% に
   集中し、確定オッズでも同水準なので断面の取り違えではなくプール側の性質（1.1 倍の下限に張り付いた大本命など。原因は未確定）。
   q_market はプール内で正規化しているので和 1 は保たれるが、overround_tan はレース文脈の列として残してある。
3. **払戻が確定オッズと合わない開催日**（キャンペーン等で公式払戻が上乗せされた日と推定）: 2023-12-28 の 23R で馬連・馬単・
   三連複・枠連の公式払戻が確定オッズ×100 より 2-7% 高い（単勝は一致）。ワイドは 2022-12-28（13R）・2023-12-28（11R）・
   2022-05-29（4R）で公式払戻が確定 [Lo, Hi] の外。payout は公式（kekka / ワイド parquet）を使っているので決済は正しいが、
   これらの日は close_odds ≠ payout/100。棚卸しの「2023 年だけ 3,418/3,442」はこの 1 日分。
4. **result_mismatch（16R）**: base（master_v2 由来）に 3 着以内の馬の行が無い 15R と、kekka の 1 着行が壊れている 1R
   （2020110705050104）。全候補の payout=NaN（settle_status=unsettleable）。結果を見た除外なので件数（0.04%）を明記して学習から外す。
5. **同着で払戻額を特定できない的中組**: 馬単 34 行・三連単 22 行（1 着同着・2 着同着）、ワイド 27 行（3 着同着で的中 5 組の
   うち parquet に無い 2 組）。いずれも payout=NaN、settle_status=win_amount_missing。行対応で取った同着の払戻
   （kekka_rowmap_deadheat）は「確定オッズのほぼ半分」になっていることを 162R で確認（§5）。
6. **ワイド 2015** は払戻データが無い（19,950 行が no_payout_source）。ワイドの決済は 2016-2025 だけ。
7. **候補の母集団の先読み**: 判断時点で見えていたが後で取消・除外になった馬（934 頭 / 912R）と、出走したが v6 スコアが無い馬
   （競走中止＋master 欠落、1,631 頭 / 1,536R。列 n_dnf）は候補にも p_model の正規化にも入っていない。一方 q_market（単勝・
   馬連）は判断時点のプール全体で正規化しているので、該当レースでは候補馬の q の和が 1 より小さい。
8. **複勝の q** は 1/mid の正規化で粗い（Σ1/mid÷着数の p05 が 0.77-0.91。Lo/Hi の幅が広いレースで mid が高めに出る）。
   odds_decision_lo / _hi を併せて使うこと。
9. **λ** は 2026 の本番較正済み p_win で fit した値を OOF（softmax-τ）にも市場 π にも流用した（仕様 §2 の指定）。
   本スクリプトの PL は compute_bets.pl_pair_probs と 200R で最大差 5.6e−16（§「pl_vs_compute_bets_max_abs_diff」）。
10. **推定市場確率（ワイド・馬単・三連複・三連単）は価格ではない**。単勝・馬連の T−28 から作った合成値で、和は厳密に 1（ワイド 3）。
    「q との乖離」を妙味と読むのは、これらの券種では市場の歪みでなく合成式の歪みを拾う危険がある（仕様 §9 と同じ注意）。
11. **時点**: 単勝・複勝・馬連の判断断面は全レースで発走 25-30 分前（p50 27-28 分）、単勝と馬連は全レースで同一時刻。発走時刻
    以降・15 分前より後の断面は 0 件。枠連の判断断面（O1 S0）は発走 27-32 分前で、単勝断面より最大 6 分早い。odds_t11 は
    全レースで 11 分前。確定記録は発走 5-13 分後。
12. 単勝 odds_decision は base.parquet の mkt_odds_pre（analysis/mcond/market.py）と 109,848 行すべて一致（断面選択の再現確認）。"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checks-only", action="store_true", help="保存済み parquet から検査とレポートだけ作り直す")
    ap.add_argument("--smoke", type=int, default=0, help="各年の先頭 N レースだけで動作確認（出力は *_smoke）")
    args = ap.parse_args()
    if args.smoke:
        global OUT, RACES_OUT, CHECKS_OUT, REPORT
        OUT = OUT.with_name("substrate_smoke.parquet")
        RACES_OUT = RACES_OUT.with_name("substrate_races_smoke.parquet")
        CHECKS_OUT = CHECKS_OUT.with_name("substrate_checks_smoke.json")
        REPORT = REPORT.with_name("SUBSTRATE_REPORT_smoke.md")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKS_OUT.parent.mkdir(parents=True, exist_ok=True)
    if args.checks_only:
        sub = pd.read_parquet(OUT)
        races = pd.read_parquet(RACES_OUT)
        info = json.loads(CHECKS_OUT.read_text(encoding="utf-8"))["build_info"]
    else:
        sub, races, info = build(args.smoke)
        log(f"save {len(sub):,} rows -> {OUT}")
        sub.to_parquet(OUT, index=False)
        races.to_parquet(RACES_OUT, index=False)
    log("checks")
    c = checks(sub, races, info)
    inputs = {}
    for p in [BASE_PQ, KEKKA, WIDE, LAMBDA, CHAOS_Q]:
        inputs[str(p.relative_to(BASE)).replace("\\", "/")] = sha256(p)
    inputs[str(O1_RAW)] = sha256(O1_RAW)
    inputs["data/_research/bet_decision_ai/substrate.parquet"] = sha256(OUT)
    c["inputs_sha256"] = inputs
    CHECKS_OUT.write_text(json.dumps(c, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    write_report(c, list(sub.columns), inputs)
    log(f"done. report -> {REPORT}")


if __name__ == "__main__":
    main()
