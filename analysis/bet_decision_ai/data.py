# -*- coding: utf-8 -*-
"""
data.py — 馬券判断 AI の読み込みと特徴量化（仕様 docs/research/BET_DECISION_AI_SPEC.md v1.0）
=========================================================================================
**開発コードは 2023-2025 年の行を一切読まない。** 年の絞り込みはこのファイルの `_read_years()` 1 か所で行い、
parquet の読み込み時点で year<=2022 の行だけを取り出す（pyarrow filter）。その後 `dev_only()` で再検査する。
確認期間（2023-25）を読めるのは confirm.py が `CONFIRM_TOKEN` を渡したときだけ。

- arm=main のみ（枠連 sub_wakuren は判断 AI の入力にしない。B1 a95 の決済用にだけ別関数で読む）
- 決済不能（result_mismatch の 16R = settle_status 'unsettleable'）はレースごと除外（全方策共通）
- 同着で払戻額を特定できない的中（settle_status 'win_amount_missing'）は払戻 100 円（仕様 §7）
- 入力列は仕様 §4 の許可リストだけ（ALLOWED_INPUT_COLS）。派生量はこれらの関数のみ。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
SUB_PQ = BASE / "data" / "_research" / "bet_decision_ai" / "substrate.parquet"
RACES_PQ = BASE / "data" / "_research" / "bet_decision_ai" / "substrate_races.parquet"
BASE_PQ = BASE / "data" / "_research" / "mcond" / "base.parquet"
ODIR = BASE / "data" / "Time _series_odds"
SPEC = BASE / "docs" / "research" / "BET_DECISION_AI_SPEC.md"
SPEC_SHA_PREFIX = "5cc64e6adfbd8769"

DEV_MAX_YEAR = 2022
TRAIN_YEARS = (2017, 2020)
SELECT_YEARS = (2021, 2022)
CONFIRM_YEARS = (2023, 2025)
CONFIRM_TOKEN = "__bet_decision_ai_confirm_once__"   # confirm.py だけが使う

KAPPA = 0.05
BET_TYPES = ["tan", "fuku", "umaren", "wide", "umatan", "sanrenpuku", "sanrentan"]
K = {"tan": 3, "fuku": 3, "umaren": 6, "wide": 6, "umatan": 6, "sanrenpuku": 10, "sanrentan": 10}
SLOT0 = {}
_o = 0
for _b in BET_TYPES:
    SLOT0[_b] = _o
    _o += K[_b]
NSLOT = _o   # 44

# 仕様 §4 の許可リスト（これ以外を入力にしない）
ALLOWED_INPUT_COLS = ["bet_type", "cand_rank", "p_model", "q_market", "q_src", "log_p_over_q", "odds_decision",
                      "odds_decision_lo", "odds_decision_hi", "n_field_decision", "surface", "distance", "class_norm",
                      "venue_code", "field_chaos_score", "chaos_pct_asof", "top1_p", "top2_p", "overround_tan",
                      "overround_fuku", "overround_umaren"]
FORBIDDEN_INPUT_COLS = ["payout", "payout_src", "settle_status", "close_odds", "close_odds_lo", "close_odds_hi",
                        "odds_t11", "odds_t11_lo", "odds_t11_hi", "deadheat_top3", "n_starters_close", "n_dnf",
                        "n_scratch_after_snap", "year", "date"]
# 決済・集計・baseline 用のキー列（入力には使わない）
KEY_COLS = ["rid16", "year", "date", "arm", "combo", "h1", "h2", "h3", "payout", "settle_status", "n_entries_master"]

SURFACES = ["芝", "ダ"]
CLASSES = ["新馬", "未勝利", "1勝", "2勝", "3勝", "OP", "OP(L)", "G3", "G2", "G1", "重賞(格付無)"]
VENUES = [f"{i:02d}" for i in range(1, 11)]
Q_SRCS = ["real_odds", "est_tan_umaren", "est_tan_only"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 22), b""):
            h.update(ch)
    return h.hexdigest()


def check_spec() -> str:
    s = sha256(SPEC)
    if not s.startswith(SPEC_SHA_PREFIX):
        raise SystemExit(f"仕様の sha256 が凍結版と違う: {s[:16]} != {SPEC_SHA_PREFIX}")
    return s


# ======================================================================== 年の絞り込み（唯一の入口）
def dev_only(df: pd.DataFrame, col: str = "year") -> pd.DataFrame:
    """開発期間（year<=2022）の行だけを返し、確認期間が混ざっていないことを検査する。"""
    y = pd.to_numeric(df[col], errors="raise")
    out = df[y <= DEV_MAX_YEAR]
    assert (pd.to_numeric(out[col]) <= DEV_MAX_YEAR).all()
    return out


def _read_years(path: Path, columns: list[str] | None, year_hi: int, year_lo: int = 2015,
                extra_filters: list | None = None, _confirm_token: str | None = None) -> pd.DataFrame:
    """全 parquet 読み込みの唯一の入口。year_hi > 2022 は confirm.py のトークンが無いと拒否。"""
    if year_hi > DEV_MAX_YEAR and _confirm_token != CONFIRM_TOKEN:
        raise PermissionError("確認期間（2023-2025）は confirm.py --confirm-once からしか読めない")
    filt = [("year", ">=", year_lo), ("year", "<=", year_hi)] + (extra_filters or [])
    df = pq.read_table(path, columns=columns, filters=filt).to_pandas()
    y = pd.to_numeric(df["year"])
    assert ((y >= year_lo) & (y <= year_hi)).all()
    if _confirm_token != CONFIRM_TOKEN:
        df = dev_only(df)
    return df


def load_candidates(year_lo: int = 2017, year_hi: int = DEV_MAX_YEAR, _confirm_token: str | None = None) -> pd.DataFrame:
    """arm=main の候補券（判断 AI の入力 + 決済列）。決済不能レースを除外、同着 NaN を 100 円に。"""
    cols = list(dict.fromkeys(KEY_COLS + ALLOWED_INPUT_COLS))
    df = _read_years(SUB_PQ, cols, year_hi, year_lo, [("arm", "=", "main")], _confirm_token)
    for c in ["arm", "bet_type", "q_src", "settle_status"]:
        df[c] = df[c].astype(str)
    assert set(df["arm"]) == {"main"}
    bad = set(df.loc[df["settle_status"] == "unsettleable", "rid16"])
    df = df[~df["rid16"].isin(bad)].copy()
    wam = df["settle_status"] == "win_amount_missing"
    df.loc[wam, "payout"] = 100.0
    nps = df["settle_status"] == "no_payout_source"
    if nps.any():
        raise RuntimeError(f"払戻源の無い行 {int(nps.sum())}（2015 ワイド以外で発生してはいけない）")
    assert df["payout"].notna().all(), "payout NaN が残っている"
    df.attrs["n_unsettleable_races"] = len(bad)
    df.attrs["n_deadheat_100"] = int(wam.sum())
    return df


def load_wakuren_for_b1(year_lo: int = 2017, year_hi: int = DEV_MAX_YEAR, _confirm_token: str | None = None) -> pd.DataFrame:
    """B1 a95 の枠連券の決済専用（判断 AI には使わない）。"""
    cols = ["rid16", "year", "date", "arm", "bet_type", "combo", "h1", "h2", "payout", "settle_status"]
    df = _read_years(SUB_PQ, cols, year_hi, year_lo, [("arm", "=", "sub_wakuren")], _confirm_token)
    df["settle_status"] = df["settle_status"].astype(str)
    df.loc[df["settle_status"] == "win_amount_missing", "payout"] = 100.0
    return df


def load_races(year_lo: int = 2017, year_hi: int = DEV_MAX_YEAR, _confirm_token: str | None = None) -> pd.DataFrame:
    return _read_years(RACES_PQ, None, year_hi, year_lo, None, _confirm_token)


def load_base_market(year_lo: int = 2015, year_hi: int = DEV_MAX_YEAR, _confirm_token: str | None = None) -> pd.DataFrame:
    """base.parquet の市場列（B2 と B4 の市場版 top1/top2/混戦度用）。"""
    cols = ["rid16", "ban", "year", "date", "v6_pwin", "mkt_pi_pre", "mkt_odds_pre", "fin", "fpay"]
    return _read_years(BASE_PQ, cols, year_hi, year_lo, None, _confirm_token)


def load_waku_map(year_lo: int = 2017, year_hi: int = DEV_MAX_YEAR, _confirm_token: str | None = None) -> dict:
    """master_v2 の実際の枠番（B1 a95 の枠連券の組を作る用）。読み込み直後に年で絞る。{rid16: {馬番: 枠番}}"""
    if year_hi > DEV_MAX_YEAR and _confirm_token != CONFIRM_TOKEN:
        raise PermissionError("確認期間は confirm.py からのみ")
    master = BASE / "data" / "master_v2_20130105-20251228.csv"
    m = pd.read_csv(master, encoding="utf-8-sig", dtype=str, usecols=["レースID(新/馬番無)", "枠番", "馬番"])
    m["rid16"] = m["レースID(新/馬番無)"].str.replace(r"\D", "", regex=True).str[:16]
    m = m[m["rid16"].str.len() == 16]
    m["year"] = m["rid16"].str[:4].astype(int)
    m = m[(m["year"] >= year_lo) & (m["year"] <= year_hi)]
    if _confirm_token != CONFIRM_TOKEN:
        m = dev_only(m)
    m["ban"] = pd.to_numeric(m["馬番"], errors="coerce")
    m["waku"] = pd.to_numeric(m["枠番"], errors="coerce")
    m = m.dropna(subset=["ban", "waku"])
    return {rid: dict(zip(g["ban"].astype(int), g["waku"].astype(int))) for rid, g in m.groupby("rid16", sort=False)}


def load_tan_snapshot_full_field(year_lo: int = 2017, year_hi: int = DEV_MAX_YEAR,
                                 _confirm_token: str | None = None) -> dict:
    """TANPUK 時系列から T−28 断面（substrate と同じ選び方）と確定断面の単勝オッズを全馬分。
    rid16 の年で読み込み直後に絞る（ここも唯一の入口の規則に従う）。返り値 {rid16: (pre_tan[18], fin_tan[18])}"""
    if year_hi > DEV_MAX_YEAR and _confirm_token != CONFIRM_TOKEN:
        raise PermissionError("確認期間は confirm.py からのみ")
    from analysis.bet_decision_ai import build_substrate as bs
    parts = []
    for f in sorted(ODIR.glob("TANPUK_*.csv")):
        df = pd.read_csv(f, encoding="cp932", low_memory=False, skipinitialspace=True)
        c = list(df.columns)
        df = df.rename(columns={c[0]: "rid", c[1]: "kb", c[2]: "mdhm", c[3]: "tou", c[4]: "votes"})
        df["rid16"] = df["rid"].astype(str).str.replace(r"\D", "", regex=True).str[:16]
        df = df[df["rid16"].str.len() == 16]
        df["year"] = df["rid16"].str[:4].astype(int)
        df = df[(df["year"] >= year_lo) & (df["year"] <= year_hi)]
        if _confirm_token != CONFIRM_TOKEN:
            df = dev_only(df)
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df["kb"] = pd.to_numeric(df["kb"], errors="coerce")
    df["mdhm"] = pd.to_numeric(df["mdhm"], errors="coerce")
    df = df.dropna(subset=["kb", "mdhm"]).copy()
    df["mdhm"] = df["mdhm"].astype(np.int64)
    df["ts"] = pd.to_datetime(df["rid16"].str[:4] + df["mdhm"].astype(str).str.zfill(8), format="%Y%m%d%H%M",
                              errors="coerce")
    pre, fin = bs.select_snapshots(df)
    tan_c = {}
    for x in df.columns:
        s = str(x).strip()
        if s.endswith("単") and s[:-1].isdigit():
            tan_c[int(s[:-1])] = x

    def mat(d):
        a = np.zeros((len(d), 18))
        for b, c in tan_c.items():
            a[:, b - 1] = pd.to_numeric(d[c], errors="coerce").fillna(0.0).to_numpy()
        return dict(zip(d.index, a))
    mp, mf = mat(pre), mat(fin)
    return {r: (mp[r], mf.get(r)) for r in mp}


# ======================================================================== 特徴量
def market_race_context(base_mkt: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """B4 用: 市場版の top1_p / top2_p / 混戦度（mkt_pi_pre をレース内正規化した分布のエントロピー）と
    その as-of パーセンタイル（前年までの分布での位置、substrate と同じ作り方）。"""
    b = base_mkt.dropna(subset=["mkt_pi_pre"])
    rows = []
    for rid, g in b.groupby("rid16", sort=False):
        pi = g["mkt_pi_pre"].to_numpy(float)
        pi = pi[pi > 0]
        if len(pi) < 2:
            continue
        pi = pi / pi.sum()
        ps = np.sort(pi)[::-1]
        ent = -np.sum(pi * np.log(pi))
        rows.append((rid, int(g["year"].iloc[0]), ps[0], ps[1], min(1.0, max(0.0, ent / np.log(len(pi))))))
    m = pd.DataFrame(rows, columns=["rid16", "year", "mkt_top1_p", "mkt_top2_p", "mkt_chaos"])
    m = m[m["rid16"].isin(set(races["rid16"]))]   # races は 2015 から渡す（as-of の参照分布 = substrate の平地レース）
    m["mkt_chaos_pct_asof"] = np.nan
    for y in sorted(m["year"].unique()):
        ref = np.sort(m.loc[m["year"] < y, "mkt_chaos"].to_numpy())
        if len(ref) == 0:
            continue
        mm = m["year"] == y
        m.loc[mm, "mkt_chaos_pct_asof"] = np.searchsorted(ref, m.loc[mm, "mkt_chaos"], side="right") / len(ref)
    return m.drop(columns=["year"])


def _onehot(s: pd.Series, cats: list[str], prefix: str) -> pd.DataFrame:
    s = s.astype(str)
    d = {f"{prefix}_{c}": (s == c).astype(np.float32) for c in cats}
    d[f"{prefix}_other"] = (~s.isin(cats)).astype(np.float32)
    return pd.DataFrame(d, index=s.index)


def ticket_features(df: pd.DataFrame, mode: str = "model", mkt_ctx: pd.DataFrame | None = None) -> pd.DataFrame:
    """候補券の数値特徴（許可列とその関数だけ）。mode='market' は B4: p_model→q_market、log_p_over_q=0、
    top1_p/top2_p/混戦度は市場版、cand_rank は候補内の q 順位。"""
    assert mode in ("model", "market")
    d = df.copy()
    if mode == "market":
        assert mkt_ctx is not None
        d = d.merge(mkt_ctx, on="rid16", how="left")
        d["p_model"] = d["q_market"]
        d["log_p_over_q"] = 0.0
        d["top1_p"] = d["mkt_top1_p"]
        d["top2_p"] = d["mkt_top2_p"]
        d["field_chaos_score"] = d["mkt_chaos"]
        d["chaos_pct_asof"] = d["mkt_chaos_pct_asof"]
        d["cand_rank"] = d.groupby(["rid16", "bet_type"])["q_market"].rank(ascending=False, method="first")
        d.index = df.index
    used = [c for c in d.columns if c in ALLOWED_INPUT_COLS]
    for c in FORBIDDEN_INPUT_COLS:
        assert c not in used
    eps = 1e-6
    f = pd.DataFrame(index=d.index)
    f["cand_rank"] = d["cand_rank"].astype(np.float32)
    f["log_p"] = np.log(d["p_model"].clip(lower=eps))
    f["log_q"] = np.log(d["q_market"].clip(lower=eps))
    f["lpq"] = d["log_p_over_q"].fillna(0.0)
    f["log_odds"] = np.log(d["odds_decision"].clip(lower=1.0))
    f["p_x_odds"] = np.log((d["p_model"] * d["odds_decision"]).clip(lower=eps))
    lo, hi = d["odds_decision_lo"], d["odds_decision_hi"]
    f["has_lohi"] = lo.notna().astype(np.float32)
    f["log_lo"] = np.log(lo.clip(lower=1.0)).fillna(0.0)
    f["log_hi"] = np.log(hi.clip(lower=1.0)).fillna(0.0)
    f["p_x_lo"] = np.log((d["p_model"] * lo).clip(lower=eps)).fillna(0.0)
    f["n_field"] = d["n_field_decision"].astype(float).fillna(12.0)
    f["distance"] = d["distance"].astype(float) / 1000.0
    f["chaos"] = d["field_chaos_score"].astype(float)
    f["chaos_pct"] = d["chaos_pct_asof"].astype(float).fillna(0.5)
    f["top1_p"] = d["top1_p"].astype(float)
    f["top2_p"] = d["top2_p"].astype(float)
    f["or_tan"] = d["overround_tan"].astype(float).fillna(1.25)
    f["or_fuku"] = d["overround_fuku"].astype(float).fillna(1.22)
    f["or_umaren"] = d["overround_umaren"].astype(float).fillna(1.29)
    oh = [_onehot(d["bet_type"], BET_TYPES, "bt"), _onehot(d["q_src"], Q_SRCS, "qs"),
          _onehot(d["surface"], SURFACES, "sf"), _onehot(d["class_norm"], CLASSES, "cl"),
          _onehot(d["venue_code"].astype(str).str.zfill(2), VENUES, "vn")]
    f = pd.concat([f] + oh, axis=1).astype(np.float32)
    assert np.isfinite(f.to_numpy()).all(), "特徴に非有限値"
    return f


RACE_FEATS = ["n_field", "distance", "chaos", "chaos_pct", "top1_p", "top2_p", "or_tan", "or_fuku", "or_umaren"]


class Standardizer:
    def __init__(self):
        self.mu = None
        self.sd = None
        self.cols = None

    def fit(self, f: pd.DataFrame):
        self.cols = list(f.columns)
        x = f.to_numpy(np.float64)
        self.mu = x.mean(0)
        self.sd = x.std(0)
        oh = [i for i, c in enumerate(self.cols) if c.split("_")[0] in ("bt", "qs", "sf", "cl", "vn") or c == "has_lohi"]
        self.mu[oh] = 0.0
        self.sd[oh] = 1.0
        self.sd[self.sd < 1e-8] = 1.0
        return self

    def transform(self, f: pd.DataFrame) -> np.ndarray:
        assert list(f.columns) == self.cols
        return ((f.to_numpy(np.float64) - self.mu) / self.sd).astype(np.float32)

    def to_dict(self):
        return {"cols": self.cols, "mu": self.mu.tolist(), "sd": self.sd.tolist()}

    @classmethod
    def from_dict(cls, d):
        s = cls()
        s.cols, s.mu, s.sd = d["cols"], np.array(d["mu"]), np.array(d["sd"])
        return s


def slot_index(df: pd.DataFrame) -> np.ndarray:
    return (df["bet_type"].map(SLOT0).to_numpy(int) + df["cand_rank"].to_numpy(int) - 1)


class RaceTensor:
    """レース × 44 スロットの padded 配列。X[R,S,F], mask[R,S], R_ret[R,S]（払戻/100）, races, dates。
    スロット順は (bet_type, cand_rank) の固定配置で、候補の並びは p_model 順（substrate のまま）。"""

    def __init__(self, df: pd.DataFrame, Xrow: np.ndarray):
        rids = df["rid16"].to_numpy()
        self.races, inv = np.unique(rids, return_inverse=True)
        R, S, F = len(self.races), NSLOT, Xrow.shape[1]
        slot = slot_index(df)
        assert slot.min() >= 0 and slot.max() < NSLOT
        self.X = np.zeros((R, S, F), np.float32)
        self.mask = np.zeros((R, S), bool)
        self.ret = np.zeros((R, S), np.float32)
        self.p = np.zeros((R, S), np.float32)
        self.lpq = np.zeros((R, S), np.float32)
        self.X[inv, slot] = Xrow
        self.mask[inv, slot] = True
        self.ret[inv, slot] = df["payout"].to_numpy(np.float32) / 100.0
        self.p[inv, slot] = df["p_model"].to_numpy(np.float32)
        self.lpq[inv, slot] = df["log_p_over_q"].to_numpy(np.float32)
        dmap = df.groupby("rid16")["date"].first()
        self.dates = dmap.loc[self.races].astype(str).to_numpy()
        ymap = df.groupby("rid16")["year"].first()
        self.years = ymap.loc[self.races].to_numpy(int)
        # 各スロットの (bet_type, combo) を復元するための表
        self.combo = np.full((R, S), "", dtype=object)
        self.combo[inv, slot] = df["combo"].to_numpy()

    def subset(self, m: np.ndarray) -> "RaceTensor":
        o = object.__new__(RaceTensor)
        for k in ["races", "X", "mask", "ret", "p", "lpq", "dates", "years", "combo"]:
            setattr(o, k, getattr(self, k)[m])
        return o


SLOT_BET_TYPE = np.array(sum([[b] * K[b] for b in BET_TYPES], []))
