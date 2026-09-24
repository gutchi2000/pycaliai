# -*- coding: utf-8 -*-
"""
graph_core.py — EXP17 Stage 0: 時点安全な共通対戦馬 2-hop 表現の純関数実装
==========================================================================
本ファイルは Stage 0 の合成テスト・被覆監査・等価性監査・検出力監査が共有する唯一の実装。
学習・評価・ROI は一切含まない。

契約 (spec v0.2 §3-4 と一致させる):
  - 馬 ID は `血統登録番号` の文字列 (10桁の数字)。馬名 join は禁止。不正 ID は fail-closed。
  - 履歴 = 対象日 t より厳密に前 (date < t) の JRA 平地レースの公式最終着順のみ。同日結果は使わない。
  - (h,c) の過去対戦: W_hc = h が c より先着した回数、L_hc = c が先着した回数。同着は両方に加えない。
  - p_hc = (W+alpha)/(W+L+2 alpha),  ell_hc = logit(clip(p_hc, eps, 1-eps))
  - 未対戦 pair (i,j) と共通対戦馬 c:  d_ij^(c) = ell_ic - ell_jc
  - d_ij = Σ_c ω_ijc d_ij^(c) / Σ_c ω_ijc ,  ω = 信頼度 × 新しさ × 条件一致 (Stage 0 暫定値、spec で凍結)
  - 証拠の無い pair は missing (NaN)。0 に置換しない。
  - 直接対戦済み pair は主機構 sample から除外 (別集計)。
  - Hodge 射影: min_s Σ_{(i,j)∈E} w_ij (s_i - s_j - d_ij)^2 + lambda Σ s_i^2、連結成分ごとに Σ s_i = 0。
  - uncovered 馬は s=0 かつ graph_uncovered=True。
  - p_graph = softmax(s / tau)。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HID_RE = re.compile(r"^\d{10}$")


class HorseIDError(ValueError):
    """不正 ID・ID 衝突は fail-closed。"""


# ---------------------------------------------------------------- ID 検証
def validate_hids(hids: Iterable[str], names: Optional[Iterable[str]] = None) -> None:
    """血統登録番号の形式検証と、同一 ID に複数馬名が結び付く衝突の検出 (fail-closed)。
    names が与えられた場合のみ衝突検査を行う。馬名から ID を補完することは決してしない。"""
    hids = list(hids)
    bad = [h for h in hids if not isinstance(h, str) or not HID_RE.match(h)]
    if bad:
        raise HorseIDError(f"invalid horse id(s): {bad[:5]} (n={len(bad)})")
    if names is not None:
        seen: Dict[str, str] = {}
        for h, n in zip(hids, names):
            if n is None or (isinstance(n, float) and np.isnan(n)):
                continue
            n = str(n)
            if h in seen and seen[h] != n:
                raise HorseIDError(f"horse id collision: {h} -> {{{seen[h]!r}, {n!r}}}")
            seen.setdefault(h, n)


# ---------------------------------------------------------------- 履歴ストア
@dataclass
class HistoryStore:
    """directed meetings を (h でソート, h 内で date ソート) した列指向配列で保持する。
    クエリは `date < day` で切るので day-start snapshot が構造的に保証される。"""
    h: np.ndarray          # int32 code of h
    c: np.ndarray          # int32 code of c
    date: np.ndarray       # int32 YYYYMMDD
    won: np.ndarray        # int8: 1 = h beat c, 0 = c beat h (ties are not stored)
    surf: np.ndarray       # int8: 1 = 芝, 0 = ダ
    dband: np.ndarray      # int8 distance band
    ridc: np.ndarray       # int32 code of the meeting race (rid16)
    code_of: Dict[str, int]
    hid_of: np.ndarray     # code -> hid
    start_of: np.ndarray   # code -> first row index in h-sorted arrays
    end_of: np.ndarray     # code -> end row index (exclusive)
    starts_date: Dict[int, np.ndarray] = field(default_factory=dict)  # code -> sorted dates of own starts (as-of career count 用)
    rid_of: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=object))  # rid code -> rid16

    def meetings_before(self, code_h: int, code_c: int, day: int):
        """(h,c) の date<day の対戦 (rid16, date, won) を返す。等価性監査の null 再現用。"""
        c, d, won, _, _ = self.slice_before(code_h, day)
        a, b = int(self.start_of[code_h]), int(self.end_of[code_h])
        k = len(c)
        rc = self.ridc[a:a + k]
        m = c == code_c
        return [str(x) for x in self.rid_of[rc[m]]], d[m], won[m]

    def slice_before(self, code: int, day: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        a, b = int(self.start_of[code]), int(self.end_of[code])
        if b <= a:
            e = np.empty(0, dtype=np.int32)
            return e, e, np.empty(0, np.int8), np.empty(0, np.int8), np.empty(0, np.int8)
        d = self.date[a:b]
        k = int(np.searchsorted(d, day, side="left"))   # date < day  (strict)
        return self.c[a:a + k], d[:k], self.won[a:a + k], self.surf[a:a + k], self.dband[a:a + k]

    def prior_starts(self, code: int, day: int) -> int:
        arr = self.starts_date.get(code)
        if arr is None:
            return 0
        return int(np.searchsorted(arr, day, side="left"))


def dist_band(d: float) -> int:
    if d != d:
        return 1
    return int(np.searchsorted(np.array([1400, 1800, 2200]), d, side="left"))


def build_history_store(runs: pd.DataFrame, subject_hids: Optional[Sequence[str]] = None) -> HistoryStore:
    """runs: columns [date(int YYYYMMDD), rid16(str), hid(str), fin(float, 公式着順, 同着は同値), surf(int), dband(int)].
    finisher 行のみを想定 (master_v2 は着順が数値の行のみ)。
    subject_hids を与えると h 側をその集合に限定 (メモリ節約; c 側は全馬)。"""
    validate_hids(runs["hid"].unique())
    runs = runs.sort_values(["date", "rid16", "hid"], kind="mergesort").reset_index(drop=True)
    all_h = pd.Index(sorted(runs["hid"].unique()))
    code_of = {h: i for i, h in enumerate(all_h)}
    hid_of = np.array(all_h, dtype=object)
    subj = None if subject_hids is None else np.array([code_of[h] for h in subject_hids if h in code_of], dtype=np.int32)
    subj_mask = None
    if subj is not None:
        subj_mask = np.zeros(len(all_h), dtype=bool)
        subj_mask[subj] = True

    codes = runs["hid"].map(code_of).to_numpy(np.int32)
    dates = runs["date"].to_numpy(np.int32)
    fins = runs["fin"].to_numpy(float)
    surfs = runs["surf"].to_numpy(np.int8)
    dbs = runs["dband"].to_numpy(np.int8)
    rid = runs["rid16"].to_numpy()
    rid_index = pd.Index(pd.unique(rid))
    rid_codes_all = rid_index.get_indexer(rid).astype(np.int32)

    # race boundaries
    change = np.r_[True, rid[1:] != rid[:-1]]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], len(rid)]

    H, C, D, W, S, B, RC = [], [], [], [], [], [], []
    for a, b in zip(starts, ends):
        n = b - a
        if n < 2:
            continue
        cc = codes[a:b]
        ff = fins[a:b]
        ii, jj = np.triu_indices(n, k=1)
        tie = ff[ii] == ff[jj]
        ii, jj = ii[~tie], jj[~tie]
        if len(ii) == 0:
            continue
        i_wins = (ff[ii] < ff[jj]).astype(np.int8)
        # directed both ways
        hh = np.concatenate([cc[ii], cc[jj]])
        oo = np.concatenate([cc[jj], cc[ii]])
        ww = np.concatenate([i_wins, 1 - i_wins])
        if subj_mask is not None:
            m = subj_mask[hh]
            hh, oo, ww = hh[m], oo[m], ww[m]
            if len(hh) == 0:
                continue
        H.append(hh); C.append(oo); W.append(ww)
        RC.append(np.full(len(hh), rid_codes_all[a], np.int32))
        D.append(np.full(len(hh), dates[a], np.int32))
        S.append(np.full(len(hh), surfs[a], np.int8))
        B.append(np.full(len(hh), dbs[a], np.int8))
    if H:
        h = np.concatenate(H); c = np.concatenate(C); d = np.concatenate(D)
        w = np.concatenate(W); s = np.concatenate(S); bnd = np.concatenate(B); rc = np.concatenate(RC)
    else:
        h = c = d = rc = np.empty(0, np.int32); w = s = bnd = np.empty(0, np.int8)
    order = np.lexsort((c, d, h))          # sort by h, then date, then c  (deterministic)
    h, c, d, w, s, bnd, rc = h[order], c[order], d[order], w[order], s[order], bnd[order], rc[order]
    start_of = np.searchsorted(h, np.arange(len(all_h)), side="left").astype(np.int64)
    end_of = np.searchsorted(h, np.arange(len(all_h)), side="right").astype(np.int64)

    starts_date: Dict[int, np.ndarray] = {}
    g = pd.DataFrame({"code": codes, "date": dates}).sort_values(["code", "date"], kind="mergesort")
    for code, sub in g.groupby("code", sort=False):
        starts_date[int(code)] = sub["date"].to_numpy(np.int32)
    return HistoryStore(h, c, d, w, s, bnd, rc, code_of, hid_of, start_of, end_of, starts_date,
                        np.array(rid_index, dtype=object))


# ---------------------------------------------------------------- ペア証拠
@dataclass(frozen=True)
class PairParams:
    alpha: float = 1.0          # symmetric Beta prior
    eps: float = 0.01           # logit clip
    half_life_days: Optional[float] = None   # None = no decay
    cond_bonus: float = 0.0     # weight multiplier (1 + cond_bonus * match_rate)
    lookback_days: Optional[int] = None      # None = all history
    lam: float = 1e-6           # Hodge ridge (uniqueness only)
    include_direct_in_projection: bool = False


def _agg_opponents(c: np.ndarray, d: np.ndarray, won: np.ndarray, surf: np.ndarray, db: np.ndarray,
                   day: int, tsurf: int, tdb: int, prm: PairParams):
    """対象馬 h の履歴 (date<day) を相手 c ごとに集約。戻り: dict c -> (W, L, last_date, wsum, cond_match_rate, n)"""
    if prm.lookback_days is not None and len(d):
        lo = _shift_day(day, -prm.lookback_days)
        m = d >= lo
        c, d, won, surf, db = c[m], d[m], won[m], surf[m], db[m]
    out = {}
    if len(c) == 0:
        return out
    order = np.argsort(c, kind="mergesort")
    c, d, won, surf, db = c[order], d[order], won[order], surf[order], db[order]
    bounds = np.r_[0, np.flatnonzero(c[1:] != c[:-1]) + 1, len(c)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        w = won[a:b].astype(float)
        if prm.half_life_days:
            age = np.array([_days_between(int(x), day) for x in d[a:b]], float)
            dec = np.exp(-np.log(2.0) * age / prm.half_life_days)
        else:
            dec = np.ones(b - a)
        W = float((w * dec).sum()); L = float(((1 - w) * dec).sum())
        match = float(((surf[a:b] == tsurf) & (db[a:b] == tdb)).mean())
        out[int(c[a])] = (W, L, int(d[a:b].max()), b - a, match)
    return out


_ORD: Dict[int, int] = {}


def _ordinal(day: int) -> int:
    o = _ORD.get(day)
    if o is None:
        o = pd.Timestamp(str(day)).toordinal()
        _ORD[day] = o
    return o


def _shift_day(day: int, delta: int) -> int:
    t = pd.Timestamp(str(day)) + pd.Timedelta(days=delta)
    return int(t.strftime("%Y%m%d"))


def _days_between(d0: int, d1: int) -> int:
    return _ordinal(d1) - _ordinal(d0)


def ell(W: float, L: float, prm: PairParams) -> float:
    p = (W + prm.alpha) / (W + L + 2 * prm.alpha)
    p = min(max(p, prm.eps), 1 - prm.eps)
    return float(np.log(p / (1 - p)))


@dataclass
class RacePairResult:
    hids: List[str]
    d: np.ndarray                 # (n,n) antisymmetric, NaN where no evidence
    w: np.ndarray                 # (n,n) weights (0 where no evidence)
    n_common: np.ndarray          # (n,n) int count of common opponents used
    direct: np.ndarray            # (n,n) bool: pair met directly before day
    d_direct: np.ndarray          # (n,n) direct-evidence log-odds (NaN if none) — 別集計用
    per_c: Dict[Tuple[int, int], List[Tuple[int, float, float]]]   # (i,j) -> [(c_code, d_ij^(c), omega)]
    edge_age_days: List[int]      # ages of (h,c) edges used (most recent meeting)
    cond_match: List[float]
    n_meet_hc: List[int]          # meetings per used (h,c) edge


def pair_evidence(store: HistoryStore, hids: Sequence[str], day: int, tsurf: int, tdb: int,
                  prm: PairParams = PairParams()) -> RacePairResult:
    """現在レースの出走馬 hids について、day-start snapshot で pair 証拠行列を作る。
    hids の順序に依存しない (順序を変えても同じ (i,j) 対応の値が返る)。"""
    validate_hids(hids)
    n = len(hids)
    codes = [store.code_of.get(h, -1) for h in hids]
    aggs = []
    for code in codes:
        if code < 0:
            aggs.append({})
            continue
        c, d, won, surf, db = store.slice_before(code, day)
        aggs.append(_agg_opponents(c, d, won, surf, db, day, tsurf, tdb, prm))
    D = np.full((n, n), np.nan); Wt = np.zeros((n, n)); NC = np.zeros((n, n), int)
    DIR = np.zeros((n, n), bool); DD = np.full((n, n), np.nan)
    per_c: Dict[Tuple[int, int], List[Tuple[int, float, float]]] = {}
    ages: List[int] = []; matches: List[float] = []; nmeet: List[int] = []
    for i in range(n):
        for j in range(i + 1, n):
            ai, aj = aggs[i], aggs[j]
            ci, cj = codes[i], codes[j]
            if cj >= 0 and cj in ai:
                DIR[i, j] = DIR[j, i] = True
                Wd, Ld, *_ = ai[cj]
                v = ell(Wd, Ld, prm); DD[i, j] = v; DD[j, i] = -v
            common = set(ai.keys()) & set(aj.keys())
            common.discard(ci); common.discard(cj)
            if not common:
                continue
            num = 0.0; den = 0.0; lst = []
            for c in sorted(common):
                Wi, Li, ldi, ni, mi = ai[c]
                Wj, Lj, ldj, nj, mj = aj[c]
                dc = ell(Wi, Li, prm) - ell(Wj, Lj, prm)
                rel = (ni * nj) / (ni + nj)              # harmonic-mean reliability
                omega = rel * (1.0 + prm.cond_bonus * 0.5 * (mi + mj))
                num += omega * dc; den += omega
                lst.append((c, dc, omega))
                ages.append(_days_between(ldi, day)); ages.append(_days_between(ldj, day))
                matches.append(mi); matches.append(mj); nmeet.append(ni); nmeet.append(nj)
            if den > 0:
                v = num / den
                D[i, j] = v; D[j, i] = -v
                Wt[i, j] = Wt[j, i] = den
                NC[i, j] = NC[j, i] = len(lst)
                per_c[(i, j)] = lst
    return RacePairResult(list(hids), D, Wt, NC, DIR, DD, per_c, ages, matches, nmeet)


# ---------------------------------------------------------------- Hodge 射影
def components(adj: np.ndarray) -> List[List[int]]:
    n = adj.shape[0]
    seen = np.zeros(n, bool); comps = []
    for s in range(n):
        if seen[s]:
            continue
        stack = [s]; seen[s] = True; comp = []
        while stack:
            u = stack.pop(); comp.append(u)
            for v in np.flatnonzero(adj[u]):
                if not seen[v]:
                    seen[v] = True; stack.append(int(v))
        comps.append(sorted(comp))
    return comps


def hodge_project(d: np.ndarray, w: np.ndarray, lam: float = 1e-6):
    """weighted least squares:  min Σ w_ij (s_i - s_j - d_ij)^2 + lam Σ s_i^2, per component Σ s = 0.
    戻り: s (n,), uncovered (n,) bool, comps (list), residual matrix r_ij = d_ij - (s_i - s_j) (NaN where no edge)."""
    n = d.shape[0]
    obs = ~np.isnan(d) & (w > 0)
    np.fill_diagonal(obs, False)
    if not np.allclose(np.nan_to_num(d) + np.nan_to_num(d).T, 0, atol=1e-12):
        raise ValueError("d must be antisymmetric")
    Wm = np.where(obs, w, 0.0)
    deg = Wm.sum(1)
    Lap = np.diag(deg) - Wm
    b = (Wm * np.nan_to_num(d)).sum(1)
    s = np.zeros(n)
    uncovered = deg <= 0
    comps = components(obs)
    for comp in comps:
        idx = np.array(comp)
        if len(idx) == 1:
            continue
        A = Lap[np.ix_(idx, idx)] + lam * np.eye(len(idx))
        sol = np.linalg.solve(A, b[idx])
        sol = sol - sol.mean()
        s[idx] = sol
    r = np.where(obs, d - (s[:, None] - s[None, :]), np.nan)
    return s, uncovered, comps, r


def softmax_scores(s: np.ndarray, tau: float) -> np.ndarray:
    z = s / tau
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()
