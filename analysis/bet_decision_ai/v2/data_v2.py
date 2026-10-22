# -*- coding: utf-8 -*-
"""
data_v2.py — 馬券判断 AI v2 のデータ層（仕様 docs/research/BET_DECISION_AI_V2_SPEC.md v1.0 §2-§4）
=====================================================================================================
- 読み込みの **唯一の入口** は `load_rows()` と `load_calib_base()`。年の絞り込みと結果列（res_* / base の win）の
  ロックはここ（`_guard`）だけで行う。2024 年以降の結果列を読もうとしたら ResultLockError で止める。
  解除できるのは確認用スクリプト（confirm_v2.py を `--confirm-once` で実行したとき）の `confirm_unlock()` だけ。
- §3 スコア欠損の市場確率置き換え（欠損を示す列は入力に使わない）
- §4-2 確率の尺度合わせ: 年 Y の p_win を、Y より前の OOF 3 年分（base.parquet、2015 年以降）で fit した isotonic で写像。
  OOF は 2015-2023 しか無いので、2024 以降の年はすべて 2021-23 の写像（仕様 §4 の注意書き）。
  2・3 着内確率は、そろえた p_win から λ 補正 PL（v1 build_substrate.pl_tensors、λ は substrate と同じ）で作り直す。
- §4 入力の許可リスト（overround_* は使わない）。
v1 のコードは変更しない（pl_tensors を import するだけ）。
"""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from analysis.bet_decision_ai.build_substrate import pl_tensors  # noqa: E402  v1 を流用（変更しない）

SPEC = BASE / "docs" / "research" / "BET_DECISION_AI_V2_SPEC.md"
SPEC_SHA256_PREFIX = "7e073a99140d5f0c"           # v1.1（2026-10-23 改版、§10）
SPEC_V10_SHA256_PREFIX = "d41323d6db13e48f"       # v1.0（学習器確認 v1.0 を実施した版、記録用）
SUB = BASE / "data" / "_research" / "bet_decision_ai" / "v2" / "substrate_v2.parquet"
BASE_PQ = BASE / "data" / "_research" / "mcond" / "base.parquet"
LAMBDA = BASE / "data" / "harville_lambda.json"

RESULT_LOCK_YEAR = 2024                 # この年以降の結果列はロック
TRAIN_YEARS = (2021, 2022)
SELECT_YEAR = 2023
CONFIRM_YEARS = (2024, 2025)
HOLDOUT_YEAR = 2026
CALIB_WINDOW = 3
CALIB_MIN_YEAR = 2015
ISO_FLOOR = 1e-4                        # isotonic の下限（0 確率で log が発散しないように）

# ---------------------------------------------------------------- §4 入力の許可リスト
PRICE_COLS = ["odds_tan", "odds_fuku_lo", "odds_fuku_hi", "odds_fuku_mid", "q_tan_mkt", "q_fuku_mkt", "tan_odds_rank",
              "mkt_entropy_tan", "mkt_q_top1", "mkt_q_gap12", "votes_tan_s1", "votes_fuku_s1", "fuku_places"]
MODEL_COLS = ["p_bet", "log_p_over_q", "model_rank", "model_entropy", "model_p_top1", "model_p_gap12", "p_win", "p_top3"]
CONTEXT_COLS = ["bet_type", "n_field_s1", "surface", "distance", "class_norm", "venue_code"]
CAT_COLS = ["bet_type", "surface", "class_norm", "venue_code"]
FORBIDDEN_INPUT_PREFIX = ("res_", "overround_", "score_day_")
FORBIDDEN_INPUTS = {"score_missing", "score_src", "score_dup_n", "n_scored", "period_role", "year", "date", "score"}

# 固定のカテゴリ表（年をまたいでコードが変わらないように）
CATS = {
    "bet_type": ["tan", "fuku"],
    "surface": ["芝", "ダ"],
    "class_norm": ["新馬", "未勝利", "1勝", "2勝", "3勝", "OP", "OP(L)", "G3", "G2", "G1", "重賞(格付無)"],
    "venue_code": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"],
}

# substrate から読む非結果列（入力の素・キー）
_BASE_COLS = ["rid16", "year", "date", "umaban", "bet_type", "s1_timing_ok", "venue_code", "surface", "distance",
              "class_norm", "n_field_s1", "odds_tan", "odds_fuku_lo", "odds_fuku_hi", "odds_fuku_mid", "fuku_places",
              "votes_tan_s1", "votes_fuku_s1", "q_tan_mkt", "q_fuku_mkt", "tan_odds_rank", "mkt_entropy_tan",
              "mkt_q_top1", "mkt_q_gap12", "score", "p_win_sm", "score_day_gate_fail"]
_RESULT_COLS = ["res_status", "res_payout_100"]


class ResultLockError(RuntimeError):
    pass


class _Unlock:
    """confirm_unlock() だけが作る解除札。"""
    __slots__ = ("purpose",)

    def __init__(self, purpose: str):
        self.purpose = purpose


_UNLOCK_SECRET = object()


def confirm_unlock() -> _Unlock:
    """確認用スクリプト（analysis.bet_decision_ai.v2.confirm_v2）を `--confirm-once` で実行したときだけ解除札を返す。"""
    caller = inspect.stack()[1]
    mod = inspect.getmodule(caller.frame)
    name = getattr(mod, "__name__", "")
    fname = Path(caller.filename).name
    if fname != "confirm_v2.py" or not (name.endswith("confirm_v2") or name == "__main__"):
        raise ResultLockError(f"confirm_unlock は confirm_v2.py からしか呼べない（呼び出し元 {caller.filename}）")
    if "--confirm-once" not in sys.argv:
        raise ResultLockError("confirm_unlock には --confirm-once の実行が必要")
    return _Unlock("confirm_v2 --confirm-once")


def _guard(years: tuple[int, int], wants_results: bool, unlock: _Unlock | None) -> None:
    """年の絞り込みと結果列ロックの唯一の判定。"""
    lo, hi = int(years[0]), int(years[1])
    assert lo <= hi
    if wants_results and hi >= RESULT_LOCK_YEAR and not isinstance(unlock, _Unlock):
        raise ResultLockError(f"{RESULT_LOCK_YEAR} 年以降の結果列は読めない（要求 {lo}-{hi}、結果列あり）")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 22), b""):
            h.update(ch)
    return h.hexdigest()


def check_spec() -> str:
    s = sha256(SPEC)
    if not s.startswith(SPEC_SHA256_PREFIX):
        raise RuntimeError(f"仕様の sha256 が凍結値と違う: {s}")
    return s


# ---------------------------------------------------------------- 読み込み（唯一の入口）
def load_rows(years: tuple[int, int], with_results: bool, unlock: _Unlock | None = None) -> pd.DataFrame:
    """substrate v2 の候補行。with_results=False なら res_* 列を一切読まない。s1_timing_ok=False は除外。"""
    _guard(years, with_results, unlock)
    cols = _BASE_COLS + (_RESULT_COLS if with_results else [])
    tbl = pq.read_table(SUB, columns=cols, filters=[("year", ">=", int(years[0])), ("year", "<=", int(years[1]))])
    df = tbl.to_pandas()
    for c in ("bet_type", "venue_code", "surface", "class_norm"):
        df[c] = df[c].astype(str)
    df["year"] = df["year"].astype(int)
    assert df["year"].between(int(years[0]), int(years[1])).all()
    if with_results:
        _guard((int(df["year"].min()), int(df["year"].max())), True, unlock)
    else:
        assert not any(c.startswith("res_") for c in df.columns)
    df = df[df["s1_timing_ok"].astype(bool)].drop(columns=["s1_timing_ok"])
    df = df.sort_values(["rid16", "umaban", "bet_type"]).reset_index(drop=True)
    return df


def load_calib_base(years: tuple[int, int]) -> pd.DataFrame:
    """base.parquet の OOF p_win と勝ち（結果）。結果列なので 2024 以降はロック（解除札を受け付けない）。"""
    _guard(years, True, None)
    tbl = pq.read_table(BASE_PQ, columns=["year", "v6_src", "v6_pwin", "win"],
                        filters=[("year", ">=", int(years[0])), ("year", "<=", int(years[1]))])
    b = tbl.to_pandas()
    assert b["year"].max() < RESULT_LOCK_YEAR
    b = b[b["v6_src"] == "oof"]
    return b


# ---------------------------------------------------------------- §4-2 較正
def calib_years(Y: int) -> tuple[int, int]:
    """年 Y の較正窓: Y より前で OOF がある年（2015-2023）の直近 3 年。"""
    last = min(Y - 1, RESULT_LOCK_YEAR - 1)
    first = max(CALIB_MIN_YEAR, last - CALIB_WINDOW + 1)
    return first, last


_ISO_CACHE: dict = {}


def calibrator(Y: int) -> IsotonicRegression:
    w = calib_years(Y)
    if w not in _ISO_CACHE:
        b = load_calib_base(w)
        assert sorted(b["year"].unique().tolist()) == list(range(w[0], w[1] + 1)), (w, b["year"].unique())
        iso = IsotonicRegression(y_min=ISO_FLOOR, y_max=1.0, increasing=True, out_of_bounds="clip")
        iso.fit(b["v6_pwin"].to_numpy(float), b["win"].to_numpy(float))
        _ISO_CACHE[w] = iso
    return _ISO_CACHE[w]


def calibrator_info() -> dict:
    return {f"{k[0]}-{k[1]}": {"n_thresholds": int(len(v.X_thresholds_))} for k, v in _ISO_CACHE.items()}


# ---------------------------------------------------------------- 特徴量化
def _norm_entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    if len(p) <= 1:
        return 0.0
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def horse_probs(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """馬単位（単勝行）で §3 置き換え + §4-2 較正 + λ 補正 PL。返り値は (rid16, umaban) ごとのモデル列。"""
    lam = json.loads(LAMBDA.read_text(encoding="utf-8"))
    l1, l2 = float(lam["lambda1"]), float(lam["lambda2"])
    h = df[df["bet_type"] == "tan"][["rid16", "year", "umaban", "score", "p_win_sm", "q_tan_mkt", "fuku_places"]].copy()
    h = h.sort_values(["rid16", "umaban"]).reset_index(drop=True)
    h["p_cal_raw"] = np.nan
    for Y, g in h.groupby("year"):
        iso = calibrator(int(Y))
        m = g["p_win_sm"].notna()
        h.loc[g.index[m], "p_cal_raw"] = iso.predict(g.loc[m, "p_win_sm"].to_numpy(float))
    rid = h["rid16"].to_numpy()
    starts = np.flatnonzero(np.r_[True, rid[1:] != rid[:-1]])
    ends = np.r_[starts[1:], len(h)]
    pc = h["p_cal_raw"].to_numpy(float)
    q = h["q_tan_mkt"].to_numpy(float)
    fp = h["fuku_places"].to_numpy(float)
    n = len(h)
    p_win = np.full(n, np.nan)
    p_top2 = np.full(n, np.nan)
    p_top3 = np.full(n, np.nan)
    mrank = np.full(n, np.nan)
    ment = np.full(n, np.nan)
    mtop1 = np.full(n, np.nan)
    mgap = np.full(n, np.nan)
    stats = {"races": 0, "races_with_missing": 0, "horses_missing": 0, "races_all_missing": 0}
    umab = h["umaban"].to_numpy(int)
    for s, e in zip(starts, ends):
        sl = slice(s, e)
        miss = ~np.isfinite(pc[sl])
        pr = pc[sl].copy()
        qq = q[sl]
        stats["races"] += 1
        if miss.all():
            stats["races_all_missing"] += 1
            p = qq / qq.sum()
        else:
            ps = pr[~miss]
            ps = ps / ps.sum()
            qm = qq[miss].sum()
            p = np.empty(e - s)
            p[~miss] = ps * (1.0 - qm)
            p[miss] = qq[miss]
            if miss.any():
                stats["races_with_missing"] += 1
                stats["horses_missing"] += int(miss.sum())
        p_win[sl] = p
        k = e - s
        if k >= 3:
            A, P3 = pl_tensors(p, l1, l2)
            p_top2[sl] = A.sum(1) + A.sum(0)
            p_top3[sl] = P3.sum((1, 2)) + P3.sum((0, 2)) + P3.sum((0, 1))
        else:
            p_top2[sl] = 1.0
            p_top3[sl] = 1.0
        order = np.lexsort((umab[sl], -p))       # p_win 降順、同値は馬番
        r = np.empty(k)
        r[order] = np.arange(1, k + 1)
        mrank[sl] = r
        srt = np.sort(p)[::-1]
        ment[sl] = _norm_entropy(p)
        mtop1[sl] = srt[0]
        mgap[sl] = srt[0] - srt[1] if k >= 2 else srt[0]
    h["p_win"] = p_win
    h["p_top2"] = p_top2
    h["p_top3"] = p_top3
    h["p_place"] = np.where(fp == 2, p_top2, p_top3)
    h["model_rank"] = mrank
    h["model_entropy"] = ment
    h["model_p_top1"] = mtop1
    h["model_p_gap12"] = mgap
    out = h[["rid16", "umaban", "p_win", "p_top3", "p_place", "model_rank", "model_entropy", "model_p_top1",
             "model_p_gap12"]]
    return out, stats


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """load_rows の出力にモデル列を付け、§4 の許可リストの特徴量表を返す（df と同じ行順）。"""
    hp, stats = horse_probs(df)
    d = df.merge(hp, on=["rid16", "umaban"], how="left", validate="many_to_one")
    assert len(d) == len(df)
    tan = d["bet_type"].to_numpy() == "tan"
    d["p_bet"] = np.where(tan, d["p_win"], d["p_place"])
    qb = np.where(tan, d["q_tan_mkt"], d["q_fuku_mkt"])
    d["log_p_over_q"] = np.log(d["p_bet"].to_numpy(float) / qb)
    stats["p_bet_nonfinite"] = int((~np.isfinite(d["p_bet"])).sum())
    stats["lpq_nonfinite"] = int((~np.isfinite(d["log_p_over_q"])).sum())
    return d, stats


def feature_list(mode: str) -> list[str]:
    if mode == "model":
        cols = PRICE_COLS + MODEL_COLS + CONTEXT_COLS
    elif mode == "nomodel":
        cols = PRICE_COLS + CONTEXT_COLS
    else:
        raise ValueError(mode)
    for c in cols:
        assert not c.startswith(FORBIDDEN_INPUT_PREFIX) and c not in FORBIDDEN_INPUTS, c
    return cols


def make_X(d: pd.DataFrame, mode: str) -> pd.DataFrame:
    cols = feature_list(mode)
    X = d[cols].copy()
    for c in CAT_COLS:
        if c in X:
            X[c] = pd.Categorical(X[c].astype(str), categories=CATS[c])
    for c in cols:
        if c not in CAT_COLS:
            X[c] = X[c].astype(float)
    return X


# ---------------------------------------------------------------- §10 超過回収のセル（券種 × 判断時点オッズ十分位）
def decision_odds(d: pd.DataFrame) -> np.ndarray:
    """券の判断時点オッズ: 単勝 = odds_tan、複勝 = odds_fuku_mid（S1 の複勝幅の中点）。"""
    return np.where(d["bet_type"].to_numpy() == "tan", d["odds_tan"].to_numpy(float), d["odds_fuku_mid"].to_numpy(float))


def odds_decile_edges(d_train: pd.DataFrame) -> dict:
    """学習期間の候補で、券種ごとの判断時点オッズの 10%〜90% 点（9 本）。"""
    o = decision_odds(d_train)
    bt = d_train["bet_type"].to_numpy()
    return {b: [float(x) for x in np.quantile(o[bt == b], np.arange(1, 10) / 10.0)] for b in ("tan", "fuku")}


def odds_cell(d: pd.DataFrame, edges: dict) -> np.ndarray:
    """セル番号 0..19（単勝 0-9、複勝 10-19）。境界ちょうどは上の十分位。"""
    o = decision_odds(d)
    bt = d["bet_type"].to_numpy()
    cell = np.empty(len(d), np.int64)
    for k, b in enumerate(("tan", "fuku")):
        m = bt == b
        cell[m] = np.searchsorted(np.asarray(edges[b]), o[m], side="right") + 10 * k
    return cell


def payout_R(d: pd.DataFrame) -> np.ndarray:
    """払戻倍率 R = 払戻 ÷ 100（外れ 0、返還 1、競走中止 0）。結果列が無い表では呼べない。"""
    assert "res_payout_100" in d.columns
    R = d["res_payout_100"].to_numpy(float) / 100.0
    if not np.isfinite(R).all():
        raise ValueError(f"決済できない行がある: {int((~np.isfinite(R)).sum())}")
    return R
