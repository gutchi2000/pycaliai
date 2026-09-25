# -*- coding: utf-8 -*-
"""
market_build.py — EXP18: 構造 loader の出力から race ごとの市場配列を作る (結果を使わない)
=========================================================================================
snapshot の定義:
  terminal_close_market    区分4 (確定)。記録は発走後だが情報時点は締切
  historical_pre_snapshot  区分1 のうちレース当日かつ確定記録の 15 分以上前の最後の 1 本 (EXP16A と同契約)
  am9                      区分1 のうちレース当日で 09:00 に最も近い 1 本 (crux_joint の anchor 定義)
TANPUK と UMAREN は (rid16, 区分, 月日時分) の**完全一致**で結合する。一致しない snapshot は fail-closed。

base race set (結果を使わず TANPUK とレース情報だけで定義):
  平地 (トラックコード(JV) 51..59 以外) / starter = terminal 単勝オッズ > 1.0 の馬番 (EXP16A と同定義) /
  starter >= 5 / 全 starter が terminal で複勝 Lo・Hi を持つ
その後、UMAREN terminal 格子が C(starter,2) で完全でない race を「対象市場で決済不能」として除外する。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .loaders import MIN_GAP_PRE, NMAX, UMAREN_PAIRS, umaren_cols

PAIR_POS = {p: t for t, p in enumerate(UMAREN_PAIRS)}


def pool_matrices(tan: pd.DataFrame, um: pd.DataFrame):
    W = np.column_stack([pd.to_numeric(tan[f"{b}単"], errors="coerce") for b in range(1, NMAX + 1)])
    LO = np.column_stack([pd.to_numeric(tan[f"{b}複Lo"], errors="coerce") for b in range(1, NMAX + 1)])
    HI = np.column_stack([pd.to_numeric(tan[f"{b}複Hi"], errors="coerce") for b in range(1, NMAX + 1)])
    U = um[umaren_cols(um)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return W.astype(float), LO.astype(float), HI.astype(float), U


def snapshot_index(tan: pd.DataFrame, um: pd.DataFrame, info: pd.DataFrame) -> pd.DataFrame:
    """race ごとに terminal / pre / am9 の行番号 (tan, um) を返す。結合は完全一致"""
    key_u = {(r, k, m): i for i, (r, k, m) in enumerate(zip(um["rid16"], um["kubun"], um["mdhm"]))}
    rows = []
    tan = tan.assign(_i=np.arange(len(tan)))
    for rid, g in tan.groupby("rid16", sort=False):
        rec = {"rid16": rid}
        date = int(info.loc[rid, "date"]) if rid in info.index else None
        post = float(info.loc[rid, "post_min"]) if rid in info.index else np.nan
        g4 = g[g["kubun"] == 4]
        g1 = g[g["kubun"] == 1]
        if len(g4):
            t = g4.iloc[-1]
            rec["term_tan"] = int(t["_i"])
            rec["term_um"] = key_u.get((rid, 4, t["mdhm"]), -1)
            rec["term_min"] = float(t["snap_min"])
            rec["term_minus_post"] = float(t["snap_min"] - post) if post == post else np.nan
        if date is not None and len(g1):
            mmdd = f"{date % 10000:04d}"
            same = g1[g1["mmdd"] == mmdd]
            if len(same) and "term_min" in rec:
                ok = same[rec["term_min"] - same["snap_min"] >= MIN_GAP_PRE]
                if len(ok):
                    p = ok.iloc[-1]
                    rec["pre_tan"] = int(p["_i"])
                    rec["pre_um"] = key_u.get((rid, 1, p["mdhm"]), -1)
                    rec["pre_minus_post"] = float(p["snap_min"] - post) if post == post else np.nan
            if len(same):
                a = same.iloc[int(np.argmin(np.abs(same["snap_min"].to_numpy() - 540)))]
                rec["am9_tan"] = int(a["_i"])
                rec["am9_um"] = key_u.get((rid, 1, a["mdhm"]), -1)
                rec["am9_min"] = float(a["snap_min"])
        rows.append(rec)
    return pd.DataFrame(rows).set_index("rid16")


def race_arrays(W, LO, HI, U, tan_i: int, um_i: int):
    """1 snapshot → starter (単勝 > 1.0) と各配列。UMAREN は starter 上の C(n,2) (pair_index 順)"""
    w = W[tan_i]
    bans = np.flatnonzero(np.isfinite(w) & (w > 1.0)) + 1
    lo = LO[tan_i, bans - 1]
    hi = HI[tan_i, bans - 1]
    n = len(bans)
    a, b = np.triu_indices(n, k=1)
    if um_i >= 0:
        cols = [PAIR_POS[(int(bans[x]), int(bans[y]))] for x, y in zip(a, b)]
        uo = U[um_i, cols]
    else:
        uo = np.full(len(a), np.nan)
    return {"bans": bans, "n": n, "win": w[bans - 1], "place_lo": lo, "place_hi": hi, "umaren": uo}


def classify_zero_cells(U_row: np.ndarray, bans: np.ndarray, tou: int) -> dict:
    """terminal UMAREN の 0.0 セルを 頭数外 / 取消・返還可能性 (starter 外) / starter 内欠損 に分ける"""
    S = set(int(x) for x in bans)
    out = {"zero_out_of_field": 0, "zero_scratched_or_refund": 0, "zero_within_starters": 0,
           "valid_within_starters": 0, "eq_1_0_within_starters": 0}
    for (i, j), t in PAIR_POS.items():
        v = U_row[t]
        if i > tou or j > tou:
            if v == 0:
                out["zero_out_of_field"] += 1
            continue
        inside = i in S and j in S
        if v == 0 or not np.isfinite(v):
            if inside:
                out["zero_within_starters"] += 1
            else:
                out["zero_scratched_or_refund"] += 1
        elif inside:
            out["valid_within_starters"] += 1
            out["eq_1_0_within_starters"] += int(v == 1.0)
    return out


def base_race_status(info_row, arr_term) -> str:
    """base race set の判定 (結果を使わない)。戻り値は除外理由か 'base'"""
    if info_row is None:
        return "excl_no_race_info"
    if bool(info_row["is_jump"]):
        return "excl_jump"
    if arr_term["n"] < 5:
        return "excl_small_field"
    if not (np.all(np.isfinite(arr_term["place_lo"])) and np.all(arr_term["place_lo"] > 0)
            and np.all(np.isfinite(arr_term["place_hi"])) and np.all(arr_term["place_hi"] > 0)):
        return "excl_place_odds_missing"
    return "base"


def umaren_grid_complete(arr) -> bool:
    u = arr["umaren"]
    return bool(np.all(np.isfinite(u)) and np.all(u >= 1.0))


def tan_entropy(win_odds: np.ndarray) -> float:
    p = (1.0 / win_odds) / (1.0 / win_odds).sum()
    return float(-(p * np.log(p)).sum())
