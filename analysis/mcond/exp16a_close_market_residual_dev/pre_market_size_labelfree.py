# -*- coding: utf-8 -*-
"""
pre_market_size_labelfree.py — EXP16A 副腕 pre_market_size_tertiles の label-free 準備
=====================================================================================
spec v0.4 `subsets.learned_boundaries_per_eval_year.items` に事前登録されたまま未実行だった
`pre_market_size_tertiles` (historical_pre_snapshot 時点の単勝票数の三分位) を開ける前の、
**結果 (着順・winner) を一切読まない** 準備段。

やること:
  1. 正式 race set (official_races_all.npz) の各レースについて、race_population.py と
     **同一の規則** で historical_pre_snapshot 行を選び直し、その行の `単勝票数` を取る
     (区分1・レース当日・最後の確定行の 15 分以上前の最後の 1 本)。
  2. 選び直した行の de-vig が npz の pi_pre と一致することを全レースで assert する
     (= EXP16A 本体と同じスナップショットを使っている証明)。
  3. 評価年 Y ごとに Y-1 以前 (2016..Y-1) の正式 race から三分位境界を作り、年別・分位別の
     レース数 / 開催日数を数える。

読まないもの: winner_pos / 着順 / Q1 / 2024-2025。
公開物には票数の生値・境界の生値を書かない (境界は sha256 のみ記録)。
出力: out/pre_market_size_labelfree.json
実行: python -m analysis.mcond.exp16a_close_market_residual_dev.pre_market_size_labelfree
"""
from __future__ import annotations

import hashlib
import json
import sys
import time

import numpy as np

from .provenance import OUT, MIN_GAP_PRE, load_tanpuk, odds_matrix, devig
from .build_oof import RESEARCH

EVAL_YEARS = [2019, 2020, 2021, 2022, 2023]
FIT_YEAR_MIN = 2016
OUT_FILE = OUT / "pre_market_size_labelfree.json"


def pre_votes_by_race() -> dict:
    """正式 race set の rid16 順に、pre スナップの単勝票数を返す (outcome 非参照)"""
    d = np.load(RESEARCH / "official_races_all.npz", allow_pickle=True)
    rid = d["rid16"].astype(str)
    year = d["year"].astype(int)
    off = d["offsets"]
    ban = d["ban"]
    pi_pre = d["pi_pre"]
    assert year.max() <= 2023 and year.min() >= 2016, "封印期間 (2024/2025) が混入している"
    t = load_tanpuk(set(rid))
    assert (t["rid16"].str[:4].astype(int) <= 2023).all()
    votes, max_err, snap_gap = {}, 0.0, {}
    groups = {k: g for k, g in t.groupby("rid16", sort=False)}
    for i, r in enumerate(rid):
        g = groups[r]
        mmdd = r[4:8]
        fin_rows = g[g["kubun"] == 4]
        pre_c = g[(g["kubun"] == 1) & (g["mmdd"] == mmdd)]
        f0 = fin_rows.iloc[-1]
        gap = f0.snap_min - pre_c["snap_min"]
        ok = pre_c[gap >= MIN_GAP_PRE]
        p0 = ok.iloc[-1]
        od_fin, od_pre = odds_matrix(f0), odds_matrix(p0)
        starters = sorted(od_fin)
        pip_d, _, _ = devig(od_pre, starters)
        bs = [int(b) for b in ban[off[i]:off[i + 1]]]
        sp = np.array([pip_d[b] for b in bs])
        sp = sp / sp.sum()
        err = float(np.abs(sp - pi_pre[off[i]:off[i + 1]]).max())
        max_err = max(max_err, err)
        v = float(p0["votes_tan"])
        votes[r] = v
        snap_gap[r] = float(f0.snap_min - p0.snap_min)
    assert max_err < 1e-9, f"pi_pre の再現誤差 {max_err}"
    vv = np.array([votes[r] for r in rid])
    assert np.isfinite(vv).all() and (vv > 0).all(), "票数の欠損/0 がある"
    return {"rid": rid, "year": year, "day": d["day"].astype(str), "votes": vv,
            "pi_pre_max_abs_err": max_err}


def boundaries_by_year(V: dict) -> dict:
    """評価年 Y の三分位境界を Y-1 以前 (2016..Y-1) の正式 race の pre 票数から作る"""
    out = {}
    for Y in EVAL_YEARS:
        m = (V["year"] >= FIT_YEAR_MIN) & (V["year"] <= Y - 1)
        x = V["votes"][m]
        out[Y] = (float(np.quantile(x, 1 / 3)), float(np.quantile(x, 2 / 3)))
    return out


def assign_tertile(V: dict, bnd: dict) -> np.ndarray:
    """0 = 小 (<= t1), 1 = 中 (<= t2), 2 = 大。favorite_pre_odds の cells_for_year と同じ境界規約"""
    tq = np.full(len(V["rid"]), -1, dtype=int)
    for Y in EVAL_YEARS:
        m = V["year"] == Y
        t1, t2 = bnd[Y]
        x = V["votes"][m]
        tq[m] = np.where(x <= t1, 0, np.where(x <= t2, 1, 2))
    return tq


def main():
    t0 = time.time()
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    V = pre_votes_by_race()
    bnd = boundaries_by_year(V)
    tq = assign_tertile(V, bnd)
    names = ["T1_small", "T2_mid", "T3_large"]
    counts = {}
    for Y in EVAL_YEARS:
        m = V["year"] == Y
        counts[str(Y)] = {names[k]: {"races": int((tq[m] == k).sum()),
                                     "meeting_days": int(len(set(V["day"][m][tq[m] == k])))}
                          for k in range(3)}
    pooled = {names[k]: {"races": int(((tq == k) & np.isin(V["year"], EVAL_YEARS)).sum())}
              for k in range(3)}
    # 年別の票数水準は比 (年中央値 / 2016 中央値) だけを記録する (生値を出さない)
    med = {int(y): float(np.median(V["votes"][V["year"] == y])) for y in range(2016, 2024)}
    ratio = {str(y): med[y] / med[2016] for y in med}
    bnd_hash = hashlib.sha256(json.dumps({str(k): v for k, v in bnd.items()},
                                         sort_keys=True).encode()).hexdigest()
    res = {
        "role": "EXP16A 副腕 pre_market_size_tertiles の label-free 準備 (結果を読まない)",
        "spec": "v0.4 (commit e7603678) subsets.learned_boundaries_per_eval_year",
        "created": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "read": ["official_races_all.npz (rid16/year/day/ban/pi_pre/offsets のみ)", "TANPUK_*.csv"],
        "not_read": ["winner_pos", "着順", "Q1", "2024/2025"],
        "snapshot_rule": "race_population.py と同一 (区分1・当日・最後の確定行の 15 分以上前の最後の 1 本)",
        "pi_pre_reproduction_max_abs_err": V["pi_pre_max_abs_err"],
        "pre_minus_final_snapshot_gap_min": "race_population.py と同一の行選択であり、pi_pre 一致で検証",
        "boundary_rule": "評価年 Y ごとに 2016..Y-1 の正式 race の pre 単勝票数の 1/3・2/3 分位。"
                         "x <= t1 → T1, x <= t2 → T2, それ以外 T3",
        "boundaries_sha256": bnd_hash,
        "boundaries_raw": "公開物に出さない (票数の生値に当たるため)",
        "year_median_votes_ratio_to_2016": ratio,
        "counts_by_eval_year": counts,
        "pooled_counts_2019_2023": pooled,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    OUT_FILE.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ["pi_pre_reproduction_max_abs_err",
                                           "year_median_votes_ratio_to_2016",
                                           "counts_by_eval_year", "pooled_counts_2019_2023"]},
                     ensure_ascii=False, indent=1))
    print(f"[saved] {OUT_FILE} ({res['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
