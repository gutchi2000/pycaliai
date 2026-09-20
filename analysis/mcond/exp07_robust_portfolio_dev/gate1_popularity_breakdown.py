# -*- coding: utf-8 -*-
"""
gate1_popularity_breakdown.py — item 9の人気帯別内訳を軽量に補足する。
P6のCVaR探索(高コスト)は再実行せず、候補選定(較正tansho上位3頭選択、安価)だけを
再計算し、1番人気候補(単勝オッズ最小=最有力候補)の人気順位帯でP6-P1差を分類する。
STAGE2A_RESULTS.json(既存の決済済みP1/P6損益)と結合する。
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
BASE = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
import pl_probs as PL  # noqa: E402
import joblib  # noqa: E402

from analysis.mcond.exp07_robust_portfolio_dev.stage2a_run import (  # noqa: E402
    load_candidates_no_results, load_historical_pre_snapshot_odds, select_candidates, TOP_N,
)


def popularity_rank_of_top_pick(race_df: pd.DataFrame, top: pd.DataFrame, odds_for_race: dict) -> int | None:
    """calibrated tansho確率1位の候補馬が、historical_pre_snapshotの単勝オッズで
    レース全体(オッズが取得できた馬に限る)の何番人気かを返す(1=最も人気=最低オッズ)。"""
    top1_ban = int(top.sort_values("cal_tansho_p", ascending=False).iloc[0]["ban"])
    all_bans = [int(b) for b in race_df["ban"]]
    odds_list = []
    for b in all_bans:
        o = odds_for_race.get(b)
        if o is not None and o.get("tansho_odds") is not None:
            odds_list.append((b, o["tansho_odds"]))
    if not odds_list:
        return None
    odds_list.sort(key=lambda x: x[1])
    ranks = {b: i + 1 for i, (b, _) in enumerate(odds_list)}
    return ranks.get(top1_ban)


def band_of(rank: int | None) -> str:
    if rank is None:
        return "unknown"
    if rank <= 1:
        return "1番人気"
    if rank <= 3:
        return "2-3番人気"
    if rank <= 6:
        return "4-6番人気"
    return "7番人気以下"


def main():
    d = json.loads((HERE / "out" / "STAGE2A_RESULTS.json").read_text(encoding="utf-8"))
    results_by_rid = {r["rid16"]: r for r in d["results"]}

    calib = joblib.load(BASE / "models" / "pl_calibrators_v6.pkl")
    calibrators = calib["calibrators"]

    band_diff = defaultdict(list)
    n_no_rank = 0
    for year in (2024, 2025):
        print(f"[popularity] {year}: loading candidates/odds (軽量、P6探索なし)...")
        df = load_candidates_no_results(year)
        odds = load_historical_pre_snapshot_odds(year)
        race_ids = df["rid16"].unique().tolist()
        for i, rid16 in enumerate(race_ids):
            r = results_by_rid.get(rid16)
            if r is None:
                continue  # 異常レース等でStage2A結果に無い
            race_df = df[df["rid16"] == rid16]
            top = select_candidates(race_df, calibrators)
            if top is None:
                continue
            odds_for_race = odds.get(rid16, {})
            rank = popularity_rank_of_top_pick(race_df, top, odds_for_race)
            if rank is None:
                n_no_rank += 1
                continue
            band = band_of(rank)
            band_diff[band].append(r["p6_profit"] - r["p1_profit"])
            if (i + 1) % 1000 == 0:
                print(f"[popularity] {year}: {i+1}/{len(race_ids)} done")

    print("=" * 70)
    print("9c. 人気帯別 (1番人気候補の単勝人気順位で分類、P6-P1差)")
    out = {}
    order = ["1番人気", "2-3番人気", "4-6番人気", "7番人気以下", "unknown"]
    for band in order:
        vals = band_diff.get(band, [])
        if not vals:
            continue
        arr = np.array(vals, dtype=float)
        print(f"  {band}: n={len(arr)}  平均差={arr.mean():.2f}円/レース  合計差={arr.sum():.0f}円")
        out[band] = {"n": len(arr), "mean_diff": float(arr.mean()), "sum_diff": float(arr.sum())}
    print(f"  (人気順位ランク不能: {n_no_rank}件)")

    (HERE / "out" / "GATE1_POPULARITY_BREAKDOWN.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {HERE / 'out' / 'GATE1_POPULARITY_BREAKDOWN.json'}")


if __name__ == "__main__":
    main()
