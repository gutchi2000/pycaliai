# -*- coding: utf-8 -*-
"""
dump_v6_scores.py — 本番 v6 の OOS 馬別確率を1枚の parquet に焼く
=================================================================
audit_marks.py と同じ経路 (backtest_pl_ev.score_test + pl_probs + pl_calibrators_v6)
を使い、valid(2023)+test(2024-) 全レースの馬別 p_win / p_fuku を出す。

deep_bet_search が使うサブストレートの確率源。これを v5 由来の
analysis/_tmp_marks_flat.parquet の代わりに使うことで、探索結果が
そのまま本番 v6 の話になる。

出力: data/_policy/v6_scores.parquet  (rid, ban, score, p_win, p_fuku, jyun)
実行: python -m analysis.dump_v6_scores --model v6
"""
from __future__ import annotations
import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
warnings.filterwarnings("ignore")

import pl_probs as PL                                    # noqa: E402
import backtest_pl_ev as be                              # noqa: E402
from backtest_pl_ev import score_test, COL_RID, COL_BAN  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v6")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tag = args.model

    be.MODEL_PKL = BASE / f"models/unified_rank_{tag}.pkl"
    be.CAL_PKL = BASE / f"models/pl_calibrators_{tag}.pkl"
    be.CURVE_PKL = BASE / f"data/pl_payout_curve_{tag}.pkl"
    assert be.MODEL_PKL.exists(), f"{be.MODEL_PKL} not found"
    print(f"[model] {be.MODEL_PKL}")

    te = score_test(include_valid=True)
    print(f"[rows] {len(te):,}  races {te[COL_RID].nunique():,}")

    cal = joblib.load(be.CAL_PKL) if be.CAL_PKL.exists() else None
    cals = cal.get("calibrators") if isinstance(cal, dict) else None
    print(f"[cal] {be.CAL_PKL if cals else 'none'}")

    from backtest_pl_ev import all_fukusho_vec_fast

    out = []
    for rid, g in te.groupby(COL_RID, sort=False):
        if len(g) < 5:
            continue
        g = g.sort_values(COL_BAN)
        sc = g["_score"].to_numpy(float)
        w = PL.pl_weights(sc)
        p_win = PL.all_tansho(w)
        p_fuku = all_fukusho_vec_fast(w)
        if cals is not None:
            p_win = np.clip(cals["tansho"].predict(p_win), 0.0, 1.0)
            p_fuku = np.clip(cals["fukusho"].predict(p_fuku), 0.0, 1.0)
        rid_s = str(int(rid)) if isinstance(rid, (int, np.integer)) else str(rid)
        jyun = pd.to_numeric(g["着順"], errors="coerce").to_numpy()
        for b, s, pw, pf, jy in zip(g[COL_BAN].to_numpy(int), sc, p_win, p_fuku, jyun):
            out.append((rid_s, int(b), float(s), float(pw), float(pf), float(jy)))

    df = pd.DataFrame(out, columns=["rid", "ban", "score", "p_win", "p_fuku", "jyun"])
    dest = Path(args.out) if args.out else BASE / "data/_policy/v6_scores.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    print(f"saved {len(df):,} rows / {df.rid.nunique():,} races -> {dest}")
    df["yr"] = df.rid.str[:4]
    print(df.groupby("yr").rid.nunique().to_string())


if __name__ == "__main__":
    main()
