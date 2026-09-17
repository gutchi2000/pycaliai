# -*- coding: utf-8 -*-
"""
serve_coverage_gap_table.py — 学習時と serve の特徴充足率を全120本ぶん突き合わせる
================================================================================
serve_dead_feature_ablation の結論:
  「完全に死んでいる17本」はほぼ無害 (-0.75pt)。効いているのは
  「学習時は埋まっていたのに serve で半分しか埋まらない」部分欠損の側 (-3.1pt)。
既存の canary は完全死の gain% だけ見ているので、この本丸を素通りさせている。

修理すべき順に並べた表を出す。
実行: python -m analysis.serve_coverage_gap_table
"""
from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
warnings.filterwarnings("ignore")

import backtest_pl_ev as be                          # noqa: E402
from backtest_pl_ev import apply_encoders, COL_JYUN, COL_RID  # noqa: E402


def main() -> None:
    be.MODEL_PKL = BASE / "models/unified_rank_v6.pkl"
    bundle = joblib.load(be.MODEL_PKL)
    model, feats, encs = bundle["model"], bundle["feature_cols"], bundle["encoders"]
    gains = model.feature_importance(importance_type="gain")
    gain_pct = {f: 100.0 * g / gains.sum() for f, g in zip(feats, gains)}

    serve_cov = json.loads((BASE / "data/serve_feature_baseline.json").read_text(
        encoding="utf-8"))["baseline_cov"]

    df = pd.read_csv(be.MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]).copy()
    for key, mod, fn in [("race_relative_mode", "race_relative_feats", "add_race_relative_feats"),
                         ("course_affinity_mode", "course_affinity_feats", "add_course_affinity_feats"),
                         ("grade_feats_mode", "grade_feats", "add_grade_feats")]:
        if bundle.get(key):
            df = getattr(__import__(mod), fn)(df, mode=bundle[key])
    tr = apply_encoders(df[df["split"] == "train"].copy(), encs)[feats]
    tr = tr.apply(pd.to_numeric, errors="coerce")
    train_cov = tr.notna().mean()

    rows = []
    for f in feats:
        sc = serve_cov.get(f, np.nan)
        tc = float(train_cov.get(f, np.nan))
        rows.append(dict(feature=f, train=tc, serve=sc, gap=(sc - tc) if np.isfinite(sc) else np.nan,
                         gain=gain_pct.get(f, 0.0)))
    d = pd.DataFrame(rows)
    d["gain_at_risk"] = (-d["gap"]).clip(lower=0) * d["gain"]

    print("=== serve で学習時より充足率が落ちている特徴 (gain 加重で修理優先順) ===")
    print(f"{'特徴':<26}{'train':>7}{'serve':>7}{'gap':>8}{'gain%':>7}{'risk':>7}")
    tot = 0.0
    for r in d.sort_values("gain_at_risk", ascending=False).head(30).itertuples():
        if r.gain_at_risk <= 0:
            break
        tot += r.gain_at_risk
        print(f"{r.feature:<26}{r.train:>7.3f}{r.serve:>7.3f}{r.gap:>+8.3f}"
              f"{r.gain:>7.2f}{r.gain_at_risk:>7.2f}")
    allrisk = d.gain_at_risk.sum()
    print(f"\n  gain×欠損 で失っている量: 上位30本で {tot:.2f} / 全体 {allrisk:.2f} "
          f"(全 gain の {allrisk:.2f}%)")

    print("\n=== 参考: 完全死(serve=0) の内訳 ===")
    dead = d[(d.serve == 0.0)].sort_values("gain", ascending=False)
    print(f"  {len(dead)}本 / gain 合計 {dead.gain.sum():.2f}%  "
          f"(うち train でも欠損していたもの: "
          f"{int((dead.train < 0.5).sum())}本)")
    for r in dead.itertuples():
        note = "  ← train でも欠損なので無害" if r.train < 0.5 else ""
        print(f"    {r.feature:<26} train={r.train:.3f} gain={r.gain:.2f}%{note}")

    d.sort_values("gain_at_risk", ascending=False).to_csv(
        BASE / "reports/deep_bet_search/serve_coverage_gap.csv", index=False,
        encoding="utf-8-sig")
    print("\n書き出し: reports/deep_bet_search/serve_coverage_gap.csv")


if __name__ == "__main__":
    main()
