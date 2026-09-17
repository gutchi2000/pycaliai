# -*- coding: utf-8 -*-
"""
serve_dead_feature_ablation.py — serve 劣化の原因を offline 上で再現する
=======================================================================
測定済みの事実:
  offline(2023-25) の AI1位 勝率 31.75% ≈ 市場1番人気 32.97% (差 -1.22pt)
  2026 as-served   の AI1位 勝率 22.10% ≪ 市場1番人気 29.56% (差 -7.46pt)
  → 期間差では説明できない (対照 = 同じレースの市場)。

仮説: serve では一部の特徴が死んでおり (data/serve_feature_baseline.json)、
      欠損は NaN ではなく **-9999** で埋められる (backtest_pl_ev.score_test:224 と
      同じ契約)。学習時に -9999 をほぼ見ていない特徴が serve で全馬 -9999 になると、
      木は学習時に通っていない枝に全馬を流し込み、順位が壊れる。

検証: offline の同じレースで、serve の欠損パターンを人工的に再現して再スコアし、
      勝率がどこまで落ちるかを見る。22% 付近まで落ちれば原因の再現に成功。

実行: python -m analysis.serve_dead_feature_ablation
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

import backtest_pl_ev as be                                     # noqa: E402
from backtest_pl_ev import apply_encoders, COL_RID, COL_BAN, COL_JYUN  # noqa: E402

MISSING_FILL = -9999.0


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def main() -> None:
    be.MODEL_PKL = BASE / "models/unified_rank_v6.pkl"
    bundle = joblib.load(be.MODEL_PKL)
    model, feats, encs = bundle["model"], bundle["feature_cols"], bundle["encoders"]

    base_cov = json.loads((BASE / "data/serve_feature_baseline.json").read_text(
        encoding="utf-8"))["baseline_cov"]
    dead = [f for f in feats if base_cov.get(f, 1.0) == 0.0]
    partial = {f: base_cov[f] for f in feats if 0.0 < base_cov.get(f, 1.0) < 0.95}
    print(f"feature_cols={len(feats)}  serve完全死={len(dead)}  serve部分欠損={len(partial)}")
    print(f"  完全死: {dead}")

    df = pd.read_csv(be.MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]).copy()
    for mode_key, mod, fn in [("race_relative_mode", "race_relative_feats", "add_race_relative_feats"),
                              ("course_affinity_mode", "course_affinity_feats", "add_course_affinity_feats"),
                              ("grade_feats_mode", "grade_feats", "add_grade_feats")]:
        mode = bundle.get(mode_key)
        if mode:
            df = getattr(__import__(mod), fn)(df, mode=mode)
    te = df[df["split"].isin(["test", "valid"])].copy()
    te = apply_encoders(te, encs)
    Xdf = te[feats].apply(pd.to_numeric, errors="coerce")
    print(f"rows={len(te):,} races={te[COL_RID].nunique():,}")

    # 学習時に各特徴が実際どれくらい埋まっていたか (train split)
    tr = df[df["split"] == "train"]
    tr_enc = apply_encoders(tr.copy(), encs)[feats].apply(pd.to_numeric, errors="coerce")
    train_cov = tr_enc.notna().mean()

    jyun = te[COL_JYUN].to_numpy()
    rid = te[COL_RID].astype(str).to_numpy()

    def evaluate(X, label):
        sc = model.predict(X.fillna(MISSING_FILL).values)
        d = pd.DataFrame({"rid": rid, "sc": sc, "jyun": jyun})
        idx = d.groupby("rid").sc.idxmax()
        top = d.loc[idx]
        n = len(top)
        win = int((top.jyun == 1).sum())
        t3 = int((top.jyun <= 3).sum())
        lo, hi = wilson(win, n)
        print(f"  {label:<40} n={n:>5}  勝率={100*win/n:>5.2f}% [{lo:.1f},{hi:.1f}]  "
              f"top3={100*t3/n:>5.2f}%")
        return 100 * win / n, 100 * t3 / n

    print("\n=== 段階的に serve の欠損を再現する ===")
    base_w, base_t = evaluate(Xdf, "A: offline そのまま (対照)")

    Xb = Xdf.copy()
    for f in dead:
        Xb[f] = np.nan
    evaluate(Xb, f"B: 完全死 {len(dead)}特徴を欠損化")

    rng = np.random.default_rng(0)
    Xc = Xb.copy()
    for f, cov in partial.items():
        mask = rng.random(len(Xc)) > cov
        Xc.loc[mask, f] = np.nan
    evaluate(Xc, f"C: B + 部分欠損 {len(partial)}特徴を serve 率まで間引き")

    Xd = Xdf.copy()
    for f, cov in partial.items():
        mask = rng.random(len(Xd)) > cov
        Xd.loc[mask, f] = np.nan
    evaluate(Xd, f"D: 部分欠損のみ ({len(partial)}特徴)")

    print("\n=== 参考: 2026 as-served 実測 ===")
    print("  E: 実 serve (2026 bundle)                  n= 1353  勝率=22.10%  top3=51.00%")

    print("\n=== 特徴ごとの単独寄与 (その1本だけ殺した時の勝率低下) ===")
    rows = []
    for f in dead + [k for k, v in sorted(partial.items(), key=lambda x: x[1])[:12]]:
        X1 = Xdf.copy()
        X1[f] = np.nan
        sc = model.predict(X1.fillna(MISSING_FILL).values)
        d = pd.DataFrame({"rid": rid, "sc": sc, "jyun": jyun})
        top = d.loc[d.groupby("rid").sc.idxmax()]
        w = 100 * float((top.jyun == 1).mean())
        rows.append((f, base_cov.get(f, 1.0), float(train_cov.get(f, np.nan)), w, w - base_w))
    rows.sort(key=lambda r: r[4])
    print(f"  {'特徴':<26}{'serve被覆':>9}{'train被覆':>10}{'勝率':>8}{'Δ':>8}")
    for f, sc_, tc, w, d_ in rows:
        print(f"  {f:<26}{sc_:>9.3f}{tc:>10.3f}{w:>8.2f}{d_:>+8.2f}")


if __name__ == "__main__":
    main()
