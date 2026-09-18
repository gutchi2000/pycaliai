# -*- coding: utf-8 -*-
"""
prev_hosei_proxy.py — TARGET の補正タイムが途切れた期間を自前の推定値で埋められるか
=================================================================================
背景:
  prev_hosei は v6 の gain 第2位 (7.47%)。TARGET の補正タイムは 2026-05-31 で止まっており
  (analysis/hosei_cutoff_probe.py)、それ以降に前走がある馬は serve で prev_hosei が欠損する。
  ユーザー手元の TARGET もこれが最新 = 待っても埋まらない。

方針:
  学習 split で「serve でも取れる他の特徴 → prev_hosei」の回帰器を作り、
  欠損時にその推定値で埋める。

採用ゲート (offline test+valid 10,365R):
  ◎勝率/top3 を 3 条件で比較
    real    : 本物の prev_hosei (上限)
    missing : prev_hosei を欠損 (= 今の serve で 6/1 以降に起きること)
    proxy   : 推定値で埋める
  proxy が missing を有意に上回り、real との差の大半を回収したら配線する。

出力: models/prev_hosei_proxy.pkl  (feature_cols / model / 評価値)
実行: python -m analysis.prev_hosei_proxy
"""
from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
warnings.filterwarnings("ignore")

import backtest_pl_ev as be                                     # noqa: E402
from backtest_pl_ev import apply_encoders, COL_JYUN, COL_RID    # noqa: E402

FILL = -9999.0
TARGETS = ["prev_hosei", "prev_hosei9"]
OUT = BASE / "models/prev_hosei_proxy.pkl"


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

    base = json.loads((BASE / "data/serve_feature_baseline.json").read_text(encoding="utf-8"))
    serve_cov = base["baseline_cov"]
    # serve で安定して取れる特徴だけを説明変数に使う (serve で死んでる特徴に頼ると意味がない)
    xcols = [f for f in feats if f not in TARGETS and serve_cov.get(f, 0.0) >= 0.60]
    print(f"説明変数: {len(xcols)} 本 (serve 充足 60% 以上, prev_hosei 系を除く)")

    df = pd.read_csv(be.MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]).copy()
    for key, mod, fn in [("race_relative_mode", "race_relative_feats", "add_race_relative_feats"),
                         ("course_affinity_mode", "course_affinity_feats", "add_course_affinity_feats"),
                         ("grade_feats_mode", "grade_feats", "add_grade_feats")]:
        if bundle.get(key):
            df = getattr(__import__(mod), fn)(df, mode=bundle[key])
    enc = apply_encoders(df.copy(), encs)
    X = enc[feats].apply(pd.to_numeric, errors="coerce")
    split = df["split"].to_numpy()

    proxies = {}
    for t in TARGETS:
        y = X[t]
        tr = (split == "train") & y.notna().to_numpy()
        va = (split == "valid") & y.notna().to_numpy()
        te = (split == "test") & y.notna().to_numpy()
        reg = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.03, num_leaves=127,
                                min_child_samples=50, subsample=0.8, subsample_freq=1,
                                colsample_bytree=0.8, verbose=-1)
        reg.fit(X.loc[tr, xcols], y[tr], eval_set=[(X.loc[va, xcols], y[va])],
                callbacks=[lgb.early_stopping(100, verbose=False)])
        pred = reg.predict(X.loc[te, xcols])
        yt = y[te].to_numpy()
        r2 = 1 - ((yt - pred) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
        corr = np.corrcoef(yt, pred)[0, 1]
        print(f"  {t}: test R²={r2:.3f}  相関={corr:.3f}  MAE={np.abs(yt-pred).mean():.2f}  "
              f"(sd={yt.std():.2f}, 反復={reg.best_iteration_})")
        proxies[t] = reg

    # ---- 採用ゲート: ◎の精度 ----
    ev = np.isin(split, ["test", "valid"])
    Xe = X[ev].copy()
    rid = df.loc[ev, COL_RID].astype(str).to_numpy()
    jy = df.loc[ev, COL_JYUN].to_numpy()

    def top(Xm):
        sc = model.predict(Xm.fillna(FILL).values)
        d = pd.DataFrame({"rid": rid, "sc": sc, "jy": jy})
        t = d.loc[d.groupby("rid").sc.idxmax()]
        return t.set_index("rid")

    real = top(Xe)
    Xm = Xe.copy()
    Xm[TARGETS] = np.nan
    miss = top(Xm)
    Xp = Xe.copy()
    for t in TARGETS:
        Xp[t] = proxies[t].predict(Xe[xcols])
    prox = top(Xp)
    # 現実的な形: 本物があるところは本物、無いところだけ推定 (学習時も15%は欠損していた)
    Xh = Xe.copy()
    for t in TARGETS:
        na = Xh[t].isna()
        Xh.loc[na, t] = proxies[t].predict(Xe.loc[na, xcols])
    hyb = top(Xh)

    print("\n=== 採用ゲート (offline 2023-25, 10,365R) ===")
    res = {}
    for name, t in [("real (上限)", real), ("missing (今の6/1以降)", miss),
                    ("proxy 全置換", prox), ("real+欠損だけproxy", hyb)]:
        n = len(t)
        w = int((t.jy == 1).sum())
        t3 = int((t.jy <= 3).sum())
        lo, hi = wilson(w, n)
        res[name] = (100 * w / n, 100 * t3 / n)
        print(f"  {name:<22} ◎勝率={100*w/n:>5.2f}% [{lo:.1f},{hi:.1f}]  ◎top3={100*t3/n:>5.2f}%")

    # 対応比較 proxy vs missing
    j = miss.join(prox, lsuffix="_m", rsuffix="_p")
    a = int(((j.jy_p == 1) & (j.jy_m != 1)).sum())
    b = int(((j.jy_p != 1) & (j.jy_m == 1)).sum())
    from math import comb
    p = min(1.0, 2 * sum(comb(a + b, i) for i in range(min(a, b) + 1)) / 2 ** (a + b)) if a + b else 1
    gap = res["real (上限)"][0] - res["missing (今の6/1以降)"][0]
    rec = res["proxy 全置換"][0] - res["missing (今の6/1以降)"][0]
    print(f"\n  proxy vs missing: ◎勝率 {rec:+.2f}pt (proxyのみ的中={a} / missingのみ={b}, McNemar p={p:.4f})")
    print(f"  real との差の回収率: {100*rec/gap:.0f}%" if gap > 0 else "  (real と missing に差なし)")

    verdict = bool(p < 0.05 and rec > 0)
    print(f"\n  判定: {'★ 採用 (配線する)' if verdict else '不採用'}")
    joblib.dump({"models": proxies, "xcols": xcols, "targets": TARGETS,
                 "gate": {k: list(v) for k, v in res.items()},
                 "mcnemar_p": p, "adopt": verdict}, OUT)
    print(f"保存: {OUT}")


if __name__ == "__main__":
    main()
