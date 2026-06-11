# -*- coding: utf-8 -*-
"""
build_pl_calibrators_serve.py
=============================
serve 条件 (本番で死んでいる特徴をマスクしたスコア) で PL calibrator を再 fit する。
build_pl_calibrators.py の fit ロジックを v6 + serve マスクで踏襲。
本番 pkl は上書きせず models/pl_calibrators_v6_serve.pkl に保存し、
test 2024-25 の serve 条件スコアで ECE を 現行cal vs serve-cal で比較評価する。

背景 (docs/audit_20260611.md / serve_skew_eval.py):
  pl_calibrators_v6.pkl は master_v2 フル特徴のスコア分布で fit されているが、
  本番 (export_weekly_marks → parse_csv) のスコアは特徴欠損分布。
  ミスマッチで本番相当の ECE が複勝2.3倍/馬連1.8倍悪化している。
  → fit 自体を serve 条件スコアで行い、本番分布に整合させる。

マスク: serve_skew_eval の死亡30特徴のうち prev_hosei/prev_hosei9 を除く
28 numeric + 騎手/調教師コード。補正はリネーム修正 (commit 9c03247c) で
本番でも生きるため fit 側でも生かす (将来の定常状態に整合)。

使い方: PYTHONUTF8=1 python build_pl_calibrators_serve.py
出力  : models/pl_calibrators_v6_serve.pkl
        reports/calibrators_v6_serve_eval.json (+ 標準出力に比較表)
"""
from __future__ import annotations

import io
import json
import sys
import warnings
from itertools import combinations, permutations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

import pl_probs as PL
import backtest_pl_ev as be
from backtest_pl_ev import apply_encoders, load_payouts
from serve_skew_eval import (SERVE_DEAD_NOW_PREFIX, SERVE_DEAD_NOW_EXACT,
                             SERVE_DEAD_NOW_CAT, evaluate)

warnings.filterwarnings("ignore")
# 注意: serve_skew_eval が import 時に sys.stdout を TextIOWrapper でラップ済み。
# ここで再ラップすると旧ラッパが GC で close され buffer ごと死ぬ (I/O on closed file)。
# PYTHONUTF8=1 で実行する前提とし、ここではラップしない。

BASE = Path(__file__).parent
MODEL_PKL = BASE / "models/unified_rank_v6.pkl"
PROD_CAL_PKL = BASE / "models/pl_calibrators_v6.pkl"
OUT_PKL = BASE / "models/pl_calibrators_v6_serve.pkl"
OUT_JSON = BASE / "reports/calibrators_v6_serve_eval.json"

COL_RID = "レースID(新/馬番無)"
COL_JYUN = "着順"
COL_BAN = "馬番"


def main():
    print("=" * 64)
    print("build_pl_calibrators_serve.py  (fit=valid2023, serve28 マスク)")
    print("=" * 64)

    bundle = joblib.load(MODEL_PKL)
    model, feats, encs = bundle["model"], bundle["feature_cols"], bundle["encoders"]

    # serve マスク (補正 prev_hosei* と調教 trnH_/trnW_ はリネーム修正済 → 生かす)
    dead_num = [c for c in feats
                if c.startswith(SERVE_DEAD_NOW_PREFIX) or c in SERVE_DEAD_NOW_EXACT]
    dead_cat = [c for c in feats if c in SERVE_DEAD_NOW_CAT]
    print(f"[mask] numeric={len(dead_num)} + cat={len(dead_cat)} "
          f"(補正・調教は生かす)")

    print(f"[load] {be.MASTER_CSV.name}")
    df = pd.read_csv(be.MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]).copy()
    df = df[df["split"].isin(["valid", "test"])].copy()
    df["year"] = df[COL_RID].astype(str).str[:4].astype(int)

    def serve_scores(frame: pd.DataFrame) -> np.ndarray:
        d = frame.copy()
        for c in dead_cat:
            if c in d.columns:
                d[c] = "__NaN__"
        d = apply_encoders(d, encs)
        for c in dead_num:
            if c in d.columns:
                d[c] = np.nan
        X = d[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
        return model.predict(X)

    print("[predict] serve 条件スコア (valid+test 一括) ...")
    df["_score"] = serve_scores(df)

    # ===== Step 1: fit (valid=2023, build_pl_calibrators.py と同一ループ) =====
    vl = df[df["split"] == "valid"]
    print(f"[fit] valid rows={len(vl):,} races={vl[COL_RID].nunique():,}")

    buckets = {k: {"p": [], "y": []} for k in
               ["tansho", "fukusho", "umatan", "umaren", "wide",
                "sanrentan", "sanrenpuku"]}
    n_race = 0
    for rid, g in vl.groupby(COL_RID, sort=False):
        if len(g) < 3:
            continue
        g = g.reset_index(drop=True)
        jyun = g[COL_JYUN].astype(int).values
        w = PL.pl_weights(g["_score"].values)
        n = len(w)
        order = np.argsort(jyun)
        win, plc, sho = order[0], order[1], order[2]
        top3 = {win, plc, sho}
        top2 = {win, plc}

        for i in range(n):
            buckets["tansho"]["p"].append(PL.p_tansho(w, i))
            buckets["tansho"]["y"].append(1 if i == win else 0)
        fuku = PL.all_fukusho(w)
        for i in range(n):
            buckets["fukusho"]["p"].append(fuku[i])
            buckets["fukusho"]["y"].append(1 if i in top3 else 0)
        for i, j in combinations(range(n), 2):
            buckets["umaren"]["p"].append(PL.p_umaren(w, i, j))
            buckets["umaren"]["y"].append(1 if {i, j} == top2 else 0)
        for i, j in combinations(range(n), 2):
            buckets["wide"]["p"].append(PL.p_wide(w, i, j))
            buckets["wide"]["y"].append(1 if {i, j}.issubset(top3) else 0)
        for i, j, k in combinations(range(n), 3):
            buckets["sanrenpuku"]["p"].append(PL.p_sanrenpuku(w, i, j, k))
            buckets["sanrenpuku"]["y"].append(1 if {i, j, k} == top3 else 0)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                buckets["umatan"]["p"].append(PL.p_umatan(w, i, j))
                buckets["umatan"]["y"].append(1 if (i == win and j == plc) else 0)
        for i, j, k in permutations(range(n), 3):
            buckets["sanrentan"]["p"].append(PL.p_sanrentan(w, i, j, k))
            buckets["sanrentan"]["y"].append(
                1 if (i == win and j == plc and k == sho) else 0)

        n_race += 1
        if n_race % 500 == 0:
            print(f"  ..{n_race} races")

    print(f"[fit] races={n_race}")
    calibrators = {}
    for name, d in buckets.items():
        p = np.asarray(d["p"], dtype=np.float64)
        y = np.asarray(d["y"], dtype=np.int8)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p, y)
        calibrators[name] = iso
        print(f"  {name:12s} n={len(p):>10,} raw={p.mean():.5f} "
              f"cal={iso.predict(p).mean():.5f} emp={y.mean():.5f}")

    # 本番 calibrator の無退避上書き防止 (audit 2026-06-11)
    from utils import backup_model
    backup_model(OUT_PKL)
    joblib.dump({
        "calibrators": calibrators,
        "source_model": MODEL_PKL.name,
        "fit_split": "valid=2023 (serve マスクスコア)",
        "serve_mask_numeric": dead_num,
        "serve_mask_cat": dead_cat,
        "n_races": n_race,
        "seed": 42,
    }, OUT_PKL)
    print(f"[saved] {OUT_PKL}")

    # ===== Step 2: test 2024-25 (serve 条件) で ECE 比較 =====
    payouts = load_payouts()
    prod = joblib.load(PROD_CAL_PKL)["calibrators"]
    scores_df = df[[COL_RID, COL_BAN, COL_JYUN, "year", "_score"]].copy()

    print("\n[eval] serveスコア + 現行cal(v6) ...")
    res_prod = evaluate(scores_df, payouts, prod)
    print("[eval] serveスコア + serve-cal ...")
    res_serve = evaluate(scores_df, payouts, calibrators)

    cmp = {"fit": "valid2023 serve28", "prod_cal": res_prod, "serve_cal": res_serve}
    OUT_JSON.write_text(json.dumps(cmp, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print(f"\n=== test 2024-25 (serve 条件スコア) — 現行cal vs serve-cal ===")
    p, s = res_prod["test_2024_2025"], res_serve["test_2024_2025"]
    for label, key in [("ECE 単勝(◎)", "ece_tansho_hon"),
                       ("ECE 複勝(◎)", "ece_fuku_hon"),
                       ("ECE 馬連(◎-〇)", "ece_umaren_hono")]:
        print(f"  {label:<16} 現行={p[key]:.4f}  serve={s[key]:.4f}  "
              f"Δ={s[key]-p[key]:+.4f}")
    print("  (印指標はスコア同一のため両者で不変。改善判定は ECE のみで行う)")
    print(f"[saved] {OUT_JSON}")


if __name__ == "__main__":
    main()
