"""
build_pl_calibrators.py
=======================
unified_rank_v1 のスコア → PL 確率 → valid=2023 の実的中で Isotonic fit。
7 馬券分のキャリブレータを models/pl_calibrators_v1.pkl に保存。

再現性: seed=42, deterministic=True (モデル側), valid split は master の split 列.
"""
from __future__ import annotations

import io
import sys
import warnings
from itertools import permutations, combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

import pl_probs as PL

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE       = Path(__file__).parent
MASTER_CSV = BASE / "data/master_v2_20130105-20251228.csv"
MODEL_PKL  = BASE / "models/unified_rank_v1.pkl"
OUT_PKL    = BASE / "models/pl_calibrators_v1.pkl"

COL_RID  = "レースID(新/馬番無)"
COL_JYUN = "着順"
COL_BAN  = "馬番"


def load_valid(race_relative_mode: str | None = None,
               course_affinity_mode: str | None = None,
               grade_feats_mode: str | None = None):
    print(f"[load] {MASTER_CSV}")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]).copy()
    if race_relative_mode:
        from race_relative_feats import add_race_relative_feats
        df = add_race_relative_feats(df, mode=race_relative_mode)
        print(f"  + race_relative_feats(mode={race_relative_mode})")
    if course_affinity_mode:
        from course_affinity_feats import add_course_affinity_feats
        df = add_course_affinity_feats(df, mode=course_affinity_mode)
        print(f"  + course_affinity_feats(mode={course_affinity_mode})")
    if grade_feats_mode:
        from grade_feats import add_grade_feats
        df = add_grade_feats(df, mode=grade_feats_mode)
        print(f"  + grade_feats(mode={grade_feats_mode})")
    vl = df[df["split"] == "valid"].copy()
    print(f"  valid rows={len(vl):,}  races={vl[COL_RID].nunique():,}")
    return vl


def apply_encoders(df, encs):
    df = df.copy()
    for c, le in encs.items():
        if c not in df.columns: continue
        v = df[c].astype(str).fillna("__NaN__")
        known = set(le.classes_)
        v = v.where(v.isin(known), "__NaN__")
        df[c] = le.transform(v)
    return df


def main():
    print("=" * 60)
    print("build_pl_calibrators.py  (valid=2023)")
    print("=" * 60)

    bundle = joblib.load(MODEL_PKL)
    model  = bundle["model"]
    feats  = bundle["feature_cols"]
    encs   = bundle["encoders"]
    rr_mode = bundle.get("race_relative_mode")
    ca_mode = bundle.get("course_affinity_mode")
    gf_mode = bundle.get("grade_feats_mode")
    print(f"[model] feats={len(feats)}  best_iter={model.best_iteration}  "
          f"race_relative={rr_mode}  course_affinity={ca_mode}  grade_feats={gf_mode}")

    vl = load_valid(race_relative_mode=rr_mode, course_affinity_mode=ca_mode,
                    grade_feats_mode=gf_mode)
    vl = apply_encoders(vl, encs)
    X  = vl[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    vl["_score"] = model.predict(X)

    # 馬券種ごとに (pred_prob, actual) ペアを収集
    buckets = {k: {"p": [], "y": []} for k in
               ["tansho", "fukusho", "umatan", "umaren", "wide", "sanrentan", "sanrenpuku"]}

    n_race = 0
    for rid, g in vl.groupby(COL_RID, sort=False):
        if len(g) < 3:
            continue
        g = g.reset_index(drop=True)
        jyun = g[COL_JYUN].astype(int).values
        w = PL.pl_weights(g["_score"].values)
        n = len(w)

        # 着順 → 実際の 1/2/3 着 index
        order = np.argsort(jyun)  # 着順昇順
        win, plc, sho = order[0], order[1], order[2]
        top3 = {win, plc, sho}
        top2 = {win, plc}

        # --- 単勝 (N 件/R) ---
        for i in range(n):
            buckets["tansho"]["p"].append(PL.p_tansho(w, i))
            buckets["tansho"]["y"].append(1 if i == win else 0)

        # --- 複勝 (N 件/R) ---
        fuku = PL.all_fukusho(w)
        for i in range(n):
            buckets["fukusho"]["p"].append(fuku[i])
            buckets["fukusho"]["y"].append(1 if i in top3 else 0)

        # --- 馬連 (C(N,2) 件/R) ---
        for i, j in combinations(range(n), 2):
            buckets["umaren"]["p"].append(PL.p_umaren(w, i, j))
            buckets["umaren"]["y"].append(1 if {i, j} == top2 else 0)

        # --- ワイド (C(N,2) 件/R) ---
        for i, j in combinations(range(n), 2):
            buckets["wide"]["p"].append(PL.p_wide(w, i, j))
            buckets["wide"]["y"].append(1 if ({i, j}.issubset(top3)) else 0)

        # --- 三連複 (C(N,3) 件/R) ---
        for i, j, k in combinations(range(n), 3):
            buckets["sanrenpuku"]["p"].append(PL.p_sanrenpuku(w, i, j, k))
            buckets["sanrenpuku"]["y"].append(1 if {i, j, k} == top3 else 0)

        # --- 馬単 (N*(N-1) 件/R) ---
        for i in range(n):
            for j in range(n):
                if i == j: continue
                buckets["umatan"]["p"].append(PL.p_umatan(w, i, j))
                buckets["umatan"]["y"].append(1 if (i == win and j == plc) else 0)

        # --- 三連単 (N*(N-1)*(N-2) 件/R) — 重い → サンプリング ---
        # 全 perm は N=18 で 5k/R. 2650R なら 13M, OK. そのまま.
        for i, j, k in permutations(range(n), 3):
            buckets["sanrentan"]["p"].append(PL.p_sanrentan(w, i, j, k))
            buckets["sanrentan"]["y"].append(
                1 if (i == win and j == plc and k == sho) else 0
            )

        n_race += 1
        if n_race % 500 == 0:
            print(f"  ..{n_race} races")

    print(f"\n[fit] races={n_race}")
    calibrators = {}
    for name, d in buckets.items():
        p = np.asarray(d["p"], dtype=np.float64)
        y = np.asarray(d["y"], dtype=np.int8)
        # Isotonic (monotone increasing, clamp [0,1])
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p, y)
        calibrators[name] = iso

        # 簡易 log: raw vs calibrated の平均
        raw_mean  = p.mean()
        cal_mean  = iso.predict(p).mean()
        emp_mean  = y.mean()
        print(f"  {name:12s}  n={len(p):>9,}  raw={raw_mean:.5f}  cal={cal_mean:.5f}  emp={emp_mean:.5f}")

    OUT_PKL.parent.mkdir(exist_ok=True)
    # 本番 calibrator の無退避上書き防止 (audit 2026-06-11): 日付付きで退避してから書く
    from utils import backup_model
    backup_model(OUT_PKL)
    joblib.dump({
        "calibrators": calibrators,
        "source_model": MODEL_PKL.name,
        "fit_split":    "valid=2023",
        "n_races":      n_race,
        "seed":         42,
    }, OUT_PKL)
    print(f"\n[saved] {OUT_PKL}")


if __name__ == "__main__":
    main()
