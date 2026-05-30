"""
build_pl_calibrators_course.py
==============================
案 3: コース別 (芝/ダ × 距離区分) で別々の Isotonic calibrator を fit。

区分 (8 個):
  芝_short:  芝 ≤ 1400m
  芝_mile:   芝 1401-1800m
  芝_middle: 芝 1801-2200m
  芝_long:   芝 ≥ 2201m
  ダ_short:  ダ ≤ 1400m
  ダ_mile:   ダ 1401-1800m
  ダ_middle: ダ 1801-2200m
  ダ_long:   ダ ≥ 2201m

各区分内で valid=2023 のレースを集計し、 isotonic fit。
区分内サンプル不足 (<200 samples per bucket) はフォールバック (全体 fit) を使う。

入力モデル: --model v6 (default)
出力: models/pl_calibrators_{tag}_course.pkl
  構造:
    {
      "calibrators_by_bin": {
        "芝_short": {tansho: iso, fukusho: iso, ...},
        ...
      },
      "fallback_calibrators": {tansho: iso, ...},  # 全体 fit
      "source_model": "unified_rank_v6.pkl",
      "bin_def": {...},
    }

評価モード (--evaluate): 区分別 ECE を全体 fit と比較。

実行例:
  python build_pl_calibrators_course.py
  python build_pl_calibrators_course.py --evaluate
"""
from __future__ import annotations
import argparse
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

COL_RID  = "レースID(新/馬番無)"
COL_JYUN = "着順"
COL_BAN  = "馬番"

DIST_BINS = [
    ("short",  0,    1400),
    ("mile",   1401, 1800),
    ("middle", 1801, 2200),
    ("long",   2201, 9999),
]

MIN_SAMPLES_PER_BIN = 200  # 区分内 samples 未満ならフォールバック (全体 fit)


def race_bin(td: str, dist: int) -> str | None:
    if pd.isna(td) or pd.isna(dist): return None
    td = str(td)
    if td not in ["芝", "ダ"]: return None
    d = int(dist)
    for name, lo, hi in DIST_BINS:
        if lo <= d <= hi:
            return f"{td}_{name}"
    return None


def collect_buckets(vl, model, feats):
    """全レースを 1 周し、 ペア (pred_prob, actual, bin) を集める."""
    X  = vl[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    vl["_score"] = model.predict(X)

    # bin別 + 全体  buckets
    types = ["tansho", "fukusho", "umatan", "umaren",
             "wide", "sanrentan", "sanrenpuku"]
    buckets_by_bin = {}   # bin -> {type: {"p":[], "y":[]}}
    buckets_all    = {k: {"p": [], "y": []} for k in types}

    n_race = 0
    bin_counts = {}
    for rid, g in vl.groupby(COL_RID, sort=False):
        if len(g) < 3: continue
        g = g.reset_index(drop=True)
        td   = g["芝・ダ"].iloc[0] if "芝・ダ" in g.columns else None
        dist = g["距離"].iloc[0]    if "距離"    in g.columns else None
        bin_name = race_bin(td, dist)
        if bin_name is None: continue
        bin_counts[bin_name] = bin_counts.get(bin_name, 0) + 1

        jyun = g[COL_JYUN].astype(int).values
        w = PL.pl_weights(g["_score"].values)
        n = len(w)
        order = np.argsort(jyun)
        win, plc, sho = int(order[0]), int(order[1]), int(order[2])
        top3 = {win, plc, sho}
        top2 = {win, plc}

        # bin 別 と 全体 両方に push する
        bb = buckets_by_bin.setdefault(bin_name, {k: {"p": [], "y": []} for k in types})

        for i in range(n):
            p_t = PL.p_tansho(w, i)
            y_t = 1 if i == win else 0
            buckets_all["tansho"]["p"].append(p_t); buckets_all["tansho"]["y"].append(y_t)
            bb["tansho"]["p"].append(p_t);          bb["tansho"]["y"].append(y_t)

        fuku = PL.all_fukusho(w)
        for i in range(n):
            y_f = 1 if i in top3 else 0
            buckets_all["fukusho"]["p"].append(fuku[i]); buckets_all["fukusho"]["y"].append(y_f)
            bb["fukusho"]["p"].append(fuku[i]);          bb["fukusho"]["y"].append(y_f)

        for i, j in combinations(range(n), 2):
            p = PL.p_umaren(w, i, j)
            y = 1 if {i, j} == top2 else 0
            buckets_all["umaren"]["p"].append(p); buckets_all["umaren"]["y"].append(y)
            bb["umaren"]["p"].append(p);          bb["umaren"]["y"].append(y)

        for i, j in combinations(range(n), 2):
            p = PL.p_wide(w, i, j)
            y = 1 if {i, j}.issubset(top3) else 0
            buckets_all["wide"]["p"].append(p); buckets_all["wide"]["y"].append(y)
            bb["wide"]["p"].append(p);          bb["wide"]["y"].append(y)

        for i, j, k in combinations(range(n), 3):
            p = PL.p_sanrenpuku(w, i, j, k)
            y = 1 if {i, j, k} == top3 else 0
            buckets_all["sanrenpuku"]["p"].append(p); buckets_all["sanrenpuku"]["y"].append(y)
            bb["sanrenpuku"]["p"].append(p);          bb["sanrenpuku"]["y"].append(y)

        for i in range(n):
            for j in range(n):
                if i == j: continue
                p = PL.p_umatan(w, i, j)
                y = 1 if (i == win and j == plc) else 0
                buckets_all["umatan"]["p"].append(p); buckets_all["umatan"]["y"].append(y)
                bb["umatan"]["p"].append(p);          bb["umatan"]["y"].append(y)

        for i, j, k in permutations(range(n), 3):
            p = PL.p_sanrentan(w, i, j, k)
            y = 1 if (i == win and j == plc and k == sho) else 0
            buckets_all["sanrentan"]["p"].append(p); buckets_all["sanrentan"]["y"].append(y)
            bb["sanrentan"]["p"].append(p);          bb["sanrentan"]["y"].append(y)

        n_race += 1
        if n_race % 500 == 0:
            print(f"  ..{n_race} races")
    return buckets_all, buckets_by_bin, n_race, bin_counts


def fit_calibrators(buckets):
    cals = {}
    for name, d in buckets.items():
        p = np.asarray(d["p"], dtype=np.float64)
        y = np.asarray(d["y"], dtype=np.int8)
        if len(p) < 50:
            cals[name] = None  # too small
            continue
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p, y)
        cals[name] = iso
    return cals


def apply_encoders(df, encs):
    df = df.copy()
    for c, le in encs.items():
        if c not in df.columns: continue
        v = df[c].astype(str).fillna("__NaN__")
        known = set(le.classes_)
        v = v.where(v.isin(known), "__NaN__")
        df[c] = le.transform(v)
    return df


def ece_high_p(p, y, threshold=0.1):
    """高 p 帯の |予測 mean - 実勢 mean|."""
    p = np.asarray(p); y = np.asarray(y)
    mask = p >= threshold
    if mask.sum() < 30: return None
    return float(abs(p[mask].mean() - y[mask].mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v6", choices=["v5", "v6", "v7", "v8"])
    ap.add_argument("--evaluate", action="store_true",
                    help="区分別 ECE と全体 fit ECE を比較")
    args = ap.parse_args()
    tag = args.model

    MODEL_PKL = BASE / f"models/unified_rank_{tag}.pkl"
    OUT_PKL   = BASE / f"models/pl_calibrators_{tag}_course.pkl"

    print("=" * 70)
    print(f"build_pl_calibrators_course.py  (model={tag}, valid=2023)")
    print("=" * 70)

    bundle = joblib.load(MODEL_PKL)
    model  = bundle["model"]
    feats  = bundle["feature_cols"]
    encs   = bundle["encoders"]
    rr_mode = bundle.get("race_relative_mode")
    ca_mode = bundle.get("course_affinity_mode")
    print(f"[model] feats={len(feats)}  best_iter={model.best_iteration}  "
          f"race_relative={rr_mode}  course_affinity={ca_mode}")

    print(f"\n[load] {MASTER_CSV}")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]).copy()
    if rr_mode:
        from race_relative_feats import add_race_relative_feats
        df = add_race_relative_feats(df, mode=rr_mode)
    if ca_mode:
        from course_affinity_feats import add_course_affinity_feats
        df = add_course_affinity_feats(df, mode=ca_mode)
    vl = df[df["split"] == "valid"].copy()
    # 区分判定用に「芝・ダ」「距離」は encoder 適用前の値を残す
    vl["_td_raw"]   = vl["芝・ダ"]
    vl["_dist_raw"] = pd.to_numeric(vl["距離"], errors="coerce")
    vl = apply_encoders(vl, encs)
    # encoder 適用後に元の値を上書き
    vl["芝・ダ"] = vl["_td_raw"]
    vl["距離"]   = vl["_dist_raw"]
    print(f"  valid rows={len(vl):,}  races={vl[COL_RID].nunique():,}")

    print("\n[collect] (rid 単位で buckets を集計)")
    b_all, b_by_bin, n_race, bin_counts = collect_buckets(vl, model, feats)
    print(f"\n[races] {n_race:,}")
    print("[bin counts]")
    for bn in sorted(bin_counts):
        print(f"  {bn:>12s}: {bin_counts[bn]:>5,} races")

    print("\n[fit] fallback (全体)")
    fallback = fit_calibrators(b_all)
    for name, iso in fallback.items():
        if iso is None: continue
        p = np.asarray(b_all[name]["p"]); y = np.asarray(b_all[name]["y"])
        print(f"  {name:12s} n={len(p):>9,}  raw={p.mean():.5f}  "
              f"cal={iso.predict(p).mean():.5f}  emp={y.mean():.5f}")

    print("\n[fit] bin 別")
    cals_by_bin = {}
    for bn, buckets in b_by_bin.items():
        n_min = min(len(d["p"]) for d in buckets.values())
        if n_min < MIN_SAMPLES_PER_BIN:
            print(f"  {bn:>12s} skipped (min samples = {n_min} < {MIN_SAMPLES_PER_BIN})")
            continue
        cals_by_bin[bn] = fit_calibrators(buckets)
        print(f"  {bn:>12s} fit OK")

    out = {
        "calibrators_by_bin":   cals_by_bin,
        "fallback_calibrators": fallback,
        "source_model":         MODEL_PKL.name,
        "fit_split":            "valid=2023",
        "n_races":              n_race,
        "bin_counts":           bin_counts,
        "dist_bin_def":         DIST_BINS,
        "min_samples_per_bin":  MIN_SAMPLES_PER_BIN,
        "seed":                 42,
    }
    OUT_PKL.parent.mkdir(exist_ok=True)
    joblib.dump(out, OUT_PKL)
    print(f"\n[saved] {OUT_PKL}")

    if args.evaluate:
        print("\n" + "=" * 70)
        print("評価: 区分別 ECE_high_p vs 全体 fit ECE_high_p")
        print("=" * 70)
        for name in ["tansho", "fukusho", "umaren", "wide"]:
            p_all = np.asarray(b_all[name]["p"])
            y_all = np.asarray(b_all[name]["y"])
            iso_all = fallback[name]
            # 全体 fit を全体評価
            ece_global_all   = ece_high_p(p_all, y_all)
            ece_calib_global = ece_high_p(iso_all.predict(p_all), y_all)
            print(f"\n=== {name} ===")
            print(f"  [全体 fit]  raw_ECE={ece_global_all:.5f}  cal_ECE={ece_calib_global:.5f}")

            # 区分別 fit を 区分内で評価して比重平均
            n_total = 0; ece_weighted = 0.0; ece_global_eval_w = 0.0
            for bn, buckets in b_by_bin.items():
                if bn not in cals_by_bin: continue
                p_b = np.asarray(buckets[name]["p"])
                y_b = np.asarray(buckets[name]["y"])
                if len(p_b) == 0: continue
                iso_b = cals_by_bin[bn][name]
                ece_b_self = ece_high_p(iso_b.predict(p_b), y_b)
                ece_b_glob = ece_high_p(iso_all.predict(p_b), y_b)
                if ece_b_self is None: continue
                n_total += len(p_b)
                ece_weighted     += ece_b_self * len(p_b)
                ece_global_eval_w += (ece_b_glob if ece_b_glob is not None else 0.0) * len(p_b)
                ece_b_glob_str = f"{ece_b_glob:.5f}" if ece_b_glob is not None else "N/A"
                print(f"    {bn:>12s} n={len(p_b):>7,}  bin_self_ECE={ece_b_self:.5f}  "
                      f"global_fit_ECE_on_bin={ece_b_glob_str}")
            if n_total > 0:
                avg_self = ece_weighted / n_total
                avg_glob = ece_global_eval_w / n_total
                print(f"  [区分別 fit 加重平均]  ECE={avg_self:.5f}")
                print(f"  [全体 fit を区分別評価 加重平均]  ECE={avg_glob:.5f}")
                improvement = avg_glob - avg_self
                pct = improvement / avg_glob * 100 if avg_glob > 0 else 0
                print(f"  改善: {improvement:+.5f}  ({pct:+.1f}%)")


if __name__ == "__main__":
    main()
