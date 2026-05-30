"""
build_payout_curve.py
=====================
train+valid 期 (≤ 2023) のレースで
「PL補正後 prob bin × 馬券種」ごとの **実配当中央値テーブル**を作る。

これを payout_table として EV = p_cal × median_pay から期待 ROI を出す。
リーク無し (test=2024-2025 は使わない).

出力: data/pl_payout_curve_v1.pkl
"""
from __future__ import annotations

import io
import sys
import warnings
from itertools import combinations, permutations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import pl_probs as PL

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE       = Path(__file__).parent
MASTER_CSV = BASE / "data/master_v2_20130105-20251228.csv"
KEKKA_CSV  = BASE / "data/kekka_20130105-20251228.csv"
WIDE_PQ    = BASE / "data/wide_payouts_2016-2025.parquet"
MODEL_PKL  = BASE / "models/unified_rank_v1.pkl"
CAL_PKL    = BASE / "models/pl_calibrators_v1.pkl"
OUT_PKL    = BASE / "data/pl_payout_curve_v1.pkl"

COL_RID  = "レースID(新/馬番無)"
COL_JYUN = "着順"
COL_BAN  = "馬番"

# prob 区間 (log bin に近い、的中帯を細かく)
BINS = np.array([0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.01])


def load_payouts():
    df = pd.read_csv(KEKKA_CSV, encoding="cp932", low_memory=False)
    df.columns = ["rid_horse", "ban", "ped", "jyun",
                  "tansho", "fukusho", "wakuren", "umaren",
                  "umatan", "sanrenpuku", "sanrentan"]
    df["rid"] = df["rid_horse"].astype(str).str[:16]
    df["ban"] = pd.to_numeric(df["ban"], errors="coerce").astype("Int64")
    df["jyun"] = pd.to_numeric(df["jyun"], errors="coerce").astype("Int64")
    for c in ["tansho", "fukusho", "umaren", "umatan", "sanrenpuku", "sanrentan"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    out = {}
    for rid, g in df.groupby("rid", sort=False):
        r1 = g[g["jyun"] == 1]
        r2 = g[g["jyun"] == 2]
        r3 = g[g["jyun"] == 3]
        if len(r1) == 0 or len(r2) == 0 or len(r3) == 0:
            continue
        w_row, p_row, s_row = r1.iloc[0], r2.iloc[0], r3.iloc[0]
        out[rid] = {
            "win": int(w_row["ban"]), "plc": int(p_row["ban"]), "sho": int(s_row["ban"]),
            "tansho":      w_row["tansho"],
            "fuku_win":    w_row["fukusho"],
            "fuku_plc":    p_row["fukusho"],
            "fuku_sho":    s_row["fukusho"],
            "umaren":      w_row["umaren"],
            "umatan":      w_row["umatan"],
            "sanrenpuku":  w_row["sanrenpuku"],
            "sanrentan":   w_row["sanrentan"],
        }
    return out


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
    print("build_payout_curve.py  (train+valid ≤ 2023)")
    print("=" * 60)

    bundle = joblib.load(MODEL_PKL)
    model, feats, encs = bundle["model"], bundle["feature_cols"], bundle["encoders"]
    rr_mode = bundle.get("race_relative_mode")
    ca_mode = bundle.get("course_affinity_mode")
    cal = joblib.load(CAL_PKL)["calibrators"]
    print(f"[model] race_relative={rr_mode}  course_affinity={ca_mode}")

    print("[load] payouts")
    payouts = load_payouts()

    # ワイド 払戻 (odds_test.csv 由来, 2016-2025)
    print("[load] wide payouts parquet")
    wide_df = pd.read_parquet(WIDE_PQ)
    wide_df["race_id"] = wide_df["race_id"].astype(str)
    wide_map = {}
    for _, r in wide_df.iterrows():
        combos = []
        for k in (1, 2, 3):
            i, j, pay = r[f"w{k}_i"], r[f"w{k}_j"], r[f"w{k}_pay"]
            if pd.notna(i) and pd.notna(j) and pd.notna(pay):
                combos.append((int(i), int(j), float(pay)))
        if combos:
            wide_map[r["race_id"]] = combos
    print(f"  wide races: {len(wide_map):,}")

    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]).copy()
    if rr_mode:
        from race_relative_feats import add_race_relative_feats
        df = add_race_relative_feats(df, mode=rr_mode)
        print(f"  + race_relative_feats(mode={rr_mode})")
    if ca_mode:
        from course_affinity_feats import add_course_affinity_feats
        df = add_course_affinity_feats(df, mode=ca_mode)
        print(f"  + course_affinity_feats(mode={ca_mode})")
    tv = df[df["split"].isin(["train", "valid"])].copy()
    print(f"[data] train+valid rows={len(tv):,}  races={tv[COL_RID].nunique():,}")

    tv = apply_encoders(tv, encs)
    X = tv[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    tv["_score"] = model.predict(X)

    # bet_type → list of (prob_cal, pay)
    hits = {k: [] for k in ["tansho", "fukusho", "umaren", "umatan", "wide", "sanrenpuku", "sanrentan"]}

    n_race = 0
    for rid, g in tv.groupby(COL_RID, sort=False):
        if len(g) < 3: continue
        rid_s = str(int(rid)) if isinstance(rid, (int, np.integer)) else str(rid)
        if rid_s not in payouts: continue
        po = payouts[rid_s]

        g = g.sort_values(COL_BAN).reset_index(drop=True)
        ban = g[COL_BAN].astype(int).values
        b2i = {int(b): i for i, b in enumerate(ban)}
        wi, pi, si = b2i.get(po["win"]), b2i.get(po["plc"]), b2i.get(po["sho"])
        if wi is None or pi is None or si is None: continue
        n_race += 1

        w = PL.pl_weights(g["_score"].values)

        # 各馬券、実際に当たった combo の (prob_cal, pay) を記録
        if po["tansho"] and not pd.isna(po["tansho"]):
            p = cal["tansho"].predict([PL.p_tansho(w, wi)])[0]
            hits["tansho"].append((p, float(po["tansho"])))
        if po["fuku_win"] and not pd.isna(po["fuku_win"]):
            fu = PL.all_fukusho(w)
            for idx, pay in [(wi, po["fuku_win"]), (pi, po["fuku_plc"]), (si, po["fuku_sho"])]:
                if pay and not pd.isna(pay):
                    p = cal["fukusho"].predict([fu[idx]])[0]
                    hits["fukusho"].append((p, float(pay)))
        if po["umaren"] and not pd.isna(po["umaren"]):
            p = cal["umaren"].predict([PL.p_umaren(w, wi, pi)])[0]
            hits["umaren"].append((p, float(po["umaren"])))
        # ワイド (3 combos / race) — race_id で結合
        w_combos = wide_map.get(rid_s)
        if w_combos:
            for ui, uj, wpay in w_combos:
                ii = b2i.get(ui); jj = b2i.get(uj)
                if ii is None or jj is None:
                    continue
                p = cal["wide"].predict([PL.p_wide(w, ii, jj)])[0]
                hits["wide"].append((p, wpay))
        if po["umatan"] and not pd.isna(po["umatan"]):
            p = cal["umatan"].predict([PL.p_umatan(w, wi, pi)])[0]
            hits["umatan"].append((p, float(po["umatan"])))
        if po["sanrenpuku"] and not pd.isna(po["sanrenpuku"]):
            p = cal["sanrenpuku"].predict([PL.p_sanrenpuku(w, wi, pi, si)])[0]
            hits["sanrenpuku"].append((p, float(po["sanrenpuku"])))
        if po["sanrentan"] and not pd.isna(po["sanrentan"]):
            p = cal["sanrentan"].predict([PL.p_sanrentan(w, wi, pi, si)])[0]
            hits["sanrentan"].append((p, float(po["sanrentan"])))

        if n_race % 5000 == 0:
            print(f"  ..{n_race} races")

    print(f"\n[aggregate] races={n_race}")

    # bin 別中央値
    curves = {}
    for bet, rows in hits.items():
        if not rows:
            curves[bet] = None; continue
        p = np.array([r[0] for r in rows])
        pay = np.array([r[1] for r in rows])
        bin_idx = np.digitize(p, BINS) - 1
        bin_idx = np.clip(bin_idx, 0, len(BINS) - 2)
        medians = np.full(len(BINS) - 1, np.nan)
        means   = np.full(len(BINS) - 1, np.nan)
        counts  = np.zeros(len(BINS) - 1, dtype=int)
        for b in range(len(BINS) - 1):
            mask = bin_idx == b
            if mask.any():
                medians[b] = np.median(pay[mask])
                means[b]   = np.mean(pay[mask])
                counts[b]  = int(mask.sum())
        curves[bet] = {"bins": BINS, "median_pay": medians, "mean_pay": means, "count": counts}
        print(f"\n  [{bet}]  hits={len(rows):,}")
        for b in range(len(BINS) - 1):
            if counts[b] > 0:
                print(f"    p ∈ [{BINS[b]:.3f},{BINS[b+1]:.3f})  n={counts[b]:>6,}  "
                      f"med={medians[b]:>8.0f}  mean={means[b]:>8.0f}")

    joblib.dump({
        "curves":  curves,
        "bins":    BINS,
        "source":  MODEL_PKL.name,
        "cal":     CAL_PKL.name,
        "split":   "train+valid (≤2023)",
        "n_races": n_race,
    }, OUT_PKL)
    print(f"\n[saved] {OUT_PKL}")


if __name__ == "__main__":
    main()
