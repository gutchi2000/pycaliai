"""
audit_auc_ece.py
================
4馬券 (単勝・複勝・馬連・馬単) の AUC と ECE を valid(2023) / test(2024) / test(2025) で計測。

AUC: 識別力 (ランキング性能)
ECE: Expected Calibration Error (確率の正確さ)

raw (PL 素) と cal (Isotonic 後) の両方を出す。
cal が raw より良くなっていればキャリブレーションは効いてる。
test で大きく劣化してたら過学習 or calibrator drift。
"""
from __future__ import annotations
import io, sys, warnings
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss

import pl_probs as PL
from backtest_pl_ev import (
    score_test, apply_encoders, all_umaren_mat, all_umatan_mat, all_fukusho_vec_fast,
    BASE, MASTER_CSV, MODEL_PKL, CAL_PKL, COL_RID, COL_JYUN, COL_BAN,
)

warnings.filterwarnings("ignore")
# stdout wrapper set by backtest_pl_ev import


def ece(probs, labels, n_bins=15):
    """Expected Calibration Error (equal-width bins)."""
    probs = np.asarray(probs); labels = np.asarray(labels)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
    e = 0.0; N = len(probs)
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0: continue
        conf = probs[m].mean()
        acc  = labels[m].mean()
        e += (m.sum() / N) * abs(conf - acc)
    return e


def brier(probs, labels):
    return float(((np.asarray(probs) - np.asarray(labels)) ** 2).mean())


def score_df(split_mask, split_label, master_df, bundle):
    model, feats, encs = bundle["model"], bundle["feature_cols"], bundle["encoders"]
    sub = master_df[split_mask].copy()
    sub = apply_encoders(sub, encs)
    X = sub[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    sub["_score"] = model.predict(X)
    print(f"  [{split_label}] rows={len(sub):,}  races={sub[COL_RID].nunique():,}")
    return sub


def collect(sub, cal):
    """各馬券の (p_raw, p_cal, y) を全レース集約."""
    buckets = {k: {"raw": [], "cal": [], "y": []}
               for k in ["tansho", "fukusho", "umaren", "umatan"]}
    n_race = 0
    for rid, g in sub.groupby(COL_RID, sort=False):
        if len(g) < 3: continue
        g = g.reset_index(drop=True)
        jyun = g[COL_JYUN].astype(int).values
        w = PL.pl_weights(g["_score"].values)
        n = len(w)
        order = np.argsort(jyun)
        win, plc, sho = order[0], order[1], order[2]
        top3 = {win, plc, sho}
        top2 = {win, plc}
        n_race += 1

        # 単勝 (N)
        p_raw = w / w.sum()
        p_cal = cal["tansho"].predict(p_raw)
        y = np.zeros(n); y[win] = 1
        buckets["tansho"]["raw"].append(p_raw); buckets["tansho"]["cal"].append(p_cal); buckets["tansho"]["y"].append(y)

        # 複勝 (N)
        p_raw = all_fukusho_vec_fast(w)
        p_cal = cal["fukusho"].predict(p_raw)
        y = np.zeros(n); y[win] = 1; y[plc] = 1; y[sho] = 1
        buckets["fukusho"]["raw"].append(p_raw); buckets["fukusho"]["cal"].append(p_cal); buckets["fukusho"]["y"].append(y)

        # 馬連 (C(N,2))
        Pm = all_umaren_mat(w)
        iu, ju = np.triu_indices(n, k=1)
        p_raw = Pm[iu, ju]
        p_cal = cal["umaren"].predict(p_raw)
        y = np.array([({int(iu[k]), int(ju[k])} == top2) for k in range(len(iu))], dtype=float)
        buckets["umaren"]["raw"].append(p_raw); buckets["umaren"]["cal"].append(p_cal); buckets["umaren"]["y"].append(y)

        # 馬単 (N*(N-1))
        Um = all_umatan_mat(w)
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        mask = ii != jj
        p_raw = Um[mask]
        p_cal = cal["umatan"].predict(p_raw)
        ii_f = ii[mask]; jj_f = jj[mask]
        y = ((ii_f == win) & (jj_f == plc)).astype(float)
        buckets["umatan"]["raw"].append(p_raw); buckets["umatan"]["cal"].append(p_cal); buckets["umatan"]["y"].append(y)

    # concat
    out = {}
    for k, d in buckets.items():
        out[k] = {
            "raw": np.concatenate(d["raw"]),
            "cal": np.concatenate(d["cal"]),
            "y":   np.concatenate(d["y"]),
        }
    return out, n_race


def report(name, data):
    r = data["raw"]; c = data["cal"]; y = data["y"]
    # AUC
    try:
        auc_r = roc_auc_score(y, r)
        auc_c = roc_auc_score(y, c)
    except ValueError:
        auc_r = auc_c = float("nan")
    ece_r = ece(r, y); ece_c = ece(c, y)
    br_r = brier(r, y); br_c = brier(c, y)
    emp = y.mean()
    mean_r = r.mean(); mean_c = c.mean()
    print(f"  {name:8s}  n={len(y):>10,}  emp={emp:.5f}  "
          f"meanP raw={mean_r:.5f}/cal={mean_c:.5f}  "
          f"AUC raw={auc_r:.4f}/cal={auc_c:.4f}  "
          f"ECE raw={ece_r:.5f}/cal={ece_c:.5f}  "
          f"Brier raw={br_r:.5f}/cal={br_c:.5f}")
    return {
        "n": len(y), "emp": emp,
        "auc_raw": auc_r, "auc_cal": auc_c,
        "ece_raw": ece_r, "ece_cal": ece_c,
        "brier_raw": br_r, "brier_cal": br_c,
    }


def main():
    print("=" * 95)
    print("audit_auc_ece.py  (4馬券: 単勝/複勝/馬連/馬単)")
    print("=" * 95)

    bundle = joblib.load(MODEL_PKL)
    cal = joblib.load(CAL_PKL)["calibrators"]

    print(f"[load] master_v2")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]).copy()
    df["year"] = df[COL_RID].astype(str).str[:4].astype(int)

    for label, mask in [
        ("valid=2023",  df["split"] == "valid"),
        ("test=2024",   (df["split"] == "test") & (df["year"] == 2024)),
        ("test=2025",   (df["split"] == "test") & (df["year"] == 2025)),
    ]:
        print(f"\n>>> {label} <<<")
        sub = score_df(mask, label, df, bundle)
        data, n_race = collect(sub, cal)
        print(f"  races={n_race:,}")
        print()
        for bet in ["tansho", "fukusho", "umaren", "umatan"]:
            report(bet, data[bet])


if __name__ == "__main__":
    main()
