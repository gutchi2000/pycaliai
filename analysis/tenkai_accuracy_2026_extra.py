"""tenkai_accuracy_2026 の補足（CI・月別・新馬除外・JRDB 前日予想との比較・平均の上限）。分析のみ。

  python -m analysis.tenkai_accuracy_2026_extra --root E:/PyCaLiAI --cache <tenkai_accuracy_2026 と同じ --cache>
"""
from __future__ import annotations

import argparse
import json
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.tenkai_accuracy_2026 import PLACE, auc, brier3, load_tenkai, pace_block, wilson


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="E:/PyCaLiAI")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--start", default="20260418")
    ap.add_argument("--end", default="20261115")
    a = ap.parse_args()
    root, cache = Path(a.root), Path(a.cache)
    R = pd.read_parquet(cache.with_suffix(".races.parquet"))
    S = pd.read_parquet(cache.with_suffix(".horses.parquet"))
    meta = json.loads((root / "reports/tenkai/meta.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(11)
    out = {}

    # 平均の確率の上限（σ と区切り幅から）
    Phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))
    for s, (c1, c2) in meta["cuts"].items():
        w = (c2 - c1) / 2 / meta["sigma"]
        out[f"p_mid_max_{s}"] = round(2 * Phi(w) - 1, 3)

    TR, TH = load_tenkai(root / "reports/tenkai", root / "site/data", a.start, a.end)
    T = TR.join(R, on="race_key", rsuffix="_t", how="inner")
    out["klass_values"] = T["klass"].value_counts().head(40).to_dict()
    T["shinba"] = T["klass"].astype(str).str.contains("新馬")
    T["plab"] = T[["p_slow", "p_mid", "p_high"]].to_numpy().argmax(1)
    T["ptop"] = T[["p_slow", "p_mid", "p_high"]].max(axis=1)
    out["p_mid_observed_max"] = float(T["p_mid"].max())

    # 新馬を除く
    out["pace_B_excl_shinba"] = pace_block(T[~T["shinba"]], "pace_b_cls")
    out["pace_B_shinba"] = pace_block(T[T["shinba"]], "pace_b_cls")
    # 予想の向き × 確度
    rows = []
    for lab, nm in ((0, "スロー"), (2, "ハイ"), (1, "平均")):
        for sure, m in (("公算が大きい(≥50%)", T["ptop"] >= 0.5), ("寄り(<50%)", T["ptop"] < 0.5)):
            g = T[(T["plab"] == lab) & m & T["pace_b_cls"].notna()]
            if len(g) == 0:
                continue
            d = np.bincount(g["pace_b_cls"].astype(int), minlength=3) / len(g)
            k = int((g["pace_b_cls"] == lab).sum())
            rows.append({"予想": nm, "確度": sure, "n": int(len(g)), "実際 スロー/平均/ハイ": [round(x, 3) for x in d],
                         "的中": round(k / len(g), 3), "ci": [round(x, 3) for x in wilson(k, len(g))]})
    out["direction_x_sure_B"] = rows

    # 月別
    T["month"] = T["date"].str[4:6]
    mon = []
    for mth, g in T.groupby("month"):
        b = pace_block(g, "pace_b_cls")
        gl = g[g["late_y"].notna()]
        mon.append({"month": mth, "races": int(len(g)), "pace_acc_B": round(b["acc"], 3),
                    "bss_B": round(b["bss_unif"], 3), "late_auc": round(auc(gl["late_y"], gl["late_p"]), 3)})
    out["by_month"] = mon

    # 届くか: AUC 差（コースの率との差）のブートストラップ
    cr = json.loads((root / "reports/tenkai/course_late_rate.json").read_text(encoding="utf-8"))
    T["course_rate"] = (T["place"] + T["surface"] + T["dist"].astype(int).astype(str)).map(cr["course"]).fillna(
        T["surface"].map(cr["surface"]))
    tl = T[T["late_y"].notna() & T["late_p"].notna()].reset_index(drop=True)
    y, p, c = tl["late_y"].to_numpy(), tl["late_p"].to_numpy(), tl["course_rate"].to_numpy()
    bs, bb = [], []
    for _ in range(1000):
        s = rng.integers(0, len(y), len(y))
        bs.append(auc(y[s], p[s]) - auc(y[s], c[s]))
        bb.append(np.mean((p[s] - y[s]) ** 2) - np.mean((c[s] - y[s]) ** 2))
    out["late_auc_minus_course_ci"] = [round(float(np.percentile(bs, q)), 3) for q in (2.5, 50, 97.5)]
    out["late_brier_minus_course_ci"] = [round(float(np.percentile(bb, q)), 5) for q in (2.5, 50, 97.5)]

    # 番手: tenkai − 前 5 走平均 の Spearman 差（レース単位の対応ありブートストラップ）
    Sx = S[(S["date"] >= pd.Timestamp(a.start)) & S["ran"]].copy()
    for cc in ("c1", "c2", "c3", "c4"):
        Sx[cc] = Sx[cc].where(Sx[cc] > 0)
    Sx["early"] = Sx["c1"].fillna(Sx["c2"]).fillna(Sx["c3"]).fillna(Sx["c4"])
    H = TH.merge(Sx[["race_key", "umaban", "early", "c4", "N", "rk"]], on=["race_key", "umaban"], how="inner")
    # 素朴基準（前 5 走平均）
    Sh = S[S["ran"]].copy()
    for cc in ("c1", "c2", "c3", "c4"):
        Sh[cc] = Sh[cc].where(Sh[cc] > 0)
    Sh["e_rel"] = (Sh["c1"].fillna(Sh["c2"]).fillna(Sh["c3"]).fillna(Sh["c4"]) - 1) / np.maximum(Sh["N"] - 1, 1)
    Sh = Sh[Sh["e_rel"].notna() & (Sh["ped"] != "")].sort_values(["ped", "date", "race_key"])
    Sh["b5"] = Sh.groupby("ped")["e_rel"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    H = H.merge(Sh[["race_key", "umaban", "b5"]], on=["race_key", "umaban"], how="left")
    H["b5"] = H["b5"].fillna(0.5)
    # JRDB 前日 KYI の 道中順位（参考ベンチマーク）
    K = pd.read_parquet(root / "data/jrdb/kyi_tenkai.parquet", columns=["racekey", "ban", "k_d_jun", "k_pace_q"])
    K = K[K["racekey"].str[2:4] == "26"].rename(columns={"racekey": "rk", "ban": "umaban"})
    K["umaban"] = K["umaban"].astype(int)
    H = H.merge(K[["rk", "umaban", "k_d_jun"]], on=["rk", "umaban"], how="left")
    sp = {"pe": [], "b5": [], "k_d_jun": [], "pl4": [], "k4": []}
    for _, g in H.groupby("race_key"):
        g = g[g["early"].notna() & g["c4"].notna()]
        if len(g) < 5 or g["k_d_jun"].isna().any():
            continue
        ar, a4 = g["early"].rank(), g["c4"].rank()
        sp["pe"].append(g["pe"].rank().corr(ar))
        sp["b5"].append(g["b5"].rank().corr(ar))
        sp["k_d_jun"].append(g["k_d_jun"].rank().corr(ar))
        sp["pl4"].append(g["pl"].rank().corr(a4))
        sp["k4"].append(g["k_d_jun"].rank().corr(a4))
    sp = {k: np.array(v, dtype=float) for k, v in sp.items()}
    ok = ~np.isnan(sp["pe"]) & ~np.isnan(sp["b5"]) & ~np.isnan(sp["k_d_jun"])

    def pair_ci(x, y):
        d = x - y
        bs = [np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(1000)]
        return [round(float(np.percentile(bs, q)), 3) for q in (2.5, 50, 97.5)]
    out["pos_bench_races"] = int(ok.sum())
    out["pos_bench_mean"] = {k: round(float(np.nanmean(v[ok])), 3) for k, v in sp.items()}
    out["early_tenkai_minus_base5_ci"] = pair_ci(sp["pe"][ok], sp["b5"][ok])
    out["early_tenkai_minus_jrdb_ci"] = pair_ci(sp["pe"][ok], sp["k_d_jun"][ok])
    out["c4_tenkai_minus_jrdb_ci"] = pair_ci(sp["pl4"][ok], sp["k4"][ok])

    # ペース: JRDB 前日のペース予想（H/M/S）との比較（同じレース、正解 B）
    kp = K.groupby("rk")["k_pace_q"].first().map({-1.0: 0, 0.0: 1, 1.0: 2})
    T["jrdb_pred"] = T["rk"].map(kp)
    q = T[T["jrdb_pred"].notna() & T["pace_b_cls"].notna()]
    out["jrdb_pace_pred"] = {"n": int(len(q)), "tenkai_acc": round(float((q["plab"] == q["pace_b_cls"]).mean()), 3),
                             "jrdb_acc": round(float((q["jrdb_pred"] == q["pace_b_cls"]).mean()), 3),
                             "jrdb_pred_dist": (q["jrdb_pred"].value_counts(normalize=True).sort_index().round(3).to_dict()),
                             "agree_tenkai_jrdb": round(float((q["plab"] == q["jrdb_pred"]).mean()), 3),
                             "jrdb_acc_vs_jrdb_actual": round(float((q["jrdb_pred"] == q["jrdb_cls"])[q["jrdb_cls"].notna()].mean()), 3),
                             "tenkai_acc_vs_jrdb_actual": round(float((q["plab"] == q["jrdb_cls"])[q["jrdb_cls"].notna()].mean()), 3)}
    # 順位相関の比較（連続値）: tenkai p_high − p_slow vs JRDB は 3 値だけなので一致率のみ
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
