# -*- coding: utf-8 -*-
"""
hosei_bug_impact.py — prev_hosei の off-by-one + 低カバレッジが精度をどれだけ削るか
=================================================================================
verify_hosei_offbyone.py で確定した事実:
  serve の prev_hosei = 「前々走」の補正タイム (94.15% の行でズレ, 平均絶対差 7.95)
  serve の prev_hosei 充足率 = 46.3% (学習時 85.4%) → 残りは -9999 で埋まる

これを offline の同じレース群に人工的に注入して、AI1位の勝率がどこまで落ちるかを測る。
2026 as-served の実測 (勝率 22.10% / top3 51.00%) に届けば、
「serve 劣化の主因は prev_hosei のバグ」と言い切れる。

実行: python -m analysis.hosei_bug_impact
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
warnings.filterwarnings("ignore")

import backtest_pl_ev as be                                   # noqa: E402
from backtest_pl_ev import apply_encoders, COL_JYUN, COL_RID, COL_BAN  # noqa: E402

H_MASTER = BASE / "data/hosei/H_20130105-20251228.csv"
KEKKA = Path(r"E:\競馬過去走データ\kekka_20130105-20251228_v2.csv")
FILL = -9999.0
SERVE_COV = 0.463          # data/serve_feature_baseline.json
SERVE_WRONG_RATE = 0.9415  # verify_hosei_offbyone.py


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hh = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - hh), 100 * (c + hh)


def serve_value_map():
    """rid18 -> serve が実際に入れている値 (= H[前走].前走補正 = 前々走の補正)"""
    h = pd.read_csv(H_MASTER, encoding="cp932", dtype={"レースID(新)": str})
    h["rid18"] = h["レースID(新)"].astype(str).str.strip().str.zfill(18)
    for c in ("前走補正", "前走補9"):
        h[c] = pd.to_numeric(h[c], errors="coerce")
    hm = h.set_index("rid18")[["前走補正", "前走補9"]]

    k = pd.read_csv(KEKKA, encoding="cp932", low_memory=False,
                    usecols=["日付", "馬名", "レースID(新)"])
    k["rid18"] = k["レースID(新)"].astype(str).str.strip().str.zfill(18)
    k["d"] = pd.to_numeric(k["日付"], errors="coerce")
    k = k.dropna(subset=["d", "馬名"]).sort_values(["馬名", "d"])
    k["prev18"] = k.groupby("馬名")["rid18"].shift(1)
    k = k.dropna(subset=["prev18"]).join(
        hm.rename(columns={"前走補正": "sv_hosei", "前走補9": "sv_h9"}), on="prev18")
    return (k.set_index("rid18")[["sv_hosei", "sv_h9"]]
            .groupby(level=0).first())


def main() -> None:
    be.MODEL_PKL = BASE / "models/unified_rank_v6.pkl"
    bundle = joblib.load(be.MODEL_PKL)
    model, feats, encs = bundle["model"], bundle["feature_cols"], bundle["encoders"]

    df = pd.read_csv(be.MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]).copy()
    for key, mod, fn in [("race_relative_mode", "race_relative_feats", "add_race_relative_feats"),
                         ("course_affinity_mode", "course_affinity_feats", "add_course_affinity_feats"),
                         ("grade_feats_mode", "grade_feats", "add_grade_feats")]:
        if bundle.get(key):
            df = getattr(__import__(mod), fn)(df, mode=bundle[key])
    te = df[df["split"].isin(["test", "valid"])].copy()
    te["rid18"] = (te[COL_RID].astype("int64").astype(str).str.zfill(16)
                   + te[COL_BAN].astype(int).astype(str).str.zfill(2))
    enc = apply_encoders(te.copy(), encs)
    X0 = enc[feats].apply(pd.to_numeric, errors="coerce")
    jyun = te[COL_JYUN].to_numpy()
    rid = te[COL_RID].astype(str).to_numpy()
    print(f"rows={len(te):,} races={te[COL_RID].nunique():,}")

    def ev(X, label):
        sc = model.predict(X.fillna(FILL).values)
        d = pd.DataFrame({"rid": rid, "sc": sc, "jyun": jyun})
        top = d.loc[d.groupby("rid").sc.idxmax()]
        n = len(top)
        w = int((top.jyun == 1).sum())
        t3 = int((top.jyun <= 3).sum())
        lo, hi = wilson(w, n)
        print(f"  {label:<46} 勝率={100*w/n:>5.2f}% [{lo:.1f},{hi:.1f}]  top3={100*t3/n:>5.2f}%")
        return 100 * w / n, 100 * t3 / n

    print("\n=== prev_hosei バグの影響 ===")
    base = ev(X0, "A: 正しい prev_hosei (学習と同じ) 【対照】")

    print("  serve の実値マップを構築中...", flush=True)
    sv = serve_value_map()
    got = te.rid18.map(sv.sv_hosei)
    got9 = te.rid18.map(sv.sv_h9)
    print(f"  マップできた行: {100*got.notna().mean():.1f}%")

    X1 = X0.copy()
    X1["prev_hosei"] = got.to_numpy()
    X1["prev_hosei9"] = got9.to_numpy()
    ev(X1, "B: 値だけ serve と同じ (前々走の値) にする")

    rng = np.random.default_rng(0)
    drop = rng.random(len(X1)) > SERVE_COV
    X2 = X1.copy()
    X2.loc[drop, ["prev_hosei", "prev_hosei9"]] = np.nan
    ev(X2, f"C: B + 充足率を serve 実測 {SERVE_COV:.3f} まで落とす")

    X3 = X0.copy()
    X3.loc[drop, ["prev_hosei", "prev_hosei9"]] = np.nan
    ev(X3, "D: 値は正しいが充足率だけ落とす (欠損単独の影響)")

    X4 = X0.copy()
    X4[["prev_hosei", "prev_hosei9"]] = np.nan
    ev(X4, "E: prev_hosei を完全に捨てる")

    print("\n=== 参考 ===")
    print("  実 serve 2026 (bundle 実測)                    勝率=22.10%  top3=51.00%")
    print("  同期間の市場1番人気                              勝率=29.56%  top3=59.13%")
    print(f"  offline の市場1番人気                           勝率=32.97%  top3=64.05%")
    print("  → 2026 は母集団自体が -3.4pt 難しい。C が 25-26% 付近なら")
    print("     『prev_hosei バグ + 母集団』で serve 劣化はほぼ説明できる。")


if __name__ == "__main__":
    main()
