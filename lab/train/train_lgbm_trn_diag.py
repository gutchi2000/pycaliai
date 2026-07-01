"""
train_lgbm_trn_diag.py
======================
「調教(坂路)を新特徴量として足すと test 印精度が62%の天井を破るか」を測る。

設計: 同一パラメータ(v6 best)で 2本学習し、差分で調教の純効果を isolate。
  A) baseline   : v6 feats のみ (LabelEncoded cat → numeric, v6 と同じ)
  B) +坂路       : v6 feats + 坂路 as-of 特徴量5列

リークフリー保証:
  - 坂路は merge_asof(backward, tol=14d, allow_exact_matches=False)
    = レース日より厳密に前の、14日以内の最新追い切りのみ。未来情報なし。
  - 時系列split (train<=2022 / valid2023 / test2024-25) 厳守。

比較基準: 本番v6 (test 2024-25) ◎top1=30.24% ◎top3=62.03% win∈top5=78.20%
"""
from __future__ import annotations
import io
import sys
import time
import warnings
from itertools import groupby
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE       = Path(__file__).parent
MASTER_CSV = BASE / "data/master_v2_20130105-20251228.csv"
TRAIN_DIR  = BASE / "data/training"
SEED = 42

COL_RID  = "レースID(新/馬番無)"
COL_JYUN = "着順"
COL_BAN  = "馬番"

LEAK_COLS = {
    "着順", "fukusho_flag", "roi_target",
    "レースID(新)", "レースID(新/馬番無)",
    "馬名", "レース名", "発走時刻", "date_dt", "日付",
    "血統登録番号", "split",
}
CAT_COLS = [
    "場所", "芝・ダ", "コース区分", "芝(内・外)", "馬場状態", "天気",
    "クラス名", "種牡馬", "父タイプ名", "母父馬", "母父タイプ名", "毛色",
    "馬主(最新/仮想)", "生産者", "騎手コード", "調教師コード",
    "年齢限定", "限定", "性別限定", "指定条件", "重量種別", "性別",
    "ブリンカー", "前走場所", "前芝・ダ", "前走馬場状態", "前走競走種別", "前好走",
]
# v6 best params (reports/optuna_v6_marks.json)
V6_PARAMS = dict(num_leaves=59, max_depth=12, min_data_in_leaf=197,
                 learning_rate=0.0508, feature_fraction=0.876,
                 bagging_fraction=0.7032, lambda_l1=0.001108, lambda_l2=7.538)
V6_BEST_ITER = 469
TRN_FEATS = ["trn_h_time1", "trn_h_lap1", "trn_h_accel", "trn_h_days", "trn_h_has"]
V6_REF = {"hon_top1": 0.3024, "hon_top3": 0.6203, "winner_in_top5": 0.7820}


def parse_date(s):
    return pd.to_datetime(s.astype(str).str.replace(r"[/\-]", "", regex=True),
                          format="%Y%m%d", errors="coerce")


def add_hill_asof(df):
    """坂路 as-of 結合 (leak-free)。df に _rdate(レース日) が必要。"""
    print("  [坂路] as-of 結合 ...")
    h = pd.read_csv(TRAIN_DIR / "H-20150401-20260313.csv", encoding="cp932")
    h = h.rename(columns={"年月日": "wd", "馬名": "馬名", "Time1": "t1", "Lap1": "l1", "Lap4": "l4"})
    h["date_dt"] = parse_date(h["wd"])
    h["馬名"] = h["馬名"].astype(str).str.strip()
    for c in ["t1", "l1", "l4"]:
        h[c] = pd.to_numeric(h[c], errors="coerce")
    h = h[(h["t1"] > 40) & (h["t1"] < 100)].dropna(subset=["date_dt", "馬名"])
    h["wdate"] = h["date_dt"]
    R = h[["馬名", "date_dt", "wdate", "t1", "l1", "l4"]].sort_values("date_dt").reset_index(drop=True)

    # master_v2 に既存の date_dt があると merge で suffix 衝突するので除去 (_rdate を使う)
    df = df.drop(columns=[c for c in ["date_dt"] if c in df.columns])
    L = df.sort_values("_rdate").reset_index(drop=True)
    merged = pd.merge_asof(
        L, R, left_on="_rdate", right_on="date_dt", by="馬名",
        direction="backward", tolerance=pd.Timedelta(days=14),
        allow_exact_matches=False,
    )
    merged["trn_h_time1"] = merged["t1"]
    merged["trn_h_lap1"]  = merged["l1"]
    merged["trn_h_accel"] = merged["l1"] - merged["l4"]
    merged["trn_h_days"]  = (merged["_rdate"] - merged["wdate"]).dt.days
    merged["trn_h_has"]   = merged["t1"].notna().astype(int)
    cov = merged["trn_h_has"].mean() * 100
    print(f"    結合後カバレッジ {cov:.1f}%")
    return merged.drop(columns=["date_dt", "wdate", "t1", "l1", "l4"])


def prep():
    print(f"[load] {MASTER_CSV}")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]).copy()
    df["label"] = np.clip(6 - df[COL_JYUN].astype(int), 0, 5).astype(int)
    df["_rdate"] = parse_date(df["日付"])
    df = df.dropna(subset=["_rdate"])
    df = add_hill_asof(df)

    base_feats = [c for c in df.columns if c not in LEAK_COLS and c != "label"
                  and c not in TRN_FEATS and c != "_rdate"]
    # LabelEncode (v6 と同じ)
    encs = {}
    tr = df[df["split"] == "train"]
    for c in CAT_COLS:
        if c not in df.columns: continue
        le = LabelEncoder()
        le.fit(pd.concat([tr[c].astype(str).fillna("__NaN__"),
                          pd.Series(["__NaN__"])], ignore_index=True))
        encs[c] = le
    for c, le in encs.items():
        v = df[c].astype(str).fillna("__NaN__")
        v = v.where(v.isin(set(le.classes_)), "__NaN__")
        df[c] = le.transform(v)
    return df, base_feats


def make_ds(d, feats):
    d = d.sort_values(COL_RID).reset_index(drop=True)
    X = d[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    y = d["label"].values.astype(int)
    g = np.array([len(list(gr)) for _, gr in groupby(d[COL_RID])])
    return lgb.Dataset(X, label=y, group=g, free_raw_data=False)


def train(d_tr, d_vl, feats):
    params = dict(objective="lambdarank", lambdarank_truncation_level=5,
                  metric="ndcg", eval_at=[5], **V6_PARAMS,
                  bagging_freq=5, verbose=-1, n_jobs=-1, seed=SEED,
                  deterministic=True, force_col_wise=True, feature_pre_filter=False)
    m = lgb.train(params, make_ds(d_tr, feats), num_boost_round=int(V6_BEST_ITER * 1.1),
                  valid_sets=[make_ds(d_vl, feats)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def evaluate(d_te, model, feats):
    d = d_te.sort_values(COL_RID).reset_index(drop=True)
    X = d[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    d["_s"] = model.predict(X)
    n = h1 = h3 = w5 = 0
    for rid, g in d.groupby(COL_RID, sort=False):
        if len(g) < 5: continue
        s = g["_s"].values; jy = g[COL_JYUN].astype(int).values
        n += 1
        order = np.argsort(-s); hon = int(order[0])
        top5 = set(int(x) for x in order[:5])
        sij = np.argsort(jy); wi, pi_, si = int(sij[0]), int(sij[1]), int(sij[2])
        if hon == wi: h1 += 1
        if hon in {wi, pi_, si}: h3 += 1
        if wi in top5: w5 += 1
    return {"n": n, "hon_top1": h1/n, "hon_top3": h3/n, "winner_in_top5": w5/n}


def main():
    t0 = time.time()
    df, base_feats = prep()
    tr = df[df["split"] == "train"]; vl = df[df["split"] == "valid"]; te = df[df["split"] == "test"]
    print(f"  train={len(tr):,}  valid={len(vl):,}  test={len(te):,}  base_feats={len(base_feats)}")

    print("\n[A] baseline (坂路なし) 学習 ...")
    mA = train(tr, vl, base_feats)
    rA = evaluate(te, mA, base_feats)

    print("[B] +坂路 学習 ...")
    feats_B = base_feats + TRN_FEATS
    mB = train(tr, vl, feats_B)
    rB = evaluate(te, mB, feats_B)

    print("\n" + "=" * 66)
    print(f"=== 坂路調教の純効果  test 2024-25 ({rA['n']:,} races) ===")
    print("=" * 66)
    print(f"{'指標':16s} {'A:baseline':>11s} {'B:+坂路':>10s} {'差(B-A)':>9s} {'本番v6':>8s}")
    for k in ["hon_top1", "hon_top3", "winner_in_top5"]:
        a, b = rA[k]*100, rB[k]*100
        ref = V6_REF[k]*100
        print(f"{k:16s} {a:10.2f}% {b:9.2f}% {b-a:+8.2f}pt {ref:7.2f}%")
    print("=" * 66)

    # 坂路特徴量の重要度
    imp = pd.Series(mB.feature_importance(importance_type="gain"), index=feats_B)
    print("\n[坂路特徴量の gain importance (全体順位)]")
    rank = imp.rank(ascending=False).astype(int)
    for f in TRN_FEATS:
        print(f"  {f:14s} gain={imp[f]:12,.0f}  全{len(feats_B)}中 {rank[f]}位")
    print(f"\n[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
