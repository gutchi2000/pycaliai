"""目的関数の検証: v6 と同じ特徴・同じ学習データで、勝敗を log-loss で直接学習したモデルを作る (分析専用)。

データ: data/master_v2_20130105-20251228.csv (v6 C1 の学習に使った候補 master と SHA-256 一致)。
  学習 = split "train" (〜2022)、早期停止 = split "valid" (2023)、補助評価 = split "test" (2024-25)。
特徴: models/unified_rank_v6.pkl の feature_cols (120 列) と encoders をそのまま使う
  (配信側の X と同じエンコードにするため)。欠損は -9999 (v6 と同じ)。

作るもの (事前に固定):
  R0 (再現確認): v6 と同じ lambdarank・同じハイパラ・seed 42・515 回・同じ標本重み。
      本番 pkl と予測が一致すれば「同じ特徴・同じデータ」が成り立っている。
  M1 (主): 目的関数 = binary log-loss (1 着か)。ハイパラは v6 から lambdarank 固有のものを除いてそのまま、
      標本重みなし、valid の binary_logloss で早期停止 (100 回、上限 5,000 回)。確率は配信時にレース内で正規化。
  M2 (副): 目的関数 = レース内 softmax の勝ち馬対数尤度 (条件付きロジット = Plackett-Luce の 1 着部分)。
      同じハイパラ、標本重みなし、valid の勝ち馬 NLL で早期停止。確率 = レース内 softmax(スコア)。
  温度 (オフライン補助用): v6 スコアの softmax(s/T) の T を valid 2023 の勝ち馬 NLL で fit。
      (本検証の温度は凍結データの前半で fit する。analysis.objective_check_eval を参照)

出力: <out>/models/{r0,m1,m2}.txt、<out>/train_summary.json
使い方: python -m analysis.objective_check_train --out reports/objective_check
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

BASE = Path("E:/PyCaLiAI")
sys.path.insert(0, str(BASE))
import optuna_v6_marks as ovm  # noqa: E402

RANK_ONLY = {"objective", "lambdarank_truncation_level", "metric", "eval_at", "num_iterations"}


def load_split():
    pkl = joblib.load(BASE / "models" / "unified_rank_v6.pkl")
    feats, encs = pkl["feature_cols"], pkl["encoders"]
    df = pd.read_csv(BASE / "data" / "master_v2_20130105-20251228.csv", encoding="utf-8-sig", low_memory=False)
    df[ovm.COL_JYUN] = pd.to_numeric(df[ovm.COL_JYUN], errors="coerce")
    df = df.dropna(subset=[ovm.COL_JYUN, ovm.COL_RID, "split"]).copy()
    df["label"] = np.clip(6 - df[ovm.COL_JYUN].astype(int), 0, 5).astype(int)
    df["win"] = (df[ovm.COL_JYUN].astype(int) == 1).astype(int)
    for c, le in encs.items():
        if c not in df.columns:
            continue
        v = df[c].astype(str).fillna("__NaN__")
        v = v.where(v.isin(set(le.classes_)), "__NaN__")
        df[c] = le.transform(v)
    # R0 は本番の学習と同じく master の並びのまま make_dataset に渡す (並べ替えはその中で行われる)
    raw_tr = df[df["split"] == "train"].copy()
    win_pay = ovm.load_winner_tansho_pay()
    raw_tr["rid_s"] = raw_tr[ovm.COL_RID].astype(str)
    raw_tr["winner_tansho"] = raw_tr["rid_s"].map(win_pay).fillna(100.0)
    parts = {s: df[df["split"] == s].sort_values(ovm.COL_RID, kind="stable").reset_index(drop=True)
             for s in ("train", "valid", "test")}
    return pkl, feats, parts, raw_tr


def X_of(d, feats):
    return d[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values


def groups_of(d):
    rid = d[ovm.COL_RID].to_numpy()
    _, start, counts = np.unique(rid, return_index=True, return_counts=True)
    order = np.argsort(start)
    return start[order], counts[order]


def race_softmax(s, starts, counts):
    """starts は昇順・連続 (レース ID で並べた行) を前提にレース内 softmax。"""
    s = np.asarray(s, float)
    mx = np.maximum.reduceat(s, starts)
    e = np.exp(s - np.repeat(mx, counts))
    return e / np.repeat(np.add.reduceat(e, starts), counts)


def race_normalize(p, starts, counts):
    p = np.asarray(p, float)
    return p / np.repeat(np.add.reduceat(p, starts), counts)


def winner_nll(p, y, starts, counts):
    """勝ち馬がいるレースの平均 −Σ y·log p / Σ y (同着は等分)。"""
    ysum = np.add.reduceat(y, starts)
    ll = np.add.reduceat(y * np.log(np.clip(p, 1e-12, None)), starts)
    ok = ysum > 0
    return float((-ll[ok] / ysum[ok]).mean())


def binary_ll(p, y):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def softmax_objective(starts, counts):
    def fobj(preds, data):
        y = data.get_label()
        p = race_softmax(preds, starts, counts)
        t = np.repeat(np.add.reduceat(y, starts), counts)
        has = (t > 0).astype(float)
        ys = np.where(t > 0, y / np.where(t > 0, t, 1.0), 0.0)
        grad = (p - ys) * has
        hess = np.maximum(p * (1 - p), 1e-6) * has
        return grad, hess
    return fobj


def softmax_feval(starts, counts):
    def feval(preds, data):
        return "winner_nll", winner_nll(race_softmax(preds, starts, counts), data.get_label(), starts, counts), False
    return feval


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    (out / "models").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pkl, feats, parts, raw_tr = load_split()
    tr, vl, te = parts["train"], parts["valid"], parts["test"]
    Xtr, Xvl, Xte = X_of(tr, feats), X_of(vl, feats), X_of(te, feats)
    gtr, gvl, gte = groups_of(tr), groups_of(vl), groups_of(te)
    ytr, yvl, yte = tr["win"].to_numpy(float), vl["win"].to_numpy(float), te["win"].to_numpy(float)
    print(f"rows train={len(tr):,} valid={len(vl):,} test={len(te):,}  feats={len(feats)}  t={time.time()-t0:.0f}s", flush=True)
    summary = {"rows": {"train": len(tr), "valid": len(vl), "test": len(te)},
               "races": {"train": len(gtr[0]), "valid": len(gvl[0]), "test": len(gte[0])}}

    base = dict(pkl["model"].params)
    # ---- R0: 再現確認 ----
    ds_r0, _, _ = ovm.make_dataset(raw_tr, feats, alpha=pkl["sample_weight_alpha"])
    del raw_tr
    p_r0 = {k: v for k, v in base.items() if k != "num_iterations"}
    r0 = lgb.train(p_r0, ds_r0, num_boost_round=pkl["num_boost_round"])
    r0.save_model(str(out / "models" / "r0.txt"))
    s_prod_vl, s_r0_vl = pkl["model"].predict(Xvl), r0.predict(Xvl)
    summary["r0_reproduction"] = {"max_abs_diff_valid": float(np.abs(s_prod_vl - s_r0_vl).max()),
                                  "corr_valid": float(np.corrcoef(s_prod_vl, s_r0_vl)[0, 1])}
    print("R0", summary["r0_reproduction"], f"t={time.time()-t0:.0f}s", flush=True)

    params = {k: v for k, v in base.items() if k not in RANK_ONLY}
    # ---- M1: binary log-loss ----
    p1 = dict(params, objective="binary", metric="binary_logloss")
    m1 = lgb.train(p1, lgb.Dataset(Xtr, label=ytr), num_boost_round=5000,
                   valid_sets=[lgb.Dataset(Xvl, label=yvl)],
                   callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)])
    m1.save_model(str(out / "models" / "m1.txt"))
    summary["m1"] = {"best_iteration": m1.best_iteration, "params": p1}
    print("M1 best_iter", m1.best_iteration, f"t={time.time()-t0:.0f}s", flush=True)

    # ---- M2: race softmax (条件付きロジット) ----
    p2 = dict(params, objective=softmax_objective(*gtr), metric="None")
    m2 = lgb.train(p2, lgb.Dataset(Xtr, label=ytr, init_score=np.zeros(len(ytr))), num_boost_round=5000,
                   valid_sets=[lgb.Dataset(Xvl, label=yvl, init_score=np.zeros(len(yvl)))],
                   feval=softmax_feval(*gvl),
                   callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)])
    m2.save_model(str(out / "models" / "m2.txt"))
    summary["m2"] = {"best_iteration": m2.best_iteration,
                     "params": {k: v for k, v in p2.items() if k != "objective"} | {"objective": "custom race softmax"}}
    print("M2 best_iter", m2.best_iteration, f"t={time.time()-t0:.0f}s", flush=True)

    # ---- オフライン補助評価 (valid 2023 は早期停止・温度 fit に使用、test 2024-25 が補助評価) ----
    cal = joblib.load(BASE / "models" / "pl_calibrators_v6.pkl")["calibrators"]["tansho"]
    s6_vl = s_prod_vl
    T_off = minimize_scalar(lambda T: winner_nll(race_softmax(s6_vl / T, *gvl), yvl, *gvl),
                            bounds=(0.2, 5.0), method="bounded").x
    summary["offline_T_valid2023"] = float(T_off)

    def probs(X, g):
        s6 = pkl["model"].predict(X)
        pl = race_softmax(s6, *g)
        return {"v6_pl": pl,
                "v6_pl_iso": race_normalize(np.clip(cal.predict(pl), 1e-6, None), *g),
                "v6_T": race_softmax(s6 / T_off, *g),
                "m1": race_normalize(m1.predict(X, num_iteration=m1.best_iteration), *g),
                "m2": race_softmax(m2.predict(X, num_iteration=m2.best_iteration), *g)}

    for name, X, g, y in (("valid2023", Xvl, gvl, yvl), ("test2024_25", Xte, gte, yte)):
        P = probs(X, g)
        summary[f"offline_{name}"] = {k: {"winner_nll": winner_nll(p, y, *g), "binary_logloss": binary_ll(p, y),
                                          "brier": float(((p - y) ** 2).mean())} for k, p in P.items()}
        print(name, json.dumps(summary[f"offline_{name}"]), flush=True)
        if name == "test2024_25":
            np.savez_compressed(out / "offline_test_probs.npz", y=y, starts=g[0], counts=g[1],
                                rid=te[ovm.COL_RID].astype(str).to_numpy(), **P)
    summary["elapsed_s"] = time.time() - t0
    (out / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
