# -*- coding: utf-8 -*-
"""
train_r0.py — R0-clean (正式基準) と R0-prodref (参考) の学習
=============================================================
base_train 2016-2021 で学習、2022 NDCG@5 で early stopping、格子点は 2022 win logloss (τ 最尤) で選択。
2023 は学習・ES・較正・選択に使わない (予測を保存するだけ。評価は evaluate.py)。
出力: data/_research/mcond/exp15/scores/{model}_s{seed}.npz, out/r0_train.json
"""
from __future__ import annotations

import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import common as C

SCORES = C.CACHE / "scores"


def winner_tansho(df):
    k = pd.read_csv(C.KEKKA, encoding="cp932", low_memory=False)
    k.columns = ["rid_horse", "ban", "ped", "jyun", "tansho", "fukusho", "wakuren", "umaren",
                 "umatan", "sanrenpuku", "sanrentan"]
    k["rid_s"] = k["rid_horse"].astype(str).str[:16]
    k["jyun"] = pd.to_numeric(k["jyun"], errors="coerce")
    k["tansho"] = pd.to_numeric(k["tansho"], errors="coerce")
    w = k[k["jyun"] == 1].drop_duplicates("rid_s")
    return df["rid16"].map(dict(zip(w["rid_s"], w["tansho"]))).fillna(100.0).to_numpy()


def groups(rid):
    cut = np.flatnonzero(rid[1:] != rid[:-1]) + 1
    return np.diff(np.concatenate([[0], cut, [len(rid)]]))


def fit_one(X, df, cfg, seed, weight=None):
    fx = C.SPEC["models"]["R0-clean"]["fixed"]
    tr = (df["period"] == "train").to_numpy()
    sl = (df["period"] == "sel").to_numpy()
    rid = df["rid16"].to_numpy()
    y = df["rel"].to_numpy()
    dtr = lgb.Dataset(X[tr], label=y[tr], group=groups(rid[tr]),
                      weight=None if weight is None else weight[tr], free_raw_data=False)
    dsl = lgb.Dataset(X[sl], label=y[sl], group=groups(rid[sl]), reference=dtr)
    params = {
        "objective": "lambdarank", "lambdarank_truncation_level": fx["lambdarank_truncation_level"],
        "metric": "ndcg", "eval_at": fx["eval_at"], "bagging_freq": fx["bagging_freq"],
        "learning_rate": cfg["learning_rate"], "num_leaves": cfg["num_leaves"],
        "max_depth": cfg["max_depth"], "min_data_in_leaf": cfg["min_data_in_leaf"],
        "feature_fraction": cfg["feature_fraction"], "bagging_fraction": cfg["bagging_fraction"],
        "lambda_l1": cfg["lambda_l1"], "lambda_l2": cfg["lambda_l2"],
        "verbose": -1, "n_jobs": -1, "seed": int(seed), "deterministic": fx["deterministic"],
        "force_col_wise": fx["force_col_wise"], "feature_pre_filter": fx["feature_pre_filter"],
    }
    t0 = time.time()
    m = lgb.train(params, dtr, num_boost_round=fx["max_rounds"], valid_sets=[dsl],
                  callbacks=[lgb.early_stopping(fx["early_stopping_rounds"], verbose=False)])
    ev = (df["period"].isin(["sel", "dev"])).to_numpy()
    s = np.full(len(df), np.nan)
    s[ev] = m.predict(X[ev], num_iteration=m.best_iteration)
    races_sel = C.race_index(df, sl)
    tau = C.fit_tau(s, df["win"].to_numpy(), races_sel)
    p = C.softmax_race(s, races_sel, tau)
    rm = C.race_metrics(df, races_sel, s, p)
    info = {"best_iter": int(m.best_iteration), "tau": tau, "sel_ll": float(rm["ll"].mean()),
            "sel_brier": float(rm["brier"].mean()), "seconds": time.time() - t0,
            "n_leaves_total": int(sum(t["num_leaves"] for t in m.dump_model()["tree_info"][:m.best_iteration]))}
    return s, info


def save(name, seed, df, s, info):
    SCORES.mkdir(parents=True, exist_ok=True)
    ev = (df["period"].isin(["sel", "dev"])).to_numpy()
    np.savez(SCORES / f"{name}_s{seed}.npz", rows=np.where(ev)[0], score=s[ev],
             key=(df["rid16"] + "_" + df["ban"].astype(str)).to_numpy()[ev],
             info=json.dumps(info))


def main():
    df = C.load_rows()
    con = C.load_contract()
    enc = C.Encoded(df, con, "clean")
    X = enc.X_r0
    spec = C.SPEC["models"]["R0-clean"]
    seeds = C.SPEC["seeds"]
    log = {"grid": {}, "seeds": {}, "prodref": {}}
    for cfg in spec["grid"]:
        _, info = fit_one(X, df, cfg, seeds["selection_seed"])
        log["grid"][cfg["name"]] = info
        print(f"[grid] {cfg['name']:18s} sel_ll={info['sel_ll']:.5f} iter={info['best_iter']} "
              f"tau={info['tau']:.3f} ({info['seconds']:.0f}s)", flush=True)
    best = min(spec["grid"], key=lambda c: log["grid"][c["name"]]["sel_ll"])
    log["selected"] = best["name"]
    print(f"[select] {best['name']}", flush=True)
    for sd in seeds["main"]:
        s, info = fit_one(X, df, best, sd)
        save("R0-clean", sd, df, s, info)
        log["seeds"][str(sd)] = info
        print(f"[R0-clean s{sd}] sel_ll={info['sel_ll']:.5f} iter={info['best_iter']}", flush=True)
    # R0-prodref (参考): v6 の 120 列 + v6 race 重み
    encp = C.Encoded(df, con, "prodref")
    alpha = 0.03083978412534253
    w = 1.0 + alpha * np.log1p(winner_tansho(df) / 100.0)
    for sd in seeds["main"]:
        s, info = fit_one(encp.X_r0, df, best, sd, weight=w)
        save("R0-prodref", sd, df, s, info)
        log["prodref"][str(sd)] = info
        print(f"[R0-prodref s{sd}] sel_ll={info['sel_ll']:.5f} iter={info['best_iter']}", flush=True)
    log["n_features"] = {"clean": len(enc.cols), "prodref": len(encp.cols)}
    C.dump(log, "r0_train.json")


if __name__ == "__main__":
    main()
