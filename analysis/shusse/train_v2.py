# -*- coding: utf-8 -*-
"""
analysis/shusse/train_v2.py — 出世予測 AI v2: 全 k・3 ヘッド・評価・アブレーション
==================================================================================
k ∈ {1,2,3,5} の基準点ごとに LightGBM 3 ヘッドを学習する。
  D: P(離脱)              binary（全対象馬、y = 1 − ran_12m）
  A: P(L0..L4 | 走った)   multiclass（走った馬のみ）。期待レベルを valid で QWK 最大化した閾値で丸める
  B: P(重賞掲示板 24m)    binary（24 か月窓が閉じた馬）
統合: P(重賞掲示板 12m) = (1 − P(離脱)) · P(L4 | 走った)

ハイパーパラメータは valid（2020-21 デビュー）で leaves {15,31,63} × min_leaf {30,60,120}（lr 0.03、
早期停止）の logloss 最小を選び、test（2022 デビュー）を見る前に固定する。
2023 デビューで 12 か月窓が閉じた馬は test2023_partial として記述用に出す（選択バイアスあり、判定には使わない）。

実行（root から）:
    venv311\\Scripts\\python.exe -m analysis.shusse.train_v2

出力（analysis/shusse/out_v2/）:
    dataset_k{k}.parquet, pred_test_k{k}.parquet, model_spec_k{k}.json
    models/lgbm_{A|B|D}_k{k}.txt, asof_{trn|sire|brd}_{ev|deb}_k{k}.parquet
    metrics_v2.json, ablation_v2.json, v1_comparison.json, feature_labels.json
    feature_importance_{A|B|D}_k{k}.csv, calibration_{D|B|G12}_k{k}.csv, confusion_A_k{k}.csv
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, cohen_kappa_score, log_loss, roc_auc_score

from analysis.shusse.build_career import COL_HORSE
from analysis.shusse.features_v2 import (
    CAT_COLS, KS, N_LEVELS, SPLITS, TARGET_LEVEL_NAME, TE_M, build_asof_tables, build_features, build_targets,
    eligible_mask, feature_cols, feature_groups, feature_labels, prep_matrix, save_asof_tables, save_json, te_priors,
)

OUT = Path(__file__).resolve().parent / "out_v2"
MODELS = OUT / "models"
MODELS.mkdir(parents=True, exist_ok=True)
SEED = 20261031
N_BOOT = 300
GRID = {"num_leaves": [15, 31, 63], "min_data_in_leaf": [30, 60, 120]}
BASE_PARAMS = dict(learning_rate=0.03, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                   lambda_l2=1.0, verbose=-1, seed=SEED, num_threads=8)
LV = np.arange(N_LEVELS)
NAMED = ["ミッキーカプチーノ", "ガストリック", "ピピオラ", "ドゥレッツァ", "リバティアイランド", "ドゥラエレーデ",
         "ソールオリエンス", "タスティエーラ"]


# --------------------------------------------------------------------------- metrics
def qwk(y, p) -> float:
    return float(cohen_kappa_score(y, p, weights="quadratic"))


def auc(y, p) -> float:
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else float("nan")


def cum_auc(y_level, prob) -> dict:
    return {f"auc_ge_L{k}": auc((y_level >= k).astype(int), prob[:, k:].sum(axis=1)) for k in range(1, N_LEVELS)}


def cum_auc_score(y_level, score) -> dict:
    return {f"auc_ge_L{k}": auc((y_level >= k).astype(int), score) for k in range(1, N_LEVELS)}


def fit_cutpoints(expected, y) -> list[float]:
    cuts = [0.5, 1.5, 2.5, 3.5]
    grid = np.arange(0.05, 4.0, 0.05)
    best = qwk(y, np.digitize(expected, cuts))
    for _ in range(3):
        for i in range(4):
            for v in grid:
                lo = cuts[i - 1] if i > 0 else -np.inf
                hi = cuts[i + 1] if i < 3 else np.inf
                if not (lo < v < hi):
                    continue
                trial = cuts.copy()
                trial[i] = float(v)
                s = qwk(y, np.digitize(expected, trial))
                if s > best + 1e-9:
                    best, cuts = s, trial
    return cuts


def boot_ci(fn, *arrays, n: int = N_BOOT, seed: int = SEED) -> list[float]:
    rng = np.random.default_rng(seed)
    N = len(arrays[0])
    vals = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        try:
            v = fn(*[a[idx] for a in arrays])
        except ValueError:
            continue
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return [float("nan"), float("nan")]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def calib_table(y, p, bins: int = 10) -> pd.DataFrame:
    q = pd.qcut(p, bins, labels=False, duplicates="drop")
    t = pd.DataFrame({"y": y, "p": p, "bin": q}).groupby("bin").agg(n=("y", "size"), pred_mean=("p", "mean"), obs_rate=("y", "mean"))
    t["gap"] = t["obs_rate"] - t["pred_mean"]
    return t.reset_index()


def ece(y, p, bins: int = 10) -> float:
    t = calib_table(y, p, bins)
    return float((t["n"] / t["n"].sum() * t["gap"].abs()).sum())


def binary_metrics(y, p, ci: bool = False) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    r = {"n": int(len(y)), "base_rate": float(y.mean()), "auc": auc(y, p), "brier": float(brier_score_loss(y, p)),
         "logloss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])), "ece10": ece(y, p)}
    if ci:
        r["auc_ci95"] = boot_ci(auc, y, p)
    return r


def ordinal_metrics(y, P, cuts, ci: bool = False) -> dict:
    y = np.asarray(y).astype(int)
    pred_cut = np.digitize(P @ LV, cuts)
    r = {"n": int(len(y)), "qwk_argmax": qwk(y, P.argmax(axis=1)), "qwk_expected_cut": qwk(y, pred_cut),
         "multi_logloss": float(log_loss(y, P, labels=LV)), **cum_auc(y, P),
         "level_dist": {int(k): float(v) for k, v in pd.Series(y).value_counts(normalize=True).sort_index().items()}}
    if ci:
        r["qwk_expected_cut_ci95"] = boot_ci(qwk, y, pred_cut)
        r["auc_ge_L4_ci95"] = boot_ci(lambda a, b: auc((a >= 4).astype(int), b), y, P[:, 4])
        r["auc_ge_L1_ci95"] = boot_ci(lambda a, b: auc((a >= 1).astype(int), b), y, P[:, 1:].sum(axis=1))
    return r


def rule_ordinal_metrics(y, rule_level, rule_score, ci: bool = False) -> dict:
    y = np.asarray(y).astype(int)
    r = {"n": int(len(y)), "qwk": qwk(y, rule_level), **cum_auc_score(y, rule_score)}
    if ci:
        r["qwk_ci95"] = boot_ci(qwk, y, np.asarray(rule_level))
        r["auc_ge_L4_ci95"] = boot_ci(lambda a, b: auc((a >= 4).astype(int), b), y, np.asarray(rule_score, dtype=float))
    return r


def rule_table_prob(train: pd.DataFrame, target: str, m: float = TE_M):
    """規則ベースラインの確率: train の (rule_level, n_wins) 別経験率（縮約）。"""
    tab = train.groupby(["rule_level", "n_wins"])[target].agg(["mean", "size"]).reset_index()
    g = float(train[target].mean())
    tab["p_rule"] = (tab["mean"] * tab["size"] + m * g) / (tab["size"] + m)
    tab = tab[["rule_level", "n_wins", "p_rule"]]

    def apply(d: pd.DataFrame) -> np.ndarray:
        return d[["rule_level", "n_wins"]].merge(tab, on=["rule_level", "n_wins"], how="left")["p_rule"].fillna(g).values
    return apply


# --------------------------------------------------------------------------- training
def head_params(head: str) -> dict:
    if head == "A":
        return {"objective": "multiclass", "num_class": N_LEVELS}
    return {"objective": "binary"}


def fit(params: dict, Mtr, ytr, Mva, yva, cats: list[str]):
    dtr = lgb.Dataset(Mtr, ytr, categorical_feature=cats, free_raw_data=False)
    dva = lgb.Dataset(Mva, yva, categorical_feature=cats, reference=dtr, free_raw_data=False)
    m = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def select_params(head: str, Mtr, ytr, Mva, yva, cats: list[str]) -> tuple[dict, dict]:
    """valid logloss 最小のグリッド点。test は見ない。"""
    rows = []
    best = None
    for nl, ml in itertools.product(GRID["num_leaves"], GRID["min_data_in_leaf"]):
        p = {**BASE_PARAMS, **head_params(head), "num_leaves": nl, "min_data_in_leaf": ml}
        m = fit(p, Mtr, ytr, Mva, yva, cats)
        ll = float(m.best_score["valid_0"]["multi_logloss" if head == "A" else "binary_logloss"])
        rows.append({"num_leaves": nl, "min_data_in_leaf": ml, "valid_logloss": ll, "best_iter": int(m.best_iteration)})
        if best is None or ll < best[0]:
            best = (ll, p, int(m.best_iteration))
    return best[1], {"grid": rows, "chosen": {"num_leaves": best[1]["num_leaves"], "min_data_in_leaf": best[1]["min_data_in_leaf"],
                                              "valid_logloss": best[0], "best_iter": best[2]}}


def predict(m, M, head: str) -> np.ndarray:
    return m.predict(M, num_iteration=m.best_iteration)


def top_contrib(m, M, feats: list[str], head: str, cls: int | None = None, n: int = 3) -> list[str]:
    """各馬の寄与上位 n 特徴（符号付き）を JSON 文字列で返す。multiclass は cls のブロック。"""
    C = m.predict(M, num_iteration=m.best_iteration, pred_contrib=True)
    F = len(feats)
    if head == "A":
        C = C[:, cls * (F + 1):(cls + 1) * (F + 1)]
    C = C[:, :F]
    out = []
    for row in C:
        idx = np.argsort(-np.abs(row))[:n]
        out.append(json.dumps([[feats[i], round(float(row[i]), 4)] for i in idx], ensure_ascii=False))
    return out


def importance(m, feats: list[str]) -> pd.DataFrame:
    imp = pd.DataFrame({"feature": feats, "gain": m.feature_importance("gain"), "split": m.feature_importance("split")})
    imp["gain_share"] = (imp["gain"] / imp["gain"].sum()).round(4)
    lab = feature_labels(feats)
    imp["label_ja"] = imp["feature"].map(lab)
    return imp.sort_values("gain", ascending=False)


# --------------------------------------------------------------------------- dataset per k
def make_dataset(starts: pd.DataFrame, horses: pd.DataFrame, k: int) -> tuple[pd.DataFrame, dict, dict]:
    targets_all = build_targets(starts, k)
    elig = eligible_mask(horses, targets_all, k)
    tg = targets_all[targets_all[COL_HORSE].isin(elig[elig].index)].copy()
    h = horses.set_index(COL_HORSE)
    tg["debut_year"] = h.loc[tg[COL_HORSE], "debut_year"].values
    tg["split"] = "none"
    for name, (a, b) in SPLITS.items():
        tg.loc[tg["debut_year"].between(a, b), "split"] = name
    tg = tg[tg["split"] != "none"]
    priors = te_priors(tg, tg.loc[tg["split"] == "train", COL_HORSE])
    tables = build_asof_tables(horses, targets_all, k)
    save_asof_tables(tables, OUT, k)
    X = build_features(starts, horses, k, tables, priors, horse_ids=tg[COL_HORSE])
    X = X.merge(tg.drop(columns=["debut_year"]), on=[COL_HORSE, "base_date"], how="inner")
    assert len(X) == len(tg), (len(X), len(tg))
    return X, priors, tables


def run_k(starts: pd.DataFrame, horses: pd.DataFrame, k: int, metrics: dict) -> dict:
    t0 = time.time()
    X, priors, _ = make_dataset(starts, horses, k)
    X.to_parquet(OUT / f"dataset_k{k}.parquet", index=False)
    feats = feature_cols(X)
    cat_in = [c for c in CAT_COLS if c in feats]
    tr, va, te, t23 = (X[X["split"] == s].reset_index(drop=True) for s in ("train", "valid", "test", "test2023_partial"))
    _, cats = prep_matrix(tr, feats)
    mk = lambda d: prep_matrix(d, feats, cats)[0]
    Mtr, Mva, Mte, Mt23 = mk(tr), mk(va), mk(te), mk(t23)
    mk_tr = {s: d for s, d in zip(("train", "valid", "test", "test2023_partial"), (tr, va, te, t23))}
    res: dict = {"k": k, "n": {s: int(len(d)) for s, d in mk_tr.items()}, "n_features": len(feats),
                 "target_dist": {s: {"ran_12m": float(d["ran_12m"].mean()),
                                     "y_level_given_ran": {int(a): float(b) for a, b in d.loc[d["ran_12m"] == 1, "y_level"].value_counts(normalize=True).sort_index().items()},
                                     "y_level_v1": {int(a): float(b) for a, b in d["y_level_v1"].value_counts(normalize=True).sort_index().items()},
                                     "y_graded12": float(d["y_graded12"].mean()),
                                     "y_graded24": float(d.loc[d["h24_complete"], "y_graded24"].mean()) if d["h24_complete"].any() else None,
                                     "n_h24": int(d["h24_complete"].sum())} for s, d in mk_tr.items()}}
    spec: dict = {"k": k, "features": feats, "categorical": cats, "te_priors": priors, "te_m": TE_M, "splits": SPLITS,
                  "eligibility": "true_debut & debut_age in {2,3} & n_flat_starts >= k & base_date+365 <= 2025-12-28",
                  "target_level_name": TARGET_LEVEL_NAME, "heads": {}}
    models: dict = {}

    # ---- D: 離脱
    yD = "y_dropout"
    pD, gD = select_params("D", Mtr, tr[yD], Mva, va[yD], cat_in)
    mD = fit(pD, Mtr, tr[yD], Mva, va[yD], cat_in)
    mD.save_model(str(MODELS / f"lgbm_D_k{k}.txt"))
    models["D"] = mD
    spec["heads"]["D"] = {"target": yD, "params": pD, "best_iter": int(mD.best_iteration), "grid": gD}
    ruleD = rule_table_prob(tr, yD)
    res["D"] = {}
    for s, d, M in (("valid", va, Mva), ("test", te, Mte), ("test2023_partial", t23, Mt23)):
        p = predict(mD, M, "D")
        res["D"][f"lgbm_{s}"] = binary_metrics(d[yD], p, ci=(s == "test"))
        res["D"][f"rule_{s}"] = binary_metrics(d[yD], ruleD(d), ci=(s == "test"))
        if s == "test":
            calib_table(d[yD].values, p).to_csv(OUT / f"calibration_D_k{k}.csv", index=False, encoding="utf-8-sig")
    importance(mD, feats).to_csv(OUT / f"feature_importance_D_k{k}.csv", index=False, encoding="utf-8-sig")

    # ---- A: 到達クラス（走った馬のみ）
    trA, vaA = tr[tr["ran_12m"] == 1].reset_index(drop=True), va[va["ran_12m"] == 1].reset_index(drop=True)
    MtrA, MvaA = mk(trA), mk(vaA)
    yA = "y_level"
    pA, gA = select_params("A", MtrA, trA[yA].astype(int), MvaA, vaA[yA].astype(int), cat_in)
    mA = fit(pA, MtrA, trA[yA].astype(int), MvaA, vaA[yA].astype(int), cat_in)
    mA.save_model(str(MODELS / f"lgbm_A_k{k}.txt"))
    models["A"] = mA
    PvaA = predict(mA, MvaA, "A")
    cuts = fit_cutpoints(PvaA @ LV, vaA[yA].astype(int).values)
    spec["heads"]["A"] = {"target": yA, "population": "ran_12m == 1", "params": pA, "best_iter": int(mA.best_iteration), "cutpoints": cuts, "grid": gA}
    res["A"] = {"cutpoints": cuts}
    for s, d, M in (("valid", va, Mva), ("test", te, Mte), ("test2023_partial", t23, Mt23)):
        ran = (d["ran_12m"] == 1).values
        P = predict(mA, M, "A")
        y = d[yA].values[ran].astype(int)
        res["A"][f"lgbm_{s}"] = ordinal_metrics(y, P[ran], cuts, ci=(s == "test"))
        res["A"][f"rule_{s}"] = rule_ordinal_metrics(y, d["rule_level"].values[ran], d["rule_score"].values[ran], ci=(s == "test"))
        unwon = ran & (d["rule_level"].values == 0)
        yu = d[yA].values[unwon].astype(int)
        res["A"][f"lgbm_{s}_unwon"] = ordinal_metrics(yu, P[unwon], cuts, ci=(s == "test"))
        res["A"][f"rule_{s}_unwon"] = rule_ordinal_metrics(yu, d["rule_level"].values[unwon], d["rule_score"].values[unwon], ci=(s == "test"))
        if s == "test":
            pred_cut = np.digitize(P[ran] @ LV, cuts)
            cm = pd.crosstab(pd.Series(y, name="actual"), pd.Series(pred_cut, name="pred")).reindex(index=LV, columns=LV).fillna(0).astype(int)
            cm.to_csv(OUT / f"confusion_A_k{k}.csv", encoding="utf-8-sig")
            res["A"]["confusion_test_expected_cut"] = cm.values.tolist()
    importance(mA, feats).to_csv(OUT / f"feature_importance_A_k{k}.csv", index=False, encoding="utf-8-sig")

    # ---- B: 重賞掲示板 24m（窓が閉じた馬）
    yB = "y_graded24"
    trB, vaB = tr[tr["h24_complete"]].reset_index(drop=True), va[va["h24_complete"]].reset_index(drop=True)
    MtrB, MvaB = mk(trB), mk(vaB)
    pB, gB = select_params("B", MtrB, trB[yB].astype(int), MvaB, vaB[yB].astype(int), cat_in)
    mB = fit(pB, MtrB, trB[yB].astype(int), MvaB, vaB[yB].astype(int), cat_in)
    mB.save_model(str(MODELS / f"lgbm_B_k{k}.txt"))
    models["B"] = mB
    spec["heads"]["B"] = {"target": yB, "population": "h24_complete", "params": pB, "best_iter": int(mB.best_iteration), "grid": gB}
    ruleB = rule_table_prob(trB, yB)
    res["B"] = {}
    for s, d, M in (("valid", va, Mva), ("test", te, Mte), ("test2023_partial", t23, Mt23)):
        ok = d["h24_complete"].values
        if ok.sum() < 50:
            continue
        p = predict(mB, M, "B")[ok]
        y = d[yB].values[ok].astype(int)
        res["B"][f"lgbm_{s}"] = binary_metrics(y, p, ci=(s == "test"))
        res["B"][f"rule_{s}"] = binary_metrics(y, ruleB(d[ok]), ci=(s == "test"))
        if s == "test":
            calib_table(y, p).to_csv(OUT / f"calibration_B_k{k}.csv", index=False, encoding="utf-8-sig")
    importance(mB, feats).to_csv(OUT / f"feature_importance_B_k{k}.csv", index=False, encoding="utf-8-sig")

    # ---- 統合 P(重賞掲示板 12m) = (1 − P(離脱)) · P(L4 | 走った)、全対象馬で評価
    res["G12"] = {}
    ruleG = rule_table_prob(tr, "y_graded12")
    for s, d, M in (("valid", va, Mva), ("test", te, Mte), ("test2023_partial", t23, Mt23)):
        pd_ = predict(mD, M, "D")
        P = predict(mA, M, "A")
        pg = (1 - pd_) * P[:, 4]
        y = d["y_graded12"].values.astype(int)
        res["G12"][f"lgbm_{s}"] = binary_metrics(y, pg, ci=(s == "test"))
        res["G12"][f"lgbm_{s}_A_only_pL4"] = {"auc": auc(y, P[:, 4])}
        res["G12"][f"rule_{s}"] = binary_metrics(y, ruleG(d), ci=(s == "test"))
        res["G12"][f"rule_{s}"]["auc_rule_score"] = auc(y, d["rule_score"].values)
        if s == "test":
            calib_table(y, pg).to_csv(OUT / f"calibration_G12_k{k}.csv", index=False, encoding="utf-8-sig")

    # ---- test / test2023 予測の保存（寄与上位 3 特徴つき）
    preds = []
    for s, d, M in (("test", te, Mte), ("test2023_partial", t23, Mt23)):
        if len(d) == 0:
            continue
        P = predict(mA, M, "A")
        pd_ = predict(mD, M, "D")
        pb = predict(mB, M, "B")
        out = d[[COL_HORSE, "馬名", "base_date", "debut_year", "split", "ran_12m", "n_starts_12m", "y_level", "y_level_v1",
                 "y_graded12", "y_graded24", "h24_complete", "rule_level", "rule_score"]].copy()
        out["p_dropout"] = pd_
        for c in range(N_LEVELS):
            out[f"p_L{c}_given_ran"] = P[:, c]
        out["expected_level_given_ran"] = P @ LV
        out["pred_level_given_ran"] = np.digitize(P @ LV, cuts)
        out["p_graded12"] = (1 - pd_) * P[:, 4]
        out["p_graded24"] = pb
        out["top3_A_L4"] = top_contrib(mA, M, feats, "A", cls=4)
        out["top3_A_L0"] = top_contrib(mA, M, feats, "A", cls=0)
        out["top3_D"] = top_contrib(mD, M, feats, "D")
        out["top3_B"] = top_contrib(mB, M, feats, "B")
        preds.append(out)
    pd.concat(preds).to_parquet(OUT / f"pred_test_k{k}.parquet", index=False)

    res["elapsed_sec"] = round(time.time() - t0, 1)
    spec["elapsed_sec"] = res["elapsed_sec"]
    save_json(spec, OUT / f"model_spec_k{k}.json")
    metrics[f"k{k}"] = res
    print(f"[k={k}] n={res['n']} feats={len(feats)} "
          f"A test QWK={res['A']['lgbm_test']['qwk_expected_cut']:.3f} (rule {res['A']['rule_test']['qwk']:.3f}) "
          f"AUC L4={res['A']['lgbm_test']['auc_ge_L4']:.3f} | D AUC={res['D']['lgbm_test']['auc']:.3f} | "
          f"B AUC={res['B'].get('lgbm_test', {}).get('auc', float('nan')):.3f} | G12 AUC={res['G12']['lgbm_test']['auc']:.3f} "
          f"| {res['elapsed_sec']}s", flush=True)
    return {"X": X, "feats": feats, "cats": cats, "params": {"A": pA, "B": pB, "D": pD}, "cuts": cuts, "models": models}


# --------------------------------------------------------------------------- ablation
def ablation(ctx: dict, k: int) -> dict:
    X, feats, cats, params = ctx["X"], ctx["feats"], ctx["cats"], ctx["params"]
    groups = feature_groups(feats)
    configs = {"perf_only": groups["perf"],
               "perf+jockey": groups["perf"] + groups["jockey"],
               "perf+pedigree_stable": groups["perf"] + groups["pedigree_stable"],
               "perf+body": groups["perf"] + groups["body"],
               "all": feats}
    tr, va, te = (X[X["split"] == s].reset_index(drop=True) for s in ("train", "valid", "test"))
    out = {"k": k, "groups": {g: v for g, v in groups.items()}, "results": {}}
    for name, fs in configs.items():
        cat_in = [c for c in CAT_COLS if c in fs]
        cs = {c: cats[c] for c in cat_in}
        mk = lambda d: prep_matrix(d, fs, cs)[0]
        r = {"n_feats": len(fs)}
        # A
        trA, vaA = tr[tr["ran_12m"] == 1], va[va["ran_12m"] == 1]
        mA = fit(params["A"], mk(trA), trA["y_level"].astype(int), mk(vaA), vaA["y_level"].astype(int), cat_in)
        cuts = fit_cutpoints(predict(mA, mk(vaA), "A") @ LV, vaA["y_level"].astype(int).values)
        teA = te[te["ran_12m"] == 1]
        P = predict(mA, mk(teA), "A")
        y = teA["y_level"].astype(int).values
        r["A_qwk_test"] = qwk(y, np.digitize(P @ LV, cuts))
        r["A_auc_L4_test"] = auc((y >= 4).astype(int), P[:, 4])
        r["A_auc_L1_test"] = auc((y >= 1).astype(int), P[:, 1:].sum(axis=1))
        r["A_logloss_test"] = float(log_loss(y, P, labels=LV))
        # D
        mD = fit(params["D"], mk(tr), tr["y_dropout"], mk(va), va["y_dropout"], cat_in)
        r["D_auc_test"] = auc(te["y_dropout"].values, predict(mD, mk(te), "D"))
        # B
        trB, vaB, teB = tr[tr["h24_complete"]], va[va["h24_complete"]], te[te["h24_complete"]]
        mB = fit(params["B"], mk(trB), trB["y_graded24"].astype(int), mk(vaB), vaB["y_graded24"].astype(int), cat_in)
        r["B_auc_test"] = auc(teB["y_graded24"].astype(int).values, predict(mB, mk(teB), "B"))
        # G12
        pg = (1 - predict(mD, mk(te), "D")) * predict(mA, mk(te), "A")[:, 4]
        r["G12_auc_test"] = auc(te["y_graded12"].values, pg)
        out["results"][name] = r
        print(f"  [ablation k={k}] {name:22s} feats={len(fs):3d} QWK={r['A_qwk_test']:.3f} AUC_L4={r['A_auc_L4_test']:.3f} D={r['D_auc_test']:.3f} B={r['B_auc_test']:.3f} G12={r['G12_auc_test']:.3f}", flush=True)
    return out


# --------------------------------------------------------------------------- v1 比較（k=3）
def v1_comparison(ctx: dict, k: int = 3) -> dict:
    """同じ特徴・同じパラメータで y_level_v1（出走ベース L4・離脱=L0）を学習し、新定義と並べる。"""
    X, feats, cats, params, mA_new, cuts_new = ctx["X"], ctx["feats"], ctx["cats"], ctx["params"], ctx["models"]["A"], ctx["cuts"]
    cat_in = [c for c in CAT_COLS if c in feats]
    mk = lambda d: prep_matrix(d, feats, cats)[0]
    tr, va, te = (X[X["split"] == s].reset_index(drop=True) for s in ("train", "valid", "test"))
    out: dict = {"k": k, "rows": {}}
    # (1) v1 設定: 全馬・y_level_v1
    m1 = fit(params["A"], mk(tr), tr["y_level_v1"].astype(int), mk(va), va["y_level_v1"].astype(int), cat_in)
    c1 = fit_cutpoints(predict(m1, mk(va), "A") @ LV, va["y_level_v1"].astype(int).values)
    P1 = predict(m1, mk(te), "A")
    y1 = te["y_level_v1"].astype(int).values
    out["rows"]["v1_def_all_horses"] = {"desc": "v1 定義（出走ベース L4、離脱=L0）、全対象馬", **ordinal_metrics(y1, P1, c1, ci=True),
                                       "rule": rule_ordinal_metrics(y1, te["rule_level"].values, te["rule_score"].values)}
    ran = (te["ran_12m"] == 1).values
    out["rows"]["v1_def_model_eval_on_ran"] = {"desc": "同じ v1 モデルを走った馬だけで評価（y_level_v1）", **ordinal_metrics(y1[ran], P1[ran], c1, ci=True)}
    # (2) 離脱分離のみ: 走った馬・y_level_v1
    trR, vaR = tr[tr["ran_12m"] == 1], va[va["ran_12m"] == 1]
    m2 = fit(params["A"], mk(trR), trR["y_level_v1"].astype(int), mk(vaR), vaR["y_level_v1"].astype(int), cat_in)
    c2 = fit_cutpoints(predict(m2, mk(vaR), "A") @ LV, vaR["y_level_v1"].astype(int).values)
    P2 = predict(m2, mk(te[ran]), "A")
    out["rows"]["dropout_separated_v1_L4"] = {"desc": "離脱分離のみ（走った馬で学習・評価、L4 は出走ベース）", **ordinal_metrics(y1[ran], P2, c2, ci=True),
                                             "rule": rule_ordinal_metrics(y1[ran], te["rule_level"].values[ran], te["rule_score"].values[ran])}
    # (3) 新定義: 走った馬・y_level（掲示板 L4）
    Pn = predict(mA_new, mk(te), "A")
    yn = te["y_level"].values
    out["rows"]["new_def_dropout_separated_board_L4"] = {"desc": "新定義（離脱分離 + 重賞掲示板 L4）", **ordinal_metrics(yn[ran].astype(int), Pn[ran], cuts_new, ci=True),
                                                        "rule": rule_ordinal_metrics(yn[ran].astype(int), te["rule_level"].values[ran], te["rule_score"].values[ran])}
    # 定義変更による L4 の変化（test）
    lv1 = te["y_level_v1"].values
    out["L4_reclassification_test"] = {
        "v1_L4_count": int((lv1 == 4).sum()),
        "v1_L4_that_are_dropout": 0,
        "v1_L4_now_L4_board": int(((lv1 == 4) & ran & (yn == 4)).sum()),
        "v1_L4_now_L3_unplaced": int(((lv1 == 4) & ran & (yn == 3)).sum()),
        "v1_L0_that_are_dropout": int(((lv1 == 0) & ~ran).sum()),
        "v1_L0_count": int((lv1 == 0).sum()),
    }
    # 名指しの馬（v1 の弱点例）
    pD = predict(ctx["models"]["D"], mk(te), "D")
    rows = []
    for nm in NAMED:
        i = np.where(te["馬名"].values == nm)[0]
        if len(i) == 0:
            continue
        i = int(i[0])
        rows.append({"馬名": nm, "v1_model_pL4": round(float(P1[i, 4]), 3), "v1_model_pred_level": int(np.digitize(P1[i] @ LV, c1)),
                     "new_p_dropout": round(float(pD[i]), 3), "new_pL4_given_ran": round(float(Pn[i, 4]), 3),
                     "new_pL3_given_ran": round(float(Pn[i, 3]), 3), "new_p_graded12": round(float((1 - pD[i]) * Pn[i, 4]), 3),
                     "new_pred_level_given_ran": int(np.digitize(Pn[i] @ LV, cuts_new)),
                     "actual_ran_12m": int(te["ran_12m"].values[i]), "actual_y_level_v1": int(lv1[i]),
                     "actual_y_level_new": (None if not ran[i] else int(yn[i]))})
    out["named_horses"] = rows
    return out


# --------------------------------------------------------------------------- main
def main() -> None:
    T0 = time.time()
    starts = pd.read_parquet(OUT / "starts_v2.parquet")
    horses = pd.read_parquet(OUT / "horses_v2.parquet")
    metrics: dict = {"seed": SEED, "grid": GRID, "base_params": BASE_PARAMS, "n_boot": N_BOOT}
    ctxs = {}
    all_feats: set[str] = set()
    for k in KS:
        ctxs[k] = run_k(starts, horses, k, metrics)
        all_feats |= set(ctxs[k]["feats"])
    save_json(feature_labels(sorted(all_feats)), OUT / "feature_labels.json")
    save_json(metrics, OUT / "metrics_v2.json")
    abl = {f"k{k}": ablation(ctxs[k], k) for k in (1, 3)}
    save_json(abl, OUT / "ablation_v2.json")
    cmp3 = v1_comparison(ctxs[3], 3)
    save_json(cmp3, OUT / "v1_comparison.json")
    metrics["elapsed_total_sec"] = round(time.time() - T0, 1)
    save_json(metrics, OUT / "metrics_v2.json")
    print(json.dumps(cmp3["rows"], ensure_ascii=False, indent=1, default=float)[:3000])
    print(pd.DataFrame(cmp3["named_horses"]).to_string())
    print(f"total {metrics['elapsed_total_sec']}s")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
