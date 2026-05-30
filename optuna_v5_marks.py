"""
optuna_v5_marks.py
==================
v5 = 印精度を最大化する LightGBM LambdaRank。

cowork 連携を前提とした設計:
  - PyCaLiAI 側は「印 (◎〇▲△△) と PL 確率」を高品質で出す
  - 馬券構築は cowork 側に任せる
  - 学習目的は ROI ではなく **印の精度 + 確率の信頼性**

v4 との差分:
  1. ラベル: 11-着順 → clip(6-着順, 0, 5)  (top-5 にフォーカス)
  2. lambdarank_truncation_level = 5 固定 (v4 は trial で振っていた)
  3. eval_at = [1, 3, 5]
  4. sample_weight: 1 + alpha * log1p(tansho_pay / 100)
     (穴好走を重く → 市場コピー脱却)
  5. Optuna objective = composite (印精度の加重合成)

objective = (
    0.30 * NDCG@5
  + 0.25 * ◎_top3_rate          # ◎ が複勝圏に入る率
  + 0.20 * top3_subset_top5_rate # {1,2,3着} が top-5 に揃う率
  + 0.15 * winner_in_top5_rate   # 1着馬が top-5 に入る率
  + 0.10 * ◎_top2_rate           # ◎ 連対率
)

5-fold CV で valid 内 race を fit/eval 分割。
fit fold は学習に使わず指標測定のみ (LGBM 学習は train≤2022 固定)。
v4 と同じ feats / encoders / split を使う。

再現性: seed=42, deterministic=True, force_col_wise=True。

実行: python optuna_v5_marks.py [--n-trials 30]

出力:
  models/unified_rank_v5.pkl
  reports/optuna_v5_marks.json
"""
from __future__ import annotations
import argparse
import io
import json
import sys
import warnings
from itertools import groupby
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

import pl_probs as PL

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE       = Path(__file__).parent
MASTER_CSV = BASE / "data/master_v2_20130105-20251228.csv"
KEKKA_CSV  = BASE / "data/kekka_20130105-20251228.csv"
OUT_MODEL  = BASE / "models/unified_rank_v5.pkl"
OUT_JSON   = BASE / "reports/optuna_v5_marks.json"

SEED       = 42
N_TRIALS_DEFAULT = 30
N_FOLDS    = 5

np.random.seed(SEED)

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

# composite weights (合計 = 1.0)
W_NDCG5            = 0.30
W_HON_TOP3         = 0.25
W_TOP3_SUBSET_TOP5 = 0.20
W_WINNER_IN_TOP5   = 0.15
W_HON_TOP2         = 0.10


# ============================================================
# データ準備
# ============================================================
def load_winner_tansho_pay():
    """rid_s -> 1着馬の単勝配当 (sample_weight 用)."""
    df = pd.read_csv(KEKKA_CSV, encoding="cp932", low_memory=False)
    df.columns = ["rid_horse", "ban", "ped", "jyun",
                  "tansho", "fukusho", "wakuren", "umaren",
                  "umatan", "sanrenpuku", "sanrentan"]
    df["rid_s"] = df["rid_horse"].astype(str).str[:16]
    df["jyun"] = pd.to_numeric(df["jyun"], errors="coerce")
    df["tansho"] = pd.to_numeric(df["tansho"], errors="coerce")
    win = df[df["jyun"] == 1].drop_duplicates("rid_s")
    return dict(zip(win["rid_s"].values, win["tansho"].values))


def prep():
    print(f"[load] {MASTER_CSV}")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]).copy()

    # v5 ラベル: clip(6 - 着順, 0, 5)
    # 1着 -> 5, 2着 -> 4, 3着 -> 3, 4着 -> 2, 5着 -> 1, 6位以下 -> 0
    df["label"] = np.clip(6 - df[COL_JYUN].astype(int), 0, 5).astype(int)

    tr = df[df["split"] == "train"].copy()
    vl = df[df["split"] == "valid"].copy()
    print(f"  train={len(tr):,}  valid={len(vl):,}")

    # encoders (train で fit)
    encs = {}
    for c in CAT_COLS:
        if c not in tr.columns: continue
        le = LabelEncoder()
        vals = tr[c].astype(str).fillna("__NaN__")
        le.fit(pd.concat([vals, pd.Series(["__NaN__"])], ignore_index=True))
        encs[c] = le

    def _apply(d):
        d = d.copy()
        for c, le in encs.items():
            if c not in d.columns: continue
            v = d[c].astype(str).fillna("__NaN__")
            known = set(le.classes_)
            v = v.where(v.isin(known), "__NaN__")
            d[c] = le.transform(v)
        return d

    tr = _apply(tr); vl = _apply(vl)
    feats = [c for c in tr.columns if c not in LEAK_COLS and c != "label"]

    # sample_weight (train のみ)
    win_pay = load_winner_tansho_pay()
    tr["rid_s"] = tr[COL_RID].astype(str).str[:16] if tr[COL_RID].dtype == object \
                  else tr[COL_RID].astype(str)
    tr["winner_tansho"] = tr["rid_s"].map(win_pay).fillna(100.0)

    return tr, vl, feats, encs


def make_dataset(d, feats, alpha=0.0):
    """alpha=0 → weight 全て 1. alpha>0 で穴好走を重く."""
    d = d.sort_values(COL_RID).reset_index(drop=True)
    X = d[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    y = d["label"].values.astype(int)
    g = np.array([len(list(gr)) for _, gr in groupby(d[COL_RID])])
    if alpha > 0 and "winner_tansho" in d.columns:
        w = (1.0 + alpha * np.log1p(d["winner_tansho"].values / 100.0)).astype(float)
    else:
        w = np.ones(len(d), dtype=float)
    return lgb.Dataset(X, label=y, group=g, weight=w, free_raw_data=False), X, d


# ============================================================
# 評価 (印精度 composite)
# ============================================================
def ndcg_at_k(label_arr, score_arr, k=5):
    n = len(label_arr)
    if n < k: k = n
    order = np.argsort(-score_arr)[:k]
    gains = (2.0 ** label_arr[order] - 1)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (gains * discounts).sum()
    iorder = np.argsort(-label_arr)[:k]
    igains = (2.0 ** label_arr[iorder] - 1)
    idcg = (igains * discounts).sum()
    return float(dcg / idcg) if idcg > 0 else 0.0


def evaluate_marks(vl_scored):
    """vl_scored から印精度 composite を計算."""
    n_race = 0
    ndcg5_sum = 0.0
    hon_top3 = 0
    hon_top2 = 0
    top3_subset_top5 = 0
    winner_in_top5 = 0

    for rid, g in vl_scored.groupby(COL_RID, sort=False):
        if len(g) < 5: continue
        scores = g["_score"].values
        jyun = g[COL_JYUN].astype(int).values
        n_race += 1
        order = np.argsort(-scores)
        hon = int(order[0])
        top3_pred = set(int(x) for x in order[:3])
        top5_pred = set(int(x) for x in order[:5])

        # NDCG@5
        rel = np.maximum(0, 5 - (jyun - 1))
        rel = np.clip(rel, 0, 5).astype(float)
        ndcg5_sum += ndcg_at_k(rel, scores, k=5)

        # 実際の 1, 2, 3 着 index (g 内 index)
        sorted_idx_by_jyun = np.argsort(jyun)
        wi = int(sorted_idx_by_jyun[0])
        pi_ = int(sorted_idx_by_jyun[1])
        si = int(sorted_idx_by_jyun[2])
        actual_top3 = {wi, pi_, si}

        if hon in actual_top3:
            hon_top3 += 1
        if hon in {wi, pi_}:
            hon_top2 += 1
        if actual_top3.issubset(top5_pred):
            top3_subset_top5 += 1
        if wi in top5_pred:
            winner_in_top5 += 1

    if n_race == 0:
        return None
    return {
        "n_race": n_race,
        "ndcg5": ndcg5_sum / n_race,
        "hon_top3_rate": hon_top3 / n_race,
        "hon_top2_rate": hon_top2 / n_race,
        "top3_subset_top5_rate": top3_subset_top5 / n_race,
        "winner_in_top5_rate": winner_in_top5 / n_race,
    }


def composite_score(metrics):
    return (
        W_NDCG5            * metrics["ndcg5"]
      + W_HON_TOP3         * metrics["hon_top3_rate"]
      + W_TOP3_SUBSET_TOP5 * metrics["top3_subset_top5_rate"]
      + W_WINNER_IN_TOP5   * metrics["winner_in_top5_rate"]
      + W_HON_TOP2         * metrics["hon_top2_rate"]
    )


def compute_cv_composite(model, vl_df, X_vl):
    """valid を 5-fold split. 各 fold で印指標を計算 → 平均."""
    # ここは fit/eval 分割不要 (LGBM 学習は train で完了済)。
    # ただし安定性確認のため race 単位の bootstrap-ish CV をやる。
    vl_scored = vl_df.copy()
    vl_scored["_score"] = model.predict(X_vl)

    unique_rids = vl_scored[COL_RID].drop_duplicates().values
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    scores = []
    metrics_all = None
    for _, eval_idx in kf.split(unique_rids):
        rids_eval = unique_rids[eval_idx]
        sub = vl_scored[vl_scored[COL_RID].isin(rids_eval)]
        m = evaluate_marks(sub)
        if m is None: continue
        scores.append(composite_score(m))
        if metrics_all is None:
            metrics_all = {k: [v] for k, v in m.items()}
        else:
            for k, v in m.items():
                metrics_all[k].append(v)
    composite_mean = float(np.mean(scores))
    metrics_mean = {k: float(np.mean(v)) for k, v in metrics_all.items()}
    return composite_mean, metrics_mean


# ============================================================
# Optuna
# ============================================================
_CACHE = {}


def objective(trial):
    alpha = trial.suggest_float("alpha", 0.0, 2.0)
    params = {
        "objective":                  "lambdarank",
        "lambdarank_truncation_level": 5,  # 固定 (top-5 印精度に集中)
        "metric":                     "ndcg",
        "eval_at":                    [5],
        "learning_rate":              trial.suggest_float("lr", 0.01, 0.1, log=True),
        "num_leaves":                 trial.suggest_int("num_leaves", 31, 255, log=True),
        "max_depth":                  trial.suggest_int("max_depth", -1, 12),
        "min_data_in_leaf":           trial.suggest_int("min_data_in_leaf", 20, 200),
        "feature_fraction":           trial.suggest_float("ff", 0.6, 1.0),
        "bagging_fraction":           trial.suggest_float("bf", 0.6, 1.0),
        "bagging_freq":               5,
        "lambda_l1":                  trial.suggest_float("l1", 1e-3, 10, log=True),
        "lambda_l2":                  trial.suggest_float("l2", 1e-3, 10, log=True),
        "verbose": -1, "n_jobs": -1, "seed": SEED,
        "deterministic": True, "force_col_wise": True, "feature_pre_filter": False,
    }

    # alpha が変わったら dataset を作り直す
    ds_tr, _, _ = make_dataset(_CACHE["tr"], _CACHE["feats"], alpha=alpha)
    ds_vl = _CACHE["ds_vl"]

    model = lgb.train(params, ds_tr, num_boost_round=2000,
                      valid_sets=[ds_vl],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    best_iter = model.best_iteration
    trial.set_user_attr("best_iter", best_iter)

    composite, metrics = compute_cv_composite(model, _CACHE["vl_df"], _CACHE["X_vl"])
    trial.set_user_attr("ndcg5", metrics["ndcg5"])
    trial.set_user_attr("hon_top3_rate", metrics["hon_top3_rate"])
    trial.set_user_attr("hon_top2_rate", metrics["hon_top2_rate"])
    trial.set_user_attr("top3_subset_top5_rate", metrics["top3_subset_top5_rate"])
    trial.set_user_attr("winner_in_top5_rate", metrics["winner_in_top5_rate"])

    print(f"  trial {trial.number:>2}: composite={composite*100:6.3f}  "
          f"ndcg5={metrics['ndcg5']:.4f}  ◎top3={metrics['hon_top3_rate']*100:5.2f}%  "
          f"top3⊂top5={metrics['top3_subset_top5_rate']*100:5.2f}%  "
          f"win∈top5={metrics['winner_in_top5_rate']*100:5.2f}%  "
          f"alpha={alpha:.2f}  iter={best_iter}")
    sys.stdout.flush()
    return composite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT)
    args = ap.parse_args()

    print("=" * 80)
    print(f"optuna_v5_marks.py  (n_trials={args.n_trials}, n_folds={N_FOLDS}, seed={SEED})")
    print(f"  objective = composite mark accuracy")
    print(f"    NDCG@5 ({W_NDCG5}) + ◎top3 ({W_HON_TOP3}) + top3⊂top5 ({W_TOP3_SUBSET_TOP5})")
    print(f"    + win∈top5 ({W_WINNER_IN_TOP5}) + ◎top2 ({W_HON_TOP2})")
    print(f"  ラベル: clip(6 - 着順, 0, 5)  (top-5 focus)")
    print(f"  sample_weight: 1 + alpha * log1p(tansho_pay/100)")
    print("=" * 80)

    tr, vl, feats, encs = prep()

    ds_vl, X_vl, vl_df = make_dataset(vl, feats, alpha=0.0)  # valid は weight=1
    print(f"  feats: {len(feats)}")

    _CACHE["tr"] = tr
    _CACHE["vl_df"] = vl_df
    _CACHE["X_vl"] = X_vl
    _CACHE["ds_vl"] = ds_vl
    _CACHE["feats"] = feats

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    print(f"\n[best] composite = {study.best_value*100:.3f}")
    bp = study.best_params
    print(f"  best_iter = {study.best_trial.user_attrs['best_iter']}")
    print(f"  alpha     = {bp['alpha']:.3f}")
    for k in ["ndcg5", "hon_top3_rate", "hon_top2_rate",
              "top3_subset_top5_rate", "winner_in_top5_rate"]:
        v = study.best_trial.user_attrs[k]
        if "rate" in k:
            print(f"  {k:24s} = {v*100:.2f}%")
        else:
            print(f"  {k:24s} = {v:.4f}")

    # ========= retrain on train (best params, alpha 適用) =========
    print(f"\n[retrain] best params on train (alpha={bp['alpha']:.3f})")
    params = {
        "objective": "lambdarank",
        "lambdarank_truncation_level": 5,
        "metric": "ndcg", "eval_at": [1, 3, 5],
        "learning_rate": bp["lr"],
        "num_leaves": bp["num_leaves"],
        "max_depth": bp["max_depth"],
        "min_data_in_leaf": bp["min_data_in_leaf"],
        "feature_fraction": bp["ff"],
        "bagging_fraction": bp["bf"],
        "bagging_freq": 5,
        "lambda_l1": bp["l1"], "lambda_l2": bp["l2"],
        "verbose": -1, "n_jobs": -1, "seed": SEED,
        "deterministic": True, "force_col_wise": True, "feature_pre_filter": False,
    }
    best_iter = study.best_trial.user_attrs["best_iter"]
    ds_tr_final, _, _ = make_dataset(tr, feats, alpha=bp["alpha"])
    model = lgb.train(params, ds_tr_final, num_boost_round=int(best_iter * 1.1),
                      valid_sets=[ds_vl],
                      callbacks=[lgb.log_evaluation(period=200)])

    OUT_MODEL.parent.mkdir(exist_ok=True)
    joblib.dump({
        "model": model, "feature_cols": feats, "encoders": encs,
        "cat_cols": CAT_COLS, "seed": SEED,
        "master_csv": MASTER_CSV.name,
        "optuna_best_params": bp,
        "optuna_best_composite": study.best_value,
        "n_folds": N_FOLDS,
        "label_scheme": "clip(6 - 着順, 0, 5)",
        "sample_weight_alpha": bp["alpha"],
        "description": "unified_rank_v5: LambdaRank with top-5 focus label & "
                       "payout-weighted sampling. Optuna obj = composite mark "
                       "accuracy (NDCG@5 + ◎top3 + top3⊂top5 + win∈top5 + ◎top2). "
                       f"30 trials, 5-fold CV, seed=42.",
    }, OUT_MODEL)
    print(f"[saved] {OUT_MODEL}")

    OUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "n_trials": args.n_trials, "n_folds": N_FOLDS, "seed": SEED,
            "best_composite": study.best_value,
            "best_params": bp, "best_iter": best_iter,
            "best_metrics": {k: study.best_trial.user_attrs[k] for k in
                             ["ndcg5", "hon_top3_rate", "hon_top2_rate",
                              "top3_subset_top5_rate", "winner_in_top5_rate"]},
            "label_scheme": "clip(6 - 着順, 0, 5)",
            "composite_weights": {
                "ndcg5": W_NDCG5, "hon_top3": W_HON_TOP3,
                "top3_subset_top5": W_TOP3_SUBSET_TOP5,
                "winner_in_top5": W_WINNER_IN_TOP5,
                "hon_top2": W_HON_TOP2,
            },
            "all_trials": [
                {"number": t.number, "value": t.value, "params": t.params,
                 "best_iter": t.user_attrs.get("best_iter"),
                 "ndcg5": t.user_attrs.get("ndcg5"),
                 "hon_top3_rate": t.user_attrs.get("hon_top3_rate"),
                 "top3_subset_top5_rate": t.user_attrs.get("top3_subset_top5_rate"),
                 "winner_in_top5_rate": t.user_attrs.get("winner_in_top5_rate"),
                 "hon_top2_rate": t.user_attrs.get("hon_top2_rate")}
                for t in study.trials
            ],
        }, f, indent=2, ensure_ascii=False)
    print(f"[saved] {OUT_JSON}")


if __name__ == "__main__":
    main()
