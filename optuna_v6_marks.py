"""
optuna_v6_marks.py
==================
v6 = v5 の calibration を改善するために sample_weight 偏重を緩めた版。

v5 との主な差分:
  1. **sample_weight alpha 範囲**: [0.0, 2.0] → [0.0, 1.5]
     v5 audit で「alpha=1.325 だと穴馬 raw_score が systematically 嵩上げされ
     calibration が tail で崩壊」が判明。alpha 上限を下げて、Optuna が低 alpha
     を選びやすくする。
  2. **Optuna 目的関数**: composite mark accuracy (v5 と同じ) - 0.5 * ECE_high_p
     ECE_high_p = predicted p_win が高い領域 (top-3 per race) での
     |予測平均 - 実勢平均|。calibration awareness を加える。
  3. ranking 性能 (NDCG@5 等) と calibration 性能 (ECE) のトレードオフを
     Optuna に解かせる。alpha が自然に下がる想定。

v5 の audit 結果 (参考):
  - 単勝高 EV (≥1.2) ROI: 0.734 (controllable な負け以下)
  - 複勝高 EV (≥1.2) ROI: 0.909
  - 高 EV bin で p_obs << p_est (model 過信)
  → v6 では高 p_pred 帯の calibration を penalty で抑制

実行: python optuna_v6_marks.py [--n-trials 40]

出力:
  models/unified_rank_v6.pkl
  reports/optuna_v6_marks.json
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
OUT_MODEL  = BASE / "models/unified_rank_v6.pkl"
OUT_JSON   = BASE / "reports/optuna_v6_marks.json"

SEED       = 42
N_TRIALS_DEFAULT = 40   # v5 は 30, v6 は alpha + ECE トレードオフ探索のため少し増やす
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

# v6 新規: ECE penalty 重み
W_ECE_PENALTY      = 0.5  # composite から差し引く ECE の係数


# ============================================================
# データ準備 (v5 と同じ)
# ============================================================
def load_winner_tansho_pay():
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
    df["label"] = np.clip(6 - df[COL_JYUN].astype(int), 0, 5).astype(int)

    tr = df[df["split"] == "train"].copy()
    vl = df[df["split"] == "valid"].copy()
    print(f"  train={len(tr):,}  valid={len(vl):,}")

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

    win_pay = load_winner_tansho_pay()
    tr["rid_s"] = tr[COL_RID].astype(str).str[:16] if tr[COL_RID].dtype == object \
                  else tr[COL_RID].astype(str)
    tr["winner_tansho"] = tr["rid_s"].map(win_pay).fillna(100.0)

    return tr, vl, feats, encs


def make_dataset(d, feats, alpha=0.0):
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
# 評価 (印精度 + ECE)
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


def evaluate_marks_with_ece(vl_scored, p_threshold=0.10):
    """v5 の評価指標 + ECE (高 p_pred 帯) を返す。

    ECE = | mean(p_win[mask]) - mean(actual[mask]) |
      where mask = p_win >= p_threshold (上位 ~3 頭 per race 相当)

    モデルが「強い」と思った馬の予測確率と実勢確率の絶対誤差。
    過信 (p_pred > p_actual) も過小 (p_pred < p_actual) もペナルティ。
    """
    n_race = 0
    ndcg5_sum = 0.0
    hon_top3 = 0
    hon_top2 = 0
    top3_subset_top5 = 0
    winner_in_top5 = 0

    p_high_all = []
    actual_high_all = []

    for rid, g in vl_scored.groupby(COL_RID, sort=False):
        if len(g) < 5: continue
        scores = g["_score"].values
        jyun = g[COL_JYUN].astype(int).values
        n_race += 1

        # PL 確率変換
        w = PL.pl_weights(scores)
        p_win = PL.all_tansho(w)

        order = np.argsort(-scores)
        hon = int(order[0])
        top3_pred = set(int(x) for x in order[:3])
        top5_pred = set(int(x) for x in order[:5])

        rel = np.maximum(0, 5 - (jyun - 1))
        rel = np.clip(rel, 0, 5).astype(float)
        ndcg5_sum += ndcg_at_k(rel, scores, k=5)

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

        # ECE (高 p_pred 帯) サンプル収集
        win_mask = (jyun == 1).astype(int)
        high_p_mask = p_win >= p_threshold
        if high_p_mask.sum() > 0:
            p_high_all.extend(p_win[high_p_mask].tolist())
            actual_high_all.extend(win_mask[high_p_mask].tolist())

    if n_race == 0:
        return None

    # ECE 計算
    p_arr = np.array(p_high_all)
    a_arr = np.array(actual_high_all)
    if len(p_arr) > 100:
        ece_high = abs(float(p_arr.mean()) - float(a_arr.mean()))
    else:
        ece_high = 0.0   # サンプル少なすぎる場合 penalty 無し

    return {
        "n_race": n_race,
        "ndcg5": ndcg5_sum / n_race,
        "hon_top3_rate": hon_top3 / n_race,
        "hon_top2_rate": hon_top2 / n_race,
        "top3_subset_top5_rate": top3_subset_top5 / n_race,
        "winner_in_top5_rate": winner_in_top5 / n_race,
        "ece_high_p": ece_high,
        "n_high_p_samples": len(p_arr),
        "p_pred_high_mean": float(p_arr.mean()) if len(p_arr) else 0.0,
        "actual_high_mean": float(a_arr.mean()) if len(a_arr) else 0.0,
    }


def composite_score(metrics):
    base = (
        W_NDCG5            * metrics["ndcg5"]
      + W_HON_TOP3         * metrics["hon_top3_rate"]
      + W_TOP3_SUBSET_TOP5 * metrics["top3_subset_top5_rate"]
      + W_WINNER_IN_TOP5   * metrics["winner_in_top5_rate"]
      + W_HON_TOP2         * metrics["hon_top2_rate"]
    )
    # v6: ECE penalty を引く
    return base - W_ECE_PENALTY * metrics["ece_high_p"]


def compute_cv_composite(model, vl_df, X_vl):
    vl_scored = vl_df.copy()
    vl_scored["_score"] = model.predict(X_vl)

    unique_rids = vl_scored[COL_RID].drop_duplicates().values
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    scores = []
    metrics_all = None
    for _, eval_idx in kf.split(unique_rids):
        rids_eval = unique_rids[eval_idx]
        sub = vl_scored[vl_scored[COL_RID].isin(rids_eval)]
        m = evaluate_marks_with_ece(sub)
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
    # v6: alpha 上限を 1.5 に下げる (v5 は 2.0 だった)
    alpha = trial.suggest_float("alpha", 0.0, 1.5)
    params = {
        "objective":                  "lambdarank",
        "lambdarank_truncation_level": 5,
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

    ds_tr, _, _ = make_dataset(_CACHE["tr"], _CACHE["feats"], alpha=alpha)
    ds_vl = _CACHE["ds_vl"]

    model = lgb.train(params, ds_tr, num_boost_round=2000,
                      valid_sets=[ds_vl],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    best_iter = model.best_iteration
    trial.set_user_attr("best_iter", best_iter)

    composite, metrics = compute_cv_composite(model, _CACHE["vl_df"], _CACHE["X_vl"])
    for k in ["ndcg5", "hon_top3_rate", "hon_top2_rate",
              "top3_subset_top5_rate", "winner_in_top5_rate",
              "ece_high_p", "p_pred_high_mean", "actual_high_mean"]:
        trial.set_user_attr(k, metrics[k])

    print(f"  trial {trial.number:>2}: composite={composite*100:6.3f}  "
          f"ndcg5={metrics['ndcg5']:.4f}  ◎top3={metrics['hon_top3_rate']*100:5.2f}%  "
          f"ECE={metrics['ece_high_p']:.4f}  "
          f"alpha={alpha:.2f}  iter={best_iter}")
    sys.stdout.flush()
    return composite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT)
    args = ap.parse_args()

    print("=" * 80)
    print(f"optuna_v6_marks.py  (n_trials={args.n_trials}, n_folds={N_FOLDS}, seed={SEED})")
    print(f"  objective = composite_v5 - {W_ECE_PENALTY} × ECE_high_p")
    print(f"    composite_v5 = NDCG@5 ({W_NDCG5}) + ◎top3 ({W_HON_TOP3}) + "
          f"top3⊂top5 ({W_TOP3_SUBSET_TOP5})")
    print(f"                + win∈top5 ({W_WINNER_IN_TOP5}) + ◎top2 ({W_HON_TOP2})")
    print(f"    ECE_high_p = |mean(p_pred) - mean(actual)| where p_pred >= 0.10")
    print(f"  alpha 範囲: [0.0, 1.5]  (v5 は [0.0, 2.0])")
    print(f"  ラベル: clip(6 - 着順, 0, 5)")
    print("=" * 80)

    tr, vl, feats, encs = prep()
    ds_vl, X_vl, vl_df = make_dataset(vl, feats, alpha=0.0)
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
              "top3_subset_top5_rate", "winner_in_top5_rate", "ece_high_p"]:
        v = study.best_trial.user_attrs[k]
        if "rate" in k:
            print(f"  {k:24s} = {v*100:.2f}%")
        else:
            print(f"  {k:24s} = {v:.4f}")

    # ========= retrain on train (best params) =========
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
        "ece_penalty_weight": W_ECE_PENALTY,
        "description": "unified_rank_v6: v5 + ECE penalty in Optuna objective. "
                       "alpha range narrowed to [0.0, 1.5] to discourage extreme "
                       "payout weighting. Goal: better calibration at high p_pred "
                       "tail, sacrificing NDCG slightly. "
                       f"{args.n_trials} trials, 5-fold CV, seed=42.",
    }, OUT_MODEL)
    print(f"[saved] {OUT_MODEL}")

    OUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "n_trials": args.n_trials, "n_folds": N_FOLDS, "seed": SEED,
            "best_composite": study.best_value,
            "best_params": bp, "best_iter": best_iter,
            "ece_penalty_weight": W_ECE_PENALTY,
            "best_metrics": {k: study.best_trial.user_attrs[k] for k in
                             ["ndcg5", "hon_top3_rate", "hon_top2_rate",
                              "top3_subset_top5_rate", "winner_in_top5_rate",
                              "ece_high_p", "p_pred_high_mean", "actual_high_mean"]},
            "label_scheme": "clip(6 - 着順, 0, 5)",
            "composite_weights": {
                "ndcg5": W_NDCG5, "hon_top3": W_HON_TOP3,
                "top3_subset_top5": W_TOP3_SUBSET_TOP5,
                "winner_in_top5": W_WINNER_IN_TOP5,
                "hon_top2": W_HON_TOP2,
                "ece_penalty": -W_ECE_PENALTY,
            },
            "all_trials": [
                {"number": t.number, "value": t.value, "params": t.params,
                 "best_iter": t.user_attrs.get("best_iter"),
                 "ndcg5": t.user_attrs.get("ndcg5"),
                 "hon_top3_rate": t.user_attrs.get("hon_top3_rate"),
                 "ece_high_p": t.user_attrs.get("ece_high_p")}
                for t in study.trials
            ],
        }, f, indent=2, ensure_ascii=False)
    print(f"[saved] {OUT_JSON}")


if __name__ == "__main__":
    main()
