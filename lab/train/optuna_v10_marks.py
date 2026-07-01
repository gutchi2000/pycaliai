# -*- coding: utf-8 -*-
"""
optuna_v10_marks.py
===================
v10 = 本番 v6 の学習プロセス監査指摘 (docs/audit_20260611.md) を反映した再学習。
(v7=ワイドROI目的 / v8=affinity leak / v9 は別系統の失敗実験。本スクリプトは v6 直系)

v6 との差分 (全て監査指摘対応):
  1. **split スライド** (鮮度): train = 〜2023 / valid = 2024 / test = 2025 封印。
     v6 は train〜2022 で3年以上凍結、LabelEncoder の未知カテゴリ率も増加していた。
     test=2025 は学習・HP選択・early stopping のどこにも使わず、採用判定の
     一発比較 (audit_marks) のためだけに温存する。
  2. **タイム文字列列の修理**: 「前走走破タイム」等が 'M.SS.T' 文字列のまま
     to_numeric → 100%NaN で無言死亡していた。utils.parse_time_str を
     パターン自動検出で適用。
  3. **死亡特徴の検出と除外**: coerce 後 100%NaN の特徴は除外して警告
     (母馬などの object 高カーディナリティ列が -9999 定数で残る問題)。
  4. **ECE_high_p の多ビン化**: v6 は高p帯単一ビンの符号付きバイアスで
     過信と過小が相殺し得た。p∈[0.10,0.15,0.20,0.30,1.0] の4ビン加重 |gap| 平均に。
  5. **CV の正直化**: v6 の「5-fold CV」は同一モデルのレース分割評価で
     fold 間 std を見ていなかった。objective = mean − 1.0×std の保守選択にし、
     ノイズ水準の差での argmax を抑制。fold std も記録。

維持 (意図的に変えない):
  - fillna(-9999): LGBM native NaN の方が筋は良いが、推論パス
    (export_marks_json/backtest_pl_ev/serve calibrator) 全部が -9999 規約で
    動いており、連動変更のリスクが大きい。native NaN は別実験へ。
  - ラベル / composite 重み / lambdarank 設定: v6 と同一 (比較可能性のため)。

実行: PYTHONUTF8=1 python optuna_v10_marks.py [--n-trials 40]
出力: models/unified_rank_v10.pkl / reports/optuna_v10_marks.json
採用判定: audit_marks --model v10 を test=2025 のみで v6 と比較 (1回だけ)。
採用時の連動再構築: pl_calibrators (valid=2024 fit) / serve cal / class_prior /
                    payout curve / chaos_quantiles。
"""
from __future__ import annotations
import argparse
import io
import json
import re
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
from utils import parse_time_str

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE       = Path(__file__).parent
MASTER_CSV = BASE / "data/master_v2_20130105-20251228.csv"
KEKKA_CSV  = BASE / "data/kekka_20130105-20251228.csv"
OUT_MODEL  = BASE / "models/unified_rank_v10.pkl"
OUT_JSON   = BASE / "reports/optuna_v10_marks.json"

SEED       = 42
N_TRIALS_DEFAULT = 40
N_FOLDS    = 5
CV_STD_PENALTY = 1.0     # objective = mean − この係数 × fold std

TRAIN_MAX_YEAR = 2023    # train: 〜2023
VALID_YEAR     = 2024    # valid: 2024 (early stopping + HP選択 + 後続 calibrator fit)
# test = 2025 は封印 (このスクリプトでは一切評価しない)

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

W_NDCG5            = 0.30
W_HON_TOP3         = 0.25
W_TOP3_SUBSET_TOP5 = 0.20
W_WINNER_IN_TOP5   = 0.15
W_HON_TOP2         = 0.10
W_ECE_PENALTY      = 0.5

_TIME_PAT = re.compile(r"^\s*\d+\.\d{2}\.\d\s*$")


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


def fix_time_string_cols(df: pd.DataFrame) -> list[str]:
    """'M.SS.T' 形式の object 列を秒 float に変換 (in-place)。変換列名を返す。
    監査指摘: 前走走破タイム等が文字列のまま coerce → 100%NaN で死亡していた。
    """
    fixed = []
    for c in df.columns:
        if c in LEAK_COLS or c in CAT_COLS or df[c].dtype != object:
            continue
        sample = df[c].dropna().astype(str)
        if len(sample) == 0:
            continue
        sample = sample.head(200)
        if (sample.str.match(_TIME_PAT).mean()) > 0.8:
            df[c] = parse_time_str(df[c])
            fixed.append(c)
    return fixed


def prep():
    print(f"[load] {MASTER_CSV}")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID]).copy()
    df["label"] = np.clip(6 - df[COL_JYUN].astype(int), 0, 5).astype(int)

    # --- v10: year ベース split (master の split 列は使わない) ---
    df["_year"] = df[COL_RID].astype(str).str[:4].astype(int)
    tr = df[df["_year"] <= TRAIN_MAX_YEAR].copy()
    vl = df[df["_year"] == VALID_YEAR].copy()
    n_test_sealed = int((df["_year"] >= VALID_YEAR + 1).sum())
    print(f"  train(〜{TRAIN_MAX_YEAR})={len(tr):,}  valid({VALID_YEAR})={len(vl):,}  "
          f"test({VALID_YEAR+1}〜)={n_test_sealed:,} ← 封印 (採用判定まで未使用)")

    # --- v10: タイム文字列列の修理 ---
    fixed_tr = fix_time_string_cols(tr)
    fix_time_string_cols(vl)
    if fixed_tr:
        print(f"  [time fix] {len(fixed_tr)} 列を parse_time_str: {fixed_tr}")

    encs = {}
    for c in CAT_COLS:
        if c not in tr.columns:
            continue
        le = LabelEncoder()
        vals = tr[c].astype(str).fillna("__NaN__")
        le.fit(pd.concat([vals, pd.Series(["__NaN__"])], ignore_index=True))
        encs[c] = le

    def _apply(d):
        d = d.copy()
        for c, le in encs.items():
            if c not in d.columns:
                continue
            v = d[c].astype(str).fillna("__NaN__")
            known = set(le.classes_)
            v = v.where(v.isin(known), "__NaN__")
            d[c] = le.transform(v)
        return d

    tr = _apply(tr)
    vl = _apply(vl)
    feats = [c for c in tr.columns
             if c not in LEAK_COLS and c not in ("label", "_year")]

    # --- v10: 死亡特徴 (coerce 後 100%NaN) の検出と除外 ---
    probe = tr[feats].apply(pd.to_numeric, errors="coerce")
    dead = [c for c in feats if probe[c].isna().all()]
    if dead:
        print(f"  [dead feats] coerce後100%NaN の {len(dead)} 列を除外: {dead}")
        feats = [c for c in feats if c not in dead]

    win_pay = load_winner_tansho_pay()
    tr["rid_s"] = tr[COL_RID].astype(str).str[:16] if tr[COL_RID].dtype == object \
                  else tr[COL_RID].astype(str)
    tr["winner_tansho"] = tr["rid_s"].map(win_pay).fillna(100.0)

    return tr, vl, feats, encs, fixed_tr, dead


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


def ndcg_at_k(label_arr, score_arr, k=5):
    n = len(label_arr)
    if n < k:
        k = n
    order = np.argsort(-score_arr)[:k]
    gains = (2.0 ** label_arr[order] - 1)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (gains * discounts).sum()
    iorder = np.argsort(-label_arr)[:k]
    igains = (2.0 ** label_arr[iorder] - 1)
    idcg = (igains * discounts).sum()
    return float(dcg / idcg) if idcg > 0 else 0.0


# v10: 高 p 帯を 4 ビンに分割 (v6 は単一ビンで過信と過小が相殺し得た)
ECE_BIN_EDGES = [0.10, 0.15, 0.20, 0.30, 1.01]


def evaluate_marks_with_ece(vl_scored):
    n_race = 0
    ndcg5_sum = 0.0
    hon_top3 = hon_top2 = top3_subset_top5 = winner_in_top5 = 0
    p_high_all, actual_high_all = [], []

    for rid, g in vl_scored.groupby(COL_RID, sort=False):
        if len(g) < 5:
            continue
        scores = g["_score"].values
        jyun = g[COL_JYUN].astype(int).values
        n_race += 1
        w = PL.pl_weights(scores)
        p_win = PL.all_tansho(w)
        order = np.argsort(-scores)
        hon = int(order[0])
        top5_pred = set(int(x) for x in order[:5])

        rel = np.clip(np.maximum(0, 5 - (jyun - 1)), 0, 5).astype(float)
        ndcg5_sum += ndcg_at_k(rel, scores, k=5)

        sj = np.argsort(jyun)
        wi, pi_, si = int(sj[0]), int(sj[1]), int(sj[2])
        actual_top3 = {wi, pi_, si}
        hon_top3 += int(hon in actual_top3)
        hon_top2 += int(hon in {wi, pi_})
        top3_subset_top5 += int(actual_top3.issubset(top5_pred))
        winner_in_top5 += int(wi in top5_pred)

        win_mask = (jyun == 1).astype(int)
        hm = p_win >= ECE_BIN_EDGES[0]
        if hm.sum() > 0:
            p_high_all.extend(p_win[hm].tolist())
            actual_high_all.extend(win_mask[hm].tolist())

    if n_race == 0:
        return None

    # v10: 多ビン加重 |gap| 平均
    p_arr = np.array(p_high_all)
    a_arr = np.array(actual_high_all)
    ece_high = 0.0
    if len(p_arr) > 100:
        total = 0
        acc = 0.0
        for lo, hi in zip(ECE_BIN_EDGES[:-1], ECE_BIN_EDGES[1:]):
            m = (p_arr >= lo) & (p_arr < hi)
            nb = int(m.sum())
            if nb < 20:
                continue
            acc += nb * abs(float(p_arr[m].mean()) - float(a_arr[m].mean()))
            total += nb
        ece_high = acc / total if total else 0.0

    return {
        "n_race": n_race,
        "ndcg5": ndcg5_sum / n_race,
        "hon_top3_rate": hon_top3 / n_race,
        "hon_top2_rate": hon_top2 / n_race,
        "top3_subset_top5_rate": top3_subset_top5 / n_race,
        "winner_in_top5_rate": winner_in_top5 / n_race,
        "ece_high_p": ece_high,
        "n_high_p_samples": len(p_arr),
    }


def composite_score(m):
    base = (W_NDCG5 * m["ndcg5"]
            + W_HON_TOP3 * m["hon_top3_rate"]
            + W_TOP3_SUBSET_TOP5 * m["top3_subset_top5_rate"]
            + W_WINNER_IN_TOP5 * m["winner_in_top5_rate"]
            + W_HON_TOP2 * m["hon_top2_rate"])
    return base - W_ECE_PENALTY * m["ece_high_p"]


def compute_cv_composite(model, vl_df, X_vl):
    """fold ごとの composite から (mean, std, 平均指標) を返す。"""
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
        if m is None:
            continue
        scores.append(composite_score(m))
        if metrics_all is None:
            metrics_all = {k: [v] for k, v in m.items()}
        else:
            for k, v in m.items():
                metrics_all[k].append(v)
    mean = float(np.mean(scores))
    std = float(np.std(scores))
    metrics_mean = {k: float(np.mean(v)) for k, v in metrics_all.items()}
    return mean, std, metrics_mean


_CACHE = {}


def objective(trial):
    alpha = trial.suggest_float("alpha", 0.0, 1.5)
    params = {
        "objective": "lambdarank",
        "lambdarank_truncation_level": 5,
        "metric": "ndcg", "eval_at": [5],
        "learning_rate": trial.suggest_float("lr", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255, log=True),
        "max_depth": trial.suggest_int("max_depth", -1, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),
        "feature_fraction": trial.suggest_float("ff", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bf", 0.6, 1.0),
        "bagging_freq": 5,
        "lambda_l1": trial.suggest_float("l1", 1e-3, 10, log=True),
        "lambda_l2": trial.suggest_float("l2", 1e-3, 10, log=True),
        "verbose": -1, "n_jobs": -1, "seed": SEED,
        "deterministic": True, "force_col_wise": True, "feature_pre_filter": False,
    }
    ds_tr, _, _ = make_dataset(_CACHE["tr"], _CACHE["feats"], alpha=alpha)
    model = lgb.train(params, ds_tr, num_boost_round=2000,
                      valid_sets=[_CACHE["ds_vl"]],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    trial.set_user_attr("best_iter", model.best_iteration)

    mean, std, metrics = compute_cv_composite(model, _CACHE["vl_df"], _CACHE["X_vl"])
    # v10: 保守選択 = mean − 1.0×std (fold ノイズ水準の argmax を抑制)
    value = mean - CV_STD_PENALTY * std
    trial.set_user_attr("composite_mean", mean)
    trial.set_user_attr("composite_std", std)
    for k in ["ndcg5", "hon_top3_rate", "hon_top2_rate",
              "top3_subset_top5_rate", "winner_in_top5_rate", "ece_high_p"]:
        trial.set_user_attr(k, metrics[k])
    print(f"  trial {trial.number:>2}: mean-std={value*100:6.3f} "
          f"(mean={mean*100:6.3f} std={std*100:.3f})  "
          f"◎top3={metrics['hon_top3_rate']*100:5.2f}%  "
          f"ECE4bin={metrics['ece_high_p']:.4f}  alpha={alpha:.2f}  "
          f"iter={model.best_iteration}")
    sys.stdout.flush()
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT)
    args = ap.parse_args()

    print("=" * 80)
    print(f"optuna_v10_marks.py  (n_trials={args.n_trials}, seed={SEED})")
    print(f"  split: train〜{TRAIN_MAX_YEAR} / valid={VALID_YEAR} / "
          f"test={VALID_YEAR+1} 封印")
    print(f"  objective = (composite − {W_ECE_PENALTY}×ECE_4bin) の "
          f"fold mean − {CV_STD_PENALTY}×std")
    print("=" * 80)

    tr, vl, feats, encs, fixed_cols, dead_cols = prep()
    ds_vl, X_vl, vl_df = make_dataset(vl, feats, alpha=0.0)
    print(f"  feats: {len(feats)}")

    _CACHE.update(tr=tr, vl_df=vl_df, X_vl=X_vl, ds_vl=ds_vl, feats=feats)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    bt = study.best_trial
    bp = study.best_params
    print(f"\n[best] mean-std = {study.best_value*100:.3f}  "
          f"(mean={bt.user_attrs['composite_mean']*100:.3f} "
          f"std={bt.user_attrs['composite_std']*100:.3f})")
    print(f"  best_iter={bt.user_attrs['best_iter']}  alpha={bp['alpha']:.3f}")

    # ========= retrain (v6 と同じ best_iter×1.1。trial との乖離を保存時に記録) =========
    print(f"\n[retrain] best params on train (alpha={bp['alpha']:.3f})")
    params = {
        "objective": "lambdarank", "lambdarank_truncation_level": 5,
        "metric": "ndcg", "eval_at": [1, 3, 5],
        "learning_rate": bp["lr"], "num_leaves": bp["num_leaves"],
        "max_depth": bp["max_depth"], "min_data_in_leaf": bp["min_data_in_leaf"],
        "feature_fraction": bp["ff"], "bagging_fraction": bp["bf"],
        "bagging_freq": 5, "lambda_l1": bp["l1"], "lambda_l2": bp["l2"],
        "verbose": -1, "n_jobs": -1, "seed": SEED,
        "deterministic": True, "force_col_wise": True, "feature_pre_filter": False,
    }
    best_iter = bt.user_attrs["best_iter"]
    ds_tr_final, _, _ = make_dataset(tr, feats, alpha=bp["alpha"])
    model = lgb.train(params, ds_tr_final, num_boost_round=int(best_iter * 1.1),
                      valid_sets=[ds_vl],
                      callbacks=[lgb.log_evaluation(period=200)])
    # 保存直前に valid 指標を再計測して trial 値との乖離を記録 (監査指摘 14)
    mean_f, std_f, metrics_f = compute_cv_composite(model, vl_df, X_vl)
    print(f"[retrain check] composite mean={mean_f*100:.3f} "
          f"(trial時 {bt.user_attrs['composite_mean']*100:.3f}) "
          f"◎top3={metrics_f['hon_top3_rate']*100:.2f}%")

    OUT_MODEL.parent.mkdir(exist_ok=True)
    joblib.dump({
        "model": model, "feature_cols": feats, "encoders": encs,
        "cat_cols": CAT_COLS, "seed": SEED,
        "master_csv": MASTER_CSV.name,
        "optuna_best_params": bp,
        "optuna_best_value_mean_minus_std": study.best_value,
        "n_folds": N_FOLDS,
        "split": {"train_max_year": TRAIN_MAX_YEAR, "valid_year": VALID_YEAR,
                  "test_sealed": VALID_YEAR + 1},
        "time_fixed_cols": fixed_cols,
        "dead_cols_excluded": dead_cols,
        "label_scheme": "clip(6 - 着順, 0, 5)",
        "sample_weight_alpha": bp["alpha"],
        "ece_penalty_weight": W_ECE_PENALTY,
        "retrain_valid_metrics": metrics_f,
        "description": "unified_rank_v10: v6 direct successor with audit fixes "
                       f"(split slide train<={TRAIN_MAX_YEAR}/valid={VALID_YEAR}, "
                       "time-string repair, dead-feature exclusion, 4-bin ECE "
                       "penalty, mean-std conservative HP selection). test=2025 "
                       "sealed for one-shot adoption comparison vs v6.",
    }, OUT_MODEL)
    print(f"[saved] {OUT_MODEL}")

    OUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "n_trials": args.n_trials, "seed": SEED,
            "split": {"train_max_year": TRAIN_MAX_YEAR, "valid_year": VALID_YEAR},
            "best_value_mean_minus_std": study.best_value,
            "best_params": bp, "best_iter": best_iter,
            "best_composite_mean": bt.user_attrs["composite_mean"],
            "best_composite_std": bt.user_attrs["composite_std"],
            "time_fixed_cols": fixed_cols,
            "dead_cols_excluded": dead_cols,
            "retrain_valid_metrics": metrics_f,
            "best_metrics": {k: bt.user_attrs[k] for k in
                             ["ndcg5", "hon_top3_rate", "hon_top2_rate",
                              "top3_subset_top5_rate", "winner_in_top5_rate",
                              "ece_high_p"]},
            "all_trials": [
                {"number": t.number, "value": t.value, "params": t.params,
                 "composite_mean": t.user_attrs.get("composite_mean"),
                 "composite_std": t.user_attrs.get("composite_std"),
                 "hon_top3_rate": t.user_attrs.get("hon_top3_rate"),
                 "ece_high_p": t.user_attrs.get("ece_high_p")}
                for t in study.trials
            ],
        }, f, indent=2, ensure_ascii=False)
    print(f"[saved] {OUT_JSON}")


if __name__ == "__main__":
    main()
