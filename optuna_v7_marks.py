"""
optuna_v7_marks.py
==================
v7 = v6 + ワイド ROI を目的関数に組込 + race-relative 新特徴量 (race_relative_feats v2)。

v6 との差分:
  1. **特徴量レイヤー**: race_relative_feats.add_race_relative_feats(mode='v2')
     を train/valid 両方に適用。22 新規列 (z-score, rank, peer_max/min,
     kako5_agari3f_volatility, kako5_pos_volatility, peer_count) を append。
  2. **Optuna 目的関数**:
        composite_v7 = composite_v6
                     + W_WIDE_BONUS * (wide_roi_3box - 1.0)   # ◎〇▲ box 3 点流し ROI
                     - W_WIDE_ECE * ECE_wide                  # box 3 点 ペア帯 calibration
     ROI 1.0 (= 控除率分カバー) を基準点とする加点設計。
     ROI > 1.0 で正の加点、ROI < 1.0 で減点。
  3. **目的**: Cowork 実績の弱点であるワイド (97.8%) を黒字側に押し上げる。
                同時にワイドペアの calibration も改善し、Cowork 判断材料を強化。

実行: python optuna_v7_marks.py [--n-trials 40]

出力:
  models/unified_rank_v7.pkl
  reports/optuna_v7_marks.json
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
from race_relative_feats import add_race_relative_feats, added_columns

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE       = Path(__file__).parent
MASTER_CSV = BASE / "data/master_v2_20130105-20251228.csv"
KEKKA_CSV  = BASE / "data/kekka_20130105-20251228.csv"
WIDE_PARQ  = BASE / "data/wide_payouts_2016-2025.parquet"
OUT_MODEL  = BASE / "models/unified_rank_v7.pkl"
OUT_JSON   = BASE / "reports/optuna_v7_marks.json"

SEED       = 42
N_TRIALS_DEFAULT = 40
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

# 印精度 composite weights
W_NDCG5            = 0.30
W_HON_TOP3         = 0.25
W_TOP3_SUBSET_TOP5 = 0.20
W_WINNER_IN_TOP5   = 0.15
W_HON_TOP2         = 0.10

# v6: high p_pred 帯 ECE penalty
W_ECE_PENALTY      = 0.5

# v7 新規: ワイド ROI ボーナスと ワイド box ECE penalty
W_WIDE_BONUS       = 0.30   # (wide_roi_3box - 1.0) に掛ける
W_WIDE_ECE         = 0.30   # box 3 点ペアの ECE に掛ける


# ============================================================
# データ準備
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


def load_wide_payouts():
    """rid_s -> [(ban_i, ban_j, pay), x3] (馬番 1-indexed, pay=100円当たり円)"""
    p = pd.read_parquet(WIDE_PARQ)
    out = {}
    for r in p.itertuples(index=False):
        rid = str(r.race_id)[:16]
        out[rid] = [
            (int(r.w1_i), int(r.w1_j), int(r.w1_pay)),
            (int(r.w2_i), int(r.w2_j), int(r.w2_pay)),
            (int(r.w3_i), int(r.w3_j), int(r.w3_pay)),
        ]
    return out


def prep():
    print(f"[load] {MASTER_CSV}")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]).copy()
    df["label"] = np.clip(6 - df[COL_JYUN].astype(int), 0, 5).astype(int)

    # ★v7 新特徴量レイヤー
    print(f"[v7] race_relative_feats (mode=v2)")
    df = add_race_relative_feats(df, mode="v2")
    added = added_columns(mode="v2")
    print(f"     +{len(added)} 列: {added[:6]}... ({len(added)} total)")

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

    # ban 列を vl に保持 (ROI 計算で使う)
    vl["_ban"] = pd.to_numeric(vl[COL_BAN], errors="coerce").astype("Int64")

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
# 評価: 印精度 + ECE_high_p + Wide ROI/ECE
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


def evaluate_marks_with_wide(vl_scored, wide_pays, p_threshold=0.10):
    """印精度 + ECE_high_p (v6) + Wide ROI (3-box) + ECE_wide。

    Wide ROI:
        ◎〇▲ box 3 点 (◎-〇, ◎-▲, 〇-▲) 各 100 円購入と仮定。
        wide_pays[rid_s] の 3 当たりペアと一致した点の払戻を合計。
        ROI = sum(payout_hit) / (300 * n_race)
    """
    n_race = 0
    ndcg5_sum = 0.0
    hon_top3 = 0
    hon_top2 = 0
    top3_subset_top5 = 0
    winner_in_top5 = 0

    p_high_all = []
    actual_high_all = []

    wide_stake = 0
    wide_return = 0
    wide_hits  = 0
    p_box_all = []
    actual_box_all = []
    n_wide_race = 0

    for rid, g in vl_scored.groupby(COL_RID, sort=False):
        if len(g) < 5: continue
        # 馬番昇順で sort (wide_pays は馬番ベース)
        g = g.sort_values(COL_BAN).reset_index(drop=True)
        scores = g["_score"].values
        jyun = g[COL_JYUN].astype(int).values
        ban  = pd.to_numeric(g[COL_BAN], errors="coerce").astype(int).values
        n_race += 1

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

        win_mask = (jyun == 1).astype(int)
        high_p_mask = p_win >= p_threshold
        if high_p_mask.sum() > 0:
            p_high_all.extend(p_win[high_p_mask].tolist())
            actual_high_all.extend(win_mask[high_p_mask].tolist())

        # ===== Wide ROI / ECE_wide =====
        rid_s = str(rid)[:16] if not isinstance(rid, (int, np.integer)) else str(int(rid))
        if rid_s not in wide_pays:
            continue
        n_wide_race += 1
        idx_hon = int(order[0]); idx_tai = int(order[1]); idx_san = int(order[2])
        # ペア (馬番 ベース)
        ban_hon = int(ban[idx_hon]); ban_tai = int(ban[idx_tai]); ban_san = int(ban[idx_san])
        box_ban = [
            tuple(sorted((ban_hon, ban_tai))),
            tuple(sorted((ban_hon, ban_san))),
            tuple(sorted((ban_tai, ban_san))),
        ]
        # 払戻ペア辞書 (馬番)
        pay_map = {tuple(sorted((wi_, wj_))): wp for (wi_, wj_, wp) in wide_pays[rid_s]}
        wide_stake += 300  # 100 円 × 3 点
        for pair in box_ban:
            if pair in pay_map:
                wide_return += pay_map[pair]
                wide_hits += 1

        # ECE_wide サンプル: 印 box 3 点の予測確率 vs 実当たり
        all_wide_p = PL.all_wide(w)
        for (i_, j_) in [(idx_hon, idx_tai), (idx_hon, idx_san), (idx_tai, idx_san)]:
            key = (min(i_, j_), max(i_, j_))
            pred_p = all_wide_p[key]
            pair_ban = tuple(sorted((int(ban[i_]), int(ban[j_]))))
            hit = 1 if pair_ban in pay_map else 0
            p_box_all.append(pred_p)
            actual_box_all.append(hit)

    if n_race == 0:
        return None

    p_arr = np.array(p_high_all)
    a_arr = np.array(actual_high_all)
    ece_high = abs(float(p_arr.mean()) - float(a_arr.mean())) if len(p_arr) > 100 else 0.0

    pb_arr = np.array(p_box_all)
    ab_arr = np.array(actual_box_all)
    ece_wide = abs(float(pb_arr.mean()) - float(ab_arr.mean())) if len(pb_arr) > 100 else 0.0

    wide_roi = wide_return / wide_stake if wide_stake > 0 else 0.0

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
        # v7 新規
        "wide_n_race": n_wide_race,
        "wide_stake":  int(wide_stake),
        "wide_return": int(wide_return),
        "wide_roi":    wide_roi,
        "wide_hit_rate": (wide_hits / (3 * n_wide_race)) if n_wide_race > 0 else 0.0,
        "ece_wide": ece_wide,
        "n_wide_box_samples": len(pb_arr),
    }


def composite_score(metrics):
    base = (
        W_NDCG5            * metrics["ndcg5"]
      + W_HON_TOP3         * metrics["hon_top3_rate"]
      + W_TOP3_SUBSET_TOP5 * metrics["top3_subset_top5_rate"]
      + W_WINNER_IN_TOP5   * metrics["winner_in_top5_rate"]
      + W_HON_TOP2         * metrics["hon_top2_rate"]
    )
    # v6
    ece_pen = W_ECE_PENALTY * metrics["ece_high_p"]
    # v7
    wide_bonus = W_WIDE_BONUS * (metrics["wide_roi"] - 1.0)
    wide_ece_pen = W_WIDE_ECE * metrics["ece_wide"]
    return base - ece_pen + wide_bonus - wide_ece_pen


def compute_cv_composite(model, vl_df, X_vl, wide_pays):
    vl_scored = vl_df.copy()
    vl_scored["_score"] = model.predict(X_vl)

    unique_rids = vl_scored[COL_RID].drop_duplicates().values
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    scores = []
    metrics_all = None
    for _, eval_idx in kf.split(unique_rids):
        rids_eval = unique_rids[eval_idx]
        sub = vl_scored[vl_scored[COL_RID].isin(rids_eval)]
        m = evaluate_marks_with_wide(sub, wide_pays)
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

    composite, metrics = compute_cv_composite(
        model, _CACHE["vl_df"], _CACHE["X_vl"], _CACHE["wide_pays"]
    )
    for k in ["ndcg5", "hon_top3_rate", "hon_top2_rate",
              "top3_subset_top5_rate", "winner_in_top5_rate",
              "ece_high_p", "p_pred_high_mean", "actual_high_mean",
              "wide_roi", "wide_hit_rate", "ece_wide"]:
        trial.set_user_attr(k, metrics[k])

    print(f"  trial {trial.number:>2}: composite={composite*100:6.3f}  "
          f"ndcg5={metrics['ndcg5']:.4f}  ◎top3={metrics['hon_top3_rate']*100:5.2f}%  "
          f"wide_ROI={metrics['wide_roi']*100:5.1f}%  "
          f"ECE_h={metrics['ece_high_p']:.4f}  ECE_w={metrics['ece_wide']:.4f}  "
          f"α={alpha:.2f}  iter={best_iter}")
    sys.stdout.flush()
    return composite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT)
    args = ap.parse_args()

    print("=" * 80)
    print(f"optuna_v7_marks.py  (n_trials={args.n_trials}, n_folds={N_FOLDS}, seed={SEED})")
    print(f"  objective = composite_v6")
    print(f"            + {W_WIDE_BONUS} × (wide_roi_3box - 1.0)")
    print(f"            - {W_WIDE_ECE} × ECE_wide")
    print(f"  base = NDCG@5({W_NDCG5}) + ◎top3({W_HON_TOP3}) + top3⊂top5({W_TOP3_SUBSET_TOP5})")
    print(f"       + win∈top5({W_WINNER_IN_TOP5}) + ◎top2({W_HON_TOP2})")
    print(f"  v6 ECE penalty: -{W_ECE_PENALTY} × ECE_high_p (p_win>=0.10)")
    print(f"  v7 wide bonus:  ◎〇▲ box 3 点流し (100 円 × 3 点/R) ROI を valid=2023 で測定")
    print(f"  α 範囲: [0.0, 1.5]   ラベル: clip(6 - 着順, 0, 5)")
    print("=" * 80)

    tr, vl, feats, encs = prep()
    ds_vl, X_vl, vl_df = make_dataset(vl, feats, alpha=0.0)
    print(f"  feats: {len(feats)}")

    print(f"[load] {WIDE_PARQ.name}")
    wide_pays = load_wide_payouts()
    print(f"  wide_payouts races={len(wide_pays):,}")

    _CACHE["tr"] = tr
    _CACHE["vl_df"] = vl_df
    _CACHE["X_vl"] = X_vl
    _CACHE["ds_vl"] = ds_vl
    _CACHE["feats"] = feats
    _CACHE["wide_pays"] = wide_pays

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    print(f"\n[best] composite = {study.best_value*100:.3f}")
    bp = study.best_params
    print(f"  best_iter = {study.best_trial.user_attrs['best_iter']}")
    print(f"  alpha     = {bp['alpha']:.3f}")
    for k in ["ndcg5", "hon_top3_rate", "hon_top2_rate",
              "top3_subset_top5_rate", "winner_in_top5_rate",
              "ece_high_p", "ece_wide", "wide_roi", "wide_hit_rate"]:
        v = study.best_trial.user_attrs[k]
        if "rate" in k or k == "wide_roi":
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
        "wide_bonus_weight": W_WIDE_BONUS,
        "wide_ece_weight":   W_WIDE_ECE,
        "race_relative_mode": "v2",
        "description": "unified_rank_v7: v6 + wide ROI bonus + wide ECE penalty + "
                       "race_relative_feats(mode=v2) (peer_max/min, "
                       "kako5_agari3f_volatility, pos_volatility, peer_count). "
                       "Aim: improve Cowork ワイド ROI (Cowork 実績 97.8% -> 黒字側) and "
                       "馬連/複勝 calibration. "
                       f"{args.n_trials} trials, 5-fold CV, seed=42.",
    }, OUT_MODEL)
    print(f"[saved] {OUT_MODEL}")

    OUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "n_trials": args.n_trials, "n_folds": N_FOLDS, "seed": SEED,
            "best_composite": study.best_value,
            "best_params": bp, "best_iter": best_iter,
            "weights": {
                "ndcg5": W_NDCG5, "hon_top3": W_HON_TOP3,
                "top3_subset_top5": W_TOP3_SUBSET_TOP5,
                "winner_in_top5": W_WINNER_IN_TOP5,
                "hon_top2": W_HON_TOP2,
                "ece_penalty": -W_ECE_PENALTY,
                "wide_bonus":  W_WIDE_BONUS,
                "wide_ece":   -W_WIDE_ECE,
            },
            "best_metrics": {k: study.best_trial.user_attrs[k] for k in
                             ["ndcg5", "hon_top3_rate", "hon_top2_rate",
                              "top3_subset_top5_rate", "winner_in_top5_rate",
                              "ece_high_p", "p_pred_high_mean", "actual_high_mean",
                              "wide_roi", "wide_hit_rate", "ece_wide"]},
            "label_scheme": "clip(6 - 着順, 0, 5)",
            "all_trials": [
                {"number": t.number, "value": t.value, "params": t.params,
                 "best_iter": t.user_attrs.get("best_iter"),
                 "ndcg5": t.user_attrs.get("ndcg5"),
                 "hon_top3_rate": t.user_attrs.get("hon_top3_rate"),
                 "ece_high_p": t.user_attrs.get("ece_high_p"),
                 "wide_roi": t.user_attrs.get("wide_roi"),
                 "ece_wide": t.user_attrs.get("ece_wide")}
                for t in study.trials
            ],
        }, f, indent=2, ensure_ascii=False)
    print(f"[saved] {OUT_JSON}")


if __name__ == "__main__":
    main()
