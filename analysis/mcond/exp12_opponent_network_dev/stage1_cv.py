# -*- coding: utf-8 -*-
"""
stage1_cv.py
=============
EXP12 Stage1最小反証。2023年development、meeting-day forward-chaining CVで
O0(v6+市場)→O1(単純過去field-strength平均)→O2(既存ELO/Glicko/EXP02)→
O3(O2+個体識別1-hop)を比較する。主比較はO3 vs O2。

前処理は各train foldのみでfit。ハイパーパラメータ固定(LogisticRegression
l2 C=1.0 class_weight=None)。2024・2025年は使わない。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

OUT_DIR = Path(__file__).parent / "out"
RANDOM_STATE = 20260921
N_BLOCKS = 6
BOOT_N = 3000

O0_COLS = ["v6_p_win", "market_p_win"]
O2_COLS = O0_COLS + ["elo_T1M1_horse", "g2_mu", "dyn_skill_mu"]
O3_EXTRA_COLS = [
    "unique_opponent_count", "opponent_current_strength_mean", "opponent_current_strength_max",
    "opponent_current_strength_top3_mean", "beaten_opponent_strength_mean",
    "lost_to_opponent_strength_mean", "strongest_beaten_opponent",
    "recency_weighted_opponent_strength", "opponent_strength_dispersion",
    "opponent_later_proved_strong_count", "external_history_gap",
]
O3_COLS = O2_COLS + O3_EXTRA_COLS


def make_pipeline() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(penalty="l2", C=1.0, class_weight=None, max_iter=1000,
                                    random_state=RANDOM_STATE)),
    ])


def forward_chaining_folds(meeting_days_sorted, n_blocks):
    blocks = np.array_split(np.array(meeting_days_sorted), n_blocks)
    folds = []
    for k in range(n_blocks - 1):
        train_md = set(np.concatenate(blocks[: k + 1]).tolist())
        test_md = set(blocks[k + 1].tolist())
        folds.append((train_md, test_md))
    return folds


def meeting_day_paired_bootstrap(diff, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(diff)
    boots = rng.choice(diff, size=(n_boot, n), replace=True).mean(axis=1)
    return dict(mean=float(diff.mean()), ci_lo=float(np.percentile(boots, 2.5)),
                ci_hi=float(np.percentile(boots, 97.5)), p_negative=float((boots < 0).mean()))


def race_level_multinomial_logloss(df: pd.DataFrame, pred_col: str) -> float:
    """レース単位: 各馬の予測確率をレース内で正規化しmultinomial loglossを計算。"""
    losses = []
    for rid, g in df.groupby("rid"):
        p = g[pred_col].values.astype(float)
        p = np.clip(p, 1e-9, None)
        p = p / p.sum()
        y = g["win"].values
        if y.sum() == 1:
            losses.append(-np.log(p[y == 1][0]))
    return float(np.mean(losses)) if losses else np.nan


def winner_rank_by_pred(df: pd.DataFrame, pred_col: str) -> pd.Series:
    """勝ち馬がpred_colの予測順位で何位だったか(1位=的中)。"""
    out = []
    for rid, g in df.groupby("rid"):
        g = g.sort_values(pred_col, ascending=False).reset_index(drop=True)
        win_idx = g.index[g["win"] == 1]
        if len(win_idx) == 1:
            out.append(win_idx[0] + 1)
    return pd.Series(out)


def run():
    ev = pd.read_parquet(OUT_DIR / "evaluation_table_2023.parquet")
    o1_check = pd.read_parquet(OUT_DIR / "o1_check_2023.parquet") if (OUT_DIR / "o1_check_2023.parquet").exists() else None

    ev = ev.dropna(subset=["v6_p_win", "market_p_win"]).copy()
    if o1_check is not None:
        o1_check2 = o1_check.rename(columns={"opponent_avg_quality_prev": "o1_opponent_avg_quality_prev"})
        ev = ev.merge(o1_check2[["rid", "ban", "o1_opponent_avg_quality_prev"]], on=["rid", "ban"], how="left")
    else:
        ev["o1_opponent_avg_quality_prev"] = np.nan

    o1_cols_actual = O0_COLS + ["o1_opponent_avg_quality_prev"]

    meeting_days_sorted = sorted(ev["meeting_day"].unique())
    folds = forward_chaining_folds(meeting_days_sorted, N_BLOCKS)
    print(f"[stage1_cv] meeting_day数={len(meeting_days_sorted)} fold数={len(folds)}")

    model_cols = dict(O0=O0_COLS, O1=o1_cols_actual, O2=O2_COLS, O3=O3_COLS)
    oof_preds = {name: np.full(len(ev), np.nan) for name in model_cols}

    fold_reports = []
    for i, (train_md, test_md) in enumerate(folds):
        train_mask = ev["meeting_day"].isin(train_md).values
        test_mask = ev["meeting_day"].isin(test_md).values
        y_train = ev.loc[train_mask, "win"].values
        y_test = ev.loc[test_mask, "win"].values
        fr = dict(fold=i + 1, n_train=int(train_mask.sum()), n_test=int(test_mask.sum()))
        for name, cols in model_cols.items():
            pipe = make_pipeline()
            Xtr = ev.loc[train_mask, cols].values
            Xte = ev.loc[test_mask, cols].values
            pipe.fit(Xtr, y_train)
            pred = pipe.predict_proba(Xte)[:, 1]
            oof_preds[name][test_mask] = pred
            fr[f"logloss_{name}"] = float(log_loss(y_test, pred, labels=[0, 1]))
            fr[f"brier_{name}"] = float(brier_score_loss(y_test, pred))
        fold_reports.append(fr)
        print(f"  fold{i+1}: n_train={fr['n_train']} n_test={fr['n_test']} "
              f"O2={fr['logloss_O2']:.4f} O3={fr['logloss_O3']:.4f} "
              f"diff={fr['logloss_O3']-fr['logloss_O2']:+.5f}")

    covered = ~np.isnan(oof_preds["O3"])
    oof = ev[covered].copy()
    for name in model_cols:
        oof[f"pred_{name}"] = oof_preds[name][covered]

    y = oof["win"].values
    agg = {}
    for name in model_cols:
        p = oof[f"pred_{name}"].values
        agg[f"logloss_{name}"] = float(log_loss(y, p, labels=[0, 1]))
        agg[f"brier_{name}"] = float(brier_score_loss(y, p))
        agg[f"prauc_{name}"] = float(average_precision_score(y, p))
        agg[f"race_ll_{name}"] = race_level_multinomial_logloss(oof.assign(**{f"_p": p}), "_p")

    print("\n[集約結果(OOF)]")
    for name in model_cols:
        print(f"  {name}: logloss={agg[f'logloss_{name}']:.5f} brier={agg[f'brier_{name}']:.5f} "
              f"pr_auc={agg[f'prauc_{name}']:.5f} race_multiloss={agg[f'race_ll_{name}']:.5f}")

    diff_ll = agg["logloss_O3"] - agg["logloss_O2"]
    diff_br = agg["brier_O3"] - agg["brier_O2"]
    print(f"\n  O3-O2: logloss差={diff_ll:+.5f}  Brier差={diff_br:+.5f}")

    oof["ll_o2"] = -(y * np.log(np.clip(oof["pred_O2"], 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - oof["pred_O2"], 1e-12, 1)))
    oof["ll_o3"] = -(y * np.log(np.clip(oof["pred_O3"], 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - oof["pred_O3"], 1e-12, 1)))
    day_diff = oof.groupby("meeting_day").apply(lambda d: (d["ll_o3"] - d["ll_o2"]).mean(), include_groups=False)
    boot = meeting_day_paired_bootstrap(day_diff.values, BOOT_N, RANDOM_STATE)
    print(f"\n  開催日paired bootstrap(logloss差 O3-O2): mean={boot['mean']:+.5f} "
          f"CI95=[{boot['ci_lo']:+.5f},{boot['ci_hi']:+.5f}]")

    n_folds_o3_better = sum(1 for fr in fold_reports if fr["logloss_O3"] < fr["logloss_O2"])

    # 較正slope
    from sklearn.linear_model import LinearRegression
    def calib(pred, y):
        lr = LinearRegression().fit(pred.reshape(-1, 1), y)
        return float(lr.coef_[0]), float(lr.intercept_)
    slope_o2, icpt_o2 = calib(oof["pred_O2"].values, y)
    slope_o3, icpt_o3 = calib(oof["pred_O3"].values, y)
    print(f"  較正slope: O2={slope_o2:.3f} O3={slope_o3:.3f}")

    result = dict(
        n_meeting_days=len(meeting_days_sorted), n_folds=len(folds), fold_reports=fold_reports,
        agg=agg, logloss_diff_o3_minus_o2=diff_ll, brier_diff_o3_minus_o2=diff_br,
        bootstrap=boot, n_folds_o3_better_logloss=n_folds_o3_better, n_folds_total=len(fold_reports),
        calib_slope_o2=slope_o2, calib_slope_o3=slope_o3,
    )
    import json
    with open(OUT_DIR / "stage1_cv_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[saved] {OUT_DIR / 'stage1_cv_result.json'}")
    return result


if __name__ == "__main__":
    run()
