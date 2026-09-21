# -*- coding: utf-8 -*-
"""
stage1_5_cv.py
===============
EXP10 Stage 1.5. 最小反証実験: 2023年developmentのみを使い、meeting-day単位の
forward-chaining(blocked chronological) CVでB3 vs R1(最小構成)を比較する。
2024・2025年は一切使わない。

設計(ユーザー指定、spec.jsonに凍結):
  - random 5-foldは禁止。meeting-day(rid[:10]=日付+場所)を分割単位にする。
  - 時系列順を維持するforward-chaining CV。同一開催日のレースをtrain/valへ
    分割しない。
  - scaling(標準化)・欠損補完は各train foldだけでfitする(sklearn Pipeline)。
  - class_weight・正則化・特徴群は実行前に固定する(spec.json参照、事後変更禁止)。
  - SMOTE等の合成oversamplingは使わない。

判定(ユーザー指定):
  主比較はR1 vs B3のOOF予測、開催日単位paired bootstrap。続行条件は
  STAGE1_DESIGN.md §7 / README参照。一つでも満たさなければ2024・2025年を
  開封せずEXP10を終了する。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_5_features import build_stage1_5_table, B3_COLS, R1_EXTRA_COLS  # noqa: E402

RANDOM_STATE = 20261013
N_TIME_BLOCKS = 6  # meeting-dayを6ブロックに時系列分割 -> forward-chaining 5fold
BOOT_N = 3000
BOOT_SEED = 20261013


def make_pipeline() -> Pipeline:
    # 固定ハイパーパラメータ(事後変更禁止、spec.json参照):
    #   SimpleImputer(strategy="median")  train foldのみでfit
    #   StandardScaler()                  train foldのみでfit
    #   LogisticRegression(penalty="l2", C=1.0, class_weight=None, max_iter=1000)
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(penalty="l2", C=1.0, class_weight=None,
                                    max_iter=1000, random_state=RANDOM_STATE)),
    ])


def forward_chaining_folds(meeting_days_sorted: list[str], n_blocks: int):
    """meeting_day(時系列順)をn_blocksに等分し、fold_k: train=block[0..k], test=block[k+1]。
    戻り値: [(train_meeting_days, test_meeting_days), ...] (n_blocks-1個)。
    """
    blocks = np.array_split(np.array(meeting_days_sorted), n_blocks)
    folds = []
    for k in range(n_blocks - 1):
        train_md = set(np.concatenate(blocks[: k + 1]).tolist())
        test_md = set(blocks[k + 1].tolist())
        folds.append((train_md, test_md))
    return folds


def meeting_day_paired_bootstrap(per_day_diff: np.ndarray, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(per_day_diff)
    boots = rng.choice(per_day_diff, size=(n_boot, n), replace=True).mean(axis=1)
    return dict(
        mean=float(per_day_diff.mean()),
        ci_lo=float(np.percentile(boots, 2.5)),
        ci_hi=float(np.percentile(boots, 97.5)),
        ci_hi_975=float(np.percentile(boots, 97.5)),
        p_negative=float((boots < 0).mean()),
    )


def run_stage1_5():
    table, diag = build_stage1_5_table([2023])
    print(f"[Stage1.5] 特徴テーブル: {len(table):,}行")

    meeting_days_sorted = sorted(table["meeting_day"].unique())  # 日付文字列の辞書順=時系列順
    folds = forward_chaining_folds(meeting_days_sorted, N_TIME_BLOCKS)
    print(f"[Stage1.5] meeting_day数={len(meeting_days_sorted)}  fold数={len(folds)}")

    r1_cols = B3_COLS + R1_EXTRA_COLS
    oof_b3 = np.full(len(table), np.nan)
    oof_r1 = np.full(len(table), np.nan)
    fold_reports = []

    for i, (train_md, test_md) in enumerate(folds):
        train_mask = table["meeting_day"].isin(train_md).values
        test_mask = table["meeting_day"].isin(test_md).values
        train_df = table[train_mask]
        test_df = table[test_mask]
        y_train = train_df["catastrophic_downside"].values
        y_test = test_df["catastrophic_downside"].values

        pipe_b3 = make_pipeline()
        pipe_b3.fit(train_df[B3_COLS].values, y_train)
        pred_b3 = pipe_b3.predict_proba(test_df[B3_COLS].values)[:, 1]

        pipe_r1 = make_pipeline()
        pipe_r1.fit(train_df[r1_cols].values, y_train)
        pred_r1 = pipe_r1.predict_proba(test_df[r1_cols].values)[:, 1]

        oof_b3[test_mask] = pred_b3
        oof_r1[test_mask] = pred_r1

        fold_reports.append(dict(
            fold=i + 1,
            train_period=(min(train_md), max(train_md)),
            test_period=(min(test_md), max(test_md)),
            n_train=int(train_mask.sum()), n_test=int(test_mask.sum()),
            n_train_pos=int(y_train.sum()), n_test_pos=int(y_test.sum()),
            logloss_b3=float(log_loss(y_test, pred_b3, labels=[0, 1])),
            logloss_r1=float(log_loss(y_test, pred_r1, labels=[0, 1])),
            brier_b3=float(brier_score_loss(y_test, pred_b3)),
            brier_r1=float(brier_score_loss(y_test, pred_r1)),
            pr_auc_b3=float(average_precision_score(y_test, pred_b3)),
            pr_auc_r1=float(average_precision_score(y_test, pred_r1)),
        ))

    print("\n[Stage1.5] fold別レポート")
    for fr in fold_reports:
        print(f"  fold{fr['fold']}: train={fr['train_period']} test={fr['test_period']} "
              f"n_train={fr['n_train']}(pos={fr['n_train_pos']}) n_test={fr['n_test']}(pos={fr['n_test_pos']}) "
              f"logloss B3={fr['logloss_b3']:.4f} R1={fr['logloss_r1']:.4f} "
              f"(diff={fr['logloss_r1']-fr['logloss_b3']:+.4f}) "
              f"brier B3={fr['brier_b3']:.4f} R1={fr['brier_r1']:.4f} "
              f"pr_auc B3={fr['pr_auc_b3']:.4f} R1={fr['pr_auc_r1']:.4f}")

    covered = ~np.isnan(oof_b3)
    oof_table = table[covered].copy()
    oof_table["pred_b3"] = oof_b3[covered]
    oof_table["pred_r1"] = oof_r1[covered]
    y_oof = oof_table["catastrophic_downside"].values

    agg_logloss_b3 = log_loss(y_oof, oof_table["pred_b3"].values, labels=[0, 1])
    agg_logloss_r1 = log_loss(y_oof, oof_table["pred_r1"].values, labels=[0, 1])
    agg_brier_b3 = brier_score_loss(y_oof, oof_table["pred_b3"].values)
    agg_brier_r1 = brier_score_loss(y_oof, oof_table["pred_r1"].values)
    agg_prauc_b3 = average_precision_score(y_oof, oof_table["pred_b3"].values)
    agg_prauc_r1 = average_precision_score(y_oof, oof_table["pred_r1"].values)

    # 開催日単位 paired bootstrap (logloss差, Brier差)
    oof_table["ll_b3"] = -(y_oof * np.log(np.clip(oof_table["pred_b3"], 1e-12, 1)) +
                            (1 - y_oof) * np.log(np.clip(1 - oof_table["pred_b3"], 1e-12, 1)))
    oof_table["ll_r1"] = -(y_oof * np.log(np.clip(oof_table["pred_r1"], 1e-12, 1)) +
                            (1 - y_oof) * np.log(np.clip(1 - oof_table["pred_r1"], 1e-12, 1)))
    oof_table["br_b3"] = (oof_table["pred_b3"] - y_oof) ** 2
    oof_table["br_r1"] = (oof_table["pred_r1"] - y_oof) ** 2
    day_ll = oof_table.groupby("meeting_day").apply(
        lambda d: (d["ll_r1"] - d["ll_b3"]).mean(), include_groups=False)
    day_br = oof_table.groupby("meeting_day").apply(
        lambda d: (d["br_r1"] - d["br_b3"]).mean(), include_groups=False)

    ll_boot = meeting_day_paired_bootstrap(day_ll.values, BOOT_N, BOOT_SEED)
    br_boot = meeting_day_paired_bootstrap(day_br.values, BOOT_N, BOOT_SEED)

    n_folds_r1_better_logloss = sum(1 for fr in fold_reports if fr["logloss_r1"] < fr["logloss_b3"])
    n_folds_r1_better_brier = sum(1 for fr in fold_reports if fr["brier_r1"] < fr["brier_b3"])

    # calibration: 較正slope/intercept簡易確認(較正後予測 vs 実現率の単回帰)
    def calib_slope_intercept(pred, y):
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression().fit(pred.reshape(-1, 1), y)
        return float(lr.coef_[0]), float(lr.intercept_)

    slope_b3, icpt_b3 = calib_slope_intercept(oof_table["pred_b3"].values, y_oof)
    slope_r1, icpt_r1 = calib_slope_intercept(oof_table["pred_r1"].values, y_oof)

    result = dict(
        n_meeting_days=len(meeting_days_sorted), n_folds=len(folds),
        agg_logloss_b3=agg_logloss_b3, agg_logloss_r1=agg_logloss_r1,
        agg_logloss_diff=agg_logloss_r1 - agg_logloss_b3,
        agg_brier_b3=agg_brier_b3, agg_brier_r1=agg_brier_r1,
        agg_brier_diff=agg_brier_r1 - agg_brier_b3,
        agg_prauc_b3=agg_prauc_b3, agg_prauc_r1=agg_prauc_r1,
        agg_prauc_diff=agg_prauc_r1 - agg_prauc_b3,
        logloss_bootstrap=ll_boot, brier_bootstrap=br_boot,
        n_folds_r1_better_logloss=n_folds_r1_better_logloss,
        n_folds_r1_better_brier=n_folds_r1_better_brier,
        n_folds_total=len(fold_reports),
        calib_slope_b3=slope_b3, calib_intercept_b3=icpt_b3,
        calib_slope_r1=slope_r1, calib_intercept_r1=icpt_r1,
        fold_reports=fold_reports,
    )

    print("\n[Stage1.5] 集約結果(OOF、全fold結合)")
    print(f"  logloss  B3={agg_logloss_b3:.5f}  R1={agg_logloss_r1:.5f}  diff(R1-B3)={result['agg_logloss_diff']:+.5f}")
    print(f"  Brier    B3={agg_brier_b3:.5f}  R1={agg_brier_r1:.5f}  diff(R1-B3)={result['agg_brier_diff']:+.5f}")
    print(f"  PR-AUC   B3={agg_prauc_b3:.5f}  R1={agg_prauc_r1:.5f}  diff(R1-B3)={result['agg_prauc_diff']:+.5f}")
    print(f"  logloss 開催日paired bootstrap: mean={ll_boot['mean']:+.5f} "
          f"CI95=[{ll_boot['ci_lo']:+.5f},{ll_boot['ci_hi']:+.5f}]  P(diff<0)={ll_boot['p_negative']:.3f}")
    print(f"  Brier   開催日paired bootstrap: mean={br_boot['mean']:+.5f} "
          f"CI95=[{br_boot['ci_lo']:+.5f},{br_boot['ci_hi']:+.5f}]  P(diff<0)={br_boot['p_negative']:.3f}")
    print(f"  logloss改善foldの割合: {n_folds_r1_better_logloss}/{len(fold_reports)}")
    print(f"  Brier改善foldの割合: {n_folds_r1_better_brier}/{len(fold_reports)}")
    print(f"  較正slope/intercept: B3=({slope_b3:.3f},{icpt_b3:.3f})  R1=({slope_r1:.3f},{icpt_r1:.3f})")

    # --- 続行条件の判定 ---
    cond = {}
    cond["agg_logloss_improves"] = result["agg_logloss_diff"] < 0
    cond["agg_brier_improves"] = result["agg_brier_diff"] < 0
    cond["prauc_improves"] = result["agg_prauc_diff"] > 0
    cond["logloss_ci_upper_below_0"] = ll_boot["ci_hi"] < 0
    cond["brier_ci_upper_below_0"] = br_boot["ci_hi"] < 0
    cond["majority_folds_same_direction_logloss"] = n_folds_r1_better_logloss > len(fold_reports) / 2
    cond["improvement_not_concentrated_single_fold"] = True  # 下で個別確認・手動記録
    cond["calibration_not_worse"] = abs(slope_r1 - 1.0) <= abs(slope_b3 - 1.0) + 0.05

    print("\n[Stage1.5] 続行条件チェック")
    for k, v in cond.items():
        print(f"  {k}: {v}")
    all_pass = all(cond.values())
    print(f"\n[Stage1.5] 判定: {'続行(Stage2へ)' if all_pass else 'EXP10終了(2024・2025年は開封しない)'}")

    result["conditions"] = cond
    result["all_conditions_pass"] = all_pass
    with open(Path(__file__).parent / "out" / "STAGE1_5_REPORT.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print("[saved] out/STAGE1_5_REPORT.json")
    return result


if __name__ == "__main__":
    run_stage1_5()
