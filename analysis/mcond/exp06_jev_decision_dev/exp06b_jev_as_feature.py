# -*- coding: utf-8 -*-
"""
exp06b_jev_as_feature.py — EXP06B: Jevを判断器ではなく特徴として使う探索分析
================================================================================
2026-09-20夜、ユーザー指定。**EXP06本体(Jev direct decision gate)のGate救済では
ない**。EXP06はGate3 FAILで既に終了しており、この結論は変わらない。本ファイルは
2024/2025の結果を見た「後」に立てる完全に別の探索仮説(EXP06B)として明確に分離する。
ROIは実施しない。

R0 = LR_CONTROL相当(Jevが受け取ったのと同じ入力のみ)で実際の結果(top3)を予測する
     単純ロジスティック回帰。
R1 = R0の入力 + Jevの連続値出力(Q1 noul確率、Q2 scoreと各levelの確率、Q4の
     AI/MARKET/BLEND/ABSTAIN確率、Q5のBET/PASS_*各確率)。argmaxではなく連続値を使う。

手順(事前固定、2024/2025を見た後に変更しない):
  - 2023(development)だけでfit。目的変数は実際のtop3ラベル(2023年結果、fit専用)。
  - 前処理(StandardScaler)・正則化強度(LogisticRegressionCVでC選択、2023のみで
    cross-validation)は2023で固定し、2024/2025では一切再調整しない。
  - 2024・2025へそのまま適用してBrier/loglossを計算、レース単位ブートストラップCIで
    R1-R0の差を評価。

判定基準(事前固定):
  - R1がR0を2024・2025の両方でBrier改善(R1-R0の95%CI上限<0): 「Jev確率を補助特徴と
    する前向きshadow候補」= exploratory PASS。ただし前向きshadowで再確認するまで
    有効性確定とは呼ばない。
  - 両年とも改善なし: 「数値表データに対するJev研究は終了」= FAIL。
  - 片年だけ改善: 「不安定」= FAIL(終了)。

実行: python -m analysis.mcond.exp06_jev_decision_dev.exp06b_jev_as_feature
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.exp06_jev_decision_dev import stage_b_gate_eval as GE  # noqa: E402
from analysis.mcond.exp06_jev_decision_dev.ood_support import CONTINUOUS_COLS, CATEGORICAL_COLS  # noqa: E402

HERE = Path(__file__).resolve().parent
EPS = 1e-9
N_BOOTSTRAP = 2000
RNG_SEED = 20260920

BASE_CONT_COLS = CONTINUOUS_COLS + ["in_distribution_support", "similar_past_case_count"]
BASE_CAT_COLS = CATEGORICAL_COLS
Q4_OPTIONS = ["AI", "MARKET", "BLEND", "ABSTAIN"]
Q5_OPTIONS = ["BET", "PASS_NO_EDGE", "PASS_UNCERTAIN", "PASS_OOD", "PASS_PRICE_RISK"]
Q2_LEVELS = [str(i) for i in range(5)]


def _jev_feature_row(answers: dict) -> dict:
    q1 = answers.get("Q1") or {}
    q2 = answers.get("Q2") or {}
    q2p = q2.get("probabilities") or {}
    q4p = (answers.get("Q4") or {}).get("probabilities") or {}
    q5p = (answers.get("Q5") or {}).get("probabilities") or {}
    row = {"jev_q1_noul": float(q1.get("noul", np.nan)), "jev_q2_score": float(q2.get("score", np.nan))}
    for lv in Q2_LEVELS:
        row[f"jev_q2_p{lv}"] = float(q2p.get(lv, 0.0))
    for opt in Q4_OPTIONS:
        row[f"jev_q4_p_{opt}"] = float(q4p.get(opt, 0.0))
    for opt in Q5_OPTIONS:
        row[f"jev_q5_p_{opt}"] = float(q5p.get(opt, 0.0))
    return row


def _onehot_fit(d: pd.DataFrame, cols: list[str]):
    vocab = {c: sorted(d[c].astype(str).unique()) for c in cols}
    return _onehot_apply(d, cols, vocab), vocab


def _onehot_apply(d: pd.DataFrame, cols: list[str], vocab: dict) -> np.ndarray:
    blocks = []
    for c in cols:
        v = vocab[c]
        vidx = {val: i for i, val in enumerate(v)}
        block = np.zeros((len(d), len(v)))
        for i, val in enumerate(d[c].astype(str)):
            j = vidx.get(val)
            if j is not None:
                block[i, j] = 1.0
        blocks.append(block)
    return np.hstack(blocks) if blocks else np.zeros((len(d), 0))


def build_XR0_R1(d2023: pd.DataFrame, d2425: pd.DataFrame, dev_answers: dict, pri_answers: dict):
    jev_cols_2023 = pd.DataFrame([_jev_feature_row(dev_answers[rid]) for rid in d2023.index], index=d2023.index)
    jev_cols_2425 = pd.DataFrame([_jev_feature_row(pri_answers[rid]) for rid in d2425.index], index=d2425.index)
    jev_feature_names = list(jev_cols_2023.columns)

    cat_onehot_2023, vocab = _onehot_fit(d2023, BASE_CAT_COLS)
    cat_onehot_2425 = _onehot_apply(d2425, BASE_CAT_COLS, vocab)

    cont_2023 = np.nan_to_num(d2023[BASE_CONT_COLS].to_numpy(dtype=float), nan=0.0)
    cont_2425 = np.nan_to_num(d2425[BASE_CONT_COLS].to_numpy(dtype=float), nan=0.0)

    X0_2023 = np.hstack([cont_2023, cat_onehot_2023])
    X0_2425 = np.hstack([cont_2425, cat_onehot_2425])

    jev_2023_arr = np.nan_to_num(jev_cols_2023.to_numpy(dtype=float), nan=0.0)
    jev_2425_arr = np.nan_to_num(jev_cols_2425.to_numpy(dtype=float), nan=0.0)
    X1_2023 = np.hstack([X0_2023, jev_2023_arr])
    X1_2425 = np.hstack([X0_2425, jev_2425_arr])

    y_2023 = d2023["_result_top3"].to_numpy()
    y_2425 = d2425["_result_top3"].to_numpy()
    return {
        "X0_2023": X0_2023, "X1_2023": X1_2023, "y_2023": y_2023,
        "X0_2425": X0_2425, "X1_2425": X1_2425, "y_2425": y_2425,
        "year_2425": d2425["year"].to_numpy(), "jev_feature_names": jev_feature_names,
    }


def _fit_frozen(X_2023: np.ndarray, y_2023: np.ndarray):
    """前処理(StandardScaler)・正則化(LogisticRegressionCVでC選択)を2023のみで固定する。"""
    scaler = StandardScaler().fit(X_2023)
    Xs = scaler.transform(X_2023)
    clf = LogisticRegressionCV(cv=5, max_iter=5000, Cs=10, scoring="neg_brier_score",
                               random_state=RNG_SEED)
    clf.fit(Xs, y_2023)
    return scaler, clf


def _predict(scaler, clf, X: np.ndarray) -> np.ndarray:
    return clf.predict_proba(scaler.transform(X))[:, 1]


def _bootstrap_diff(p0: np.ndarray, p1: np.ndarray, y: np.ndarray, metric_fn, n=N_BOOTSTRAP, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    m = len(y)
    diffs = np.empty(n)
    for b in range(n):
        idx = rng.integers(0, m, size=m)
        e0 = float(np.mean(metric_fn(p0[idx], y[idx])))
        e1 = float(np.mean(metric_fn(p1[idx], y[idx])))
        diffs[b] = e1 - e0
    point = float(np.mean(metric_fn(p1, y)) - np.mean(metric_fn(p0, y)))
    return {"point_r1_minus_r0": point, "ci_lower_2.5": float(np.percentile(diffs, 2.5)),
           "ci_upper_97.5": float(np.percentile(diffs, 97.5))}


def main() -> dict:
    d2023, d2425, _ = GE.build_frames()
    dev_answers = GE._load_jev_answers(GE.SBR.DEVELOPMENT_LOG_PATH)
    pri_answers = GE._load_jev_answers(GE.SBR.PRIMARY_LOG_PATH)
    d2023 = d2023[d2023.index.isin(dev_answers.keys())].copy()
    d2425 = d2425[d2425.index.isin(pri_answers.keys())].copy()

    data = build_XR0_R1(d2023, d2425, dev_answers, pri_answers)

    scaler0, clf0 = _fit_frozen(data["X0_2023"], data["y_2023"])
    scaler1, clf1 = _fit_frozen(data["X1_2023"], data["y_2023"])

    p0_2425 = _predict(scaler0, clf0, data["X0_2425"])
    p1_2425 = _predict(scaler1, clf1, data["X1_2425"])
    y_2425 = data["y_2425"]
    year_2425 = data["year_2425"]

    result = {
        "evaluated_at": "2026-09-20", "hypothesis": "EXP06B (post-hoc探索、EXP06本体のGate救済ではない)",
        "no_roi_evaluated": True,
        "n_2023_fit": int(len(d2023)), "n_2024_2025_scored": int(len(d2425)),
        "jev_features_added": data["jev_feature_names"],
        "frozen_note": "前処理・正則化は2023のみで固定、2024/2025では一切再調整していない。",
        "by_year": {},
    }
    overall_pass = []
    for yr in (2024, 2025):
        mask = year_2425 == yr
        p0, p1, y = p0_2425[mask], p1_2425[mask], y_2425[mask]
        brier_diff = _bootstrap_diff(p0, p1, y, lambda p, yy: (p - yy) ** 2, seed=RNG_SEED + yr)
        logloss_diff = _bootstrap_diff(
            p0, p1, y,
            lambda p, yy: -(yy * np.log(np.clip(p, EPS, 1 - EPS)) + (1 - yy) * np.log(1 - np.clip(p, EPS, 1 - EPS))),
            seed=RNG_SEED + yr + 1000)
        improved = brier_diff["ci_upper_97.5"] < 0
        result["by_year"][str(yr)] = {
            "n": int(mask.sum()),
            "r0_brier_mean": float(np.mean((p0 - y) ** 2)), "r1_brier_mean": float(np.mean((p1 - y) ** 2)),
            "brier_r1_minus_r0": brier_diff, "logloss_r1_minus_r0": logloss_diff,
            "r1_improves_over_r0": improved,
        }
        overall_pass.append(improved)

    if all(overall_pass):
        verdict = "exploratory_PASS"
        verdict_note = ("R1がR0を2024・2025の両方でBrier改善(差の95%CI上限<0)。Jev確率を補助特徴と"
                        "する前向きshadow候補。有効性は前向きshadowで再確認するまで確定としない。")
    elif not any(overall_pass):
        verdict = "FAIL"
        verdict_note = "両年ともR1の改善なし。数値表データに対するJev研究はここで終了とする。"
    else:
        verdict = "FAIL"
        verdict_note = "片年のみ改善(不安定)。安定した効果と認めず、ここで終了とする。"

    result["verdict"] = verdict
    result["verdict_note"] = verdict_note
    return result


if __name__ == "__main__":
    result = main()
    out_path = HERE / "out" / "EXP06B_JEV_AS_FEATURE.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[exp06b] wrote {out_path}")
    print(f"[exp06b] verdict={result['verdict']}")
