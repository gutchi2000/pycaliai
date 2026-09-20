# -*- coding: utf-8 -*-
"""
stage_b_sensitivity_stability.py — 保存済み200レース×3回のみを使う再現性分析
================================================================================
2026-09-20夜、ユーザー指定。追加API呼び出しは一切行わない
(stage_b_sensitivity_results.jsonlを読むだけ)。単発回答を平均値へ置き換えたり、
主評価(stage_b_gate_eval.pyのGate1-3判定)を書き換えたりしない。

出力: out/STAGE_B_SENSITIVITY_STABILITY.json

実行: python -m analysis.mcond.exp06_jev_decision_dev.stage_b_sensitivity_stability
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.exp06_jev_decision_dev import stage_b_gate_eval as GE  # noqa: E402
from analysis.mcond.exp06_jev_decision_dev import stage_b_run as SBR  # noqa: E402

HERE = Path(__file__).resolve().parent
Q4_OPTIONS = ["AI", "MARKET", "BLEND", "ABSTAIN"]
Q5_OPTIONS = ["BET", "PASS_NO_EDGE", "PASS_UNCERTAIN", "PASS_OOD", "PASS_PRICE_RISK"]


def _load_sensitivity_by_race() -> dict:
    by_race = defaultdict(list)
    for line in (HERE / "out" / "stage_b_sensitivity_results.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if not rec.get("ok"):
            continue
        by_race[rec["rid16"]].append(rec["answers"])
    return by_race


def _vec_probs(answers: dict, qid: str, options: list[str]) -> np.ndarray | None:
    q = answers.get(qid) or {}
    probs = q.get("probabilities") or {}
    if not probs:
        return None
    return np.array([float(probs.get(o, 0.0)) for o in options], dtype=float)


def _pairwise_mean_abs_diff(vecs: list[np.ndarray]) -> float:
    diffs = [float(np.mean(np.abs(a - b))) for a, b in combinations(vecs, 2)]
    return float(np.mean(diffs)) if diffs else float("nan")


def _pairwise_max_abs_diff(vecs: list[np.ndarray]) -> float:
    diffs = [float(np.max(np.abs(a - b))) for a, b in combinations(vecs, 2)]
    return float(np.max(diffs)) if diffs else float("nan")


def _icc_oneway(scores_by_race: list[list[float]]) -> float:
    """ICC(1) 一元配置ランダム効果モデル。k=各レースの反復数(全レース同一想定)。"""
    arr = np.array(scores_by_race, dtype=float)  # shape (n_races, k)
    n, k = arr.shape
    grand_mean = arr.mean()
    row_means = arr.mean(axis=1)
    ss_between = k * np.sum((row_means - grand_mean) ** 2)
    ss_within = np.sum((arr - row_means[:, None]) ** 2)
    df_between = n - 1
    df_within = n * (k - 1)
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within if df_within > 0 else np.nan
    denom = ms_between + (k - 1) * ms_within
    if denom == 0 or not np.isfinite(denom):
        return float("nan")
    return float((ms_between - ms_within) / denom)


def per_race_metrics(by_race: dict) -> pd.DataFrame:
    rows = []
    for rid16, answers_list in by_race.items():
        if len(answers_list) < 2:
            continue
        q1_vals = [float((a.get("Q1") or {}).get("noul", np.nan)) for a in answers_list]
        q2_scores = [float((a.get("Q2") or {}).get("score", np.nan)) for a in answers_list]
        q2_vecs = [v for v in (_vec_probs(a, "Q2", [str(i) for i in range(5)]) for a in answers_list) if v is not None]
        q4_vecs = [v for v in (_vec_probs(a, "Q4", Q4_OPTIONS) for a in answers_list) if v is not None]
        q5_vecs = [v for v in (_vec_probs(a, "Q5", Q5_OPTIONS) for a in answers_list) if v is not None]
        q4_choices = [(a.get("Q4") or {}).get("choice") for a in answers_list]
        q5_choices = [(a.get("Q5") or {}).get("choice") for a in answers_list]
        is_bet = [c == "BET" for c in q5_choices]

        row = {
            "rid16": rid16, "n_reps": len(answers_list),
            "q1_mean_abs_diff": _pairwise_mean_abs_diff([np.array([v]) for v in q1_vals]),
            "q1_max_abs_diff": _pairwise_max_abs_diff([np.array([v]) for v in q1_vals]),
            "q2_score": q2_scores,
            "q2_mean_abs_diff": _pairwise_mean_abs_diff(q2_vecs) if len(q2_vecs) >= 2 else np.nan,
            "q2_max_abs_diff": _pairwise_max_abs_diff(q2_vecs) if len(q2_vecs) >= 2 else np.nan,
            "q4_mean_abs_diff": _pairwise_mean_abs_diff(q4_vecs) if len(q4_vecs) >= 2 else np.nan,
            "q4_max_abs_diff": _pairwise_max_abs_diff(q4_vecs) if len(q4_vecs) >= 2 else np.nan,
            "q5_mean_abs_diff": _pairwise_mean_abs_diff(q5_vecs) if len(q5_vecs) >= 2 else np.nan,
            "q5_max_abs_diff": _pairwise_max_abs_diff(q5_vecs) if len(q5_vecs) >= 2 else np.nan,
            "q4_unanimous": len(set(q4_choices)) == 1,
            "q5_unanimous": len(set(q5_choices)) == 1,
            "bet_pass_unanimous": len(set(is_bet)) == 1,
        }
        rows.append(row)
    return pd.DataFrame(rows).set_index("rid16")


def _stratum_summary(df: pd.DataFrame) -> dict:
    return {
        "n": int(len(df)),
        "q1_mean_abs_diff": float(df["q1_mean_abs_diff"].mean()) if len(df) else None,
        "q2_mean_abs_diff": float(df["q2_mean_abs_diff"].mean()) if len(df) else None,
        "q4_mean_abs_diff": float(df["q4_mean_abs_diff"].mean()) if len(df) else None,
        "q5_mean_abs_diff": float(df["q5_mean_abs_diff"].mean()) if len(df) else None,
        "q4_argmax_agreement_rate": float(df["q4_unanimous"].mean()) if len(df) else None,
        "q5_argmax_agreement_rate": float(df["q5_unanimous"].mean()) if len(df) else None,
        "bet_pass_reversal_rate": float(1 - df["bet_pass_unanimous"].mean()) if len(df) else None,
        "q4_category_reversal_rate": float(1 - df["q4_unanimous"].mean()) if len(df) else None,
    }


def main() -> dict:
    by_race = _load_sensitivity_by_race()
    metrics = per_race_metrics(by_race)

    icc_q2 = _icc_oneway(metrics["q2_score"].tolist())

    overall = _stratum_summary(metrics)
    overall["q2_score_icc_oneway"] = icc_q2
    overall["max_abs_diff_observed"] = {
        "q1": float(metrics["q1_max_abs_diff"].max()),
        "q2": float(metrics["q2_max_abs_diff"].max()),
        "q4": float(metrics["q4_max_abs_diff"].max()),
        "q5": float(metrics["q5_max_abs_diff"].max()),
    }

    print("[stability] loading state for year/surface/support quintile join...")
    d2023, d2425, _ = GE.build_frames()
    state_all = pd.concat([d2023, d2425])
    joined = metrics.join(state_all[["year", "surface", "in_distribution_support"]], how="left")
    joined["support_quintile"] = pd.qcut(joined["in_distribution_support"], 5,
                                         labels=["q1_low", "q2", "q3", "q4", "q5_high"],
                                         duplicates="drop")

    by_year = {str(y): _stratum_summary(g) for y, g in joined.groupby("year")}
    by_surface = {str(s): _stratum_summary(g) for s, g in joined.groupby("surface")}
    by_quintile = {str(q): _stratum_summary(g) for q, g in joined.groupby("support_quintile", observed=True)}

    return {
        "evaluated_at": "2026-09-20", "note": "追加API呼び出しなし、保存済み200レース×3回のみ使用",
        "n_races_with_ge2_valid_reps": int(len(metrics)),
        "overall": overall,
        "breakdown_by_year": by_year,
        "breakdown_by_surface": by_surface,
        "breakdown_by_support_quintile": by_quintile,
        "small_sample_caveat": "各層n=数十件程度のため、層別の値は傾向の参考に留め断定しない。",
    }


if __name__ == "__main__":
    result = main()
    out_path = HERE / "out" / "STAGE_B_SENSITIVITY_STABILITY.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[stability] wrote {out_path}")
