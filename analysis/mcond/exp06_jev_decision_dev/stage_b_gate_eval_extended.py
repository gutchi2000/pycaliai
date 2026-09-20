# -*- coding: utf-8 -*-
"""
stage_b_gate_eval_extended.py — Gate2完全統制版 + Gate3層別breakdown (2026-09-20夜)
================================================================================
ユーザー指定。stage_b_gate_eval.py(2026-09-20昼、正式なGate1-3判定、コミット済み
14494c75)の結果・verdictは一切変更しない。本ファイルは追加の診断出力のみを行う
(そのため独立ファイルとして分離、frozen scriptへは書き戻さない)。

C. Gate2完全統制版: 元の4統制変数(in_distribution_support/model_rank_disagreement/
   feature_missing_rate/model_prob_variance)に加え、m4_top_prob/entropy_m4/
   market_prob/market_entropy/ai_market_divergence/similar_past_case_count/
   field_size、およびカテゴリ変数(popularity_band/venue/surface/distance_band/
   class_band/year)の一次元one-hotを全て追加統制する。odds_change_rate等の
   T-10/T-20由来項目はavailability=falseのため引き続き使わない。

D. Gate3層別breakdown: 2024/2025・競馬場・人気帯・芝ダート・距離帯・support
   quintileの各区分で、区分内のJevの実coverageに合わせてLR_CONTROLと単純OOD
   ゲートを再選定し(全体coverageのままではなく区分ごとに揃える)、平均Brier/
   loglossを比較する。overall FAIL(stage_b_gate_eval.py側)は変更しない。

実行: python -m analysis.mcond.exp06_jev_decision_dev.stage_b_gate_eval_extended
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.exp06_jev_decision_dev import stage_b_gate_eval as GE  # noqa: E402

HERE = Path(__file__).resolve().parent
EPS = 1e-9
N_BOOTSTRAP = 2000
RNG_SEED = 20260920

FULL_CONT_CONTROLS = [
    "m4_top_prob", "entropy_m4", "market_prob", "market_entropy", "ai_market_divergence",
    "model_rank_disagreement", "model_prob_variance", "in_distribution_support",
    "similar_past_case_count", "feature_missing_rate", "field_size",
]
FULL_CAT_CONTROLS = ["popularity_band", "venue", "surface", "distance_band", "class_band"]


def _onehot(d: pd.DataFrame, cols: list[str], vocab: dict[str, list[str]] | None = None):
    if vocab is None:
        vocab = {c: sorted(d[c].astype(str).unique()) for c in cols}
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
    return np.hstack(blocks) if blocks else np.zeros((len(d), 0)), vocab


def _build_full_X(d: pd.DataFrame, include_year: bool, vocab=None, year_vocab=None):
    cont = d[FULL_CONT_CONTROLS].to_numpy(dtype=float)
    cont = np.nan_to_num(cont, nan=0.0)
    cat, vocab = _onehot(d, FULL_CAT_CONTROLS, vocab)
    blocks = [cont, cat]
    if include_year:
        year_block, year_vocab = _onehot(d.assign(_year_str=d["year"].astype(str)), ["_year_str"], year_vocab)
        blocks.append(year_block)
    return np.hstack(blocks), vocab, year_vocab


def gate2_full_controls(d2425: pd.DataFrame) -> dict:
    y = d2425["brier"].to_numpy()
    X_controls, vocab, year_vocab = _build_full_X(d2425, include_year=True)
    X_full = np.column_stack([X_controls, d2425["risk_prob"].to_numpy(dtype=float)])

    def coef_and_r2gain(y_arr, X_c, X_f):
        lr_c = LinearRegression().fit(X_c, y_arr)
        ss_res_c = float(np.sum((y_arr - lr_c.predict(X_c)) ** 2))
        lr_f = LinearRegression().fit(X_f, y_arr)
        resid_f = y_arr - lr_f.predict(X_f)
        ss_res_f = float(np.sum(resid_f ** 2))
        r2gain = (ss_res_c - ss_res_f) / max(ss_res_c, EPS)
        return float(lr_f.coef_[-1]), r2gain

    rng = np.random.default_rng(RNG_SEED)
    n = len(y)
    boot_coef = np.empty(N_BOOTSTRAP)
    boot_r2 = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        boot_coef[b], boot_r2[b] = coef_and_r2gain(y[idx], X_controls[idx], X_full[idx])
    point_coef, point_r2 = coef_and_r2gain(y, X_controls, X_full)
    ci_lo, ci_hi = float(np.percentile(boot_coef, 2.5)), float(np.percentile(boot_coef, 97.5))

    per_year = {}
    for yr in (2024, 2025):
        dy = d2425[d2425["year"] == yr]
        Xy_controls, _, _ = _build_full_X(dy, include_year=False, vocab=vocab)
        Xy_full = np.column_stack([Xy_controls, dy["risk_prob"].to_numpy(dtype=float)])
        yy = dy["brier"].to_numpy()
        rng_y = np.random.default_rng(RNG_SEED + yr)
        ny = len(yy)
        bc = np.empty(N_BOOTSTRAP)
        br2 = np.empty(N_BOOTSTRAP)
        for b in range(N_BOOTSTRAP):
            idx = rng_y.integers(0, ny, size=ny)
            bc[b], br2[b] = coef_and_r2gain(yy[idx], Xy_controls[idx], Xy_full[idx])
        pc, pr2 = coef_and_r2gain(yy, Xy_controls, Xy_full)
        per_year[str(yr)] = {
            "n": int(ny),
            "risk_prob_coefficient": {"point": pc, "ci_lower_2.5": float(np.percentile(bc, 2.5)),
                                      "ci_upper_97.5": float(np.percentile(bc, 97.5))},
            "incremental_r2": {"point": pr2, "ci_lower_2.5": float(np.percentile(br2, 2.5)),
                               "ci_upper_97.5": float(np.percentile(br2, 97.5))},
        }

    sign_2024 = per_year["2024"]["risk_prob_coefficient"]["point"] > 0
    sign_2025 = per_year["2025"]["risk_prob_coefficient"]["point"] > 0
    sign_consistent = sign_2024 == sign_2025
    ci_excludes_zero_both_years = (
        per_year["2024"]["risk_prob_coefficient"]["ci_lower_2.5"] > 0
        and per_year["2025"]["risk_prob_coefficient"]["ci_lower_2.5"] > 0
    )
    pass_full = ci_lo > 0 and ci_excludes_zero_both_years
    return {
        "controls_full_list": FULL_CONT_CONTROLS + FULL_CAT_CONTROLS + ["year"],
        "controls_excluded_note": "odds_change_rate等T-10/T-20由来項目はavailability=falseのため未使用。",
        "n": n,
        "pooled_2024_2025": {
            "risk_prob_coefficient": {"point": point_coef, "ci_lower_2.5": ci_lo, "ci_upper_97.5": ci_hi},
            "incremental_r2": {"point": point_r2, "ci_lower_2.5": float(np.percentile(boot_r2, 2.5)),
                               "ci_upper_97.5": float(np.percentile(boot_r2, 97.5))},
        },
        "per_year": per_year,
        "sign_consistent_across_years": sign_consistent,
        "verdict": "PASS" if pass_full else "FAIL",
        "verdict_note": (
            "PASS条件(本追加検定用に事前固定): pooled係数95%CI下限>0 かつ 2024・2025単年"
            "それぞれの係数95%CI下限>0(=両年でrisk_prob項がゼロを含まない正の効果)。"
        ),
    }


def gate3_breakdown(d2023: pd.DataFrame, d2425: pd.DataFrame, fit_cols: list[str]) -> dict:
    clf, transform, median_brier_2023 = GE._fit_lr_control(d2023, fit_cols)
    d = d2425.copy()
    d["p_high_error_lr"] = clf.predict_proba(transform(d))[:, 1]

    def matched_coverage_compare(sub: pd.DataFrame) -> dict:
        n = len(sub)
        if n < 5:
            return {"n": n, "note": "n<5、算出スキップ"}
        coverage_rate = float(sub["is_bet"].mean())
        n_bet = max(int(round(coverage_rate * n)), 1) if coverage_rate > 0 else 0
        out = {"n": n, "coverage_rate": coverage_rate, "n_bet": n_bet}
        if n_bet == 0 or n_bet >= n:
            out["note"] = "coverage 0%または100%、比較スキップ"
            return out
        jev_mask = sub["is_bet"].to_numpy()
        lr_order = np.argsort(sub["p_high_error_lr"].to_numpy())
        lr_mask = np.zeros(n, dtype=bool)
        lr_mask[lr_order[:n_bet]] = True
        ood_order = np.argsort(-sub["in_distribution_support"].to_numpy())
        ood_mask = np.zeros(n, dtype=bool)
        ood_mask[ood_order[:n_bet]] = True
        for metric in ("brier", "logloss"):
            arr = sub[metric].to_numpy()
            out[metric] = {
                "jev_mean": float(np.mean(arr[jev_mask])) if jev_mask.any() else None,
                "n_jev_bet": int(jev_mask.sum()),
                "lr_control_mean": float(np.mean(arr[lr_mask])),
                "ood_only_mean": float(np.mean(arr[ood_mask])),
            }
        out["small_sample"] = n_bet < 20
        return out

    breakdown = {}
    breakdown["by_year"] = {str(y): matched_coverage_compare(g) for y, g in d.groupby("year")}
    breakdown["by_venue"] = {str(v): matched_coverage_compare(g) for v, g in d.groupby("venue")}
    breakdown["by_popularity_band"] = {str(p): matched_coverage_compare(g) for p, g in d.groupby("popularity_band")}
    breakdown["by_surface"] = {str(s): matched_coverage_compare(g) for s, g in d.groupby("surface")}
    breakdown["by_distance_band"] = {str(b): matched_coverage_compare(g) for b, g in d.groupby("distance_band")}
    d["support_quintile"] = pd.qcut(d["in_distribution_support"], 5,
                                    labels=["q1_low", "q2", "q3", "q4", "q5_high"], duplicates="drop")
    breakdown["by_support_quintile"] = {str(q): matched_coverage_compare(g)
                                        for q, g in d.groupby("support_quintile", observed=True)}
    breakdown["method_note"] = (
        "各区分でJevの実coverage(区分内BET率)に合わせてLR_CONTROL・単純OODゲートの"
        "採用件数を区分内で再選定した(全体coverageの固定閾値ではない)。n_bet<20の区分は"
        "small_sample=trueとして断定しない。overall(全区分プール)のFAIL判定はstage_b_"
        "gate_eval.py(14494c75)のまま変更しない。"
    )
    return breakdown


def main() -> dict:
    d2023, d2425, fit_cols = GE.build_frames()
    gate2_full = gate2_full_controls(d2425)
    gate3_bd = gate3_breakdown(d2023, d2425, fit_cols)
    return {
        "evaluated_at": "2026-09-20", "note": "追加API呼び出しなし。正式なGate1-3判定"
                                              "(stage_b_gate_eval.py)は変更しない。",
        "gate2_full_controls": gate2_full,
        "gate3_breakdown": gate3_bd,
    }


if __name__ == "__main__":
    result = main()
    out_path = HERE / "out" / "STAGE_B_GATE_EVAL_EXTENDED.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[gate_eval_extended] wrote {out_path}")
    print(f"[gate_eval_extended] Gate2_full={result['gate2_full_controls']['verdict']}")
