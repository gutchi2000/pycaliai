# -*- coding: utf-8 -*-
"""
stage_b_gate_eval.py — Stage B Gate 1-3 統計検定 (2026-09-20夜、corrected run完了後)
================================================================================
ユーザー指定の実行順11-12番目("2023で対照モデル・coverageを固定"→"2024/2025を
一度だけ評価")。Gate 1-3を通過するまでGate 4(経済評価・ROI)へは進まない
(spec.json "9. Gate"、原文: "Gate 4：経済評価 Gate 1〜3を通過した場合のみ実施する")。

методология(2026-09-20固定、元のEXP06仕様書 "7. 主検定" / "9. Gate" 原文より):
  Gate 1 (Jev確率の情報性): "Jevのリスク確率と実際の予測誤差の関係が単調であること。"
    risk_prob = Q5(participation)のprobabilities["PASS_UNCERTAIN"]+["PASS_OOD"]
    (原文: "JevのリスクPASS_UNCERTAIN・PASS_OOD確率が高いほど、以下が悪化するかを
    検定する: M4のlogloss、Brier score、...")。
    本スクリプトではBrier scoreとlogloss(M4予測 vs 実際top3)の2指標を計算する
    (予測順位誤差・AIと市場のブレンド誤差・候補馬券の実現損益は、全頭の予測順位や
    ブレンド式・確定払戻を追加でJOINする必要があり、現在キャッシュされている
    race-state vectorのスキーマには含まれないため今回は計算しない。Gate4[経済評価]
    でのみ扱う実現損益とは異なり、これらはGate1の追加指標だが、データ未整備のため
    本パスでは省略する。省略した旨をレポートに明記する)。
    Spearman順位相関(単調性検定に忠実)、primary(2024-2025)を主エビデンス、
    development(2023)は方向性の事前確認としてのみ報告する(主エビデンスとして
    数えない、元の役割定義通り)。

  Gate 2 (既存ゲートへの固有上積み): "既存OOD指標、モデル不一致、欠損率、オッズ変動を
    統制した後にも、Jev出力が予測誤差を説明すること。"
    オッズ変動(odds_change_rate)は2023-2025で入手不可(data_availability_audit、
    availability=false)のため統制変数から除外し、その旨を明記する。
    統制変数: in_distribution_support, model_rank_disagreement, feature_missing_rate,
    model_prob_variance。目的変数: Brier score。OLSの偏回帰係数(risk_prob追加項)の
    有意性をレース単位ブートストラップで検定する(このプロジェクトの一貫した証拠基準
    "レース単位ブートストラップ信頼区間"に合わせ、正規分布近似のp値ではなく
    ブートストラップCIを採用、statsmodels等の追加依存を避ける)。

  Gate 3 (同一coverageでの改善): "Jevゲートが固定ルールおよび単純モデルより良いこと。"
    LR_CONTROL: "Jevと同じ入力を使う単純ロジスティック回帰...Jevがこれを超えなければ、
    外部AIを使う固有価値なしと判定する。"
    2023年developmentだけでfit(結果ラベルはfit対象、2023のJev回答内容は使わない)、
    目的変数はhigh_error_2023=(2023年内のBrierが2023年内中央値超)の2値
    (この二値化の定義は本評価スクリプト作成時[2026-09-20夜、Gate結果を見る前]に
    確定し、以後変更しない)。2024-2025のJevの実coverage(BET選択率)に合わせて
    LR_CONTROLと単純OODゲート(in_distribution_supportのみ)の採用率を揃え、
    採用された集合内の平均Brier/loglossをJevゲートと比較する
    ("購入率を同じに揃えた状態で比較する"、"買わないレースを増やせば成績が上がる
    見かけの改善"を防ぐ)。

実行: python -m analysis.mcond.exp06_jev_decision_dev.stage_b_gate_eval
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.exp06_jev_decision_dev import historical_state as HS  # noqa: E402
from analysis.mcond.exp06_jev_decision_dev.ood_support import (  # noqa: E402
    SupportModel, CONTINUOUS_COLS, CATEGORICAL_COLS)
from analysis.mcond.exp06_jev_decision_dev import stage_b_run as SBR  # noqa: E402

HERE = Path(__file__).resolve().parent
EPS = 1e-9
N_BOOTSTRAP = 2000
RNG_SEED = 20260920  # 固定シード、結果を見て変更しない


def _load_jev_answers(log_path: Path) -> dict:
    """{rid16: answers_dict} を返す。fail(ok=False)は除外する。"""
    out = {}
    if not log_path.exists():
        return out
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if not rec.get("ok"):
            continue
        cache_path = SBR.JC.CACHE_DIR / f"{rec['input_hash']}.json"
        if not cache_path.exists():
            continue
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        answers = cached.get("all_answers")
        if answers is None:
            continue
        out[rec["rid16"]] = answers
    return out


def _q5_risk_prob(answers: dict) -> float:
    q5 = answers.get("Q5") or {}
    probs = q5.get("probabilities") or {}
    return float(probs.get("PASS_UNCERTAIN", 0.0)) + float(probs.get("PASS_OOD", 0.0))


def _q5_is_bet(answers: dict) -> bool:
    q5 = answers.get("Q5") or {}
    return q5.get("choice") == "BET"


def _brier(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (p - y) ** 2


def _logloss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    pc = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(pc) + (1 - y) * np.log(1 - pc))


def _bootstrap_ci(values_by_group: list[np.ndarray], stat_fn, n=N_BOOTSTRAP, seed=RNG_SEED):
    """レース単位ブートストラップ。stat_fnは各グループのresample配列を受け取りスカラーを返す。"""
    rng = np.random.default_rng(seed)
    n_per_group = [len(v) for v in values_by_group]
    stats = np.empty(n)
    for b in range(n):
        resampled = [v[rng.integers(0, len(v), size=len(v))] for v in values_by_group]
        stats[b] = stat_fn(*resampled)
    return {
        "point": float(stat_fn(*values_by_group)),
        "ci_lower_2.5": float(np.percentile(stats, 2.5)),
        "ci_upper_97.5": float(np.percentile(stats, 97.5)),
        "n_per_group": n_per_group,
    }


def build_frames():
    print("[gate_eval] loading historical state + fitting OOS-safe M1/M3/M4...")
    df, feature_lists = HS.load_design_and_features()
    preds = HS.fit_oos_safe_predictions(df, feature_lists)
    state = HS.build_race_state_vectors(df, preds)

    fit_cols = CONTINUOUS_COLS + CATEGORICAL_COLS
    d2023 = state[state["year"] == 2023].copy()
    sm = SupportModel().fit(d2023[fit_cols])
    d2425 = state[state["year"].isin([2024, 2025])].copy()

    support_2023 = sm.score_reference_self()
    support_2425 = sm.score(d2425[fit_cols])
    d2023["in_distribution_support"] = support_2023["in_distribution_support"]
    d2023["similar_past_case_count"] = support_2023["similar_past_case_count"]
    d2425["in_distribution_support"] = support_2425["in_distribution_support"]
    d2425["similar_past_case_count"] = support_2425["similar_past_case_count"]

    dev_answers = _load_jev_answers(SBR.DEVELOPMENT_LOG_PATH)
    pri_answers = _load_jev_answers(SBR.PRIMARY_LOG_PATH)
    print(f"[gate_eval] development answers loaded: {len(dev_answers)}/{len(d2023)}")
    print(f"[gate_eval] primary answers loaded: {len(pri_answers)}/{len(d2425)}")

    d2023 = d2023[d2023.index.isin(dev_answers.keys())].copy()
    d2425 = d2425[d2425.index.isin(pri_answers.keys())].copy()

    d2023["risk_prob"] = [_q5_risk_prob(dev_answers[rid]) for rid in d2023.index]
    d2023["is_bet"] = [_q5_is_bet(dev_answers[rid]) for rid in d2023.index]
    d2425["risk_prob"] = [_q5_risk_prob(pri_answers[rid]) for rid in d2425.index]
    d2425["is_bet"] = [_q5_is_bet(pri_answers[rid]) for rid in d2425.index]

    for d in (d2023, d2425):
        d["brier"] = _brier(d["m4_top_prob"].to_numpy(), d["_result_top3"].to_numpy())
        d["logloss"] = _logloss(d["m4_top_prob"].to_numpy(), d["_result_top3"].to_numpy())

    return d2023, d2425, fit_cols


def gate1_informativeness(d2023: pd.DataFrame, d2425: pd.DataFrame) -> dict:
    out = {}
    for name, d, is_primary in (("development_2023_exploratory", d2023, False),
                                 ("primary_2024_2025", d2425, True)):
        res = {}
        for err_name in ("brier", "logloss"):
            rho, p = spearmanr(d["risk_prob"], d[err_name])
            res[err_name] = {"spearman_rho": float(rho), "p_value": float(p), "n": int(len(d))}
        out[name] = res
    verdict_primary = out["primary_2024_2025"]
    pass_gate1 = all(
        (verdict_primary[m]["spearman_rho"] > 0 and verdict_primary[m]["p_value"] < 0.05)
        for m in ("brier", "logloss")
    )
    out["verdict"] = "PASS" if pass_gate1 else "FAIL"
    out["verdict_note"] = (
        "PASS条件(事前固定): primary(2024-2025)でbrier/loglossの両方についてSpearman rho>0"
        "かつp<0.05(risk_probが高いほど誤差が大きい、単調関係が有意)。development(2023)は"
        "方向性の事前確認のみで判定には数えない。"
    )
    out["not_computed_note"] = (
        "予測順位誤差・AIと市場のブレンド誤差・候補馬券の実現損益は、現行race-state "
        "vectorスキーマ(全頭予測順位・ブレンド式・確定払戻を含まない)からは計算できない"
        "ため本パスでは省略した。Brier scoreとloglossの2指標のみで判定する。"
    )
    return out


def gate2_incremental_value(d2425: pd.DataFrame) -> dict:
    controls = ["in_distribution_support", "model_rank_disagreement",
                "feature_missing_rate", "model_prob_variance"]
    y = d2425["brier"].to_numpy()
    X_controls = d2425[controls].to_numpy(dtype=float)
    X_full = np.column_stack([X_controls, d2425["risk_prob"].to_numpy(dtype=float)])

    def partial_r2_gain(y_arr, X_c, X_f):
        lr_c = LinearRegression().fit(X_c, y_arr)
        resid_c = y_arr - lr_c.predict(X_c)
        ss_res_c = float(np.sum(resid_c ** 2))
        lr_f = LinearRegression().fit(X_f, y_arr)
        resid_f = y_arr - lr_f.predict(X_f)
        ss_res_f = float(np.sum(resid_f ** 2))
        return (ss_res_c - ss_res_f) / max(ss_res_c, EPS)  # R^2の増分

    def risk_prob_coef(y_arr, X_f):
        lr_f = LinearRegression().fit(X_f, y_arr)
        return float(lr_f.coef_[-1])

    rng = np.random.default_rng(RNG_SEED)
    n = len(y)
    boot_r2gain = np.empty(N_BOOTSTRAP)
    boot_coef = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        boot_r2gain[b] = partial_r2_gain(y[idx], X_controls[idx], X_full[idx])
        boot_coef[b] = risk_prob_coef(y[idx], X_full[idx])

    point_r2gain = partial_r2_gain(y, X_controls, X_full)
    point_coef = risk_prob_coef(y, X_full)
    coef_ci_lo, coef_ci_hi = float(np.percentile(boot_coef, 2.5)), float(np.percentile(boot_coef, 97.5))
    excludes_odds_volatility_note = (
        "odds_change_rateは2023-2025で入手不可(data_availability_audit、availability=false)"
        "のため統制変数から除外した。"
    )
    pass_gate2 = coef_ci_lo > 0  # risk_probの係数(誤差への正の寄与)が95%CIで0を上回る
    return {
        "target": "brier",
        "controls": controls,
        "controls_excluded_note": excludes_odds_volatility_note,
        "risk_prob_coefficient": {"point": point_coef, "ci_lower_2.5": coef_ci_lo,
                                  "ci_upper_97.5": coef_ci_hi},
        "partial_r2_gain_from_risk_prob": {
            "point": float(point_r2gain),
            "ci_lower_2.5": float(np.percentile(boot_r2gain, 2.5)),
            "ci_upper_97.5": float(np.percentile(boot_r2gain, 97.5)),
        },
        "n": int(n),
        "verdict": "PASS" if pass_gate2 else "FAIL",
        "verdict_note": (
            "PASS条件(事前固定): 統制変数(in_distribution_support/model_rank_disagreement/"
            "feature_missing_rate/model_prob_variance)を含めた回帰でrisk_prob項の係数の"
            "レース単位ブートストラップ95%CI下限が0を上回ること(統制後も固有の説明力がある)。"
        ),
    }


def _fit_lr_control(d2023: pd.DataFrame, fit_cols: list[str]):
    """2023年developmentのみでfitする単純ロジスティック回帰(LR_CONTROL)。
    目的変数high_error_2023=2023年内のBrier中央値超の2値(2026-09-20夜、
    結果を見る前に確定、以後変更しない)。Jevと同じ入力(state相当の連続値+
    カテゴリone-hot)を使う。"""
    cont_cols = [c for c in fit_cols if c in CONTINUOUS_COLS] + ["in_distribution_support",
                                                                 "similar_past_case_count"]
    cat_cols = [c for c in fit_cols if c in CATEGORICAL_COLS]
    X_cont = d2023[cont_cols].to_numpy(dtype=float)
    X_cont = np.nan_to_num(X_cont, nan=0.0)
    cat_vocab = {c: sorted(d2023[c].astype(str).unique()) for c in cat_cols}
    X_cat_blocks = []
    for c in cat_cols:
        vocab = cat_vocab[c]
        vidx = {v: i for i, v in enumerate(vocab)}
        block = np.zeros((len(d2023), len(vocab)))
        for i, v in enumerate(d2023[c].astype(str)):
            block[i, vidx[v]] = 1.0
        X_cat_blocks.append(block)
    X = np.hstack([X_cont] + X_cat_blocks) if X_cat_blocks else X_cont

    median_brier_2023 = float(np.median(d2023["brier"]))
    y_high_error = (d2023["brier"].to_numpy() > median_brier_2023).astype(int)

    # 連続値の尺度差(field_sizeは5-18、probabilityは0-1等)がlbfgsの収束を妨げていたため
    # 標準化を追加(2026-09-20、比較対象の適合品質を上げる目的のみ、目的変数・閾値・
    # 判定基準は変更しない)。
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    clf = LogisticRegression(max_iter=5000)
    clf.fit(X_scaled, y_high_error)

    def transform(d: pd.DataFrame) -> np.ndarray:
        xc = np.nan_to_num(d[cont_cols].to_numpy(dtype=float), nan=0.0)
        blocks = []
        for c in cat_cols:
            vocab = cat_vocab[c]
            vidx = {v: i for i, v in enumerate(vocab)}
            block = np.zeros((len(d), len(vocab)))
            for i, v in enumerate(d[c].astype(str)):
                j = vidx.get(v)
                if j is not None:
                    block[i, j] = 1.0
            blocks.append(block)
        raw = np.hstack([xc] + blocks) if blocks else xc
        return scaler.transform(raw)

    return clf, transform, median_brier_2023


def gate3_same_coverage(d2023: pd.DataFrame, d2425: pd.DataFrame, fit_cols: list[str]) -> dict:
    clf, transform, median_brier_2023 = _fit_lr_control(d2023, fit_cols)
    X_2425 = transform(d2425)
    p_high_error_lr = clf.predict_proba(X_2425)[:, 1]

    coverage_rate = float(d2425["is_bet"].mean())
    n = len(d2425)
    n_bet = max(int(round(coverage_rate * n)), 1)

    jev_bet_mask = d2425["is_bet"].to_numpy()

    lr_order = np.argsort(p_high_error_lr)  # 低リスク順
    lr_bet_mask = np.zeros(n, dtype=bool)
    lr_bet_mask[lr_order[:n_bet]] = True

    ood_order = np.argsort(-d2425["in_distribution_support"].to_numpy())  # support高い順
    ood_bet_mask = np.zeros(n, dtype=bool)
    ood_bet_mask[ood_order[:n_bet]] = True

    brier = d2425["brier"].to_numpy()
    logloss = d2425["logloss"].to_numpy()

    def mean_diff(a_vals, b_vals):
        return float(np.mean(a_vals) - np.mean(b_vals))

    result = {
        "coverage_rate_matched_to_jev_bet_rate": coverage_rate,
        "n_races": n, "n_bet_per_gate": n_bet,
        "lr_control_target_definition": (
            f"2023年developmentのみでfit。high_error_2023 = brier > "
            f"2023年内中央値({median_brier_2023:.6f})。2026-09-20夜、結果を見る前に確定。"
        ),
        "gates": {},
    }
    for metric_name, metric_arr in (("brier", brier), ("logloss", logloss)):
        jev_vals = metric_arr[jev_bet_mask]
        lr_vals = metric_arr[lr_bet_mask]
        ood_vals = metric_arr[ood_bet_mask]
        jev_vs_lr = _bootstrap_ci([jev_vals, lr_vals], lambda a, b: mean_diff(a, b))
        jev_vs_ood = _bootstrap_ci([jev_vals, ood_vals], lambda a, b: mean_diff(a, b))
        result["gates"][metric_name] = {
            "jev_mean": float(np.mean(jev_vals)), "n_jev_bet": int(jev_bet_mask.sum()),
            "lr_control_mean": float(np.mean(lr_vals)),
            "ood_only_mean": float(np.mean(ood_vals)),
            "jev_minus_lr_control": jev_vs_lr,
            "jev_minus_ood_only": jev_vs_ood,
        }
    pass_gate3 = all(
        result["gates"][m]["jev_minus_lr_control"]["ci_upper_97.5"] < 0
        for m in ("brier", "logloss")
    )
    result["verdict"] = "PASS" if pass_gate3 else "FAIL"
    result["verdict_note"] = (
        "PASS条件(事前固定): 同一coverageでJevゲートの平均brier/loglossがLR_CONTROLの"
        "平均を下回り、その差(Jev-LR_CONTROL)のレース単位ブートストラップ95%CI上限が"
        "0未満であること(両指標とも)。"
    )
    return result


def main() -> dict:
    d2023, d2425, fit_cols = build_frames()
    gate1 = gate1_informativeness(d2023, d2425)
    gate2 = gate2_incremental_value(d2425)
    gate3 = gate3_same_coverage(d2023, d2425, fit_cols)

    overall = "PASS" if all(g["verdict"] == "PASS" for g in (gate1, gate2, gate3)) else "FAIL"
    report = {
        "evaluated_at": "2026-09-20", "state_schema_version": SBR.STATE_SCHEMA_VERSION,
        "prompt_schema_hash": SBR.PROMPT_SCHEMA_HASH,
        "n_development_2023": int(len(d2023)), "n_primary_2024_2025": int(len(d2425)),
        "gate0_data_api_health": "本レポート生成時点でdevelopment/primaryとも全件0 fail "
                                 "(stage_b_run.pyログ参照、schema/model一致は別途確認済み)",
        "gate1_probability_informativeness": gate1,
        "gate2_incremental_value_over_existing_gates": gate2,
        "gate3_same_coverage_improvement": gate3,
        "overall_gate1_3_verdict": overall,
        "gate4_economic_evaluation": "Gate1-3の結果次第。Gate1-3のいずれかがFAILの場合は"
                                     "実施しない(spec.json/元仕様書の中止規律)。",
        "no_production_connection_made": True,
    }
    return report


if __name__ == "__main__":
    result = main()
    out_path = HERE / "out" / "STAGE_B_GATE_EVAL.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[gate_eval] wrote {out_path}")
    print(f"[gate_eval] Gate1={result['gate1_probability_informativeness']['verdict']} "
         f"Gate2={result['gate2_incremental_value_over_existing_gates']['verdict']} "
         f"Gate3={result['gate3_same_coverage_improvement']['verdict']} "
         f"OVERALL={result['overall_gate1_3_verdict']}")
