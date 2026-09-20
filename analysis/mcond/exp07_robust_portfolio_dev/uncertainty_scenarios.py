# -*- coding: utf-8 -*-
"""
uncertainty_scenarios.py — 不確実性集合の生成(仕様書追加指示4、2026-09-20夜)

各馬券確率を独立に上下させない。馬能力(PL latent score)を摂動し、共同着順分布
(build_scenarios.build_top3_states)を毎回まるごと再計算することで、常に券種間で
整合したシナリオ確率を作る(PLの定義上、正の重みベクトルからは必ず有効な確率分布が
得られるため、個別確率を独立にいじって和が壊れる、という失敗モードが構造的に起きない)。

半径(スコア摂動の大きさ)の推定は2023 developmentだけで行い、2024・2025年を見て
調整しない。オッズ側の縮小分布も2023年のTANPUKアーカイブ内で前売り(区分1)→確定
(区分4)の比率を直接実測して使う(2026年のSettleAI実測[n=742、2026-06-07以降]は
developmentより後の期間のデータのため、本モジュールでは使わない)。

実行: このファイルは単体実行を想定しない。test_uncertainty_scenarios.py等から使う。
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
import pl_probs as PL  # noqa: E402
from analysis.mcond.exp07_robust_portfolio_dev import build_scenarios as BS  # noqa: E402

EPS = 1e-9


def estimate_score_noise_scale_from_2023(gate_j1_tansho_df: pd.DataFrame) -> float:
    """2023年developmentのtansho raw PL確率と既存較正済み確率のlogit差を、スコア摂動幅の
    代理として使う(較正が実際に補正した量そのものを、モデルの「もっともらしい揺れ幅」の
    実測値とみなす、2023のみ・結果を見て調整しない)。

    2026-09-20夜、実測により単純な標準偏差(np.std)は極端な穴馬(raw PLが的中率を
    大幅過大評価する低確率帯、Gate J1参照)のlogit差の外れ値に支配され2.20という
    非現実的に大きい値になると判明したため、**IQRベースの頑健な尺度推定量
    (IQR/1.349、正規分布での標準偏差との対応)を正式採用**する(この決定自体も
    2023データのみに基づき、2024/2025を見る前に確定した)。"""
    raw_p = np.clip(gate_j1_tansho_df["raw_p"].to_numpy(), EPS, 1 - EPS)
    cal_p = np.clip(gate_j1_tansho_df["cal_p"].to_numpy(), EPS, 1 - EPS)
    logit_raw = np.log(raw_p / (1 - raw_p))
    logit_cal = np.log(cal_p / (1 - cal_p))
    residual = logit_cal - logit_raw
    q75, q25 = np.percentile(residual, [75, 25])
    iqr = q75 - q25
    return float(iqr / 1.349) if iqr > 0 else float(np.std(residual))


def estimate_odds_drift_distribution_from_2023(tanpuk_2023: pd.DataFrame) -> dict:
    """TANPUKアーカイブ内の同一レース・同一馬について、区分1(前売り、historical_pre_snapshot
    に相当する最終スナップショット)と区分4(確定)の単勝オッズ比[確定/前売り]の分布を
    2023年のみで実測する。呼び出し側は tanpuk_2023 に区分1/4混在の生データを渡す
    (レースID, 区分, 月日時分, 頭数, 単勝票数, 複勝票数, 1単..18単 の列を持つDataFrame)。"""
    odds_cols = [c for c in tanpuk_2023.columns if c.endswith("単") and c != "単勝票数"]
    pre = tanpuk_2023[tanpuk_2023["区分"] == 1].sort_values("月日時分").groupby("レースID").last()
    fin = tanpuk_2023[tanpuk_2023["区分"] == 4].groupby("レースID").last()
    common = pre.index.intersection(fin.index)
    ratios = []
    for rid in common:
        for c in odds_cols:
            pre_o, fin_o = pre.loc[rid, c], fin.loc[rid, c]
            if pd.notna(pre_o) and pd.notna(fin_o) and pre_o > 0 and fin_o > 0:
                ratios.append(fin_o / pre_o)
    ratios = np.array(ratios, dtype=float)
    ratios = ratios[(ratios > 0) & np.isfinite(ratios)]
    return {
        "n": int(len(ratios)),
        "mean_ratio": float(np.mean(ratios)) if len(ratios) else None,
        "p10": float(np.percentile(ratios, 10)) if len(ratios) else None,
        "p50": float(np.percentile(ratios, 50)) if len(ratios) else None,
        "p90": float(np.percentile(ratios, 90)) if len(ratios) else None,
        "worst_case_shrinkage_p10": float(np.percentile(ratios, 10)) if len(ratios) else None,
    }


def generate_score_perturbation_scenarios(
    scores: np.ndarray, *, n_draws: int, sigma: float, seed: int,
) -> list[np.ndarray]:
    """スコアへi.i.d.ガウス摂動を加えたn_draws個のドローを返す(重みではなくスコアへ
    加えるため、正値制約を気にせず、その後 pl_weights() で正の重みへ変換する)。"""
    rng = np.random.default_rng(seed)
    return [scores + rng.normal(0.0, sigma, size=scores.shape) for _ in range(n_draws)]


def build_uncertainty_state_probability_scenarios(
    scores: np.ndarray, *, n_draws: int, sigma: float, seed: int,
) -> dict:
    """各drawで共同着順分布をまるごと再計算し、Gate J0を再確認した上で
    state_probability_scenarios(state_payoffsと同じ状態順序を共有する確率ベクトルのリスト)
    を返す。全drawが同じstates順序(permutations(range(n),3))を共有するため、
    build_scenarios.state_payoffs_for_ticket() と組み合わせてそのまま
    robust_ticket_portfolio.optimise_portfolio() の state_probability_scenarios へ渡せる。"""
    base_states, base_probs = BS.build_top3_states(PL.pl_weights(scores))
    draws = generate_score_perturbation_scenarios(scores, n_draws=n_draws, sigma=sigma, seed=seed)

    scenarios = []
    gate_j0_results = []
    for perturbed_scores in draws:
        w = PL.pl_weights(perturbed_scores)
        states, probs = BS.build_top3_states(w)
        assert states == base_states, "摂動後も状態の列挙順は不変であるべき(出走頭数不変のため)"
        gate_j0 = BS.gate_j0_checks(w)
        gate_j0_results.append(gate_j0["overall_pass"])
        scenarios.append(probs)

    return {
        "base_states": base_states,
        "base_probabilities": base_probs,
        "scenarios": scenarios,
        "n_draws": n_draws, "sigma": sigma,
        "all_draws_passed_gate_j0": bool(all(gate_j0_results)),
        "n_gate_j0_failures": int(sum(1 for r in gate_j0_results if not r)),
    }
