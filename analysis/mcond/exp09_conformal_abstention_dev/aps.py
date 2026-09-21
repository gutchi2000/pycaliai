# -*- coding: utf-8 -*-
"""
aps.py — Stage2実装順ステップ2-6: APS(Adaptive Prediction Sets, non-randomized)。

spec.json `aps_definition`/`prediction_set_generation`/`coverage_scope`/
`participation_score_and_tie_break` に固定した定義をそのまま実装する。

nonconformity score: S_i = Σ_{j=1}^{r_i} p_(j)  (確率降順、真の勝ち馬の順位r_iまでの
累積確率質量)。
quantile: k=ceil((n+1)*(1-alpha))、q_hat=calibration score集合のk番目に小さい値
(一般的なpercentile関数は使わない、有限標本補正)。alpha=0.10固定。
k>nはfail-closed(q_hat=1.0)。
non-randomized APS固定(再現性優先)。

実行: 単体実行しない。evaluate.py / stage2_run.py から import して使う。
"""
from __future__ import annotations
import hashlib
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

ALPHA = 0.10
NOMINAL_COVERAGE = 1.0 - ALPHA
TIE_BREAK_SEED = 20260921


def _tansho_probs(scores: np.ndarray) -> np.ndarray:
    w = np.exp(scores - np.max(scores))
    return w / w.sum()


def nonconformity_score(probs: np.ndarray, true_idx: int) -> float:
    """S_i = Σ_{j=1}^{r_i} p_(j)。probsは正規化済み単勝確率ベクトル(降順である
    必要はない、この関数内でソートする)、true_idxはそのベクトル内での
    真の勝ち馬のインデックス。"""
    order = np.argsort(-probs)  # 降順
    rank = int(np.where(order == true_idx)[0][0])  # 0-indexed
    return float(np.sum(probs[order[: rank + 1]]))


@dataclass
class QHatResult:
    q_hat: float
    n: int
    k: int
    fail_closed: bool


def compute_q_hat(calibration_scores: np.ndarray, alpha: float = ALPHA) -> QHatResult:
    """k=ceil((n+1)*(1-alpha))、q_hat=k番目に小さい値(1-indexed)。
    一般的なnp.percentile等は使わない。k>nならfail-closed(q_hat=1.0)。"""
    n = len(calibration_scores)
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return QHatResult(q_hat=1.0, n=n, k=k, fail_closed=True)
    sorted_scores = np.sort(calibration_scores)
    q_hat = float(sorted_scores[k - 1])  # k番目に小さい値(1-indexed→0-indexed)
    return QHatResult(q_hat=q_hat, n=n, k=k, fail_closed=False)


def prediction_set(probs: np.ndarray, q_hat: float) -> np.ndarray:
    """non-randomized APS予測集合: 確率降順に累積し、累積和がq_hatに初めて
    到達(または超える)するところまでの馬のインデックス集合を返す。"""
    order = np.argsort(-probs)
    cum = np.cumsum(probs[order])
    stop = int(np.searchsorted(cum, q_hat, side="left"))
    stop = min(stop, len(probs) - 1)
    return order[: stop + 1]


def fractional_effective_set_size(probs: np.ndarray, q_hat: float) -> float:
    """tie-break第2段階: 予測集合サイズが同じレース間の連続順位付け用。
    境界(最後に追加した馬)がq_hat到達にどれだけ「必要」だったかの割合。
    小さいほど「境界馬への依存が小さい=自信がある」ため参加寄りとみなす。

    定義: 予測集合サイズをmとする。m-1番目まで(境界馬を含まない)の累積確率をC。
    境界馬(m番目)の確率をp_m。fractional = (q_hat - C) / p_m (0-1の範囲、
    境界馬の確率質量のうちq_hat到達に「使われた」割合)。"""
    order = np.argsort(-probs)
    sorted_p = probs[order]
    cum = np.cumsum(sorted_p)
    stop = int(np.searchsorted(cum, q_hat, side="left"))
    stop = min(stop, len(probs) - 1)
    prev_cum = cum[stop - 1] if stop > 0 else 0.0
    boundary_p = sorted_p[stop]
    if boundary_p <= 0:
        return 0.0
    frac = (q_hat - prev_cum) / boundary_p
    return float(np.clip(frac, 0.0, 1.0))


def deterministic_hash_rank(rid16: str, seed: int = TIE_BREAK_SEED) -> int:
    """tie-break第3段階: race_id+固定seedからの決定論的hash(先頭4バイトを
    整数化)。結果ラベルを一切使わない。"""
    h = hashlib.sha256(f"{seed}:{rid16}".encode("utf-8")).digest()
    return int.from_bytes(h[:4], byteorder="big")


def aps_derived_abstention_score(prediction_set_size: int, frac_effective: float,
                                 rid16: str) -> tuple:
    """3段階tie-breakをまとめた比較可能なタプル(昇順ソートで参加優先順になる)。
    "prediction-set sizeだけ"ではなくこのタプル全体がAPS-derived abstention score。"""
    return (prediction_set_size, frac_effective, deterministic_hash_rank(rid16))


def build_calibration_scores(race_frames: dict) -> pd.DataFrame:
    """2023 eligible race集合からnonconformity score S_iを計算する
    (calibration専用、q_hat算出の入力)。"""
    rows = []
    for rid16, g in race_frames.items():
        scores = g["v6_score"].to_numpy(dtype=float)
        probs = _tansho_probs(scores)
        true_idx = int(np.where(g["win"].to_numpy() == 1)[0][0])
        s = nonconformity_score(probs, true_idx)
        rows.append({"rid16": rid16, "nonconformity_score": s})
    return pd.DataFrame(rows)


def build_prediction_sets(race_frames: dict, q_hat: float) -> pd.DataFrame:
    """2024・2025 eligible race集合(2023 calibration setとは独立、q_hatのみ再利用)
    に対し予測集合・prediction_set_size・covered(真の勝ち馬を含むか)・
    APS-derived abstention scoreを計算する。"""
    rows = []
    for rid16, g in race_frames.items():
        scores = g["v6_score"].to_numpy(dtype=float)
        probs = _tansho_probs(scores)
        true_idx = int(np.where(g["win"].to_numpy() == 1)[0][0])
        pset = prediction_set(probs, q_hat)
        size = len(pset)
        covered = bool(true_idx in pset)
        frac = fractional_effective_set_size(probs, q_hat)
        tie_key = aps_derived_abstention_score(size, frac, rid16)
        rows.append({
            "rid16": rid16, "prediction_set_size": size, "covered": covered,
            "frac_effective_set_size": frac, "abstention_score": tie_key,
            "n_field": len(g),
        })
    return pd.DataFrame(rows)


def empirical_coverage(pred_sets: pd.DataFrame, rid16_subset: set | None = None) -> float:
    """指定subset(Noneなら全件)のempirical coverage(真の勝ち馬を含んだ割合)。"""
    df = pred_sets if rid16_subset is None else pred_sets[pred_sets["rid16"].isin(rid16_subset)]
    if len(df) == 0:
        return float("nan")
    return float(df["covered"].mean())
