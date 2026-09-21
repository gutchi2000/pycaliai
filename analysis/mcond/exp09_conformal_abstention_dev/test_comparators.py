# -*- coding: utf-8 -*-
"""
test_comparators.py — comparators.pyの合成テスト。実データ(2024-2025)は使わない
(方式5/7は本番の実artifact(モデル/calibrator)を使うが、これらは構造的な
モデル成果物であり2024-2025年の性能・結果データではない)。
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.exp09_conformal_abstention_dev.comparators import (  # noqa: E402
    score_max_probability, score_entropy, fit_ood_model, score_ood_support,
    fit_lr_control, score_lr_control, _race_level_stats, score_feature_missing,
)


def _mk_race(rid16, scores, winner_idx):
    n = len(scores)
    fin = [2] * n
    fin[winner_idx] = 1
    win = [0] * n
    win[winner_idx] = 1
    return rid16, pd.DataFrame({"ban": range(1, n + 1), "v6_score": scores, "fin": fin, "win": win})


def _mk_frames(n_races=40, seed=0):
    rng = np.random.default_rng(seed)
    frames = {}
    for i in range(n_races):
        n = rng.integers(5, 12)
        scores = rng.normal(size=n)
        winner = int(np.argmax(scores)) if rng.random() < 0.6 else int(rng.integers(0, n))
        rid, df = _mk_race(f"r{i}", scores, winner)
        frames[rid] = df
    return frames


def test_race_level_stats_entropy_bounds():
    frames = _mk_frames(20, seed=1)
    stats = _race_level_stats(frames)
    assert (stats["entropy"] >= 0).all()
    assert (stats["entropy"] <= 1.0001).all()
    assert (stats["max_prob"] > 0).all()
    assert (stats["max_prob"] <= 1.0001).all()


def test_max_probability_score_sign_convention():
    """max_prob score = -max(prob)。確信度が高い(max_probが大きい)ほど
    scoreは小さい(参加寄り)。"""
    frames = _mk_frames(30, seed=2)
    out = score_max_probability(frames)
    stats = _race_level_stats(frames)
    merged = out.merge(stats, on="rid16")
    # scoreの順位とmax_probの順位が完全に逆であること
    assert (merged["score"].rank().to_numpy() == merged["max_prob"].rank(ascending=False).to_numpy()).all()


def test_entropy_score_matches_direct_computation():
    frames = _mk_frames(15, seed=3)
    out = score_entropy(frames)
    stats = _race_level_stats(frames)
    merged = out.merge(stats, on="rid16")
    np.testing.assert_allclose(merged["score"].to_numpy(), merged["entropy"].to_numpy())


def test_ood_model_fit_and_score_shapes():
    train_frames = _mk_frames(60, seed=4)
    model = fit_ood_model(train_frames)
    eval_frames = _mk_frames(20, seed=5)
    out = score_ood_support(model, eval_frames)
    assert len(out) == 20
    assert out["score"].notna().all()


def test_ood_model_query_similar_to_reference_gets_lower_score():
    """訓練データに極めて近い(同じ分布からサンプルした)クエリは、訓練データから
    遠いクエリよりも高いin_distribution_support(=低いscore)を持つ傾向がある
    ことの緩やかな確認(統計的性質、個別レース単位の厳密保証ではない)。"""
    rng = np.random.default_rng(6)
    train_frames = {}
    for i in range(80):
        n = 8
        scores = rng.normal(loc=0, scale=1, size=n)
        winner = int(np.argmax(scores))
        rid, df = _mk_race(f"tr{i}", scores, winner)
        train_frames[rid] = df
    model = fit_ood_model(train_frames)

    similar_frames = {}
    for i in range(20):
        scores = rng.normal(loc=0, scale=1, size=8)
        rid, df = _mk_race(f"sim{i}", scores, int(np.argmax(scores)))
        similar_frames[rid] = df
    far_frames = {}
    for i in range(20):
        scores = rng.normal(loc=20, scale=5, size=8)  # 大きく外れた分布
        rid, df = _mk_race(f"far{i}", scores, int(np.argmax(scores)))
        far_frames[rid] = df

    sim_out = score_ood_support(model, similar_frames)
    far_out = score_ood_support(model, far_frames)
    assert sim_out["score"].mean() < far_out["score"].mean()


def test_lr_control_fit_and_score():
    train_frames = _mk_frames(100, seed=7)
    clf, scaler = fit_lr_control(train_frames)
    eval_frames = _mk_frames(30, seed=8)
    out = score_lr_control(clf, scaler, eval_frames)
    assert len(out) == 30
    assert ((out["score"] >= 0) & (out["score"] <= 1)).all()


def test_score_feature_missing_year_filter_regression():
    """バグ回帰テスト: score_feature_missing([2023])が2023年以外のレースを
    含まないこと(2026-09-21実装中に発覚、yearsフィルタが未適用で全年
    (n≈44,907レース)が混入していた実害バグ)。"""
    out2023 = score_feature_missing([2023])
    n_2023 = len(out2023)
    # 2023年のレース数はEXP07で確立済みの実測(n=3,347前後)を大幅に下回らない
    # ことと、全年合計(約44,907)より十分小さいことの両方を確認する
    assert 2000 < n_2023 < 5000, f"score_feature_missing([2023])のレース数が異常: {n_2023}"
    all_rid16_years = {r[:4] for r in out2023["rid16"]}
    assert all_rid16_years == {"2023"}, f"2023年以外が混入: {all_rid16_years}"


def test_lr_control_target_definition_uses_no_future_data():
    """LR_CONTROLのfit関数が2023(train_frames)のwin/fin列だけを使い、
    別途渡されたeval_framesを一切参照しないことの構造的確認。"""
    import inspect
    from analysis.mcond.exp09_conformal_abstention_dev.comparators import fit_lr_control
    sig = inspect.signature(fit_lr_control)
    assert list(sig.parameters.keys()) == ["race_frames_2023"]
