# -*- coding: utf-8 -*-
"""
test_stage_b_sample.py — 層化抽出ロジックの検定 (実データ・実モデルfit不要、高速)。
実データでの実行結果(200レース、感度分析用)はSTAGE_B_SENSITIVITY_SAMPLE.jsonを参照
(2026-09-20に一度だけ生成、以後再抽出しない)。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.exp06_jev_decision_dev import stage_b_sample as SBS  # noqa: E402


def _synthetic_population(n=2000, seed=0):
    rng = pd.Series(range(n))
    import numpy as np
    r = np.random.default_rng(seed)
    return pd.DataFrame({
        "year": r.choice([2024, 2025], size=n),
        "surface": r.choice(["turf", "dirt"], size=n),
        "venue": r.choice(["A", "B", "C"], size=n, p=[0.6, 0.3, 0.1]),
    }, index=[f"race_{i}" for i in range(n)])


def test_proportional_allocation_sums_to_requested_n():
    df = _synthetic_population()
    idx = SBS.proportional_stratified_sample(df, ["year", "surface", "venue"], 200, seed=1)
    assert len(idx) == 200
    assert len(set(idx)) == 200  # 重複なし(非復元抽出)


def test_proportional_allocation_reflects_population_share():
    """母集団比の大きいストラタム(venue=A, 60%)からより多く抽出されること。"""
    df = _synthetic_population()
    idx = SBS.proportional_stratified_sample(df, ["venue"], 200, seed=1)
    sampled = df.loc[idx]
    counts = sampled["venue"].value_counts()
    assert counts["A"] > counts["B"] > counts["C"]


def test_deterministic_given_fixed_seed():
    df = _synthetic_population()
    idx1 = SBS.proportional_stratified_sample(df, ["year", "surface"], 100, seed=42)
    idx2 = SBS.proportional_stratified_sample(df, ["year", "surface"], 100, seed=42)
    assert list(idx1) == list(idx2)


def test_load_or_build_sample_does_not_regenerate_existing_file(tmp_path, monkeypatch):
    """既存のfrozenファイルがあれば再抽出せずそれを読むだけであること
    (結果ラベルを見て後から選び直す、という抜け道を防ぐ)。"""
    frozen = {"created_at": "2020-01-01", "race_ids": ["dummy1", "dummy2"], "n_sample": 2}
    p = tmp_path / "frozen_sample.json"
    p.write_text(json.dumps(frozen), encoding="utf-8")
    monkeypatch.setattr(SBS, "SAMPLE_PATH", p)

    def fail_if_called():
        raise AssertionError("build_sample() should not be called when a frozen file exists")
    monkeypatch.setattr(SBS, "build_sample", fail_if_called)

    result = SBS.load_or_build_sample()
    assert result["race_ids"] == ["dummy1", "dummy2"]


def test_random_seed_is_fixed_constant():
    assert SBS.RANDOM_SEED == 20260920


def test_n_sample_matches_user_specified_200():
    assert SBS.N_SAMPLE == 200
