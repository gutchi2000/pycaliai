# -*- coding: utf-8 -*-
"""
test_stage_b_run.py — stage_b_run.py の純粋ロジック検定 (実API・実データ不要、高速)。
2026-09-20夜、state_schema_v3(unknown_category_rate省略)への訂正を検定する。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.exp06_jev_decision_dev import stage_b_run as SBR  # noqa: E402


def _fake_row():
    return pd.Series({
        "venue": "東京", "surface": "turf", "distance_band": "mile", "class_band": "low",
        "field_size": 16, "m1_top_prob": 0.6, "m3_top_prob": 0.6, "m4_top_prob": 0.6,
        "entropy_m1": 2.2, "entropy_m3": 2.2, "entropy_m4": 2.2,
        "model_rank_disagreement": 0.0, "model_prob_variance": 0.0001,
        "market_prob": 0.55, "market_entropy": 2.2, "ai_market_divergence": 0.05,
        "popularity_band": "1-3", "feature_missing_rate": 0.02,
    })


def test_build_jev_state_omits_unknown_category_rate_key():
    """unknown_category_rateはstateのキーとして一切存在しないこと(0や nullでもない)。"""
    state = SBR.build_jev_state(_fake_row(), in_dist_support=0.5, similar_count=10)
    assert "unknown_category_rate" not in state


def test_build_jev_state_marks_unknown_category_rate_unavailable():
    state = SBR.build_jev_state(_fake_row(), in_dist_support=0.5, similar_count=10)
    assert state["availability"]["unknown_category_rate"] is False
    assert state["availability"]["odds_trajectory"] is False
    assert state["availability"]["ticket_candidates"] is False


def test_build_jev_state_unavailable_fields_are_none_not_zero():
    state = SBR.build_jev_state(_fake_row(), in_dist_support=0.5, similar_count=10)
    for k in ("odds_t20", "odds_t10", "candidate_ticket_count", "ev_point_estimate"):
        assert state[k] is None


def test_build_jev_state_does_not_leak_race_id():
    """rid16やanon_race_id自体はbuild_jev_stateの引数にすら含まれない設計であること
    (呼び出し側=run_batch()がquery_jevへ別引数として渡す、stateには入れない)。"""
    state = SBR.build_jev_state(_fake_row(), in_dist_support=0.5, similar_count=10)
    for v in state.values():
        assert not (isinstance(v, str) and v.startswith("HIST"))


def test_prompt_schema_hash_bumped_from_quarantined_v2():
    assert SBR.PROMPT_SCHEMA_HASH != "exp06_v2_20260920"
    assert SBR.STATE_SCHEMA_VERSION == "v3"


def test_cumulative_tokens_includes_quarantine_dir(tmp_path, monkeypatch):
    """20M上限のカウントに旧schema(隔離済み)の実消費分も含まれること
    (2026-09-20ユーザー指定)。"""
    quarantine = tmp_path / "quarantine_20260920_old_schema" / "jev_cache"
    quarantine.mkdir(parents=True)
    (quarantine / "old1.json").write_text(json.dumps({"usage": {"input_tokens": 1000}}), encoding="utf-8")
    (quarantine / "old2.json").write_text(json.dumps({"usage": {"input_tokens": 2000}}), encoding="utf-8")

    current = tmp_path / "current_cache"
    current.mkdir()
    (current / "new1.json").write_text(json.dumps({"usage": {"input_tokens": 500}}), encoding="utf-8")

    monkeypatch.setattr(SBR, "QUARANTINE_DIR", tmp_path / "quarantine_20260920_old_schema")
    monkeypatch.setattr(SBR.JC, "CACHE_DIR", current)

    total = SBR._cumulative_input_tokens_including_quarantine()
    assert total == 1000 + 2000 + 500


def test_role_separates_development_from_primary_log_paths():
    """2023(development)と2024-2025(primary)が物理的に別ファイルへ書かれること
    (主成績への混入を構造的に防ぐ、2026-09-20ユーザー指定)。"""
    assert SBR.DEVELOPMENT_LOG_PATH != SBR.PRIMARY_LOG_PATH
    assert "2023" in SBR.DEVELOPMENT_LOG_PATH.name
    assert "2024_2025" in SBR.PRIMARY_LOG_PATH.name


def test_anon_id_format_distinguishes_schema_v3():
    """v3のanon_idはv2(HIST_NNNNNN)と衝突しないプレフィックスを使うこと
    (対応表を混同しないための設計確認)。"""
    race_id_map = {}
    reverse_map = {}
    anon = SBR._anon_id_for("2024010101010101", race_id_map, reverse_map)
    assert anon.startswith("HISTV3_")
