# -*- coding: utf-8 -*-
"""
test_jev_client.py — jev_client.py の純粋ロジック部分のテスト(実APIコール不要)。
実行: python -m pytest analysis/mcond/exp06_jev_decision_dev/test_jev_client.py -q
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.exp06_jev_decision_dev import jev_client as JC  # noqa: E402


def test_compute_input_hash_deterministic():
    state = {"a": 1, "b": 2}
    h1 = JC.compute_input_hash("schema1", state)
    h2 = JC.compute_input_hash("schema1", state)
    assert h1 == h2
    assert len(h1) == 16


def test_compute_input_hash_differs_on_state_change():
    h1 = JC.compute_input_hash("schema1", {"a": 1})
    h2 = JC.compute_input_hash("schema1", {"a": 2})
    assert h1 != h2


def test_compute_input_hash_differs_on_schema_change():
    h1 = JC.compute_input_hash("schema1", {"a": 1})
    h2 = JC.compute_input_hash("schema2", {"a": 1})
    assert h1 != h2


def test_query_jev_missing_api_key_never_leaks_key_value(monkeypatch, tmp_path):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(JC, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(JC, "RESPONSES_DIR", tmp_path / "responses")
    result = JC.query_jev("R1", "schema1", {"x": 1}, [], "modelhash", "2099-01-01T00:00:00")
    assert result["ok"] is False
    assert "TYPESAFE_API_KEY" in result["error"]
    # キーの値自体(未設定なので値は無いが、念のためNoneや空文字列以外の
    # 秘密情報が紛れ込んでいないことを確認する意図)
    assert "api_key" not in result
    assert "key" not in json.dumps(result).lower().replace("typesafe_api_key", "")


def test_query_jev_uses_cache_on_second_call(monkeypatch, tmp_path):
    monkeypatch.setenv("TYPESAFE_API_KEY", "dummy-for-test-not-real")
    monkeypatch.setattr(JC, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(JC, "RESPONSES_DIR", tmp_path / "responses")

    call_count = {"n": 0}

    def fake_call(prompt_schema_hash, state, questions):
        call_count["n"] += 1
        return {"model": "jev-test", "answers": {"Q1": "yes"}, "probabilities": {"Q1": {"yes": 1.0}},
               "confidence": 0.9, "usage": {"tokens": 10}, "http_status": 200}

    monkeypatch.setattr(JC, "_call_jev_api_raw", fake_call)

    r1 = JC.query_jev("R1", "schema1", {"x": 1}, [], "modelhash", "2099-01-01T00:00:00")
    assert r1["ok"] is True
    assert r1["from_cache"] is False
    assert call_count["n"] == 1

    r2 = JC.query_jev("R1", "schema1", {"x": 1}, [], "modelhash", "2099-01-01T00:00:00")
    assert r2["from_cache"] is True
    assert call_count["n"] == 1  # 2回目はAPIを叩かない(再課金防止)


def test_query_jev_append_only_response_log(monkeypatch, tmp_path):
    monkeypatch.setenv("TYPESAFE_API_KEY", "dummy-for-test-not-real")
    monkeypatch.setattr(JC, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(JC, "RESPONSES_DIR", tmp_path / "responses")

    def fake_call(prompt_schema_hash, state, questions):
        return {"model": "jev-test", "answers": {}, "probabilities": {}, "confidence": 0.5,
               "usage": {}, "http_status": 200}

    monkeypatch.setattr(JC, "_call_jev_api_raw", fake_call)
    JC.query_jev("R1", "schema1", {"x": 1}, [], "modelhash", "2099-01-01T00:00:00")
    JC.query_jev("R2", "schema1", {"x": 2}, [], "modelhash", "2099-01-01T00:00:00")

    logs = list((tmp_path / "responses").glob("*.jsonl"))
    assert len(logs) == 1
    lines = logs[0].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2  # 2件とも追記されている(上書きされていない)


def test_spec_json_matches_documented_questions():
    spec = json.loads((BASE / "analysis" / "mcond" / "exp06_jev_decision_dev" / "spec.json")
                      .read_text(encoding="utf-8"))
    ids = [q["id"] for q in spec["questions"]]
    assert ids == ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
    assert spec["questions"][3]["options"] == ["AI", "MARKET", "BLEND", "ABSTAIN"]
    assert spec["questions"][4]["options"] == ["BET", "PASS_NO_EDGE", "PASS_UNCERTAIN",
                                                "PASS_OOD", "PASS_PRICE_RISK"]
    assert spec["prompt_language"] == "en"
    assert "TYPESAFE_API_KEY" in " ".join(spec["absolute_conditions"])
