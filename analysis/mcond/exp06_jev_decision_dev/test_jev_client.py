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


def test_query_jev_fatal_error_no_retry_no_sleep(monkeypatch, tmp_path):
    """401/403/422等は設定/リクエスト自体の問題なので即時FAILしリトライしない
    (2026-09-20ユーザー指定のHTTP規則)。"""
    monkeypatch.setenv("TYPESAFE_API_KEY", "dummy-for-test-not-real")
    monkeypatch.setattr(JC, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(JC, "RESPONSES_DIR", tmp_path / "responses")

    sleeps = []
    monkeypatch.setattr(JC.time, "sleep", lambda s: sleeps.append(s))

    call_count = {"n": 0}

    def fatal_call(prompt_schema_hash, state, questions):
        call_count["n"] += 1
        raise JC.JevFatalError("jev api fatal status=401")

    monkeypatch.setattr(JC, "_call_jev_api_raw", fatal_call)
    result = JC.query_jev("R1", "schema1", {"x": 1}, [], "modelhash", "2099-01-01T00:00:00")
    assert result["ok"] is False
    assert call_count["n"] == 1  # リトライしていない
    assert sleeps == []  # バックオフのsleepも一切していない
    assert result["retry_count"] == 0


def test_query_jev_transient_error_uses_exponential_backoff(monkeypatch, tmp_path):
    """timeout/接続エラー/一時的5xxもrate limitと同じ有限回の指数バックオフ。"""
    monkeypatch.setenv("TYPESAFE_API_KEY", "dummy-for-test-not-real")
    monkeypatch.setattr(JC, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(JC, "RESPONSES_DIR", tmp_path / "responses")
    monkeypatch.setattr(JC, "RETRY_BACKOFF_S", 0.01)

    sleeps = []
    monkeypatch.setattr(JC.time, "sleep", lambda s: sleeps.append(s))

    def transient_call(prompt_schema_hash, state, questions):
        raise JC.JevTransientError("jev api timeout")

    monkeypatch.setattr(JC, "_call_jev_api_raw", transient_call)
    result = JC.query_jev("R1", "schema1", {"x": 1}, [], "modelhash", "2099-01-01T00:00:00")
    assert result["ok"] is False
    assert sleeps == [0.01 * (2 ** i) for i in range(JC.MAX_RETRIES)]


@pytest.mark.parametrize("status,expected_exc", [
    (401, JC.JevFatalError), (403, JC.JevFatalError), (422, JC.JevFatalError),
    (400, JC.JevFatalError), (404, JC.JevFatalError),
    (429, JC.JevRateLimitError), (529, JC.JevRateLimitError),
    (500, JC.JevTransientError), (503, JC.JevTransientError),
])
def test_call_jev_api_raw_classifies_status_codes(monkeypatch, status, expected_exc):
    monkeypatch.setenv("TYPESAFE_API_KEY", "dummy-for-test-not-real")

    class FakeResp:
        status_code = status
        def json(self):
            return {}
        def raise_for_status(self):
            pass

    class FakeRequests:
        exceptions = __import__("requests").exceptions
        @staticmethod
        def post(*a, **k):
            return FakeResp()

    monkeypatch.setitem(sys.modules, "requests", FakeRequests)
    with pytest.raises(expected_exc):
        JC._call_jev_api_raw("schema1", {"x": 1}, [])


def test_call_jev_api_raw_never_leaks_headers_in_exception_message(monkeypatch):
    """例外メッセージにAuthorizationヘッダーやキーの値が絶対に含まれないことを確認する
    (2026-09-20ユーザー指定)。"""
    monkeypatch.setenv("TYPESAFE_API_KEY", "super-secret-value-must-not-leak")

    class FakeResp:
        status_code = 401
        def json(self):
            return {}
        def raise_for_status(self):
            pass

    class FakeRequests:
        exceptions = __import__("requests").exceptions
        @staticmethod
        def post(*a, **k):
            # k["headers"] にAuthorizationが含まれているはずだが、例外側には渡さない
            assert "super-secret-value-must-not-leak" in k["headers"]["Authorization"]
            return FakeResp()

    monkeypatch.setitem(sys.modules, "requests", FakeRequests)
    with pytest.raises(JC.JevFatalError) as exc_info:
        JC._call_jev_api_raw("schema1", {"x": 1}, [])
    assert "super-secret-value-must-not-leak" not in str(exc_info.value)


def test_questions_to_api_payload_matches_spec_types():
    """2026-09-20、ユーザー提示の実成功fixtureに合わせた形: questionsはidをキーとする
    dict、各値は{type, instructions, criteria}のフラット構造(id/name/scale/options等の
    余剰キーを含まない)。"""
    spec = json.loads((BASE / "analysis" / "mcond" / "exp06_jev_decision_dev" / "spec.json")
                      .read_text(encoding="utf-8"))
    payload = JC._questions_to_api_payload(spec["questions"])
    assert isinstance(payload, dict)
    assert set(payload.keys()) == {"Q1", "Q2", "Q3", "Q4", "Q5", "Q6"}

    assert payload["Q1"]["type"] == "noul"
    assert set(payload["Q1"].keys()) == {"type", "instructions", "criteria"}
    assert set(payload["Q1"]["criteria"].keys()) == {"true", "false"}

    assert payload["Q2"]["type"] == "score"
    assert set(payload["Q2"].keys()) == {"type", "instructions", "criteria"}
    assert isinstance(payload["Q2"]["criteria"], list) and len(payload["Q2"]["criteria"]) == 5

    assert payload["Q4"]["type"] == "choice"
    assert set(payload["Q4"].keys()) == {"type", "instructions", "criteria"}
    assert payload["Q4"]["criteria"] == {
        "AI": "Trust the independent predictive model's estimate over the market price.",
        "MARKET": "Trust the market price over the independent predictive model's estimate.",
        "BLEND": "Blend the predictive model's estimate and the market price.",
        "ABSTAIN": "Neither source is trustworthy enough to base a decision on.",
    }


def test_questions_to_api_payload_matches_known_good_fixture_shape():
    """2026-09-20にユーザーが実APIへ送信しHTTP 200で成功した既知のfixture
    (推測ではなく実データ)と、_questions_to_api_payload()の出力の"形"(キー集合と
    型)が一致することを構造的に検証する(内容そのものは別の質問文なので文字列は
    比較しない)。"""
    known_good_questions_payload = {
        "Q1": {
            "type": "noul",
            "instructions": "The available evidence is sufficient for an automated decision.",
            "criteria": {"true": "Evidence is sufficient", "false": "Evidence is insufficient"},
        },
        "Q2": {
            "type": "score",
            "instructions": "Rate the risk that this case is outside the supported distribution.",
            "criteria": ["Clearly supported", "Mostly supported", "Borderline",
                        "Weak support", "Clearly outside support"],
        },
        "Q3": {
            "type": "choice",
            "instructions": "Choose the safest action.",
            "criteria": {"BET": "Make an automated decision",
                        "PASS_UNCERTAIN": "Do not act because evidence is uncertain",
                        "PASS_OOD": "Do not act because the case is outside support"},
        },
    }
    spec = json.loads((BASE / "analysis" / "mcond" / "exp06_jev_decision_dev" / "spec.json")
                      .read_text(encoding="utf-8"))
    payload = JC._questions_to_api_payload(spec["questions"])

    def shape(q: dict) -> tuple:
        crit = q["criteria"]
        crit_shape = "list_of_str" if isinstance(crit, list) else (
            "dict_true_false" if set(crit.keys()) == {"true", "false"} else "dict_options")
        return (set(q.keys()), q["type"], crit_shape)

    # Q1(noul)はknown_goodのQ1と同じ形、Q2(score)はknown_goodのQ2と同じ形、
    # Q4(choice、trust_source)はknown_goodのQ3(choice)と同じ形であるはず
    assert shape(payload["Q1"]) == shape(known_good_questions_payload["Q1"])
    assert shape(payload["Q2"]) == shape(known_good_questions_payload["Q2"])
    assert shape(payload["Q4"]) == shape(known_good_questions_payload["Q3"])
    # criteriaがlistのScoreは全要素が文字列であること
    assert all(isinstance(x, str) for x in payload["Q2"]["criteria"])
    # criteriaがdictのChoiceは全値が文字列であること(optionsそのものをキーに使う)
    assert all(isinstance(v, str) for v in payload["Q4"]["criteria"].values())
    # 入れ子ラッパー("score":{...}や"choice":{...})を作っていないこと
    # (誤りの例: {"type":"score","score":{"criteria":[...]}} のように型名をキーに
    # した入れ子を作ってしまう回帰を防ぐ)
    for qid, q in payload.items():
        assert "score" not in q, f"{qid}に入れ子ラッパー'score'キーが混入している"
        assert "choice" not in q, f"{qid}に入れ子ラッパー'choice'キーが混入している"
        assert "noul" not in q, f"{qid}に入れ子ラッパー'noul'キーが混入している"


def test_query_jev_rate_limit_uses_exponential_backoff(monkeypatch, tmp_path):
    monkeypatch.setenv("TYPESAFE_API_KEY", "dummy-for-test-not-real")
    monkeypatch.setattr(JC, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(JC, "RESPONSES_DIR", tmp_path / "responses")
    monkeypatch.setattr(JC, "RETRY_BACKOFF_S", 0.01)  # テストを高速化

    sleeps = []
    monkeypatch.setattr(JC.time, "sleep", lambda s: sleeps.append(s))

    call_count = {"n": 0}

    def flaky_call(prompt_schema_hash, state, questions):
        call_count["n"] += 1
        raise JC.JevRateLimitError("429")

    monkeypatch.setattr(JC, "_call_jev_api_raw", flaky_call)
    result = JC.query_jev("R1", "schema1", {"x": 1}, [], "modelhash", "2099-01-01T00:00:00")
    assert result["ok"] is False
    assert call_count["n"] == JC.MAX_RETRIES + 1
    # 指数バックオフ: 0.01*(2**0), 0.01*(2**1), ... と倍々に増えるはず
    assert sleeps == [0.01 * (2 ** i) for i in range(JC.MAX_RETRIES)]


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


# ---- 2026-09-20、ユーザー指定のStage B準備事項(モデル固定・単発応答ポリシー・
# spec修正記録) ----
def test_jev_model_is_pinned_not_alias():
    """主評価はjev-latestではなく実測バージョンIDへ固定する(ユーザー指定)。"""
    assert JC.JEV_MODEL == "jev-1.13.0"
    assert "latest" not in JC.JEV_MODEL


def test_compute_input_hash_includes_model():
    """同じstate+schemaでもmodelが違えば別キャッシュキーになること
    (モデルバージョンをキャッシュキーに含める、という指定の実装確認)。"""
    h1 = JC.compute_input_hash("schema1", {"a": 1}, "jev-1.13.0")
    h2 = JC.compute_input_hash("schema1", {"a": 1}, "jev-1.14.0")
    assert h1 != h2
    # model省略時はJEV_MODEL(固定バージョン)が暗黙に使われる
    h3 = JC.compute_input_hash("schema1", {"a": 1})
    assert h3 == h1


def test_query_jev_cache_key_uses_pinned_model(monkeypatch, tmp_path):
    """query_jev()が実際にJEV_MODELをキャッシュキーへ渡していることを確認する
    (compute_input_hashのmodel引数が実際に呼び出し経路で使われているか)。"""
    monkeypatch.setenv("TYPESAFE_API_KEY", "dummy-for-test-not-real")
    monkeypatch.setattr(JC, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(JC, "RESPONSES_DIR", tmp_path / "responses")

    def fake_call(prompt_schema_hash, state, questions):
        return {"model": JC.JEV_MODEL, "answers": {}, "probabilities": {}, "confidence": {},
               "usage": {}, "http_status": 200}

    monkeypatch.setattr(JC, "_call_jev_api_raw", fake_call)
    result = JC.query_jev("R1", "schema1", {"x": 1}, [], "modelhash", "2099-01-01T00:00:00")
    expected_hash = JC.compute_input_hash("schema1", {"x": 1}, JC.JEV_MODEL)
    assert result["input_hash"] == expected_hash


def test_spec_json_has_api_conformance_amendment_record():
    """API適合修正が「競馬結果を見る前」に行われたことをspec.json自身に
    記録していること(ユーザー指定の説明責任要件)。"""
    spec = json.loads((BASE / "analysis" / "mcond" / "exp06_jev_decision_dev" / "spec.json")
                      .read_text(encoding="utf-8"))
    amendment = spec["api_conformance_amendment"]
    assert amendment["race_results_seen"] is False
    assert amendment["payout_or_roi_seen"] is False
    assert "Stage B" in amendment["amended_at"]
    assert spec["primary_model"] == "jev-1.13.0"
    assert spec["primary_evaluation_policy"]["single_shot_only"]
    assert spec["prompt_language_note"].count("日本語") >= 1  # JA=Stage A限定である旨の記載確認
