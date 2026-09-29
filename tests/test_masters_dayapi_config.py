# -*- coding: utf-8 -*-
"""masters_dayapi.load_config の読込み順と fail-closed (live API には接続しない)。

優先順位: 環境変数 > data/masters_dayapi.json (.gitignore 済み) > 例外。
値はすべてテスト用のダミーで、実設定ファイルは読まない (CONFIG_PATH を tmp へ差し替える)。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import requests

import masters_dayapi as dayapi

BASE = Path(__file__).resolve().parents[1]
ENV_URL_VAL = "https://env.example.invalid/data"
ENV_KEY_VAL = "env-dummy-key"
JSON_URL_VAL = "https://json.example.invalid/data"
JSON_KEY_VAL = "json-dummy-key"


@pytest.fixture
def cfg_path(tmp_path, monkeypatch):
    path = tmp_path / "masters_dayapi.json"
    monkeypatch.setattr(dayapi, "CONFIG_PATH", path)
    monkeypatch.delenv(dayapi.ENV_URL, raising=False)
    monkeypatch.delenv(dayapi.ENV_API_KEY, raising=False)
    return path


def _write(path: Path, **kw) -> None:
    path.write_text(json.dumps(kw), encoding="utf-8")


def _no_network(*_a, **_k):
    raise AssertionError("live API に接続しようとした")


def test_env_only(cfg_path, monkeypatch):
    monkeypatch.setenv(dayapi.ENV_URL, ENV_URL_VAL)
    monkeypatch.setenv(dayapi.ENV_API_KEY, ENV_KEY_VAL)
    cfg = dayapi.load_config()
    assert (cfg["url"], cfg["api_key"]) == (ENV_URL_VAL, ENV_KEY_VAL)


def test_json_only(cfg_path):
    _write(cfg_path, url=JSON_URL_VAL, api_key=JSON_KEY_VAL)
    cfg = dayapi.load_config()
    assert (cfg["url"], cfg["api_key"]) == (JSON_URL_VAL, JSON_KEY_VAL)


def test_env_overrides_json(cfg_path, monkeypatch):
    _write(cfg_path, url=JSON_URL_VAL, api_key=JSON_KEY_VAL)
    monkeypatch.setenv(dayapi.ENV_URL, ENV_URL_VAL)
    monkeypatch.setenv(dayapi.ENV_API_KEY, ENV_KEY_VAL)
    cfg = dayapi.load_config()
    assert (cfg["url"], cfg["api_key"]) == (ENV_URL_VAL, ENV_KEY_VAL)


def test_env_overrides_per_field(cfg_path, monkeypatch):
    """netkeiba_api.load_config と同じく項目ごとに env が勝つ。"""
    _write(cfg_path, url=JSON_URL_VAL, api_key=JSON_KEY_VAL)
    monkeypatch.setenv(dayapi.ENV_API_KEY, ENV_KEY_VAL)
    cfg = dayapi.load_config()
    assert (cfg["url"], cfg["api_key"]) == (JSON_URL_VAL, ENV_KEY_VAL)


def test_neither_fails_closed(cfg_path):
    assert not cfg_path.exists()
    with pytest.raises(dayapi.DayApiConfigError) as ei:
        dayapi.load_config()
    assert "url" in str(ei.value) and "api_key" in str(ei.value)


@pytest.mark.parametrize("payload", [
    {"url": JSON_URL_VAL},                    # key 欠落
    {"api_key": JSON_KEY_VAL},                # url 欠落
    {"url": "   ", "api_key": JSON_KEY_VAL},  # 空白だけ
    {},
])
def test_partial_config_fails_closed(cfg_path, payload):
    _write(cfg_path, **payload)
    with pytest.raises(dayapi.DayApiConfigError):
        dayapi.load_config()


def test_broken_json_fails_closed(cfg_path):
    cfg_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(dayapi.DayApiConfigError):
        dayapi.load_config()


def test_config_error_message_has_no_values(cfg_path, monkeypatch):
    monkeypatch.setenv(dayapi.ENV_API_KEY, ENV_KEY_VAL)   # url だけ欠落
    with pytest.raises(dayapi.DayApiConfigError) as ei:
        dayapi.load_config()
    assert ENV_KEY_VAL not in str(ei.value)


def test_config_error_is_dayapi_error():
    """呼び出し側 (masters_vote.post_datetime) は DayApiError で weekly CSV へ戻る。"""
    assert issubclass(dayapi.DayApiConfigError, dayapi.DayApiError)


def test_get_without_config_never_touches_network(cfg_path, monkeypatch):
    monkeypatch.setattr(dayapi.requests, "get", _no_network)
    with pytest.raises(dayapi.DayApiConfigError):
        dayapi.racecards("20991231")


def test_get_sends_configured_values(cfg_path, monkeypatch):
    _write(cfg_path, url=JSON_URL_VAL, api_key=JSON_KEY_VAL)
    seen = {}

    class _Resp:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"data": {"timetable": [], "runtable": []}}

    def _fake_get(url, headers=None, **kw):
        seen.update(url=url, headers=headers, **kw)
        return _Resp()

    monkeypatch.setattr(dayapi.requests, "get", _fake_get)
    assert dayapi.racecards("20991231") == ([], [])
    assert seen["url"] == JSON_URL_VAL
    assert seen["headers"]["api-key"] == JSON_KEY_VAL
    assert seen["headers"]["id"] == "20991231"


def test_network_error_message_hides_endpoint(cfg_path, monkeypatch):
    _write(cfg_path, url=JSON_URL_VAL, api_key=JSON_KEY_VAL)

    def _boom(*_a, **_k):
        raise requests.ConnectionError(f"cannot reach {JSON_URL_VAL}")

    monkeypatch.setattr(dayapi.requests, "get", _boom)
    with pytest.raises(dayapi.DayApiError) as ei:
        dayapi.racecards("20991231")
    assert JSON_URL_VAL not in str(ei.value)
    assert "ConnectionError" in str(ei.value)


def test_module_has_no_embedded_endpoint_or_key():
    """埋込み定数を持たない (値そのものはテストに書かない)。"""
    assert not hasattr(dayapi, "API_KEY")
    assert not hasattr(dayapi, "URL")
    src = (BASE / "masters_dayapi.py").read_text(encoding="utf-8")
    assert not re.search(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", src)
    assert not re.search(r"https?://(?!example\.)[A-Za-z0-9]", src)
    assert not re.search(r"^\s*(API_KEY|URL)\s*=", src, flags=re.M)


def test_example_config_has_empty_values():
    ex = json.loads((BASE / "data" / "masters_dayapi.json.example").read_text(encoding="utf-8"))
    assert ex["url"] == "" and ex["api_key"] == ""
