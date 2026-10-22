"""本番成果物の期待 hash 照合 (2026-09-20 C0 退行の再発防止)。"""
import hashlib
import json

import pytest

from production_policy import PolicyError, verify_artifact_pins


def _pins(tmp_path, files: dict) -> object:
    p = tmp_path / "pins.json"
    p.write_text(json.dumps({"schema_version": 1, "generation": "test",
                             "pins": files}), encoding="utf-8")
    return p


def _blob(tmp_path, name, data: bytes):
    f = tmp_path / name
    f.write_bytes(data)
    return str(f), hashlib.sha256(data).hexdigest()


def test_matching_pins_pass(tmp_path):
    path, digest = _blob(tmp_path, "m.pkl", b"model-c1")
    verify_artifact_pins.cache_clear()
    out = verify_artifact_pins(_pins(tmp_path, {path: digest}))
    assert out["pins_generation"] == "test"


def test_mismatch_is_fail_closed(tmp_path):
    path, _ = _blob(tmp_path, "m.pkl", b"model-c0")
    wrong = hashlib.sha256(b"model-c1").hexdigest()
    verify_artifact_pins.cache_clear()
    with pytest.raises(PolicyError, match="期待版と不一致"):
        verify_artifact_pins(_pins(tmp_path, {path: wrong}))


def test_missing_artifact_is_fail_closed(tmp_path):
    verify_artifact_pins.cache_clear()
    with pytest.raises(PolicyError):
        verify_artifact_pins(_pins(tmp_path, {str(tmp_path / "none.pkl"): "0" * 64}))


def test_empty_pins_is_fail_closed(tmp_path):
    verify_artifact_pins.cache_clear()
    with pytest.raises(PolicyError, match="pins 欠落"):
        verify_artifact_pins(_pins(tmp_path, {}))


def test_repository_artifacts_match_pins():
    """本番ツリーの成果物が pin と一致すること (LFS 実体が必要)。"""
    verify_artifact_pins.cache_clear()
    verify_artifact_pins()
