# -*- coding: utf-8 -*-
"""予測経路の版の記録 (serve_path.py) と下流への写し。

仕様: docs/research/SERVE_REPRODUCIBILITY_FIX_SPEC_20261004.md §7
"""
from __future__ import annotations

import hashlib
import json

import production_policy as pp
import serve_history_feats as shf
import serve_path as sp
import shadow_a95


def test_unversioned_bundle_is_legacy():
    assert sp.LEGACY_VERSION == "sp-20261004-0"
    assert sp.version_of(None) == sp.LEGACY_VERSION
    assert sp.version_of({"races": []}) == sp.LEGACY_VERSION
    assert sp.stamp_fields({"races": []}) == {"serve_path_version": sp.LEGACY_VERSION,
                                              "serve_path_sha256": None}


def test_record_hashes_the_files_actually_loaded(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    hist = data / "_horse_history.parquet"
    hist.write_bytes(b"hist")
    maps = data / "serve_code_maps.json"
    maps.write_bytes(b"{}")
    model = tmp_path / "model.pkl"
    model.write_bytes(b"model")
    cal = tmp_path / "cal.pkl"
    cal.write_bytes(b"cal")
    stats = {"hist_path": str(hist), "maps_path": str(maps), "hist_rows": 3,
             "hist_max_date": 20261003, "hist_columns": ["date", "name"]}
    rec = sp.build_record(tmp_path, model_path=model, calibrator_path=cal,
                          hosei_proxy=True, history_stats=stats)
    assert rec["version"] == shf.SERVE_PATH_VERSION
    assert rec["rolling_order_key"] == shf.ROLLING_ORDER_KEY
    assert rec["identity_mode"] == shf.IDENTITY_MODE
    assert rec["name_normalization"] == shf.NAME_NORMALIZATION
    assert rec["history"] == {"path": "data/_horse_history.parquet",
                              "sha256": hashlib.sha256(b"hist").hexdigest(),
                              "loaded": True, "rows": 3, "max_date": 20261003,
                              "columns": ["date", "name"]}
    assert rec["code_maps"]["sha256"] == hashlib.sha256(b"{}").hexdigest()
    assert rec["model"]["sha256"] == hashlib.sha256(b"model").hexdigest()
    assert rec["calibrator"]["sha256"] == hashlib.sha256(b"cal").hexdigest()
    assert rec["hosei_proxy"] is True
    assert set(rec["git"]) == {"commit", "py_dirty"}

    fields = sp.stamp_fields({"serve_path": rec})
    assert fields == {"serve_path_version": rec["version"],
                      "serve_path_sha256": sp.record_sha256(rec)}


def test_record_without_history_is_marked_not_loaded(tmp_path):
    rec = sp.build_record(tmp_path)
    assert rec["history"]["loaded"] is False
    assert rec["history"]["sha256"] is None


def test_policy_stamp_copies_declared_serve_path_version(tmp_path, monkeypatch):
    pp.clear_policy_caches()
    try:
        pol = json.loads(pp.POLICY_PATH.read_text(encoding="utf-8"))
        # 本番 policy が版を宣言していれば stamp はそれを写す、未宣言なら載せない (2026-11-04 反映で宣言済み)
        if pol.get("serve_path_version"):
            assert pp.policy_stamp()["policy_serve_path_version"] == str(pol["serve_path_version"])
        else:
            assert "policy_serve_path_version" not in pp.policy_stamp()
        pol["serve_path_version"] = "sp-test-1"
        path = tmp_path / "production_policy.json"
        path.write_text(json.dumps(pol, ensure_ascii=False), encoding="utf-8")
        monkeypatch.setattr(pp, "POLICY_PATH", path)
        pp.clear_policy_caches()
        assert pp.policy_stamp()["policy_serve_path_version"] == "sp-test-1"
    finally:
        monkeypatch.undo()
        pp.clear_policy_caches()


def _race(rid: str) -> dict:
    hs = [{"umaban": i, "p_win": p, "tansho_odds": o}
          for i, p, o in ((1, 0.30, 2.5), (2, 0.20, 4.0), (3, 0.15, 6.0),
                          (4, 0.12, 9.0), (5, 0.10, 12.0), (6, 0.08, 20.0))]
    return {"race_id": rid, "horses": hs}


def test_shadow_a95_decisions_record_serve_path(tmp_path, monkeypatch):
    monkeypatch.setattr(shadow_a95, "LEDGER", tmp_path / "ledger")
    monkeypatch.setattr(shadow_a95, "BUNDLE_DIR", tmp_path)
    rec = {"version": "sp-test-1", "history": {"sha256": "x"}}
    bundle = {"serve_path": rec, "races": [_race("2099123105010101")]}
    (tmp_path / "20991231_bundle.json").write_text(json.dumps(bundle), encoding="utf-8")
    assert shadow_a95.cmd_decide("20991231", shadow_a95.load_policy()) == 0
    out = json.loads((tmp_path / "ledger" / "decisions" / "20991231.json").read_text(encoding="utf-8"))
    assert out["serve_path_version"] == "sp-test-1"
    assert out["serve_path_sha256"] == sp.record_sha256(rec)


def test_shadow_a95_decisions_mark_unversioned_bundle_as_legacy(tmp_path, monkeypatch):
    monkeypatch.setattr(shadow_a95, "LEDGER", tmp_path / "ledger")
    monkeypatch.setattr(shadow_a95, "BUNDLE_DIR", tmp_path)
    bundle = {"races": [_race("2099123105010101")]}
    (tmp_path / "20991231_bundle.json").write_text(json.dumps(bundle), encoding="utf-8")
    assert shadow_a95.cmd_decide("20991231", shadow_a95.load_policy()) == 0
    out = json.loads((tmp_path / "ledger" / "decisions" / "20991231.json").read_text(encoding="utf-8"))
    assert out["serve_path_version"] == sp.LEGACY_VERSION
    assert out["serve_path_sha256"] is None
