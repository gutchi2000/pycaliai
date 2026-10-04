# -*- coding: utf-8 -*-
"""予測入力の品質ゲート (馬同定) と予測経路の版による見送り・cohort 分離。

仕様: docs/research/SERVE_REPRODUCIBILITY_FIX_SPEC_20261004.md §3.3 / §7 / §8 (案 B)
"""
from __future__ import annotations

import json

import pytest

import compute_bets
import input_quality as iq
import production_policy as pp
import shadow_a95
import t20_site_bets
from serve_path import LEGACY_VERSION

RID = "2099123105010101"     # conftest: 2099 始まりは平地扱い


def _horses(n: int = 8) -> list[dict]:
    return [{"umaban": i, "horse_name": f"ウマ{i}", "p_win": round(0.3 - 0.03 * i, 3),
             "tansho_odds": 2.0 + i, "mark": ["◎", "〇", "▲", "△", "△"][i - 1] if i <= 5 else ""}
            for i in range(1, n + 1)]


def _race(statuses: dict[int, str] | None = None, version: str = "sp-20261004-1") -> dict:
    race = {"race_id": RID, "race_meta": {"field_size": 8}, "race_confidence": {}, "horses": _horses()}
    ident = {(RID, h["umaban"]): (statuses or {}).get(h["umaban"], "hit") for h in race["horses"]}
    iq.annotate_race(race, ident, version)
    return race


def test_annotate_race_marks_ambiguous_and_unknown_horses():
    race = _race({3: "ambiguous", 5: "new"})
    assert race["input_quality"]["bet_eligible"] is False
    assert race["input_quality"]["identity"] == {"hit": 6, "new": 1, "ambiguous": 1, "unknown": 0}
    assert [h["identity_status"] for h in race["horses"]][2] == "ambiguous"
    assert race["input_quality"]["ambiguous_horses"][0]["umaban"] == 3

    race2 = {"race_id": RID, "horses": _horses(6)}
    iq.annotate_race(race2, {}, "sp-x")             # 同定状態が無い馬は unknown → 見送り
    assert race2["input_quality"]["bet_eligible"] is False
    assert race2["input_quality"]["identity"]["unknown"] == 6

    assert _race({5: "new"})["input_quality"]["bet_eligible"] is True


def test_day_summary_two_percent_threshold():
    ok = [_race() for _ in range(6)]                  # 48 頭
    ok[0] = _race({1: "ambiguous"})                   # 1/48 = 2.08% > 2%
    s = iq.day_summary(ok, "sp-x")
    assert s["n_horses"] == 48 and s["n_ambiguous"] == 1
    assert s["day_pass"] is False and len(s["skipped_races"]) == 1
    many = [_race() for _ in range(7)] + [_race({1: "ambiguous"})]   # 1/64 = 1.56%
    assert iq.day_summary(many, "sp-x")["day_pass"] is True
    exact = [_race() for _ in range(6)] + [{"race_id": "x", "horses": [],
                                            "input_quality": {"identity": {"hit": 1, "ambiguous": 1}}}]
    assert iq.day_summary(exact, "sp-x")["n_horses"] == 50
    assert iq.day_summary(exact, "sp-x")["day_pass"] is True          # ちょうど 2% は合格


def test_race_skip_reason_identity_and_version():
    assert iq.race_skip_reason(_race()) is None
    assert "判別不能" in iq.race_skip_reason(_race({2: "ambiguous"}))
    assert iq.race_skip_reason(_race(), "sp-20261004-1") is None
    assert "版不一致" in iq.race_skip_reason(_race(version="sp-20261004-1"), "sp-20261004-2")
    legacy = {"race_id": RID, "horses": _horses()}                    # 記録の無い旧 bundle
    assert iq.race_skip_reason(legacy) is None
    assert f"bundle={LEGACY_VERSION}" in iq.race_skip_reason(legacy, "sp-20261004-1")


@pytest.fixture
def policy_with_version(tmp_path, monkeypatch):
    def _set(version: str | None, start: int | None = None):
        pol = json.loads(pp.POLICY_PATH.read_text(encoding="utf-8"))
        if version:
            pol["serve_path_version"] = version
        if start:
            pol["prospective"]["start_date"] = start
        path = tmp_path / "production_policy.json"
        path.write_text(json.dumps(pol, ensure_ascii=False), encoding="utf-8")
        monkeypatch.setattr(pp, "POLICY_PATH", path)
        pp.clear_policy_caches()
    yield _set
    monkeypatch.undo()
    pp.clear_policy_caches()


def test_compute_bets_skips_ambiguous_identity_race():
    pp.clear_policy_caches()
    out = compute_bets.compute_race_bets(_race({4: "ambiguous"}))
    assert out["race_nature"] == "見送り" and out["bets"] == []
    assert "判別不能" in out["input_quality_gate"]
    assert "a95_all" not in out


def test_compute_bets_skips_when_bundle_version_differs_from_policy(policy_with_version):
    policy_with_version("sp-20261004-1")
    legacy = {"race_id": RID, "race_meta": {"field_size": 8}, "race_confidence": {}, "horses": _horses()}
    out = compute_bets.compute_race_bets(legacy)
    assert out["race_nature"] == "見送り" and "版不一致" in out["input_quality_gate"]


def test_t20_preview_skips_ambiguous_identity_race():
    pp.clear_policy_caches()
    bets, why, done = t20_site_bets.a95_site_tickets(_race({1: "ambiguous"}), {"tansho": {}}, RID)
    assert (bets, done) == ([], True) and "品質ゲート" in why


def test_shadow_a95_decision_records_identity_gate():
    pol = shadow_a95.load_policy()
    bundle = {"races": [_race({2: "ambiguous"}), _race()]}
    decs = shadow_a95.build_decisions(bundle, pol)
    assert decs[0] == {"rid": RID, "n": 8, "valid": False, "reason": "input_quality_identity"}
    assert decs[1]["valid"] is True


def test_shadow_a95_cohort_follows_production_policy(policy_with_version):
    pol = shadow_a95.load_policy()
    pp.clear_policy_caches()
    assert shadow_a95.cohort(pol) == (LEGACY_VERSION, str(pol["registration"]["start_date"]))
    policy_with_version("sp-20261004-1", 20261010)
    assert shadow_a95.cohort(pol) == ("sp-20261004-1", "20261010")


def test_shadow_a95_report_never_pools_versions(tmp_path, monkeypatch, policy_with_version, capsys):
    pol = shadow_a95.load_policy()
    monkeypatch.setattr(shadow_a95, "LEDGER", tmp_path)
    (tmp_path / "settlements").mkdir()
    arm = {"valid": True, "stake": 10000, "ret": 0.0, "hit": False, "by_kind": {}}
    for date, ver in (("20261004", None), ("20261011", "sp-20261004-1")):
        rec = {"policy_sha256": pol["policy_sha256"], "date": date, "preregistered": True,
               "races": [{"valid": True, "band": "<2倍", "arms": {"a95_equal": dict(arm)}}]}
        if ver:
            rec["serve_path_version"] = ver
        (tmp_path / "settlements" / f"{date}.json").write_text(json.dumps(rec), encoding="utf-8")
    policy_with_version("sp-20261004-1", 20261010)
    rows = shadow_a95._collect(pol)
    assert [(r["date"], r["fwd"], r["version"]) for r in rows] == [
        ("20261004", False, LEGACY_VERSION), ("20261011", True, "sp-20261004-1")]
    shadow_a95.cmd_report(pol)
    text = capsys.readouterr().out
    assert f"旧版 cohort {LEGACY_VERSION}" in text
