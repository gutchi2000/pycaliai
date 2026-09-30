import json
from pathlib import Path

import pytest

import wide_residual_shadow as wrs
from production_policy import raw_at_percentile


RID = "2026082905010101"


@pytest.fixture(autouse=True)
def _pin_parent_policy(monkeypatch):
    """v3 は親 production policy (topdown-serve34-p667-20260825) に事前登録で固定されている。
    本線 policy が切り替わった後 (2026-10-03〜 a95) も v3 のロジック自体を検査し続けるため、
    このテストファイル内では親 policy id を登録時のものに固定する。"""
    parent = json.loads(wrs.POLICY_PATH.read_text(encoding="utf-8"))["parent_production_policy"]
    real = wrs.load_policy()
    monkeypatch.setattr(wrs, "load_policy", lambda: {**real, "policy_id": parent})


def test_cohort_closed_when_production_policy_differs(monkeypatch):
    real = wrs.load_policy()                      # fixture で親 id に固定済み → open
    assert wrs.cohort_open() is True
    monkeypatch.setattr(wrs, "load_policy", lambda: {**real, "policy_id": "some-other-policy"})
    assert wrs.cohort_open() is False
    with pytest.raises(wrs.ShadowPolicyError):
        wrs.load_shadow_policy()                  # 閉じた cohort へは計算経路でも入れない


def _race():
    return {
        "race_id": RID,
        "race_meta": {"field_size": 8},
        "race_confidence": {"field_chaos_score": raw_at_percentile(0.50)},
        "horses": [
            {"umaban": 1, "mark": "◎", "p_win": 0.35, "p_sho": 0.70},
            {"umaban": 2, "mark": "〇", "p_win": 0.25, "p_sho": 0.60},
            {"umaban": 3, "mark": "▲", "p_win": 0.20, "p_sho": 0.50},
            {"umaban": 4, "mark": "△", "p_win": 0.20, "p_sho": 0.45},
            {"umaban": 5, "mark": "", "p_win": 0.00, "p_sho": 0.00},
            {"umaban": 6, "mark": "", "p_win": 0.00, "p_sho": 0.00},
            {"umaban": 7, "mark": "", "p_win": 0.00, "p_sho": 0.00},
            {"umaban": 8, "mark": "", "p_win": 0.00, "p_sho": 0.00},
        ],
    }


def _market():
    pairs = {
        f"{i}-{j}": [float(i + j + 1), float(i + j + 1.2)]
        for i in range(1, 9)
        for j in range(i + 1, 9)
    }
    return {"ok": True, "race_id": RID, "fetched": "2026-08-29T10:00:00",
            "tansho": {str(i): 3.0 + i for i in range(1, 9)}, "wide": pairs}


def _pair_probs(_horses):
    fair = wrs.market_wide_fair(_market()["wide"])
    wide = {pair: max(0.0, p - 0.10) for pair, p in fair.items()}
    for pair in ((1, 2), (1, 3)):
        wide[pair] = fair[pair] + 0.02
    return {}, wide


def test_arm_is_model_top2_then_residual_filter_and_never_real_money():
    out = wrs.compute_shadow(_race(), _market(), pair_probability_fn=_pair_probs)
    assert out["triggered"] is True
    assert out["real_money_enabled"] is False
    assert out["parent_policy"]["artifact_sha256"]["serve_calibrator"]
    assert [row["selection"] for row in out["arm_a"]] == ["1-2", "1-3"]
    assert all(row["virtual_stake_yen"] == 100 for row in out["arm_a"])
    assert all(0.0 <= row["residual"] < 0.05 for row in out["arm_a"])
    assert len(out["control_m1"]) == 2
    assert len(out["control_m2"]) == 2


def test_incomplete_wide_market_fails_closed():
    market = _market()
    market["wide"].pop("1-2")
    with pytest.raises(wrs.ShadowPolicyError, match="価格が不完全"):
        wrs.compute_shadow(_race(), market, pair_probability_fn=_pair_probs)


def test_hard_gate_returns_no_shadow_ticket():
    race = _race()
    race["race_meta"]["field_size"] = 7
    out = wrs.compute_shadow(race, _market(), pair_probability_fn=_pair_probs)
    assert out["hard_gate_passed"] is False
    assert out["arm_a"] == []
    assert out["triggered"] is False


def test_daily_merge_replaces_same_race_and_rejects_policy_mix(tmp_path: Path):
    path = tmp_path / "20260829_shadow.json"
    first = wrs.compute_shadow(_race(), _market(), pair_probability_fn=_pair_probs)
    wrs.merge_daily_shadow(path, [first])
    second = dict(first)
    second["triggered"] = False
    wrs.merge_daily_shadow(path, [second])
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert len(saved["races"]) == 1
    assert saved["races"][0]["triggered"] is False
    saved["policy"]["policy_id"] = "old-policy"
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(wrs.ShadowPolicyError, match="cohort混在"):
        wrs.merge_daily_shadow(path, [first])
