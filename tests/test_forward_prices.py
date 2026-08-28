from pathlib import Path

import pytest

from forward_prices import (
    archive_decision_record,
    archive_market_snapshot,
    build_decision_record,
    fair_win_probabilities,
    read_snapshot,
)


RID = "2026082905010101"
STAMP = {"policy_id": "test-policy", "policy_sha256": "a" * 64}


def _market():
    return {
        "race_id": RID,
        "fetched": "2026-08-29T10:00:00.123",
        "tansho": {"1": 2.0, "2": 4.0, "3": 8.0},
        "fukusho": {"1": [1.2, 1.4], "2": [1.8, 2.1]},
        "wide": {"1-2": [2.5, 2.9]},
        "overround_tan": 0.875,
    }


def _race():
    return {
        "race_id": RID,
        "horses": [
            {"umaban": 1, "mark": "◎", "p_win": 0.60, "p_sho": 0.85},
            {"umaban": 2, "mark": "〇", "p_win": 0.25, "p_sho": 0.60},
            {"umaban": 3, "mark": "▲", "p_win": 0.15, "p_sho": 0.40},
        ],
    }


def test_market_snapshots_are_immutable_and_readable(tmp_path: Path):
    market = _market()
    first = archive_market_snapshot(market, "t10", stamp=STAMP, root=tmp_path)
    second = archive_market_snapshot(market, "t10", stamp=STAMP, root=tmp_path)
    assert first == second
    assert len(list(tmp_path.rglob("*.json.gz"))) == 1
    saved = read_snapshot(first)
    assert saved["stage"] == "t10"
    assert saved["market"] == market
    assert saved["policy"]["policy_id"] == "test-policy"


def test_market_probability_is_devigged():
    fair = fair_win_probabilities(_market()["tansho"])
    assert sum(fair.values()) == pytest.approx(1.0)
    assert fair[1] > fair[2] > fair[3]


def test_decision_contains_market_residual_and_engine_pair(tmp_path: Path):
    primary = {"race_id": RID, "race_nature": "topdown", "bets": []}
    shadow = {"race_id": RID, "race_nature": "本命勝負", "bets": []}
    record = build_decision_record(
        _race(), _market(), primary, shadow, mode="default",
        model_umaren={(1, 2): 0.30}, model_wide={(1, 2): 0.55}, stamp=STAMP)
    path = archive_decision_record(record, root=tmp_path)
    saved = read_snapshot(path)
    hon = next(x for x in saved["horses"] if x["umaban"] == 1)
    assert hon["win_market_residual"] == pytest.approx(
        hon["p_win_model"] - hon["p_win_market_devig"])
    assert saved["primary"] == primary
    assert saved["shadow"] == shadow
    assert saved["pairs"][0]["p_wide_model"] == pytest.approx(0.55)
