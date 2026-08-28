from pathlib import Path

from analysis.forward_price_eval import evaluate
from forward_prices import (
    archive_decision_record,
    archive_market_snapshot,
    build_decision_record,
)


RID = "2026082905010101"
POLICY_ID = "test-forward-policy"
STAMP = {"policy_id": POLICY_ID}


def _market(fetched: str, odds1: float = 2.0):
    return {
        "race_id": RID,
        "fetched": fetched,
        "tansho": {"1": odds1, "2": 4.0, "3": 8.0},
        "fukusho": {"1": [1.2, 1.4]},
        "wide": {"1-2": [2.5, 2.9]},
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


def _archive_decision_and_t10(root: Path, stamp=STAMP):
    market = _market("2026-08-29T10:00:00.000")
    archive_market_snapshot(market, "t10", stamp=stamp, root=root)
    primary = {"race_id": RID, "race_nature": "topdown", "bets": []}
    record = build_decision_record(
        _race(), market, primary, None, mode="default", stamp=stamp)
    archive_decision_record(record, root=root)


def test_eval_requires_close_for_every_decision(tmp_path: Path):
    _archive_decision_and_t10(tmp_path)
    report, lineage = evaluate(tmp_path, 20260829, POLICY_ID)
    assert lineage == []
    assert report["coverage"]["paired_t10"] == 1
    assert report["coverage"]["paired_close"] == 0
    assert report["integrity_errors"] == [f"{RID}: close価格なし"]


def test_eval_pairs_exact_t10_hash_and_close(tmp_path: Path):
    _archive_decision_and_t10(tmp_path)
    close = _market("2026-08-29T10:11:00.000", odds1=1.8)
    archive_market_snapshot(close, "close", stamp=STAMP, root=tmp_path)
    report, lineage = evaluate(tmp_path, 20260829, POLICY_ID)
    assert lineage == []
    assert report["integrity_errors"] == []
    assert report["coverage"]["paired_t10"] == 1
    assert report["coverage"]["paired_close"] == 1


def test_eval_rejects_policy_mixing(tmp_path: Path):
    _archive_decision_and_t10(tmp_path, stamp={"policy_id": "wrong-policy"})
    _, lineage = evaluate(tmp_path, 20260829, POLICY_ID)
    assert lineage
    assert all("policy mismatch" in msg for msg in lineage)
