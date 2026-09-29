"""wide residual forward 評価器が schema v2 の `close_late` と v1 の `close` の両方を締切録として読むこと。"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import forward_prices as FP
from analysis.evaluate_wide_residual_forward import load_price_lineage

DATE = "20261003"
RID = "2026100306040908"
STAMP = {"policy_id": "test-policy", "policy_sha256": "a" * 64}


def _market(fetched: str) -> dict:
    return {"race_id": RID, "fetched": fetched, "ok": True, "tansho": {"1": 2.5, "2": 4.0}}


def _write_v1(root: Path, stage: str, name: str, market_sha: str = "b" * 64) -> None:
    rec = {"schema_version": 1, "record_type": "market_snapshot", "stage": stage, "race_id": RID,
           "observed_at": "2026-10-03T13:56:00", "market": {}, "market_sha256": market_sha}
    p = root / DATE / name
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        json.dump(rec, fh)


def test_v1_close_record_is_read_as_close(tmp_path):
    _write_v1(tmp_path, "close", f"{RID}_close_20261003135600000_aaaaaaaaaa.json.gz")
    assert load_price_lineage(DATE, tmp_path)[RID]["has_close"] is True


def test_v2_close_request_is_stored_as_close_late_and_read(tmp_path):
    p = FP.archive_market_snapshot(_market("2026-10-03T13:56:00.100"), "close", stamp=STAMP, root=tmp_path)
    assert FP.read_snapshot(p)["stage"] == "close_late"
    assert load_price_lineage(DATE, tmp_path)[RID]["has_close"] is True


def test_v2_close_late_written_directly_is_read(tmp_path):
    FP.archive_market_snapshot(_market("2026-10-03T13:56:00.200"), "close_late", stamp=STAMP, root=tmp_path)
    assert load_price_lineage(DATE, tmp_path)[RID]["has_close"] is True


def test_t10_hash_is_kept_and_other_v2_stages_are_not_close(tmp_path):
    m = _market("2026-10-03T13:45:00.100")
    FP.archive_market_snapshot(m, "t10", stamp=STAMP, root=tmp_path)
    for stage, t in (("t2_candidate", "13:53"), ("final_rt_candidate", "18:30"),
                     ("final_stock_candidate", "18:40"), ("trio_t10", "13:45")):
        FP.archive_market_snapshot(_market(f"2026-10-03T{t}:00.300"), stage, stamp=STAMP, root=tmp_path)
    lin = load_price_lineage(DATE, tmp_path)[RID]
    assert lin["t10_hashes"] == {FP.payload_sha256(m)}
    assert lin["has_close"] is False
