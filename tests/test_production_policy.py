import copy

import pytest

from production_policy import (
    hard_skip_reasons,
    load_policy,
    load_quantiles,
    policy_stamp,
    raw_at_percentile,
)


def _valid_inputs():
    raw = raw_at_percentile(0.50)
    return ({"field_size": 12}, {"field_chaos_score": raw},
            {"mark": "◎", "p_win": 0.20, "tansho_odds": 4.0})


def test_policy_and_quantile_reference_are_pinned():
    policy = load_policy()
    quantiles = load_quantiles()
    assert policy["chaos_reference"]["reference_id"] == quantiles["reference_id"]
    assert len(quantiles["quantiles"]["field_chaos_score"]) == 101
    assert policy_stamp()["policy_id"] == policy["policy_id"]


def test_hard_skip_is_fail_closed_on_missing_inputs():
    meta, conf, hon = _valid_inputs()
    assert hard_skip_reasons(meta, conf, hon) == []
    assert any("chaos欠損" in x for x in hard_skip_reasons(meta, {}, hon))
    assert any("field_size欠損" in x for x in hard_skip_reasons({}, conf, hon))
    missing_odds = copy.deepcopy(hon)
    missing_odds.pop("tansho_odds")
    assert any("tansho_odds欠損" in x
               for x in hard_skip_reasons(meta, conf, missing_odds))


def test_chaos_gate_is_percentile_based():
    policy = load_policy()
    q = float(policy["chaos_reference"]["skip_percentile"])
    meta, _, hon = _valid_inputs()
    below = {"field_chaos_score": raw_at_percentile(q - 0.01)}
    at_gate = {"field_chaos_score": raw_at_percentile(q)}
    assert not any("chaos raw=" in x for x in hard_skip_reasons(meta, below, hon))
    assert any("chaos raw=" in x for x in hard_skip_reasons(meta, at_gate, hon))


def test_policy_stamp_hashes_all_declared_artifacts():
    stamp = policy_stamp()
    declared = set((load_policy().get("artifacts") or {}).keys())
    assert set(stamp["artifact_sha256"]) == declared
    for digest in stamp["artifact_sha256"].values():
        assert len(digest) == 64
        int(digest, 16)
