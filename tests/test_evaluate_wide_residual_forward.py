from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from analysis.evaluate_wide_residual_forward import (
    FormalLookLockError, evaluate, load_raw_results, run_and_write)


def _policy(*, quality_tickets=300, quality_days=42,
            futility_tickets=1000, formal_tickets=2000, formal_days=273):
    return {
        "schema_version": 1,
        "policy_id": "wide-residual-shadow-v3-test",
        "effective_from": 20260829,
        "status": "shadow_only",
        "real_money_enabled": False,
        "parent_production_policy": "production-test",
        "candidate": {
            "bet_type": "ワイド",
            "take_model_top_n_before_filter": 2,
            "residual_lower_inclusive": 0.0,
            "residual_upper_exclusive": 0.05,
            "max_tickets_per_race": 2,
            "stake_per_ticket_yen": 100,
            "max_odds_mid": 50.0,
        },
        "data_quality_checkpoint": {
            "minimum_settled_tickets": quality_tickets,
            "minimum_distinct_race_days": quality_days,
            "promotion_allowed": False,
            "expected_result": "inconclusive",
        },
        "futility_checkpoint": {
            "minimum_settled_tickets": futility_tickets,
            "promotion_allowed": False,
            "stop_if_absolute_roi_one_sided_95_upper_lte_pct": 100.0,
            "stop_if_delta_vs_primary_one_sided_95_upper_lte_pt": 0.0,
        },
        "formal_look": {
            "minimum_settled_tickets": formal_tickets,
            "minimum_distinct_race_days": formal_days,
            "bootstrap_blocks": "race_day",
            "bootstrap_repetitions": 100,
            "bootstrap_seed": 42,
            "absolute_roi_ci95_lower_gt_pct": 100.0,
            "paired_delta_vs_primary_ci95_lower_gt_pt": 0.0,
            "roi_after_removing_top_3_payouts_gt_pct": 100.0,
            "decision_market_hash_match_pct": 100.0,
            "close_price_coverage_gte_pct": 95.0,
            "number_of_formal_looks": 1,
        },
        "required_stress_reporting": {
            "odds_low_bands": [[0.0, 5.0], [5.0, 10.0], [10.0, None]],
            "remove_top_payout_counts": [1, 2, 3],
            "time_splits": ["first_half", "second_half", "monthly"],
            "leave_one_race_day_out": True,
            "settlement_fail_closed_on_tie_refund_ambiguity": True,
        },
    }


def _write_policy(tmp_path: Path, **kwargs):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(_policy(**kwargs), ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return path


def _stamp(policy_path: Path):
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    return {
        "policy_id": policy["policy_id"],
        "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        "parent_production_policy": policy["parent_production_policy"],
        "status": "shadow_only",
        "real_money_enabled": False,
    }


def _parent_policy(seed="a"):
    digest = (seed * 64)[:64]
    return {
        "policy_id": "production-test",
        "policy_schema_version": 1,
        "policy_sha256": digest,
        "chaos_reference_id": "chaos-test",
        "chaos_reference_sha256": digest,
        "chaos_skip_percentile": .667,
        "chaos_skip_raw_equivalent": .858,
        "artifact_sha256": {
            "rank_model": digest,
            "serve_calibrator": digest,
            "serve_feature_baseline": digest,
        },
    }


def _candidate(selection, p_model, p_market, odds_low):
    residual = p_model - p_market
    return {
        "selection": selection,
        "p_model": p_model,
        "p_market_fair": p_market,
        "residual": residual,
        "odds_t10_low": odds_low,
        "odds_t10_high": odds_low + 1.0,
        "odds_t10_mid": odds_low + 0.5,
        "virtual_stake_yen": 100,
    }


def _race(rid, stamp, market_hash="hash-a", parent_policy=None):
    first = _candidate("1-2", .30, .28, 3.0)
    second = _candidate("1-3", .27, .25, 6.0)
    market_second = _candidate("2-3", .24, .23, 4.0)
    return {
        "race_id": rid,
        "observed_at": f"{rid[:4]}-{rid[4:6]}-{rid[6:8]}T09:50:00",
        "policy": stamp,
        "market_sha256": market_hash,
        "parent_policy": parent_policy or _parent_policy(),
        "real_money_enabled": False,
        "hard_gate_passed": True,
        "hard_gate_reasons": [],
        "triggered": True,
        "arm_a": [first],
        "control_m1": [first, second],
        "control_m2": [market_second, first],
    }


def _write_shadow(shadow_dir: Path, date: str, stamp, races):
    shadow_dir.mkdir(parents=True, exist_ok=True)
    path = shadow_dir / f"{date}_shadow.json"
    path.write_text(json.dumps({"policy": stamp, "races": races}, ensure_ascii=False),
                    encoding="utf-8")


def _result(place="札幌", race_no=1, positions=None):
    positions = positions or {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8}
    return {
        "rows": [{"馬番": horse, "確定着順": position}
                 for horse, position in positions.items()],
        "place": place,
        "race_no": race_no,
        "race_status": "confirmed",
        "refunded_horses": [],
        "non_refunded_zero_horses": [],
    }


def _lineage(*race_ids, parent_policy=None):
    parent_policy = parent_policy or _parent_policy()
    return {
        rid: {"decision_hashes": ["hash-a"],
              "decisions": [{"market_sha256": "hash-a", "policy": parent_policy}],
              "t10_hashes": ["hash-a"], "has_close": True}
        for rid in race_ids
    }


def test_outcomes_do_not_change_frozen_candidate_population(tmp_path):
    policy_path = _write_policy(tmp_path)
    stamp = _stamp(policy_path)
    shadow_dir = tmp_path / "shadow"
    rid = "2026082901010101"
    _write_shadow(shadow_dir, "20260829", stamp, [_race(rid, stamp)])

    result_one = {rid: _result(positions={1: 1, 2: 2, 3: 3, 4: 4,
                                         5: 5, 6: 6, 7: 7, 8: 8})}
    result_two = {rid: _result(positions={4: 1, 5: 2, 6: 3, 1: 4,
                                         2: 5, 3: 6, 7: 7, 8: 8})}
    wide_one = {(2026, 8, 29, "札幌", 1): {"1-2": 300, "1-3": 500, "2-3": 400}}
    wide_two = {(2026, 8, 29, "札幌", 1): {"4-5": 700, "4-6": 800, "5-6": 900}}

    def run(results, wide):
        return evaluate(
            shadow_dir=shadow_dir, policy_path=policy_path,
            result_loader=lambda _date: results,
            wide_map_loader=lambda: wide,
            lineage_loader=lambda _date: _lineage(rid),
            bootstrap_repetitions=20,
        )

    first, second = run(result_one, wide_one), run(result_two, wide_two)
    assert first["candidate_population"] == second["candidate_population"]
    assert first["arms"]["arm_a"]["tickets"] == second["arms"]["arm_a"]["tickets"] == 1
    assert first["arms"]["arm_a"]["return_yen"] != second["arms"]["arm_a"]["return_yen"]
    assert first["promotion"] is second["promotion"] is False


def test_same_race_population_and_unsettled_race_is_excluded_from_all_arms(tmp_path):
    policy_path = _write_policy(tmp_path, quality_tickets=1, quality_days=1)
    stamp = _stamp(policy_path)
    shadow_dir = tmp_path / "shadow"
    rid1 = "2026082901010101"
    rid2 = "2026082901010201"
    _write_shadow(shadow_dir, "20260829", stamp,
                  [_race(rid1, stamp), _race(rid2, stamp)])
    results = {rid1: _result(race_no=1)}  # rid2 is deliberately not settled yet.
    wide = {(2026, 8, 29, "札幌", 1): {"1-2": 300, "1-3": 500, "2-3": 400}}

    report = evaluate(
        shadow_dir=shadow_dir, policy_path=policy_path,
        result_loader=lambda _date: results,
        wide_map_loader=lambda: wide,
        lineage_loader=lambda _date: _lineage(rid1, rid2),
        bootstrap_repetitions=20,
    )

    assert report["candidate_population"]["triggered_valid_races"] == 2
    assert report["quality"]["result_missing"] == 1
    assert report["same_population"]["race_counts_by_arm"] == {
        "arm_a": 1, "control_m1": 1, "control_m2": 1,
    }
    assert report["checkpoints"]["quality"]["reached"] is True
    assert report["checkpoints"]["formal"]["reached"] is False
    assert report["promotion"] is False


def test_zero_position_ambiguity_fails_closed_for_every_arm(tmp_path):
    policy_path = _write_policy(tmp_path)
    stamp = _stamp(policy_path)
    shadow_dir = tmp_path / "shadow"
    rid = "2026082901010101"
    _write_shadow(shadow_dir, "20260829", stamp, [_race(rid, stamp)])
    ambiguous = _result(positions={1: 1, 2: 2, 3: 3, 4: 0,
                                   5: 5, 6: 6, 7: 7, 8: 8})
    wide = {(2026, 8, 29, "札幌", 1): {"1-2": 300, "1-3": 500, "2-3": 400}}

    report = evaluate(
        shadow_dir=shadow_dir, policy_path=policy_path,
        result_loader=lambda _date: {rid: ambiguous},
        wide_map_loader=lambda: wide,
        lineage_loader=lambda _date: _lineage(rid),
        bootstrap_repetitions=10,
    )

    assert report["quality"]["settlement_unresolved"] == 1
    assert report["same_population"]["settled_races"] == 0
    assert all(report["arms"][arm]["races"] == 0
               for arm in ("arm_a", "control_m1", "control_m2"))
    assert report["promotion"] is False


def test_raw_result_loader_assigns_first_fifteen_columns_by_meaning(tmp_path):
    result_dir = tmp_path / "kekka"
    result_dir.mkdir()
    header = ",".join(f"h{i}" for i in range(15))
    row = ",".join([
        "260829", "札幌", "1", "1", "3", "テスト馬", "1",
        "202608290101010103", "250", "120", "", "830", "1400", "2200", "9000",
    ])
    (result_dir / "20260829.csv").write_bytes((header + "\n" + row + "\n").encode("cp932"))

    loaded = load_raw_results("20260829", result_dir)
    parsed = loaded["2026082901010101"]["rows"][0]
    assert parsed["馬番"] == 3
    assert parsed["確定着順"] == 1
    assert parsed["単勝配当"] == "250"
    assert parsed["馬連配当"] == "830"
    assert parsed["三連複配当"] == "2200"


def test_parent_policy_and_artifact_hashes_match_decision_snapshot(tmp_path):
    policy_path = _write_policy(tmp_path, formal_tickets=1, formal_days=1)
    stamp = _stamp(policy_path)
    parent = _parent_policy()
    shadow_dir = tmp_path / "shadow"
    rid = "2026082901010101"
    _write_shadow(shadow_dir, "20260829", stamp,
                  [_race(rid, stamp, parent_policy=parent)])
    wide = {(2026, 8, 29, "札幌", 1):
            {"1-2": 300, "1-3": 500, "2-3": 400}}

    report = evaluate(
        shadow_dir=shadow_dir, policy_path=policy_path,
        result_loader=lambda _date: {rid: _result()},
        wide_map_loader=lambda: wide,
        lineage_loader=lambda _date: _lineage(rid, parent_policy=parent),
        bootstrap_repetitions=10,
    )

    assert report["lineage"]["parent_policy_match_pct"] == 100.0
    assert report["lineage"]["full_lineage_match_pct"] == 100.0
    assert report["lineage"]["cohort_parent_policy_consistent"] is True
    assert report["checkpoints"]["formal"]["conditions"][
        "parent_policy_artifact_hash_match"] is True


def test_missing_parent_artifact_hash_is_invalid_candidate(tmp_path):
    policy_path = _write_policy(tmp_path)
    stamp = _stamp(policy_path)
    parent = _parent_policy()
    del parent["artifact_sha256"]["serve_calibrator"]
    shadow_dir = tmp_path / "shadow"
    rid = "2026082901010101"
    _write_shadow(shadow_dir, "20260829", stamp,
                  [_race(rid, stamp, parent_policy=parent)])

    report = evaluate(
        shadow_dir=shadow_dir, policy_path=policy_path,
        result_loader=lambda _date: {}, wide_map_loader=lambda: {},
        lineage_loader=lambda _date: {}, bootstrap_repetitions=10,
    )

    assert report["quality"]["invalid_candidate"] == 1
    assert report["candidate_population"]["triggered_valid_races"] == 0
    assert report["promotion"] is False


def test_decision_parent_policy_mismatch_blocks_formal_promotion(tmp_path):
    policy_path = _write_policy(tmp_path, formal_tickets=1, formal_days=1)
    stamp = _stamp(policy_path)
    parent = _parent_policy("a")
    decision_parent = _parent_policy("b")
    shadow_dir = tmp_path / "shadow"
    rid = "2026082901010101"
    _write_shadow(shadow_dir, "20260829", stamp,
                  [_race(rid, stamp, parent_policy=parent)])
    wide = {(2026, 8, 29, "札幌", 1):
            {"1-2": 300, "1-3": 500, "2-3": 400}}

    report = evaluate(
        shadow_dir=shadow_dir, policy_path=policy_path,
        result_loader=lambda _date: {rid: _result()},
        wide_map_loader=lambda: wide,
        lineage_loader=lambda _date: _lineage(rid, parent_policy=decision_parent),
        bootstrap_repetitions=10,
    )

    assert report["checkpoints"]["formal"]["reached"] is True
    assert report["lineage"]["parent_policy_match_pct"] == 0.0
    assert report["checkpoints"]["formal"]["conditions"][
        "parent_policy_artifact_hash_match"] is False
    assert report["promotion"] is False


def test_mixed_parent_policy_stamps_are_rejected_from_one_cohort(tmp_path):
    policy_path = _write_policy(tmp_path)
    stamp = _stamp(policy_path)
    shadow_dir = tmp_path / "shadow"
    rid1 = "2026082901010101"
    rid2 = "2026082901010201"
    _write_shadow(
        shadow_dir, "20260829", stamp,
        [_race(rid1, stamp, parent_policy=_parent_policy("a")),
         _race(rid2, stamp, parent_policy=_parent_policy("b"))],
    )

    report = evaluate(
        shadow_dir=shadow_dir, policy_path=policy_path,
        result_loader=lambda _date: {}, wide_map_loader=lambda: {},
        lineage_loader=lambda _date: {}, bootstrap_repetitions=10,
    )

    assert report["quality"]["parent_policy_cohort_mismatch"] == 1
    assert report["quality"]["invalid_candidate"] == 1
    assert report["candidate_population"]["triggered_valid_races"] == 1
    assert report["lineage"]["cohort_parent_policy_consistent"] is False
    assert report["promotion"] is False


def test_formal_report_is_locked_before_new_shadow_data_can_trigger_second_look(tmp_path):
    policy_path = _write_policy(tmp_path, formal_tickets=1, formal_days=1)
    stamp = _stamp(policy_path)
    parent = _parent_policy()
    shadow_dir = tmp_path / "shadow"
    output = tmp_path / "formal.json"
    rid1 = "2026082901010101"
    rid2 = "2026082901010201"
    _write_shadow(shadow_dir, "20260829", stamp,
                  [_race(rid1, stamp, parent_policy=parent)])
    results = {rid1: _result(race_no=1)}
    wide = {(2026, 8, 29, "札幌", 1):
            {"1-2": 300, "1-3": 500, "2-3": 400}}

    first, first_locked = run_and_write(
        out=output, shadow_dir=shadow_dir, policy_path=policy_path,
        result_loader=lambda _date: results,
        wide_map_loader=lambda: wide,
        lineage_loader=lambda _date: _lineage(rid1, parent_policy=parent),
        bootstrap_repetitions=10,
    )
    assert first["checkpoints"]["formal"]["reached"] is True
    assert first_locked is False
    original_bytes = output.read_bytes()
    original_manifest = first["candidate_population"]["manifest_sha256"]

    # Add a second, settled race which would change a fresh evaluation.
    _write_shadow(
        shadow_dir, "20260829", stamp,
        [_race(rid1, stamp, parent_policy=parent),
         _race(rid2, stamp, parent_policy=parent)],
    )
    results[rid2] = _result(race_no=2)
    wide[(2026, 8, 29, "札幌", 2)] = {
        "1-2": 600, "1-3": 700, "2-3": 800}
    fresh = evaluate(
        shadow_dir=shadow_dir, policy_path=policy_path,
        result_loader=lambda _date: results,
        wide_map_loader=lambda: wide,
        lineage_loader=lambda _date: _lineage(rid1, rid2, parent_policy=parent),
        bootstrap_repetitions=10,
    )
    assert fresh["candidate_population"]["manifest_sha256"] != original_manifest

    locked_report, locked = run_and_write(
        out=output, shadow_dir=shadow_dir, policy_path=policy_path,
        # If the lock is checked too late, these loaders make the test fail.
        result_loader=lambda _date: (_ for _ in ()).throw(AssertionError("second look")),
        wide_map_loader=lambda: (_ for _ in ()).throw(AssertionError("second look")),
        lineage_loader=lambda _date: (_ for _ in ()).throw(AssertionError("second look")),
        bootstrap_repetitions=10,
    )
    assert locked is True
    assert locked_report["candidate_population"]["manifest_sha256"] == original_manifest
    assert output.read_bytes() == original_bytes


def test_existing_formal_report_from_another_policy_fails_closed(tmp_path):
    policy_path = _write_policy(tmp_path)
    output = tmp_path / "formal.json"
    existing = {
        "policy_sha256": "0" * 64,
        "candidate_population": {"manifest_sha256": "1" * 64},
        "checkpoints": {"formal": {"reached": True}},
    }
    output.write_text(json.dumps(existing), encoding="utf-8")
    before = output.read_bytes()

    with pytest.raises(FormalLookLockError, match="another policy"):
        run_and_write(out=output, policy_path=policy_path)
    assert output.read_bytes() == before


def test_existing_collecting_report_can_be_updated(tmp_path):
    policy_path = _write_policy(tmp_path)
    output = tmp_path / "collecting.json"
    output.write_text(json.dumps({
        "policy_sha256": _stamp(policy_path)["policy_sha256"],
        "checkpoints": {"formal": {"reached": False}},
    }), encoding="utf-8")

    report, locked = run_and_write(
        out=output, policy_path=policy_path,
        result_loader=lambda _date: {}, wide_map_loader=lambda: {},
        lineage_loader=lambda _date: {}, bootstrap_repetitions=10,
    )
    assert locked is False
    assert report["checkpoints"]["state"] == "collecting_quality"
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == 1


def _races_with_zero():
    return {
        "2026082901010101": {
            "rows": [
                {"馬番": 1, "確定着順": 1},
                {"馬番": 2, "確定着順": 2},
                {"馬番": 3, "確定着順": 3},
                {"馬番": 7, "確定着順": 0},     # 取消 or 競走中止
                {"馬番": 8, "確定着順": 0},
            ],
            "race_status": "confirmed",
            "refunded_horses": [],
            "non_refunded_zero_horses": [],
            "zero_classification": "none",
        }
    }


def test_zero_positions_fall_back_to_conservative_started(tmp_path):
    """変更フィードが無い開催日は「全て出走扱い」に倒す。

    出走扱い = 券を返還でなく外れとして決済する = ROI を過小評価する側であり、
    昇格判定を緩める方向には決して働かない。
    """
    from analysis.evaluate_wide_residual_forward import classify_zero_positions

    races = _races_with_zero()
    classify_zero_positions("20260829", races, changes_dir=tmp_path)
    rec = races["2026082901010101"]

    assert rec["refunded_horses"] == []
    assert rec["non_refunded_zero_horses"] == [7, 8]
    assert rec["zero_classification"] == "conservative_all_started"


def test_zero_positions_use_jv_change_feed_when_present(tmp_path):
    """JV-Link AV レコード (t10.ps1 → changes.ps1) があれば返還を権威的に分類する。"""
    from analysis.evaluate_wide_residual_forward import classify_zero_positions

    (tmp_path / "20260829.json").write_text(json.dumps({
        "date": "20260829",
        "races": {"2026082901010101": {
            "cancels": [{"umaban": 7, "name": "テスト", "kind": "出走取消"}]}},
    }, ensure_ascii=False), encoding="utf-8")

    races = _races_with_zero()
    classify_zero_positions("20260829", races, changes_dir=tmp_path)
    rec = races["2026082901010101"]

    assert rec["refunded_horses"] == [7]        # 取消 = 返還
    assert rec["non_refunded_zero_horses"] == [8]   # 中止 = 出走して外れ
    assert rec["zero_classification"] == "jv_change_feed"
