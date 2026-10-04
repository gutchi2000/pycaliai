# -*- coding: utf-8 -*-
"""観測計画 v2.1 Dry 修正（2026-10-05）: 監査側 F1〜F3・B3〜B5・タスク記録の検出（F7）・監査の二重起動防止。"""
from __future__ import annotations

import gzip
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import forward_prices as FP  # noqa: E402
import jv_journal  # noqa: E402
import jv_records as JR  # noqa: E402
import obs_task_runs as TR  # noqa: E402
from test_obs_phase2 import ALL5, R2, RID, SPEC_OF, STAMP, _event, _full_day, _put, build  # noqa: E402

PROD = Path(r"E:\PyCaLiAI")


# ---------------------------------------------------------------- F2: 期待組数（発走前は出走頭数欄が 0）
@pytest.mark.parametrize("kind", ["O1", "O2", "O3", "O4", "O5"])
@pytest.mark.parametrize("n_reg,scr", [(12, ()), (16, (3, 9)), (18, (18,)), (10, (1,))])
def test_prerace_record_with_zero_running_field_uses_registered_minus_scratched(kind, n_reg, scr):
    s = JR.structure_record(build(kind, n_reg=n_reg, scratched=scr, n_run=0, kubun="1"), SPEC_OF[kind], RID)
    assert s["n_running"] == 0 and s["structure_version"] == JR.STRUCTURE_VERSION
    assert s["complete"], s["anomalies"]
    for block, c in s["counts"].items():
        if block == "wakuren":
            continue
        assert c["priced_match"] is True and c["running_basis"] == "registered_minus_scratched"
        assert c["scratched"] == sorted(scr)


def test_missing_combo_is_still_a_mismatch_when_running_field_is_zero():
    rec = build("O2", n_reg=12, n_run=0, kubun="1")
    k = rec.index("0102")                                   # 1-2 の組を未発売 filler にする（取消ではない欠け）
    rec = rec[:k + 4] + "*" * 9 + rec[k + 13:]
    s = JR.structure_record(rec, "0B32", RID)
    c = s["counts"]["umaren"]
    assert c["scratched"] == [] and c["priced_match"] is False and not s["complete"]


def test_record_without_any_priced_slot_is_underivable_and_incomplete():
    rec = build("O5", n_reg=10, n_run=0, kubun="1", scratched=tuple(range(1, 11)))
    s = JR.structure_record(rec, "0B35", RID)
    c = s["counts"]["trio"]
    assert c["running_basis"] == "underivable" and c["priced_match"] is False and not s["complete"]


def test_wakuren_not_on_sale_expects_no_slots():
    rec = build("O1", n_reg=8, n_run=0, kubun="1")
    body = rec.rstrip("\r\n")
    body = body[:41] + "0" + body[42:603] + " " * (927 - 603) + body[927:]
    s = JR.structure_record(body + "\r\n", "0B31", RID)
    c = s["counts"]["wakuren"]
    assert c["on_sale"] is False and c["expected_nonblank"] == 0 and c["nonblank_match"] is True
    on = JR.structure_record(build("O1", n_reg=8, n_run=0, kubun="1"), "0B31", RID)["counts"]["wakuren"]
    assert on["on_sale"] is True and on["expected_nonblank"] == JR.wakuren_slot_count(8)


def test_final_record_running_field_disagreeing_with_derived_is_an_anomaly():
    s = JR.structure_record(build("O2", n_reg=12, scratched=(3,), n_run=12), "0B32", RID)
    assert any("shusso field 12" in a for a in s["anomalies"]) and not s["complete"]


def _real_scratch_records():
    out = []
    for root in ("forward_prices", "forward_prices_dry"):
        for p in sorted((PROD / "data" / root / "20261004").glob("2026100405040205_*.json.gz")):
            rec = FP.read_snapshot(p)
            for cap in rec.get("jv_captures") or []:
                if cap.get("records"):
                    out.append((FP.canonical_stage(rec["stage"]), cap["spec"], cap["records"][-1]))
    return out


REAL_SCRATCH = _real_scratch_records() if (PROD / "data" / "forward_prices" / "20261004").is_dir() else []


@pytest.mark.skipif(not REAL_SCRATCH, reason="10/04 東京 5R（取消 1 頭）の実録が手元に無い")
def test_real_records_with_a_scratch_match_expected_counts_in_every_stage():
    """B3: 取消を含む実録（10/04 東京 5R、10 番取消）。発走前（出走頭数欄 0）も確定後も期待値と一致する。"""
    stages, scr_by_stage = set(), {}
    for st, spec, r in REAL_SCRATCH:
        s = JR.structure_record(r["raw"], spec, "2026100405040205", r.get("stream", "rt"))
        stages.add((st, s["kubun"]))
        for block, c in s["counts"].items():
            if block == "wakuren":
                assert c["nonblank_match"] is True, (st, spec)
                continue
            assert c["priced_match"] is True, (st, spec, block, c)
            scr_by_stage.setdefault(st, set()).add(tuple(c["scratched"]))
    assert all(len(v) == 1 for v in scr_by_stage.values()), scr_by_stage      # stage 内の全券種で取消馬が一致
    assert {(10,)} in scr_by_stage.values()
    assert {k for _, k in stages} >= {"1", "4"}                   # 発走前（区分 1）と確定後（区分 4）の両方


# ---------------------------------------------------------------- F1 / F3 / B4 / B5: 監査
def _audit(**kw):
    from analysis import obs_stage0_audit as A
    return A.audit(["20261003"], **kw)


def _split_day(tmp_path):
    """F1: 観測 collector の stage は dry store、本番経路の stage は本番 store に置く。"""
    root, jroot, sched = _full_day(tmp_path)
    dry_root, prod_root = tmp_path / "fwd_dry", tmp_path / "fwd_prod"
    for p in sorted((root / "20261003").glob("*.json.gz")):
        st = FP.canonical_stage(FP.read_snapshot(p)["stage"])
        dest = (prod_root if st in ("t10", "close_late") else dry_root) / "20261003"
        dest.mkdir(parents=True, exist_ok=True)
        p.rename(dest / p.name)
    return dry_root, prod_root, jroot, sched


def test_dry_audit_reads_production_stages_from_the_production_store(tmp_path):
    from analysis import obs_stage0_audit as A
    dry_root, prod_root, jroot, sched = _split_day(tmp_path)
    old = _audit(forward_root=dry_root, journal_root=jroot, schedule=sched, scope="night",
                 task_runs_root=tmp_path / "runs")
    assert old["missing"]["t10/0B31"]["missing_count"] == 2               # 旧来の読み方（dry だけ）なら欠損
    sources = [{"name": "dry", "forward_root": dry_root, "journal_root": jroot, "stages": A.OBS_STAGES},
               {"name": "production", "forward_root": prod_root, "journal_root": tmp_path / "nojournal",
                "stages": A.PROD_STAGES}]
    rep = _audit(sources=sources, schedule=sched, scope="night", task_runs_root=tmp_path / "runs")
    assert rep["contract"]["met"], rep["contract"]
    assert rep["stage_sources"]["t10"] == ["production"] and rep["stage_sources"]["t2_candidate"] == ["dry"]


def test_dry_sources_ignore_production_stages_found_in_the_dry_store(tmp_path):
    from analysis import obs_stage0_audit as A
    dry_root, prod_root, jroot, sched = _split_day(tmp_path)
    sources = [{"name": "dry", "forward_root": dry_root, "journal_root": jroot, "stages": A.OBS_STAGES},
               {"name": "production", "forward_root": tmp_path / "empty", "journal_root": tmp_path / "nojournal",
                "stages": A.PROD_STAGES}]
    for p in sorted((prod_root / "20261003").glob("*.json.gz")):          # 本番 stage の録を dry に置いても数えない
        p.rename(dry_root / "20261003" / p.name)
    rep = _audit(sources=sources, schedule=sched, scope="night", task_runs_root=tmp_path / "runs")
    assert rep["missing"]["t10/0B31"]["missing_count"] == 2 and "t10" not in rep["stage_sources"]


def test_night_scope_excludes_stock_from_the_contract(tmp_path):
    root, jroot, sched = _full_day(tmp_path, drop=("final_stock_candidate",))
    night = _audit(forward_root=root, journal_root=jroot, schedule=sched, scope="night", task_runs_root=tmp_path / "r")
    assert night["contract"]["met"] and night["missing"]["final_stock_candidate/0B31"]["required"] is False
    stock = _audit(forward_root=root, journal_root=jroot, schedule=sched, scope="stock", task_runs_root=tmp_path / "r")
    assert stock["decision"] == "CONTRACT_NOT_MET"


def test_overlap_primary_is_interval_intersection_and_30s_window_is_descriptive(tmp_path):
    """t10 と三連複を 20 秒ずらす: ±30 秒では重なりだが、取得区間は交差しない。"""
    root, jroot, sched = tmp_path / "fwd", tmp_path / "journal" / "20261003", None
    for k, rid in enumerate((RID, R2)):
        m = 20 + 20 * k
        _put(root, jroot, rid, "t10", ALL5[:4], f"15:{m:02d}:00", 200 + k, "jvlink_odds")
        _put(root, jroot, rid, "trio_t10", ["0B35", "0B31"], f"15:{m:02d}:20", 300 + k, "trio_shadow")
    sched = {"20261003": {RID: "2026-10-03T15:30:00", R2: "2026-10-03T15:50:00"}}
    rep = _audit(forward_root=root, journal_root=tmp_path / "journal", schedule=sched, scope="night",
                 task_runs_root=tmp_path / "r")
    assert rep["concurrency"]["definition"] == "intersection" and rep["concurrency"]["overlapped"]["n"] == 0
    assert rep["concurrency_window30_descriptive"]["overlapped"]["n"] > 0
    assert rep["concurrency_verdict"] == "INSUFFICIENT_OVERLAP_OBSERVED"


def test_concurrency_success_does_not_depend_on_the_stored_capture_ok(tmp_path):
    """F3: 保存時の cap.ok（旧版の式で False）に関係なく、今の式で成否を出す。"""
    root, jroot, sched = _full_day(tmp_path)
    for p in sorted((root / "20261003").glob("*.json.gz")):
        rec = FP.read_snapshot(p)
        for cap in rec.get("jv_captures") or []:
            cap["ok"] = False
            for r in cap["records"]:
                r.pop("structure_version", None)                      # 旧版の録（jvrec-1）
                for c in r["counts"].values():
                    c["priced_match"] = False
        with gzip.open(p, "wt", encoding="utf-8") as fh:
            json.dump(rec, fh)
    rep = _audit(forward_root=root, journal_root=tmp_path / "journal", schedule=sched, scope="full",
                 task_runs_root=tmp_path / "r")
    assert rep["concurrency"]["overlapped"]["rate"] == 1.0 and rep["contract"]["met"], rep["contract"]
    assert "restructure_equal" not in rep["raw_parser"]                    # 旧版どうしは比べない（B5）
    rv = rep["restructure_version"]
    assert rv["older_records_not_compared"]["jvrec-1"] > 0 and rv["older_records_that_would_differ"]["jvrec-1"] > 0


def test_same_version_restructure_difference_is_still_a_failure(tmp_path):
    root, jroot, sched = _full_day(tmp_path)
    p = sorted((root / "20261003").glob("*_t2_candidate_*.json.gz"))[0]
    rec = FP.read_snapshot(p)
    rec["jv_captures"][0]["records"][-1]["kubun"] = "9"                 # 同じ版で保存値と再構造化が食い違う
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        json.dump(rec, fh)
    rep = _audit(forward_root=root, journal_root=tmp_path / "journal", schedule=sched, scope="full",
                 task_runs_root=tmp_path / "r")
    assert rep["raw_parser"]["restructure_equal"]["fail"] == 1 and rep["decision"] == "CONTRACT_NOT_MET"


def test_scratch_inconsistency_between_specs_is_a_coverage_mismatch(tmp_path):
    root, jroot = tmp_path / "fwd", tmp_path / "journal" / "20261003"
    caps = []
    for spec, scr in (("0B31", (3,)), ("0B32", (5,))):                # 単勝は 3 番、馬連は 5 番が取消
        kind = next(k for k, s in SPEC_OF.items() if s == spec)
        meta = {"fetch_started_at": f"2026-10-03T15:28:00.10{len(caps)}+09:00",
                "fetch_finished_at": f"2026-10-03T15:28:00.60{len(caps)}+09:00", "rc_init": 0, "rc_open": 0,
                "error": None, "stream": "rt"}
        caps.append(JR.capture(spec, RID, [build(kind, n_reg=12, scratched=scr, n_run=0, kubun="1")], meta))
    FP.archive_market_snapshot({"race_id": RID, "fetched": caps[0]["fetch_started_at"]}, "t2_candidate",
                               stamp=STAMP, root=root, captures=caps, scheduled_post="2026-10-03T15:30:00")
    rep = _audit(forward_root=root, journal_root=tmp_path / "journal", schedule={"20261003": {RID: "x"}},
                 scope="night", task_runs_root=tmp_path / "r")
    assert rep["coverage"]["t2_candidate/0B32"]["scratch_consistency"]["mismatch"] == 1
    assert "t2_candidate/0B32/scratch_consistency" in rep["coverage_fail_over_1pct"]


# ---------------------------------------------------------------- F7: タスクの開始・終了記録
def _runs(root, date, items):
    for task, rid, end in items:
        run = {"run_id": f"{task}{rid}", "task": task, "date": date, "race_id": rid if task in ("t2", "trio") else None,
               "started_at": "2026-10-03T15:00:00.000+09:00"}
        TR.write_start(run, dry=True, root=root)
        if end is not None:
            TR.write_end(run, exit_code=end, dry=True, root=root, child_exit_codes=[end])


def test_task_runs_complete_pass_and_killed_run_is_attributed(tmp_path):
    root, jroot, sched = _full_day(tmp_path, drop_races=(("final_rt_candidate", R2),))
    runs = tmp_path / "runs"
    _runs(runs, "20261003", [(t, r, 0) for t in ("t2", "trio") for r in (RID, R2)] + [("final", None, None)])
    rep = _audit(forward_root=root, journal_root=tmp_path / "journal", schedule=sched, scope="night",
                 task_runs_root=runs, task_runs_required_from="20261003")
    tr = rep["task_runs"]["20261003"]
    assert tr["recorded"] and tr["started"] == 5 and tr["ended"] == 4 and len(tr["killed"]) == 1
    miss = rep["missing"]["final_rt_candidate/0B31"]["races"]
    assert [m["cause"] for m in miss] == ["task_killed"]
    assert any("not 100%" in r for r in rep["contract"]["reasons"])


def test_ctrl_c_exit_under_windowless_launch_is_flagged(tmp_path):
    root, jroot, sched = _full_day(tmp_path)
    runs = tmp_path / "runs"
    _runs(runs, "20261003", [(t, r, 0) for t in ("t2", "trio") for r in (RID, R2)] + [("final", None, 0xC000013A)])
    rep = _audit(forward_root=root, journal_root=tmp_path / "journal", schedule=sched, scope="night",
                 task_runs_root=runs, task_runs_required_from="20261003")
    assert any("0xC000013A" in r for r in rep["contract"]["reasons"]) and rep["decision"] == "CONTRACT_NOT_MET"


def test_task_runs_absent_before_activation_date_are_descriptive(tmp_path):
    root, jroot, sched = _full_day(tmp_path)
    rep = _audit(forward_root=root, journal_root=tmp_path / "journal", schedule=sched, scope="night",
                 task_runs_root=tmp_path / "none")
    assert rep["task_runs"]["20261003"] == {"recorded": False, "expected": 5, "required": False}
    assert rep["contract"]["met"]
    req = _audit(forward_root=root, journal_root=tmp_path / "journal", schedule=sched, scope="night",
                 task_runs_root=tmp_path / "none", task_runs_required_from="20261003")
    assert any("no task run records" in r for r in req["contract"]["reasons"])


def test_task_run_records_are_exclusive_and_never_overwritten(tmp_path):
    run = {"run_id": "abc", "task": "final", "date": "20261003", "started_at": "2026-10-03T18:30:00.000+09:00"}
    TR.write_start(run, dry=True, root=tmp_path)
    with pytest.raises(FileExistsError):
        TR.write_start(run, dry=True, root=tmp_path)
    assert TR.run_state(TR.read_runs("20261003", dry=True, root=tmp_path), ("final", "20261003")) == "killed"


# ---------------------------------------------------------------- 監査の二重起動防止
def test_audit_lock_blocks_a_second_run_and_replaces_only_stale_locks(tmp_path):
    from analysis import obs_stage0_audit as A
    lock = tmp_path / "a.lock"
    assert A.acquire_lock(lock)[0] is True
    ok, why = A.acquire_lock(lock)
    assert ok is False and "lock held" in why
    old = time.time() - 4000
    os.utime(lock, (old, old))
    ok, why = A.acquire_lock(lock, stale_sec=1800)
    assert ok is True and "stale" in why
    A.release_lock(lock)
    assert not lock.exists()


def test_recent_identical_ledger_row_is_detected(tmp_path, monkeypatch):
    from analysis import obs_stage0_audit as A
    monkeypatch.setattr(A, "_git_head", lambda: "h1")
    ledger = tmp_path / "ledger.jsonl"
    rep = {"dates": ["20261010"], "scope": "night", "decision": "CONTRACT_NOT_MET", "exit_code": 4,
           "contract": {"met": False, "reasons": []}, "concurrency_reasons": []}
    A.record_ledger(rep, ledger, dry=True, report_path=str(tmp_path / "obs_stage0_20261010_dry_night.json"))
    assert A.recent_duplicate(ledger, ["20261010"], True, "night", "obs_stage0_20261010_dry_night")
    assert A.recent_duplicate(ledger, ["20261010"], True, "stock", "obs_stage0_20261010_dry_stock") is None
    assert A.recent_duplicate(ledger, ["20261010"], False, "night", "obs_stage0_20261010_night") is None


def test_main_returns_already_running_and_writes_nothing_when_locked(tmp_path, monkeypatch):
    from analysis import obs_stage0_audit as A
    lock = tmp_path / "held.lock"
    lock.write_text("x")
    monkeypatch.setattr(A, "LOCK_PATH", lock)
    monkeypatch.setattr(A, "STAGE0_LEDGER", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(sys, "argv", ["obs_stage0_audit", "--dates", "20261010", "--dry",
                                      "--out", str(tmp_path / "rep.json")])
    monkeypatch.setattr(A, "acquire_lock", lambda path=lock, stale_sec=0: (False, "lock held"))
    assert A.main() == A.EXIT_ALREADY_RUNNING
    assert not (tmp_path / "ledger.jsonl").exists() and not (tmp_path / "rep.json").exists()
