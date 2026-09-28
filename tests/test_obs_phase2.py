# -*- coding: utf-8 -*-
"""
test_obs_phase2.py — 観測計画 v2.1 Phase 2（shadow collector）の回帰テスト
=========================================================================
JV-Link は呼ばない。録は実録と同じレイアウトの合成録（CRLF 付き）で作る。実録 raw が手元にあれば
（E:/PyCaLiAI/reports/...）読取専用で parser を照合し、無ければその 1 本だけ skip する。

python -m pytest tests/test_obs_phase2.py -q
"""
from __future__ import annotations

import gzip
import importlib.util
import itertools
import json
import subprocess
import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import forward_prices as FP  # noqa: E402
import jv_journal  # noqa: E402
import jv_records as JR  # noqa: E402

RID = "2026100306040911"
STAMP = {"policy_id": "test-policy", "policy_sha256": "a" * 64}
BASE_COMMIT = "bf952d3f"


# ---------------------------------------------------------------- synthetic JV records
def _head(kind, rid, n_reg, n_run, kubun="4", announce="10031530"):
    return f"{kind}{kubun}20261003{rid}{announce}{n_reg:02d}{n_run:02d}"


def _o(v, w):
    return f"{int(round(v * 10)):0{w}d}"


def build(kind, rid=RID, n_reg=12, n_run=None, scratched=(), announce="10031530", kubun="4"):
    """実録と同じ固定長の合成録（末尾 CRLF）。scratched の馬を含む組は '*' 埋め（未発売）。"""
    n_run = n_reg - len(scratched) if n_run is None else n_run
    s = set(scratched)
    head = _head(kind, rid, n_reg, n_run, kubun, announce)
    if kind == "O1":
        tan = "".join((f"{i:02d}" + ("****" if i in s else _o(0.8 * n_reg + 0.1 * i, 4)) + "01") if i <= n_reg else " " * 8
                      for i in range(1, 29))
        fuk = "".join((f"{i:02d}" + ("********" if i in s else _o(1.1 + i / 10, 4) + _o(1.3 + i / 10, 4)) + "01")
                      if i <= n_reg else " " * 12 for i in range(1, 29))
        frames = min(n_reg, 8)
        same = min(8, max(0, n_reg - 8))
        waku = ""
        for a, b in itertools.combinations_with_replacement(range(1, 9), 2):
            valid = (a < b and b <= frames) or (a == b and a > 8 - same)
            waku += (f"{a}{b}" + _o(3.0 + a + b, 5) + "01") if valid else " " * 9
        body = head + "7773" + tan + fuk + waku + "00000100000" + "00000200000" + "00000030000"
    elif kind in ("O2", "O3"):
        slots = ""
        for i, j in itertools.combinations(range(1, 19), 2):
            if j > n_reg:
                slots += " " * (13 if kind == "O2" else 17)
                continue
            if i in s or j in s:
                slots += f"{i:02d}{j:02d}" + "*" * (9 if kind == "O2" else 13)
            elif kind == "O2":
                slots += f"{i:02d}{j:02d}" + _o(10 + i + j, 6) + "001"
            else:
                slots += f"{i:02d}{j:02d}" + _o(2 + i, 5) + _o(3 + j, 5) + "001"
        body = head + "7" + slots + "00000500000"
    elif kind == "O4":
        slots = ""
        for i in range(1, 19):
            for j in range(1, 19):
                if i == j:
                    continue
                if max(i, j) > n_reg:
                    slots += " " * 13
                elif i in s or j in s:
                    slots += f"{i:02d}{j:02d}" + "*" * 9
                else:
                    slots += f"{i:02d}{j:02d}" + _o(20 + i + j, 6) + "001"
        body = head + "7" + slots + "00000400000"
    elif kind == "O5":
        slots = ""
        for a, b, c in itertools.combinations(range(1, 19), 3):
            if c > n_reg:
                slots += " " * 15
            elif {a, b, c} & s:
                slots += f"{a:02d}{b:02d}{c:02d}" + "*" * 9
            else:
                slots += f"{a:02d}{b:02d}{c:02d}" + _o(50 + a + b + c, 6) + "001"
        body = head + "7" + slots + "00000900000"
    else:
        raise ValueError(kind)
    assert len(body) == JR.RECORD_LEN[kind], (kind, len(body))
    return body + "\r\n"


SPEC_OF = {"O1": "0B31", "O2": "0B32", "O3": "0B33", "O4": "0B34", "O5": "0B35"}


def fake_fetch(recs_by_spec, ok=True):
    def fetch(rid, spec, max_rec=200):
        meta = {"fetch_started_at": "2026-10-03T15:28:00.100+09:00", "fetch_finished_at": "2026-10-03T15:28:00.900+09:00",
                "rc_init": 0, "rc_open": 0 if ok else -1, "error": None, "stream": "rt"}
        recs = recs_by_spec.get(spec, []) if ok else []
        meta["n_records_returned"] = len(recs)
        return recs, meta
    return fetch


# ---------------------------------------------------------------- jv_records
@pytest.mark.parametrize("kind", ["O1", "O2", "O3", "O4", "O5"])
@pytest.mark.parametrize("n_reg,scr", [(12, ()), (16, (3, 9)), (8, ()), (18, (18,))])
def test_structure_record_counts_flags_votes_and_validated_parser(kind, n_reg, scr):
    s = JR.structure_record(build(kind, n_reg=n_reg, scratched=scr), SPEC_OF[kind], RID)
    n_run = n_reg - len(scr)
    assert s["length_ok"] and s["race_key_ok"] and s["validated_parser_match"] is True
    assert s["complete"] and not s["anomalies"]
    assert s["kubun"] == "4" and s["announced_at"] == "2026-10-03T15:30+09:00"
    assert (s["n_registered"], s["n_running"]) == (n_reg, n_run)
    blocks = s["counts"]
    if kind == "O1":
        assert blocks["tansho"]["priced"] == n_run and blocks["tansho"]["unpriced"] == len(scr)
        assert blocks["wakuren"]["nonblank_match"] is True
        assert s["votes_total"] == {"tansho": 100000, "fukusho": 200000, "wakuren": 30000}
        assert s["hatsubai_flag"]["hatsubai_wakuren"] == "7" and len(s["wakuren_block_raw"]) == 324
    else:
        (b, c), = blocks.items()
        assert c["priced"] == c["expected_priced"] and c["nonblank"] == c["expected_nonblank"]
        assert c["unpriced"] == c["expected_nonblank"] - c["expected_priced"]
        assert s["hatsubai_flag"] == {"hatsubai": "7"}
    assert s["raw"].endswith("\r\n")


def test_anomalies_race_key_length_and_announce_0000():
    s = JR.structure_record(build("O3"), "0B33", "2026100306040912")
    assert not s["race_key_ok"] and not s["complete"]
    s = JR.structure_record(build("O3")[:-30] + "\r\n", "0B33", RID)
    assert not s["length_ok"] and not s["complete"]
    s = JR.structure_record(build("O5", announce="10030000"), "0B35", RID)
    assert "announce_0000" in s["anomalies"]
    s = JR.structure_record(build("O5", announce="10030000"), "0B35", RID, stream="stock")
    assert "announce_0000" not in s["anomalies"]           # 蓄積系の確定録は発表時分 0 が正常
    assert JR.announced_at("00000000", RID) is None


def test_kubun_value_is_stored_not_interpreted():
    for k in "012345679":
        s = JR.structure_record(build("O2", kubun=k), "0B32", RID)
        assert s["kubun"] == k and s["complete"]
    assert not any("final" in str(v) for v in JR.structure_record(build("O2"), "0B32", RID).values()
                   if isinstance(v, str) and v != "rt")


REAL = sorted(Path("E:/PyCaLiAI/reports/live_odds/raw").glob("*_0B3*.txt")) + \
    sorted(Path("E:/PyCaLiAI/reports/trio_portfolio_shadow_v2/raw/_smoke").glob("*/*_0B3*.txt"))


@pytest.mark.skipif(not REAL, reason="実録 raw が手元に無い")
def test_real_jv_records_parse_complete_and_match_validated_parsers():
    import re
    for f in REAL:
        spec = re.search(r"(0B3\d)", f.name).group(1)
        rid = re.search(r"(\d{16})", str(f)).group(1)
        rec = [r for r in f.read_text(encoding="utf-8", errors="replace").split("\n") if r.strip()][-1]
        s = JR.structure_record(rec, spec, rid)
        assert s["length_ok"] and s["race_key_ok"] and s["complete"], (f, s["anomalies"])
        assert s["validated_parser_match"] is True, f


def test_capture_keeps_all_records_and_marks_latest():
    recs = [build("O2", announce="10031520"), build("O2", announce="10031530")]
    cap = JR.capture("0B32", RID, recs, {"fetch_started_at": "x", "fetch_finished_at": "y"})
    assert cap["n_records_kind"] == 2 and cap["ok"] and cap["announced_at"] == "2026-10-03T15:30+09:00"


# ---------------------------------------------------------------- forward_prices schema v2
def _market(stage_time="2026-10-03T15:20:00.100"):
    return {"race_id": RID, "fetched": stage_time, "ok": True, "tansho": {"1": 2.5}}


def test_close_is_stored_as_close_late_and_readers_alias(tmp_path):
    p = FP.archive_market_snapshot(_market(), "close", stamp=STAMP, root=tmp_path)
    rec = FP.read_snapshot(p)
    assert rec["stage"] == "close_late" and rec["stage_requested"] == "close"
    assert "_close_late_" in p.name and rec["schema_version"] == 2
    assert FP.canonical_stage("close") == "close_late" and FP.canonical_stage("t10") == "t10"
    with pytest.raises(ValueError):
        FP.archive_market_snapshot(_market(), "final", stamp=STAMP, root=tmp_path)


def test_v1_record_remains_readable(tmp_path):
    v1 = {"schema_version": 1, "record_type": "market_snapshot", "stage": "close", "race_id": RID,
          "observed_at": "2026-09-27T13:56:00", "market": {}}
    p = tmp_path / RID[:8] / f"{RID}_close_20260927135600000_abc.json.gz"
    p.parent.mkdir(parents=True)
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        json.dump(v1, fh)
    rec = FP.read_snapshot(p)
    assert FP.canonical_stage(rec["stage"]) == "close_late"


def test_captures_are_stored_beside_the_unchanged_market(tmp_path):
    cap = JR.capture("0B35", RID, [build("O5")], {"fetch_started_at": "a", "fetch_finished_at": "b"})
    m = _market()
    p = FP.archive_market_snapshot(m, "t2_candidate", stamp=STAMP, root=tmp_path, captures=[cap])
    rec = FP.read_snapshot(p)
    assert rec["market"] == m and rec["jv_captures"][0]["records"][0]["raw"] == build("O5")
    assert rec["source"] == "JV-Link:0B35" and rec["market_sha256"] == FP.payload_sha256(m)


# ---------------------------------------------------------------- jvlink_odds: production path unchanged
def _load_base_jvlink_odds(tmp_path):
    src = subprocess.run(["git", "-c", "safe.directory=*", "show", f"{BASE_COMMIT}:jvlink_odds.py"],
                         cwd=BASE, capture_output=True).stdout
    p = tmp_path / "jvlink_odds_base.py"
    p.write_bytes(src)
    spec = importlib.util.spec_from_file_location("jvlink_odds_base", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _recs_by_spec(n_reg=14, scratched=(5,)):
    return {SPEC_OF[k]: [build(k, n_reg=n_reg, scratched=scratched)] for k in ("O1", "O2", "O3", "O4")}


def _run_main(mod, argv, monkeypatch, tmp_path, archive_root):
    import forward_prices
    real = forward_prices.archive_market_snapshot

    def archive(*a, **k):
        k.setdefault("root", archive_root)
        k.setdefault("stamp", STAMP)
        return real(*a, **k)
    monkeypatch.setattr(forward_prices, "archive_market_snapshot", archive)
    monkeypatch.setattr(sys, "argv", ["jvlink_odds.py", *argv])
    return mod.main()


@pytest.mark.parametrize("stage", ["t10", "close"])
def test_live_odds_json_and_exit_code_identical_to_base_commit(tmp_path, monkeypatch, stage):
    import jvlink_odds as new
    base = _load_base_jvlink_odds(tmp_path)
    by = _recs_by_spec()
    monkeypatch.setattr(base, "fetch_records", lambda rid, spec, max_rec=200: list(by.get(spec, [])))
    monkeypatch.setattr(new, "_fetch_records_raw",
                        lambda rid, spec, max_rec, meta: (meta.update(rc_init=0, rc_open=0) or list(by.get(spec, []))))
    monkeypatch.setattr(new, "_CAPTURE_LOG", [])
    monkeypatch.setenv("PYCALIAI_JV_JOURNAL_ROOT", str(tmp_path / "journal"))
    out_b, out_n = tmp_path / "live_base", tmp_path / "live_new"
    rc_b = _run_main(base, ["--race", RID, "--stage", stage, "--out-dir", str(out_b)], monkeypatch, tmp_path, tmp_path / "fb")
    rc_n = _run_main(new, ["--race", RID, "--stage", stage, "--out-dir", str(out_n)], monkeypatch, tmp_path, tmp_path / "fn")
    jb = json.loads((out_b / f"{RID}.json").read_text(encoding="utf-8"))
    jn = json.loads((out_n / f"{RID}.json").read_text(encoding="utf-8"))
    jb.pop("fetched"); jn.pop("fetched")
    assert rc_b == rc_n == 0 and jb == jn                      # latest view (compute_bets 入力) は不変
    rec = FP.read_snapshot(next((tmp_path / "fn").rglob("*.json.gz")))
    assert rec["stage"] == ("close_late" if stage == "close" else "t10")
    assert [c["spec"] for c in rec["jv_captures"]] == ["0B31", "0B32", "0B33", "0B34"]
    assert all(c["ok"] for c in rec["jv_captures"])
    old = FP.read_snapshot(next((tmp_path / "fb").rglob("*.json.gz")))
    assert {k: v for k, v in rec["market"].items() if k != "fetched"} == \
           {k: v for k, v in old["market"].items() if k != "fetched"}
    assert len(list((tmp_path / "journal").rglob("*.json"))) == 4


def test_capture_failures_never_change_exit_code_or_live_view(tmp_path, monkeypatch):
    import jv_records
    import jvlink_odds as new
    by = _recs_by_spec()
    monkeypatch.setattr(new, "_fetch_records_raw", lambda rid, spec, max_rec, meta: list(by.get(spec, [])))
    monkeypatch.setenv("PYCALIAI_JV_JOURNAL_ROOT", str(tmp_path / "journal"))

    def boom(*a, **k):
        raise RuntimeError("structuring failed")
    monkeypatch.setattr(jv_records, "capture", boom)
    monkeypatch.setattr(new, "_CAPTURE_LOG", [])
    rc = _run_main(new, ["--race", RID, "--stage", "t10", "--out-dir", str(tmp_path / "live")], monkeypatch, tmp_path, tmp_path / "f1")
    rec = FP.read_snapshot(next((tmp_path / "f1").rglob("*.json.gz")))
    assert rc == 0 and "structuring failed" in rec["capture_error"] and "jv_captures" not in rec
    monkeypatch.undo()


def test_raw_archive_failure_falls_back_to_market_only(tmp_path, monkeypatch):
    import forward_prices
    import jvlink_odds as new
    by = _recs_by_spec()
    monkeypatch.setattr(new, "_fetch_records_raw", lambda rid, spec, max_rec, meta: list(by.get(spec, [])))
    monkeypatch.setattr(new, "_CAPTURE_LOG", [])
    monkeypatch.setenv("PYCALIAI_JV_JOURNAL_ROOT", str(tmp_path / "journal"))
    real = forward_prices.archive_market_snapshot

    def archive(*a, **k):
        if k.get("captures"):
            raise OSError("disk full on raw")
        return real(*a, root=tmp_path / "f2", stamp=STAMP, **{x: y for x, y in k.items() if x not in ("root", "stamp")})
    monkeypatch.setattr(forward_prices, "archive_market_snapshot", archive)
    monkeypatch.setattr(sys, "argv", ["jvlink_odds.py", "--race", RID, "--stage", "t10", "--out-dir", str(tmp_path / "live")])
    assert new.main() == 0
    rec = FP.read_snapshot(next((tmp_path / "f2").rglob("*.json.gz")))
    assert "raw付き保存失敗" in rec["capture_error"]


def test_fetch_records_signature_and_empty_on_open_failure(monkeypatch, tmp_path):
    import jvlink_odds as new
    monkeypatch.setenv("PYCALIAI_JV_JOURNAL_ROOT", str(tmp_path / "journal"))
    monkeypatch.setattr(new, "_CAPTURE_LOG", [])

    def raw(rid, spec, max_rec, meta):
        meta.update(rc_init=0, rc_open=-1)
        return []
    monkeypatch.setattr(new, "_fetch_records_raw", raw)
    assert new.fetch_records(RID, "0B31") == []
    ev = json.loads(next((tmp_path / "journal").rglob("*.json")).read_text(encoding="utf-8"))
    assert ev["rc_open"] == -1 and ev["n_records_returned"] == 0 and ev["spec"] == "0B31"


# ---------------------------------------------------------------- jvlink_obs (shadow)
def test_t2_candidate_collects_all_five_specs_and_never_writes_live_odds(tmp_path, monkeypatch):
    import jvlink_obs
    monkeypatch.chdir(tmp_path)
    by = {SPEC_OF[k]: [build(k, n_reg=16, scratched=(2,))] for k in SPEC_OF}
    p = jvlink_obs.collect_rt(RID, "t2_candidate", "2026-10-03T15:30:00", fetch=fake_fetch(by),
                              root=tmp_path / "fwd", stamp=STAMP)
    rec = FP.read_snapshot(p)
    assert rec["stage"] == "t2_candidate" and [c["spec"] for c in rec["jv_captures"]] == list(jvlink_obs.SPECS_ALL)
    assert rec["market"]["ok"] and len(rec["market"]["trio"]) == JR.n_comb(15, 3)
    assert all(c["ok"] for c in rec["jv_captures"])
    written = {q.relative_to(tmp_path).parts[0] for q in tmp_path.rglob("*") if q.is_file()}
    assert written == {"fwd"}                                  # reports/live_odds 等には一切書かない


def test_t2_candidate_records_failed_fetch_without_raising(tmp_path):
    import jvlink_obs
    p = jvlink_obs.collect_rt(RID, "t2_candidate", None, fetch=fake_fetch({}, ok=False), root=tmp_path,
                              stamp=STAMP)
    rec = FP.read_snapshot(p)
    assert rec["market"]["ok"] is False and all(c["n_records_kind"] == 0 for c in rec["jv_captures"])


def test_stock_routing_keeps_only_o1_to_o5_of_target_races():
    import jvlink_obs
    keep = {RID}.__contains__
    assert jvlink_obs.stock_route(build("O5"), keep) == "0B35"
    assert jvlink_obs.stock_route(build("O1"), keep) == "0B31"
    assert jvlink_obs.stock_route("HR7" + "20261003" + RID + "x" * 50, keep) is None     # 払戻は読み捨て
    assert jvlink_obs.stock_route("SE7" + "20261003" + RID + "x" * 50, keep) is None     # 着順も読み捨て
    assert jvlink_obs.stock_route(build("O5", rid="2026100306040912"), keep) is None


def test_archive_stock_one_record_per_race_with_version_history(tmp_path):
    import jvlink_obs
    kept = {RID: {"0B35": [build("O5", kubun="4"), build("O5", kubun="5", announce="00000000")],
                  "0B31": [build("O1")]}}
    meta = {"fetch_started_at": "2026-10-03T18:30:00.000+09:00", "fetch_finished_at": "2026-10-03T18:31:00.000+09:00",
            "rc_init": 0, "rc_open": 0, "error": None, "from_ts": "20261003000000"}
    res = jvlink_obs.archive_stock(kept, meta, {RID: "2026-10-03T15:30:00", "2026100306040912": "x"},
                                   root=tmp_path, stamp=STAMP)
    assert res["saved"] == 1 and res["scheduled_races_without_stock_records"] == ["2026100306040912"]
    rec = FP.read_snapshot(next(tmp_path.rglob("*.json.gz")))
    trio = next(c for c in rec["jv_captures"] if c["spec"] == "0B35")
    assert rec["stage"] == "final_stock_candidate" and trio["stream"] == "stock"
    assert [r["kubun"] for r in trio["records"]] == ["4", "5"]


# ---------------------------------------------------------------- trio collector
def test_trio_collect_race_adds_timing_votes_manifest_and_forward_mirror(tmp_path, monkeypatch):
    import forward_prices
    import jvlink_trio_odds as T
    for name, sub in (("OUT_DIR", ""), ("RAW_DIR", "raw"), ("SNAP_DIR", "snapshots"), ("STOCK_DIR", "raw_stock")):
        monkeypatch.setattr(T, name, tmp_path / "trio" / sub if sub else tmp_path / "trio")
    monkeypatch.setattr(T, "MANIFEST", tmp_path / "trio" / "manifest.jsonl")
    monkeypatch.setattr(T, "BASE", tmp_path)
    monkeypatch.setattr(forward_prices, "FORWARD_ROOT", tmp_path / "fwd")
    monkeypatch.setattr(T, "fetch_records_timed", fake_fetch({"0B35": [build("O5")], "0B31": [build("O1")]}))
    real = forward_prices.archive_market_snapshot
    monkeypatch.setattr(forward_prices, "archive_market_snapshot",
                        lambda *a, **k: real(*a, **{**k, "stamp": STAMP}))
    snap = T.collect_race(RID, "20261003", dry=True)
    assert snap["ok"] and snap["votes_total"] == 900000 and snap["fetch"]["0B35"]["fetch_started_at"]
    assert "_dry" in snap["raw"]["0B35"].replace("\\", "/")
    assert not str(snap["forward_v2"]).startswith("ERROR")
    rec = FP.read_snapshot(next((tmp_path / "forward_prices_dry").rglob("*.json.gz")))
    assert rec["stage"] == "trio_t10" and {c["spec"] for c in rec["jv_captures"]} == {"0B35", "0B31"}
    rows = [json.loads(x) for x in (tmp_path / "trio" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    raws = [r for r in rows if r["kind"] == "raw"]
    assert len(raws) == 2 and all(r["sha256"] and r["announce"] == "10031530" for r in raws)


# ---------------------------------------------------------------- Stage 0 audit
def _store_day(root, jroot, specs_ok=True, announce="10031530", overlap_fail=False):
    """2 レース: 各レースの t2_candidate（5 spec）と本番 t10（4 spec）、journal は同時刻帯で重ねる。"""
    import os
    rids = [RID, "2026100309040911"]
    t = {RID: "15:28:00", rids[1]: "15:28:10"}
    for k, rid in enumerate(rids):
        for stage, specs, pid in (("t2_candidate", SPEC_OF.values(), 100 + k), ("t10", list(SPEC_OF.values())[:4], 200 + k)):
            caps = []
            for m, spec in enumerate(specs):
                kind = next(x for x, s in SPEC_OF.items() if s == spec)
                started = f"2026-10-03T{t[rid]}.{m:03d}+09:00"
                meta = {"fetch_started_at": started, "fetch_finished_at": started.replace(".", ".9", 1)[:-6][:23] + "+09:00",
                        "rc_init": 0, "rc_open": 0, "error": None, "stream": "rt"}
                bad = overlap_fail and stage == "t2_candidate" and m < 3
                recs = [build(kind, rid=rid, announce=announce)] if not bad else [build(kind, rid="2026100306040999")]
                caps.append(JR.capture(spec, rid, recs, meta))
                jroot.mkdir(parents=True, exist_ok=True)
                proc = "jvlink_obs" if stage == "t2_candidate" else "jvlink_odds"
                (jroot / f"{started[:10].replace('-', '')}_{pid}_{spec}_{stage}.json").write_text(json.dumps(
                    {"process": proc, "stage": stage, "race_id": rid, "spec": spec, "pid": pid,
                     "n_records_returned": 1, **meta}), encoding="utf-8")
            FP.archive_market_snapshot({"race_id": rid, "fetched": caps[0]["fetch_started_at"]}, stage, stamp=STAMP,
                                       root=root, captures=caps, scheduled_post="2026-10-03T15:30:00")
    return {"20261003": {r: "2026-10-03T15:30:00" for r in rids}}


def _audit(tmp_path, **kw):
    from analysis import obs_stage0_audit as A
    root, jroot = tmp_path / "fwd", tmp_path / "journal" / "20261003"
    sched = _store_day(root, jroot, **kw)
    return A.audit(["20261003"], root, tmp_path / "journal", schedule=sched)


def test_stage0_audit_clean_day(tmp_path):
    rep = _audit(tmp_path)
    assert rep["decision"] == "CONCURRENCY_OK" and rep["concurrency"]["overlapped"]["n"] > 0
    assert rep["concurrency"]["overlapped"]["rate"] == 1.0 and rep["concurrency"]["joined_to_capture"] == 18
    assert rep["missing"]["t2_candidate/0B35"]["missing_rate"] == 0.0
    assert all(v["fail"] == 0 for v in rep["raw_parser"].values())
    assert rep["coverage_fail_over_1pct"] == [] and rep["corruption_0000"] == []
    assert rep["files_opened_outcome_like"] == []


def test_stage0_audit_requires_queue_on_overlap_failures(tmp_path):
    rep = _audit(tmp_path, overlap_fail=True)
    assert rep["decision"] == "QUEUE_SERIALIZATION_REQUIRED"
    assert rep["concurrency"]["overlapped"]["rate"] < 0.95


def test_stage0_audit_requires_queue_on_any_0000_corruption(tmp_path):
    rep = _audit(tmp_path, announce="10030000")
    assert rep["decision"] == "QUEUE_SERIALIZATION_REQUIRED" and rep["corruption_0000"]


def test_stage0_audit_source_reads_no_outcome():
    """結果・払戻を読む関数/モジュールを import も呼出もしない（禁止パス marker は block 用の定数として持つだけ）。"""
    import ast
    tree = ast.parse((BASE / "analysis" / "obs_stage0_audit.py").read_text(encoding="utf-8"))
    mods = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} |             {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not mods & {"generate_results", "update_live_results", "settle_masters_vote", "backtest_pl_ev"}
    assert not names & {"load_kekka_all", "get_winner", "get_race_kk", "read_payouts", "payout_table"}


# ---------------------------------------------------------------- 500R guard
def test_performance_guard_blocks_before_500(tmp_path):
    from analysis import obs_guard as G
    with pytest.raises(PermissionError):
        G.assert_performance_allowed("trio", tmp_path, n=499)
    assert G.assert_performance_allowed("trio", tmp_path, n=500) == 500
    cap = JR.capture("0B35", RID, [build("O5")], {})
    for st in ("trio_t10", "final_stock_candidate"):
        FP.archive_market_snapshot({"race_id": RID, "fetched": f"2026-10-03T{15 if st == 'trio_t10' else 18}:00:00"},
                                   st, stamp=STAMP, root=tmp_path, captures=[cap])
    assert G.count_label_free_valid_races("trio", tmp_path) == 1
    with pytest.raises(PermissionError):
        G.assert_performance_allowed("trio", tmp_path)


# ---------------------------------------------------------------- dry-day start condition
def test_dry_check(tmp_path):
    from analysis import obs_dry_check as D
    trio = tmp_path / "trio"
    raw = trio / "raw" / "_dry" / "20261003" / RID
    raw.mkdir(parents=True)
    (raw / "t_0B35.txt").write_text("x", encoding="utf-8")
    man = [{"kind": "raw", "race_id": RID, "sha256": "s", "path": f"reports/x/raw/_dry/20261003/{RID}/t_0B35.txt"},
           {"kind": "snapshot", "race_id": RID, "announce_dt": "2026-10-03T15:20",
            "path": f"reports/x/snapshots/_dry/20261003/{RID}_t.json"}]
    (trio / "manifest.jsonl").write_text("\n".join(json.dumps(m) for m in man), encoding="utf-8")
    rep = D.check("20261003", [RID], trio, tmp_path / "fwd_dry")
    assert not rep["ok"] and not rep["failures"][0]["raw_0B31"]
    (raw / "t_0B31.txt").write_text("x", encoding="utf-8")
    cap = JR.capture("0B35", RID, [build("O5")], {})
    for st in ("trio_t10", "t2_candidate"):
        FP.archive_market_snapshot({"race_id": RID, "fetched": f"2026-10-03T15:{20 if st == 'trio_t10' else 28}:00"},
                                   st, stamp=STAMP, root=tmp_path / "fwd_dry", captures=[cap])
    assert D.check("20261003", [RID], trio, tmp_path / "fwd_dry")["ok"]


# ---------------------------------------------------------------- scheduler scripts
@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell parser")
def test_powershell_scripts_parse_and_register_needs_apply():
    for f in ("obs_schedule.ps1", "obs_register_tasks.ps1"):
        b = (BASE / f).read_bytes()
        assert b.startswith(b"\xef\xbb\xbf") and all(c < 128 for c in b[3:])
        cmd = ("$e=$null;$t=$null;[void][System.Management.Automation.Language.Parser]::ParseFile("
               f"'{BASE / f}',[ref]$t,[ref]$e);$e.Count")
        out = subprocess.run(["powershell.exe", "-NoProfile", "-Command", cmd], capture_output=True, text=True)
        assert out.stdout.strip() == "0", (f, out.stdout, out.stderr)
    src = (BASE / "obs_register_tasks.ps1").read_text(encoding="utf-8-sig")
    assert "if (-not $Apply)" in src and src.index("if (-not $Apply)") < src.index("Register-ScheduledTask -TaskName $name")
    sched = (BASE / "obs_schedule.ps1").read_text(encoding="utf-8-sig")
    for forbidden in ("masters_vote", "compute_bets", "validate_cowork", "live_odds", "notify", "git ", "sync-hf"):
        assert forbidden not in sched.split("##############################################################")[-1]
