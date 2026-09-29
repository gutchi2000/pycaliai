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
R2 = "2026100309040911"
ALL5 = list(SPEC_OF.values())


def _put(root, jroot, rid, stage, specs, hms, pid, proc, *, journal_stage=None, announce="10031530",
         bad=False, stream="rt", journal=True, n_run_override=None):
    """1 stage 分の capture 付き録を forward store に、spec ごとの取得を journal に書く。"""
    caps = []
    for m, spec in enumerate(specs):
        kind = next(x for x, s in SPEC_OF.items() if s == spec)
        started = f"2026-10-03T{hms}.{100 + m:03d}+09:00"
        finished = f"2026-10-03T{hms}.{600 + m:03d}+09:00"
        meta = {"fetch_started_at": started, "fetch_finished_at": finished, "rc_init": 0, "rc_open": 0,
                "error": None, "stream": stream}
        if bad:
            recs = [build(kind, rid="2026100306040999", announce=announce)]
        elif n_run_override is not None:
            recs = [build(kind, rid=rid, n_reg=12, scratched=(3,), n_run=n_run_override, announce=announce)]
        else:
            recs = [build(kind, rid=rid, announce=announce)]
        caps.append(JR.capture(spec, rid, recs, meta, stream=stream))
        if journal:
            jroot.mkdir(parents=True, exist_ok=True)
            (jroot / f"{started[11:19].replace(':', '')}{100 + m}_{pid}_{spec}_{stage}.json").write_text(json.dumps(
                {"process": proc, "stage": journal_stage or stage, "race_id": rid, "spec": spec, "pid": pid,
                 "n_records_returned": 1, **meta}), encoding="utf-8")
    FP.archive_market_snapshot({"race_id": rid, "fetched": caps[0]["fetch_started_at"]}, stage, stamp=STAMP,
                               root=root, captures=caps, scheduled_post="2026-10-03T15:30:00")


def _event(jroot, name, **ev):
    jroot.mkdir(parents=True, exist_ok=True)
    (jroot / f"{name}.json").write_text(json.dumps(ev), encoding="utf-8")


def _full_day(tmp_path, *, overlap=True, drop=(), bad_t2=False, announce="10031530", coverage_bad=False,
              drop_races=(), stock_error=None):
    """2 レース × 必須 6 stage。overlap=True なら本番 t10 と三連複 shadow が同時刻（別プロセス）。"""
    root, jroot = tmp_path / "fwd", tmp_path / "journal" / "20261003"
    for k, rid in enumerate((RID, R2)):
        base_m = 20 + 20 * k                     # 15:20 / 15:40
        t10 = f"15:{base_m:02d}:00"
        trio = t10 if overlap else f"15:{base_m + 3:02d}:00"
        if "t10" not in drop and ("t10", rid) not in drop_races:
            _put(root, jroot, rid, "t10", ALL5[:4], t10, 200 + k, "jvlink_odds", announce=announce)
        if "trio_t10" not in drop and ("trio_t10", rid) not in drop_races:
            _put(root, jroot, rid, "trio_t10", ["0B35", "0B31"], trio, 300 + k, "trio_shadow", announce=announce)
        if "t2_candidate" not in drop and ("t2_candidate", rid) not in drop_races:
            _put(root, jroot, rid, "t2_candidate", ALL5, f"15:{base_m + 8:02d}:00", 400 + k, "jvlink_obs",
                 announce=announce, bad=bad_t2, n_run_override=(12 if coverage_bad and k == 0 else None))
        if "close_late" not in drop and ("close_late", rid) not in drop_races:
            _put(root, jroot, rid, "close_late", ALL5[:4], f"15:{base_m + 11:02d}:00", 500 + k, "jvlink_odds",
                 journal_stage="close", announce=announce)
        if "final_rt_candidate" not in drop and ("final_rt_candidate", rid) not in drop_races:
            _put(root, jroot, rid, "final_rt_candidate", ALL5, f"18:{30 + k:02d}:00", 600, "jvlink_obs")
        if "final_stock_candidate" not in drop and ("final_stock_candidate", rid) not in drop_races:
            _put(root, jroot, rid, "final_stock_candidate", ALL5, "18:40:00", 700, "jvlink_obs", stream="stock",
                 journal=False, announce="00000000")
    if "final_stock_candidate" not in drop:
        _event(jroot, "184000_700_STOCK", process="jvlink_obs", stage="final_stock_candidate", race_id="STOCK",
               spec="RACE", pid=700, rc_init=0, rc_open=0, n_records_returned=10, error=stock_error,
               fetch_started_at="2026-10-03T18:40:00.000+09:00", fetch_finished_at="2026-10-03T18:44:00.000+09:00")
    return root, tmp_path / "journal", {"20261003": {RID: "2026-10-03T15:30:00", R2: "2026-10-03T15:50:00"}}


def _run_audit(root, jroot, sched):
    from analysis import obs_stage0_audit as A
    return A.audit(["20261003"], root, jroot, schedule=sched)


def test_stage0_audit_clean_day_contract_met_and_concurrency_ok(tmp_path):
    rep = _run_audit(*_full_day(tmp_path))
    assert rep["contract"]["met"], rep["contract"]
    assert rep["decision"] == "CONCURRENCY_OK" and rep["exit_code"] == 0
    c = rep["concurrency"]
    assert c["overlapped"]["n"] > 0 and c["overlapped"]["rate"] == 1.0 and c["unidentified_events"] == 0
    assert c["assumes_schedule_non_overlap"] is False
    assert "trio_t10" in c["by_kind"]["t10"]["partner_kinds"]          # 同一レースでも別プロセスなら重なり
    assert all(v["missing_rate"] == 0.0 for v in rep["missing"].values() if v.get("required"))
    assert all(v["fail"] == 0 for v in rep["raw_parser"].values())
    assert rep["files_opened_outcome_like"] == []


def test_stage0_audit_missing_required_stage_is_contract_not_met(tmp_path):
    rep = _run_audit(*_full_day(tmp_path, drop=("final_stock_candidate",)))
    assert rep["decision"] == "CONTRACT_NOT_MET" and rep["exit_code"] == 4
    assert any("final_stock_candidate/0B35" in r for r in rep["contract"]["reasons"])


def test_stage0_audit_coverage_mismatch_over_1pct_is_contract_not_met(tmp_path):
    rep = _run_audit(*_full_day(tmp_path, coverage_bad=True))
    assert rep["decision"] == "CONTRACT_NOT_MET" and rep["exit_code"] == 4
    assert rep["coverage_fail_over_1pct"] and any("coverage" in r for r in rep["contract"]["reasons"])


def test_stage0_audit_unidentified_process_is_contract_not_met(tmp_path):
    root, jroot, sched = _full_day(tmp_path)
    _event(jroot / "20261003", "x_unknown", process="unknown", stage=None, race_id=RID, spec="0B31", pid=999,
           rc_init=0, rc_open=0, n_records_returned=1,
           fetch_started_at="2026-10-03T12:00:00.000+09:00", fetch_finished_at="2026-10-03T12:00:01.000+09:00")
    rep = _run_audit(root, jroot, sched)
    assert rep["decision"] == "CONTRACT_NOT_MET" and rep["concurrency"]["unidentified_events"] == 1


def test_stage0_audit_overlap_failures_report_queue_verdict_even_when_contract_fails(tmp_path):
    root, jroot, sched = _full_day(tmp_path, bad_t2=True)
    for k, rid in enumerate((RID, R2)):                    # jvlink_changes の取得を T−2 と同時刻に重ねる
        hms = f"15{28 + 20 * k:02d}00"
        _event(jroot / "20261003", f"{hms}_changes_{k}", process="jvlink_changes", stage="changes",
               race_id=rid, spec="0B15", pid=800 + k, rc_init=0, rc_open=-1, n_records_returned=0,
               fetch_started_at=f"2026-10-03T15:{28 + 20 * k:02d}:00.200+09:00",
               fetch_finished_at=f"2026-10-03T15:{28 + 20 * k:02d}:00.400+09:00")
    rep = _run_audit(root, jroot, sched)
    assert rep["decision"] == "CONTRACT_NOT_MET"                   # 失敗は契約 (race_key) も壊す
    assert rep["concurrency_verdict"] == "QUEUE_SERIALIZATION_REQUIRED"
    t2 = rep["concurrency"]["by_kind"]["t2_candidate"]
    assert t2["over_n"] == 10 and t2["over_rate"] == 0.0
    assert "jvlink_changes" in t2["partner_kinds"]
    assert "jvlink_changes" in rep["concurrency"]["by_kind"]         # changes は重なり相手として識別される


def test_stage0_audit_0000_corruption_requires_queue(tmp_path):
    rep = _run_audit(*_full_day(tmp_path, announce="10030000"))
    assert rep["contract"]["met"] and rep["corruption_0000"]
    assert rep["decision"] == "QUEUE_SERIALIZATION_REQUIRED" and rep["exit_code"] == 3


def test_stage0_audit_insufficient_overlap_is_its_own_exit(tmp_path):
    rep = _run_audit(*_full_day(tmp_path, overlap=False))
    assert rep["contract"]["met"] and rep["concurrency"]["overlapped"]["n"] == 0
    assert rep["decision"] == "INSUFFICIENT_OVERLAP_OBSERVED" and rep["exit_code"] == 5


def test_decision_exit_codes_are_distinct():
    from analysis import obs_stage0_audit as A
    assert A.DECISION_EXIT == {"CONCURRENCY_OK": 0, "QUEUE_SERIALIZATION_REQUIRED": 3, "CONTRACT_NOT_MET": 4,
                               "INSUFFICIENT_OVERLAP_OBSERVED": 5}


@pytest.mark.parametrize("ev,kind", [
    ({"process": "jvlink_odds", "stage": "t10"}, "t10"),
    ({"process": "jvlink_odds", "stage": "close"}, "close_late"),
    ({"process": "jvlink_odds", "stage": "close_late"}, "close_late"),
    ({"process": "jvlink_odds", "stage": "t20"}, "t20"),
    ({"process": "jvlink_odds", "stage": "vote"}, "vote"),
    ({"process": "jvlink_odds", "stage": "exp05fs_t35"}, "exp05fs_t35"),
    ({"process": "exp05fs_calendar", "stage": "calendar"}, "exp05fs_calendar"),
    ({"process": "jvlink_changes", "stage": "changes"}, "jvlink_changes"),
    ({"process": "jvlink_obs", "stage": "t2_candidate"}, "t2_candidate"),
    ({"process": "jvlink_obs", "stage": "final_rt_candidate"}, "final_rt_candidate"),
    ({"process": "jvlink_obs", "stage": "final_stock_candidate", "spec": "RACE"}, "final_stock_candidate(STOCK)"),
    ({"process": "trio_shadow", "stage": "trio_t10"}, "trio_t10"),
    ({"process": "analysis/bodyweight_forward/collector", "stage": None}, "analysis/bodyweight_forward/collector"),
    ({"process": "unknown", "stage": None}, "UNIDENTIFIED:unknown"),
])
def test_every_jvlink_fetch_kind_is_identifiable(ev, kind):
    from analysis import obs_stage0_audit as A
    assert A.event_kind(ev) == kind


def test_journal_never_records_unknown_process(tmp_path, monkeypatch):
    monkeypatch.setenv("PYCALIAI_JV_JOURNAL_ROOT", str(tmp_path))
    monkeypatch.setattr(jv_journal, "_CONTEXT", {"process": "unknown", "stage": None, "race_id": None, "dry": False})
    monkeypatch.setattr(sys, "argv", [str(BASE / "analysis" / "bodyweight_forward" / "collector.py")])
    p = jv_journal.write_event(RID, "0B11", {"fetch_started_at": "2026-10-03T09:00:00.000+09:00"})
    ev = json.loads(p.read_text(encoding="utf-8"))
    assert ev["process"] == "analysis/bodyweight_forward/collector" and ev["process_source"] == "argv"
    monkeypatch.setattr(sys, "argv", ["-c"])
    ev = json.loads(jv_journal.write_event(RID, "0B11", {"fetch_started_at": "2026-10-03T09:00:01.000+09:00"})
                    .read_text(encoding="utf-8"))
    assert ev["process"] == "python"                                  # 監査では UNIDENTIFIED → 契約未達


def test_jvlink_changes_sets_explicit_process_name():
    import ast
    tree = ast.parse((BASE / "jvlink_changes.py").read_text(encoding="utf-8"))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    calls = [n for n in ast.walk(main) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "set_context"]
    assert any(k.arg == "process" and getattr(k.value, "value", None) == "jvlink_changes"
               for c in calls for k in c.keywords)


def test_exp05fs_calendar_session_is_journaled_even_on_failure(tmp_path, monkeypatch):
    from analysis.mcond.exp05_forward_shadow import jvlink_race_calendar as cal
    monkeypatch.setenv("PYCALIAI_JV_JOURNAL_ROOT", str(tmp_path))
    monkeypatch.setattr(cal, "_fetch_races_jv", lambda *a, **k: ([], "JVOpen失敗 rc=-1"))
    assert cal.fetch_races("20261003") == ([], "JVOpen失敗 rc=-1")         # 戻り値は不変
    ev = json.loads(next(tmp_path.rglob("*.json")).read_text(encoding="utf-8"))
    assert ev["process"] == "exp05fs_calendar" and ev["spec"] == "RACE" and ev["error"] == "JVOpen失敗 rc=-1"
    from analysis import obs_stage0_audit as A
    assert A.event_kind(ev) == "exp05fs_calendar"


def test_stage0_audit_source_reads_no_outcome():
    """結果・払戻を読む関数/モジュールを import も呼出もしない（禁止パス marker は block 用の定数として持つだけ）。"""
    import ast
    tree = ast.parse((BASE / "analysis" / "obs_stage0_audit.py").read_text(encoding="utf-8"))
    mods = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
            {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not mods & {"generate_results", "update_live_results", "settle_masters_vote", "backtest_pl_ev"}
    assert not names & {"load_kekka_all", "get_winner", "get_race_kk", "read_payouts", "payout_table"}


# ---------------------------------------------------------------- 500R guard
def test_performance_guard_blocks_before_500(tmp_path):
    from analysis import obs_guard as G
    with pytest.raises(PermissionError):
        G._require(499, "trio")
    assert G._require(500, "trio") == 500
    cap = JR.capture("0B35", RID, [build("O5")], {})
    for st in ("trio_t10", "final_stock_candidate"):
        FP.archive_market_snapshot({"race_id": RID, "fetched": f"2026-10-03T{15 if st == 'trio_t10' else 18}:00:00"},
                                   st, stamp=STAMP, root=tmp_path, captures=[cap])
    assert G.count_label_free_valid_races("trio", tmp_path) == 1
    with pytest.raises(PermissionError):
        G.assert_performance_allowed("trio", tmp_path, tmp_path / "no_ledger.jsonl")


def test_guard_has_no_count_override_argument():
    import inspect
    from analysis import obs_guard as G
    assert list(inspect.signature(G.assert_performance_allowed).parameters) == ["stream", "root", "ledger"]


def test_repository_has_no_unguarded_performance_path():
    """追跡 .py 全件で、観測 stage × 結果・払戻系を扱いながら guard を呼ばない module が 0 件。"""
    from analysis import obs_guard as G
    files = G.tracked_python_files()
    assert len(files) > 100
    assert G.find_unguarded_modules(files=files) == []


GUARD_CASES = {
    "unguarded_import": ('from generate_results import load_kekka_all\nSTAGE = "trio_t10"\n', True),
    "unguarded_name_import": ('from build_site import parse_wide_kekka\nS = ["t2_candidate"]\n', True),
    "unguarded_string": ('p = "data/kekka/20261003.csv"\nstage = "final_stock_candidate"\n', True),
    "unguarded_japanese": ('col = "確定着順"\nstage = "final_rt_candidate"\n', True),
    "unguarded_dynamic": ('import importlib\nm = importlib.import_module("generate_results")\ns = "trio_t10"\n', True),
    "guarded": ('from analysis.obs_guard import assert_performance_allowed\n'
                'from generate_results import load_kekka_all\n'
                'def main():\n    assert_performance_allowed("trio")\n    return "trio_t10"\n', False),
    "guard_root_override": ('from analysis.obs_guard import assert_performance_allowed\n'
                            'import generate_results\n'
                            'assert_performance_allowed("trio", root="elsewhere")\ns = "trio_t10"\n', True),
    "guard_ledger_override": ('from analysis.obs_guard import assert_performance_allowed\n'
                              'import generate_results\n'
                              'assert_performance_allowed("trio", ledger="fake.jsonl")\ns = "trio_t10"\n', True),
    "guard_positional_override": ('from analysis.obs_guard import assert_performance_allowed\n'
                                  'import generate_results\n'
                                  'assert_performance_allowed("trio", "elsewhere")\ns = "trio_t10"\n', True),
    "stage_only_label_free": ('STAGES = ("t2_candidate", "trio_t10")\n', False),
    "outcome_only_no_obs_stage": ('from generate_results import load_kekka_all\nstage = "t10"\n', False),
    "docstring_and_blocklist_ignored": ('"""payout や 着順 は読まない。"""\nFORBIDDEN_MARKERS = ("kekka", "payout")\n'
                                        'STAGE = "t2_candidate"\n', False),
    "prose_mention_not_counted": ('ROLE = "label-free: no outcome, payout or ROI is read"\nS = "trio_t10"\n', False),
    "path_with_space_still_counted": ('P = "E:/data dir/kekka/x.csv"\nS = "trio_t10"\n', True),
}


@pytest.mark.parametrize("name", sorted(GUARD_CASES))
def test_guard_static_check_cases(tmp_path, name):
    from analysis import obs_guard as G
    src, bad = GUARD_CASES[name]
    p = tmp_path / "analysis" / f"{name}.py"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(src, encoding="utf-8")
    found = G.find_unguarded_modules(base=tmp_path, files=[p])
    assert bool(found) is bad, (name, found)


def test_guard_check_cli_fails_on_violation(tmp_path, monkeypatch):
    from analysis import obs_guard as G
    p = tmp_path / "analysis" / "bad.py"
    p.parent.mkdir(parents=True)
    p.write_text('import generate_results\nx = "trio_t10"\n', encoding="utf-8")
    monkeypatch.setattr(G, "BASE", tmp_path)
    monkeypatch.setattr(G, "tracked_python_files", lambda base=tmp_path: [p])
    monkeypatch.setattr(sys, "argv", ["obs_guard", "--check"])
    assert G.main() == 1


# ---------------------------------------------------------------- Stage 0 missing threshold, attribution, ledger
def test_missing_threshold_is_max_of_1pct_and_one_race():
    from analysis import obs_stage0_audit as A
    assert A.missing_threshold(2) == 1.0 and A.missing_threshold(50) == 1.0
    assert A.missing_threshold(150) == pytest.approx(1.5) and A.missing_threshold(300) == pytest.approx(3.0)


def test_one_missing_race_over_two_days_is_tolerated_and_attributed(tmp_path):
    rep = _run_audit(*_full_day(tmp_path, drop_races={("final_stock_candidate", R2)}))
    m = rep["missing"]["final_stock_candidate/0B35"]
    assert m["missing_count"] == 1 and m["threshold"] == 1.0 and m["over_threshold"] is False
    assert m["races"] == [{"race_id": R2, "cause": "stock_race_absent",
                           "detail": "STOCK session ok but no O1-O5 record for the race"}]
    assert rep["contract"]["met"] and rep["decision"] == "CONCURRENCY_OK"
    assert rep["missing_cause_summary"] == {"stock_race_absent": 5}


def test_two_missing_races_exceed_threshold_with_causes_and_restart(tmp_path):
    rep = _run_audit(*_full_day(tmp_path, drop_races={("t2_candidate", RID), ("t2_candidate", R2)}))
    m = rep["missing"]["t2_candidate/0B33"]
    assert m["missing_count"] == 2 and m["over_threshold"] is True
    assert {r["cause"] for r in m["races"]} == {"task_not_fired"}
    assert rep["decision"] == "CONTRACT_NOT_MET" and rep["exit_code"] == 4
    assert any("t2_candidate/0B33 (2 > 1)" in r for r in rep["contract"]["reasons"])
    assert rep["collection_continues"] is True and rep["performance_blocked_by_stage0"] is True
    assert rep["stage0_restart"] and rep["stage0_passed"] is False


def test_race_key_mismatch_is_attributed(tmp_path):
    rep = _run_audit(*_full_day(tmp_path, bad_t2=True))
    races = rep["missing"]["t2_candidate/0B35"]["races"]
    assert {r["cause"] for r in races} == {"race_key_mismatch"} and len(races) == 2
    assert rep["decision"] == "CONTRACT_NOT_MET"


def test_stock_session_failure_is_attributed(tmp_path):
    rep = _run_audit(*_full_day(tmp_path, stock_error="JVOpen rc=-1",
                                drop_races={("final_stock_candidate", RID), ("final_stock_candidate", R2)}))
    assert {r["cause"] for r in rep["missing"]["final_stock_candidate/0B31"]["races"]} == {"stock_session_failed"}


def _cap(records, **meta):
    return {"records": records, "fetch_started_at": meta.pop("started", "2026-10-03T15:28:00.100+09:00"), **meta}


@pytest.mark.parametrize("case,expected", [
    ("capture_rc", "fetch_rc"), ("capture_empty", "no_records"), ("capture_wrong_key", "race_key_mismatch"),
    ("capture_ok_but_missing", "unattributed"), ("record_v1", "record_without_raw_v1"),
    ("record_v2_no_spec", "spec_capture_absent"), ("journal_rc", "fetch_rc"), ("journal_empty", "no_records"),
    ("journal_records_not_stored", "record_not_stored"), ("nothing", "task_not_fired"),
    ("stock_no_session", "task_not_fired"), ("stock_failed", "stock_session_failed"),
    ("stock_absent", "stock_race_absent"),
])
def test_attribute_missing_every_cause(case, expected):
    from analysis import obs_stage0_audit as A
    st, spec, rid = ("final_stock_candidate" if case.startswith("stock") else "t2_candidate"), "0B35", RID
    cap_by, rec_by, jidx, sess = {}, {}, {}, []
    good = {"race_key_ok": True}
    if case == "capture_rc":
        cap_by[(st, spec, rid)] = [_cap([], rc_init=0, rc_open=-1)]
    elif case == "capture_empty":
        cap_by[(st, spec, rid)] = [_cap([], rc_init=0, rc_open=0, n_records_returned=2)]
    elif case == "capture_wrong_key":
        cap_by[(st, spec, rid)] = [_cap([{"race_key_ok": False, "race_key": "x"}], rc_init=0, rc_open=0)]
    elif case == "capture_ok_but_missing":
        cap_by[(st, spec, rid)] = [_cap([good], rc_init=0, rc_open=0)]
    elif case == "record_v1":
        rec_by[(st, rid)] = "v1"
    elif case == "record_v2_no_spec":
        rec_by[(st, rid)] = "v2"
    elif case.startswith("journal"):
        ev = {"rc_init": 0, "rc_open": -1 if case == "journal_rc" else 0,
              "n_records_returned": 0 if case == "journal_empty" else 3, "fetch_started_at": "2026-10-03T15:28"}
        jidx[(st, rid, spec)] = [ev]
    elif case == "stock_failed":
        sess = [{"fetch_started_at": "2026-10-03T18:30:00", "rc_init": 0, "rc_open": -1, "error": "x"}]
    elif case == "stock_absent":
        sess = [{"fetch_started_at": "2026-10-03T18:30:00", "rc_init": 0, "rc_open": 0}]
    elif case == "stock_no_session":
        sess = [{"fetch_started_at": "2026-10-02T18:30:00", "rc_init": 0, "rc_open": 0}]   # 前日のセッションは対象外
    cause, detail = A.attribute_missing(st, spec, rid, cap_by, rec_by, jidx, sess)
    assert cause == expected and cause in A.MISSING_CAUSES and detail


def test_unattributed_missing_is_contract_not_met(tmp_path, monkeypatch):
    from analysis import obs_stage0_audit as A
    monkeypatch.setattr(A, "attribute_missing", lambda *a, **k: ("unattributed", "forced"))
    rep = _run_audit(*_full_day(tmp_path, drop_races={("close_late", R2)}))
    assert rep["missing"]["close_late/0B31"]["missing_count"] == 1                   # 閾値内でも
    assert rep["decision"] == "CONTRACT_NOT_MET"
    assert any("without an attributed cause" in r for r in rep["contract"]["reasons"])


def test_ledger_and_guard_block_after_contract_not_met_and_count_from_pass_window(tmp_path, monkeypatch):
    from analysis import obs_guard as G
    from analysis import obs_stage0_audit as A
    ledger = tmp_path / "ledger.jsonl"
    with pytest.raises(PermissionError, match="no real Stage 0"):
        G.assert_performance_allowed("trio", tmp_path / "fwd", ledger)
    rep = _run_audit(*_full_day(tmp_path / "d1", drop_races={("t2_candidate", RID), ("t2_candidate", R2)}))
    A.record_ledger(rep, ledger, dry=False, report_path="r1.json")
    row = json.loads(ledger.read_text(encoding="utf-8").splitlines()[-1])
    assert row["decision"] == "CONTRACT_NOT_MET" and row["missing_cause_summary"] == {"task_not_fired": 10}
    with pytest.raises(PermissionError, match="restart Stage 0"):
        G.assert_performance_allowed("trio", tmp_path / "fwd", ledger)
    ok = _run_audit(*_full_day(tmp_path / "d2"))
    A.record_ledger(ok, ledger, dry=False, report_path="r2.json")
    A.record_ledger(rep, ledger, dry=True, report_path="dry.json")                   # dry は guard が無視
    assert G.latest_stage0(ledger)["decision"] == "CONCURRENCY_OK"
    fwd = tmp_path / "fwd"
    cap = JR.capture("0B35", RID, [build("O5")], {})
    for day, rid in (("20261003", RID), ("20260927", "2026092706040911")):
        for st, hh in (("trio_t10", 15), ("final_stock_candidate", 18)):
            FP.archive_market_snapshot({"race_id": rid, "fetched": f"{day[:4]}-{day[4:6]}-{day[6:]}T{hh}:00:00"}, st,
                                       stamp=STAMP, root=fwd, captures=[JR.capture("0B35", rid, [build("O5", rid=rid)], {})])
    assert G.count_label_free_valid_races("trio", fwd) == 2
    assert G.count_label_free_valid_races("trio", fwd, since="20261003") == 1          # 通過窓より前は数えない
    monkeypatch.setattr(G, "MIN_RACES_FOR_PERFORMANCE", 1)
    assert G.assert_performance_allowed("trio", fwd, ledger) == 1
    A.record_ledger(rep, ledger, dry=False, report_path="r3.json")                   # 後で再び契約未達
    with pytest.raises(PermissionError):
        G.assert_performance_allowed("trio", fwd, ledger)


def test_audit_and_guard_share_ledger_path_and_pass_code():
    from analysis import obs_guard as G
    from analysis import obs_stage0_audit as A
    assert G.STAGE0_LEDGER == A.STAGE0_LEDGER and G.STAGE0_PASS == A.STAGE0_PASS == "CONCURRENCY_OK"


def test_observation_plan_v211_hash_and_probe_rule():
    import hashlib
    doc = BASE / "docs" / "research" / "OBSERVATION_PLAN_20260928.md"
    rec = (BASE / "docs" / "research" / "OBSERVATION_PLAN_20260928.sha256").read_text(encoding="utf-8").split()
    assert rec[0] == hashlib.sha256(doc.read_bytes()).hexdigest() and rec[2] == "v2.1.1"
    text = doc.read_text(encoding="utf-8")
    for probe in ("jvlink_probe.py", "jvlink_race_day_probe.py", "jvlink_shadow_probe.py"):
        assert probe in text.split("## 12.")[1]
    assert "a23ef19352b73ecdeab2ba1c68de2da12e54b5bb5e8a4ad7925900f621185d8c" in text        # v2.1 の hash を保持


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
