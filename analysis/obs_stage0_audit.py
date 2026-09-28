# -*- coding: utf-8 -*-
"""
obs_stage0_audit.py — 観測計画 v2.1 Phase 2 Stage 0（最初の 2 開催日）label-free 監査
====================================================================================
監査するのは次だけ（実行部への指示どおり）。outcome・払戻・ROI・性能・帯選択は読まない / 計算しない。

  1. 欠損率        stage × spec ごとに、予定レースのうち録（capture）が無い割合
  2. 時刻          取得所要（spec 別・5 spec 合計）、発表遅延（取得完了 − 発表月日時分）、発走までの秒
  3. 全組被覆      発売中の組数 = 出走頭数から決まる期待値、非空白 slot = 登録頭数からの期待値
  4. raw/parser 一致 保存 raw の再構造化 = 保存済み構造化、slot parser = 本番の実録突合済み parser
  5. 並走成功率    T−2 / 本番 T−10 / 三連複 shadow の取得が別プロセス同士で ±30 秒以内に重なった
                   取得の成功率を単独取得と比べる（成功 = rc 0・録あり・race_key 一致・全組被覆）。
                   同一レースの本番 T−10 と三連複 shadow の同時取得も別プロセスなので重なりに数える
  +  00:00 型破損  RT 録の発表時分 00:00、または発走予定 00:00

判定（§3 (6)）: 重なり時の成功率 < 95%、または 00:00 型破損が 1 件でもあれば
QUEUE_SERIALIZATION_REQUIRED（性能評価へ進まず、単一プロセスのキュー直列化を先に行う）。

python -m analysis.obs_stage0_audit --dates 20261003 20261004 [--dry]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import jv_records as JR  # noqa: E402
from forward_prices import FORWARD_ROOT, canonical_stage, read_snapshot  # noqa: E402

JOURNAL_ROOT = BASE / "data" / "jvlink_fetch_journal"
OVERLAP_SEC = 30.0
MIN_OVERLAP_SUCCESS = 0.95
MAX_COVERAGE_MISMATCH = 0.01
FIVE_SPEC_P95_SEC = 60.0
STAGES = ("t10", "t20", "close_late", "t2_candidate", "trio_t10", "final_rt_candidate",
          "final_stock_candidate")
CONCURRENCY_KINDS = {("jvlink_odds", "t10"): "t10", ("jvlink_obs", "t2_candidate"): "t2_candidate",
                     ("trio_shadow", "trio_t10"): "trio_t10"}
FORBIDDEN_MARKERS = ("kekka", "payout", "haraimodoshi", "wide_payouts", "live_results", "results.json",
                     "rows_2013_2023")

_OPENED: set[str] = set()


def _track_opens() -> None:
    def hook(event, args):
        if event == "open" and args and isinstance(args[0], (str, os.PathLike)):
            _OPENED.add(os.fsdecode(args[0]).replace("\\", "/"))
    sys.addaudithook(hook)


def _ts(s) -> datetime | None:
    try:
        d = datetime.fromisoformat(str(s))
        return d if d.tzinfo else d.astimezone()
    except Exception:
        return None


def _q(xs: list[float]) -> dict:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return {"n": 0}
    p95 = xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))]
    return {"n": len(xs), "p50": round(median(xs), 3), "p95": round(p95, 3), "max": round(xs[-1], 3)}


def scheduled_races(date: str) -> dict[str, str]:
    from jvlink_trio_odds import build_schedule
    return {rid: pt.isoformat() for pt, rid in build_schedule(date)}


def load_records(root: Path, dates: list[str]) -> list[tuple[Path, dict]]:
    out = []
    for d in dates:
        for p in sorted((root / d).glob("*.json.gz")):
            rec = read_snapshot(p)
            if rec.get("record_type") == "market_snapshot":
                rec["_stage"] = canonical_stage(rec.get("stage"))
                out.append((p, rec))
    return out


def load_journal(root: Path, dates: list[str]) -> list[dict]:
    ev = []
    for d in dates:
        for p in sorted((root / d).glob("*.json")):
            try:
                ev.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                ev.append({"_unreadable": str(p)})
    return ev


def audit(dates: list[str], forward_root: Path = FORWARD_ROOT, journal_root: Path = JOURNAL_ROOT,
          schedule: dict[str, dict[str, str]] | None = None) -> dict:
    schedule = schedule if schedule is not None else {d: scheduled_races(d) for d in dates}
    sched_all = {rid: post for d in dates for rid, post in schedule.get(d, {}).items()}
    recs = load_records(forward_root, dates)
    events = load_journal(journal_root, dates)

    # ---- 1. 欠損率 / 3. 全組被覆 / 4. raw・parser 一致 / 2. 時刻（capture 単位）
    have = defaultdict(set)                      # (stage, spec) -> rids with a capture holding >=1 record
    v1_only = defaultdict(set)                   # stage -> rids with records lacking jv_captures
    cov = defaultdict(lambda: defaultdict(lambda: [0, 0]))   # (stage,spec) -> block -> [mismatch, n]
    parse_checks = defaultdict(lambda: [0, 0])  # check -> [fail, n]
    dur = defaultdict(list)
    span = defaultdict(list)
    ann_delay = defaultdict(list)
    to_post = defaultdict(lambda: defaultdict(list))
    corruption: list[dict] = []
    capture_index: dict[tuple, dict] = {}
    for path, rec in recs:
        st, rid = rec["_stage"], rec.get("race_id")
        caps = rec.get("jv_captures")
        if not caps:
            v1_only[st].add(rid)
            continue
        sp = _ts(rec.get("scheduled_post"))
        if rec.get("scheduled_post") and str(rec["scheduled_post"])[11:16] == "00:00":
            corruption.append({"race_id": rid, "stage": st, "what": "scheduled_post 00:00"})
        starts, ends = [], []
        for cap in caps:
            spec = cap.get("spec")
            key = (st, spec)
            if cap.get("records"):
                have[key].add(rid)
            t0, t1 = _ts(cap.get("fetch_started_at")), _ts(cap.get("fetch_finished_at"))
            if t0 and t1:
                dur[key].append((t1 - t0).total_seconds())
                starts.append(t0)
                ends.append(t1)
            capture_index[(st, rid, spec, str(cap.get("fetch_started_at")))] = cap
            for r in cap.get("records", []):
                for block, c in (r.get("counts") or {}).items():
                    m = c.get("priced_match") if block != "wakuren" else c.get("nonblank_match")
                    if m is not None:
                        cov[key][block][0] += int(not m)
                        cov[key][block][1] += 1
                again = JR.structure_record(r["raw"], spec, rid, r.get("stream", "rt"))
                same = all(again.get(k) == r.get(k) for k in ("odds_parsed", "counts", "kubun",
                                                              "announced_at", "votes_total", "hatsubai_flag"))
                for name, ok in (("restructure_equal", same),
                                 ("validated_parser_match", r.get("validated_parser_match") is True),
                                 ("race_key_ok", bool(r.get("race_key_ok"))),
                                 ("length_ok", bool(r.get("length_ok"))),
                                 ("raw_sha256", hashlib.sha256(r["raw"].encode("utf-8", "replace")).hexdigest()
                                  == r.get("raw_sha256"))):
                    parse_checks[name][0] += int(not ok)
                    parse_checks[name][1] += 1
                if "announce_0000" in (r.get("anomalies") or []):
                    corruption.append({"race_id": rid, "stage": st, "spec": spec, "what": "RT announce 00:00",
                                       "announce_raw": r.get("announce_raw")})
            last = cap["records"][-1] if cap.get("records") else None
            if last and cap.get("stream") == "rt":
                ta = _ts(last.get("announced_at"))
                if ta and t1:
                    ann_delay[key].append((t1 - ta).total_seconds())
                if sp and ta:
                    to_post[key]["announce_basis"].append((sp - ta).total_seconds())
                if sp and t1:
                    to_post[key]["fetch_basis"].append((sp - t1).total_seconds())
        if starts:
            span[st].append((max(ends) - min(starts)).total_seconds())

    missing = {}
    specs_by_stage = defaultdict(set)
    for st, spec in have:
        specs_by_stage[st].add(spec)
    for st in STAGES:
        for spec in sorted(specs_by_stage.get(st, set())):
            n = len(sched_all)
            got = len(have[(st, spec)] & set(sched_all))
            missing[f"{st}/{spec}"] = {"scheduled": n, "with_capture": got,
                                       "missing_rate": round(1 - got / n, 4) if n else None}
        if v1_only.get(st):
            missing[f"{st}/v1_without_raw"] = {"races": len(v1_only[st])}
    coverage = {f"{st}/{spec}": {b: {"mismatch": m, "n": n, "rate": round(m / n, 4) if n else None}
                                 for b, (m, n) in blocks.items()}
                for (st, spec), blocks in cov.items()}
    coverage_fail = sorted(f"{k}/{b}" for k, blocks in coverage.items() for b, v in blocks.items()
                           if v["rate"] is not None and v["rate"] > MAX_COVERAGE_MISMATCH and b != "wakuren")
    wakuren_fail = sorted(k for k, blocks in coverage.items()
                          if (blocks.get("wakuren") or {}).get("rate") not in (None, 0.0)
                          and blocks["wakuren"]["rate"] > MAX_COVERAGE_MISMATCH)

    # ---- 5. 並走成功率（journal）
    def kind(e):
        return CONCURRENCY_KINDS.get((e.get("process"), e.get("stage")))

    def success(e) -> tuple[bool, bool]:
        base_ok = e.get("rc_init") == 0 and e.get("rc_open") == 0 and (e.get("n_records_returned") or 0) > 0
        cap = capture_index.get((canonical_stage(e.get("stage")), e.get("race_id"), e.get("spec"),
                                 str(e.get("fetch_started_at"))))
        if cap is None:
            return base_ok, False
        return base_ok and bool(cap.get("ok")), True

    evs = [e for e in events if kind(e)]
    spans = [(_ts(e.get("fetch_started_at")), _ts(e.get("fetch_finished_at")) or _ts(e.get("fetch_started_at")))
             for e in evs]
    solo, over = [0, 0], [0, 0]
    joined = 0
    overlap_rows = []
    for i, e in enumerate(evs):
        s0, s1 = spans[i]
        if s0 is None:
            continue
        partners = []
        for j, f in enumerate(evs):
            if i == j or f.get("pid") == e.get("pid"):      # 同一プロセス内の逐次取得は並走ではない
                continue
            f0, f1 = spans[j]
            if f0 is None:
                continue
            if (f0 - s1).total_seconds() <= OVERLAP_SEC and (s0 - f1).total_seconds() <= OVERLAP_SEC:
                partners.append(kind(f))
        ok, has_cap = success(e)
        joined += int(has_cap)
        bucket = over if partners else solo
        bucket[0] += int(ok)
        bucket[1] += 1
        if partners:
            overlap_rows.append({"kind": kind(e), "race_id": e.get("race_id"), "spec": e.get("spec"),
                                 "started": e.get("fetch_started_at"), "ok": ok,
                                 "partners": sorted(set(partners))})
    rate = lambda b: round(b[0] / b[1], 4) if b[1] else None
    conc = {"window_sec": OVERLAP_SEC, "events": len(evs), "joined_to_capture": joined,
            "solo": {"ok": solo[0], "n": solo[1], "rate": rate(solo)},
            "overlapped": {"ok": over[0], "n": over[1], "rate": rate(over)},
            "overlapped_failures": [r for r in overlap_rows if not r["ok"]][:50],
            "by_kind": {k: {"events": sum(1 for e in evs if kind(e) == k)} for k in set(CONCURRENCY_KINDS.values())}}

    five_spec = _q(span.get("t2_candidate", []))
    decision_reasons = []
    if over[1] and over[0] / over[1] < MIN_OVERLAP_SUCCESS:
        decision_reasons.append(f"overlapped success {over[0]}/{over[1]} < {MIN_OVERLAP_SUCCESS}")
    if corruption:
        decision_reasons.append(f"00:00-type corruption x{len(corruption)}")
    if decision_reasons:
        decision = "QUEUE_SERIALIZATION_REQUIRED"
    elif not over[1]:
        decision = "INSUFFICIENT_OVERLAP_OBSERVED"
    else:
        decision = "CONCURRENCY_OK"
    result_like = sorted(p for p in _OPENED if any(m in p.lower() for m in FORBIDDEN_MARKERS))
    assert not result_like, f"outcome-like file opened: {result_like}"
    return {
        "role": "observation plan v2.1 Phase 2 Stage 0 (label-free). No outcome, payout, ROI, performance "
                "or band selection is read or computed.",
        "dates": dates, "scheduled_races": len(sched_all), "records": len(recs), "journal_events": len(events),
        "missing": missing,
        "timing": {"fetch_duration_sec": {f"{k[0]}/{k[1]}": _q(v) for k, v in dur.items()},
                   "t2_candidate_5spec_span_sec": five_spec,
                   "t2_5spec_p95_over_60s": bool(five_spec.get("p95", 0) > FIVE_SPEC_P95_SEC),
                   "announce_delay_sec": {f"{k[0]}/{k[1]}": _q(v) for k, v in ann_delay.items()},
                   "seconds_to_post": {f"{k[0]}/{k[1]}": {b: _q(v) for b, v in d.items()}
                                       for k, d in to_post.items()}},
        "coverage": coverage, "coverage_fail_over_1pct": coverage_fail,
        "wakuren_rule_mismatch_over_1pct": wakuren_fail,
        "raw_parser": {k: {"fail": f, "n": n} for k, (f, n) in parse_checks.items()},
        "concurrency": conc, "corruption_0000": corruption,
        "decision": decision, "decision_reasons": decision_reasons,
        "files_opened_outcome_like": result_like,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="+", required=True)
    ap.add_argument("--dry", action="store_true", help="forward_prices_dry / journal_dry を監査")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    _track_opens()
    froot = FORWARD_ROOT.parent / "forward_prices_dry" if args.dry else FORWARD_ROOT
    jroot = JOURNAL_ROOT.parent / "jvlink_fetch_journal_dry" if args.dry else JOURNAL_ROOT
    rep = audit(args.dates, froot, jroot)
    out = Path(args.out) if args.out else BASE / "reports" / f"obs_stage0_{'_'.join(args.dates)}{'_dry' if args.dry else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("dates", "scheduled_races", "records", "decision", "decision_reasons",
                                          "coverage_fail_over_1pct")}, ensure_ascii=False))
    print(f"[obs_stage0] -> {out}")
    return 0 if rep["decision"] != "QUEUE_SERIALIZATION_REQUIRED" else 3


if __name__ == "__main__":
    raise SystemExit(main())
