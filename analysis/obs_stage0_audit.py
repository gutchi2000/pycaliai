# -*- coding: utf-8 -*-
"""
obs_stage0_audit.py — 観測計画 v2.1 Phase 2 Stage 0（最初の 2 開催日）label-free 監査
====================================================================================
監査するのは次だけ。outcome・払戻・ROI・性能・帯選択は読まない / 計算しない。

  1. 欠損率        必須 stage × spec ごとに、予定レースのうち録（capture）が無い割合
  2. 時刻          取得所要（spec 別・5 spec 合計）、発表遅延（取得完了 − 発表月日時分）、発走までの秒
  3. 全組被覆      発売中の組数 = 出走頭数からの期待値、非空白 slot = 登録頭数からの期待値（枠連を含む）
  4. raw/parser 一致 保存 raw の再構造化 = 保存済み構造化、slot parser = 本番の実録突合済み parser
  5. 並走成功率    取得ジャーナルの全 JV-Link 取得（本番 t10/t20/close_late/vote/exp05fs_t35、T−2、三連複、
                   final RT、蓄積系 STOCK、jvlink_changes、EXP05-F calendar、その他）を重なり相手にする。
                   別プロセスの取得と ±30 秒以内に重なった取得の成功率を、単独取得の成功率と比べる。
                   スケジュールが重ならないことは仮定しない（実際の開始・終了時刻だけで判定する）。
                   成功率を測るのは forward store の capture と突き合わせられる価格取得
                   （成功 = rc 0・録あり・race_key 一致・全組被覆）。それ以外（changes・calendar・STOCK 等）は
                   重なり相手としてだけ数え、rc の分布を記述する。
  +  00:00 型破損  RT 録の発表時分 00:00、または発走予定 00:00

判定と終了コード（契約 → 並走の順）:
  CONTRACT_NOT_MET (4)              必須 stage/spec の欠損率 > 1%、被覆不一致率 > 1%（枠連を含む）、
                                    raw/parser 不一致が 1 件以上、または process 名が識別できない取得がある
  QUEUE_SERIALIZATION_REQUIRED (3)  契約は満たすが、重なり時の成功率 < 95%、または 00:00 型破損が 1 件以上
  INSUFFICIENT_OVERLAP_OBSERVED (5) 契約は満たすが、重なった価格取得が 1 件も観測されない
  CONCURRENCY_OK (0)                上記以外
契約と並走の判定は両方とも常に計算して報告する（先に当たった方が decision になる）。

python -m analysis.obs_stage0_audit --dates 20261003 20261004 [--dry]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
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
MAX_MISSING_RATE = 0.01
FIVE_SPEC_P95_SEC = 60.0
ALL5 = ("0B31", "0B32", "0B33", "0B34", "0B35")
PROD4 = ("0B31", "0B32", "0B33", "0B34")
# Phase 2 の必須 stage × spec（予定レース全件に録が要る）。t20 / vote / exp05fs_t35 は他系統の契約なので記述のみ。
REQUIRED_STAGE_SPECS = {"t10": PROD4, "close_late": PROD4, "t2_candidate": ALL5, "trio_t10": ("0B35", "0B31"),
                        "final_rt_candidate": ALL5, "final_stock_candidate": ALL5}
DESCRIPTIVE_STAGES = ("t20", "vote", "exp05fs_t35")
DECISION_EXIT = {"CONCURRENCY_OK": 0, "QUEUE_SERIALIZATION_REQUIRED": 3, "CONTRACT_NOT_MET": 4,
                 "INSUFFICIENT_OVERLAP_OBSERVED": 5}
UNIDENTIFIED_PROCESSES = {"", "unknown", "python", "pythonw", "-c", "None"}
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


def event_kind(e: dict) -> str:
    """取得ジャーナルの 1 件を識別名にする。process 名が無い・不明な取得は 'UNIDENTIFIED:...'。"""
    proc = str(e.get("process") or "")
    st = canonical_stage(e.get("stage")) if e.get("stage") else ""
    if proc in UNIDENTIFIED_PROCESSES:
        return f"UNIDENTIFIED:{proc or '-'}"
    if proc == "jvlink_odds":
        return st or "jvlink_odds"
    if proc == "jvlink_obs":
        return "final_stock_candidate(STOCK)" if e.get("spec") == "RACE" else (st or "jvlink_obs")
    if proc == "trio_shadow":
        return "trio_t10"
    if proc == "jvlink_changes":
        return "jvlink_changes"
    if proc == "exp05fs_calendar":
        return "exp05fs_calendar"
    return f"{proc}:{st}" if st else proc


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


def concurrency(events: list[dict], capture_index: dict) -> dict:
    """全取得を重なり相手にした並走成功率。スケジュールの非重複は仮定しない。"""
    evs = [e for e in events if "_unreadable" not in e]
    spans = []
    for e in evs:
        t0 = _ts(e.get("fetch_started_at"))
        t1 = _ts(e.get("fetch_finished_at")) or t0
        spans.append((t0, t1))
    kinds = [event_kind(e) for e in evs]

    def measured(e) -> tuple[bool | None, bool]:
        """(success or None when not measurable, joined_to_capture)"""
        cap = capture_index.get((canonical_stage(e.get("stage")) if e.get("stage") else None, e.get("race_id"),
                                 e.get("spec"), str(e.get("fetch_started_at"))))
        if cap is None:
            return None, False
        base_ok = e.get("rc_init") == 0 and e.get("rc_open") == 0 and (e.get("n_records_returned") or 0) > 0
        return bool(base_ok and cap.get("ok")), True

    per_kind = defaultdict(lambda: {"events": 0, "solo_ok": 0, "solo_n": 0, "over_ok": 0, "over_n": 0,
                                    "partner_kinds": Counter(), "rc_nonzero": 0, "errors": 0})
    solo, over = [0, 0], [0, 0]
    failures = []
    for i, e in enumerate(evs):
        k = kinds[i]
        pk = per_kind[k]
        pk["events"] += 1
        pk["rc_nonzero"] += int(e.get("rc_init") not in (0, None) or e.get("rc_open") not in (0, None))
        pk["errors"] += int(bool(e.get("error")))
        s0, s1 = spans[i]
        partners = []
        if s0 is not None:
            for j, f in enumerate(evs):
                if i == j or f.get("pid") == e.get("pid"):        # 同一プロセス内の逐次取得は並走ではない
                    continue
                f0, f1 = spans[j]
                if f0 is None:
                    continue
                if (f0 - s1).total_seconds() <= OVERLAP_SEC and (s0 - f1).total_seconds() <= OVERLAP_SEC:
                    partners.append(kinds[j])
        pk["partner_kinds"].update(set(partners))
        ok, joined = measured(e)
        if ok is None:
            continue
        if partners:
            over[0] += int(ok); over[1] += 1
            pk["over_ok"] += int(ok); pk["over_n"] += 1
            if not ok:
                failures.append({"kind": k, "race_id": e.get("race_id"), "spec": e.get("spec"),
                                 "started": e.get("fetch_started_at"), "partners": sorted(set(partners))})
        else:
            solo[0] += int(ok); solo[1] += 1
            pk["solo_ok"] += int(ok); pk["solo_n"] += 1
    rate = lambda a, n: round(a / n, 4) if n else None
    table = {k: {**{x: v[x] for x in ("events", "solo_n", "over_n", "rc_nonzero", "errors")},
                 "solo_rate": rate(v["solo_ok"], v["solo_n"]), "over_rate": rate(v["over_ok"], v["over_n"]),
                 "partner_kinds": dict(v["partner_kinds"])} for k, v in sorted(per_kind.items())}
    return {"window_sec": OVERLAP_SEC, "events": len(evs), "unreadable": len(events) - len(evs),
            "measured": solo[1] + over[1],
            "solo": {"ok": solo[0], "n": solo[1], "rate": rate(*solo)},
            "overlapped": {"ok": over[0], "n": over[1], "rate": rate(*over)},
            "overlapped_failures": failures[:50], "by_kind": table,
            "unidentified_events": sum(1 for k in kinds if k.startswith("UNIDENTIFIED")),
            "assumes_schedule_non_overlap": False}


def audit(dates: list[str], forward_root: Path = FORWARD_ROOT, journal_root: Path = JOURNAL_ROOT,
          schedule: dict[str, dict[str, str]] | None = None, required: dict | None = None) -> dict:
    schedule = schedule if schedule is not None else {d: scheduled_races(d) for d in dates}
    required = REQUIRED_STAGE_SPECS if required is None else required
    sched_all = {rid: post for d in dates for rid, post in schedule.get(d, {}).items()}
    recs = load_records(forward_root, dates)
    events = load_journal(journal_root, dates)

    have = defaultdict(set)
    v1_only = defaultdict(set)
    cov = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    parse_checks = defaultdict(lambda: [0, 0])
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

    n_sched = len(sched_all)
    missing, missing_fail = {}, []
    for st, specs in required.items():
        for spec in specs:
            got = have[(st, spec)] & set(sched_all)
            rate_m = round(1 - len(got) / n_sched, 4) if n_sched else None
            missing[f"{st}/{spec}"] = {"scheduled": n_sched, "with_capture": len(got), "missing_rate": rate_m,
                                       "missing_races": sorted(set(sched_all) - got)[:30], "required": True}
            if rate_m is None or rate_m > MAX_MISSING_RATE:
                missing_fail.append(f"{st}/{spec}")
    for st in DESCRIPTIVE_STAGES:
        for spec in PROD4:
            got = have[(st, spec)] & set(sched_all)
            if got:
                missing[f"{st}/{spec}"] = {"scheduled": n_sched, "with_capture": len(got), "required": False}
    for st, rids in v1_only.items():
        missing[f"{st}/v1_without_raw"] = {"races": len(rids), "required": False}
    coverage = {f"{st}/{spec}": {b: {"mismatch": m, "n": n, "rate": round(m / n, 4) if n else None}
                                 for b, (m, n) in blocks.items()}
                for (st, spec), blocks in cov.items()}
    coverage_fail = sorted(f"{k}/{b}" for k, blocks in coverage.items() for b, v in blocks.items()
                           if v["rate"] is not None and v["rate"] > MAX_COVERAGE_MISMATCH)
    parse_fail = sorted(k for k, (f, n) in parse_checks.items() if f > 0)
    conc = concurrency(events, capture_index)

    contract_reasons = []
    if missing_fail:
        contract_reasons.append(f"required stage/spec missing > {MAX_MISSING_RATE:.0%}: {missing_fail}")
    if coverage_fail:
        contract_reasons.append(f"coverage mismatch > {MAX_COVERAGE_MISMATCH:.0%}: {coverage_fail}")
    if parse_fail:
        contract_reasons.append(f"raw/parser check failures: {parse_fail}")
    if conc["unidentified_events"]:
        contract_reasons.append(f"journal events without an identifiable process: {conc['unidentified_events']}")
    queue_reasons = []
    ov = conc["overlapped"]
    if ov["n"] and ov["ok"] / ov["n"] < MIN_OVERLAP_SUCCESS:
        queue_reasons.append(f"overlapped success {ov['ok']}/{ov['n']} < {MIN_OVERLAP_SUCCESS}")
    if corruption:
        queue_reasons.append(f"00:00-type corruption x{len(corruption)}")
    concurrency_verdict = ("QUEUE_SERIALIZATION_REQUIRED" if queue_reasons
                           else "INSUFFICIENT_OVERLAP_OBSERVED" if not ov["n"] else "CONCURRENCY_OK")
    decision = "CONTRACT_NOT_MET" if contract_reasons else concurrency_verdict

    five_spec = _q(span.get("t2_candidate", []))
    result_like = sorted(p for p in _OPENED if any(m in p.lower() for m in FORBIDDEN_MARKERS))
    assert not result_like, f"outcome-like file opened: {result_like}"
    return {
        "role": "observation plan v2.1 Phase 2 Stage 0 (label-free). No outcome, payout, ROI, performance "
                "or band selection is read or computed.",
        "dates": dates, "scheduled_races": n_sched, "records": len(recs), "journal_events": len(events),
        "missing": missing,
        "timing": {"fetch_duration_sec": {f"{k[0]}/{k[1]}": _q(v) for k, v in dur.items()},
                   "t2_candidate_5spec_span_sec": five_spec,
                   "t2_5spec_p95_over_60s": bool(five_spec.get("p95", 0) > FIVE_SPEC_P95_SEC),
                   "announce_delay_sec": {f"{k[0]}/{k[1]}": _q(v) for k, v in ann_delay.items()},
                   "seconds_to_post": {f"{k[0]}/{k[1]}": {b: _q(v) for b, v in d.items()}
                                       for k, d in to_post.items()}},
        "coverage": coverage, "coverage_fail_over_1pct": coverage_fail,
        "raw_parser": {k: {"fail": f, "n": n} for k, (f, n) in parse_checks.items()},
        "concurrency": conc, "corruption_0000": corruption,
        "contract": {"met": not contract_reasons, "reasons": contract_reasons},
        "concurrency_verdict": concurrency_verdict, "concurrency_reasons": queue_reasons,
        "decision": decision, "exit_code": DECISION_EXIT[decision],
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
    print(json.dumps({k: rep[k] for k in ("dates", "scheduled_races", "records", "decision", "contract",
                                          "concurrency_verdict", "concurrency_reasons")}, ensure_ascii=False))
    print(f"[obs_stage0] -> {out}")
    return rep["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
