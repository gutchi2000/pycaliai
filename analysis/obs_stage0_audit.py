# -*- coding: utf-8 -*-
"""
obs_stage0_audit.py — 観測計画 v2.1 Phase 2 Stage 0（最初の 2 開催日）label-free 監査
====================================================================================
監査するのは次だけ。outcome・払戻・ROI・性能・帯選択は読まない / 計算しない。

  1. 欠損率        必須 stage × spec ごとに、予定レースのうち録（capture）が無い割合
  2. 時刻          取得所要（spec 別・5 spec 合計）、発表遅延（取得完了 − 発表月日時分）、発走までの秒
  3. 全組被覆      発売中の組数 = 発売中の頭数からの期待値（jv_records の構造化版 STRUCTURE_VERSION の式。
                   発走前は 登録頭数 − 取消）、非空白 slot = 登録頭数からの期待値（枠連は発売フラグ '0' なら 0）。
                   同じ stage 録の中で、単勝から導いた取消馬と他の券種から導いた取消馬が一致すること。
                   契約に数えるのは範囲内（--scope）の stage だけ。範囲外は記述。
  4. raw/parser 一致 保存 raw の再構造化 = 保存済み構造化（同じ構造化版の録どうしだけ比べる。旧版の録を
                   今の式で構造化し直した差は不一致に数えず、件数だけ記述する）、slot parser = 本番の
                   実録突合済み parser
  5. 並走成功率    取得ジャーナルの全 JV-Link 取得を重なり相手にする。主判定は**取得区間の交差**（実際に
                   同時に JV-Link を使っていた取得どうし）。±30 秒の近接は記述として併記する。
                   成功 = rc 0・録あり・最新録の race_key 一致・録長一致・malformed 0・（枠連以外で）
                   発売中の組数 = 期待値。被覆の合否（契約）とは別に数える。
                   成功率を測るのは forward store の capture と突き合わせられる価格取得。それ以外
                   （changes・calendar・STOCK 等）は重なり相手としてだけ数える。
  6. タスク記録    観測タスクの開始・終了（obs_task_runs、F7）。開始のみの run = 強制終了。
                   TASK_RUNS_REQUIRED_FROM 以降の開催日は、期待タスクの開始・終了 100% が契約。
  +  00:00 型破損  RT 録の発表時分 00:00、または発走予定 00:00

範囲（--scope）:
  night（既定）  当夜の契約。t10・close_late・t2_candidate・trio_t10・final_rt_candidate
  stock          蓄積系だけ。final_stock_candidate（配信後に評価する）
  full           両方

保存先（--dry）: 観測 collector の stage（t2_candidate・trio_t10・final_*・conc_test）は *_dry の store から、
本番経路の stage（t10・close_late・t20・vote・exp05fs_t35）は本番 store から読む。並走の相手は
dry と本番の journal の和。report の stage_sources に stage ごとの読込み元を記録する。

判定と終了コード（契約 → 並走の順）:
  CONTRACT_NOT_MET (4)              必須 stage/spec の欠損数 > max(1% × 予定レース数, 1 race)（2 開催日合計）、
                                    原因を帰属できない欠損が 1 件以上、被覆不一致率 > 1%（枠連を含む）、
                                    raw/parser 不一致が 1 件以上、process 名が識別できない取得がある、
                                    またはタスク記録の契約（6）を満たさない
  QUEUE_SERIALIZATION_REQUIRED (3)  契約は満たすが、重なり時の成功率 < 95%、または 00:00 型破損が 1 件以上
  INSUFFICIENT_OVERLAP_OBSERVED (5) 契約は満たすが、重なった価格取得が 1 件も観測されない
  CONCURRENCY_OK (0)                上記以外（Stage 0 通過）
  ALREADY_RUNNING (7)               監査そのものの二重起動（lock を取れない、または同じ条件の実行が直近にある）。
                                    report・ledger とも何も書かずに終わる
契約と並走の判定は両方とも常に計算して報告する（先に当たった方が decision になる）。

欠損 race の原因帰属（全件。原因不明のまま集計しない）:
  task_killed            そのタスクの run が開始だけで終了記録が無い（起動器ごと強制終了）
  task_abnormal_exit     そのタスクの run が 0 以外の終了コードで終わった
  task_not_fired         その race・stage・spec の取得ジャーナルも録も無い
  fetch_rc               取得の rc_init / rc_open が 0 でない、または例外
  race_key_mismatch      録はあるが、最新録のレースキーが要求 race と一致しない
  no_records             取得は rc 0 だが、その種別の録が 0 件
  record_not_stored      取得ジャーナルでは録ありだが、forward store に録が無い（保存失敗）
  spec_capture_absent    その stage の録はあるが、その spec の capture が無い
  record_without_raw_v1  その stage の録が schema v1（raw なし）
  stock_session_failed   蓄積系セッション（STOCK）が失敗
  stock_race_absent      蓄積系セッションは成功したが、その race の O1〜O5 録が無い
  unattributed           上記のどれにも当たらない → それ自体で CONTRACT_NOT_MET

本監査は判定・理由・欠損 race・原因を report と ledger（data/obs_stage0_ledger.jsonl、追記専用）へ書く。
既存の report は上書きしない（同名があれば時刻を付けた別名で書く）。収集処理は止めない。
性能・ROI・帯選択へは進まない（analysis.obs_guard が ledger を見て拒否する）。

python -m analysis.obs_stage0_audit --dates 20261010 --dry [--scope night|stock|full] [--rejudge-of LABEL]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import jv_records as JR  # noqa: E402
import obs_task_runs as TR  # noqa: E402
from forward_prices import FORWARD_ROOT, canonical_stage, read_snapshot  # noqa: E402

AUDIT_VERSION = "stage0-audit-2"
JOURNAL_ROOT = BASE / "data" / "jvlink_fetch_journal"
DRY_FORWARD_ROOT = FORWARD_ROOT.parent / "forward_prices_dry"
DRY_JOURNAL_ROOT = JOURNAL_ROOT.parent / "jvlink_fetch_journal_dry"
STAGE0_LEDGER = BASE / "data" / "obs_stage0_ledger.jsonl"
LOCK_PATH = BASE / "data" / ".locks" / "obs_stage0_audit.lock"
LOCK_STALE_SEC = 1800
DUPLICATE_WINDOW_SEC = 900
EXIT_ALREADY_RUNNING = 7
STAGE0_PASS = "CONCURRENCY_OK"
MISSING_CAUSES = ("task_killed", "task_abnormal_exit", "task_not_fired", "fetch_rc", "race_key_mismatch",
                  "no_records", "record_not_stored", "spec_capture_absent", "record_without_raw_v1",
                  "stock_session_failed", "stock_race_absent", "unattributed")
OVERLAP_SEC = 30.0                 # 記述用の近接窓（主判定は取得区間の交差）
MIN_OVERLAP_SUCCESS = 0.95
MAX_COVERAGE_MISMATCH = 0.01
MAX_MISSING_RATE = 0.01
FIVE_SPEC_P95_SEC = 60.0
ALL5 = ("0B31", "0B32", "0B33", "0B34", "0B35")
PROD4 = ("0B31", "0B32", "0B33", "0B34")
NIGHT_REQUIRED = {"t10": PROD4, "close_late": PROD4, "t2_candidate": ALL5, "trio_t10": ("0B35", "0B31"),
                  "final_rt_candidate": ALL5}
STOCK_REQUIRED = {"final_stock_candidate": ALL5}
SCOPES = {"night": NIGHT_REQUIRED, "stock": STOCK_REQUIRED, "full": {**NIGHT_REQUIRED, **STOCK_REQUIRED}}
REQUIRED_STAGE_SPECS = SCOPES["full"]
DESCRIPTIVE_STAGES = ("t20", "vote", "exp05fs_t35")
# Dry のとき観測 collector の stage は dry store、本番経路の stage は本番 store から読む（F1）
OBS_STAGES = ("t2_candidate", "trio_t10", "final_rt_candidate", "final_stock_candidate", "conc_test")
PROD_STAGES = ("t10", "close_late", "t20", "vote", "exp05fs_t35")
# この日付以降の開催日はタスクの開始・終了記録 100% が契約（F7 の窓なし起動器が動き始める日）
TASK_RUNS_REQUIRED_FROM = "20261010"
SCOPE_TASKS = {"night": ("t2", "trio", "final"), "stock": ("stock",), "full": ("t2", "trio", "final", "stock")}
STAGE_TASK = {"t2_candidate": "t2", "trio_t10": "trio", "final_rt_candidate": "final",
              "final_stock_candidate": "stock"}
DECISION_EXIT = {"CONCURRENCY_OK": 0, "QUEUE_SERIALIZATION_REQUIRED": 3, "CONTRACT_NOT_MET": 4,
                 "INSUFFICIENT_OVERLAP_OBSERVED": 5}
UNIDENTIFIED_PROCESSES = {"", "unknown", "python", "pythonw", "-c", "None"}
FORBIDDEN_MARKERS = ("kekka", "payout", "haraimodoshi", "wide_payouts", "live_results", "results.json",
                     "rows_2013_2023")
RESTRUCTURE_FIELDS = ("odds_parsed", "counts", "kubun", "announced_at", "votes_total", "hatsubai_flag")

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
    if proc in ("jvlink_changes", "exp05fs_calendar"):
        return proc
    if proc.startswith("obs_conc_test"):
        return "conc_test"
    return f"{proc}:{st}" if st else proc


def scheduled_races(date: str) -> dict[str, str]:
    from jvlink_trio_odds import build_schedule
    return {rid: pt.isoformat() for pt, rid in build_schedule(date)}


def store_sources(dry: bool) -> list[dict]:
    """監査の読込み元。Dry は stage ごとに dry store と本番 store を分ける（F1）。"""
    if not dry:
        return [{"name": "production", "forward_root": FORWARD_ROOT, "journal_root": JOURNAL_ROOT, "stages": None}]
    return [{"name": "dry", "forward_root": DRY_FORWARD_ROOT, "journal_root": DRY_JOURNAL_ROOT, "stages": OBS_STAGES},
            {"name": "production", "forward_root": FORWARD_ROOT, "journal_root": JOURNAL_ROOT,
             "stages": PROD_STAGES}]


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


def _rc_bad(x: dict) -> bool:
    return x.get("rc_init") not in (0, None) or x.get("rc_open") not in (0, None) or bool(x.get("error"))


def capture_success(cap: dict, last: dict | None) -> bool:
    """並走の成否（F3）。被覆の契約とは独立: rc 0・録あり・race_key・録長・malformed 0・
    （枠連以外で）発売中の組数 = 期待値。last は今の構造化版で構造化し直した最新録。"""
    if _rc_bad(cap) or last is None:
        return False
    if not (last.get("race_key_ok") and last.get("length_ok")):
        return False
    for block, c in (last.get("counts") or {}).items():
        if c.get("malformed"):
            return False
        if block != "wakuren" and not c.get("priced_match"):
            return False
    return True


def _overlaps(s0, s1, f0, f1, mode: str) -> bool:
    if mode == "intersection":
        return f0 < s1 and s0 < f1
    return (f0 - s1).total_seconds() <= OVERLAP_SEC and (s0 - f1).total_seconds() <= OVERLAP_SEC


def concurrency(events: list[dict], success_index: dict, mode: str = "intersection") -> dict:
    """全取得を重なり相手にした並走成功率。mode='intersection'（主判定: 取得区間の交差）/
    'window30'（記述: ±30 秒の近接）。スケジュールの非重複は仮定しない。"""
    evs = [e for e in events if "_unreadable" not in e]
    spans = []
    for e in evs:
        t0 = _ts(e.get("fetch_started_at"))
        t1 = _ts(e.get("fetch_finished_at")) or t0
        spans.append((t0, t1))
    kinds = [event_kind(e) for e in evs]

    def measured(e) -> bool | None:
        ok = success_index.get((canonical_stage(e.get("stage")) if e.get("stage") else None, e.get("race_id"),
                                e.get("spec"), str(e.get("fetch_started_at"))))
        if ok is None:
            return None
        base_ok = e.get("rc_init") == 0 and e.get("rc_open") == 0 and (e.get("n_records_returned") or 0) > 0
        return bool(base_ok and ok)

    per_kind = defaultdict(lambda: {"events": 0, "solo_ok": 0, "solo_n": 0, "over_ok": 0, "over_n": 0,
                                    "partner_kinds": Counter(), "rc_nonzero": 0, "errors": 0})
    solo, over = [0, 0], [0, 0]
    failures = []
    pairs = set()
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
                if _overlaps(s0, s1, f0, f1, mode):
                    partners.append(kinds[j])
                    pairs.add((min(i, j), max(i, j)))
        pk["partner_kinds"].update(set(partners))
        ok = measured(e)
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
    return {"definition": mode, "window_sec": OVERLAP_SEC if mode == "window30" else 0.0,
            "events": len(evs), "unreadable": len(events) - len(evs), "measured": solo[1] + over[1],
            "overlapping_event_pairs": len(pairs),
            "solo": {"ok": solo[0], "n": solo[1], "rate": rate(*solo)},
            "overlapped": {"ok": over[0], "n": over[1], "rate": rate(*over)},
            "overlapped_failures": failures[:50], "by_kind": table,
            "unidentified_events": sum(1 for k in kinds if k.startswith("UNIDENTIFIED")),
            "assumes_schedule_non_overlap": False}


def missing_threshold(n_scheduled: int) -> float:
    """許容欠損数 = max(1% × 予定レース数, 1 race)。欠損数がこれを超えたら契約未達。"""
    return max(MAX_MISSING_RATE * n_scheduled, 1.0)


def attribute_missing(st: str, spec: str, rid: str, cap_by: dict, rec_by: dict, jidx: dict,
                      stock_sessions: list[dict], task_state=None) -> tuple[str, str]:
    """欠損 1 件の原因 (cause, detail)。どれにも当たらなければ 'unattributed'。"""
    caps = cap_by.get((st, spec, rid), [])
    if caps:
        last = max(caps, key=lambda c: str(c.get("fetch_started_at")))
        recs = last.get("records") or []
        if recs and not recs[-1].get("race_key_ok"):
            return "race_key_mismatch", f"record race_key {recs[-1].get('race_key')!r}"
        if _rc_bad(last):
            return "fetch_rc", f"rc_init={last.get('rc_init')} rc_open={last.get('rc_open')} error={last.get('error')}"
        if not recs:
            return "no_records", f"rc 0, {last.get('n_records_returned')} records returned, 0 of kind"
        return "unattributed", "capture with matching records exists but was not counted"
    if (st, rid) in rec_by:
        return (("record_without_raw_v1", "schema v1 record without jv_captures") if rec_by[(st, rid)] == "v1"
                else ("spec_capture_absent", f"record for {st} has no {spec} capture"))
    evs = jidx.get((st, rid, spec), [])
    if evs:
        e = max(evs, key=lambda x: str(x.get("fetch_started_at")))
        if _rc_bad(e):
            return "fetch_rc", f"journal rc_init={e.get('rc_init')} rc_open={e.get('rc_open')} error={e.get('error')}"
        if not (e.get("n_records_returned") or 0):
            return "no_records", "journal: rc 0 and 0 records"
        return "record_not_stored", f"journal: {e.get('n_records_returned')} records fetched, no stored record"
    state = task_state(st, rid) if task_state else None
    if state == "killed":
        return "task_killed", "task run has a start record and no end record (forced termination)"
    if state == "abnormal":
        return "task_abnormal_exit", "task run ended with a non-zero exit code"
    if st == "final_stock_candidate":
        sess = [s for s in stock_sessions if str(s.get("fetch_started_at", ""))[:10].replace("-", "") >= rid[:8]]
        if not sess:
            return "task_not_fired", "no STOCK session on/after the race date"
        s = max(sess, key=lambda x: str(x.get("fetch_started_at")))
        if _rc_bad(s):
            return "stock_session_failed", f"STOCK rc_init={s.get('rc_init')} rc_open={s.get('rc_open')} error={s.get('error')}"
        return "stock_race_absent", "STOCK session ok but no O1-O5 record for the race"
    return "task_not_fired", "no journal event and no record for this race/stage/spec"


def _scratched_by_block(last: dict) -> dict[str, list]:
    return {b: c.get("scratched") for b, c in (last.get("counts") or {}).items() if c.get("scratched") is not None}


def expected_tasks(schedule: dict[str, dict[str, str]], scope: str) -> dict[str, set]:
    out = {}
    for d, races in schedule.items():
        keys = set()
        for t in SCOPE_TASKS[scope]:
            if t in ("t2", "trio"):
                keys |= {(t, rid) for rid in races}
            else:
                keys.add((t, d))
        out[d] = keys
    return out


def audit(dates: list[str], forward_root: Path = FORWARD_ROOT, journal_root: Path = JOURNAL_ROOT,
          schedule: dict[str, dict[str, str]] | None = None, required: dict | None = None, *,
          sources: list[dict] | None = None, scope: str = "night", dry: bool = False,
          task_runs_root: Path | None = None, task_runs_required_from: str = TASK_RUNS_REQUIRED_FROM) -> dict:
    schedule = schedule if schedule is not None else {d: scheduled_races(d) for d in dates}
    required = SCOPES[scope] if required is None else required
    sources = sources if sources is not None else [
        {"name": "given", "forward_root": forward_root, "journal_root": journal_root, "stages": None}]
    sched_all = {rid: post for d in dates for rid, post in schedule.get(d, {}).items()}
    recs, events, stage_sources = [], [], defaultdict(set)
    for src in sources:
        for p, rec in load_records(Path(src["forward_root"]), dates):
            if src["stages"] is None or rec["_stage"] in src["stages"]:
                recs.append((p, rec))
                stage_sources[rec["_stage"]].add(src["name"])
        events += load_journal(Path(src["journal_root"]), dates)

    have = defaultdict(set)
    v1_only = defaultdict(set)
    cov = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    parse_checks = defaultdict(lambda: [0, 0])
    version_skips = Counter()
    version_would_differ = Counter()
    dur = defaultdict(list)
    span = defaultdict(list)
    ann_delay = defaultdict(list)
    to_post = defaultdict(lambda: defaultdict(list))
    corruption: list[dict] = []
    success_index: dict[tuple, bool] = {}
    cap_by: dict[tuple, list] = defaultdict(list)
    rec_by: dict[tuple, str] = {}
    for path, rec in recs:
        st, rid = rec["_stage"], rec.get("race_id")
        caps = rec.get("jv_captures")
        if not caps:
            v1_only[st].add(rid)
            rec_by.setdefault((st, rid), "v1")
            continue
        rec_by[(st, rid)] = "v2"
        sp = _ts(rec.get("scheduled_post"))
        if rec.get("scheduled_post") and str(rec["scheduled_post"])[11:16] == "00:00":
            corruption.append({"race_id": rid, "stage": st, "what": "scheduled_post 00:00"})
        starts, ends = [], []
        last_again: dict[str, dict] = {}
        for cap in caps:
            spec = cap.get("spec")
            key = (st, spec)
            cap_by[(st, spec, rid)].append(cap)
            if cap.get("records") and cap["records"][-1].get("race_key_ok"):
                have[key].add(rid)                      # 有効 = その種別の録があり、最新録のレースキーが一致
            t0, t1 = _ts(cap.get("fetch_started_at")), _ts(cap.get("fetch_finished_at"))
            if t0 and t1:
                dur[key].append((t1 - t0).total_seconds())
                starts.append(t0)
                ends.append(t1)
            again_last = None
            for r in cap.get("records", []):
                again = JR.structure_record(r["raw"], spec, rid, r.get("stream", "rt"))
                again_last = again
                for block, c in (again.get("counts") or {}).items():
                    m = c.get("priced_match") if block != "wakuren" else c.get("nonblank_match")
                    if m is not None:
                        cov[key][block][0] += int(not m)
                        cov[key][block][1] += 1
                stored_version = r.get("structure_version", JR.LEGACY_STRUCTURE_VERSION)
                same = all(again.get(k) == r.get(k) for k in RESTRUCTURE_FIELDS)
                checks = [("validated_parser_match", r.get("validated_parser_match") is True),
                          ("race_key_ok", bool(r.get("race_key_ok"))),
                          ("length_ok", bool(r.get("length_ok"))),
                          ("raw_sha256", hashlib.sha256(r["raw"].encode("utf-8", "replace")).hexdigest()
                           == r.get("raw_sha256"))]
                if stored_version == JR.STRUCTURE_VERSION:
                    checks.append(("restructure_equal", same))
                else:                                    # 旧版の録の再構造化差は不一致に数えない（B5）
                    version_skips[stored_version] += 1
                    version_would_differ[stored_version] += int(not same)
                for name, ok in checks:
                    parse_checks[name][0] += int(not ok)
                    parse_checks[name][1] += 1
                if "announce_0000" in (r.get("anomalies") or []):
                    corruption.append({"race_id": rid, "stage": st, "spec": spec, "what": "RT announce 00:00",
                                       "announce_raw": r.get("announce_raw")})
            if again_last is not None:
                last_again[spec] = again_last
            success_index[(st, rid, spec, str(cap.get("fetch_started_at")))] = capture_success(cap, again_last)
            last = cap["records"][-1] if cap.get("records") else None
            if last and cap.get("stream") == "rt":
                ta = _ts(last.get("announced_at"))
                if ta and t1:
                    ann_delay[key].append((t1 - ta).total_seconds())
                if sp and ta:
                    to_post[key]["announce_basis"].append((sp - ta).total_seconds())
                if sp and t1:
                    to_post[key]["fetch_basis"].append((sp - t1).total_seconds())
        # 同じ stage 録の中で、単勝から導いた取消馬 = 他の券種から導いた取消馬（B3）
        ref = _scratched_by_block(last_again["0B31"]).get("tansho") if "0B31" in last_again else None
        if ref is not None:
            for spec, a in last_again.items():
                for block, scr in _scratched_by_block(a).items():
                    if (spec, block) == ("0B31", "tansho"):
                        continue
                    cov[(st, spec)]["scratch_consistency"][0] += int(sorted(scr) != sorted(ref))
                    cov[(st, spec)]["scratch_consistency"][1] += 1
        if starts:
            span[st].append((max(ends) - min(starts)).total_seconds())

    n_sched = len(sched_all)
    jidx: dict[tuple, list] = defaultdict(list)
    stock_sessions = []
    for e in events:
        if "_unreadable" in e:
            continue
        if e.get("process") == "jvlink_obs" and e.get("spec") == "RACE":
            stock_sessions.append(e)
        elif e.get("stage"):
            jidx[(canonical_stage(e.get("stage")), e.get("race_id"), e.get("spec"))].append(e)

    # タスクの開始・終了記録（F7）
    exp_tasks = expected_tasks({d: schedule.get(d, {}) for d in dates}, scope)
    runs_by_date = {d: TR.read_runs(d, dry=dry, root=task_runs_root) for d in dates}
    task_runs = {d: TR.summarize(runs_by_date[d], exp_tasks[d]) for d in dates}
    task_reasons = []
    for d in dates:
        s = task_runs[d]
        s["required"] = d >= task_runs_required_from
        if not s["required"]:
            continue
        if not s["recorded"]:
            task_reasons.append(f"{d}: no task run records (obs_task_runs)")
            continue
        if s["missing_start"] or s["missing_end"]:
            task_reasons.append(f"{d}: task start/end records not 100% "
                                f"(started {s['started']}/{s['expected']}, ended {s['ended']}/{s['expected']})")
        if s["ctrl_c_exit"]:
            task_reasons.append(f"{d}: 0xC000013A recurred under windowless launch x{len(s['ctrl_c_exit'])}")
        elif s["abnormal_exit"]:
            task_reasons.append(f"{d}: tasks ended with non-zero exit x{len(s['abnormal_exit'])}")

    def task_state(st: str, rid: str) -> str | None:
        t = STAGE_TASK.get(st)
        if t is None:
            return None
        d = rid[:8]
        return TR.run_state(runs_by_date.get(d), (t, rid if t in ("t2", "trio") else d))

    threshold = missing_threshold(n_sched)
    missing, missing_fail, unattributed = {}, [], []
    cause_summary: Counter = Counter()
    for st, specs in required.items():
        for spec in specs:
            got = have[(st, spec)] & set(sched_all)
            races = []
            for rid in sorted(set(sched_all) - got):
                cause, detail = attribute_missing(st, spec, rid, cap_by, rec_by, jidx, stock_sessions, task_state)
                races.append({"race_id": rid, "cause": cause, "detail": detail})
                cause_summary[cause] += 1
                if cause == "unattributed":
                    unattributed.append(f"{st}/{spec}/{rid}")
            n_miss = len(races)
            over = (n_sched == 0) or (n_miss > threshold)
            missing[f"{st}/{spec}"] = {"scheduled": n_sched, "with_capture": len(got), "missing_count": n_miss,
                                       "threshold": threshold, "over_threshold": over,
                                       "missing_rate": round(n_miss / n_sched, 4) if n_sched else None,
                                       "races": races, "required": True}
            if over:
                missing_fail.append(f"{st}/{spec} ({n_miss} > {threshold:g})" if n_sched else f"{st}/{spec} (no schedule)")
    for st, specs in {**SCOPES["full"], **{s: PROD4 for s in DESCRIPTIVE_STAGES}}.items():
        if st in required:
            continue
        for spec in specs:
            got = have[(st, spec)] & set(sched_all)
            if got or st in SCOPES["full"]:
                missing[f"{st}/{spec}"] = {"scheduled": n_sched, "with_capture": len(got),
                                           "missing_count": n_sched - len(got), "required": False}
    for st, rids in v1_only.items():
        missing[f"{st}/v1_without_raw"] = {"races": len(rids), "required": False}
    coverage = {f"{st}/{spec}": {b: {"mismatch": m, "n": n, "rate": round(m / n, 4) if n else None}
                                 for b, (m, n) in blocks.items()}
                for (st, spec), blocks in cov.items()}
    over_1pct = sorted(f"{k}/{b}" for k, blocks in coverage.items() for b, v in blocks.items()
                       if v["rate"] is not None and v["rate"] > MAX_COVERAGE_MISMATCH)
    coverage_fail = [x for x in over_1pct if x.split("/")[0] in required]
    coverage_fail_descriptive = [x for x in over_1pct if x.split("/")[0] not in required]
    parse_fail = sorted(k for k, (f, n) in parse_checks.items() if f > 0)
    conc = concurrency(events, success_index, "intersection")
    conc30 = concurrency(events, success_index, "window30")

    contract_reasons = []
    if missing_fail:
        contract_reasons.append(f"required stage/spec missing count > max(1%, 1 race): {missing_fail}")
    if unattributed:
        contract_reasons.append(f"missing races without an attributed cause: {unattributed[:20]}")
    if coverage_fail:
        contract_reasons.append(f"coverage mismatch > {MAX_COVERAGE_MISMATCH:.0%}: {coverage_fail}")
    if parse_fail:
        contract_reasons.append(f"raw/parser check failures: {parse_fail}")
    if conc["unidentified_events"]:
        contract_reasons.append(f"journal events without an identifiable process: {conc['unidentified_events']}")
    contract_reasons += task_reasons
    queue_reasons = []
    ov = conc["overlapped"]
    if ov["n"] and ov["ok"] / ov["n"] < MIN_OVERLAP_SUCCESS:
        queue_reasons.append(f"overlapped success {ov['ok']}/{ov['n']} < {MIN_OVERLAP_SUCCESS} (interval intersection)")
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
        "audit_version": AUDIT_VERSION, "structure_version": JR.STRUCTURE_VERSION,
        "dates": dates, "scope": scope, "required_stages": sorted(required),
        "stage_sources": {k: sorted(v) for k, v in sorted(stage_sources.items())},
        "scheduled_races": n_sched, "records": len(recs), "journal_events": len(events),
        "missing_threshold_races": threshold, "missing_cause_summary": dict(cause_summary),
        "missing": missing,
        "timing": {"fetch_duration_sec": {f"{k[0]}/{k[1]}": _q(v) for k, v in dur.items()},
                   "t2_candidate_5spec_span_sec": five_spec,
                   "t2_5spec_p95_over_60s": bool(five_spec.get("p95", 0) > FIVE_SPEC_P95_SEC),
                   "announce_delay_sec": {f"{k[0]}/{k[1]}": _q(v) for k, v in ann_delay.items()},
                   "seconds_to_post": {f"{k[0]}/{k[1]}": {b: _q(v) for b, v in d.items()}
                                       for k, d in to_post.items()}},
        "coverage": coverage, "coverage_fail_over_1pct": coverage_fail,
        "coverage_fail_over_1pct_out_of_scope": coverage_fail_descriptive,
        "raw_parser": {k: {"fail": f, "n": n} for k, (f, n) in parse_checks.items()},
        "restructure_version": {"current": JR.STRUCTURE_VERSION,
                                "older_records_not_compared": dict(version_skips),
                                "older_records_that_would_differ": dict(version_would_differ)},
        "task_runs": task_runs,
        "concurrency": conc, "concurrency_window30_descriptive": conc30, "corruption_0000": corruption,
        "contract": {"met": not contract_reasons, "reasons": contract_reasons},
        "concurrency_verdict": concurrency_verdict, "concurrency_reasons": queue_reasons,
        "decision": decision, "exit_code": DECISION_EXIT[decision],
        "stage0_passed": decision == STAGE0_PASS,
        "collection_continues": True,
        "performance_blocked_by_stage0": decision != STAGE0_PASS,
        "stage0_restart": (None if decision == STAGE0_PASS else
                           "修正を入れた後の開催日から Stage 0 の 2 開催日を数え直す（それまで性能・ROI・帯選択へ進まない）"),
        "files_opened_outcome_like": result_like,
    }


def _git_head() -> str | None:
    try:
        import subprocess
        return subprocess.run(["git", "-c", "safe.directory=*", "rev-parse", "HEAD"], cwd=BASE,
                              capture_output=True, text=True).stdout.strip() or None
    except Exception:
        return None


def record_ledger(rep: dict, ledger: Path, *, dry: bool, report_path: str, run_id: str | None = None,
                  rejudge_of: str | None = None, rerun_of: str | None = None) -> dict:
    """Stage 0 判定を追記専用 ledger へ 1 行書く（性能 guard が読む）。"""
    row = {"generated_at": datetime.now().astimezone().isoformat(timespec="seconds"), "dates": rep["dates"],
           "dry": bool(dry), "scope": rep.get("scope"), "decision": rep["decision"], "exit_code": rep["exit_code"],
           "contract_met": rep["contract"]["met"], "reasons": rep["contract"]["reasons"] + rep["concurrency_reasons"],
           "missing_cause_summary": rep.get("missing_cause_summary"), "code_head": _git_head(),
           "audit_version": rep.get("audit_version"), "structure_version": rep.get("structure_version"),
           "report": report_path, "run_id": run_id or uuid.uuid4().hex,
           "rejudge_of": rejudge_of, "rerun_of": rerun_of}
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def acquire_lock(path: Path = LOCK_PATH, stale_sec: float = LOCK_STALE_SEC) -> tuple[bool, str]:
    """監査の二重起動防止。排他作成で lock を取る。stale_sec より古い lock は 1 回だけ取り替える。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(2):
        try:
            fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            os.write(fd, f"{os.getpid()} {datetime.now().isoformat()}".encode())
            os.close(fd)
            return True, "acquired" if attempt == 0 else "acquired (stale lock replaced)"
        except FileExistsError:
            try:
                age = time.time() - path.stat().st_mtime
            except FileNotFoundError:
                continue
            if age < stale_sec or attempt:
                return False, f"lock held ({path}, age {age:.0f}s)"
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    return False, "lock not acquired"


def release_lock(path: Path = LOCK_PATH) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def recent_duplicate(ledger: Path, dates: list[str], dry: bool, scope: str, report_stem: str,
                     window_sec: float = DUPLICATE_WINDOW_SEC) -> dict | None:
    """同じ (dates, dry, scope, code_head, report 名) の行が window_sec 以内にあれば、その行を返す。"""
    if not ledger.exists():
        return None
    head = _git_head()
    now = datetime.now().astimezone()
    for line in reversed(ledger.read_text(encoding="utf-8").splitlines()):
        try:
            r = json.loads(line)
        except Exception:
            continue
        t = _ts(r.get("generated_at"))
        if t is None or (now - t) > timedelta(seconds=window_sec):
            continue
        if (r.get("dates") == dates and bool(r.get("dry")) == bool(dry) and r.get("scope", "full") == scope
                and r.get("code_head") == head and Path(str(r.get("report"))).stem.startswith(report_stem)):
            return r
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="+", required=True)
    ap.add_argument("--dry", action="store_true",
                    help="観測 collector の stage は *_dry、本番経路の stage は本番の store から読む")
    ap.add_argument("--scope", choices=sorted(SCOPES), default="night")
    ap.add_argument("--out", default=None)
    ap.add_argument("--rejudge-of", default=None, help="既存記録の再判定のとき、元の判定を示すラベル")
    ap.add_argument("--allow-rerun", action="store_true",
                    help="同じ条件の直近の実行があっても実行する（ledger に rerun_of を残す）")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    suffix = ("_dry" if args.dry else "") + ("" if args.scope == "full" else f"_{args.scope}")
    if args.rejudge_of:
        suffix += "_rejudge"
    out = Path(args.out) if args.out else BASE / "reports" / f"obs_stage0_{'_'.join(args.dates)}{suffix}.json"
    ok, why = acquire_lock()
    if not ok:
        print(f"[obs_stage0] ALREADY_RUNNING: {why}. nothing written")
        return EXIT_ALREADY_RUNNING
    try:
        dup = recent_duplicate(STAGE0_LEDGER, args.dates, args.dry, args.scope, out.stem)
        if dup and not args.allow_rerun:
            print(f"[obs_stage0] ALREADY_RUNNING: identical audit recorded at {dup.get('generated_at')} "
                  f"(run_id {dup.get('run_id')}). nothing written (use --allow-rerun to record a rerun)")
            return EXIT_ALREADY_RUNNING
        _track_opens()
        rep = audit(args.dates, sources=store_sources(args.dry), scope=args.scope, dry=args.dry)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():                                   # 既存の report は上書きしない
            out = out.with_name(f"{out.stem}_{datetime.now().strftime('%Y%m%dT%H%M%S')}{out.suffix}")
        out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        row = record_ledger(rep, STAGE0_LEDGER, dry=args.dry, report_path=str(out), rejudge_of=args.rejudge_of,
                            rerun_of=(dup or {}).get("run_id"))
    finally:
        release_lock()
    print(json.dumps({k: rep[k] for k in ("dates", "scope", "scheduled_races", "records", "decision", "exit_code",
                                          "contract", "concurrency_verdict", "concurrency_reasons",
                                          "missing_cause_summary", "stage0_restart")}, ensure_ascii=False))
    for key, m in rep["missing"].items():
        for r in m.get("races", []):
            print(f"[obs_stage0] missing {key} {r['race_id']} cause={r['cause']} ({r['detail']})")
    print(f"[obs_stage0] -> {out}  (ledger: {STAGE0_LEDGER}, run_id {row['run_id']})")
    return rep["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
