# -*- coding: utf-8 -*-
"""
obs_task_runs.py — 観測タスクの開始・終了の追記専用記録（観測計画 v2.1 Dry 修正 F7）
====================================================================================
観測タスク（T−2・三連複 T−10・FINAL・蓄積系・スケジューラ）を起動するたびに、窓なし起動器
（obs_task.py）が「開始」を 1 件、終わったら「終了」を 1 件書く。1 件 1 ファイルで、既存ファイルは
上書きしない（排他作成）。開始があり終了が無い run は、起動器ごと強制終了されたものとして検出する
（analysis/obs_stage0_audit.py の task_killed）。

    data/obs_task_runs[_dry]/{date}/{開始時刻}_{run_id}_start.json
    data/obs_task_runs[_dry]/{date}/{開始時刻}_{run_id}_end.json

標準ライブラリのみ（32-bit / 64-bit のどちらからも import 可）。書込は fail-open にしない:
開始記録を書けなければ起動器はタスクを実行しない（記録の無い取得を作らない）。
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
RUNS_ROOT = BASE / "data" / "obs_task_runs"
DRY_RUNS_ROOT = BASE / "data" / "obs_task_runs_dry"
CTRL_C_EXIT = 0xC000013A          # STATUS_CONTROL_C_EXIT（console に Ctrl+C / close が届いた終了）
TASK_KINDS = ("t2", "trio", "final", "stock", "schedule", "probe", "conc_test")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def runs_root(dry: bool) -> Path:
    env = os.environ.get("PYCALIAI_OBS_TASK_RUNS_ROOT")
    if env:
        return Path(env)
    return DRY_RUNS_ROOT if dry else RUNS_ROOT


def _write_exclusive(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0))
    try:
        os.write(fd, json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    return path


def _stem(run: dict) -> str:
    started = "".join(ch for ch in str(run["started_at"])[:23] if ch.isdigit())
    return f"{started}_{run['run_id']}"


def write_start(run: dict, *, dry: bool, root: Path | None = None) -> Path:
    """run = {run_id, task, date, race_id?, task_name?, argv, launcher_pid, started_at, ...}。"""
    assert run.get("task") in TASK_KINDS, run.get("task")
    base = (root or runs_root(dry)) / str(run["date"])
    return _write_exclusive(base / f"{_stem(run)}_start.json", {"event": "start", **run})


def write_end(run: dict, *, exit_code: int | None, dry: bool, root: Path | None = None, **extra) -> Path:
    base = (root or runs_root(dry)) / str(run["date"])
    payload = {"event": "end", "run_id": run["run_id"], "task": run["task"], "date": run["date"],
               "race_id": run.get("race_id"), "started_at": run["started_at"], "finished_at": now_iso(),
               "exit_code": exit_code, **extra}
    return _write_exclusive(base / f"{_stem(run)}_end.json", payload)


def read_runs(date: str, *, dry: bool, root: Path | None = None) -> dict | None:
    """{run_id: {"start": {...}|None, "end": {...}|None}}。日付フォルダが無ければ None（記録なし）。"""
    d = (root or runs_root(dry)) / str(date)
    if not d.is_dir():
        return None
    runs: dict[str, dict] = {}
    for p in sorted(d.glob("*.json")):
        try:
            ev = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            ev = {"event": "unreadable", "run_id": p.stem, "path": str(p)}
        r = runs.setdefault(str(ev.get("run_id")), {"start": None, "end": None, "unreadable": []})
        if ev.get("event") in ("start", "end"):
            r[ev["event"]] = ev
        else:
            r["unreadable"].append(str(p))
    return runs


def task_key(ev: dict) -> tuple[str, str]:
    """(task, race_id) — race 単位でないタスク（final・stock・schedule）は race_id の代わりに日付。"""
    return str(ev.get("task")), str(ev.get("race_id") or ev.get("date"))


def summarize(runs: dict | None, expected: set[tuple[str, str]]) -> dict:
    """期待タスクに対する開始・終了の記録状況。runs が None（記録フォルダが無い）なら recorded=False。"""
    if runs is None:
        return {"recorded": False, "expected": len(expected)}
    by_key: dict[tuple, list] = {}
    killed, abnormal, ctrl_c, unreadable = [], [], [], []
    for rid, r in runs.items():
        unreadable += r.get("unreadable", [])
        ev = r["start"] or r["end"]
        if ev is None:
            continue
        by_key.setdefault(task_key(ev), []).append(r)
        if r["start"] and not r["end"]:
            killed.append({"run_id": rid, "task": ev.get("task"), "race_id": ev.get("race_id"),
                           "started_at": r["start"].get("started_at")})
        if r["end"] and r["end"].get("exit_code") not in (0,):
            code = r["end"].get("exit_code")
            item = {"run_id": rid, "task": ev.get("task"), "race_id": ev.get("race_id"), "exit_code": code,
                    "child_exit_codes": r["end"].get("child_exit_codes")}
            abnormal.append(item)
            codes = [code] + list(r["end"].get("child_exit_codes") or [])
            if any(c is not None and (int(c) & 0xFFFFFFFF) == CTRL_C_EXIT for c in codes):
                ctrl_c.append(item)
    started = {k for k, rs in by_key.items() if any(r["start"] for r in rs)}
    ended = {k for k, rs in by_key.items() if any(r["start"] and r["end"] for r in rs)}
    return {"recorded": True, "expected": len(expected),
            "started": len(expected & started), "ended": len(expected & ended),
            "missing_start": sorted("/".join(k) for k in expected - started),
            "missing_end": sorted("/".join(k) for k in (expected & started) - ended),
            "killed": killed, "abnormal_exit": abnormal, "ctrl_c_exit": ctrl_c,
            "unreadable": unreadable, "unexpected_tasks": sorted("/".join(k) for k in set(by_key) - expected)}


def run_state(runs: dict | None, key: tuple[str, str]) -> str | None:
    """欠損帰属用: そのタスクの最新 run が 'killed'（開始のみ）/ 'abnormal'（終了コード≠0）/ 'ok' / None。"""
    if not runs:
        return None
    rs = [r for r in runs.values() if (r["start"] or r["end"]) and task_key(r["start"] or r["end"]) == key]
    if not rs:
        return None
    last = max(rs, key=lambda r: str((r["start"] or r["end"]).get("started_at")))
    if last["start"] and not last["end"]:
        return "killed"
    if last["end"] and last["end"].get("exit_code") != 0:
        return "abnormal"
    return "ok"
