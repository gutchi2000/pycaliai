"""Append-only JV-Link 0B11 bodyweight collector.

This records only information available at collection time. It never loads
results, payouts, odds, predictions, or bets, and it does not feed production.
Run with 32-bit Python because JV-Link is a 32-bit COM component.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORE = ROOT / "data" / "forward_bodyweight"
CALENDAR_DIR = ROOT / "data" / "_research" / "mcond" / "exp05fs_calendar"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jvlink_changes import _wh_bytes, parse_wh_detail  # noqa: E402
from jvlink_odds import fetch_records  # noqa: E402


def _atomic_create(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with open(tmp, "xb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    try:
        os.rename(tmp, path)
    except FileExistsError:
        pass
    finally:
        tmp.unlink(missing_ok=True)


def _record_bytes(rec: str) -> bytes:
    return _wh_bytes(rec)


def collect_race(race_id: str, *, store: Path = DEFAULT_STORE,
                 now: datetime | None = None,
                 fetcher: Callable[[str, str], list[str]] = fetch_records) -> dict:
    now = now or datetime.now().astimezone()
    observed_at = now.isoformat(timespec="milliseconds")
    recs = [r for r in fetcher(race_id, "0B11")
            if r.startswith("WH") and r[11:27] == race_id]
    attempt_id = f"{now.strftime('%H%M%S_%f')}_{race_id}"
    attempt = {"schema_version": 1, "race_id": race_id,
               "observed_at": observed_at, "source": "JVLink/JVRTOpen/0B11",
               "success": bool(recs), "record_count": len(recs)}
    if not recs:
        attempt["reason"] = "no_WH_record"
        _atomic_create(store / "attempts" / race_id[:8] / f"{attempt_id}.json", attempt)
        return attempt

    rec = recs[-1]
    raw = _record_bytes(rec)
    raw_sha = hashlib.sha256(raw).hexdigest()
    rows = parse_wh_detail(rec)
    snapshot = {
        "schema_version": 1, "race_id": race_id, "observed_at": observed_at,
        "source": "JVLink/JVRTOpen/0B11", "raw_cp932_sha256": raw_sha,
        "raw_record_cp932_hex": raw.hex(), "horses": rows,
        "normal_weight_count": sum(r["weight_status"] == "normal" for r in rows),
        "special_weight_count": sum(r["weight_status"] != "normal" for r in rows),
    }
    snap_path = store / "snapshots" / race_id[:8] / race_id / f"{raw_sha}.json"
    existed = snap_path.exists()
    _atomic_create(snap_path, snapshot)
    attempt.update({"raw_cp932_sha256": raw_sha,
                    "snapshot": str(snap_path.relative_to(store)).replace("\\", "/"),
                    "new_snapshot": not existed, "horse_count": len(rows)})
    _atomic_create(store / "attempts" / race_id[:8] / f"{attempt_id}.json", attempt)
    return attempt


def load_calendar(date: str, path: Path | None = None) -> list[tuple[str, datetime]]:
    path = path or CALENDAR_DIR / f"{date}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("target_date") != date or not data.get("sanity_check_result", {}).get("passed"):
        raise ValueError("calendar identity/sanity check failed")
    ids, times = data.get("race_ids", []), data.get("post_times", [])
    if len(ids) != len(times) or len(set(ids)) != len(ids):
        raise ValueError("calendar race/time cardinality invalid")
    out = []
    for rid, hhmm in zip(ids, times):
        post = datetime.strptime(f"{date} {hhmm}", "%Y%m%d %H:%M").astimezone()
        out.append((rid, post))
    return out


def due_races(calendar: list[tuple[str, datetime]], now: datetime,
              lead_start: int = 90, lead_end: int = 5):
    return [(rid, post) for rid, post in calendar
            if post - timedelta(minutes=lead_start) <= now <= post - timedelta(minutes=lead_end)]


def poll(date: str, *, store: Path = DEFAULT_STORE, now: datetime | None = None) -> dict:
    now = now or datetime.now().astimezone()
    due = due_races(load_calendar(date), now)
    results = [collect_race(rid, store=store, now=now) for rid, _ in due]
    return {"date": date, "observed_at": now.isoformat(timespec="seconds"),
            "due_races": len(due), "successes": sum(r["success"] for r in results),
            "new_snapshots": sum(r.get("new_snapshot", False) for r in results),
            "results": results}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    ap.add_argument("--race")
    ap.add_argument("--store", type=Path, default=DEFAULT_STORE)
    args = ap.parse_args()
    try:
        result = collect_race(args.race, store=args.store) if args.race else poll(args.date, store=args.store)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except FileNotFoundError:
        print(json.dumps({"date": args.date, "status": "no_verified_calendar"}, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({"date": args.date, "status": "error", "error": str(exc)},
                         ensure_ascii=False), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
