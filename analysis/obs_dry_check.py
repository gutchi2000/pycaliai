# -*- coding: utf-8 -*-
"""
obs_dry_check.py — 観測計画 v2.1 §5.1 の三連複 collector 登録開始条件（-Dry 1 開催日）の検査
=========================================================================================
Dry 開催日について、予定の全レースで次が揃っているかを見る（label-free、結果は読まない）。
  - reports/trio_portfolio_shadow_v2/raw/_dry/{date}/{rid}/ に 0B35 と同時点の 0B31 の raw
  - manifest.jsonl の raw 行に sha256、snapshot 行に発表月日時分（announce_dt）
  - forward_prices_dry に trio_t10（raw 付き）と t2_candidate の録

python -m analysis.obs_dry_check --date 20261003
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from forward_prices import FORWARD_ROOT, canonical_stage, read_snapshot  # noqa: E402

TRIO_DIR = BASE / "reports" / "trio_portfolio_shadow_v2"


def check(date: str, rids: list[str], trio_dir: Path = TRIO_DIR,
          fwd_dry: Path = FORWARD_ROOT.parent / "forward_prices_dry") -> dict:
    raw_root = trio_dir / "raw" / "_dry" / date
    man = []
    mp = trio_dir / "manifest.jsonl"
    if mp.exists():
        for line in mp.read_text(encoding="utf-8").splitlines():
            try:
                man.append(json.loads(line))
            except Exception:
                pass
    stages = {}
    for p in (fwd_dry / date).glob("*.json.gz") if (fwd_dry / date).is_dir() else []:
        rec = read_snapshot(p)
        stages.setdefault(rec.get("race_id"), set()).add(canonical_stage(rec.get("stage")))
    rows, fails = [], []
    for rid in rids:
        files = sorted(p.name for p in (raw_root / rid).glob("*.txt")) if (raw_root / rid).is_dir() else []
        raw_rows = [m for m in man if m.get("kind") == "raw" and m.get("race_id") == rid
                    and f"/_dry/{date}/" in str(m.get("path", "")).replace("\\", "/")]
        snap_rows = [m for m in man if m.get("kind") == "snapshot" and m.get("race_id") == rid
                     and f"/_dry/{date}/" in str(m.get("path", "")).replace("\\", "/")]
        r = {"race_id": rid,
             "raw_0B35": any(f.endswith("_0B35.txt") for f in files),
             "raw_0B31": any(f.endswith("_0B31.txt") for f in files),
             "manifest_sha256_all": bool(raw_rows) and all(m.get("sha256") for m in raw_rows),
             "manifest_announce_all": bool(snap_rows) and all(m.get("announce_dt") for m in snap_rows),
             "forward_trio_t10": "trio_t10" in stages.get(rid, set()),
             "forward_t2_candidate": "t2_candidate" in stages.get(rid, set())}
        r["ok"] = all(v for k, v in r.items() if k != "race_id")
        rows.append(r)
        if not r["ok"]:
            fails.append(r)
    return {"date": date, "races": len(rids), "ok": bool(rids) and not fails, "failures": fails}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    args = ap.parse_args()
    from jvlink_trio_odds import build_schedule
    rids = [rid for _, rid in build_schedule(args.date)]
    rep = check(args.date, rids)
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
