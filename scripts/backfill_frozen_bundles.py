# -*- coding: utf-8 -*-
"""
backfill_frozen_bundles.py — 過去の bundle を発走前の版で凍結する (2026-11-13 監査 P1-5)

各開催日について、reports/cowork_input/{date}_bundle.json の git 履歴から
「当日 09:30 (第 1R より前) より前の最後のコミット」を取り出し、
reports/cowork_input_frozen/ に書く。sidecar に出所 (commit) と live との一致を記録する。
発走前のコミットが無い日 (初回コミットが発走後) は凍結しない = サイトでは「未検証」になる。
既に凍結版がある日は触らない (--force で作り直し)。

実行: python scripts/backfill_frozen_bundles.py [--root E:/PyCaLiAI] [--dry] [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path


def git(root: Path, *a) -> bytes:
    return subprocess.run(["git", "-c", "safe.directory=*", "-C", str(root), *a],
                          capture_output=True, check=False).stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    live_dir = root / "reports" / "cowork_input"
    fdir = root / "reports" / "cowork_input_frozen"
    rows = []
    for bp in sorted(live_dir.glob("*_bundle.json")):
        d = bp.name[:8]
        dst = fdir / f"{d}_bundle.json"
        if dst.exists() and not args.force:
            rows.append({"date": d, "action": "exists"})
            continue
        rel = f"reports/cowork_input/{d}_bundle.json"
        until = f"{d[:4]}-{d[4:6]}-{d[6:]} 09:30"
        sha = git(root, "log", "-1", "--format=%H %cI", f"--until={until}", "--", rel).decode().strip()
        if not sha:
            rows.append({"date": d, "action": "no_pre_race_commit"})
            continue
        commit, cdate = sha.split()
        raw = git(root, "show", f"{commit}:{rel}")
        if not raw:
            rows.append({"date": d, "action": "git_show_failed", "commit": commit})
            continue
        same = raw == bp.read_bytes()
        meta = {"date": d, "frozen_at": datetime.now().isoformat(timespec="seconds"),
                "first_post": f"{d[:4]}-{d[4:6]}-{d[6:]}T09:30", "sha256": hashlib.sha256(raw).hexdigest(),
                "source": f"git:{commit} ({cdate})", "identical_to_live_at_backfill": same,
                "verified_pre_race": True}
        rows.append({"date": d, "action": "frozen", "commit": commit[:10], "identical_to_live": same})
        if not args.dry:
            fdir.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(raw)
            (fdir / f"{d}_bundle.provenance.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                                              encoding="utf-8")
    n = {k: sum(r["action"] == k for r in rows) for k in ("frozen", "exists", "no_pre_race_commit", "git_show_failed")}
    print(json.dumps(n, ensure_ascii=False))
    for r in rows:
        if r["action"] != "exists" and not (r["action"] == "frozen" and r.get("identical_to_live")):
            print(json.dumps(r, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
