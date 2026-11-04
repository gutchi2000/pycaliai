# -*- coding: utf-8 -*-
"""
scripts/kaiko_warm.py — 回顧記事の事前生成（Phase C の末尾から best-effort で呼ばれる）
================================================================================
鍵はローカルに置かず、pycaliai.com の Worker（worker/kaiko.js）に生成させる。
  GET  https://pycaliai.com/api/kaiko?date=YYYYMMDD  → missing（未生成の race_id）
  POST https://pycaliai.com/api/kaiko/gen {date, race_id}  → 1 件生成して KV に保存
sync-hf-umami.ps1（wrangler deploy）で当日の site/data/{date}.json が本番に乗った後に呼ぶこと。

使い方:
    venv311\\Scripts\\python.exe scripts\\kaiko_warm.py [YYYYMMDD ...]
    引数なし: site/data/manifest.json の結果あり開催日のうち直近 2 日。
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://pycaliai.com"
ORIGIN = "https://pycaliai.com"


def _req(method: str, path: str, body: dict | None = None, timeout: int = 120):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    # Cloudflare は既定の Python-urllib UA を 403 で弾くので、識別できる UA を名乗る
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"content-type": "application/json", "origin": ORIGIN,
                                          "user-agent": "PyCaLiAI-kaiko-warm/1.0 (+https://pycaliai.com)"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8"))
        except Exception:
            return e.code, {"error": str(e)}


def warm(date: str) -> tuple[int, int]:
    st, lst = _req("GET", f"/api/kaiko?date={date}")
    if st != 200:
        print(f"  {date}: list {st} {lst.get('error')}")
        return 0, 1
    missing = [x for x in lst.get("missing", []) if x != "day"] + (["day"] if "day" in lst.get("missing", []) else [])
    print(f"  {date}: {lst.get('n_races')}R, 未生成 {len(missing)} 件")
    made = failed = 0
    for rid in missing:
        st, j = _req("POST", "/api/kaiko/gen", {"date": date, "race_id": rid})
        if st == 200 and j.get("stored") is False:
            failed += 1
            print(f"    NG  {rid} 生成はできたが KV 保存に失敗（Worker ログを確認）")
        elif st == 200:
            made += 1
            print(f"    ok  {rid} ({j['item'].get('model')}, in {j['item']['usage']['input_tokens']} / out {j['item']['usage']['output_tokens']})")
        elif st == 409:
            time.sleep(5)
            failed += 1
        else:
            failed += 1
            print(f"    NG  {rid} {st} {j.get('error')} ({j.get('code')})")
            if j.get("code") in ("daily_limit", "month_limit", "no_key"):
                break
    return made, failed


def main() -> int:
    dates = sys.argv[1:]
    if not dates:
        mf = json.loads((ROOT / "site" / "data" / "manifest.json").read_text(encoding="utf-8"))
        dates = [d["date"] for d in mf.get("dates", []) if d.get("has_results")][:2]
    print(f"[kaiko_warm] {dates}")
    tot_made = tot_failed = 0
    for d in dates:
        m, f = warm(d)
        tot_made += m
        tot_failed += f
    print(f"[kaiko_warm] made={tot_made} failed={tot_failed}")
    return 0 if tot_failed == 0 else 2


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sys.exit(main())
