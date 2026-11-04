# -*- coding: utf-8 -*-
"""
scripts/kaiko_warm.py — 回顧記事・見どころの事前生成（Phase A / Phase C の末尾から best-effort で呼ばれる）
================================================================================
鍵はローカルに置かず、pycaliai.com の Worker（worker/kaiko.js）に生成させる。
  GET  https://pycaliai.com/api/kaiko?date=YYYYMMDD  → missing（未生成の race_id / "day" / "preview"）
  POST https://pycaliai.com/api/kaiko/gen {date, race_id}  → 1 件生成して KV に保存
sync-hf-umami.ps1（wrangler deploy）で当日の site/data/{date}.json が本番に乗った後に呼ぶこと。

使い方:
    venv311\\Scripts\\python.exe scripts\\kaiko_warm.py [YYYYMMDD ...] [--discord]
    引数なし: site/data/manifest.json の開催日のうち、結果待ちの日（見どころ）と結果ありの直近 2 日（回顧）。
    --discord: この実行で新しく生成した総括・見どころを Discord（notify_config.json の webhook）へ投稿する。
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://pycaliai.com"
ORIGIN = "https://pycaliai.com"
NOTIFY_CFG = ROOT / "notify_config.json"   # t10_runner.py と共用（gitignore、webhook URL は秘匿）
UA = "PyCaLiAI-kaiko-warm/1.0 (+https://pycaliai.com)"   # Cloudflare は既定の Python-urllib UA を 403 で弾く


def _req(method: str, path: str, body: dict | None = None, timeout: int = 150):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"content-type": "application/json", "origin": ORIGIN, "user-agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8"))
        except Exception:
            return e.code, {"error": str(e)}


def _webhook_url() -> str:
    url = os.environ.get("PYCALIAI_DISCORD_WEBHOOK", "").strip()
    if url:
        return url
    try:
        return str(json.loads(NOTIFY_CFG.read_text(encoding="utf-8")).get("discord_webhook", "") or "").strip()
    except (OSError, ValueError):
        return ""


def discord(text: str) -> bool:
    """Discord webhook へ送信（2000 字上限）。未設定/失敗は非致命。"""
    url = _webhook_url()
    if not url:
        print("  [discord] webhook 未設定 (notify_config.json)")
        return False
    body = json.dumps({"content": text[:1990]}).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", "User-Agent": "PyCaLiAI-kaiko (private bot)"})
    try:
        urllib.request.urlopen(req, timeout=10)
        return True
    except Exception as e:
        print(f"  [discord] 送信失敗 (非致命): {e}")
        return False


def _fmt(d8: str) -> str:
    return f"{int(d8[4:6])}/{int(d8[6:8])}"


def warm(date: str, created: list) -> tuple[int, int]:
    st, lst = _req("GET", f"/api/kaiko?date={date}")
    if st != 200:
        print(f"  {date}: list {st} {lst.get('error')}")
        return 0, 1
    missing = lst.get("missing", [])
    order = [x for x in missing if x not in ("day", "preview")] + [x for x in ("day", "preview") if x in missing]
    print(f"  {date}: {'結果待ち' if lst.get('upcoming') else str(lst.get('n_races')) + 'R'}, 未生成 {len(order)} 件")
    made = failed = 0
    for rid in order:
        st, j = _req("POST", "/api/kaiko/gen", {"date": date, "race_id": rid})
        if st == 200 and j.get("stored") is False:
            failed += 1
            print(f"    NG  {rid} 生成はできたが KV 保存に失敗（Worker ログを確認）")
        elif st == 200:
            made += 1
            it = j["item"]
            print(f"    ok  {rid} ({it.get('model')}, in {it['usage']['input_tokens']} / out {it['usage']['output_tokens']})")
            if j.get("created") and rid in ("day", "preview"):
                created.append((date, rid, it))
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
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    want_discord = "--discord" in sys.argv[1:]
    mf = json.loads((ROOT / "site" / "data" / "manifest.json").read_text(encoding="utf-8"))
    if args:
        dates = args
    else:
        dates = [d["date"] for d in mf.get("dates", []) if not d.get("has_results")][:2] + \
                [d["date"] for d in mf.get("dates", []) if d.get("has_results")][:2]
    print(f"[kaiko_warm] {dates}{' +discord' if want_discord else ''}")
    tot_made = tot_failed = 0
    created: list = []
    for d in dates:
        m, f = warm(d, created)
        tot_made += m
        tot_failed += f
    print(f"[kaiko_warm] made={tot_made} failed={tot_failed}")
    if want_discord:
        for date, rid, it in created:
            head = f"**{_fmt(date)} {'回顧' if rid == 'day' else '今日の見どころ'}**"
            x = it.get("x_post") or ""
            msg = f"{head}\n{it.get('text', '')}\n\n> X 下書き: {x}\n{BASE}/kaiko#{date}"
            ok = discord(msg)
            print(f"  [discord] {date} {rid}: {'sent' if ok else 'skip'}")
    return 0 if tot_failed == 0 else 2


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sys.exit(main())
