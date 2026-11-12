# -*- coding: utf-8 -*-
"""
am9_snapshot.py — 開催日の朝 9 時に全レースの単勝・複勝オッズを 1 回だけ保存する (CW-14, 2026-11-13)
=====================================================================================
目的: 「朝 9 時に売れていて発走前に売れなくなった馬は過小評価される」(docs/research/POSTCOVID_SEARCH_20261113.md) の前向き確認。
過去 13 年の検証は TANPUK 時系列の「当日 9 時前後」のスナップを使っており、2026 年は同じ時点の記録が無かった。

やること: data/weekly/{date}.csv のレース一覧を読み、1 レースずつ 32-bit の jvlink_odds.py を stage=am9 で呼ぶ。
  - 出力は reports/am9_odds/{date}/{rid}.json と data/forward_prices/{date}/*_am9_*.json.gz (schema v2、票数計つき)。
  - 本番の reports/live_odds・買い目・投票には一切書かない (--out-dir を分けている)。読むだけの収集。
  - 失敗したレースは 1 回だけ再試行。全滅または 2 割以上失敗なら Discord (私用) に 1 通。正常時は無言。
実行: タスク PyCaLiAI_AM9 (土日 09:00、am9_register.ps1 で登録)。手動: python scripts/am9_snapshot.py [--date YYYYMMDD] [--dry]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY32 = ["py", "-3.12-32"]
NON_DIGIT = re.compile(r"[^0-9]")


def race_ids(date: str) -> list[str]:
    """出走表 (前夜に置かれる) → 無ければ JV-Link の開催カレンダー (毎朝 08:20 に t35_shadow が作る) の順で探す。"""
    p = ROOT / "data" / "weekly" / f"{date}.csv"
    if not p.exists():
        cal = ROOT / "data" / "_research" / "mcond" / "exp05fs_calendar" / f"{date}.json"
        try:
            ids = json.loads(cal.read_text(encoding="utf-8")).get("race_ids") or []
            return sorted({str(r)[:16] for r in ids if str(r).startswith(date) and len(str(r)) >= 16})
        except (OSError, ValueError):
            return []
    out: list[str] = []
    with open(p, encoding="cp932", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        col = next((c for c in (reader.fieldnames or []) if "レースID" in c), None)
        if not col:
            return []
        for row in reader:
            rid = NON_DIGIT.sub("", str(row.get(col, "")))[:16]
            if len(rid) == 16 and rid.startswith(date) and rid not in out:
                out.append(rid)
    return sorted(out)


def fetch(rid: str, out_dir: Path, timeout: int = 90) -> tuple[bool, str]:
    cmd = [*PY32, "jvlink_odds.py", "--race", rid, "--stage", "am9", "--out-dir", str(out_dir)]
    try:
        r = subprocess.run(cmd, cwd=str(ROOT), env=dict(os.environ, PYTHONUTF8="1"), capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    line = next((ln for ln in (r.stdout or "").splitlines() if "[jvlink_odds]" in ln), (r.stdout or r.stderr or "").strip()[-200:])
    ok = r.returncode == 0 and "ok=True" in line
    return ok, line


def discord(msg: str) -> None:
    try:
        cfg = json.loads((ROOT / "notify_config.json").read_text(encoding="utf-8"))
        url = cfg.get("discord_webhook")
        if not url:
            return
        req = urllib.request.Request(url, data=json.dumps({"content": msg[:1900]}).encode("utf-8"),
                                     headers={"Content-Type": "application/json", "User-Agent": "pycaliai-am9"})
        urllib.request.urlopen(req, timeout=15).read()
    except Exception as e:  # noqa: BLE001
        print(f"[am9] discord 送信失敗: {type(e).__name__}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().strftime("%Y%m%d"))
    ap.add_argument("--dry", action="store_true", help="レース一覧を表示するだけ (JV-Link を呼ばない)")
    ap.add_argument("--limit", type=int, default=0, help="先頭 N レースだけ (テスト用)")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    rids = race_ids(a.date)
    if not rids:
        print(f"[am9] {a.date}: 出走表が無い (開催日でない) → 何もしない")
        return 0
    if a.limit:
        rids = rids[:a.limit]
    print(f"[am9] {a.date}: {len(rids)} レース  開始 {dt.datetime.now():%H:%M:%S}")
    if a.dry:
        print("\n".join(rids))
        return 0
    out_dir = ROOT / "reports" / "am9_odds" / a.date
    out_dir.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    for rid in rids:
        ok, line = fetch(rid, out_dir)
        if not ok:
            time.sleep(2)
            ok, line = fetch(rid, out_dir)
        print(f"  {rid} {'ok' if ok else 'NG'}  {line[:160]}")
        if not ok:
            failed.append(rid)
        time.sleep(0.3)
    print(f"[am9] 終了 {dt.datetime.now():%H:%M:%S}  成功 {len(rids) - len(failed)} / {len(rids)}")
    if failed and len(failed) >= max(1, len(rids) // 5):
        discord(f"**⚠ 朝 9 時オッズの保存に失敗 {a.date}**: {len(failed)} / {len(rids)} レース。CW-14 の前向き確認から今日の分が欠けます。"
                f"手動: `python scripts\\am9_snapshot.py --date {a.date}` (9:30 までなら有効)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
