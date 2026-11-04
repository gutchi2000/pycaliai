# -*- coding: utf-8 -*-
"""
scripts/task_watchdog.py — 土日 09:10 に T-10 / T-20 のタスクスケジューラ起動を確認し、失敗時だけ Discord へ (2026-11-04)
=====================================================================================================
検査 (開催日 = data/weekly/{今日}.csv がある日だけ。無ければ何もしない):
  1. タスク PyCaLiAI_T10 / PyCaLiAI_T20_Site の「前回の実行時刻」が今日で、「前回の結果」が 0 (または実行中)
  2. レース毎タスク PyCaLiAI_T10R_* / PyCaLiAI_T20R_* が 1 件以上登録されている (-Schedule が bundle を待って登録する)
  3. 当日の bundle reports/cowork_input/{今日}_bundle.json が存在する
どれかが落ちていれば Discord (notify_config.json の discord_webhook) に 1 通。正常時は無言。
登録: watchdog_register.ps1 (タスク PyCaLiAI_Watchdog, 土日 09:10)。手動: python scripts/task_watchdog.py [--date YYYYMMDD] [--force]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kaiko_warm import discord  # noqa: E402

RUNNING = {"267009", "267014", "0x41301", "0x41306"}   # 実行中 / 停止中(終了処理)


def _schtasks_csv(args: list[str]) -> list[dict]:
    r = subprocess.run(["schtasks", "/Query", "/FO", "CSV", "/V"] + args, capture_output=True)
    if r.returncode != 0:
        return []
    txt = None
    for enc in ("cp932", "utf-8-sig", "utf-16"):
        try:
            txt = r.stdout.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if not txt:
        return []
    rows = list(csv.DictReader(io.StringIO(txt)))
    out = []
    for x in rows:
        name = _col(x, "TaskName", "タスク名")
        if name and name not in ("TaskName", "タスク名"):
            x["TaskName"] = name          # 日本語ロケールでも TaskName で引けるよう正規化
            out.append(x)
    return out


def _col(row: dict, *cands: str):
    """列名はロケール依存 (英: "Last Run Time" / 日: "前回の実行時刻")。候補の部分一致で引く。"""
    for k in row:
        if k is None:
            continue
        for c in cands:
            if c in k:
                return row[k]
    return None


def check(date: str) -> list[str]:
    probs: list[str] = []
    today_prefix = f"{date[:4]}/{int(date[4:6])}/{int(date[6:8])}"      # schtasks の日付表記 2026/11/1 or 2026/11/01
    today_prefix2 = f"{date[:4]}/{date[4:6]}/{date[6:8]}"
    for name in ("PyCaLiAI_T10", "PyCaLiAI_T20_Site"):
        rows = _schtasks_csv(["/TN", name])
        if not rows:
            probs.append(f"{name}: タスクが見つからない (schtasks /Query 失敗)")
            continue
        row = rows[0]
        last = str(_col(row, "Last Run Time", "前回の実行時刻") or "")
        res = str(_col(row, "Last Result", "前回の結果") or "").strip()
        ran_today = last.startswith(today_prefix) or last.startswith(today_prefix2)
        if not ran_today:
            probs.append(f"{name}: 今日まだ起動していない (前回 {last or '不明'})")
        elif res not in ("0",) and res not in RUNNING:
            probs.append(f"{name}: 前回の結果 {res} (0 でない)")
    allrows = _schtasks_csv([])
    names = [r["TaskName"] for r in allrows]
    for pref, label in (("PyCaLiAI_T10R_", "T-10 レース毎タスク"), ("PyCaLiAI_T20R_", "T-20 レース毎タスク")):
        k = sum(1 for n in names if pref in n)
        if k == 0:
            probs.append(f"{label} (" + pref + "*) が 0 件 — -Schedule が bundle を待っているか失敗")
    bundle = ROOT / "reports" / "cowork_input" / f"{date}_bundle.json"
    if not bundle.exists():
        probs.append(f"bundle が無い: {bundle.name} (Phase A 未実行?)")
    return probs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().strftime("%Y%m%d"))
    ap.add_argument("--force", action="store_true", help="weekly CSV が無くても検査する")
    ap.add_argument("--dry", action="store_true", help="Discord へ送らず表示だけ")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    if not a.force and not (ROOT / "data" / "weekly" / f"{a.date}.csv").exists():
        print(f"[watchdog] {a.date}: 出走表が無い (開催日でない) → 検査しない")
        return 0
    probs = check(a.date)
    if not probs:
        print(f"[watchdog] {a.date}: T-10 / T-20 正常")
        return 0
    msg = f"**🚨 T-10/T-20 起動チェック {a.date[:4]}/{int(a.date[4:6])}/{int(a.date[6:8])}**\n" + "\n".join(f"- {p}" for p in probs) + \
          "\n手動: `.\\t10.ps1 -Schedule` / `.\\t20_site.ps1 -Schedule`"
    print(msg)
    if not a.dry:
        discord(msg)
    return 1


if __name__ == "__main__":
    sys.exit(main())
