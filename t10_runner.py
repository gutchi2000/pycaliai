# -*- coding: utf-8 -*-
"""
t10_runner.py — 当日 T-10 自動馬券ライン（スケジューラ本体）
================================================================
土日の当日に 1 回起動しておくと、bundle の全レースについて発走 lead 分前
(デフォルト T-10) に自動で:

  1. jvlink_odds.py (32-bit, py -3.12-32) → reports/live_odds/{rid16}.json
  2. compute_bets.py --race {rid16} --live-odds-dir --apply
       → reports/cowork_output/{date}_bets.json へ当該レースのみ in-place merge
  3. validate_cowork_bets.py --apply (見送りガード)
  4. 買い目をコンソール表示 + ビープ（投票は人間が IPAT で行う）

を実行する。NiceGUI (ローカル) は cowork_output を ui.timer で随時読むので
画面にもそのまま反映される。HF への push は本ランナーでは行わない（手動）。

発走時刻: data/weekly/{date}.csv の「発走時刻」列（TARGET 出走表）。
fail-safe: オッズ欠損/鮮度NG/overround 異常は compute_bets 側で見送りになる。

実行:
  venv311\\Scripts\\python.exe t10_runner.py                # 当日( bundle 自動検出)
  venv311\\Scripts\\python.exe t10_runner.py 20260614       # 日付指定
  venv311\\Scripts\\python.exe t10_runner.py 20260614 --dry # 計算のみ・書込なし
  venv311\\Scripts\\python.exe t10_runner.py 20260607 --once 2026060705030211 --max-age-min 99999
                                                            # 1レース即時処理（動作テスト用）
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

BASE = Path(__file__).parent
PY32 = ["py", "-3.12-32"]          # JV-Link は 32-bit COM
LIVE_DIR = BASE / "reports" / "live_odds"
POLL_SEC = 15

sys.stdout.reconfigure(encoding="utf-8")


def _rid16(x) -> str:
    return re.sub(r"\D", "", str(x))[:16]


def beep():
    try:
        import winsound
        winsound.Beep(880, 250); winsound.Beep(1320, 250)
    except Exception:
        print("\a", end="")


def latest_bundle_date() -> str | None:
    files = glob.glob(str(BASE / "reports" / "cowork_input" / "*_bundle.json"))
    dates = sorted({Path(f).name.split("_")[0] for f in files
                    if re.match(r"^\d{8}_", Path(f).name)}, reverse=True)
    return dates[0] if dates else None


def parse_hhmm(s: str) -> tuple[int, int] | None:
    """'15:40' / '1540' / '15:40:00' / '15時40分' → (15, 40)"""
    s = str(s or "").strip()
    m = re.match(r"^(\d{1,2})[:時](\d{2})", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.match(r"^(\d{3,4})$", s)
    if m:
        v = int(m.group(1))
        return v // 100, v % 100
    return None


def load_post_times(date_str: str) -> dict[str, str]:
    """data/weekly/{date}.csv → {rid16: '15:40'} (cp932, TARGET 出走表)"""
    path = BASE / "data" / "weekly" / f"{date_str}.csv"
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    with open(path, encoding="cp932", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        rid_col = next((c for c in (reader.fieldnames or []) if "レースID" in c), None)
        tm_col = next((c for c in (reader.fieldnames or []) if "発走" in c), None)
        if not rid_col or not tm_col:
            return {}
        for row in reader:
            rid = _rid16(row.get(rid_col, ""))
            if len(rid) == 16 and rid not in out:
                hm = parse_hhmm(row.get(tm_col, ""))
                if hm:
                    out[rid] = f"{hm[0]:02d}:{hm[1]:02d}"
    return out


def run_cmd(cmd: list[str], timeout: int = 180) -> tuple[int, str]:
    env = dict(os.environ, PYTHONUTF8="1")
    try:
        r = subprocess.run(cmd, cwd=str(BASE), env=env, timeout=timeout,
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return -1, f"timeout {timeout}s: {' '.join(cmd)}"
    except FileNotFoundError as e:
        return -1, str(e)


def show_race_bets(date_str: str, rid16: str):
    """apply 後の bets.json から当該レースを読み戻して表示。"""
    path = BASE / "reports" / "cowork_output" / f"{date_str}_bets.json"
    if not path.exists():
        return
    raw = json.loads(path.read_text(encoding="utf-8"))
    races = raw["bets"] if isinstance(raw, dict) and "bets" in raw else raw
    e = next((r for r in races if _rid16(r.get("race_id", "")) == rid16), None)
    if e is None:
        return
    if e.get("bets"):
        tot = sum(b["購入額"] for b in e["bets"])
        print(f"  🎫 {e.get('race_label','')} [{e.get('race_nature','')}] "
              f"{len(e['bets'])}点 ¥{tot:,}")
        for b in e["bets"]:
            print(f"     {b['馬券種']:3s} {b['買い目']:8s} ¥{b['購入額']:>5,}  {b.get('理由','')}")
        print(f"     → {e.get('race_reason','')}")
        beep()
    else:
        print(f"  — {e.get('race_label','')} [見送り] {e.get('race_reason','')}")


def process_race(date_str: str, bundle: Path, rid16: str, label: str,
                 max_age_min: float, dry: bool) -> bool:
    """T-10 処理 1 レース分。True=完了 (見送り含む)。"""
    now = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{now}] ▶ T-10 処理開始: {label} ({rid16})")

    # 1. JV-Link ライブオッズ (32-bit)。失敗しても compute_bets が fail-safe 見送りにする
    rc, out = run_cmd([*PY32, "jvlink_odds.py", "--race", rid16])
    line = next((l for l in out.splitlines() if "[jvlink_odds]" in l), out.strip()[-200:])
    print(f"  [1/3] jvlink_odds (exit {rc}) {line}")

    # 2. compute_bets (当該レースのみ、ライブ必須モード)
    cmd = [sys.executable, "compute_bets.py", "--bundle", str(bundle),
           "--live-odds-dir", str(LIVE_DIR), "--max-age-min", str(max_age_min),
           "--race", rid16]
    if not dry:
        cmd.append("--apply")
    rc, out = run_cmd(cmd)
    print(f"  [2/3] compute_bets (exit {rc})")
    if rc != 0:
        print("       " + out.strip().replace("\n", "\n       ")[-500:])
        return True   # 再試行しない (fail-safe 見送り扱い)
    if dry:
        # dry は compute_bets の出力をそのまま見せる
        for l in out.splitlines():
            if l.strip():
                print("       " + l)
        return True

    # 3. 見送りガード (書込後は必ず通す契約)
    rc, out = run_cmd([sys.executable, "validate_cowork_bets.py",
                       "--date", date_str, "--apply"])
    print(f"  [3/3] validate_cowork_bets (exit {rc})")
    if rc == 1:
        print("       ⚠ ガード実行不能: " + out.strip()[-300:])

    show_race_bets(date_str, rid16)
    return True


def main():
    ap = argparse.ArgumentParser(description="T-10 自動馬券ライン")
    ap.add_argument("date", nargs="?", default=None, help="YYYYMMDD (省略時は最新 bundle)")
    ap.add_argument("--lead-min", type=float, default=10.0,
                    help="発走の何分前に処理するか (default 10 = T-10)")
    ap.add_argument("--max-age-min", type=float, default=20.0,
                    help="ライブオッズ許容鮮度 (分) → compute_bets へ渡す")
    ap.add_argument("--dry", action="store_true", help="計算のみ。bets.json へ書き込まない")
    ap.add_argument("--once", default=None,
                    help="rid16 を指定して 1 レースだけ即時処理して終了 (テスト用)")
    args = ap.parse_args()

    date_str = args.date or latest_bundle_date()
    if not date_str:
        print("[ERROR] bundle が見つからない (reports/cowork_input/)"); return 1
    bundle = BASE / "reports" / "cowork_input" / f"{date_str}_bundle.json"
    if not bundle.exists():
        print(f"[ERROR] {bundle} が無い (Phase A を先に実行)"); return 1

    d = json.loads(bundle.read_text(encoding="utf-8"))
    races = d.get("races", [])
    post = load_post_times(date_str)

    # --once: 即時 1 レース (発走時刻不要)
    if args.once:
        rid = _rid16(args.once)
        rm = next((r.get("race_meta", {}) for r in races
                   if _rid16(r.get("race_id", "")) == rid), {})
        label = f"{rm.get('place','')}{rm.get('R','') or ''} {rm.get('course','')}".strip() or rid
        process_race(date_str, bundle, rid, label, args.max_age_min, args.dry)
        return 0

    # スケジュール構築
    sched = []   # (post_dt, rid16, label)
    base_day = datetime.strptime(date_str, "%Y%m%d")
    missing = []
    for r in races:
        rid = _rid16(r.get("race_id", ""))
        rm = r.get("race_meta", {})
        label = f"{rm.get('place','')} {rm.get('course','')} {rm.get('class','')}".strip()
        hm = parse_hhmm(post.get(rid, ""))
        if hm is None:
            missing.append((rid, label)); continue
        sched.append((base_day.replace(hour=hm[0], minute=hm[1]), rid, label))
    sched.sort()

    if missing:
        print(f"[WARN] 発走時刻不明 {len(missing)}R (data/weekly/{date_str}.csv に無い) → スキップ:")
        for rid, label in missing:
            print(f"   {rid} {label}")
    if not sched:
        print("[ERROR] スケジュール対象 0 レース"); return 1

    lead = timedelta(minutes=args.lead_min)
    print(f"\n========== T-10 自動馬券ライン  {date_str}  "
          f"{len(sched)}R (T-{args.lead_min:.0f}, 鮮度{args.max_age_min:.0f}分"
          f"{', DRY' if args.dry else ''}) ==========")
    for pt, rid, label in sched:
        print(f"  {pt:%H:%M} 発走 → {(pt-lead):%H:%M} 処理  {label}")
    print("  (Ctrl+C で停止。投票は人間が IPAT で行う)\n")

    done: set[str] = set()
    while len(done) < len(sched):
        now = datetime.now()
        for pt, rid, label in sched:
            if rid in done:
                continue
            if now >= pt:
                print(f"[{now:%H:%M:%S}] ✗ {label} は発走済み ({pt:%H:%M}) → スキップ")
                done.add(rid)
            elif now >= pt - lead:
                process_race(date_str, bundle, rid, label, args.max_age_min, args.dry)
                done.add(rid)
        if len(done) < len(sched):
            time.sleep(POLL_SEC)
    print(f"\n========== 全 {len(sched)}R 処理完了 ==========")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[停止] Ctrl+C")
        raise SystemExit(130)
