# -*- coding: utf-8 -*-
"""
market_snapshot.py — 発走31-38分前の市場オッズを EXP05-F 専用に取得・保存する (spec §7)
============================================================================================
t20_site_bets.py と同じ「レース毎タスク」方式。本番 T-10 (t10_runner.py) / サイト T-20
(t20_site_bets.py) / 学生大会 T-4 のどのラインにも一切触れない — オッズ取得は
reports/exp05fs_odds/ という専用ディレクトリ、forward_prices の stage は "exp05fs_t35" 専用。

31-38分前ちょうどに起動できるとは限らない (タスク起床遅延) ので、取得後に鮮度と
「31-38分前ウィンドウ内か」を検証し、外れていたら primary 評価から除外する
(spec §7: 別時刻への自動フォールバックは禁止)。

実行:
  venv311\\Scripts\\python.exe -m analysis.mcond.exp05_forward_shadow.market_snapshot \
      --once <rid16> --date 20260919 [--dry]
  venv311\\Scripts\\python.exe -m analysis.mcond.exp05_forward_shadow.market_snapshot \
      --date 20260919 --list-schedule --lead-min 35
"""
from __future__ import annotations
import argparse
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from t10_runner import build_schedule, load_post_times, latest_bundle_date  # noqa: E402

HERE = Path(__file__).resolve().parent
ODDS_DIR = BASE / "reports" / "exp05fs_odds"
PREDICT_DIR = BASE / "data" / "_research" / "mcond" / "exp05fs_predictions"
PY32 = ["py", "-3.12-32"]
JST = timezone(timedelta(hours=9))
LEAD_MIN_TARGET = 35.0
WINDOW_MIN = (31.0, 38.0)  # spec §7: 主評価に使える範囲


def _rid16(x) -> str:
    return re.sub(r"\D", "", str(x or ""))[:16]


def run_cmd(cmd, timeout=180):
    r = subprocess.run(cmd, cwd=BASE, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def fetch_and_validate(rid: str, scheduled_post: datetime) -> dict:
    """jvlink_odds.pyでオッズ取得しwindow内かを検証する。戻り値は append-only 保存用の記録。"""
    ODDS_DIR.mkdir(parents=True, exist_ok=True)
    jv_cmd = [*PY32, "jvlink_odds.py", "--race", rid, "--stage", "exp05fs_t35",
             "--out-dir", str(ODDS_DIR), "--scheduled-post", scheduled_post.isoformat()]
    rc, out = run_cmd(jv_cmd)
    line = next((l for l in out.splitlines() if "[jvlink_odds]" in l), out.strip()[-300:])
    print(f"  [1/2] jvlink_odds (exit {rc}) {line}")
    if rc != 0:
        return {"ok": False, "why": "jvlink_odds exit != 0", "raw_out": out[-500:]}

    try:
        market = json.loads((ODDS_DIR / f"{rid}.json").read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ok": False, "why": f"オッズJSON読込失敗: {exc}"}

    if not market.get("ok"):
        return {"ok": False, "why": f"オッズ ok=false ({market.get('reason', '')})", "market": market}

    fetched = market.get("fetched")
    try:
        ts = datetime.fromisoformat(str(fetched))
    except Exception:
        return {"ok": False, "why": f"fetched形式不正 ({fetched!r})", "market": market}
    sp = scheduled_post if scheduled_post.tzinfo else scheduled_post.replace(tzinfo=JST)
    ts_aware = ts if ts.tzinfo else ts.replace(tzinfo=JST)
    minutes_to_start = (sp - ts_aware).total_seconds() / 60.0

    valid_for_primary = WINDOW_MIN[0] <= minutes_to_start <= WINDOW_MIN[1]
    over = market.get("overround_tan")
    over_ok = isinstance(over, (int, float)) and 1.0 <= over <= 1.5
    tan = market.get("tansho") or {}
    tan_ok = len(tan) >= 6 and all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0
                                   for v in tan.values())

    return {"ok": True, "market": market, "minutes_to_start": minutes_to_start,
           "valid_for_primary": bool(valid_for_primary and over_ok and tan_ok),
           "window_min": list(WINDOW_MIN), "overround_ok": over_ok, "tansho_ok": tan_ok,
           "why": "" if (valid_for_primary and over_ok and tan_ok) else
                  f"window外またはオッズ異常 (minutes_to_start={minutes_to_start:.1f})"}


def process_race(date_str: str, rid: str, label: str, dry: bool,
                 scheduled_post: datetime | None = None) -> int:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{now}] EXP05-F T-35市場snapshot: {label} ({rid})")
    if scheduled_post is None:
        print("  発走時刻不明のため取得不可")
        return 1
    if scheduled_post.tzinfo is None:
        scheduled_post = scheduled_post.replace(tzinfo=JST)
    if datetime.now(JST) >= scheduled_post:
        print(f"  発走時刻超過のため取得不可 (予定発走 {scheduled_post:%H:%M})")
        return 1

    result = fetch_and_validate(rid, scheduled_post)
    result.update({"race_id": rid, "label": label, "date": date_str,
                   "scheduled_post": scheduled_post.isoformat(),
                   "captured_at": datetime.now(JST).isoformat(timespec="seconds")})
    print(f"  [2/2] ok={result['ok']} valid_for_primary={result.get('valid_for_primary')} "
          f"why={result.get('why', '')}")
    if dry:
        return 0
    from analysis.mcond.exp05_forward_shadow.predict_and_store import store_prediction, store_market_only
    try:
        store_prediction(date_str, rid, result)
    except FileNotFoundError as exc:
        # 特徴量snapshot(feature_snapshot.py --date)が無い週固有の失敗。市場は既に取得
        # できているので、これを捨てずに market_only レコードとして保存する
        # (spec: 市場データを捨てない。market_snapshot_saved=true/prediction_saved=false)。
        print(f"  [predict_and_store] 特徴量snapshot未生成のためprediction計算不可、"
             f"市場snapshotのみ保存する: {exc}")
        path = store_market_only(date_str, rid, result, reason="weekly_input_unavailable")
        print(f"  [market_only] -> {path.relative_to(BASE)}")
        (BASE / "logs").mkdir(exist_ok=True)
        with open(BASE / "logs" / "exp05fs_errors.log", "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()} {rid} weekly_input_unavailable "
                   f"(market_snapshot_saved=true, prediction_saved=false): {exc}\n")
        return 2
    except Exception as exc:
        print(f"  [predict_and_store] 失敗 (shadowは非干渉のため例外を握りつぶし専用ログのみ): {exc}")
        (BASE / "logs").mkdir(exist_ok=True)
        with open(BASE / "logs" / "exp05fs_errors.log", "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()} {rid} store_prediction失敗(未分類): {exc}\n")
        return 2
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="")
    ap.add_argument("--once", default="")
    ap.add_argument("--lead-min", type=float, default=LEAD_MIN_TARGET)
    ap.add_argument("--list-schedule", action="store_true")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    date_str = args.date or latest_bundle_date() or datetime.now().strftime("%Y%m%d")
    post = load_post_times(date_str)
    if not post:
        print(f"[ERROR] data/weekly/{date_str}.csv が無い、または発走時刻を読めない")
        return 1

    if args.list_schedule:
        races = [{"race_id": rid} for rid in post]
        sched, missing = build_schedule(date_str, races, args.lead_min)
        for post_dt, rid, _label in sched:
            print(f"{rid}\t{post_dt:%H:%M}")
        return 0

    if args.once:
        races = [{"race_id": rid, "race_meta": {}} for rid in post]
        sched, _ = build_schedule(date_str, races, args.lead_min)
        match = next(((p, r, l) for p, r, l in sched if r == _rid16(args.once)), None)
        if match is None:
            print(f"[ERROR] {args.once} の発走時刻が data/weekly/{date_str}.csv に無い")
            return 1
        post_dt, rid, label = match
        return process_race(date_str, rid, label or rid, args.dry, post_dt)

    print("使い方: --once <rid16> --date YYYYMMDD [--dry]  または  --list-schedule --lead-min 35")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
