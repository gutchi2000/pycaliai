# -*- coding: utf-8 -*-
"""
jvlink_race_day_probe.py — data/weekly/{date}.csv に依存しない「今日JRA開催があるか」の
独立確認の試み (spec「EXP05-F 開催日判定」)
========================================================================================
★必ず 32-bit Python で実行: py -3.12-32 jvlink_race_day_probe.py --date 20260921
  (jvlink_odds.py / jvlink_trio_odds.py と同型。JV-Link は 32-bit COM)

**2026-09-19 実測で判明した既知の限界 (このファイルは開催日判定の意思決定には使わない)**:
JV-Link の "RACE" dataspec (jvlink_trio_odds.py が `JVOpen("RACE", from_ts, 1)` で
蓄積系=確定済みデータの取得に使っている実績のあるもの) を使い、option=1・option=2の
両方を実機で試した。
  - 本日(開催中・レース確定分あり)の日付 → race_detected (成功)
  - 翌日・翌々日など未確定/未来の日付 → 両optionともJVOpen自体が `rc=-1` で失敗
    (「指定期間にデータなし」の意味と推測されるが未確認)
つまりこの「RACE」蓄積系dataspecは**既に確定した過去〜当日分の存在確認にしか使えず、
まだ開催されていない将来日の「開催予定があるか」の検出には使えない**ことを実測で確認した。
実運用の開催日判定 (`t35_shadow.ps1`) はこの結果を踏まえ、このprobeを使わず
「data/weekly/{date}.csv の有無」+「既知開催日オーバーライドリスト
(data/jra_known_race_days_override.json)」+「曜日」+「再試行と明示的な判定保留」の
組み合わせに切り替えた。このファイルは調査記録として残す (存在確認のみで内部レコードは
パースしていないため実害は無いが、将来日の判定には使えないことを次に触る人が
繰り返し発見しないよう明記する)。

戻り値は3値: "race_detected" / "no_race" / "probe_error"。

出力: 標準出力に上記3値のいずれか1行 + 診断情報(stderr)。
実行: py -3.12-32 jvlink_race_day_probe.py --date 20260921 [--timeout 25]
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

BASE = Path(__file__).parent
try:
    SID = (BASE / "data" / "jvlink_sid.txt").read_text(encoding="utf-8").strip().splitlines()[0] or "UNKNOWN"
except Exception:
    SID = "UNKNOWN"


def probe(date_str: str, timeout_s: float = 25.0) -> tuple[str, str]:
    """(result, detail) を返す。result in {"race_detected","no_race","probe_error"}。"""
    try:
        import win32com.client as w
    except Exception as exc:
        return "probe_error", f"win32com import失敗: {exc}"

    try:
        jv = w.Dispatch("JVDTLab.JVLink")
    except Exception as exc:
        return "probe_error", f"JVLink Dispatch失敗: {exc}"

    try:
        if jv.JVInit(SID) != 0:
            return "probe_error", "JVInit失敗"
    except Exception as exc:
        return "probe_error", f"JVInit例外: {exc}"

    from_ts = f"{date_str}000000"  # JVOpen の fromtime は YYYYMMDDHHMMSS (jvlink_trio_odds.py で実証済みの形式)
    try:
        try:
            r = jv.JVOpen("RACE", from_ts, 2)
        except Exception as exc:
            return "probe_error", f"JVOpen例外: {exc}"
        rc = r[0] if isinstance(r, tuple) else r
        if rc != 0:
            return "probe_error", f"JVOpen失敗 rc={rc}"

        found_ra = 0
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            try:
                rr = jv.JVRead(" " * 120000, 120000, " " * 256)
            except Exception as exc:
                return "probe_error", f"JVRead例外: {exc}"
            size = rr[0] if isinstance(rr, tuple) else rr
            if size == 0:
                break  # 全読込完了 (このJVOpenが対象とする範囲を読み切った)
            if size < 0:
                continue  # ファイル境界
            buf = rr[1][:size] if isinstance(rr, tuple) else ""
            if buf[:2] == "RA":
                found_ra += 1
                # 存在確認だけが目的なので1件見つかった時点で十分。
                # (このJVOpenの対象範囲を全部読み切る必要はない)
                break
        if found_ra > 0:
            return "race_detected", f"RAレコード検出 (存在確認のみ、内容は未パース)"
        return "no_race", "RAレコード0件"
    finally:
        try:
            jv.JVClose()
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYYMMDD")
    ap.add_argument("--timeout", type=float, default=25.0)
    args = ap.parse_args()
    result, detail = probe(args.date, args.timeout)
    print(result)
    print(f"[jvlink_race_day_probe] date={args.date} result={result} detail={detail}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
