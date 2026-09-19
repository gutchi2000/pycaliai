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

**2026-09-19夜、追加確認事項(ユーザー指摘)**: 上記の「未来日はrc=-1」という結果は
「前日夜」から見た未来日についてのものであり、「当日朝」に見た当日日付でも同じく
成功するとは限らない(=まだ検証していない)。「当日になれば当日分は取得できる」という
仮説自体は妥当そうだが未検証のため、`--diag-extract`モードを追加し、`t35_shadow.ps1`の
`-Schedule`内から毎朝自動的に(意思決定には使わず診断ログのみに)実行するよう配線した。
これにより2026-09-20朝の実機結果が`logs/jvlink_probe_diag_{date}.log`に自動的に
記録される。この診断モードはレコード種別ID("RA")の存在確認に加え、当日分なら既知の
data/weekly/{date}.csv があるはずなので、その中の発走時刻(HH:MM→HHMM)を生バイト列から
探索してオフセットが特定できるかも試す(=公表仕様を信用せず、既知の正解値との突合で
検証するという本プロジェクトの既定方針に従う)。まだ本番の判定ロジックへは一切配線
していない(診断専用、失敗しても-Scheduleの本フローに影響しないようtry/exceptで隔離)。

戻り値は3値: "race_detected" / "no_race" / "probe_error"。

出力: 標準出力に上記3値のいずれか1行 + 診断情報(stderr)。
実行: py -3.12-32 jvlink_race_day_probe.py --date 20260921 [--timeout 25]
診断: py -3.12-32 jvlink_race_day_probe.py --date 20260920 --diag-extract [--timeout 60]
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from datetime import datetime
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


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_known_post_times(date_str: str) -> dict[str, str]:
    """data/weekly/{date}.csv → {rid16: 'HH:MM'} を依存最小(csv+re のみ、pandas不要)で読む。
    32bit Python環境(py -3.12-32)でも動かすため t10_runner を import しない
    (t10_runner.load_post_times と同一ロジックの複製、意図的)。"""
    import csv as _csv
    import re as _re
    path = REPO_ROOT / "data" / "weekly" / f"{date_str}.csv"
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    try:
        with open(path, encoding="cp932", errors="replace", newline="") as f:
            reader = _csv.DictReader(f)
            rid_col = next((c for c in (reader.fieldnames or []) if "レースID" in c), None)
            tm_col = next((c for c in (reader.fieldnames or []) if "発走" in c), None)
            if not rid_col or not tm_col:
                return {}
            for row in reader:
                rid = _re.sub(r"\D", "", str(row.get(rid_col, "")))[:16]
                if len(rid) != 16 or rid in out:
                    continue
                s = str(row.get(tm_col, "") or "").strip()
                m = _re.match(r"^(\d{1,2})[:時](\d{2})", s) or _re.match(r"^(\d{3,4})$", s)
                if not m:
                    continue
                if ":" in s or "時" in s:
                    hh, mm = int(m.group(1)), int(m.group(2))
                else:
                    v = int(m.group(1)); hh, mm = v // 100, v % 100
                out[rid] = f"{hh:02d}:{mm:02d}"
    except Exception:
        return {}
    return out


def diag_extract(date_str: str, timeout_s: float = 60.0) -> dict:
    """診断専用(意思決定には未配線): JVOpen("RACE")で当日分のRAレコードを取得できるか、
    取得できた場合そのバイト列の中に既知の発走時刻(data/weekly/{date}.csvがあれば)の
    ASCII表現(HHMM)が見つかるか総当たりで探し、見つかったオフセットを報告する。
    公表仕様のオフセットを信用せず既知の正解値との突合で検証する方針
    (このプロジェクトの既定方針、O1/O3/O4パーサーの検証と同じやり方)。
    失敗しても例外は投げず、結果dictのerrorキーに理由を入れて返す(呼び出し側で
    非干渉に扱えるように)。"""
    out: dict = {"date": date_str, "checked_at": datetime.now().isoformat(timespec="seconds")}
    known = _load_known_post_times(date_str)
    out["known_post_times_source"] = "data/weekly" if known else None
    out["known_post_times_count"] = len(known)
    known_hhmm = sorted({v.replace(":", "") for v in known.values()})
    out["known_hhmm_values"] = known_hhmm

    try:
        import win32com.client as w
    except Exception as exc:
        out["error"] = f"win32com import失敗: {exc}"
        return out
    try:
        jv = w.Dispatch("JVDTLab.JVLink")
    except Exception as exc:
        out["error"] = f"JVLink Dispatch失敗: {exc}"
        return out
    try:
        if jv.JVInit(SID) != 0:
            out["error"] = "JVInit失敗"
            return out
    except Exception as exc:
        out["error"] = f"JVInit例外: {exc}"
        return out

    from_ts = f"{date_str}000000"
    records_examined = 0
    ra_records = 0
    offset_hits: dict[int, int] = {}
    sample_dump = None
    try:
        try:
            r = jv.JVOpen("RACE", from_ts, 1)
        except Exception as exc:
            out["error"] = f"JVOpen例外: {exc}"
            return out
        rc = r[0] if isinstance(r, tuple) else r
        out["jvopen_rc"] = rc
        if rc != 0:
            out["error"] = f"JVOpen失敗 rc={rc}"
            return out

        t0 = time.time()
        while time.time() - t0 < timeout_s:
            try:
                rr = jv.JVRead(" " * 120000, 120000, " " * 256)
            except Exception as exc:
                out["error"] = f"JVRead例外: {exc}"
                break
            size = rr[0] if isinstance(rr, tuple) else rr
            if size == 0:
                break
            if size < 0:
                continue
            buf = rr[1][:size] if isinstance(rr, tuple) else ""
            records_examined += 1
            if buf[:2] != "RA":
                continue
            ra_records += 1
            if sample_dump is None:
                sample_dump = buf[:400]
            for hhmm in known_hhmm:
                idx = buf.find(hhmm)
                if idx >= 0:
                    offset_hits[idx] = offset_hits.get(idx, 0) + 1
    finally:
        try:
            jv.JVClose()
        except Exception:
            pass

    out["records_examined"] = records_examined
    out["ra_records"] = ra_records
    out["sample_ra_record_head400"] = sample_dump
    out["hhmm_offset_hits"] = {str(k): v for k, v in sorted(offset_hits.items())}
    out["note"] = ("hhmm_offset_hitsで同じoffsetに複数レース分のヒットが集中していれば"
                   "発走時刻フィールドの位置として有望。値がバラバラなら偶然一致の疑い"
                   "(未検証のまま判定ロジックへ使わないこと)。")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYYMMDD")
    ap.add_argument("--timeout", type=float, default=25.0)
    ap.add_argument("--diag-extract", action="store_true",
                    help="意思決定には未配線の診断モード。RAレコードに既知発走時刻が"
                         "見つかるか総当たりで探索し結果をJSONで標準出力へ")
    args = ap.parse_args()

    if args.diag_extract:
        result = diag_extract(args.date, args.timeout or 60.0)
        print(json.dumps(result, ensure_ascii=False, indent=1))
        return 0

    result, detail = probe(args.date, args.timeout)
    print(result)
    print(f"[jvlink_race_day_probe] date={args.date} result={result} detail={detail}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
