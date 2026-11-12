# -*- coding: utf-8 -*-
"""
power_rating_retired.py — 登録抹消（引退）馬の一覧を JV-Link から取る（ローカル専用）
================================================================================
強さスコア（power_rating.py）のランキングから引退馬を外すための補助。JV-Link の蓄積系
dataspec「DIFN」に含まれる競走馬マスタ（UM）の抹消区分を読み、抹消済みの馬名と抹消日を
reports/power_rating/retired.csv（.gitignore 済み、JRA-VAN 由来なので配布しない）に書く。

★必ず 32-bit Python で実行（JV-Link は 32bit COM）:
    py -3.12-32 power_rating_retired.py                # 基準日の 420 日前以降に更新された UM
    py -3.12-32 power_rating_retired.py --days 800

UM は馬の情報が更新されるたびに流れる。抹消も更新なので「直近 N 日に抹消された馬」は
fromtime = N 日前 で必ず拾える。power_rating の「現役」は 1 年以内に出走した馬なので、
420 日あれば足りる。既存の retired.csv があれば読み込んで足し込む（消さない）。

UM レコード（文字位置、0 始まり。馬名より前は半角のみ）:
    [0:2]="UM"  [11:21]=血統登録番号  [21]=抹消区分(0:現役 1:抹消)
    [22:30]=登録日  [30:38]=抹消日  [38:46]=生年月日  [46:64]=馬名（全角 18 文字）
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / "reports" / "power_rating" / "retired.csv"
SID_FILE = BASE / "data" / "jvlink_sid.txt"


def _sid() -> str:
    try:
        return SID_FILE.read_text(encoding="utf-8").strip().splitlines()[0] or "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def fix_mojibake(buf: str) -> str:
    """JV-Link の文字列が「cp932 のバイト列を 1 バイト 1 文字で読んだ形」なら日本語に戻す。

    起動のしかた（PowerShell から / Bash から）で JVRead の戻りが変わる: 正しい日本語で来る
    こともあれば、Shift_JIS のバイトが cp1252 の文字として来ることもある（実測 2026-11-12）。
    全文字が 1 バイトに戻せるときだけ変換し、すでに日本語ならそのまま返す。
    """
    out = bytearray()
    for ch in buf:
        o = ord(ch)
        if o < 256:
            out.append(o)
            continue
        try:
            out += ch.encode("cp1252")
        except UnicodeEncodeError:
            return buf
    try:
        return out.decode("cp932")
    except UnicodeDecodeError:
        return buf


def parse_um(buf: str) -> tuple[str, str, str, str] | None:
    """UM レコード → (血統登録番号, 馬名, 抹消区分, 抹消日)。形が合わなければ None。"""
    if buf[:2] != "UM" or len(buf) < 64:
        return None
    buf = fix_mojibake(buf)
    if len(buf) < 64:
        return None
    ketto, dele, deldate = buf[11:21], buf[21], buf[30:38]
    name = buf[46:64].strip(" 　")
    if not ketto.isdigit() or dele not in "01" or not name:
        return None
    return ketto, name, dele, deldate if deldate.isdigit() and deldate != "00000000" else ""


def fetch(from_ts: str, timeout_s: float) -> tuple[dict[str, tuple[str, str, str]], str]:
    """Returns ({血統登録番号: (馬名, 抹消区分, 抹消日)}, エラー文字列)。同じ馬は後勝ち。"""
    try:
        import win32com.client as w
        jv = w.Dispatch("JVDTLab.JVLink")
    except Exception as exc:
        return {}, f"JVLink を開けません（32-bit Python で実行していますか）: {exc}"
    if jv.JVInit(_sid()) != 0:
        return {}, "JVInit 失敗"
    horses: dict[str, tuple[str, str, str]] = {}
    try:
        r = jv.JVOpen("DIFN", from_ts, 1)
        rc = r[0] if isinstance(r, tuple) else r
        if rc != 0:
            return {}, f"JVOpen 失敗 rc={rc} (from={from_ts})"
        n_dl = r[2] if isinstance(r, tuple) and len(r) > 2 else 0
        t0 = time.time()
        while n_dl and time.time() - t0 < timeout_s:      # ダウンロード完了待ち
            st = jv.JVStatus()
            if st < 0:
                return {}, f"JVStatus 失敗 rc={st}"
            if st >= n_dl:
                break
            time.sleep(1.0)
        while time.time() - t0 < timeout_s:
            rr = jv.JVRead(" " * 4000, 4000, " " * 256)
            size = rr[0] if isinstance(rr, tuple) else rr
            if size == 0:
                return horses, ""
            if size == -1:                                # ファイルの切れ目
                continue
            if size == -3:                                # ダウンロード中
                time.sleep(0.5)
                continue
            if size < 0:
                return horses, f"JVRead 失敗 rc={size}"
            fname = str(rr[3]) if isinstance(rr, tuple) and len(rr) > 3 else ""
            if fname and not fname.startswith("UM"):      # UM 以外のファイルは丸ごと飛ばす
                jv.JVSkip()
                continue
            rec = parse_um(rr[1][:size])
            if rec:
                horses[rec[0]] = rec[1:]
        return horses, "時間切れ（途中まで取得）"
    finally:
        try:
            jv.JVClose()
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="登録抹消馬の一覧を JV-Link から取る（ローカル専用）")
    ap.add_argument("--days", type=int, default=420, help="何日前以降の更新を読むか")
    ap.add_argument("--timeout", type=float, default=1800.0)
    args = ap.parse_args(argv)
    from_ts = (date.today() - timedelta(days=args.days)).strftime("%Y%m%d") + "000000"
    horses, err = fetch(from_ts, args.timeout)
    print(f"UM {len(horses):,} 頭を読みました (from={from_ts})" + (f" / {err}" if err else ""))
    if not horses:
        return 1
    rows: dict[str, tuple[str, str]] = {}
    if OUT.exists():
        with OUT.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                rows[r["血統登録番号"]] = (r["馬名"], r["抹消日"])
    for ketto, (name, dele, deldate) in horses.items():
        if dele == "1":
            rows[ketto] = (name, deldate)
        else:
            rows.pop(ketto, None)                         # 再登録された馬は外す
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["血統登録番号", "馬名", "抹消日"])
        for ketto in sorted(rows):
            wr.writerow([ketto, *rows[ketto]])
    n_new = sum(1 for v in horses.values() if v[1] == "1")
    print(f"抹消馬 {len(rows):,} 頭 → {OUT}（今回の取得分に抹消 {n_new:,} 頭、"
          f"取得 {datetime.now():%Y-%m-%d %H:%M}）")
    return 0 if not err else 2


if __name__ == "__main__":
    sys.exit(main())
