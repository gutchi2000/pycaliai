"""
jrdb_snapshot.py — JRDB 前日ファイル (KYI) を発走前に保存し、後で「書き換えられていないか」を照合する

背景 (2026-11-12): JRDB の前日指数 (IDM・騎手指数・情報指数) を v6 に混ぜると ◎3着内が上がったが、
検証に使った値はアーカイブから後で取ったもの。JRDB が過去の値を直していないことを示すには、
発走前に自分で保存したファイルと後日のアーカイブを突き合わせる必要がある (収集待ち台帳 CW-12)。

  capture : 開催日が今日以降の KYI を取得し、内容が前回と違うときだけ保存 (同じなら何もしない)
            data/jrdb/snapshots/Kyi/KYI{yymmdd}__{取得時刻}.zip + manifest.jsonl (取得時刻・sha256・Last-Modified)
  verify  : 開催日を過ぎた「発走前の保存分」を現在のアーカイブと照合 (sha 一致 / 3 指数の不一致行数)
  status  : 保存済みの開催日数・発走前に取れた日数

鍵は本体 notify_config.json の jrdb_user / jrdb_password (値は表示しない)。データは git 管理外。
実行 (root から): python scripts/jrdb_snapshot.py capture [--notify]
                  python scripts/jrdb_snapshot.py verify
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import re
import sys
import time
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jrdb_fetch as jf  # noqa: E402

SNAP = jf.ROOT / "data/jrdb/snapshots/Kyi"
MANIFEST = jf.ROOT / "data/jrdb/snapshots/manifest.jsonl"
PRE_RACE_HOUR = 9          # 開催日の 9:00 より前に取れたものを「発走前」と数える (第 1R は 9:50 前後)
NAME = re.compile(r"KYI(\d{6})\.zip$", re.I)


def race_date(link):
    m = NAME.search(link)
    if not m:
        return None
    yy = int(m.group(1)[:2])                       # 99 は 1999 年 (2099 と読むと全部「今日以降」になる)
    return dt.datetime.strptime(("19" if yy >= 90 else "20") + m.group(1), "%Y%m%d").date()


def read_manifest():
    if not MANIFEST.exists():
        return []
    return [json.loads(x) for x in MANIFEST.read_text(encoding="utf-8").splitlines() if x.strip()]


def kyi_rows(zbytes):
    """zip の中身 → {(racekey, 馬番): (IDM, 騎手指数, 情報指数)}"""
    out = {}
    with zipfile.ZipFile(io.BytesIO(zbytes)) as z:
        for n in z.namelist():
            for line in z.read(n).split(b"\n"):
                if len(line) < 70:
                    continue
                out[(line[0:8], line[8:10])] = (line[54:59].strip(), line[59:64].strip(), line[64:69].strip())
    return out


def notify(msg):
    try:
        cfg = json.loads((jf.ROOT / "notify_config.json").read_text(encoding="utf-8"))
        url = cfg.get("discord_webhook")
        if url:
            requests.post(url, json={"content": msg}, timeout=30)
    except Exception as e:  # 通知の失敗で本体を落とさない
        print(f"notify failed: {type(e).__name__}")


def cmd_capture(args):
    now = dt.datetime.now()
    since = dt.datetime.strptime(args.since, "%Y%m%d").date() if args.since else now.date()
    s = jf.session()
    url, st, links = jf.list_links(s, "Kyi")
    if st != 200:
        print(f"Kyi index HTTP {st}")
        if args.notify:
            notify(f"⚠️ JRDB 前日ファイルの一覧が取れません (HTTP {st})")
        return 1
    last = {}
    for m in read_manifest():
        last[m["race_date"]] = m["sha256"]
    SNAP.mkdir(parents=True, exist_ok=True)
    saved = 0
    for link in links:
        d = race_date(link)
        if d is None or d < since:
            continue
        r = s.get(link, timeout=120)
        time.sleep(1)
        if r.status_code != 200:
            print(f"{link.rsplit('/', 1)[-1]}: HTTP {r.status_code}")
            continue
        sha = hashlib.sha256(r.content).hexdigest()
        key = d.strftime("%Y%m%d")
        if last.get(key) == sha:
            continue
        name = f"KYI{d.strftime('%y%m%d')}__{now.strftime('%Y%m%dT%H%M%S')}.zip"
        (SNAP / name).write_bytes(r.content)
        pre = now < dt.datetime.combine(d, dt.time(PRE_RACE_HOUR, 0))
        rec = {"type": "Kyi", "race_date": key, "captured_at": now.isoformat(timespec="seconds"),
               "pre_race": bool(pre), "sha256": sha, "bytes": len(r.content), "file": name,
               "last_modified": r.headers.get("Last-Modified"), "revised": key in last}
        with open(MANIFEST, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        last[key] = sha
        saved += 1
        print(f"saved {name} pre_race={pre} revised={rec['revised']}")
    print(f"capture: 新規保存 {saved} 本")
    # 成績 (SED) の今年ぶんも取り直す (v7 候補の過去走特徴用、2026-11-13)。失敗しても capture の結果は変えない。
    try:
        import subprocess
        rr = subprocess.run([sys.executable, str(jf.ROOT / "scripts" / "jrdb_refresh.py")], cwd=str(jf.ROOT), capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=900)
        print((rr.stdout or "").strip().splitlines()[-1] if (rr.stdout or "").strip() else f"jrdb_refresh rc={rr.returncode}")
        if rr.returncode != 0 and args.notify:
            notify("⚠️ JRDB 成績 (SED) の更新に失敗。v7 の過去走特徴が 1 週古くなります: python scripts/jrdb_refresh.py")
    except Exception as e:  # noqa: BLE001
        print(f"jrdb_refresh 失敗: {type(e).__name__}: {e}")
    today = now.strftime("%Y%m%d")
    if args.notify and (jf.ROOT / f"data/weekly/{today}.csv").exists() and today not in last:
        notify(f"⚠️ JRDB 前日ファイル KYI{today[2:]} が発走前に保存できていません")
    return 0


def cmd_verify(args):
    today = dt.date.today().strftime("%Y%m%d")
    first = {}
    for m in read_manifest():
        if m["pre_race"] and m["race_date"] < today:
            first.setdefault(m["race_date"], m)      # その開催日で最初に取れた発走前の版
    if not first:
        print("照合できる発走前の保存分がまだありません")
        return 0
    s = jf.session()
    url, st, links = jf.list_links(s, "Kyi")
    by_date = {race_date(l).strftime("%Y%m%d"): l for l in links if race_date(l)}
    same = diff_days = rows = diff_rows = 0
    for key, m in sorted(first.items()):
        link = by_date.get(key)
        if not link:
            print(f"{key}: アーカイブに無い")
            continue
        r = s.get(link, timeout=120)
        time.sleep(1)
        if hashlib.sha256(r.content).hexdigest() == m["sha256"]:
            same += 1
            continue
        a, b = kyi_rows((SNAP / m["file"]).read_bytes()), kyi_rows(r.content)
        keys = set(a) & set(b)
        d = sum(1 for k in keys if a[k] != b[k])
        rows += len(keys)
        diff_rows += d
        diff_days += 1
        print(f"{key}: ファイルは変化。3 指数の不一致 {d}/{len(keys)} 行 (保存 {m['captured_at']})")
    print(f"verify: 開催日 {len(first)} / 完全一致 {same} / ファイル変化 {diff_days} "
          f"(3 指数の不一致 {diff_rows}/{rows} 行)")
    return 0


def cmd_status(args):
    ms = read_manifest()
    days = sorted({m["race_date"] for m in ms})
    pre = sorted({m["race_date"] for m in ms if m["pre_race"]})
    rev = sum(1 for m in ms if m.get("revised"))
    print(f"保存 {len(ms)} 本 / 開催日 {len(days)} / 発走前に取れた開催日 {len(pre)} / 同じ開催日の差し替え {rev} 回")
    if pre:
        print(f"発走前の保存: {pre[0]} 〜 {pre[-1]}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--since", help="YYYYMMDD (既定=今日)。テスト用")
    c.add_argument("--notify", action="store_true", help="開催日なのに保存が無いとき Discord へ")
    c.set_defaults(fn=cmd_capture)
    sub.add_parser("verify").set_defaults(fn=cmd_verify)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
