"""
jrdb_fetch.py — JRDB 会員データコーナー (固定長データ) の確認とダウンロード

JRDB 会員向け JRDB-API は 2026-11-12 時点で「準備中」のため、従来のデータコーナーを使う。
会員番号とパスワードは本体の notify_config.json (gitignore 済み) に置く:
  "jrdb_user": "...", "jrdb_password": "..."
このスクリプトは値を表示しない。落としたファイルは data/jrdb/raw/ (git 管理外。JRDB 規約で再配布禁止)。

使い方 (root から):
  python scripts/jrdb_fetch.py probe                       # 一覧ページを読むだけ (何が何年分あるか)
  python scripts/jrdb_fetch.py download --types Skb Sed Tyb --from 2013 --to 2026

注意: レース当日 08:00-19:00 は前年以前のデータに制限がかかる (JRDB 案内 2025-07-28)。平日に実行する。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import requests

ROOT = Path(os.environ.get("PYCALIAI_DATA_ROOT", "E:/PyCaLiAI"))
RAW = ROOT / "data/jrdb/raw"
BASE = "http://www.jrdb.com/member/datazip/"
TYPES = ["Skb", "Sed", "Tyb"]
HREF = re.compile(r'href="([^"]+\.(?:zip|lzh))"', re.I)


def session():
    cfg = json.loads((ROOT / "notify_config.json").read_text(encoding="utf-8"))
    u, p = cfg.get("jrdb_user"), cfg.get("jrdb_password")
    if not (u and p):
        sys.exit("notify_config.json に jrdb_user / jrdb_password がありません")
    s = requests.Session()
    s.auth = (u, p)
    s.headers["User-Agent"] = "jrdb-fetch/1.0"
    return s


def list_links(s, typ):
    url = f"{BASE}{typ}/index.html"
    r = s.get(url, timeout=60)
    if r.status_code != 200:
        return url, r.status_code, []
    html = r.content.decode("cp932", errors="replace")
    return url, 200, sorted({urljoin(url, h) for h in HREF.findall(html)})


def year_of(link):
    name = link.rsplit("/", 1)[-1]
    m = re.search(r"_(\d{4})\.", name)                 # 年単位: SKB_2013.zip
    if m:
        return int(m.group(1)), True
    m = re.search(r"(\d{2})(\d{2})(\d{2})\.", name)    # 日単位: SKB130105.zip
    if m:
        yy = int(m.group(1))
        return (1900 + yy if yy >= 90 else 2000 + yy), False
    return None, False


def cmd_probe(args):
    s = session()
    for typ in args.types:
        url, st, links = list_links(s, typ)
        if st != 200:
            print(f"{typ}: {url} -> HTTP {st}")
            continue
        yearly = sorted(y for y, is_y in map(year_of, links) if is_y and y)
        daily = [y for y, is_y in map(year_of, links) if not is_y and y]
        ext = sorted({l.rsplit(".", 1)[-1].lower() for l in links})
        print(f"{typ}: リンク {len(links)} 本 ({'/'.join(ext)})  年単位 {len(yearly)} 本"
              f"{' ' + str(yearly[0]) + '-' + str(yearly[-1]) if yearly else ''}  日単位 {len(daily)} 本"
              f"{' ' + str(min(daily)) + '-' + str(max(daily)) if daily else ''}")
        for l in links[:2] + links[-2:]:
            print("   ", l.replace(BASE, ""))
        time.sleep(1)


def cmd_download(args):
    now = dt.datetime.now()
    if now.weekday() >= 5 and 8 <= now.hour < 19:
        sys.exit("レース当日 08-19 時は過去データに制限がかかるので中止")
    s = session()
    total = 0
    for typ in args.types:
        url, st, links = list_links(s, typ)
        if st != 200:
            print(f"{typ}: HTTP {st} — スキップ")
            continue
        info = [(l, *year_of(l)) for l in links]
        have_year = {y for _, y, is_y in info if is_y}
        want = []
        for l, y, is_y in info:
            if y is None or not (args.frm <= y <= args.to):
                continue
            if is_y or y not in have_year:              # 年単位があれば日単位は取らない
                want.append(l)
        zips = [l for l in want if l.lower().endswith(".zip")]
        stems = {l.rsplit(".", 1)[0] for l in zips}
        want = zips + [l for l in want if l.lower().endswith(".lzh") and l.rsplit(".", 1)[0] not in stems]
        out = RAW / typ
        out.mkdir(parents=True, exist_ok=True)
        n = b = 0
        for l in want:
            dst = out / l.rsplit("/", 1)[-1]
            if dst.exists() and dst.stat().st_size > 0:
                continue
            r = s.get(l, timeout=300)
            if r.status_code != 200:
                print(f"  {dst.name}: HTTP {r.status_code}")
                continue
            dst.write_bytes(r.content)
            n += 1
            b += len(r.content)
            time.sleep(args.sleep)
        total += b
        print(f"{typ}: 対象 {len(want)} 本, 新規 {n} 本, {b / 1e6:.1f} MB -> {out}")
    print(f"合計 {total / 1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--types", nargs="+", default=TYPES)
    p.set_defaults(fn=cmd_probe)
    d = sub.add_parser("download")
    d.add_argument("--types", nargs="+", default=TYPES)
    d.add_argument("--from", dest="frm", type=int, default=2013)
    d.add_argument("--to", type=int, default=dt.date.today().year)
    d.add_argument("--sleep", type=float, default=1.0)
    d.set_defaults(fn=cmd_download)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
