# -*- coding: utf-8 -*-
"""
build_sitemap.py — site/sitemap.xml を生成する (2026-11-04、build_site.py 末尾から呼ばれる)
================================================================================
静的ページ + レース共有ページ /r/{rid16} (全開催日) + 開催日ページ /r/d/{date} + 馬ページ
(horse?name=、Worker が OG を差し込む canonical) + 騎手/調教師ページ。
URL 数は 1 ファイル 50,000 件が上限。超えたら sitemap index に分割する (今は 2026 で ~12,000)。
"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote
from xml.sax.saxutils import escape

BASE = Path(__file__).resolve().parent
SITE = BASE / "site"
ORIGIN = "https://pycaliai.com"

STATIC = [  # (path, changefreq, priority)
    ("/", "daily", "1.0"), ("/explain.html", "weekly", "0.7"), ("/kaiko.html", "daily", "0.8"),
    ("/tsushinbo.html", "weekly", "0.7"), ("/shusse.html", "weekly", "0.6"), ("/shusse_check.html", "monthly", "0.4"),
    ("/horse.html", "weekly", "0.6"), ("/jt.html", "weekly", "0.6"), ("/legal.html", "yearly", "0.2"),
]


def _lastmod(d8: str) -> str:
    return f"{d8[:4]}-{d8[4:6]}-{d8[6:8]}"


def build() -> dict:
    mf = json.loads((SITE / "data" / "manifest.json").read_text(encoding="utf-8"))
    dates = [d["date"] for d in mf.get("dates", [])]
    urls: list[str] = []

    def add(loc: str, lastmod: str | None = None, freq: str | None = None, prio: str | None = None):
        parts = [f"<loc>{escape(loc)}</loc>"]
        if lastmod:
            parts.append(f"<lastmod>{lastmod}</lastmod>")
        if freq:
            parts.append(f"<changefreq>{freq}</changefreq>")
        if prio:
            parts.append(f"<priority>{prio}</priority>")
        urls.append("<url>" + "".join(parts) + "</url>")

    for p, f, pr in STATIC:
        add(ORIGIN + p, None, f, pr)
    n_day = n_race = n_horse = n_jt = 0
    for d in dates:
        if (SITE / "r" / "d" / f"{d}.html").exists():
            add(f"{ORIGIN}/r/d/{d}", _lastmod(d), "weekly" if d >= dates[0] else "yearly", "0.5"); n_day += 1
    for p in sorted((SITE / "r").glob("????????????????.html")):
        rid = p.stem
        add(f"{ORIGIN}/r/{rid}", _lastmod(rid[:8]), None, "0.4"); n_race += 1
    idx = SITE / "data" / "h" / "_index.json"
    if idx.exists():
        for name, n_runs, last in json.loads(idx.read_text(encoding="utf-8")):
            add(f"{ORIGIN}/horse?name={quote(str(name))}", _lastmod(str(last)) if str(last).isdigit() else None, None, "0.3"); n_horse += 1
    jt = SITE / "data" / "jt" / "_index.json"
    if jt.exists():
        for row in json.loads(jt.read_text(encoding="utf-8")):
            name, kind = row[0], (row[3] if len(row) > 3 else "j")
            add(f"{ORIGIN}/jt.html?{'j' if kind == 'j' else 't'}={quote(str(name))}", None, "weekly", "0.3"); n_jt += 1
    if len(urls) > 49000:
        raise RuntimeError(f"sitemap URL 数 {len(urls)} が上限に近い → sitemap index への分割が必要")
    xml = ('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
           + "\n".join("  " + u for u in urls) + "\n</urlset>\n")
    out = SITE / "sitemap.xml"
    data = xml.encode("utf-8")
    written = not out.exists() or out.read_bytes() != data
    if written:
        out.write_bytes(data)
    return {"urls": len(urls), "days": n_day, "races": n_race, "horses": n_horse, "jt": n_jt, "written": written}


if __name__ == "__main__":
    print(build())
