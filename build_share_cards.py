# -*- coding: utf-8 -*-
"""
build_share_cards.py — レース共有カード（固定 URL ページ + OG 画像）
====================================================================
SPA の index.html は共有すると表紙しか見えない。各レースに
  https://pycaliai.com/r/{race_id16}      (site/r/{rid}.html)
  https://pycaliai.com/og/{race_id16}.png (site/og/{rid}.png, 1200x630)
を持たせ、X 等に貼ると印・勝率がカードで出るようにする。人間が開くと
/#race={rid} へ即リダイレクトし app.js がそのレースを直接開く。

入力は公開済み site/data/{date}.json だけ（オッズ・EV・払戻は元から無い）。
build_site.py の末尾から呼ばれる。単体: python build_share_cards.py [YYYYMMDD ...]

生成方針
  - HTML は全開催日ぶん（決定的な内容なので変更が無ければ git 差分ゼロ）。
  - PNG は「結果待ちの開催日 + 結果ありの直近 2 開催日」だけ生成し、それ以外は削除
    （リポジトリ肥大の抑制。古いレースの HTML は og/default.png を指す）。
  - フォントは Windows 同梱の日本語ゴシック。見つからなければ PNG は default のみ。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"
DATA = SITE / "data"
R_DIR = SITE / "r"
OG_DIR = SITE / "og"
BASE_URL = "https://pycaliai.com"

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\BIZ-UDGothicB.ttc",
    r"C:\Windows\Fonts\YuGothB.ttc",
    r"C:\Windows\Fonts\meiryo.ttc",
    r"C:\Windows\Fonts\NotoSansJP-VF.ttf",
]
FONT_REG_CANDIDATES = [
    r"C:\Windows\Fonts\BIZ-UDGothicR.ttc",
    r"C:\Windows\Fonts\YuGothM.ttc",
    r"C:\Windows\Fonts\meiryo.ttc",
]
W, H = 1200, 630
BG = (7, 11, 20)
CARD = (13, 20, 36)
LINE = (30, 43, 74)
GOLD = (245, 185, 66)
GOLD2 = (255, 217, 122)
TX = (237, 241, 251)
TX2 = (169, 182, 211)
TX3 = (115, 133, 168)
TEAL = (45, 212, 168)
RED = (242, 85, 90)
MARK_ORDER = {"◎": 0, "〇": 1, "○": 1, "▲": 2, "△": 3}
WD = ["月", "火", "水", "木", "金", "土", "日"]


def _esc(s) -> str:
    return (str(s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def _pct(v, d=1) -> str:
    try:
        return f"{float(v) * 100:.{d}f}"
    except (TypeError, ValueError):
        return "—"


def _date_label(d8: str) -> str:
    import datetime as _dt
    t = _dt.date(int(d8[:4]), int(d8[4:6]), int(d8[6:8]))
    return f"{t.month}/{t.day}({WD[t.weekday()]})"


def _marked(race: dict) -> list[dict]:
    hs = [h for h in race.get("horses") or [] if (h.get("mark") or "") in MARK_ORDER]
    hs.sort(key=lambda h: (MARK_ORDER[h["mark"]], h.get("ai_rank") or 99))
    return hs[:5]


def race_title(race: dict) -> str:
    bits = [f"{race.get('place','')} {race.get('rno','')}R", race.get("course") or "", race.get("klass") or ""]
    return " ".join(b for b in bits if b)


def _result_line(race: dict) -> str | None:
    res = race.get("result") or {}
    top3 = res.get("top3") or []
    if not top3:
        return None
    by_uma = {int(h["umaban"]): h for h in race.get("horses") or []}
    names = []
    for i, u in enumerate(top3[:3], 1):
        h = by_uma.get(int(u))
        names.append(f"{i}着 {h['name']}" if h else f"{i}着 {u}番")
    return " / ".join(names)


def _honmei_pos(race: dict) -> int | None:
    res = (race.get("result") or {}).get("order") or {}
    for h in race.get("horses") or []:
        if h.get("mark") == "◎":
            v = res.get(str(h["umaban"]))
            return int(v) if v else None
    return None


def description(day: dict, race: dict) -> str:
    parts = []
    for h in _marked(race)[:3]:
        parts.append(f"{h['mark']}{h['name']}（勝率{_pct(h.get('p_win'))}%）")
    j = race.get("judgment") or {}
    s = f"{_date_label(day['date'])} {race_title(race)} の AI 予想。" + " ".join(parts)
    if j.get("hardness"):
        s += f" 判定: {j['hardness']}"
    if j.get("headline"):
        s += f"／{j['headline']}"
    rl = _result_line(race)
    if rl:
        s += f" 結果: {rl}"
        hp = _honmei_pos(race)
        if hp:
            s += f"（◎は{hp}着）"
    return s[:300]


# ------------------------------------------------------------------ PNG
_FONT_CACHE: dict = {}


def _font(size: int, bold: bool = True):
    from PIL import ImageFont
    key = (size, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    for p in (FONT_CANDIDATES if bold else FONT_REG_CANDIDATES):
        try:
            f = ImageFont.truetype(p, size)
            _FONT_CACHE[key] = f
            return f
        except Exception:
            continue
    raise FileNotFoundError("日本語フォントが見つかりません")


def fonts_available() -> bool:
    try:
        _font(20)
        return True
    except Exception:
        return False


def _txt(draw, xy, s, size, fill, bold=True, anchor="la", maxw=None):
    f = _font(size, bold)
    if maxw:
        while s and draw.textlength(s, font=f) > maxw:
            s = s[:-1]
        if maxw and s and draw.textlength(s + "…", font=f) > maxw and len(s) > 1:
            s = s[:-1] + "…"
    draw.text(xy, s, font=f, fill=fill, anchor=anchor)


def render_png(day: dict, race: dict, out: Path) -> None:
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    # 左のゴールド帯 + 下地カード
    d.rectangle([0, 0, 14, H], fill=GOLD)
    d.rounded_rectangle([44, 150, W - 44, H - 92], radius=22, fill=CARD, outline=LINE, width=2)
    # ヘッダ
    _txt(d, (48, 44), "PyCaLiAI", 40, GOLD)
    _txt(d, (262, 54), "AI 競馬予想", 26, TX2, bold=False)
    _txt(d, (W - 48, 54), _date_label(day["date"]), 28, TX2, bold=False, anchor="ra")
    # タイトル
    _txt(d, (48, 100), race_title(race), 44, TX, maxw=W - 96 - (0 if not race.get("race_name") else 0))
    if race.get("race_name"):
        _txt(d, (W - 48, 108), race["race_name"], 30, GOLD2, anchor="ra", maxw=520)
    # 印リスト
    marked = _marked(race)
    y = 176
    row_h = 57   # 5 行で 176+285=461 < 結果行の区切り (H-150=480)
    for h in marked:
        m = h["mark"]
        col = GOLD if m == "◎" else (TX if m in ("〇", "○") else (TX2 if m == "▲" else TX3))
        d.ellipse([72, y + 3, 72 + 46, y + 49], fill=(18, 28, 49), outline=col, width=3)
        _txt(d, (95, y + 26), "○" if m == "〇" else m, 28, col, anchor="mm")
        _txt(d, (140, y + 6), h.get("name") or "", 36, TX, maxw=470)
        _txt(d, (640, y + 12), f"勝率 {_pct(h.get('p_win'))}%", 27, GOLD2 if m == "◎" else TX2)
        _txt(d, (880, y + 12), f"3着内 {_pct(h.get('p_sho'), 0)}%", 27, TX2, bold=False)
        y += row_h
    if not marked:
        _txt(d, (142, 200), "印なし（見送り / 判定不能）", 34, TX3)
    # 右上の判定チップ
    j = race.get("judgment") or {}
    if j.get("hardness"):
        chip = f"{j['hardness']}"
        f = _font(26)
        tw = d.textlength(chip, font=f)
        x1 = W - 72
        d.rounded_rectangle([x1 - tw - 36, 170, x1, 212], radius=21, outline=GOLD, width=2)
        _txt(d, (x1 - 18 - tw / 2, 191), chip, 26, GOLD, anchor="mm")
    # 結果（あれば）
    rl = _result_line(race)
    if rl:
        d.line([72, H - 150, W - 72, H - 150], fill=LINE, width=2)
        hp = _honmei_pos(race)
        tag = f"結果  {rl}" + (f"　◎{hp}着" if hp else "")
        col = TEAL if (hp and hp <= 3) else (RED if hp else TX2)
        _txt(d, (72, H - 140), tag, 28, col, maxw=W - 144)
    # フッタ
    _txt(d, (48, H - 62), "pycaliai.com", 30, GOLD)
    _txt(d, (W - 48, H - 56), "AI 予測は参考情報です。的中・利益を保証しません。", 20, TX3, bold=False, anchor="ra")
    out.parent.mkdir(parents=True, exist_ok=True)
    img.convert("P", palette=Image.ADAPTIVE, colors=96).save(out, optimize=True)


def render_default_png(out: Path) -> None:
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 14, H], fill=GOLD)
    _txt(d, (W / 2, 250), "PyCaLiAI", 110, GOLD, anchor="mm")
    _txt(d, (W / 2, 350), "AI 競馬予想 — 印・勝率・回顧・通信簿", 40, TX2, anchor="mm", bold=False)
    _txt(d, (W / 2, 440), "pycaliai.com", 34, TX, anchor="mm")
    out.parent.mkdir(parents=True, exist_ok=True)
    img.convert("P", palette=Image.ADAPTIVE, colors=64).save(out, optimize=True)


# ------------------------------------------------------------------ HTML
HTML_TPL = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — PyCaLiAI AI予想</title>
<meta name="description" content="{desc}">
<link rel="canonical" href="{url}">
<meta property="og:type" content="article">
<meta property="og:site_name" content="PyCaLiAI">
<meta property="og:title" content="{title} — AI予想">
<meta property="og:description" content="{desc}">
<meta property="og:url" content="{url}">
<meta property="og:image" content="{img}">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{title} — AI予想">
<meta name="twitter:description" content="{desc}">
<meta name="twitter:image" content="{img}">
<meta http-equiv="refresh" content="0;url=/#race={rid}">
<link rel="icon" href="/favicon.ico" sizes="any">
<style>body{{background:#070b14;color:#edf1fb;font-family:system-ui,sans-serif;display:grid;place-items:center;height:100vh;margin:0}}a{{color:#f5b942}}</style>
<script>location.replace("/#race={rid}");</script>
</head>
<body><p>{title} の AI 予想を開いています… <a href="/#race={rid}">開かない場合はこちら</a></p></body>
</html>
"""


def write_html(day: dict, race: dict, has_png: bool) -> bool:
    rid = str(race["race_id"])
    url = f"{BASE_URL}/r/{rid}"
    img = f"{BASE_URL}/og/{rid}.png" if has_png else f"{BASE_URL}/og/default.png"
    html = HTML_TPL.format(title=_esc(race_title(race) + (f" {race['race_name']}" if race.get("race_name") else "")),
                           desc=_esc(description(day, race)), url=url, img=img, rid=rid)
    p = R_DIR / f"{rid}.html"
    if p.exists() and p.read_bytes() == html.encode("utf-8"):
        return False
    p.write_bytes(html.encode("utf-8"))   # LF 固定 (Windows の text mode で CRLF にしない)
    return True


# ------------------------------------------------------------------ main
def build(manifest_entries: list[dict], hot_results: int = 2) -> dict:
    """manifest_entries: [{date, has_results, ...}] 新しい順。"""
    R_DIR.mkdir(parents=True, exist_ok=True)
    OG_DIR.mkdir(parents=True, exist_ok=True)
    have_font = fonts_available()
    if have_font and not (OG_DIR / "default.png").exists():
        render_default_png(OG_DIR / "default.png")
    dates = [e["date"] for e in manifest_entries]
    hot = {e["date"] for e in manifest_entries if not e.get("has_results")}
    hot |= set([e["date"] for e in sorted(manifest_entries, key=lambda e: e["date"], reverse=True)
                if e.get("has_results")][:hot_results])
    stats = {"html_written": 0, "html_total": 0, "png_written": 0, "png_removed": 0, "font": have_font}
    keep_png: set[str] = {"default.png"}
    for date in dates:
        p = DATA / f"{date}.json"
        if not p.exists():
            continue
        day = json.loads(p.read_text(encoding="utf-8"))
        for race in day.get("races") or []:
            rid = str(race["race_id"])
            png = OG_DIR / f"{rid}.png"
            if have_font and date in hot:
                try:
                    render_png(day, race, png)
                    stats["png_written"] += 1
                except Exception as e:  # 画像失敗は HTML を default 画像で続行
                    print(f"  [share_card png skip] {rid}: {e}")
            has_png = have_font and date in hot and png.exists()
            if has_png:
                keep_png.add(png.name)
            stats["html_total"] += 1
            if write_html(day, race, has_png):
                stats["html_written"] += 1
    # hot 以外の PNG は削除（リポジトリ肥大の抑制）
    for f in OG_DIR.glob("*.png"):
        if f.name not in keep_png:
            f.unlink()
            stats["png_removed"] += 1
    return stats


def main() -> int:
    mf = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    entries = mf.get("dates") or []
    if len(sys.argv) > 1:
        only = set(sys.argv[1:])
        entries = [e for e in entries if e["date"] in only]
    st = build(entries)
    print(f"share cards: html {st['html_written']}/{st['html_total']} 更新, png {st['png_written']} 生成 / "
          f"{st['png_removed']} 削除, font={'OK' if st['font'] else '無し'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
