# -*- coding: utf-8 -*-
"""
place_weekly.py — 週次 TARGET エクスポートを data/_inbox/ から各サブフォルダへ自動仕分け。
================================================================================
「どれをどこに置くか分からんくなる」対策。1フォルダ(data/_inbox/)に全部放り込んで
これを1回走らせると、ファイル名と中身を見て data/ の正しい場所へ振り分ける。

振り分けルール:
  ファイル名 H-*.csv / W-*.csv ............ data/training/   (坂路/WC調教・名前そのまま)
  ファイル名 OD*.CSV ...................... data/odds/       (オッズ・名前そのまま)
  ファイル名 S<日付>.csv ................. data/weekly/{日付}.csv   (★出走表 = S プレフィクス)
  ファイル名 K<日付>.csv ................. data/kako5/{日付}.csv    (★過去5走 = K プレフィクス)
  ファイル名 T<日付>.csv ................. data/tyaku/{日付}.csv    (★着度数 = T プレフィクス)
  中身 15列・「確定着順」系 ............... data/kekka/{日付}.csv    (結果・払戻 通常版)
  中身 174列・払戻成績 ................... data/bias/{日付}.csv     (★土曜結果 → 実現バイアス自動生成)
  中身 ワイド払戻(組-配当が並ぶ列) ....... data/kekka/wide_kekka.csv へ **追記** (★2026-08-25 追加)
  ファイル名「出走馬分析<日付>.csv」....... data/bunseki/{日付}.csv  (★serve回収用 新形式、2026-09追加)

  ※ 出走表/過去5走/着度数 はレースヘッダが同型 (19列) だが、馬行の列数が違う
     (出走表=46 or 99 / 着度数=53-60 / 過去5走=72)。プレフィクス無しでも
     馬行列数から自動判別する (実測 2026-07-03)。判別不能時のみ S/K/T を付けて再実行。
     着度数は serve の horse_fuku_* 特徴 (predict_weekly.parse_csv) の供給源 = 必須ファイル。
  ※ 174列の払戻成績を置くと build_realized_bias.py を自動実行し data/realized_bias.json を更新
     (=翌開催日の出走表タブに「実現バイアス」カードが出る)。
  ※ 出走馬分析 (2026-09 発見) はヘッダ行あり(先頭列 "No.")・列数は可変(TARGET側で
     項目を増減できる)。中身判定は "馬齢斤量差"/"前場所" 列の有無で行う。
     馬齢斤量差・前走馬体重(増減)・前走出走頭数・前走場所・母馬・毛色・生産者・馬主・
     騎手/調教師年齢・トラックコード(JV) 等、serve 側で長らく死んでいた特徴の
     TARGET ネイティブな供給源。parse_bunseki.py がパースする。まだ本番 parse_csv には
     未配線 (2026-09-05 時点、枠確定後データでの検証待ち)。

実行: PYTHONUTF8=1 ./venv311/Scripts/python.exe place_weekly.py [--dry]
   --dry: 移動せず、どこへ振り分けるかだけ表示。
"""
from __future__ import annotations
import csv, io, json, re, shutil, sys
from pathlib import Path

BASE = Path(__file__).parent
INBOX = BASE / "data" / "_inbox"
WIDE_KEKKA = BASE / "data" / "kekka" / "wide_kekka.csv"
DRY = "--dry" in sys.argv

# ワイド払戻の1セル: "05-07 \250 (3)" のような 組-配当(-人気) の並び。
WIDE_PAIR_RE = re.compile(r"\d+\s*[-―]\s*\d+\s*[\\¥￥]\s*\d+")

DATE_RE = re.compile(r"(20\d{6})")
RID16_RE = re.compile(r"^(20\d{6})\d{8}$")
YMD_DOT_RE = re.compile(r"^(20\d\d)\.(\d{1,2})\.(\d{1,2})$")


def log(m): print(m, flush=True)


def extract_date(name: str, path: Path) -> str | None:
    """ファイル名→中身 の順で YYYYMMDD を拾う。"""
    m = DATE_RE.search(name)
    if m:
        return m.group(1)
    try:
        with open(path, encoding="cp932", errors="replace") as f:
            r = csv.reader(f)
            next(r, None)
            row = next(r, [])
    except Exception:
        return None
    for c in row:                                   # 16桁レースID の先頭8桁
        mm = RID16_RE.match(c.strip())
        if mm:
            return mm.group(1)
    for c in row:                                   # 「2026.6.28」形式
        mm = YMD_DOT_RE.match(c.strip())
        if mm:
            return f"{mm.group(1)}{int(mm.group(2)):02d}{int(mm.group(3)):02d}"
    return None


def detect_content(path: Path) -> str | None:
    """中身(ヘッダ)から種類を判定。'bias'(174列払戻) / 'kekka'(15列結果) /
    'bunseki'(出走馬分析, ヘッダ行あり) / 'racelist'(19列) / None。"""
    try:
        with open(path, encoding="cp932", errors="replace") as f:
            header = next(csv.reader(f))
    except Exception:
        return None
    h = {c.strip() for c in header}
    n = len(header)
    if n >= 100 and ("1着馬番" in h or "馬連配当" in h or "３連単配当" in h):
        return "bias"
    if "確定着順" in h:
        return "kekka"
    # 出走馬分析 (2026-09 発見): 1行目が本物のヘッダで先頭列 "No."、
    # "馬齢斤量差"/"前場所" 等 serve 回収対象の特徴を含む。列数は
    # ユーザーがTARGET側で項目を増減するため固定せず、識別列の有無で判定する。
    if header and header[0].strip() == "No." and "馬齢斤量差" in h and "前場所" in h:
        return "bunseki"
    if n < 30 and "レースID(新)" in h and "クラス名" in h:
        return "racelist"        # 出走表/過去5走/着度数 — 馬行列数で二次判定
    return None


def detect_racelist_kind(path: Path) -> str | None:
    """racelist 系 (レースヘッダ19列) の種類を馬行の列数から判定。
    出走表=46列(99列形式も) / 着度数=53-60列(週により+2程度揺れる) / 過去5走=72列。
    """
    from collections import Counter
    cnt: Counter = Counter()
    try:
        with open(path, encoding="cp932", errors="replace") as f:
            for row in csv.reader(f):
                if len(row) > 19 and row[0].strip().isdigit():
                    cnt[len(row)] += 1
                if sum(cnt.values()) >= 200:
                    break
    except Exception:
        return None
    if not cnt:
        return None
    n = cnt.most_common(1)[0][0]
    if n in (46, 99):
        return "weekly"
    if n == 72:
        return "kako5"
    if 50 <= n <= 60:
        return "tyaku"
    return None


def _decode(raw: bytes) -> tuple[str, str]:
    """(text, encoding)。TARGET は cp932、稀に utf-8(BOM付) で出る。"""
    for enc in ("cp932", "utf-8-sig", "utf-8"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return raw.decode("cp932", errors="replace"), "cp932"


def _wide_rows(text: str) -> list[list[str]]:
    """ワイド払戻CSVの行を返す。組-配当セルを持つ行だけ拾う。"""
    out = []
    for row in csv.reader(io.StringIO(text)):
        if len(row) < 10:
            continue
        if any(WIDE_PAIR_RE.search(c or "") for c in row[5:]):
            out.append(row)
    return out


def detect_wide(path: Path) -> bool:
    """ワイド払戻エクスポートか。ヘッダ無しなので中身のパターンで判定する。"""
    try:
        text, _ = _decode(path.read_bytes())
    except OSError:
        return False
    rows = _wide_rows(text)
    total = sum(1 for _ in csv.reader(io.StringIO(text)))
    # 大半の行が組-配当を持っていれば ワイド払戻ファイル。
    return bool(rows) and total and len(rows) >= max(3, total * 0.5)


def _wide_key(row: list[str]) -> tuple:
    """(年, 月, 日, 場所, R) を照合キーにする。ゼロ埋め揺れを吸収。"""
    def n(v):
        v = str(v).strip()
        return str(int(v)) if v.lstrip("-").isdigit() else v
    return tuple(n(row[i]) for i in range(5))


def ingest_wide_kekka(path: Path) -> tuple[int, int, list[str]]:
    """ワイド払戻を data/kekka/wide_kekka.csv へ重複なく追記。
    戻り値: (追記行数, 重複スキップ数, 追記された日付 YYYYMMDD)。"""
    src_text, _ = _decode(path.read_bytes())
    new_rows = _wide_rows(src_text)
    if not new_rows:
        return 0, 0, []

    if WIDE_KEKKA.exists():
        cur_text, enc = _decode(WIDE_KEKKA.read_bytes())
    else:
        cur_text, enc = "", "cp932"
    have = {_wide_key(r) for r in _wide_rows(cur_text)}

    add, dup, dates = [], 0, set()
    for row in new_rows:
        key = _wide_key(row)
        if key in have:
            dup += 1
            continue
        have.add(key)
        add.append(row)
        try:
            dates.add(f"{int(row[0]):04d}{int(row[1]):02d}{int(row[2]):02d}")
        except (TypeError, ValueError):
            pass
    if not add or DRY:
        return len(add), dup, sorted(dates)

    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(add)
    WIDE_KEKKA.parent.mkdir(parents=True, exist_ok=True)
    tail = buf.getvalue()
    if cur_text and not cur_text.endswith("\n"):
        tail = "\n" + tail
    with open(WIDE_KEKKA, "ab") as f:
        f.write(tail.encode(enc, errors="replace"))
    return len(add), dup, sorted(dates)


def route(path: Path):
    """(行き先Path, run_realized, 注記) を返す。行き先 None = 仕分け不能。
    行き先が WIDE_KEKKA のときは移動でなく追記 (main が特別扱い)。"""
    name = path.name
    low = name.lower()
    if re.match(r"^[HW]-", name):
        return BASE / "data" / "training" / name, False, "調教"
    if low.startswith("od"):
        return BASE / "data" / "odds" / name, False, "オッズ"
    if re.match(r"^[Ss]\d", name):                          # ★出走表
        d = extract_date(name[1:], path)
        return (BASE / "data" / "weekly" / f"{d}.csv") if d else None, False, "出走表(S)"
    if re.match(r"^[Kk]\d", name):                          # ★過去5走
        d = extract_date(name[1:], path)
        return (BASE / "data" / "kako5" / f"{d}.csv") if d else None, False, "過去5走(K)"
    if re.match(r"^[Tt]\d", name):                          # ★着度数
        d = extract_date(name[1:], path)
        return (BASE / "data" / "tyaku" / f"{d}.csv") if d else None, False, "着度数(T)"
    t = detect_content(path)
    if t is None and detect_wide(path):     # ★ワイド払戻 (ヘッダ無し = 既知形式の後に判定)
        return WIDE_KEKKA, False, "ワイド払戻→wide_kekka.csv 追記"
    if t == "bias":
        d = extract_date(name, path)
        return (BASE / "data" / "bias" / f"{d}.csv") if d else None, True, "払戻成績→実現バイアス"
    if t == "kekka":
        d = extract_date(name, path)
        return (BASE / "data" / "kekka" / f"{d}.csv") if d else None, False, "結果(kekka)"
    if t == "bunseki":
        d = extract_date(name, path)
        return (BASE / "data" / "bunseki" / f"{d}.csv") if d else None, False, "出走馬分析"
    if t == "racelist":
        kind = detect_racelist_kind(path)
        d = extract_date(name, path)
        if kind and d:
            sub, label = {"weekly": ("weekly", "出走表(自動判別)"),
                          "kako5": ("kako5", "過去5走(自動判別)"),
                          "tyaku": ("tyaku", "着度数(自動判別)")}[kind]
            return BASE / "data" / sub / f"{d}.csv", False, label
        return None, False, "⚠ 出走表/過去5走/着度数を判別できず。S/K/T プレフィクスを付けて"
    return None, False, "⚠ 種類判定不能"


def main():
    if not INBOX.exists():
        INBOX.mkdir(parents=True, exist_ok=True)
        log(f"[作成] {INBOX} — ここに週次ファイルを放り込んで再実行してください。")
        return
    files = [p for p in sorted(INBOX.iterdir())
             if p.is_file() and p.suffix.lower() == ".csv"]
    if not files:
        log(f"data/_inbox/ に CSV がありません。")
        return

    log("=" * 70)
    log(f"週次仕分け {'(DRY-RUN)' if DRY else ''}  対象 {len(files)} 件")
    log("=" * 70)
    realized_targets, skipped = [], []
    for p in files:
        dest, run_realized, note = route(p)
        if dest is None:
            log(f"  ❌ {p.name:28s}  {note}")
            skipped.append(p.name)
            continue
        rel = dest.relative_to(BASE)
        if dest == WIDE_KEKKA:                              # ★追記型 (移動しない)
            n_add, n_dup, dates = ingest_wide_kekka(p)
            log(f"  → {p.name:28s} → {rel} [追記]   "
                f"[{note}] +{n_add}行 / 重複{n_dup} {dates or ''}")
            if not DRY:
                if n_add:
                    done_dir = INBOX / "_ingested"
                    done_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(p), str(done_dir / p.name))
                else:
                    log(f"     ⚠ 新規行なし。{p.name} は data/_inbox/ に残置")
                    skipped.append(p.name)
            continue
        over = " (上書き)" if dest.exists() else ""
        log(f"  → {p.name:28s} → {rel}{over}   [{note}]")
        if DRY:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(dest))
        if run_realized:
            realized_targets.append(dest)

    # 174列払戻成績が入ったら実現バイアスを再生成
    for dest in realized_targets:
        try:
            import build_realized_bias as brb
            payload = brb.build(str(dest))
            brb.OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
            log(f"\n[実現バイアス] {brb.OUT.relative_to(BASE)} 更新 "
                f"({payload['label']} → show_on={payload['show_on_date']}, {len(payload['venues'])}セグメント)")
        except Exception as e:
            log(f"\n⚠ 実現バイアス生成失敗: {e}")

    log("\n" + "=" * 70)
    if skipped:
        log(f"未仕分け {len(skipped)} 件は data/_inbox/ に残置: {', '.join(skipped)}")
    if realized_targets and not DRY:
        log("※ サイト反映は build_site.py / sync-hf-umami.ps1 で（週次フロー内）。")
    log("done.")


if __name__ == "__main__":
    main()
