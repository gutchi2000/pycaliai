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
  中身 15列・「確定着順」系 ............... data/kekka/{日付}.csv    (結果・払戻 通常版)
  中身 174列・払戻成績 ................... data/bias/{日付}.csv     (★土曜結果 → 実現バイアス自動生成)

  ※ 出走表(出走表)と過去5走は構造が完全同型で中身判定できないため、ファイル名の
     先頭1文字(S / K)で見分ける。エクスポート時か _inbox 投入時に S/K を付ける。
  ※ 174列の払戻成績を置くと build_realized_bias.py を自動実行し data/realized_bias.json を更新
     (=翌開催日の出走表タブに「実現バイアス」カードが出る)。

実行: PYTHONUTF8=1 ./venv311/Scripts/python.exe place_weekly.py [--dry]
   --dry: 移動せず、どこへ振り分けるかだけ表示。
"""
from __future__ import annotations
import csv, json, re, shutil, sys
from pathlib import Path

BASE = Path(__file__).parent
INBOX = BASE / "data" / "_inbox"
DRY = "--dry" in sys.argv

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
    """中身(ヘッダ)から種類を判定。'bias'(174列払戻) / 'kekka'(15列結果) / 'racelist'(19列) / None。"""
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
    if n < 30 and "レースID(新)" in h and "クラス名" in h:
        return "racelist"        # 出走表/過去5走 — S/K プレフィクスが要る
    return None


def route(path: Path):
    """(行き先Path, run_realized, 注記) を返す。行き先 None = 仕分け不能。"""
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
    t = detect_content(path)
    if t == "bias":
        d = extract_date(name, path)
        return (BASE / "data" / "bias" / f"{d}.csv") if d else None, True, "払戻成績→実現バイアス"
    if t == "kekka":
        d = extract_date(name, path)
        return (BASE / "data" / "kekka" / f"{d}.csv") if d else None, False, "結果(kekka)"
    if t == "racelist":
        return None, False, "⚠ 出走表/過去5走は見分け不能。S/K プレフィクスを付けて"
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
