#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
data/weekly/{date}.csv が Excel で開いて保存され壊れた場合の修復ツール。

Excel 保存で起きる破壊:
  1. レースID(新) が 16桁整数 -> 指数表記 "2.02606E+15" に丸められ全レース同値化
  2. 全行末尾に空カンマが追加され、全行がシート最大列数(47)に padding される
     -> parse_csv の 19列(レース行)/46列(馬行) 判定が全滅し races が空になる

修復内容:
  - 各行を行種別(レース行 / 馬行 / ヘッダ)で判定し、正規の幅(19 / 46)に truncate
    (Excel が足した末尾の空 padding を落とすだけ。本来データは前方に無傷で残る)
  - レース行のレースID(新) を 日付S + 場所コード + 開催回 + 日目 + R から再構成
    フォーマット: YYYYMMDD + track(2) + kai(2) + day(2) + R(2) = 16桁
    (TARGET が吐く正規IDと完全一致 -> 結果(kekka)突合も安全)

使い方:
    python scripts/repair_excel_weekly_csv.py data/weekly/20260620.csv
    python scripts/repair_excel_weekly_csv.py data/weekly/20260620.csv --inplace
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

# JRA 中央場コード
TRACK = {
    "札幌": "01", "函館": "02", "福島": "03", "新潟": "04", "東京": "05",
    "中山": "06", "中京": "07", "京都": "08", "阪神": "09", "小倉": "10",
}

DATE_RE = re.compile(r"^\d{4}\.\d{1,2}\.\d{1,2}$")
# 開催 "1函3" -> kai=1, day=3 (場所漢字を間に挟む)
KAISAI_RE = re.compile(r"^(\d+)\D+(\d+)$")

RACE_WIDTH = 19   # RACE_COLS
HORSE_WIDTH = 46  # HORSE_COLS_46（本番の TARGET 週次フォーマット）


def repair(text: str) -> tuple[str, dict]:
    out: list[str] = []
    stats = {"race": 0, "horse": 0, "header": 0, "blank": 0, "unknown": 0}
    for raw in text.split("\n"):
        line = raw.rstrip("\r")
        cols = line.split(",")
        if all(c == "" for c in cols):
            stats["blank"] += 1
            continue
        f0 = cols[0]
        f1 = cols[1] if len(cols) > 1 else ""

        # --- ヘッダ行 ---
        if f0 == "レースID(新)":
            out.append(",".join(cols[:RACE_WIDTH]))
            stats["header"] += 1
            continue
        if f0 == "枠番":
            out.append(",".join(cols[:HORSE_WIDTH]))
            stats["header"] += 1
            continue

        # --- レース情報行: 日付S(field1) が日付形式 ---
        if DATE_RE.match(f1):
            place = cols[3]
            kai_field = cols[4]
            r_field = cols[5]
            if place not in TRACK:
                raise ValueError(f"未知の場所コード: {place!r} (行: {line[:60]})")
            m = KAISAI_RE.match(kai_field)
            if not m:
                raise ValueError(f"開催欄をパースできません: {kai_field!r}")
            y, mo, d = f1.split(".")
            ymd = f"{int(y):04d}{int(mo):02d}{int(d):02d}"
            rid = f"{ymd}{TRACK[place]}{int(m.group(1)):02d}{int(m.group(2)):02d}{int(r_field):02d}"
            fixed = cols[:RACE_WIDTH]
            # padding で 19 未満になることは無いが念のため
            while len(fixed) < RACE_WIDTH:
                fixed.append("")
            fixed[0] = rid
            out.append(",".join(fixed))
            stats["race"] += 1
            continue

        # --- 馬行: 先頭が枠番(数字) ---
        if f0.isdigit():
            fixed = cols[:HORSE_WIDTH]
            while len(fixed) < HORSE_WIDTH:
                fixed.append("")
            out.append(",".join(fixed))
            stats["horse"] += 1
            continue

        stats["unknown"] += 1
        # 不明行はそのまま（落とさない）
        out.append(line)

    return "\n".join(out) + "\n", stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--inplace", action="store_true",
                    help="元ファイルを上書き（.bak が無ければ作成）")
    args = ap.parse_args()

    src: Path = args.path
    if not src.exists():
        print(f"ERROR: not found: {src}", file=sys.stderr)
        return 2

    text = src.read_bytes().decode("cp932")
    fixed_text, stats = repair(text)
    print(f"  race={stats['race']} horse={stats['horse']} "
          f"header={stats['header']} blank(dropped)={stats['blank']} "
          f"unknown={stats['unknown']}")

    if args.inplace:
        bak = src.with_suffix(src.suffix + ".pre_repair.bak")
        if not bak.exists():
            bak.write_bytes(src.read_bytes())
            print(f"  backup -> {bak}")
        src.write_bytes(fixed_text.encode("cp932"))
        print(f"  REPAIRED (inplace) -> {src}")
    else:
        dst = src.with_suffix(".repaired.csv")
        dst.write_bytes(fixed_text.encode("cp932"))
        print(f"  wrote -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
