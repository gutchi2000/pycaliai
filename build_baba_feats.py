# -*- coding: utf-8 -*-
"""
build_baba_feats.py — JRA 馬場 PDF (クッション値 + 含水率) を (場所×日付) テーブル化
================================================================================
新観測量。data/baba/{年}/{年}{track}{回}.pdf を全件パース。

2フォーマット:
  旧 (2018-2024): 開催日節ブロック。"第N日…（YYYY年M月D日～D日）" + 曜日列 +
      [クッション値×k] + 場所 + 芝/ダ含水率(ゴール前/4コーナー)×k。
      クッション値は2020-09開始なので2018-19はクッション値行なし(=NaN)。
  新 (2025-): 1測定日1レコード。"M月D日 曜日 コース 時刻 クッション値 時刻 芝GP 芝4C ダGP ダ4C"。

★leak-safe: クッション値/含水率は土日早朝(発走前)測定 = 当日レース前に既知。

出力: data/baba_feats.parquet  (場所[日本語], 日付[YYYYMMDD], cushion, shiba_gp, shiba_4c, dirt_gp, dirt_4c)
実行: python build_baba_feats.py [--test]
"""
from __future__ import annotations
import argparse, glob, json, re, sys
from datetime import date, timedelta
from pathlib import Path

import fitz
import numpy as np
import pandas as pd

import io as _io
sys.stdout = _io.TextIOWrapper(open(1, "wb", closefd=False), encoding="utf-8", line_buffering=True)

BASE = Path(__file__).parent
PDF_DIR = BASE / "data/baba"
OUT = BASE / "data/baba_feats.parquet"

TRACK = {"sapporo": "札幌", "hakodate": "函館", "fukushima": "福島", "niigata": "新潟",
         "tokyo": "東京", "nakayama": "中山", "chukyo": "中京", "kyoto": "京都",
         "hanshin": "阪神", "kokura": "小倉"}
FLOAT = re.compile(r"^\d+\.\d+$")
DATE_RANGE = re.compile(r"(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日\s*[～~から]+\s*(?:(\d{1,2})月\s*)?(\d{1,2})日")
DATE_ROW = re.compile(r"(\d{1,2})月\s*(\d{1,2})日")
TIME = re.compile(r"^\d{1,2}:\d{2}$")
COL = ["場所", "日付", "cushion", "shiba_gp", "shiba_4c", "dirt_gp", "dirt_4c"]


def track_of(fn: str) -> str:
    m = re.search(r"[a-z]+", Path(fn).stem)
    return TRACK.get(m.group(0), m.group(0)) if m else "?"


def _floats_after(lines, i):
    out = []
    while i < len(lines) and FLOAT.match(lines[i]):
        out.append(float(lines[i])); i += 1
    return out


def _date_list(yr, mo, d1, mo2, d2):
    a = date(yr, mo, d1)
    b = date(yr, mo2 or mo, d2)
    if b < a:  # 年跨ぎ保険
        b = date(yr + 1, mo2 or mo, d2)
    return [a + timedelta(days=n) for n in range((b - a).days + 1)]


def parse_old(text, track):
    lines = [l.strip() for l in text.split("\n")]
    hdr = [(i, DATE_RANGE.search(l)) for i, l in enumerate(lines) if DATE_RANGE.search(l)]
    recs = []
    for bi, (hi, m) in enumerate(hdr):
        yr, mo, d1 = int(m.group(1)), int(m.group(2)), int(m.group(3))
        mo2 = int(m.group(4)) if m.group(4) else mo
        d2 = int(m.group(5))
        dates = _date_list(yr, mo, d1, mo2, d2)
        end = hdr[bi + 1][0] if bi + 1 < len(hdr) else len(lines)
        blk = lines[hi + 1:end]
        k = len(dates)
        # 全 float-run を位置付きで列挙 (クッション値の位置は年で違う:
        #   2020=ブロック末尾 / 2021-24=ヘッダ直後 → 位置非依存で拾う)
        runs_all = []
        j = 0
        while j < len(blk):
            if FLOAT.match(blk[j]):
                vals = _floats_after(blk, j)
                runs_all.append((j, vals)); j += max(len(vals), 1)
            else:
                j += 1
        marker_starts = {p + 1 for p in range(len(blk)) if blk[p] in ("ゴール前", "４コーナー", "4コーナー")}
        runs = [v for (s, v) in runs_all if s in marker_starts]   # 芝GP,芝4C,ダGP,ダ4C
        runs = (runs + [[], [], [], []])[:4]
        non_hum = [v for (s, v) in runs_all if s not in marker_starts]
        cushion = max(non_hum, key=lambda v: (len(v) == k, len(v)), default=[])  # 日数kに最も近い run
        for di, dt in enumerate(dates):
            def g(run):
                return run[di] if di < len(run) else np.nan
            recs.append({
                "場所": track, "日付": int(dt.strftime("%Y%m%d")),
                "cushion": (cushion[di] if di < len(cushion) else np.nan),
                "shiba_gp": g(runs[0]), "shiba_4c": g(runs[1]),
                "dirt_gp": g(runs[2]), "dirt_4c": g(runs[3]),
            })
    return recs


def parse_new(text, track):
    ym = re.search(r"(\d{4})年", text)
    yr = int(ym.group(1)) if ym else None
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    recs = []
    i = 0
    while i < len(lines):
        m = DATE_ROW.search(lines[i])
        if not m or "年" in lines[i]:   # タイトル行(YYYY年)除外
            i += 1; continue
        mo, da = int(m.group(1)), int(m.group(2))
        j = i + 1; floats = []
        while j < len(lines) and not (DATE_ROW.search(lines[j]) and "年" not in lines[j]):
            if FLOAT.match(lines[j]):
                floats.append(float(lines[j]))
            j += 1
        # floats = [cushion, 芝GP, 芝4C, ダGP, ダ4C] (cushionが無い日は4個)
        if len(floats) >= 5:
            cu, hum = floats[0], floats[1:5]
        elif len(floats) == 4:
            cu, hum = np.nan, floats
        else:
            i = j; continue
        if yr:
            recs.append({"場所": track, "日付": int(f"{yr}{mo:02d}{da:02d}"),
                         "cushion": cu, "shiba_gp": hum[0], "shiba_4c": hum[1],
                         "dirt_gp": hum[2], "dirt_4c": hum[3]})
        i = j
    return recs


def parse_pdf(path):
    track = track_of(path)
    d = fitz.open(path)
    text = "\n".join(d[i].get_text() for i in range(d.page_count))
    d.close()
    is_new = ("一覧" in text) or ("測定時刻" in text) or ("測定月日" in text)
    return parse_new(text, track) if is_new else parse_old(text, track)


def main(test=False):
    files = sorted(glob.glob(str(PDF_DIR / "**/*.pdf"), recursive=True))
    print(f"[scan] {len(files)} PDFs")
    all_recs = []
    errs = []
    for f in files:
        try:
            r = parse_pdf(f)
            if not r:
                errs.append((f, "0 records"))
            all_recs.extend(r)
        except Exception as e:
            errs.append((f, repr(e)))
    df = pd.DataFrame(all_recs, columns=COL)
    df = df.drop_duplicates(["場所", "日付"]).sort_values(["日付", "場所"]).reset_index(drop=True)

    if test:
        print("\n[validate] 2021東京 第1回 (1/29-31 cushion 期待 8.4/8.8/9.3):")
        print(df[(df["場所"] == "東京") & (df["日付"].between(20210129, 20210131))].to_string())
        print("\n[validate] 2025中山 (1/4 cushion 期待 10.1, 1/5=10.2):")
        print(df[(df["場所"] == "中山") & (df["日付"].between(20250104, 20250106))].to_string())

    print(f"\n[parsed] rows={len(df):,}  期間 {df['日付'].min()}-{df['日付'].max()}")
    cush = df["cushion"].notna()
    print(f"  cushion 非欠損={cush.sum():,} ({cush.mean()*100:.0f}%) 最古={df.loc[cush,'日付'].min()}")
    print(f"  含水率(芝GP) 非欠損={df['shiba_gp'].notna().sum():,}")
    print(f"  場所別行数:\n{df['場所'].value_counts().to_string()}")
    if errs:
        print(f"\n[warn] {len(errs)} ファイルで問題: {errs[:5]}")
    if not test:
        df.to_parquet(OUT, index=False)
        print(f"[saved] {OUT}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    raise SystemExit(main(test=a.test))
