"""
jrdb_parse.py — JRDB 固定長データ (SKB / TYB / SED) を表に変換する

入力: data/jrdb/raw/{Skb,Tyb,Sed}/*.zip   (scripts/jrdb_fetch.py が落としたもの)
出力: data/jrdb/{skb,tyb,sed}.parquet       (git 管理外。JRDB 規約で再配布禁止)
仕様: http://www.jrdb.com/program/{Skb/skb,Tyb/tyb,Sed/sed}_doc.txt  (バイト位置は cp932 のバイト単位)

実行 (root から): PYTHONUTF8=1 python scripts/jrdb_parse.py
"""
from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("PYCALIAI_DATA_ROOT", "E:/PyCaLiAI"))
RAW = ROOT / "data/jrdb/raw"
OUT = ROOT / "data/jrdb"

# (列名, 開始位置(1始まり), バイト数, 型)  型: s=文字列 / n=数値 / c=コード文字列(空白→NA)
KEY = [("racekey", 1, 8, "s"), ("ban", 9, 2, "n")]
SPEC = {
    "Skb": (304, KEY + [("ped", 11, 8, "s"), ("ymd", 19, 8, "n")]
            + [(f"tokki_{i + 1}", 27 + 3 * i, 3, "c") for i in range(6)]
            + [(f"bagu_{i + 1}", 45 + 3 * i, 3, "c") for i in range(8)]
            + [(f"ashi_sogo_{i + 1}", 69 + 3 * i, 3, "c") for i in range(3)]
            + [("paddock_comment", 114, 40, "s"), ("ashi_comment", 154, 40, "s"),
               ("bagu_comment", 194, 40, "s"), ("race_comment", 234, 40, "s"),
               ("hami", 274, 3, "c"), ("bandage", 277, 3, "c"), ("teitetsu", 280, 3, "c"),
               ("tei_jotai", 283, 3, "c"), ("soe", 286, 3, "c"), ("kotsuryu", 289, 3, "c")]),
    "Tyb": (128, KEY + [("idm", 11, 5, "n"), ("jockey_idx", 16, 5, "n"), ("info_idx", 21, 5, "n"),
                        ("odds_idx", 26, 5, "n"), ("paddock_idx", 31, 5, "n"), ("sogo_idx", 41, 5, "n"),
                        ("bagu_change", 46, 1, "n"), ("ashi_info", 47, 1, "n"), ("torikeshi", 48, 1, "n"),
                        ("tan_odds", 73, 6, "n"), ("fuku_odds", 79, 6, "n"), ("odds_time", 85, 4, "c"),
                        ("bw", 89, 3, "n"), ("bw_diff", 92, 3, "n"), ("odds_mark", 95, 1, "c"),
                        ("paddock_mark", 96, 1, "c"), ("chokuzen_mark", 97, 1, "c"),
                        ("batai_cd", 98, 1, "c"), ("kehai_cd", 99, 1, "c"), ("hassou", 100, 4, "c")]),
    "Sed": (376, KEY + [("ped", 11, 8, "s"), ("ymd", 19, 8, "n"), ("name", 27, 36, "s"),
                        ("dist", 63, 4, "n"), ("td", 67, 1, "n"), ("tosu", 131, 2, "n"),
                        ("jyun", 141, 2, "n"), ("ijou", 143, 1, "n"), ("time", 144, 4, "n"),
                        ("tan_odds", 175, 6, "n"), ("ninki", 181, 2, "n"), ("idm", 183, 3, "n"),
                        ("soten", 186, 3, "n"), ("babasa", 189, 3, "n"), ("pace", 192, 3, "n"),
                        ("deokure", 195, 3, "n"), ("ichidori", 198, 3, "n"), ("furi", 201, 3, "n"),
                        ("mae_furi", 204, 3, "n"), ("naka_furi", 207, 3, "n"), ("ato_furi", 210, 3, "n"),
                        ("race_adj", 213, 3, "n"), ("course_dori", 216, 1, "n"), ("joushou", 217, 1, "n"),
                        ("class_cd", 218, 2, "c"), ("batai_cd", 220, 1, "c"), ("kehai_cd", 221, 1, "c"),
                        ("race_pace", 222, 1, "c"), ("uma_pace", 223, 1, "c"), ("ten_idx", 224, 5, "n"),
                        ("agari_idx", 229, 5, "n"), ("pace_idx", 234, 5, "n"), ("racep_idx", 239, 5, "n"),
                        ("fuku_odds", 291, 6, "n"), ("odds10_tan", 297, 6, "n"), ("odds10_fuku", 303, 6, "n"),
                        ("c1", 309, 2, "n"), ("c2", 311, 2, "n"), ("c3", 313, 2, "n"), ("c4", 315, 2, "n"),
                        ("bw", 333, 3, "n"), ("yon_course", 370, 1, "n"), ("hassou", 371, 4, "c")]),
}


def parse_type(typ):
    reclen, spec = SPEC[typ]
    rows, bad, nfiles = [], 0, 0
    for zp in sorted((RAW / typ).glob("*.zip")):
        with zipfile.ZipFile(zp) as z:
            for name in z.namelist():
                if not name.lower().endswith(".txt"):
                    continue
                nfiles += 1
                for line in z.read(name).split(b"\n"):
                    line = line.rstrip(b"\r")
                    if not line.strip():
                        continue
                    if len(line) != reclen - 2:
                        bad += 1
                        continue
                    rows.append(tuple(line[s - 1:s - 1 + n] for _, s, n, _ in spec) + (name,))
    df = pd.DataFrame(rows, columns=[c for c, *_ in spec] + ["src"])
    for c, _, _, t in spec:
        s = df[c].map(lambda b: b.decode("cp932", errors="replace").strip())
        if t == "n":
            df[c] = pd.to_numeric(s, errors="coerce")
        elif t == "c":
            df[c] = s.where(s != "", None)
        else:
            df[c] = s
    df["src"] = df["src"].astype(str)
    before = len(df)
    df = df.drop_duplicates(["racekey", "ban"], keep="last").reset_index(drop=True)
    print(f"{typ}: files={nfiles} rows={before:,} dup除去後={len(df):,} 長さ不正行={bad}")
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for typ in SPEC:
        df = parse_type(typ)
        df.to_parquet(OUT / f"{typ.lower()}.parquet", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
