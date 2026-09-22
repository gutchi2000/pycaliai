# -*- coding: utf-8 -*-
"""
alt_source_coverage_matrix.py
=============================
欠落集合 (out/missing_set_manifest.json = weekly に無く kekka にある 598 行 /
50 レース) を復元しうる候補ソースを、必須列 coverage matrix と実測照合で評価する。

馬名 join は禁止。join key は race_id(+馬番) を基本とする。
結果・払戻・ROI は一切読まない。READ-ONLY。

出力: out/alt_source_coverage_matrix.json
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "out"

REQUIRED_COLS = [
    "race_id", "race_date", "ped_id", "finish_status", "surface", "distance",
    "venue", "jockey", "trainer", "sire", "race_class", "time_provenance",
]


def log(m):
    print(m, flush=True)


def load_missing():
    man = json.load(open(OUT / "missing_set_manifest.json", encoding="utf-8"))
    rows = pd.read_csv(OUT / "missing_set_rows.csv", encoding="utf-8-sig", dtype=str)
    rows["umaban"] = pd.to_numeric(rows["umaban"], errors="coerce")
    races = pd.DataFrame(man["missing_races"])
    # rid から日付を復元 (manifest の date 列は表示用に桁落ちしているため rid を正とする)
    races["date"] = races["rid16"].str[:8].astype(int)
    rows["date"] = rows["race_id"].str[:8].astype(int)
    return man, rows, races


# ---------------- 候補ソース reader ----------------

def read_tyaku(date: int) -> pd.DataFrame:
    """data/tyaku/{date}.csv = レース単位ヘッダ (19列)。"""
    p = BASE / "data" / "tyaku" / f"{date}.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p, encoding="cp932", dtype=str, on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()
    if "レースID(新)" not in df.columns:
        return pd.DataFrame()
    df["rid16"] = df["レースID(新)"].astype(str).str.strip().str[:16]
    return df


def read_bias(date: int) -> pd.DataFrame:
    p = BASE / "data" / "bias" / f"{date}.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p, encoding="cp932", dtype=str, on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()
    if "レースID" not in df.columns:
        return pd.DataFrame()
    df["rid16"] = df["レースID"].astype(str).str.strip().str[:16]
    return df


def read_bunseki(date: int) -> pd.DataFrame:
    p = BASE / "data" / "bunseki" / f"{date}.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p, encoding="cp932", dtype=str, on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()
    if "レースID(新)" not in df.columns:
        return pd.DataFrame()
    df["rid16"] = df["レースID(新)"].astype(str).str.strip().str[:16]
    df["umaban"] = pd.to_numeric(df.get("馬番"), errors="coerce")
    return df


def read_kako5_pastruns(date: int) -> pd.DataFrame:
    """kako5 raw から「過去走」レコードを取り出す。
    過去走は 月/日/場所/TD/距離/着順/人気/上り3F のみ。年・race_id・
    血統登録番号・騎手・調教師は持たない。"""
    p = BASE / "data" / "kako5" / f"{date}.csv"
    if not p.exists():
        return pd.DataFrame()
    recs = []
    cur = None
    off = [(12, 13, 14, 15, 16, 18), (24, 25, 26, 27, 28, 30),
           (36, 37, 38, 39, 40, 42), (48, 49, 50, 51, 52, 54),
           (60, 61, 62, 63, 64, 66)]
    try:
        with open(p, encoding="cp932", errors="replace") as f:
            for row in csv.reader(f):
                if len(row) == 19 and row[0] and row[0][:4].isdigit():
                    cur = row[0][:16]
                    continue
                if len(row) == 72 and row[0].isdigit() and cur:
                    for (mo, dy, pl, td, di, po) in off:
                        if not row[mo] or not row[dy]:
                            continue
                        recs.append({"cur_rid16": cur, "cur_umaban": row[2],
                                     "p_month": row[mo], "p_day": row[dy],
                                     "p_place": row[pl], "p_td": row[td],
                                     "p_dist": row[di], "p_pos": row[po]})
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame(recs)


# ---------------- 評価 ----------------

def evaluate():
    man, miss_rows, miss_races = load_missing()
    miss_dates = sorted(miss_races["date"].unique().tolist())
    log(f"欠落: {len(miss_rows)} 行 / {len(miss_races)} レース / {len(miss_dates)} 日")

    results = {}

    # ---- 1) data/tyaku (レース単位) ----
    rec_races, cols_ok = 0, {"surface": 0, "distance": 0, "race_class": 0}
    dates_present = 0
    for d in miss_dates:
        t = read_tyaku(d)
        if t.empty:
            continue
        dates_present += 1
        rids = set(t["rid16"])
        for rid in miss_races.loc[miss_races["date"] == d, "rid16"]:
            if rid in rids:
                rec_races += 1
                r = t[t["rid16"] == rid].iloc[0]
                if str(r.get("芝・ダート", "")).strip():
                    cols_ok["surface"] += 1
                if str(r.get("距離", "")).strip():
                    cols_ok["distance"] += 1
                if str(r.get("クラス名", "")).strip():
                    cols_ok["race_class"] += 1
    results["data/tyaku"] = {
        "granularity": "race (horse 行なし)",
        "join_key": "レースID(新)[:16]",
        "dates_covering_missing": dates_present,
        "dates_needed": len(miss_dates),
        "races_recovered": rec_races,
        "races_needed": len(miss_races),
        "race_recovery_rate": round(rec_races / len(miss_races), 4),
        "horse_rows_recovered": 0,
        "column_fill": cols_ok,
        "supplies": ["surface", "distance", "race_class", "race_date", "venue"],
        "cannot_supply": ["ped_id", "jockey", "trainer", "sire",
                          "finish_status(horse)", "umaban"],
    }

    # ---- 2) data/bias (レース単位 + 1-3着のみ) ----
    rec_races_b, dates_b = 0, 0
    for d in miss_dates:
        b = read_bias(d)
        if b.empty:
            continue
        dates_b += 1
        rids = set(b["rid16"])
        rec_races_b += sum(
            1 for rid in miss_races.loc[miss_races["date"] == d, "rid16"] if rid in rids)
    results["data/bias"] = {
        "granularity": "race + 1〜3着馬のみ (全出走馬なし)",
        "join_key": "レースID[:16]",
        "dates_covering_missing": dates_b,
        "dates_needed": len(miss_dates),
        "races_recovered": rec_races_b,
        "races_needed": len(miss_races),
        "race_recovery_rate": round(rec_races_b / len(miss_races), 4),
        "horse_rows_recovered": 0,
        "supplies": ["surface", "distance", "race_class", "平・障 flag"],
        "cannot_supply": ["ped_id", "全出走馬行", "jockey(4着以下)",
                          "trainer(4着以下)"],
    }

    # ---- 3) data/bunseki (馬単位・全必須列あり) ----
    rec_rows, rec_races_s, dates_s = 0, 0, 0
    fill = {c: 0 for c in ["ped_id", "jockey", "trainer", "sire",
                           "surface", "distance", "race_class"]}
    for d in miss_dates:
        s = read_bunseki(d)
        if s.empty:
            continue
        dates_s += 1
        sub = miss_rows[miss_rows["date"] == d]
        key = set(zip(s["rid16"], s["umaban"]))
        rids = set(s["rid16"])
        rec_races_s += sum(
            1 for rid in miss_races.loc[miss_races["date"] == d, "rid16"] if rid in rids)
        for _, r in sub.iterrows():
            if (r["race_id"], r["umaban"]) in key:
                rec_rows += 1
                row = s[(s["rid16"] == r["race_id"]) & (s["umaban"] == r["umaban"])].iloc[0]
                for c, col in [("ped_id", "血統登録番号"), ("jockey", "騎手コード"),
                               ("trainer", "調教師コード"), ("sire", "種牡馬"),
                               ("surface", "芝ダ"), ("distance", "距離"),
                               ("race_class", "クラス")]:
                    v = str(row.get(col, "")).strip()
                    if v and v.lower() != "nan":
                        fill[c] += 1
    results["data/bunseki"] = {
        "granularity": "horse (全出走馬・122列)",
        "join_key": "レースID(新)[:16] + 馬番 (馬名 join 不要)",
        "dates_covering_missing": dates_s,
        "dates_needed": len(miss_dates),
        "races_recovered": rec_races_s,
        "races_needed": len(miss_races),
        "race_recovery_rate": round(rec_races_s / len(miss_races), 4),
        "horse_rows_recovered": rec_rows,
        "horse_rows_needed": len(miss_rows),
        "horse_row_recovery_rate": round(rec_rows / len(miss_rows), 4),
        "column_fill_on_recovered": fill,
        "supplies": REQUIRED_COLS[:-1],
        "cannot_supply": ["finish_status (出走前カードのため着順なし "
                          "→ kekka と合成が必要)"],
        "available_dates": sorted(
            int(p.stem) for p in (BASE / "data" / "bunseki").glob("*.csv")),
    }

    # ---- 4) data/kako5 (過去走レコード: 後日のカードから遡及) ----
    #   欠落レースは「後の開催日の kako5 に過去走として現れるか」を測る
    k5_hits, k5_checked = 0, 0
    k5_dates = sorted(int(p.stem) for p in (BASE / "data" / "kako5").glob("*.csv"))
    for d in miss_dates[:12]:  # 代表 12 日で実測
        later = [x for x in k5_dates if x > d][:3]
        want = miss_races.loc[miss_races["date"] == d]
        for _, rr in want.iterrows():
            k5_checked += 1
            mo, dy = int(str(d)[4:6]), int(str(d)[6:8])
            found = False
            for ld in later:
                pr = read_kako5_pastruns(ld)
                if pr.empty:
                    continue
                hit = pr[(pd.to_numeric(pr["p_month"], errors="coerce") == mo) &
                         (pd.to_numeric(pr["p_day"], errors="coerce") == dy)]
                if len(hit):
                    found = True
                    break
            k5_hits += int(found)
    results["data/kako5 (past-run 遡及)"] = {
        "granularity": "過去走レコード (月/日/場所/TD/距離/着順/人気/上り3F)",
        "join_key": "無し (race_id も 血統登録番号 も持たない)",
        "dates_available": len(k5_dates),
        "sampled_races_checked": k5_checked,
        "sampled_races_with_same_month_day_pastrun": k5_hits,
        "supplies": ["surface", "distance", "venue", "finish_status",
                     "race_date(年なし)"],
        "cannot_supply": ["race_id", "ped_id", "jockey", "trainer", "sire",
                          "年 (month/day のみ)", "決定的な join key"],
        "note": "過去走は当該馬が『後日また出走した』場合のみ現れる。"
                "欠落レースのみを走った馬・以後引退した馬は永久に現れない。"
                "また join key が無く、馬名照合を要するため本監査の禁止事項に抵触する。",
    }

    # ---- 5) data/走間分析 (13/13 だが期間外) ----
    so = sorted((BASE / "data" / "走間分析").glob("*.csv"))
    results["data/走間分析"] = {
        "granularity": "horse (13/13 必須列)",
        "files": [p.name for p in so],
        "period_end": "20251221 (2026 を含まない)",
        "races_recovered": 0,
        "verdict_reason": "2026 期間外のため欠落集合をまったく復元できない",
    }

    # ---- 6) 外部 E:\競馬過去走データ ----
    ext = Path(r"E:\競馬過去走データ")
    ext2026 = sorted(d.name for d in (ext / "2026").iterdir() if d.is_dir()) \
        if (ext / "2026").exists() else []
    results["外部 E:/競馬過去走データ"] = {
        "2026_subdirs": ext2026,
        "2026_subdir_range": f"{ext2026[0]}〜{ext2026[-1]}" if ext2026 else "なし",
        "single_date_files": ["全出走馬分析20260905.csv", "出走馬分析20260905.csv",
                              "一覧20260830.csv", "bias20260912.csv",
                              "bias20260913.csv", "ROI用_20260906.csv"],
        "kekka_masters": ["kekka_20130105-20251228_v2.csv (2025 まで)",
                          "raw_data/kekka_1986_2025*.csv (2025 まで)"],
        "verdict_reason": "2026 サブディレクトリは 0104〜0301 で止まっており、"
                          "欠落集合(0307〜0906)をカバーしない。"
                          "単日ファイルは 0830/0905/0906 のみ。",
    }

    return man, miss_rows, miss_races, results


def main():
    log("=" * 70)
    log("ALT SOURCE COVERAGE MATRIX (read-only)")
    log("=" * 70)
    man, miss_rows, miss_races, results = evaluate()

    for name, r in results.items():
        log(f"\n--- {name} ---")
        for k, v in r.items():
            if isinstance(v, (list, dict)) and len(str(v)) > 160:
                log(f"  {k}: {str(v)[:160]}...")
            else:
                log(f"  {k}: {v}")

    payload = {
        "generated_at": datetime.now().isoformat(),
        "missing_set": {
            "rows": len(miss_rows), "races": len(miss_races),
            "dates": int(miss_races['date'].nunique()),
        },
        "candidates": results,
    }
    with open(OUT / "alt_source_coverage_matrix.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    log("\n保存: out/alt_source_coverage_matrix.json")


if __name__ == "__main__":
    main()
