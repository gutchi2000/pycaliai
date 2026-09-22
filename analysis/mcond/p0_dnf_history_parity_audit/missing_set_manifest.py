# -*- coding: utf-8 -*-
"""
missing_set_manifest.py
=======================
`data/weekly/{date}.csv` に存在せず `data/kekka/{date}.csv` に存在する
race / horse 行の集合を machine-readable manifest として固定する。

結論表現の固定（ユーザー指示 2026-09-22）:
  - build_horse_history.py: code bug なし
  - _horse_history.parquet: weekly 入力に対して完全
  - data/weekly: 2026 履歴の authoritative source として不完全
  - _horse_history.parquet: 全出走履歴を必要とする consumer には不完全
  - legacy-v6 shadow replay: 再開条件未達
「parquet生成バグ」「再構築すれば直る」とは表現しない。

READ-ONLY。production 変更・モデル変更・ROI 評価は行わない。
出力: out/missing_set_manifest.json  (+ out/missing_set_rows.csv)
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import date as _date
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

KEKKA_DIR = BASE / "data" / "kekka"
WEEKLY_DIR = BASE / "data" / "weekly"
MASTER_MAX_DATE = 20251228  # master_v2 の最終日 (build_horse_history.py と同一境界)

WEEKDAY_JP = ["月", "火", "水", "木", "金", "土", "日"]


def log(m):
    print(m, flush=True)


def build_missing_set() -> tuple[pd.DataFrame, dict]:
    """build_horse_history.load_2026_history() と同一の join 条件で
    「kekka にあって weekly 側に対応が無い」行を抽出する。"""
    from build_horse_history import parse_weekly_light, _clean_name

    miss_rows: list[pd.DataFrame] = []
    per_date: list[dict] = []
    dates_no_weekly: list[int] = []

    for kp in sorted(KEKKA_DIR.glob("*.csv")):
        stem = kp.stem
        if not (stem.isdigit() and len(stem) == 8):
            continue
        d = int(stem)
        if d <= MASTER_MAX_DATE:
            continue
        wp = WEEKLY_DIR / f"{stem}.csv"
        if not wp.exists():
            dates_no_weekly.append(d)
            continue
        try:
            k = pd.read_csv(kp, encoding="cp932", dtype=str)
        except Exception as e:
            log(f"  ! {stem}: kekka read fail {e}")
            continue
        if "レースID(新)" not in k.columns or "確定着順" not in k.columns:
            continue

        k = k.assign(
            rid16=k["レースID(新)"].astype(str).str.strip().str[:16],
            umaban=pd.to_numeric(k["馬番"], errors="coerce"),
            pos_raw=k["確定着順"].astype(str).str.strip(),
            pos=pd.to_numeric(k["確定着順"], errors="coerce"),
        )
        k["pos"] = k["pos"].where(k["pos"] > 0)
        k["name_clean"] = k["馬名"].map(_clean_name)

        w = parse_weekly_light(wp)
        if w.empty:
            # weekly はあるがパース 0 行 → kekka 全行が欠落集合
            miss = k.copy()
            miss["miss_stage"] = "weekly_parse_empty"
            miss_rows.append(miss)
            per_date.append({
                "date": d, "kekka_rows": len(k), "weekly_rows": 0,
                "joined": 0, "name_filtered": 0, "missing": len(k),
            })
            continue

        merged = k[["rid16", "umaban", "name_clean"]].merge(
            w.rename(columns={"馬番": "umaban"})[["rid16", "umaban", "name"]],
            on=["rid16", "umaban"], how="left", indicator=True)

        join_ok = merged["_merge"] == "both"
        name_ok = join_ok & (merged["name_clean"] == merged["name"])

        k = k.reset_index(drop=True)
        miss = k.loc[~name_ok.values].copy()
        miss["miss_stage"] = np.where(
            join_ok.values[~name_ok.values], "name_filter", "weekly_join")
        if len(miss):
            miss_rows.append(miss)

        per_date.append({
            "date": d, "kekka_rows": len(k), "weekly_rows": int(len(w)),
            "joined": int(join_ok.sum()), "name_filtered": int(name_ok.sum()),
            "missing": int(len(k) - name_ok.sum()),
        })

    miss_df = (pd.concat(miss_rows, ignore_index=True)
               if miss_rows else pd.DataFrame())
    return miss_df, {"per_date": per_date, "dates_no_weekly": dates_no_weekly}


def classify_finish(pos_raw: str, pos) -> str:
    """kekka『確定着順』の finish status 分類。
    データソースの制約により 止(DNF)/外(取消)/消(除外) は区別できない
    (LIVE_SERVE_DNF_TRACE.md で既確認) → non_finish として一括。"""
    if pd.notna(pos):
        return "finished"
    s = str(pos_raw).strip()
    if s in ("", "nan", "None"):
        return "non_finish_blank"
    if s in ("0", "０"):
        return "non_finish_zero"
    return f"non_finish_other:{s}"


def main():
    log("=" * 70)
    log("MISSING SET MANIFEST — weekly に無く kekka に存在する race/horse 行")
    log("=" * 70)

    miss, meta = build_missing_set()
    per_date = pd.DataFrame(meta["per_date"])

    total_kekka = int(per_date["kekka_rows"].sum())
    total_missing = int(per_date["missing"].sum())
    n_dates = len(per_date)

    log(f"\n対象日数: {n_dates}  (weekly 欠如日: {len(meta['dates_no_weekly'])})")
    log(f"kekka 総行数: {total_kekka:,}")
    log(f"欠落行数: {total_missing:,}  ({total_missing/total_kekka*100:.3f}%)")

    if miss.empty:
        log("欠落なし")
        return

    miss["date"] = miss["日付"].astype(str).str.replace("/", "").str[:8]
    miss["date"] = pd.to_numeric(miss["date"], errors="coerce").astype("Int64")
    miss["finish_status"] = [
        classify_finish(r, p) for r, p in zip(miss["pos_raw"], miss["pos"])]
    miss["venue"] = miss["場所"].astype(str).str.strip()
    miss["month"] = (miss["date"] // 100 % 100).astype("Int64")
    miss["weekday"] = [
        WEEKDAY_JP[_date(int(d) // 10000, int(d) // 100 % 100, int(d) % 100).weekday()]
        if pd.notna(d) else "?" for d in miss["date"]]

    # --- レース単位 ---
    kekka_race_n = {}
    for kp in sorted(KEKKA_DIR.glob("*.csv")):
        if not (kp.stem.isdigit() and len(kp.stem) == 8):
            continue
        if int(kp.stem) <= MASTER_MAX_DATE:
            continue
        try:
            k = pd.read_csv(kp, encoding="cp932", dtype=str)
        except Exception:
            continue
        if "レースID(新)" not in k.columns:
            continue
        rid = k["レースID(新)"].astype(str).str.strip().str[:16]
        for r, c in rid.value_counts().items():
            kekka_race_n[r] = int(c)

    race_miss = (miss.groupby("rid16")
                 .agg(missing_rows=("rid16", "size"),
                      date=("date", "first"), venue=("venue", "first"),
                      month=("month", "first"), weekday=("weekday", "first"))
                 .reset_index())
    race_miss["kekka_rows_in_race"] = race_miss["rid16"].map(kekka_race_n)
    race_miss["whole_race_missing"] = (
        race_miss["missing_rows"] == race_miss["kekka_rows_in_race"])

    n_races_total = len(kekka_race_n)
    n_races_affected = len(race_miss)
    n_races_whole = int(race_miss["whole_race_missing"].sum())
    n_races_partial = n_races_affected - n_races_whole

    log(f"\nレース単位:")
    log(f"  kekka 総レース数: {n_races_total:,}")
    log(f"  欠落を含むレース: {n_races_affected}  ({n_races_affected/n_races_total*100:.2f}%)")
    log(f"    うち レース丸ごと欠落: {n_races_whole}")
    log(f"    うち 部分欠落: {n_races_partial}")

    # --- finish status 別 ---
    fs = miss["finish_status"].value_counts().to_dict()
    log(f"\nfinish status 別欠落行数:")
    for k_, v in fs.items():
        log(f"  {k_}: {v}")

    # --- 内訳 ---
    by_venue = miss["venue"].value_counts().to_dict()
    by_month = {str(k_): int(v) for k_, v in miss["month"].value_counts().items()}
    by_weekday = miss["weekday"].value_counts().to_dict()
    by_stage = miss["miss_stage"].value_counts().to_dict()

    log(f"\n欠落段階: {by_stage}")
    log(f"venue 別 (上位): {dict(list(sorted(by_venue.items(), key=lambda x: -x[1]))[:12])}")
    log(f"month 別: {by_month}")
    log(f"weekday 別: {by_weekday}")

    # --- 必要列ごとの不足状況 (kekka 単独で何が取れるか) ---
    kekka_cols = list(miss.columns)
    required_cols = {
        "race_id": "レースID(新)" in kekka_cols,
        "race_date": "日付" in kekka_cols,
        "血統登録番号": "血統登録番号" in kekka_cols,
        "finish_status": "確定着順" in kekka_cols,
        "surface": any("芝" in c or "ダ" in c for c in kekka_cols),
        "distance": any("距離" in c for c in kekka_cols),
        "venue": "場所" in kekka_cols,
        "jockey": any("騎手" in c for c in kekka_cols),
        "trainer": any("調教師" in c or "厩舎" in c for c in kekka_cols),
        "sire_pedigree": any("父" in c or "種牡馬" in c for c in kekka_cols),
        "race_class": any("クラス" in c or "条件" in c for c in kekka_cols),
        "umaban": "馬番" in kekka_cols,
        "horse_name": "馬名" in kekka_cols,
    }
    log(f"\nkekka 単独での必要列充足:")
    for c, ok in required_cols.items():
        log(f"  {'OK ' if ok else 'NG '} {c}")

    missing_required = [c for c, ok in required_cols.items() if not ok]

    # --- 保存 ---
    rows_out = miss[["date", "rid16", "umaban", "馬名", "venue", "Ｒ",
                     "pos_raw", "pos", "finish_status", "miss_stage",
                     "month", "weekday"]].copy()
    rows_out = rows_out.rename(columns={"rid16": "race_id", "馬名": "horse_name",
                                        "Ｒ": "race_no"})
    rows_out["血統登録番号"] = None  # kekka に存在しない (候補ソース探索対象)
    rows_out.to_csv(OUT / "missing_set_rows.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "scope": "2026 dates only (date > master_v2 max 20251228)",
        "conclusion_framing_fixed": {
            "build_horse_history_py": "code bug なし",
            "_horse_history_parquet_vs_weekly": "weekly 入力に対して完全",
            "data_weekly": "2026 履歴の authoritative source として不完全",
            "_horse_history_parquet_vs_consumers": "全出走履歴を必要とする consumer には不完全",
            "legacy_v6_shadow_replay": "再開条件未達",
        },
        "totals": {
            "n_dates": n_dates,
            "n_dates_without_weekly_file": len(meta["dates_no_weekly"]),
            "dates_without_weekly_file": meta["dates_no_weekly"],
            "kekka_rows": total_kekka,
            "missing_rows": total_missing,
            "missing_row_rate": round(total_missing / total_kekka, 6),
            "kekka_races": n_races_total,
            "races_with_any_missing": n_races_affected,
            "races_wholly_missing": n_races_whole,
            "races_partially_missing": n_races_partial,
            "race_affected_rate": round(n_races_affected / n_races_total, 6),
        },
        "by_miss_stage": by_stage,
        "by_finish_status": fs,
        "by_venue": by_venue,
        "by_month": by_month,
        "by_weekday": by_weekday,
        "required_column_availability_in_kekka": required_cols,
        "required_columns_missing_from_kekka": missing_required,
        "per_date": meta["per_date"],
        "missing_races": race_miss.sort_values("date").to_dict("records"),
        "rows_csv": "out/missing_set_rows.csv",
        "notes": [
            "血統登録番号 は kekka/weekly いずれにも存在しない → 候補ソース探索が必要",
            "止(DNF)/外(取消)/消(除外) は kekka『確定着順』で区別不能 "
            "(既確認: LIVE_SERVE_DNF_TRACE.md) → non_finish として一括計上",
        ],
    }
    with open(OUT / "missing_set_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)

    log(f"\n保存: out/missing_set_manifest.json / out/missing_set_rows.csv")


if __name__ == "__main__":
    main()
