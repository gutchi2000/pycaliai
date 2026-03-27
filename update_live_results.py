"""
update_live_results.py
kekka CSV（data/kekka/YYYYMMDD.csv）を読み、
data/live_results_2026.csv の着順・払戻列を自動埋めする。

Usage:
    python update_live_results.py --date 20260322
    python update_live_results.py           # data/kekka/ の最新ファイルを自動検出
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR    = Path(__file__).parent
LIVE_CSV    = BASE_DIR / "data" / "live_results_2026.csv"
KEKKA_DIR   = BASE_DIR / "data" / "kekka"

# kekka CSV の列インデックス（cp932, ヘッダあり）
# 0:日付 1:場所 2:R 3:枠番 4:馬番 5:馬名 6:確定着順 7:レースID
# 8:単勝払戻 9:複勝払戻 10:馬連払戻 11:馬単払戻 12:ワイド払戻
# 13:三連複払戻 14:三連単払戻
COL_DATE   = 0
COL_PLACE  = 1
COL_R      = 2
COL_BANUM  = 4
COL_CHAKU  = 6
COL_TAN    = 8   # 単勝払戻
COL_FUKU   = 9   # 複勝払戻
COL_UMAREN = 10  # 馬連払戻
COL_SANREN = 13  # 三連複払戻


def _parse_payout(val) -> int | None:
    """払戻列の値をパース。(6.6) はオッズ表示なので無効、数値のみ有効。"""
    if val is None:
        return None
    s = str(val).strip()
    if s.startswith("(") or s in ("", "nan", "NaN"):
        return None
    try:
        f = float(s)
        if pd.isna(f):
            return None
        return int(f)
    except (ValueError, TypeError):
        return None


def _kekka_date_to_live(yymmdd: str) -> str:
    """260322 → '2026.3.22' に変換（live_results の日付フォーマット）"""
    s = str(yymmdd).strip().zfill(6)
    yy = int(s[0:2])
    mm = int(s[2:4])
    dd = int(s[4:6])
    yyyy = 2000 + yy
    return f"{yyyy}.{mm}.{dd}"


def _parse_umaren_combo(buy_str: str) -> list[tuple[int, int]]:
    """
    '7-9 / 7-14' → [(7,9), (7,14)]
    買い目文字列から馬番ペアのリストを返す。
    """
    combos = []
    for part in buy_str.split("/"):
        part = part.strip()
        m = re.match(r"(\d+)-(\d+)", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            combos.append((min(a, b), max(a, b)))
    return combos


def _parse_sanrenfuku_combo(buy_str: str) -> tuple[int, int, int] | None:
    """
    '1-7-9' → (1,7,9)
    """
    m = re.match(r"(\d+)-(\d+)-(\d+)", buy_str.strip())
    if m:
        nums = sorted([int(m.group(i)) for i in range(1, 4)])
        return tuple(nums)
    return None


def update_live_results(date_str: str) -> None:
    kekka_path = KEKKA_DIR / f"{date_str}.csv"
    if not kekka_path.exists():
        logger.error(f"kekka CSV が見つかりません: {kekka_path}")
        return

    if not LIVE_CSV.exists():
        logger.error(f"live_results が見つかりません: {LIVE_CSV}")
        return

    # --- kekka 読み込み ---
    kk = pd.read_csv(kekka_path, encoding="cp932", header=0, dtype=str)
    kk.columns = list(range(len(kk.columns)))  # 列名を数値インデックスに統一

    # 日付を live_results フォーマットに変換
    kk["_live_date"] = kk[COL_DATE].apply(_kekka_date_to_live)
    kk["_place"]     = kk[COL_PLACE].astype(str).str.strip()
    kk["_R"]         = kk[COL_R].astype(str).str.strip()
    kk["_banum"]     = kk[COL_BANUM].astype(str).str.strip()
    kk["_chaku"]     = kk[COL_CHAKU].astype(str).str.strip()
    kk["_tan_pay"]   = kk[COL_TAN].apply(_parse_payout)
    kk["_fuku_pay"]  = kk[COL_FUKU].apply(_parse_payout)
    kk["_umaren_pay"]= kk[COL_UMAREN].apply(_parse_payout)
    kk["_sanren_pay"]= kk[COL_SANREN].apply(_parse_payout)

    # レースごとに 1-2着馬番セットを作成（馬連照合用）
    kk["_chaku_int"] = pd.to_numeric(kk["_chaku"], errors="coerce")
    race_top2: dict[tuple, set] = {}
    race_top3: dict[tuple, set] = {}
    for _, row in kk.iterrows():
        key = (row["_live_date"], row["_place"], row["_R"])
        if row["_chaku_int"] <= 2:
            race_top2.setdefault(key, set()).add(int(row["_banum"]))
        if row["_chaku_int"] <= 3:
            race_top3.setdefault(key, set()).add(int(row["_banum"]))

    # kekka をキーで引けるように辞書化
    kk_dict: dict[tuple, dict] = {}
    for _, row in kk.iterrows():
        key = (row["_live_date"], row["_place"], row["_R"], row["_banum"])
        kk_dict[key] = row

    # --- live_results 読み込み ---
    lr = pd.read_csv(LIVE_CSV, encoding="utf-8-sig", dtype=str)
    lr["_日付_str"] = lr["日付"].astype(str).str.strip()
    lr["_場所_str"] = lr["場所"].astype(str).str.strip()
    lr["_R_str"]   = lr["R"].astype(str).str.strip()
    lr["_馬番_str"] = lr["馬番"].astype(str).str.strip()

    updated = 0
    for idx, row in lr.iterrows():
        key = (row["_日付_str"], row["_場所_str"], row["_R_str"], row["_馬番_str"])
        if key not in kk_dict:
            continue

        k = kk_dict[key]
        race_key = (row["_日付_str"], row["_場所_str"], row["_R_str"])
        changed = False

        # ── 着順 ──
        chaku_val = str(k["_chaku"]).strip()
        if pd.isna(lr.at[idx, "着順"]) or str(lr.at[idx, "着順"]).strip() in ("", "nan"):
            lr.at[idx, "着順"] = chaku_val
            changed = True

        # ── 単勝払戻（1着馬のみ） ──
        if k["_tan_pay"] is not None:
            if pd.isna(lr.at[idx, "単勝払戻"]) or str(lr.at[idx, "単勝払戻"]).strip() in ("", "nan"):
                lr.at[idx, "単勝払戻"] = str(k["_tan_pay"])
                changed = True

        # ── 複勝払戻（3着以内） ──
        if k["_fuku_pay"] is not None:
            if pd.isna(lr.at[idx, "複勝払戻"]) or str(lr.at[idx, "複勝払戻"]).strip() in ("", "nan"):
                lr.at[idx, "複勝払戻"] = str(k["_fuku_pay"])
                changed = True

        # ── 馬連払戻（HAHO買い目と照合） ──
        buy_str = str(row.get("HAHO_馬連_買い目", "")).strip()
        if buy_str and buy_str != "nan" and "馬連払戻" in lr.columns:
            if pd.isna(lr.at[idx, "馬連払戻"]) or str(lr.at[idx, "馬連払戻"]).strip() in ("", "nan"):
                top2 = race_top2.get(race_key, set())
                for (a, b) in _parse_umaren_combo(buy_str):
                    if {a, b} <= top2 and k["_umaren_pay"] is not None:
                        lr.at[idx, "馬連払戻"] = str(k["_umaren_pay"])
                        changed = True
                        break

        # ── 三連複払戻（HAHO/HALO三連複買い目と照合） ──
        for bet_col in ["HAHO_三連複_買い目", "HALO_三連複_買い目"]:
            san_str = str(row.get(bet_col, "")).strip()
            if san_str and san_str != "nan" and "三連複払戻" in lr.columns:
                if pd.isna(lr.at[idx, "三連複払戻"]) or str(lr.at[idx, "三連複払戻"]).strip() in ("", "nan"):
                    top3 = race_top3.get(race_key, set())
                    combo = _parse_sanrenfuku_combo(san_str)
                    if combo and set(combo) <= top3 and k["_sanren_pay"] is not None:
                        lr.at[idx, "三連複払戻"] = str(k["_sanren_pay"])
                        changed = True

        if changed:
            updated += 1

    # 作業列を削除して保存
    lr = lr.drop(columns=["_日付_str", "_場所_str", "_R_str", "_馬番_str"])
    lr.to_csv(LIVE_CSV, index=False, encoding="utf-8-sig")
    logger.info(f"live_results_2026.csv 更新完了: {updated}行 更新 ({LIVE_CSV})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="", help="YYYYMMDD（省略時は data/kekka/ の最新を使用）")
    args = parser.parse_args()

    date_str = args.date
    if not date_str:
        csvs = sorted(KEKKA_DIR.glob("????????.csv"), reverse=True)
        if not csvs:
            logger.error("data/kekka/ に CSV がありません")
            return
        date_str = csvs[0].stem
        logger.info(f"自動検出: {date_str}")

    update_live_results(date_str)


if __name__ == "__main__":
    main()
