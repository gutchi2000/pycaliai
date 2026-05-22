"""
build_pedigree_data.py
======================
master_v2 から血統データを 2 種類抽出してファイル化する。

(A) data/horse_pedigree.json  : 馬名 → {sire, sire_type, broodmare_sire, broodmare_sire_type}
    最新の records から build。export_weekly_marks.py が bundle.json に注入するのに使う。

(B) data/pedigree_stats.json  : (場所, td, dist_band) → {父成績, 母父成績}
    NiceGUI 血統分析タブのコース別ランキング表示に使う。
    集計範囲: 過去 10 年。距離は ±100m 帯でグルーピング。

Usage:
    python build_pedigree_data.py
    python build_pedigree_data.py --years 10 --min-runs 15

Output:
    data/horse_pedigree.json
    data/pedigree_stats.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE = Path(__file__).parent
MASTER_CSV = BASE / "data" / "master_v2_20130105-20251228.csv"

DEFAULT_YEARS = 10
DEFAULT_MIN_RUNS = 15

# 距離帯: 厳密一致を優先、サンプル不足なら ±100m → ±200m に拡張
# ここでは ±100m 帯で集計（実用上の妥協点）
DIST_HALF_WIDTH = 100


# ===========================================================
# Loader
# ===========================================================
def load_master(years: int) -> pd.DataFrame:
    """master_v2 から必要列だけ読み込む。"""
    usecols = [
        "日付", "場所", "芝・ダ", "距離", "着順",
        "馬名", "種牡馬", "父タイプ名", "母父馬", "母父タイプ名",
    ]
    logger.info(f"master_v2 読み込み: {MASTER_CSV.name}")
    df = pd.read_csv(
        MASTER_CSV, encoding="utf-8-sig",
        usecols=usecols, dtype=str, low_memory=False, on_bad_lines="skip",
    )
    logger.info(f"  生 rows: {len(df):,}")

    df["日付"] = pd.to_numeric(df["日付"], errors="coerce")
    df["着順"] = pd.to_numeric(df["着順"], errors="coerce")
    df["距離"] = pd.to_numeric(df["距離"], errors="coerce")

    # 過去 N 年フィルタ (今日 — N 年)
    today_int = int(datetime.now().strftime("%Y%m%d"))
    cutoff = int((datetime.now().replace(year=datetime.now().year - years)).strftime("%Y%m%d"))
    df = df[(df["日付"] >= cutoff) & (df["日付"] <= today_int)].copy()
    logger.info(f"  過去{years}年 ({cutoff}-{today_int}): {len(df):,} rows")

    df = df.dropna(subset=["着順", "距離", "馬名"])
    logger.info(f"  着順/距離/馬名 valid: {len(df):,} rows")
    return df


# ===========================================================
# (A) horse_pedigree.json — 馬名 → pedigree dict
# ===========================================================
def build_horse_pedigree(df: pd.DataFrame, out_path: Path) -> dict:
    """各馬について最新 record の pedigree を取り出す。"""
    logger.info("(A) 馬名 → pedigree マップ生成中...")
    # 馬名でグルーピングして最新 record (日付 max) を選ぶ
    df_sorted = df.sort_values("日付", ascending=False)
    latest = df_sorted.drop_duplicates(subset=["馬名"], keep="first")

    result = {}
    for _, r in latest.iterrows():
        name = str(r["馬名"]).strip()
        if not name:
            continue
        entry = {}
        if pd.notna(r.get("種牡馬")) and str(r["種牡馬"]).strip():
            entry["sire"] = str(r["種牡馬"]).strip()
        if pd.notna(r.get("父タイプ名")) and str(r["父タイプ名"]).strip():
            entry["sire_type"] = str(r["父タイプ名"]).strip()
        if pd.notna(r.get("母父馬")) and str(r["母父馬"]).strip():
            entry["broodmare_sire"] = str(r["母父馬"]).strip()
        if pd.notna(r.get("母父タイプ名")) and str(r["母父タイプ名"]).strip():
            entry["broodmare_sire_type"] = str(r["母父タイプ名"]).strip()
        if entry:
            result[name] = entry

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_horses": len(result),
        "source": "master_v2",
        "horses": result,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    size_kb = out_path.stat().st_size / 1024
    logger.info(f"  -> {out_path}  ({len(result):,} 頭, {size_kb:.1f} KB)")
    return payload


# ===========================================================
# (B) pedigree_stats.json — コース別 sire ランキング
# ===========================================================
def _compute_sire_stats(sub: pd.DataFrame, col: str, min_runs: int) -> list[dict]:
    """sub を col で groupby し、出走数 >= min_runs の sire のみ集計。

    Returns: list of dicts sorted by 複勝率 desc
    """
    out = []
    for sire, grp in sub.groupby(col):
        if pd.isna(sire) or not str(sire).strip():
            continue
        n = len(grp)
        if n < min_runs:
            continue
        pos = grp["着順"].values
        wins = int(np.sum(pos == 1))
        top2 = int(np.sum(pos <= 2))
        top3 = int(np.sum(pos <= 3))

        out.append({
            "name": str(sire).strip(),
            "n_runs": n,
            "n_1": wins,
            "n_2": top2 - wins,
            "n_3": top3 - top2,
            "n_out": n - top3,
            "win_rate": round(wins / n * 100, 1),
            "rentai_rate": round(top2 / n * 100, 1),
            "fuku_rate": round(top3 / n * 100, 1),
        })

    # 複勝率降順でソート
    out.sort(key=lambda x: -x["fuku_rate"])
    return out


def _assign_ranks(stats: list[dict], baseline_fuku: float) -> list[dict]:
    """複勝率からランクを付与: SS/S/A/B/C/D
    baseline_fuku は当該コースの全体複勝率(参考)を渡し、D ランクの理由表示に使う。
    """
    for s in stats:
        f = s["fuku_rate"]
        if f >= 35:    s["rank"] = "SS"
        elif f >= 30:  s["rank"] = "S"
        elif f >= 25:  s["rank"] = "A"
        elif f >= 20:  s["rank"] = "B"
        elif f >= 15:  s["rank"] = "C"
        else:          s["rank"] = "D"
        # D ランクの不向き理由 (baseline 比)
        if s["rank"] == "D" and baseline_fuku > 0:
            ratio = f / baseline_fuku
            s["unfit_reason"] = (
                f"複勝率 {f}% は平均 {baseline_fuku:.1f}% の "
                f"約 {int(round(1/ratio))} 分の 1"
                if ratio > 0 else
                f"複勝率 {f}% (平均 {baseline_fuku:.1f}%)"
            )
    return stats


def build_pedigree_stats(df: pd.DataFrame, out_path: Path,
                          min_runs: int = DEFAULT_MIN_RUNS,
                          dist_half: int = DIST_HALF_WIDTH) -> dict:
    """場所 × TD × 距離帯 ごとに父・母父集計を作る。"""
    logger.info("(B) コース別 sire 集計中...")

    # TD 列を 芝/ダ に統一
    td_map = {"T": "芝", "D": "ダ", "芝": "芝", "ダ": "ダ", "ダート": "ダ"}
    df = df.copy()
    df["td"] = df["芝・ダ"].astype(str).map(lambda v: td_map.get(v.strip(), v.strip()))

    # 距離帯: dist // 200 * 200 (e.g. 1500-1700 → 1600 bucket)
    # ただし表示には実際の代表距離も持たせる
    df["dist_bucket"] = (df["距離"] // 200 * 200 + 100).astype(int)  # 100, 300, ... 中央値

    courses: dict[str, dict] = {}
    place_td_dist_groups = df.groupby(["場所", "td", "dist_bucket"])
    logger.info(f"  combinations: {len(place_td_dist_groups)}")

    n_emitted = 0
    for (place, td, dist_bucket), sub in place_td_dist_groups:
        if len(sub) < 200:  # 少なすぎるコースは出力しない
            continue
        baseline_fuku = (sub["着順"] <= 3).mean() * 100  # 全体複勝率 = 3/N

        sire_stats = _compute_sire_stats(sub, "種牡馬", min_runs)
        bm_sire_stats = _compute_sire_stats(sub, "母父馬", min_runs)

        if not sire_stats and not bm_sire_stats:
            continue

        sire_stats = _assign_ranks(sire_stats, baseline_fuku)
        bm_sire_stats = _assign_ranks(bm_sire_stats, baseline_fuku)

        key = f"{place}_{td}_{dist_bucket}"
        courses[key] = {
            "place": place,
            "td": td,
            "dist_bucket": int(dist_bucket),
            "dist_band": [int(dist_bucket - dist_half), int(dist_bucket + dist_half)],
            "n_total_runs": len(sub),
            "baseline_fuku_rate": round(baseline_fuku, 1),
            "sire": sire_stats[:30],          # top 30
            "broodmare_sire": bm_sire_stats[:30],
        }
        n_emitted += 1

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "years": DEFAULT_YEARS,
        "min_runs": min_runs,
        "dist_half_width": dist_half,
        "n_courses": n_emitted,
        "source": "master_v2",
        "courses": courses,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    size_kb = out_path.stat().st_size / 1024
    logger.info(f"  -> {out_path}  ({n_emitted} courses, {size_kb:.1f} KB)")
    return payload


# ===========================================================
# Main
# ===========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=DEFAULT_YEARS, help="過去 N 年")
    ap.add_argument("--min-runs", type=int, default=DEFAULT_MIN_RUNS, help="ランク付け最小出走数")
    ap.add_argument("--skip-horse", action="store_true", help="(A) horse_pedigree.json をスキップ")
    ap.add_argument("--skip-stats", action="store_true", help="(B) pedigree_stats.json をスキップ")
    args = ap.parse_args()

    if not MASTER_CSV.exists():
        logger.error(f"master_v2 未存在: {MASTER_CSV}")
        sys.exit(1)

    df = load_master(args.years)

    if not args.skip_horse:
        build_horse_pedigree(df, BASE / "data" / "horse_pedigree.json")
    if not args.skip_stats:
        build_pedigree_stats(df, BASE / "data" / "pedigree_stats.json",
                              min_runs=args.min_runs)

    logger.info("done")


if __name__ == "__main__":
    main()
