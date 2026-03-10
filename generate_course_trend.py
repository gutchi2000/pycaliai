"""
generate_course_trend.py
========================
course_trend_*.csv から course_trend.json を生成するスクリプト。

使い方:
    python generate_course_trend.py \
        --input  data/course_trend_20060105-20251228.csv \
        --output data/course_trend.json \
        --min_n  30

引数:
    --input   : 入力CSVパス（cp932想定、utf-8も自動判定）
    --output  : 出力JSONパス（省略時 data/course_trend.json）
    --min_n   : 有効と判断する最小レース数（省略時 30）
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# =========================================================
# 定数
# =========================================================
PLACES = ["中山", "阪神", "中京", "東京", "京都", "福島", "新潟", "札幌", "函館", "小倉"]
SMILES  = ["S", "M", "I", "L", "E"]
CLASSES = ["新馬", "未勝利", "1勝", "2勝", "3勝", "OP/重賞"]
SEASONS = ["春", "夏", "秋", "冬"]

CLASS_NORMALIZE: dict[str, str] = {
    "新馬": "新馬", "未勝利": "未勝利",
    "1勝": "1勝", "500万": "1勝",
    "2勝": "2勝", "1000万": "2勝",
    "3勝": "3勝", "1600万": "3勝",
    "OP": "OP/重賞", "ｵｰﾌﾟﾝ": "OP/重賞", "オープン": "OP/重賞",
    "Ｇ１": "OP/重賞", "Ｇ２": "OP/重賞", "Ｇ３": "OP/重賞",
    "OP(L)": "OP/重賞", "(L)": "OP/重賞",
    "ＪＧ１": "OP/重賞", "ＪＧ２": "OP/重賞", "ＪＧ３": "OP/重賞",
}

# =========================================================
# ヘルパー
# =========================================================
def smile_from_dist(d: int | str) -> str:
    try:
        d = int(d)
        if d <= 1300: return "S"
        elif d <= 1899: return "M"
        elif d <= 2100: return "I"
        elif d <= 2700: return "L"
        else: return "E"
    except Exception:
        return "M"


def season_from_date(date_val) -> str:
    """日付(YYMMDD int or str) → 季節。"""
    try:
        s = str(int(date_val)).zfill(6)
        m = int(s[2:4])
        if m in [3, 4, 5]:   return "春"
        elif m in [6, 7, 8]: return "夏"
        elif m in [9, 10, 11]: return "秋"
        else: return "冬"
    except Exception:
        return "春"


def load_csv(path: Path) -> pd.DataFrame:
    for enc in ["cp932", "utf-8", "utf-8-sig", "shift_jis"]:
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            logger.info(f"読み込み完了: {path} ({enc}) {len(df):,}行")
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError(f"エンコーディング判定失敗: {path}")


# =========================================================
# 集計
# =========================================================
def build_trend(df: pd.DataFrame, min_n: int, start_yymmdd: int = 0) -> dict:
    """DataFrame → course_trend dict。"""
    # 前処理
    if start_yymmdd:
        df = df[pd.to_numeric(df["日付"], errors="coerce").fillna(0) >= start_yymmdd]
    df = df[df["場所"].isin(PLACES)].copy()
    df["race_key"] = df["レースID(新)"].astype(str).str.zfill(18).str[:16]
    df["smile"]    = df["距離"].apply(smile_from_dist)
    df["season"]   = df["日付"].apply(season_from_date)
    df["cls"]      = df["クラス名"].map(CLASS_NORMALIZE)
    df = df[df["cls"].notna() & df["芝・ダ"].notna()].copy()

    winners = df[df["確定着順"] == 1].copy()
    logger.info(f"有効行: {len(df):,}行 / 勝ち馬: {len(winners):,}行")

    result: dict = {}
    total = sufficient = 0

    for place in PLACES:
        result[place] = {}
        for shida in ["芝", "ダ"]:
            result[place][shida] = {}
            for sm in SMILES:
                result[place][shida][sm] = {}
                for cls in CLASSES:
                    result[place][shida][sm][cls] = {}
                    for season in SEASONS:
                        total += 1
                        mask = (
                            (df["場所"] == place) &
                            (df["芝・ダ"] == shida) &
                            (df["smile"] == sm) &
                            (df["cls"] == cls) &
                            (df["season"] == season)
                        )
                        sub     = df[mask]
                        n_races = sub["race_key"].nunique()

                        if n_races < min_n:
                            result[place][shida][sm][cls][season] = {
                                "n": n_races, "insufficient": True
                            }
                            continue

                        win = winners[mask.reindex(winners.index, fill_value=False)]
                        sufficient += 1

                        # 好調枠番 top3
                        waku_rate: dict[str, float] = {}
                        for w in range(1, 9):
                            wins_ = int((win["枠番"] == w).sum())
                            total_w = sub[sub["枠番"] == w]["race_key"].nunique()
                            if total_w >= 5:
                                waku_rate[f"{w}枠"] = round(wins_ / total_w * 100, 1)
                        top_waku = sorted(waku_rate, key=waku_rate.get, reverse=True)[:3]  # type: ignore

                        # 脚質ランク
                        style_rows: list[dict] = []
                        for st in ["逃げ", "先行", "中団", "後方", "ﾏｸﾘ"]:
                            wins_ = int((win["脚質"] == st).sum())
                            total_s = sub[sub["脚質"] == st]["race_key"].nunique()
                            if total_s >= 5:
                                style_rows.append({
                                    "脚質": st,
                                    "勝率": round(wins_ / total_s * 100, 1)
                                })
                        style_rows.sort(key=lambda x: -x["勝率"])

                        # 好調騎手 top5（出走10以上）
                        jockey_rows: list[dict] = []
                        for jk, grp in sub.groupby("騎手"):
                            total_j = grp["race_key"].nunique()
                            if total_j < 10:
                                continue
                            wins_ = int((win["騎手"] == jk).sum())
                            jockey_rows.append({
                                "騎手": jk,
                                "勝率": round(wins_ / total_j * 100, 1),
                                "出走": total_j,
                            })
                        jockey_rows.sort(key=lambda x: -x["勝率"])

                        # 好調調教師 top5（出走5以上）
                        trainer_rows: list[dict] = []
                        for tr, grp in sub.groupby("調教師"):
                            total_t = grp["race_key"].nunique()
                            if total_t < 5:
                                continue
                            wins_ = int((win["調教師"] == tr).sum())
                            trainer_rows.append({
                                "調教師": tr,
                                "勝率": round(wins_ / total_t * 100, 1),
                                "出走": total_t,
                            })
                        trainer_rows.sort(key=lambda x: -x["勝率"])

                        # 好調血統_父
                        sire_rows: list[dict] = []
                        for si, grp in sub.groupby("種牡馬"):
                            total_s = grp["race_key"].nunique()
                            if total_s < 5:
                                continue
                            wins_ = int((win["種牡馬"] == si).sum())
                            sire_rows.append({
                                "種牡馬": si,
                                "勝率": round(wins_ / total_s * 100, 1),
                                "出走": total_s,
                            })
                        sire_rows.sort(key=lambda x: -x["勝率"])

                        # 好調血統_母父
                        bms_rows: list[dict] = []
                        for bm, grp in sub.groupby("母父馬"):
                            total_b = grp["race_key"].nunique()
                            if total_b < 5:
                                continue
                            wins_ = int((win["母父馬"] == bm).sum())
                            bms_rows.append({
                                "母父馬": bm,
                                "勝率": round(wins_ / total_b * 100, 1),
                                "出走": total_b,
                            })
                        bms_rows.sort(key=lambda x: -x["勝率"])

                        result[place][shida][sm][cls][season] = {
                            "n":             n_races,
                            "好調枠番":      top_waku,
                            "脚質ランク":    style_rows,
                            "好調騎手":      jockey_rows[:5],
                            "好調調教師":    trainer_rows[:5],
                            "好調血統_父":   sire_rows[:5],
                            "好調血統_母父": bms_rows[:5],
                        }

        logger.info(f"  {place} 完了")

    logger.info(f"集計完了（季節別）: 総セル={total}, 有効={sufficient}({sufficient/total*100:.1f}%)")

    # 通年を追加（春夏秋冬の合算）
    tsuinen_added = 0
    for place_data in result.values():
        for shida_data in place_data.values():
            for sm_data in shida_data.values():
                for cls_data in sm_data.values():
                    valid = {s: v for s, v in cls_data.items() if not v.get("insufficient")}
                    total_n = sum(v["n"] for v in valid.values())
                    if total_n < min_n:
                        cls_data["通年"] = {"n": total_n, "insufficient": True}
                        continue
                    from collections import defaultdict
                    waku_count: dict = defaultdict(int)
                    for v in valid.values():
                        for w in v.get("好調枠番", []):
                            waku_count[w] += 1
                    top_waku = sorted(waku_count, key=waku_count.get, reverse=True)[:3]  # type: ignore
                    style_n: dict = defaultdict(float)
                    style_tot: dict = defaultdict(float)
                    for v in valid.values():
                        n = v["n"]
                        for row in v.get("脚質ランク", []):
                            style_n[row["脚質"]]   += row["勝率"] * n
                            style_tot[row["脚質"]] += n
                    style_rows = [
                        {"脚質": st, "勝率": round(style_n[st] / style_tot[st], 1)}
                        for st in style_n if style_tot[st] > 0
                    ]
                    style_rows.sort(key=lambda x: -x["勝率"])
                    def _merge(key, col):
                        acc: dict = defaultdict(list)
                        for v in valid.values():
                            for row in v.get(key, []):
                                acc[row[col]].append(row)
                        res = [
                            {col: name, "勝率": round(sum(r["勝率"] for r in rows)/len(rows), 1),
                             "出走": sum(r.get("出走", 0) for r in rows)}
                            for name, rows in acc.items()
                        ]
                        res.sort(key=lambda x: -x["勝率"])
                        return res[:5]
                    cls_data["通年"] = {
                        "n":             total_n,
                        "好調枠番":      top_waku,
                        "脚質ランク":    style_rows,
                        "好調騎手":      _merge("好調騎手",   "騎手"),
                        "好調調教師":    _merge("好調調教師", "調教師"),
                        "好調血統_父":   _merge("好調血統_父",   "種牡馬"),
                        "好調血統_母父": _merge("好調血統_母父", "母父馬"),
                    }
                    tsuinen_added += 1

    logger.info(f"通年追加: {tsuinen_added}セル")
    return result


# =========================================================
# メイン
# =========================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="course_trend.json 生成スクリプト")
    parser.add_argument("--input",  required=True, help="入力CSVパス")
    parser.add_argument("--output",        default="data/course_trend.json", help="出力JSONパス")
    parser.add_argument("--min_n",         type=int, default=30,     help="有効判定の最小レース数（デフォルト30）")
    parser.add_argument("--start_yymmdd",  type=int, default=0,      help="集計開始日付 YYMMDD（例: 160101 = 2016-01-01）")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error(f"入力ファイルが見つかりません: {input_path}")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"入力: {input_path}")
    logger.info(f"出力: {output_path}")
    logger.info(f"最小サンプル数: {args.min_n}")
    if args.start_yymmdd:
        logger.info(f"集計開始日付: {args.start_yymmdd}")

    df = load_csv(input_path)
    trend = build_trend(df, args.min_n, args.start_yymmdd)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(trend, f, ensure_ascii=False, indent=2)

    logger.info(f"出力完了: {output_path} ({output_path.stat().st_size/1024:.1f}KB)")


if __name__ == "__main__":
    main()
