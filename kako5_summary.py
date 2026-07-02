"""
kako5_summary.py
================
週次 kako5 CSV (data/kako5/{YYYYMMDD}.csv) から、advisor 用の
narrative-ready history を馬ごとに抽出する。

出力する history dict は bundle.json の horses[].history に埋め込まれ、
Cowork (LLM) が自由日本語で advisor コメントを書く材料となる。

設計理念: 「自由と徹底」
  - PyCaLiAI 側は事実 (着順・人気・上り3F・脚質ラベル等) を漏れなく渡す
  - 文章生成は 100% Cowork (LLM) の自由日本語。テンプレ無し

Usage:
    from kako5_summary import build_histories
    histories = build_histories(Path("data/kako5/20260517.csv"))
    # histories[(race_id_str, umaban_int)] → dict
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _safe_float(v) -> Optional[float]:
    try:
        f = float(v)
        if f != f:  # NaN
            return None
        return f
    except (ValueError, TypeError):
        return None


def _safe_int(v) -> Optional[int]:
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None


def _td_to_label(td: str) -> str:
    """TARGET の TD コード → 表示ラベル"""
    return {
        "T": "芝", "D": "ダ", "S": "障",
        "芝": "芝", "ダ": "ダ", "ダート": "ダ",
    }.get(str(td).strip(), str(td).strip())


def _has_deogure(style: str) -> bool:
    """脚質ラベル文字列から「出遅れ」を判定。

    注意: 現行 TARGET エクスポート(46/72列)の決手は 逃げ/先行/中団/後方/
    追込/ﾏｸﾘ のみで「出遅」を含まないため、deogure_count は実質常に 0。
    出遅情報を得るには TARGET 側で備考/コメント列を含む形式が必要。
    """
    if not style:
        return False
    s = str(style)
    return any(k in s for k in ["出遅", "ﾃﾞｵ", "デオ"])


# 過去1〜5走の column offset (実CSVで実測検証済み 2026-07-03):
#   race i → (場所, TD, 距離, 馬場, 着順, 人気, レース名, 上3F, 決手, 前走間隔)
# 注: 旧版は base+6 を style、base+8 を weight_change と誤ラベルしていた
#     (実体はレース名と決手)。馬体重変化はこの CSV に存在しない。
PAST_RUN_BASE = [14, 26, 38, 50, 62]


def _extract_run(row: list[str], base: int) -> Optional[dict]:
    """row の base 列から 1 走分を抽出。着順が無効なら None."""
    if base + 9 >= len(row):
        return None
    pos = _safe_int(row[base + 4])
    if pos is None or pos == 0:
        return None
    dist = _safe_float(row[base + 2])
    return {
        "place": (row[base + 0] or "").strip() or None,
        "td": _td_to_label(row[base + 1] or "") or None,
        "dist": int(dist) if dist else None,
        "track": (row[base + 3] or "").strip() or None,
        "pos": pos,
        "ninki": _safe_int(row[base + 5]),
        "race_name": (row[base + 6] or "").strip() or None,
        "agari3f": _safe_float(row[base + 7]),
        "style": (row[base + 8] or "").strip() or None,
        "interval_weeks": _safe_int(row[base + 9]),
    }


def _build_history(past_runs: list[dict],
                   current_td: Optional[str],
                   current_dist: Optional[int],
                   current_place: Optional[str]) -> dict:
    """過去走リストから history dict を組み立て。"""
    n = len(past_runs)
    h = {"n_runs": n}

    if n == 0:
        return h

    positions = [r["pos"] for r in past_runs if r["pos"] is not None]
    if positions:
        h["avg_pos"] = round(sum(positions) / len(positions), 2)
        h["best_pos"] = min(positions)
        # トレンド: 直近を先頭にした index に対する着順の傾き
        # positions=[recent, ..., older] vs x=[0,1,...] の最小二乗。
        # 正 = 古い方ほど着順が悪い ≒ 形上昇 (recent が古いより良い)
        # 負 = 古い方ほど着順が良い ≒ 形下降
        if len(positions) >= 2:
            x_mean = (len(positions) - 1) / 2.0
            y_mean = sum(positions) / len(positions)
            num = sum((i - x_mean) * (p - y_mean)
                      for i, p in enumerate(positions))
            den = sum((i - x_mean) ** 2 for i in range(len(positions)))
            h["pos_trend"] = round(num / den, 3) if den > 0 else 0.0
        else:
            h["pos_trend"] = 0.0

    # 同条件率
    if current_td:
        same = sum(1 for r in past_runs if r.get("td") == current_td)
        h["same_td_ratio"] = round(same / n, 2)
    if current_dist:
        same = sum(1 for r in past_runs
                   if r.get("dist") is not None
                   and abs(r["dist"] - current_dist) <= 200)
        h["same_dist_ratio"] = round(same / n, 2)
    if current_place:
        same = sum(1 for r in past_runs if r.get("place") == current_place)
        h["same_place_ratio"] = round(same / n, 2)

    # 出遅れ回数 (脚質ラベルから判定)
    h["deogure_count"] = sum(1 for r in past_runs if _has_deogure(r.get("style") or ""))

    # 過去走の詳細リスト (1走前から順、最大5走)
    runs = []
    for i, r in enumerate(past_runs):
        runs.append({
            "n_ago": i + 1,
            **{k: v for k, v in r.items() if v is not None},
        })
    h["runs"] = runs

    return h


def build_histories(kako5_path: Path) -> dict[tuple[str, int], dict]:
    """kako5 CSV を読み、(race_id, umaban) → history dict のマップを返す。

    Returns:
        { (race_id_str, umaban_int): history_dict, ... }
    """
    if not kako5_path.exists():
        logger.warning(f"kako5 CSV 未存在: {kako5_path}")
        return {}

    logger.info(f"kako5 履歴抽出: {kako5_path.name}")
    histories: dict[tuple[str, int], dict] = {}
    current_header: Optional[dict] = None

    with open(kako5_path, encoding="cp932", errors="replace") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) <= 1:
                continue
            # レースヘッダ行 (19 列, 先頭にレースID)
            if len(row) == 19 and row[0] and len(row[0]) >= 10 and row[0][:4].isdigit():
                current_header = {
                    "race_id": row[0],
                    "place": (row[3] or "").strip(),
                    "td": _td_to_label(row[8] or ""),
                    "dist": int(_safe_float(row[9]) or 0) or None,
                }
                continue
            # データ行 (72 列, 先頭が数字)
            if len(row) == 72 and row[0].isdigit() and current_header:
                uma_ban = _safe_int(row[2])
                if uma_ban is None:
                    continue
                past_runs = []
                for base in PAST_RUN_BASE:
                    r = _extract_run(row, base)
                    if r is not None:
                        past_runs.append(r)
                hist = _build_history(
                    past_runs,
                    current_td=current_header["td"],
                    current_dist=current_header["dist"],
                    current_place=current_header["place"],
                )
                histories[(current_header["race_id"], uma_ban)] = hist

    logger.info(f"  {len(histories)} 頭分の history を抽出")
    return histories


def build_horse_facts(kako5_path: Path) -> dict[tuple[str, int], dict]:
    """kako5 CSV から (race_id, umaban) → {sex, age} の static fact map を返す。

    kako5 列 layout (build_histories と同じ): col9=性別, col10=年齢
    """
    if not kako5_path.exists():
        return {}
    facts: dict[tuple[str, int], dict] = {}
    current_rid: Optional[str] = None
    with open(kako5_path, encoding="cp932", errors="replace") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) <= 1:
                continue
            if len(row) == 19 and row[0] and len(row[0]) >= 10 and row[0][:4].isdigit():
                current_rid = row[0]
                continue
            if len(row) == 72 and row[0].isdigit() and current_rid:
                uma_ban = _safe_int(row[2])
                if uma_ban is None:
                    continue
                sex = (row[9] or "").strip() or None
                age = _safe_int(row[10])
                if sex or age is not None:
                    facts[(current_rid, uma_ban)] = {"sex": sex, "age": age}
    return facts


if __name__ == "__main__":
    import argparse
    import json
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYYMMDD")
    args = ap.parse_args()

    kako5_path = Path(f"data/kako5/{args.date}.csv")
    hist = build_histories(kako5_path)
    if not hist:
        print("no histories", file=sys.stderr)
        sys.exit(1)

    # 最初の 1 件をサンプル出力
    first_key = next(iter(hist))
    print(f"sample key: {first_key}")
    print(json.dumps(hist[first_key], ensure_ascii=False, indent=2))
    print(f"\ntotal: {len(hist)} horses")
