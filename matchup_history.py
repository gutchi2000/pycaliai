"""
matchup_history.py
==================
レース単位で「今回出走馬同士の過去対戦成績」を計算する。

データ源: master_v2_*.csv
- 走破タイム は master_v2 の「前走走破タイム」を chain reconstruction で取得
  (各馬の "次走" record の 前走 フィールド = その馬の "前走" 走破タイム)
- 過去 1 年限定、2 頭以上の対戦のみ抽出

Usage:
    from matchup_history import build_matchup_history
    matchup = build_matchup_history(current_races_by_id, days_back=365)
    # matchup[race_id] = [
    #     {race_id_past, date, place, race_name, course, track,
    #      horses: {horse_name: {pos, time}}},
    #     ...
    # ]
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

BASE = Path(__file__).parent
MASTER_CSV = BASE / "data" / "master_v2_20130105-20251228.csv"


def _format_time(t: str) -> str:
    """走破タイム文字列を整形。'118.3' → '1:58.3', '1:58.3' → '1:58.3'"""
    if not t or pd.isna(t):
        return ""
    t = str(t).strip()
    if ":" in t:
        return t  # 既に M:SS.s 形式
    try:
        sec = float(t)
        m = int(sec // 60)
        s = sec - m * 60
        if m > 0:
            return f"{m}:{s:04.1f}"
        return f"{s:.1f}"
    except (ValueError, TypeError):
        return t


def _load_master_for_matchup(days_back: int) -> Optional[pd.DataFrame]:
    """matchup 用に必要列だけ読み込む。"""
    if not MASTER_CSV.exists():
        logger.warning(f"master_v2 未存在: {MASTER_CSV}")
        return None
    try:
        usecols = [
            "日付", "馬名",
            "レースID(新)", "レース名",
            "前走レースID(新)", "前走走破タイム", "前走確定着順",
            "前走日付", "前走場所", "前芝・ダ", "前距離", "前走馬場状態",
        ]
        df = pd.read_csv(
            MASTER_CSV, encoding="utf-8-sig",
            usecols=usecols, dtype=str, low_memory=False, on_bad_lines="skip",
        )
        df["日付"] = pd.to_numeric(df["日付"], errors="coerce")
        df["前走確定着順"] = pd.to_numeric(df["前走確定着順"], errors="coerce")
        df["前距離"] = pd.to_numeric(df["前距離"], errors="coerce")
        # 直近 days_back 日間
        cutoff_dt = datetime.now() - timedelta(days=days_back)
        cutoff_int = int(cutoff_dt.strftime("%Y%m%d"))
        df = df[df["日付"] >= cutoff_int]
        logger.info(f"master_v2 matchup 用 load: {len(df):,} rows (直近 {days_back} 日)")
        return df
    except Exception as e:
        logger.error(f"master_v2 読み込み失敗: {e}")
        return None


def build_matchup_history(
    current_races_by_id: dict[str, list[str]],
    days_back: int = 365,
    min_overlap: int = 2,
    limit: int = 10,
) -> dict[str, list[dict]]:
    """各今回 race について、過去の対戦相手 race を抽出する。

    Args:
        current_races_by_id: {race_id: [horse_name, ...]} 今週末の出走表
        days_back: 過去何日まで遡るか
        min_overlap: 過去 race に 今回 race の馬が何頭出てたら出力するか
        limit: 1 race あたり最大何件の対戦相手 race を出すか (直近順)

    Returns:
        {race_id_today: [
            {race_id_past, date, place, race_name, course, track,
             horses: {horse_name: {pos, time}}},
            ...
        ]}
    """
    df = _load_master_for_matchup(days_back)
    if df is None or df.empty:
        return {}

    # (馬名, 前走レースID) → 前走 fields の辞書を構築
    # 走破タイム は chain reconstruction (各 row の 前走走破タイム = その馬の 前走 race のタイム)
    # 1 つの過去 race に複数 row がぶら下がるが、それぞれの row は別の馬の record なので
    # 結合した時に「その馬がその race でどう走ったか」が得られる。
    logger.info("matchup index 構築中...")
    horse_past_races: dict[str, dict[str, dict]] = defaultdict(dict)
    # 馬名 → {race_id_past: {pos, time, place, ...}}

    # race_name lookup: race_id(新) → レース名
    race_name_map: dict[str, str] = {}

    for _, r in df.iterrows():
        # race_name は本人の race の場合に取得 (今回の row の レースID(新) と レース名)
        rid_self = str(r["レースID(新)"]).strip() if pd.notna(r["レースID(新)"]) else ""
        if rid_self and rid_self not in race_name_map:
            rn = str(r["レース名"]).strip() if pd.notna(r["レース名"]) else ""
            race_name_map[rid_self] = rn

        # 前走 fields から過去 race の馬戦績を構築
        prev_rid = str(r["前走レースID(新)"]).strip() if pd.notna(r["前走レースID(新)"]) else ""
        if not prev_rid:
            continue
        if pd.isna(r["前走確定着順"]):
            continue
        hname = str(r["馬名"]).strip()
        if not hname:
            continue
        # 同じ馬・同じ過去 race の重複は overwrite (基本的に重複しないはず)
        horse_past_races[hname][prev_rid] = {
            "pos": int(r["前走確定着順"]),
            "time": _format_time(str(r["前走走破タイム"]) if pd.notna(r["前走走破タイム"]) else ""),
            "date": str(int(r["前走日付"])) if pd.notna(r["前走日付"]) else "",
            "place": str(r["前走場所"]).strip() if pd.notna(r["前走場所"]) else "",
            "surface": str(r["前芝・ダ"]).strip() if pd.notna(r["前芝・ダ"]) else "",
            "dist": int(r["前距離"]) if pd.notna(r["前距離"]) else None,
            "track": str(r["前走馬場状態"]).strip() if pd.notna(r["前走馬場状態"]) else "",
        }

    logger.info(f"  対象 馬数: {len(horse_past_races):,} 頭")
    logger.info(f"  race_name 辞書: {len(race_name_map):,} race")

    # 各今回 race について、出走馬の過去 race の union から min_overlap 以上の race を抽出
    matchup_result: dict[str, list[dict]] = {}
    n_total_matchups = 0
    for cur_rid, horse_list in current_races_by_id.items():
        # past_rid → {hname: data}
        past_overlap: dict[str, dict[str, dict]] = defaultdict(dict)
        for hname in horse_list:
            past_dict = horse_past_races.get(hname, {})
            for past_rid, data in past_dict.items():
                past_overlap[past_rid][hname] = data

        # min_overlap 以上の race だけ
        matched = []
        for past_rid, horses_data in past_overlap.items():
            if len(horses_data) >= min_overlap:
                # ref 情報は全 horse 同じ race のはず (race_id 一致なので)
                ref = next(iter(horses_data.values()))
                matched.append({
                    "race_id_past": past_rid,
                    "date": ref["date"],
                    "place": ref["place"],
                    "race_name": race_name_map.get(past_rid, ""),
                    "surface": ref["surface"],
                    "dist": ref["dist"],
                    "track": ref["track"],
                    "horses": {
                        hn: {"pos": d["pos"], "time": d["time"]}
                        for hn, d in horses_data.items()
                    },
                })

        # 直近順、上位 limit 件
        matched.sort(key=lambda x: -(int(x["date"]) if x["date"].isdigit() else 0))
        matched = matched[:limit]
        if matched:
            matchup_result[cur_rid] = matched
            n_total_matchups += len(matched)

    logger.info(f"  matchup 抽出: {len(matchup_result)} race / 計 {n_total_matchups} 件の過去対戦")
    return matchup_result


if __name__ == "__main__":
    import json
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    # サンプル: 5/23 bundle の現在出走馬で実行
    with open("reports/cowork_input/20260523_bundle.json", encoding="utf-8") as f:
        bundle = json.load(f)
    cur_races: dict[str, list[str]] = {}
    for race in bundle.get("races", []):
        rid = race["race_id"]
        names = [h.get("horse_name", "") for h in race.get("horses", []) if h.get("horse_name")]
        cur_races[rid] = names
    result = build_matchup_history(cur_races, days_back=365, min_overlap=2, limit=10)
    # 適当な race の結果を sample 表示
    sys.stdout.reconfigure(encoding="utf-8")
    for rid in list(result.keys())[:3]:
        print(f"\n=== race {rid} ({len(result[rid])} matchups) ===")
        for m in result[rid][:3]:
            print(f"  {m['date']} {m['place']} {m['race_name']} {m['surface']}{m['dist']} {m['track']}")
            for hn, d in m["horses"].items():
                print(f"     {hn}: {d['pos']}着 {d['time']}")
