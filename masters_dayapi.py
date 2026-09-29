# -*- coding: utf-8 -*-
"""masters_dayapi.py — 学生大会 当日データAPI (読み取り専用)

公式サンプル `sample_当日データapi.py` の 2026 版。
  GET <url>
    header api-key: <api_key>
           type_of_data: "Racecards" | "Odds"
           id: Racecards=YYYYMMDD / Odds=YYYYMMDD+場コード(2)+R(2)
  Racecards → {"data": {"timetable": [...], "runtable": [...]}}  (朝 9 時までに用意)
  Odds      → {"data": {"odds_rt": [{race_id, odds_type, comb, odds}, ...]}}
              ★API サーバが JRA-VAN から取るのは **発走 4 分 30 秒前**

証明書は形だけなので verify=False (公式サンプルの指示どおり)。

接続先 url と api_key はコードに持たない。load_config() が
  環境変数 MASTERS_DAYAPI_URL / MASTERS_DAYAPI_KEY
  > data/masters_dayapi.json (.gitignore 済み。形は data/masters_dayapi.json.example)
の順で読み、どちらにも無ければ DayApiConfigError で止まる (既定値へは戻さない)。

★オッズは本システムの買い目には使わない。ワイドが**単一値**で返り、JV-Link の
  低〜高レンジが無い。事前登録 policy は mid=(low+high)/2 で定義されているため、
  意味の違う値を混ぜると残差の定義が変わる。オッズ源は JV-Link のままにする。
  ここで使うのは timetable の**公式発走時刻**（当日の時刻変更に追随できる）。

id の桁構成が 2 系統あるので注意:
  投票 race_id (netkeiba 12桁) = 年(4) + 場(2) + 回(2) + 日(2) + R(2)
  Odds の id   (12桁)          = 年月日(8) + 場(2) + R(2)
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE = Path(__file__).resolve().parent
CONFIG_PATH = BASE / "data" / "masters_dayapi.json"
ENV_URL = "MASTERS_DAYAPI_URL"
ENV_API_KEY = "MASTERS_DAYAPI_KEY"

# 投票の臨界パス (T-4:30 起動 → T-3:20 までに POST) で呼ばれる。
# 落ちても weekly CSV に即フォールバックできるので、待たずに諦める。
TIMEOUT_SEC = 6

# 場所コード: 札幌=01 … 小倉=10 (公式サンプルのヘッダコメント)
PLACES = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"]
PLACE_CODE = {name: f"{i + 1:02d}" for i, name in enumerate(PLACES)}


class DayApiError(RuntimeError):
    """当日データAPI が使えない。呼び出し側はフォールバックすること。"""


class DayApiConfigError(DayApiError):
    """接続先か api-key が設定されていない。既定値へは戻さず、ここで止める。"""


# ---------------------------------------------------------------- 設定
def load_config() -> dict:
    """接続先と api-key を返す。環境変数 > data/masters_dayapi.json の順。

    netkeiba_api.load_config と同じ読み方。どちらにも無い項目があれば
    DayApiConfigError (fail-closed)。例外文には項目名だけを載せ、値は載せない。
    """
    cfg: dict = {}
    try:
        loaded = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            cfg = loaded
    except (OSError, ValueError):
        cfg = {}
    url = os.environ.get(ENV_URL) or str(cfg.get("url") or "")
    api_key = os.environ.get(ENV_API_KEY) or str(cfg.get("api_key") or "")
    cfg["url"] = url.strip()
    cfg["api_key"] = api_key.strip()
    missing = [name for name in ("url", "api_key") if not cfg[name]]
    if missing:
        raise DayApiConfigError(
            f"当日データAPI の設定が無い: {', '.join(missing)} "
            f"(環境変数 {ENV_URL} / {ENV_API_KEY} か data/{CONFIG_PATH.name})")
    return cfg


def _get(type_of_data: str, rid: str) -> Any:
    cfg = load_config()
    headers = {"api-key": cfg["api_key"], "type_of_data": type_of_data, "id": str(rid)}
    try:
        r = requests.get(cfg["url"], headers=headers, verify=False, timeout=TIMEOUT_SEC)
    except requests.RequestException as exc:
        raise DayApiError(f"{type_of_data} {rid}: 通信失敗 ({type(exc).__name__})") from exc
    if r.status_code != 200:
        raise DayApiError(f"{type_of_data} {rid}: HTTP {r.status_code} {r.text[:200]}")
    try:
        body = r.json()
    except ValueError as exc:
        raise DayApiError(f"{type_of_data} {rid}: JSON でない応答") from exc
    data = body.get("data")
    # 未生成の日付は {"message":"Data None","data":[]} が返る
    if not data or not isinstance(data, dict):
        raise DayApiError(f"{type_of_data} {rid}: データ未生成 ({body.get('message')})")
    return data


def racecards(date_str: str) -> tuple[list[dict], list[dict]]:
    """(timetable, runtable) を返す。未生成なら DayApiError。"""
    data = _get("Racecards", date_str)
    return list(data.get("timetable") or []), list(data.get("runtable") or [])


def odds_id(rid16: str) -> str:
    """TARGET 16 桁 → Odds API の id (年月日8 + 場2 + R2)。"""
    rid = re.sub(r"\D", "", str(rid16 or ""))[:16]
    if len(rid) != 16:
        raise DayApiError(f"race_id が 16 桁でない: {rid!r}")
    return rid[:8] + rid[8:10] + rid[14:16]


def post_times(date_str: str) -> dict[str, str]:
    """{rid16: 'HH:MM'} を公式 timetable から作る。

    timetable は (place, race_num, start_time)、runtable の race_id は 18 桁
    (= 16 桁レースキー + 馬番 2 桁)。両者を (place, race_num) で突き合わせる。
    """
    timetable, runtable = racecards(date_str)
    by_key: dict[tuple[str, int], str] = {}
    for row in runtable:
        rid = re.sub(r"\D", "", str(row.get("race_id") or ""))[:16]
        if len(rid) != 16:
            continue
        try:
            key = (str(row.get("place")), int(row.get("race_num")))
        except (TypeError, ValueError):
            continue
        by_key.setdefault(key, rid)

    out: dict[str, str] = {}
    for row in timetable:
        try:
            key = (str(row.get("place")), int(row.get("race_num")))
        except (TypeError, ValueError):
            continue
        rid = by_key.get(key)
        start = str(row.get("start_time") or "").strip()
        if rid and re.match(r"^\d{1,2}:\d{2}$", start):
            out[rid] = start
    if not out:
        raise DayApiError(f"{date_str}: timetable から発走時刻を作れない")
    return out


def _main() -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="当日データAPI の確認 (読み取り専用)")
    ap.add_argument("date", help="YYYYMMDD")
    ap.add_argument("--odds", default=None, help="16 桁 race_id のオッズを表示")
    args = ap.parse_args()
    try:
        if args.odds:
            data = _get("Odds", odds_id(args.odds))
            rows = data.get("odds_rt") or []
            kinds: dict[int, int] = {}
            for row in rows:
                kinds[row.get("odds_type")] = kinds.get(row.get("odds_type"), 0) + 1
            print(f"odds_rt {len(rows)}件 種別内訳 {dict(sorted(kinds.items()))}")
            return 0
        times = post_times(args.date)
        print(f"{len(times)}R の公式発走時刻:")
        for rid, hhmm in sorted(times.items(), key=lambda kv: kv[1]):
            print(f"  {rid}  {hhmm}")
    except DayApiError as exc:
        print(f"[DayApiError] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
