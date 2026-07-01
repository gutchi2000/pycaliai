# -*- coding: utf-8 -*-
"""
build_realized_bias.py — レース結果(TARGET 払戻成績CSV)から「実現トラックバイアス」を算出。
================================================================================
fetch_baba_today.py の "今日の馬場バイアス"(=クッション帯→13年プライア早見) とは別物。
こちらは【実際に走った結果】から、会場×面ごとに 前残り率(脚質) と 内外(枠) を出す。
ユーザードクトリン「同開催・翌日は"今日の実現"を一次・プライアは補助」の一次シグナル側。

入力: TARGET 払戻成績エクスポート(cp932, 1行=1レース, 174列, test.csv 形式)
  [0]レースID(YYYYMMDD+場2桁+...) [4]芝(0)/ダ(1) [5]平(0)/障(1) [6]距離 [7]頭数
  [13]1着馬番 [14]2着 [15]3着 [97/96/95/94]1着馬4/3/2/1角通過順 [172]馬場状態コード
出力: data/realized_bias.json  (build_site.py が site/data/ へ同梱→出走表タブの妙味隣に表示)
実行: PYTHONUTF8=1 ./venv311/Scripts/python.exe build_realized_bias.py <results.csv> [--date YYYYMMDD]
"""
from __future__ import annotations
import csv, datetime, json, sys
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).parent
OUT = BASE / "data" / "realized_bias.json"
PLACE = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
         "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}
BABA = {"1": "良", "2": "稍重", "3": "重", "4": "不良"}
FRONT_Q = 0.34   # 道中位置(正規化)がこれ以下なら「前(逃げ・先行)」


def _i(x):
    try:
        return int(x)
    except (ValueError, TypeError):
        return None


def build(csv_path: str, date: str | None = None) -> dict:
    rows = list(csv.reader(open(csv_path, encoding="cp932", errors="replace")))[1:]
    agg: dict[tuple[str, str], list] = defaultdict(list)
    seen_date = None
    for r in rows:
        if not r or len(r) < 174 or not r[0][:8].isdigit():
            continue
        seen_date = seen_date or r[0][:8]
        if date and r[0][:8] != date:
            continue
        place = PLACE.get(r[0][8:10])
        if not place or r[5].strip() == "1":          # 不明場 or 障害は除外
            continue
        surf = "芝" if r[4].strip() == "0" else "ダ"
        n, w = _i(r[7]), _i(r[13])
        if not n or not w or n < 2:
            continue
        pos = next((_i(r[c]) for c in (97, 96, 95, 94) if _i(r[c])), None)
        agg[(place, surf)].append((n, w, pos, BABA.get(r[172].strip(), "?")))

    venues: dict[str, dict] = {}
    for (place, surf), v in sorted(agg.items()):
        nrm = [(w - 1) / (n - 1) for n, w, *_ in v]                       # 0=最内,1=最外
        inner = [1 for n, w, *_ in v if w <= n / 2]
        fpos = [(pos - 1) / (n - 1) for n, w, pos, *_ in v if pos]        # 0=前,1=後
        front = [1 for n, w, pos, *_ in v if pos and (pos - 1) / (n - 1) <= FRONT_Q]
        draw_avg = sum(nrm) / len(nrm)
        inner_half = len(inner) / len(v)
        front_rate = len(front) / len(v)
        babas = sorted({x[3] for x in v})
        # 枠の言い回し(弱信号・小標本前提で控えめに)
        if draw_avg <= 0.40 and inner_half >= 0.60:
            waku = "内"
        elif draw_avg >= 0.60 and inner_half <= 0.40:
            waku = "外"
        else:
            waku = "フラット"
        line = (f"{place}{surf}: 前残り{front_rate*100:.0f}% / 枠{waku}"
                f"(内半{inner_half*100:.0f}%) ・{('・'.join(babas))} ・{len(v)}R")
        venues[f"{place}|{surf}"] = {
            "place": place, "surf": surf, "n": len(v),
            "front_rate": round(front_rate, 3), "draw_avg": round(draw_avg, 3),
            "inner_half": round(inner_half, 3), "waku": waku,
            "baba": "・".join(babas), "line": line,
        }
    # source_date = 結果の日(土) / show_on_date = それを表示する翌開催日(日)。
    # ドクトリン: 同開催・翌日は前日の"実現"を一次シグナルにする(土の結果→日のカード)。
    d = date or seen_date or ""
    label, show_on = "実現", ""
    if len(d) == 8 and d.isdigit():
        dt = datetime.date(int(d[:4]), int(d[4:6]), int(d[6:8]))
        wd = "月火水木金土日"[dt.weekday()]
        label = f"{dt.month}/{dt.day}({wd})実現"
        show_on = (dt + datetime.timedelta(days=1)).strftime("%Y%m%d")
    return {"source_date": d, "show_on_date": show_on, "label": label,
            "front_q": FRONT_Q, "venues": venues}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    date = None
    if "--date" in sys.argv:
        date = sys.argv[sys.argv.index("--date") + 1]
    if not args:
        sys.exit("使い方: build_realized_bias.py <results.csv> [--date YYYYMMDD]")
    payload = build(args[0], date)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[saved] {OUT}  ({len(payload['venues'])}セグメント / {payload['label']})")
    for k, v in payload["venues"].items():
        print("  ", v["line"])


if __name__ == "__main__":
    main()
