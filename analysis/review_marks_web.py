# -*- coding: utf-8 -*-
"""最新土日 bundle.json の ◎〇▲△△ 印馬を全レース抽出する(Web回顧の入力).

kekka.csv が無い段階で「印が付いた5頭」を一覧化し、Web で着順を調べて
回顧する際の作業リストにする。

usage:
    python analysis/review_marks_web.py                # 最新の土日を自動検出
    python analysis/review_marks_web.py 20260606 20260607
"""
import sys
import json
import glob
import os

sys.stdout.reconfigure(encoding="utf-8")
BASE = r"E:\PyCaLiAI"
INDIR = rf"{BASE}\reports\cowork_input"

MARK_ORDER = {"◎": 0, "〇": 1, "○": 1, "▲": 2, "△": 3}


def latest_weekend():
    """cowork_input 内の *_bundle.json から最新2日(土日想定)を返す."""
    files = glob.glob(rf"{INDIR}\*_bundle.json")
    dates = sorted({os.path.basename(f).split("_")[0] for f in files}, reverse=True)
    return list(reversed(dates[:2]))  # 古い順(土→日)


def load_bundle(date):
    path = rf"{INDIR}\{date}_bundle.json"
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def race_no(rid):
    """race_id 末尾2桁 = レース番号."""
    try:
        return int(str(rid)[-2:])
    except Exception:
        return None


def fmt_odds(h):
    o = h.get("tansho_odds")
    return f"{o}" if o is not None else "?"


def main(dates):
    for date in dates:
        b = load_bundle(date)
        if not b:
            print(f"## {date}: bundle なし\n")
            continue
        races = sorted(b["races"], key=lambda r: (r["race_meta"].get("place", ""), race_no(r["race_id"]) or 0))
        print(f"# {date}  ({b.get('model')})  {len(races)}R\n")
        for r in races:
            m = r["race_meta"]
            rid = r["race_id"]
            marked = [h for h in r["horses"] if h.get("mark")]
            marked.sort(key=lambda h: MARK_ORDER.get(h["mark"], 9))
            rc = r.get("race_confidence", {})
            chaos = rc.get("field_chaos_score")
            chaos_s = f"chaos{chaos:.2f}" if chaos is not None else ""
            print(f"## {m.get('place')}{race_no(rid)}R {m.get('course')} {m.get('class')} "
                  f"({m.get('field_size')}頭) {chaos_s}  [{rid}]")
            for h in marked:
                print(f"  {h['mark']} {h['umaban']:>2}番 {h['horse_name']}  "
                      f"単{fmt_odds(h)}倍 rank{h['ai_rank']} "
                      f"p_win{h.get('p_win', 0):.3f} p_sho{h.get('p_sho', 0):.3f} "
                      f"vs:{h.get('ai_vs_market')}")
            print()


if __name__ == "__main__":
    args = sys.argv[1:]
    dates = args if args else latest_weekend()
    main(dates)
