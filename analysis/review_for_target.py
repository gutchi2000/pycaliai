# -*- coding: utf-8 -*-
"""回顧を TARGET の結果コメント欄へ手貼りしやすい『馬番順テキスト』で出力する.

build_uma_review.py が作った data/uma_review/{date}.json を読み、
各レースを馬番昇順(=TARGET出馬表の行順)に並べ、
「馬番 印 馬名 着: 回顧 → 次走」の1行プレーンテキストにする。
上から順にコピペできる。

usage:
  python analysis/review_for_target.py                 # 最新土日
  python analysis/review_for_target.py 20260606 20260607
"""
import sys
import json
import glob
import os

sys.stdout.reconfigure(encoding="utf-8")
BASE = r"E:\PyCaLiAI"
DBDIR = rf"{BASE}\data\uma_review"
REVDIR = rf"{BASE}\reports\review"
INDIR = rf"{BASE}\reports\cowork_input"


def latest_weekend():
    files = glob.glob(rf"{INDIR}\*_bundle.json")
    dates = sorted({os.path.basename(f).split("_")[0] for f in files}, reverse=True)
    return list(reversed(dates[:2]))


def main(dates):
    lines = []
    for date in dates:
        path = rf"{DBDIR}\{date}.json"
        if not os.path.exists(path):
            lines.append(f"# {date}: data/uma_review/{date}.json なし\n")
            continue
        recs = json.load(open(path, encoding="utf-8"))
        # レース単位にまとめ、馬番昇順
        by_race = {}
        for r in recs:
            by_race.setdefault((r["place"], r["R"]), []).append(r)
        lines.append(f"\n========== {date} ==========")
        for (place, rno) in sorted(by_race, key=lambda x: (x[0], x[1])):
            rs = sorted(by_race[(place, rno)], key=lambda x: x["umaban"])
            meta = rs[0]
            lines.append(f"\n■ {place}{rno}R {meta['course']} {meta['class']}")
            for r in rs:
                mk = r["mark"]
                txt = f"{r['kaiko']} → {r['jisou']}"
                lines.append(f"{r['umaban']:>2} {mk} {r['horse_name']} [{r['result']}]: {txt}")
    out = "\n".join(lines)
    weekend = f"{dates[0]}_{dates[-1][-4:]}" if len(dates) > 1 else dates[0]
    out_path = rf"{REVDIR}\{weekend}_target.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out)
    print(out)
    print(f"\n→ {out_path}")


if __name__ == "__main__":
    args = sys.argv[1:]
    dates = args if args else latest_weekend()
    main(dates)
