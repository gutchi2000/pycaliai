# -*- coding: utf-8 -*-
"""回顧を TARGET frontier JV の「ファイルからのコメント一括登録」用CSVに書き出す.

TARGET 取込仕様 (FAQ id=611):
  形式: 開催名,馬名,レースID(馬番有り),コメント   ((ANY),(ANY),レースID,コメント パターン)
  レースID(馬番有り) = 新仕様18桁 = race_id(16桁) + 馬番(2桁ゼロ詰め)  ※馬番必須
  取込: メインメニュー →「ファイルからのコメント一括登録」→ このCSVを指定
  ※race_id+馬番 == kekkaの「レースID(新)」を検証済(513/513一致, 2026-06)

入力: data/uma_review/{date}.json (build_uma_review.py の出力)
出力: reports/review/{first}_{last}_target_import.csv (cp932, ヘッダ無し)

usage:
  python analysis/export_target_comments.py                       # 最新土日
  python analysis/export_target_comments.py 20260606 20260607
  python analysis/export_target_comments.py 20260418 20260419 ... # 複数日まとめて1ファイル
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


def sanitize(s):
    """CSV/cp932 を壊す文字を置換."""
    if s is None:
        return ""
    s = str(s)
    s = s.replace(",", "、").replace("\n", " ").replace("\r", " ")
    s = s.replace("〜", "～")  # cp932 で安全な全角チルダへ
    return s


def main(dates):
    rows = []
    n_total = n_miss = 0
    for date in dates:
        path = rf"{DBDIR}\{date}.json"
        if not os.path.exists(path):
            print(f"# {date}: {path} なし → スキップ")
            continue
        recs = json.load(open(path, encoding="utf-8"))
        for r in recs:
            rid18 = f"{r['race_id']}{int(r['umaban']):02d}"  # 18桁(馬番有り)
            kaisai = sanitize(f"{r['place']}{r['R']}R")
            name = sanitize(r["horse_name"])
            comment = sanitize(f"{r['kaiko']} → {r['jisou']}")
            rows.append(f"{kaisai},{name},{rid18},{comment}")
            n_total += 1
            if r.get("ai_miss"):
                n_miss += 1

    weekend = f"{dates[0]}_{dates[-1][-4:]}" if len(dates) > 1 else dates[0]
    out_path = rf"{REVDIR}\{weekend}_target_import.csv"
    with open(out_path, "w", encoding="cp932", errors="replace", newline="") as f:
        f.write("\n".join(rows) + "\n")
    print(f"出力 {n_total}行 (うち取りこぼし無印 {n_miss}行)")
    print(f"→ {out_path}")
    print("取込: TARGET メインメニュー →「ファイルからのコメント一括登録」→ 上記CSVを指定")


if __name__ == "__main__":
    args = sys.argv[1:]
    dates = args if args else latest_weekend()
    main(dates)
