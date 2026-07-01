# -*- coding: utf-8 -*-
"""PyCaLiAI の印/確率を TARGET frontier JV の「外部指数」CSV に書き出す.

TARGET 外部指数 取込仕様 (FAQ id=642):
  形式: 「馬単位CSV」  → 1 行 1 頭
  1 行: レースID(馬番付き),指数,順位
    レースID(馬番付き) = 16桁(新仕様) + 馬番2桁 = 18桁
        ※ export_target_comments.py の rid18 と同一。kekka「レースID(新)」と
          513/513 一致を検証済 (2026-06)。
    指数 = 整数(0〜999999) / 実数(0.0〜999.99) 等。「大きい方が優位」を選択。
    順位 = レース内 AI 順位 (ai_rank, 1=最上位)

  TARGET 側設定:
    1. 環境設定 →「外部指数の設定」で指数スロットを 1 つ登録
         - ファイル形式 : 馬単位CSV
         - レースID形式 : 16桁(新仕様) + 馬番  (= 18桁)
         - 指数表示桁  : 既定の指数 (整数 0〜999999)
         - 優位判定    : 大きい方が優位
         - フォルダ     : このスクリプトの出力先 (reports/target_index/)
    2. 出馬表/成績画面でメニュー「ファイル」→ 外部指数読込
    3. 指数列が出馬表にソート/絞り込み可能な列として表示される

入力: reports/cowork_input/{date}_bundle.json (export_weekly_marks.py の出力)
出力: reports/target_index/{date}_ai_{metric}.csv (cp932, ヘッダ無し)

指数の中身 (--metric):
  p_sho   複勝確率 (3着以内, 既定) ... 学習ターゲット = fukusho_flag で calibration 済
  p_plc   連対確率 (2着以内)
  p_win   単勝確率 (1着)
  ai_score 生 LambdaRank スコア (相対比較用、スケール任意)

usage:
  python analysis/export_target_index.py                    # 最新土日 (bundle 2 本)
  python analysis/export_target_index.py 20260607
  python analysis/export_target_index.py 20260606 20260607  # 複数日まとめて 1 ファイル
  python analysis/export_target_index.py 20260607 --metric p_win
"""
import sys
import os
import json
import glob
import argparse

sys.stdout.reconfigure(encoding="utf-8")
BASE = r"E:\PyCaLiAI"
INDIR = rf"{BASE}\reports\cowork_input"
OUTDIR = rf"{BASE}\reports\target_index"

# metric -> (bundle 内の horse キー, 整数化スケール)
#   probability は ×1000 で 0〜1000 の整数指数にする (TARGET 整数 0〜999999 / 大きい方が優位)
METRICS = {
    "p_sho": ("p_sho", 1000),
    "p_plc": ("p_plc", 1000),
    "p_win": ("p_win", 1000),
    "ai_score": ("ai_score", 100),  # 生スコア×100 (範囲 ≈ -300〜+50、負値あり→整数(-99999〜99999))
    "mark": ("mark", 1),           # 印を数値化 (下記 MARK_MAP)、大きい方が優位
}

# 印 → 指数値 (◎=5 … 無印=0)。BULLSEYE/IDEOGRAPHIC ZERO/BLACK・WHITE TRIANGLE。
MARK_MAP = {"◎": 5, "〇": 4, "○": 4, "▲": 3, "△": 2, "": 0}


def latest_weekend():
    files = glob.glob(rf"{INDIR}\*_bundle.json")
    dates = sorted({os.path.basename(f).split("_")[0] for f in files}, reverse=True)
    return list(reversed(dates[:2]))


def to_index(val, scale):
    """確率/スコア → TARGET 指数値 (整数化)."""
    if val is None:
        return None
    try:
        return int(round(float(val) * scale))
    except (TypeError, ValueError):
        return None


def main(dates, metric):
    key, scale = METRICS[metric]
    os.makedirs(OUTDIR, exist_ok=True)

    rows = []
    n_race = n_horse = n_skip = 0
    for date in dates:
        path = rf"{INDIR}\{date}_bundle.json"
        if not os.path.exists(path):
            print(f"# {date}: {path} なし → スキップ")
            continue
        bundle = json.load(open(path, encoding="utf-8"))
        for r in bundle.get("races", []):
            rid = str(r["race_id"])          # 16桁(馬番無)
            n_race += 1
            for h in r.get("horses", []):
                try:
                    ban = int(h["umaban"])
                except (KeyError, TypeError, ValueError):
                    n_skip += 1
                    continue
                rid18 = f"{rid}{ban:02d}"     # 18桁(馬番付き) = TARGET レースID(新)
                if metric == "mark":
                    idx = MARK_MAP.get((h.get("mark") or "").strip())
                else:
                    idx = to_index(h.get(key), scale)
                rank = h.get("ai_rank")
                if idx is None or rank is None:
                    n_skip += 1
                    continue
                rows.append(f"{rid18},{idx},{int(rank)}")
                n_horse += 1

    if not rows:
        print("出力対象 0 行 (bundle 未配置 or データ不足)")
        return 1

    weekend = f"{dates[0]}_{dates[-1][-4:]}" if len(dates) > 1 else dates[0]
    out_path = rf"{OUTDIR}\{weekend}_ai_{metric}.csv"
    with open(out_path, "w", encoding="cp932", errors="replace", newline="") as f:
        f.write("\n".join(rows) + "\n")

    print(f"指数 = {metric} (×{scale}, 大きい方が優位)")
    print(f"出力 {n_horse}頭 / {n_race}レース (skip {n_skip})")
    print(f"→ {out_path}")
    print("取込: TARGET 環境設定→外部指数の設定 で 馬単位CSV/18桁 を登録 → メニュー「ファイル」→外部指数読込")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dates", nargs="*", help="YYYYMMDD (省略時は最新土日)")
    ap.add_argument("--metric", default="p_sho", choices=list(METRICS) + ["all"],
                    help="指数に使う値 (default: p_sho=複勝確率、all=全指数を個別ファイルで出力)")
    args = ap.parse_args()
    dates = args.dates if args.dates else latest_weekend()
    if args.metric == "all":
        rc = 0
        for m in METRICS:
            rc |= main(dates, m)
        sys.exit(rc)
    sys.exit(main(dates, args.metric))
