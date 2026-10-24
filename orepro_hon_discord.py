"""
orepro_hon_discord.py — 俺プロ用「◎複勝1点・全レース」を Discord に会場別で送る。

形式（会場ごとに1メッセージ、1R〜12R 昇順）:
    東京１R　⑫グローリーリンク
    東京２R　①サンストックトン

根拠: analysis/orepro_weekly/REPORT.md（◎複勝1点・全R が期待pt最大、2021-25）。
実行: python orepro_hon_discord.py                # 最新 bundle
      python orepro_hon_discord.py --date 20261025
      python orepro_hon_discord.py --date 20261025 --dry   # 送らず表示のみ
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

BASE = Path(__file__).parent
sys.stdout.reconfigure(encoding="utf-8")
from t10_runner import notify  # Discord webhook 送信（未設定なら False）

CIRCLED = {i: chr(0x2460 + i - 1) for i in range(1, 21)}   # ①〜⑳
FULLW = str.maketrans("0123456789", "０１２３４５６７８９")


def latest_bundle() -> Path | None:
    files = sorted((BASE / "reports/cowork_input").glob("*_bundle.json"))
    return files[-1] if files else None


def build_messages(bundle: dict) -> list[str]:
    date = bundle["date"]
    by_place: dict[str, list[tuple[int, str]]] = {}
    for r in bundle["races"]:
        rid = str(r["race_id"]); rno = int(rid[-2:])
        place = r["race_meta"]["place"]
        hon = [h for h in r["horses"] if h.get("mark") == "◎"]
        if not hon:
            line = f"{place}{str(rno).translate(FULLW)}R　（◎なし）"
        else:
            h = hon[0]
            line = f"{place}{str(rno).translate(FULLW)}R　{CIRCLED.get(int(h['umaban']), h['umaban'])}{h['horse_name']}"
        by_place.setdefault(place, []).append((rno, line))
    msgs = []
    for place, rows in by_place.items():
        rows.sort()
        head = f"【俺プロ ◎複勝1点】{date[:4]}/{int(date[4:6])}/{int(date[6:])} {place}"
        msgs.append(head + "\n" + "\n".join(l for _, l in rows))
    return msgs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    path = (BASE / f"reports/cowork_input/{a.date}_bundle.json") if a.date else latest_bundle()
    if not path or not path.exists():
        print(f"bundle not found: {path}"); return 1
    bundle = json.loads(path.read_text(encoding="utf-8"))
    for m in build_messages(bundle):
        print(m); print("-" * 30)
        if not a.dry:
            print("sent" if notify(m) else "NOT sent (webhook 未設定/失敗)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
