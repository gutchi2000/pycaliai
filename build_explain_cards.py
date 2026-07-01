# -*- coding: utf-8 -*-
"""
build_explain_cards.py — 指定日の全レース分の説明カードを site 用に書き出す
================================================================================
explain_race.cards_for_date() を使い site/data/explain/{date}.json + manifest.json を生成。
静的サイト(explain.html)が読み込んで分析カードを描画する。

  python build_explain_cards.py 20250608         # 指定日
  EXPLAIN_NO_LLM=1 python build_explain_cards.py 20250608   # LLM解説なし(高速)

注: ELO/Glicko/脚質は data/*_feats.parquet の被覆期間(〜2025)が必要。被覆外の馬は
    数値が空欄になる(核=印/勝率/予想/基準/EV は score cache があれば常時出る)。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import explain_race as EX   # stdout ラップは explain_race 側で1回だけ(ここでは再ラップしない)

BASE = Path(__file__).parent
OUTDIR = BASE / "site" / "data" / "explain"


def main():
    date = sys.argv[1] if len(sys.argv) > 1 else "20250608"
    cards = EX.cards_for_date(date)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    payload = {"date": date, "n_races": len(cards),
               "races": sorted(cards, key=lambda c: c["race"]["rid"])}
    (OUTDIR / f"{date}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    dates = sorted([p.stem for p in OUTDIR.glob("*.json") if p.stem != "manifest"], reverse=True)
    (OUTDIR / "manifest.json").write_text(
        json.dumps({"dates": dates}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {OUTDIR / (date + '.json')}  races={len(cards)}  manifest_dates={len(dates)}")


if __name__ == "__main__":
    main()
