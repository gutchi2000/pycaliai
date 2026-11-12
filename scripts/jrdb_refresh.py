# -*- coding: utf-8 -*-
"""
jrdb_refresh.py — JRDB 成績 (SED) の今年ぶんを取り直して data/jrdb/sed.parquet を更新する (2026-11-13)
==================================================================================
v7 候補 (JRDB 過去走特徴つき) は、出走表の各馬の「5 日以上前の走」を sed.parquet から引く。
先週末の成績が入っていないと過去走の順番が 1 つずれるので、開催前に毎週これを実行する。
  1. scripts/jrdb_fetch.py download --types Sed (今年ぶんの未取得ファイルだけ。既にあるものは飛ばす)
  2. Sed だけ固定長 → 表に変換して sed.parquet を置き換える (一時ファイルに書いてから差し替え)
JRDB はレース当日 08-19 時に取得制限がある。金曜の夜か土日の朝 8 時前に動かす
 (タスク PyCaLiAI_JRDB_Snapshot = 金土 19:40・土日 07:30 の capture の後ろで自動実行される)。
実行 (root から): python scripts/jrdb_refresh.py [--no-download]
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-download", action="store_true", help="取得せず、手元の zip から変換だけやり直す")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    year = dt.date.today().year
    if not a.no_download:
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "jrdb_fetch.py"), "download", "--types", "Sed", "--from", str(year), "--to", str(year)],
                           cwd=str(ROOT), env=dict(os.environ, PYTHONUTF8="1"), capture_output=True, text=True, encoding="utf-8", errors="replace")
        print((r.stdout or "").strip()[-400:])
        if r.returncode != 0:
            print(f"[jrdb_refresh] 取得に失敗 (rc={r.returncode}): {(r.stderr or r.stdout or '').strip()[-200:]}")
            return 1
    import jrdb_parse as jp
    import pandas as pd
    out = jp.OUT / "sed.parquet"
    before = int(pd.read_parquet(out, columns=["ymd"]).ymd.max()) if out.exists() else 0
    df = jp.parse_type("Sed")
    if len(df) < 700000:
        print(f"[jrdb_refresh] 変換結果が少なすぎる ({len(df):,} 行) → 置き換えない")
        return 1
    tmp = out.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out)
    print(f"[jrdb_refresh] sed.parquet {len(df):,} 行  最終 {before} → {int(df.ymd.max())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
