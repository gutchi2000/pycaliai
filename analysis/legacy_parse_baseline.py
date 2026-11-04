# -*- coding: utf-8 -*-
"""
legacy_parse_baseline.py — docs/LEGACY_RETIREMENT_PLAN_20261004.md 段階 1 の受け入れ条件 #1
「parse_csv の出力が切り出し前後で同一」を確かめるためのスナップショット採取と照合。

  python -m analysis.legacy_parse_baseline dump --out <dir>     # 2026 年の data/weekly/*.csv 全部を parse_csv → pickle
  python -m analysis.legacy_parse_baseline check --out <dir>    # 現在の parse_csv と pickle を比較 (値・列順・dtype)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))


def _csvs():
    return sorted((BASE / "data" / "weekly").glob("2026????.csv"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dump", "check"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    logging.disable(logging.CRITICAL)
    from predict_weekly import parse_csv
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files = _csvs()
    if a.limit:
        files = files[-a.limit:]
    bad = 0
    for p in files:
        try:
            df = parse_csv(p)
        except Exception as e:  # noqa: BLE001
            print(f"{p.stem}: parse_csv 例外 {e}")
            bad += 1
            continue
        pk = out / f"{p.stem}.pkl"
        if a.cmd == "dump":
            df.to_pickle(pk)
            print(f"{p.stem}: {df.shape} → {pk.name}", flush=True)
        else:
            if not pk.exists():
                print(f"{p.stem}: 基準なし"); bad += 1; continue
            ref = pd.read_pickle(pk)
            same_cols = list(ref.columns) == list(df.columns)
            same_dtype = same_cols and all(str(ref[c].dtype) == str(df[c].dtype) for c in ref.columns)
            try:
                pd.testing.assert_frame_equal(ref, df, check_dtype=False, check_like=False)
                same_vals = True
            except AssertionError as e:
                same_vals = False
                msg = str(e).splitlines()[0]
            ok = same_cols and same_dtype and same_vals
            bad += 0 if ok else 1
            print(f"{p.stem}: {'OK' if ok else 'NG'} cols={same_cols} dtype={same_dtype} vals={same_vals}" + ("" if same_vals else f" ({msg})"), flush=True)
    print(f"[legacy_parse_baseline] {a.cmd}: {len(files)} files, NG={bad}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
