# -*- coding: utf-8 -*-
"""
v8_market_gate.py — v8 候補 (v7 + JRDB 前日 KYI + 外厩) のスコアを共通ゲートで測る (2026-11-15)
=================================================================================
入力: reports/v8_scores.parquet (train_v8_kyi.py が保存。seed 42 の V7 / V8K / V8 のスコア、valid 2023 + test 2024-25)
ゲート (gate_common): P = 本番 v7 → +z、M = 確定市場+v7 → +z、M0 = 確定市場 → +z、A10 = 10時市場+v7 → +z。
  z_V7  … 同じ条件で学習し直した v7 (対照。≈0 のはず)
  z_V8K … KYI の前日指数・展開予想を足した版
  z_V8  … KYI + 外厩
読み方: P は「v7 を良くするか」、A10 は「朝の時点 (10 時オッズ) の予想を良くするか」= 朝の補正印に効くか、
        M は「締切市場の外の情報か」。年別 (2024 / 2025) も出す。
出力: reports/v8_market_gate.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_common as gc  # noqa: E402
import pace_forward_problem_exp as pfx  # noqa: E402


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    z = pd.read_parquet(gc.REP / "v8_scores.parquet")
    res = {}
    for arm in ("V7", "V8K", "V8"):
        c = f"s_{arm}"
        g = gc.gate(z, [c], name=arm)
        res[arm] = pfx.short(g)
        res[arm]["by_year"] = pfx.gate_years(z, [c])
    (gc.REP / "v8_market_gate.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for arm, r in res.items():
        print(arm, {k: {kk: round(vv * 1e4, 2) for kk, vv in v.items()} for k, v in r["by_year"].items()})


if __name__ == "__main__":
    main()
