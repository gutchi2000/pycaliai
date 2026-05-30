"""
run_v3_pipeline.py
==================
unified_rank_v3 (Optuna-tuned) で以下を一気通貫:
  1. PL calibrators 再 fit (valid=2023) → pl_calibrators_v3.pkl
  2. payout_curve 再構築 (train+valid ≤2023) → pl_payout_curve_v3.pkl
  3. backtest_pl_kelly 相当の OOS ROI 測定 (v3 vs v1 比較)

既存スクリプトのモジュール定数を一時差し替えして main() を呼ぶ方式。
"""
from __future__ import annotations
import io, sys, warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
# stdout wrap will be set by first imported module (build_pl_calibrators)

BASE = Path(__file__).parent
V3_MODEL = BASE / "models/unified_rank_v3.pkl"
V3_CAL   = BASE / "models/pl_calibrators_v3.pkl"
V3_CURVE = BASE / "data/pl_payout_curve_v3.pkl"


def step1_calibrators():
    print("\n" + "=" * 70)
    print("[STEP 1] PL calibrators (v3)")
    print("=" * 70)
    import build_pl_calibrators as bc
    bc.MODEL_PKL = V3_MODEL
    bc.OUT_PKL   = V3_CAL
    # stdout already wrapped; avoid double wrap
    bc.sys = sys  # replace sys so its stdout wrap is no-op? easier: prevent wrap
    # the module's sys.stdout = io.TextIOWrapper has already run at import; need to undo
    # simpler: just call main (it uses print, should be OK)
    bc.main()


def step2_curve():
    print("\n" + "=" * 70)
    print("[STEP 2] payout curve (v3)")
    print("=" * 70)
    import build_payout_curve as bp
    bp.MODEL_PKL = V3_MODEL
    bp.CAL_PKL   = V3_CAL
    bp.OUT_PKL   = V3_CURVE
    bp.main()


def step3_backtest():
    print("\n" + "=" * 70)
    print("[STEP 3] backtest_pl_kelly (v3)")
    print("=" * 70)
    import backtest_pl_ev as be
    be.MODEL_PKL = V3_MODEL
    be.CAL_PKL   = V3_CAL
    be.CURVE_PKL = V3_CURVE
    import backtest_pl_kelly as bk
    bk.MODEL_PKL = V3_MODEL
    bk.CAL_PKL   = V3_CAL
    bk.CURVE_PKL = V3_CURVE
    bk.OUT_JSON  = BASE / "reports/backtest_pl_kelly_v3.json"
    bk.main()


def main():
    print("=" * 70)
    print("run_v3_pipeline.py")
    print("=" * 70)
    print(f"V3 model: {V3_MODEL}")
    assert V3_MODEL.exists(), "unified_rank_v3.pkl not found"

    step1_calibrators()
    step2_curve()
    step3_backtest()

    print("\n" + "=" * 70)
    print("DONE: v3 pipeline. Compare reports/backtest_pl_kelly_v3.json vs backtest_pl_kelly.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
