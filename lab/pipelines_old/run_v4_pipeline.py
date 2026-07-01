"""
run_v4_pipeline.py
==================
unified_rank_v4 (Optuna obj=ROI) で:
  1. PL calibrators 再 fit → pl_calibrators_v4.pkl
  2. payout_curve 再構築 → pl_payout_curve_v4.pkl
  3. backtest_pl_kelly 相当の OOS ROI 測定 → backtest_pl_kelly_v4.json
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
V4_MODEL = BASE / "models/unified_rank_v4.pkl"
V4_CAL   = BASE / "models/pl_calibrators_v4.pkl"
V4_CURVE = BASE / "data/pl_payout_curve_v4.pkl"


def step1_calibrators():
    print("\n" + "=" * 70)
    print("[STEP 1] PL calibrators (v4)")
    print("=" * 70)
    import build_pl_calibrators as bc
    bc.MODEL_PKL = V4_MODEL
    bc.OUT_PKL   = V4_CAL
    bc.main()


def step2_curve():
    print("\n" + "=" * 70)
    print("[STEP 2] payout curve (v4)")
    print("=" * 70)
    import build_payout_curve as bp
    bp.MODEL_PKL = V4_MODEL
    bp.CAL_PKL   = V4_CAL
    bp.OUT_PKL   = V4_CURVE
    bp.main()


def step3_backtest():
    print("\n" + "=" * 70)
    print("[STEP 3] backtest_pl_kelly (v4)")
    print("=" * 70)
    import backtest_pl_ev as be
    be.MODEL_PKL = V4_MODEL
    be.CAL_PKL   = V4_CAL
    be.CURVE_PKL = V4_CURVE
    import backtest_pl_kelly as bk
    bk.MODEL_PKL = V4_MODEL
    bk.CAL_PKL   = V4_CAL
    bk.CURVE_PKL = V4_CURVE
    bk.OUT_JSON  = BASE / "reports/backtest_pl_kelly_v4.json"
    bk.main()


def main():
    print("=" * 70)
    print("run_v4_pipeline.py")
    print("=" * 70)
    assert V4_MODEL.exists(), "unified_rank_v4.pkl not found"
    step1_calibrators()
    step2_curve()
    step3_backtest()
    print("\nDONE: compare reports/backtest_pl_kelly_v4.json vs v1 / v3")


if __name__ == "__main__":
    main()
