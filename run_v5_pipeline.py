"""
run_v5_pipeline.py
==================
unified_rank_v5 (Optuna obj=印精度 composite) で:
  1. PL calibrators 再 fit → pl_calibrators_v5.pkl
  2. payout_curve 再構築 → pl_payout_curve_v5.pkl
  3. (参考) backtest_pl_kelly 相当の OOS 測定 → backtest_pl_kelly_v5.json
  4. audit_marks.py --model v5 で印指標を測定

v4 との違い: モデルパスのみ。calibrator / curve / backtest ロジックは共通。

実行: python run_v5_pipeline.py
"""
from __future__ import annotations
import sys, warnings, subprocess
from pathlib import Path

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
V5_MODEL = BASE / "models/unified_rank_v5.pkl"
V5_CAL   = BASE / "models/pl_calibrators_v5.pkl"
V5_CURVE = BASE / "data/pl_payout_curve_v5.pkl"


def step1_calibrators():
    print("\n" + "=" * 70)
    print("[STEP 1] PL calibrators (v5)")
    print("=" * 70)
    import build_pl_calibrators as bc
    bc.MODEL_PKL = V5_MODEL
    bc.OUT_PKL   = V5_CAL
    bc.main()


def step2_curve():
    print("\n" + "=" * 70)
    print("[STEP 2] payout curve (v5)")
    print("=" * 70)
    import build_payout_curve as bp
    bp.MODEL_PKL = V5_MODEL
    bp.CAL_PKL   = V5_CAL
    bp.OUT_PKL   = V5_CURVE
    bp.main()


def step3_backtest():
    print("\n" + "=" * 70)
    print("[STEP 3] backtest_pl_kelly (v5) - 参考値")
    print("=" * 70)
    import backtest_pl_ev as be
    be.MODEL_PKL = V5_MODEL
    be.CAL_PKL   = V5_CAL
    be.CURVE_PKL = V5_CURVE
    import backtest_pl_kelly as bk
    bk.MODEL_PKL = V5_MODEL
    bk.CAL_PKL   = V5_CAL
    bk.CURVE_PKL = V5_CURVE
    bk.OUT_JSON  = BASE / "reports/backtest_pl_kelly_v5.json"
    bk.main()


def step4_audit_marks():
    print("\n" + "=" * 70)
    print("[STEP 4] audit_marks (v5) - v4 との比較")
    print("=" * 70)
    # subprocess で audit_marks.py を呼ぶ (state を持ち越さない)
    cmd = [sys.executable, str(BASE / "audit_marks.py"), "--model", "v5"]
    subprocess.run(cmd, check=True)


def step5_fixed_backtest():
    print("\n" + "=" * 70)
    print("[STEP 5] backtest_fixed (v5) - 馬連7点流し")
    print("=" * 70)
    cmd = [sys.executable, str(BASE / "backtest_fixed.py"), "--model", "v5"]
    subprocess.run(cmd, check=True)


def main():
    print("=" * 70)
    print("run_v5_pipeline.py")
    print("=" * 70)
    assert V5_MODEL.exists(), f"{V5_MODEL} not found - run optuna_v5_marks.py first"
    step1_calibrators()
    step2_curve()
    step4_audit_marks()      # 印指標 (新採用判定の本命)
    step5_fixed_backtest()   # 馬連7点流し ROI (参考値)
    step3_backtest()         # backtest_pl_kelly (参考値)
    print("\n" + "=" * 70)
    print("DONE")
    print("  → reports/audit_marks_v5.log で v4 との比較を確認")
    print("  → reports/backtest_fixed_v5.log で 7点流し ROI 推移を確認")
    print("=" * 70)


if __name__ == "__main__":
    main()
