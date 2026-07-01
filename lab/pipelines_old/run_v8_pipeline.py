"""
run_v8_pipeline.py
==================
unified_rank_v8 (v6 obj + race_relative_feats(v2) + course_affinity_feats(v1))
を本番ライン用に calibrator / curve / audit まで一括構築する。

ステップ:
  1. PL calibrators 再 fit       → models/pl_calibrators_v8.pkl
  2. payout_curve 再構築          → data/pl_payout_curve_v8.pkl
  3. audit_marks.py --model v8   (印精度を v5/v6 と比較)
  4. backtest_fixed --model v8   (馬連 7 点流し ROI、参考値)
  5. backtest_pl_kelly (v8)      (Kelly OOS 測定、参考値)
  6. class_prior (v8)            (印×クラス経験的中率 → data/class_prior_v8.json)

採用判定:
  test 2024-25 で v6 と同等以上の印精度 + 同等以上の ROI なら本番化候補。
  特に複勝/馬連 ROI が改善するかが焦点。

実行: python run_v8_pipeline.py
"""
from __future__ import annotations
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
V8_MODEL = BASE / "models/unified_rank_v8.pkl"
V8_CAL   = BASE / "models/pl_calibrators_v8.pkl"
V8_CURVE = BASE / "data/pl_payout_curve_v8.pkl"


def step1_calibrators():
    print("\n" + "=" * 70)
    print("[STEP 1] PL calibrators (v8)")
    print("=" * 70)
    import build_pl_calibrators as bc
    bc.MODEL_PKL = V8_MODEL
    bc.OUT_PKL   = V8_CAL
    bc.main()


def step2_curve():
    print("\n" + "=" * 70)
    print("[STEP 2] payout curve (v8)")
    print("=" * 70)
    import build_payout_curve as bp
    bp.MODEL_PKL = V8_MODEL
    bp.CAL_PKL   = V8_CAL
    bp.OUT_PKL   = V8_CURVE
    bp.main()


def step3_audit_marks():
    print("\n" + "=" * 70)
    print("[STEP 3] audit_marks (v8)")
    print("=" * 70)
    cmd = [sys.executable, str(BASE / "audit_marks.py"), "--model", "v8"]
    subprocess.run(cmd, check=True)


def step4_fixed_backtest():
    print("\n" + "=" * 70)
    print("[STEP 4] backtest_fixed (v8)")
    print("=" * 70)
    cmd = [sys.executable, str(BASE / "backtest_fixed.py"), "--model", "v8"]
    subprocess.run(cmd, check=True)


def step5_backtest():
    print("\n" + "=" * 70)
    print("[STEP 5] backtest_pl_kelly (v8)")
    print("=" * 70)
    import backtest_pl_ev as be
    be.MODEL_PKL = V8_MODEL
    be.CAL_PKL   = V8_CAL
    be.CURVE_PKL = V8_CURVE
    import backtest_pl_kelly as bk
    bk.MODEL_PKL = V8_MODEL
    bk.CAL_PKL   = V8_CAL
    bk.CURVE_PKL = V8_CURVE
    bk.OUT_JSON  = BASE / "reports/backtest_pl_kelly_v8.json"
    bk.main()


def step6_class_prior():
    print("\n" + "=" * 70)
    print("[STEP 6] class_prior (v8)")
    print("=" * 70)
    script = BASE / "scripts" / "audit_marks_by_class.py"
    if not script.exists():
        print(f"  SKIP: {script} not found")
        return
    cmd = [sys.executable, str(script), "--model", "v8"]
    subprocess.run(cmd, check=True)


def main():
    print("=" * 70)
    print("run_v8_pipeline.py")
    print("=" * 70)
    assert V8_MODEL.exists(), (
        f"{V8_MODEL} not found - run `python optuna_v8_marks.py` first"
    )
    step1_calibrators()
    step2_curve()
    step3_audit_marks()
    step4_fixed_backtest()
    step5_backtest()
    step6_class_prior()
    print("\n" + "=" * 70)
    print("DONE")
    print("  → reports/audit_marks_v8.log で印精度を確認")
    print("  → reports/backtest_fixed_v8.json で 7 点流し ROI 推移を確認")
    print("  → reports/backtest_pl_kelly_v8.json で Kelly ROI を確認")
    print("  → data/class_prior_v8.json (export_weekly_marks.py --model v8 で使う)")
    print("=" * 70)


if __name__ == "__main__":
    main()
