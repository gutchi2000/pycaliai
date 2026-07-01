"""
run_v7_pipeline.py
==================
unified_rank_v7 (v6 + wide ROI bonus + race_relative_feats v2) を本番ライン用に
calibrator / curve / audit / class_prior まで一括構築する。

ステップ:
  1. PL calibrators 再 fit       → models/pl_calibrators_v7.pkl
  2. payout_curve 再構築          → data/pl_payout_curve_v7.pkl
  3. audit_marks.py --model v7   (印精度を v5/v6 と比較)
  4. backtest_fixed --model v7   (馬連 7 点流し ROI、参考値)
  5. audit_v7_vs_v6.py           (高 EV 帯 calibration を v6 と直接比較)
  6. backtest_pl_kelly (v7)      (Kelly OOS 測定、参考値)
  7. class_prior (v7)            (印×クラス経験的中率を data/class_prior_v7.json へ)

採用判定:
  ワイド ROI が v6 比 +0.05 以上 (Cowork ワイド改善の本命)
  または 複勝/馬連 ECE が v6 比 維持以上 で本番化候補

実行: python run_v7_pipeline.py
"""
from __future__ import annotations
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
V7_MODEL = BASE / "models/unified_rank_v7.pkl"
V7_CAL   = BASE / "models/pl_calibrators_v7.pkl"
V7_CURVE = BASE / "data/pl_payout_curve_v7.pkl"


def step1_calibrators():
    print("\n" + "=" * 70)
    print("[STEP 1] PL calibrators (v7)")
    print("=" * 70)
    import build_pl_calibrators as bc
    bc.MODEL_PKL = V7_MODEL
    bc.OUT_PKL   = V7_CAL
    bc.main()


def step2_curve():
    print("\n" + "=" * 70)
    print("[STEP 2] payout curve (v7)")
    print("=" * 70)
    import build_payout_curve as bp
    bp.MODEL_PKL = V7_MODEL
    bp.CAL_PKL   = V7_CAL
    bp.OUT_PKL   = V7_CURVE
    bp.main()


def step3_audit_marks():
    print("\n" + "=" * 70)
    print("[STEP 3] audit_marks (v7) - v5/v6 との比較")
    print("=" * 70)
    cmd = [sys.executable, str(BASE / "audit_marks.py"), "--model", "v7"]
    subprocess.run(cmd, check=True)


def step4_fixed_backtest():
    print("\n" + "=" * 70)
    print("[STEP 4] backtest_fixed (v7) - 馬連 7 点流し")
    print("=" * 70)
    cmd = [sys.executable, str(BASE / "backtest_fixed.py"), "--model", "v7"]
    subprocess.run(cmd, check=True)


def step5_audit_v7_vs_v6():
    print("\n" + "=" * 70)
    print("[STEP 5] audit_v7_vs_v6 - 高 EV 帯 calibration 直接比較")
    print("=" * 70)
    script = BASE / "scripts" / "audit_v7_vs_v6.py"
    if not script.exists():
        print(f"  SKIP: {script} not found")
        return
    cmd = [sys.executable, str(script)]
    subprocess.run(cmd, check=True)


def step6_backtest():
    print("\n" + "=" * 70)
    print("[STEP 6] backtest_pl_kelly (v7) - 参考値")
    print("=" * 70)
    import backtest_pl_ev as be
    be.MODEL_PKL = V7_MODEL
    be.CAL_PKL   = V7_CAL
    be.CURVE_PKL = V7_CURVE
    import backtest_pl_kelly as bk
    bk.MODEL_PKL = V7_MODEL
    bk.CAL_PKL   = V7_CAL
    bk.CURVE_PKL = V7_CURVE
    bk.OUT_JSON  = BASE / "reports/backtest_pl_kelly_v7.json"
    bk.main()


def step7_class_prior():
    print("\n" + "=" * 70)
    print("[STEP 7] class_prior (◎〇▲△△ 信頼度マトリクス、v7 OOS)")
    print("=" * 70)
    script = BASE / "scripts" / "audit_marks_by_class.py"
    if not script.exists():
        print(f"  SKIP: {script} not found")
        return
    cmd = [sys.executable, str(script), "--model", "v7"]
    subprocess.run(cmd, check=True)
    print(f"  → data/class_prior_v7.json を更新 (export_weekly_marks.py で読まれる)")


def main():
    print("=" * 70)
    print("run_v7_pipeline.py")
    print("=" * 70)
    assert V7_MODEL.exists(), (
        f"{V7_MODEL} not found - run `python optuna_v7_marks.py` first"
    )
    step1_calibrators()
    step2_curve()
    step3_audit_marks()
    step4_fixed_backtest()
    step5_audit_v7_vs_v6()
    step6_backtest()
    step7_class_prior()
    print("\n" + "=" * 70)
    print("DONE")
    print("  → reports/audit_marks_v7.log で印精度 (v5/v6 比較) を確認")
    print("  → reports/backtest_fixed_v7.json で 7 点流し ROI 推移を確認")
    print("  → reports/audit_v7_vs_v6_<date>.md で 高 EV calibration 比較を確認")
    print("  → reports/backtest_pl_kelly_v7.json で Kelly ROI を確認")
    print("  → data/class_prior_v7.json (export_weekly_marks.py --model v7 で使う)")
    print("=" * 70)


if __name__ == "__main__":
    main()
