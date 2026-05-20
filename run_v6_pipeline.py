"""
run_v6_pipeline.py
==================
unified_rank_v6 (Optuna obj = mark composite - 0.5 * ECE_high_p) で:
  1. PL calibrators 再 fit       → models/pl_calibrators_v6.pkl
  2. payout_curve 再構築          → data/pl_payout_curve_v6.pkl
  3. audit_marks.py --model v6   (印精度を v5 と比較)
  4. backtest_fixed --model v6   (馬連 7 点流し ROI、参考値)
  5. audit_v6_vs_v5.py           (高 EV 帯 calibration を v5 と直接比較)
  6. backtest_pl_kelly (v6)      (Kelly OOS 測定、参考値)

v5 との違い: モデルパスのみ。calibrator / curve / backtest ロジックは共通。
v6 採用基準は scripts/audit_v6_vs_v5.py の冒頭を参照
  (単勝 高 EV ROI が v5 比 +0.05 以上、または 複勝 +0.03 以上で本番化)。

実行: python run_v6_pipeline.py
"""
from __future__ import annotations
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
V6_MODEL = BASE / "models/unified_rank_v6.pkl"
V6_CAL   = BASE / "models/pl_calibrators_v6.pkl"
V6_CURVE = BASE / "data/pl_payout_curve_v6.pkl"


def step1_calibrators():
    print("\n" + "=" * 70)
    print("[STEP 1] PL calibrators (v6)")
    print("=" * 70)
    import build_pl_calibrators as bc
    bc.MODEL_PKL = V6_MODEL
    bc.OUT_PKL   = V6_CAL
    bc.main()


def step2_curve():
    print("\n" + "=" * 70)
    print("[STEP 2] payout curve (v6)")
    print("=" * 70)
    import build_payout_curve as bp
    bp.MODEL_PKL = V6_MODEL
    bp.CAL_PKL   = V6_CAL
    bp.OUT_PKL   = V6_CURVE
    bp.main()


def step3_audit_marks():
    print("\n" + "=" * 70)
    print("[STEP 3] audit_marks (v6) - v5 との比較")
    print("=" * 70)
    cmd = [sys.executable, str(BASE / "audit_marks.py"), "--model", "v6"]
    subprocess.run(cmd, check=True)


def step4_fixed_backtest():
    print("\n" + "=" * 70)
    print("[STEP 4] backtest_fixed (v6) - 馬連 7 点流し")
    print("=" * 70)
    cmd = [sys.executable, str(BASE / "backtest_fixed.py"), "--model", "v6"]
    subprocess.run(cmd, check=True)


def step5_audit_v6_vs_v5():
    print("\n" + "=" * 70)
    print("[STEP 5] audit_v6_vs_v5 - 高 EV 帯 calibration 直接比較")
    print("=" * 70)
    script = BASE / "scripts" / "audit_v6_vs_v5.py"
    if not script.exists():
        print(f"  SKIP: {script} not found")
        return
    cmd = [sys.executable, str(script)]
    subprocess.run(cmd, check=True)


def step6_backtest():
    print("\n" + "=" * 70)
    print("[STEP 6] backtest_pl_kelly (v6) - 参考値")
    print("=" * 70)
    import backtest_pl_ev as be
    be.MODEL_PKL = V6_MODEL
    be.CAL_PKL   = V6_CAL
    be.CURVE_PKL = V6_CURVE
    import backtest_pl_kelly as bk
    bk.MODEL_PKL = V6_MODEL
    bk.CAL_PKL   = V6_CAL
    bk.CURVE_PKL = V6_CURVE
    bk.OUT_JSON  = BASE / "reports/backtest_pl_kelly_v6.json"
    bk.main()


def step7_class_prior():
    """クラス別 印信頼度マトリクスを生成して data/class_prior_v6.json に書き出す。
    export_weekly_marks.py がこれを読んで bundle.json の race_meta.class_prior に
    埋め込み、Cowork が race ごとに経験 prior を参照できるようにする。
    """
    print("\n" + "=" * 70)
    print("[STEP 7] class_prior (◎〇▲△△ 信頼度マトリクス、v6 OOS)")
    print("=" * 70)
    script = BASE / "scripts" / "audit_marks_by_class.py"
    if not script.exists():
        print(f"  SKIP: {script} not found")
        return
    cmd = [sys.executable, str(script), "--model", "v6"]
    subprocess.run(cmd, check=True)
    print(f"  → data/class_prior_v6.json を更新 (export_weekly_marks.py で読まれる)")


def main():
    print("=" * 70)
    print("run_v6_pipeline.py")
    print("=" * 70)
    assert V6_MODEL.exists(), (
        f"{V6_MODEL} not found - run `python optuna_v6_marks.py` first"
    )
    step1_calibrators()
    step2_curve()
    step3_audit_marks()        # 印指標 (新採用判定の本命)
    step4_fixed_backtest()     # 馬連 7 点流し ROI (参考値)
    step5_audit_v6_vs_v5()     # 高 EV 帯 calibration 比較 (採用判定の本命 2)
    step6_backtest()           # backtest_pl_kelly (参考値)
    step7_class_prior()        # クラス別 印信頼度 → bundle.json 埋込用
    print("\n" + "=" * 70)
    print("DONE")
    print("  → reports/audit_marks_v6.log で印精度 (v5 比較) を確認")
    print("  → reports/backtest_fixed_v6.log で 7 点流し ROI 推移を確認")
    print("  → reports/audit_v6_vs_v5_<date>.md で高 EV 帯 calibration 比較を確認")
    print("  → data/class_prior_v6.json が bundle.json の race_meta.class_prior に埋込まれる")
    print("=" * 70)


if __name__ == "__main__":
    main()
