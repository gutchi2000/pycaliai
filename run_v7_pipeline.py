# -*- coding: utf-8 -*-
"""
run_v7_pipeline.py — v7 候補 (v6 + JRDB 過去走特徴) の較正器・払戻カーブ・クラス別経験率を作る (2026-11-13)
================================================================================================
前提: train_v7_jrdb.py が models/unified_rank_v7.pkl を作っていること。
  1. PL calibrators  → models/pl_calibrators_v7.pkl   (valid=2023 で fit、v6 と同じ手順)
  2. payout curve    → data/pl_payout_curve_v7.pkl
  3. class prior     → data/class_prior_v7.json
本番 (v6) の成果物・production_policy.json・artifact_pins.json には触れない。
配信用の較正器 (pl_calibrators_v7_serve.pkl) は 2026 の配信スコアが貯まってから別途 fit する。
実行: PYTHONUTF8=1 ./venv311/Scripts/python.exe run_v7_pipeline.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).parent
MODEL = BASE / "models/unified_rank_v7.pkl"
CAL = BASE / "models/pl_calibrators_v7.pkl"
CURVE = BASE / "data/pl_payout_curve_v7.pkl"


def main() -> int:
    assert MODEL.exists(), f"{MODEL} が無い (先に train_v7_jrdb.py)"
    import build_pl_calibrators as bc
    bc.MODEL_PKL, bc.OUT_PKL = MODEL, CAL
    bc.main()
    import build_payout_curve as bp
    bp.MODEL_PKL, bp.CAL_PKL, bp.OUT_PKL = MODEL, CAL, CURVE
    bp.main()
    subprocess.run([sys.executable, str(BASE / "scripts" / "audit_marks_by_class.py"), "--model", "v7"], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
