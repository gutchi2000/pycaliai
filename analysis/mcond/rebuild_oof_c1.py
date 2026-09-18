# -*- coding: utf-8 -*-
"""
rebuild_oof_c1.py — v6 の expanding-window OOF スコアを P0-5 修正後 (C1) の master で作り直す
=========================================================================================
既存 data/oof_scores_v6params.parquet (2026-07-23) は P0-5 修正前の master から作られており、
trainer_fuku30/90 の同一レース内リーク (626,774 行中 21,539 行) を含む。
2016-2023 の「現行AI」側だけが不当に強くなるため、市場条件付き検定の基準には使えない。

exp_oof_scores.py をそのまま使い、出力先だけを差し替える (既存ファイルは上書きしない)。
  train ≤Y-2 / early stop Y-1 / predict Y, Y=2015..2023, v6 の optuna パラメータ・120特徴・encoder
  注: 本番 C1 モデルが使う学習重み (sample_weight_alpha) はこのスクリプトには無い。
      OOF は「その年に v6 相当の手順で作れたスコア」の近似であり、本番モデルと完全に同一ではない。
出力: data/_research/mcond/oof_scores_c1master.parquet
"""
from __future__ import annotations
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
import exp_oof_scores as E  # noqa: E402
from joint_calibration_v6 import MASTER_CSV  # noqa: E402

E.OUT = BASE / "data" / "_research" / "mcond" / "oof_scores_c1master.parquet"

# C1 の本番 bundle には optuna_best_params キーが無い (値は旧 v6 と同一で model.params に入っている)。
# encoder / feature_cols は C1 (クリーン) のものを使い、パラメータ辞書だけ補う。
_real_load = E.joblib.load


def _load_with_params(path, *a, **k):
    b = _real_load(path, *a, **k)
    if isinstance(b, dict) and "optuna_best_params" not in b and "model" in b:
        mp = b["model"].params
        b = dict(b)
        b["optuna_best_params"] = {
            "lr": mp["learning_rate"], "num_leaves": mp["num_leaves"], "max_depth": mp["max_depth"],
            "min_data_in_leaf": mp["min_data_in_leaf"], "ff": mp["feature_fraction"],
            "bf": mp["bagging_fraction"], "l1": mp["lambda_l1"], "l2": mp["lambda_l2"]}
    return b


E.joblib.load = _load_with_params

if __name__ == "__main__":
    print(f"master = {MASTER_CSV}")
    E.main()
