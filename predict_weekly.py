# -*- coding: utf-8 -*-
"""
predict_weekly.py (shim) — 旧 8 モデルアンサンブル本体は legacy/predict_weekly.py へ移動 (2026-11-04)
================================================================================
本番が使う読み込み部 (parse_csv と列定義) は weekly_parse.py にあり、ここから同じ名前で再 export する。
それ以外の旧予測・買い目関数 (ensemble_predict / assign_marks / get_bets / main ...) は参照されたときだけ
legacy.predict_weekly を読み込んで転送する (PEP 562)。研究スクリプトの `from predict_weekly import X` は無修正で動く。
旧パイプラインの実行: python -m legacy.predict_weekly --csv data/weekly/YYYYMMDD.csv
"""
from __future__ import annotations

from weekly_parse import (  # noqa: F401
    CLASS_NORMALIZE, COLUMN_MAP, HORSE_COLS_33, HORSE_COLS_44, HORSE_COLS_46, HORSE_COLS_48, HORSE_COLS_49,
    HORSE_COLS_99, HOSSEI_DIR, KAKO5_DIR, RACE_COLS, TYAKU_DIR, TYAKU_HORSE_COLS, TYAKU_HORSE_COLS_52,
    TYAKU_HORSE_COLS_53, TYAKU_HORSE_SCHEMAS, _load_hosei, _load_kako5_warnings, _load_tyaku, parse_csv,
)


def __getattr__(name: str):
    if name.startswith("__"):
        raise AttributeError(name)
    import importlib
    legacy = importlib.import_module("legacy.predict_weekly")
    return getattr(legacy, name)


if __name__ == "__main__":
    import runpy
    runpy.run_module("legacy.predict_weekly", run_name="__main__", alter_sys=True)
