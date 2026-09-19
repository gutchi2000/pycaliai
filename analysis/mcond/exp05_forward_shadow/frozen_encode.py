# -*- coding: utf-8 -*-
"""
frozen_encode.py — EXP04で凍結されたC1の型付け・one-hotカテゴリを新しい週次データへ適用する
================================================================================================
analysis/mcond/exp04_invariant_info_dev/out/feature_typing.json (学習期間2016-2021のみから
決めた kept_numeric / kept_onehot カテゴリ、EXP04で確定・以後変更しない) をそのまま読み込み、
同じ規則を新しい行に適用するだけ。未知カテゴリは全ダミー0 (基準カテゴリと同じ扱い)。
学習・選択は一切しない (const的な適用のみ)。
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
FEATURE_TYPING = BASE / "analysis/mcond/exp04_invariant_info_dev/out/feature_typing.json"

import sys as _sys  # noqa: E402
_sys.path.insert(0, str(BASE))
from category_normalize import normalize_categorical  # noqa: E402  (正本はcategory_normalize.py。export_weekly_marks.pyと共有)


def load_typing() -> dict:
    return json.loads(FEATURE_TYPING.read_text(encoding="utf-8"))


def encode_c1(df: pd.DataFrame, typing: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """df: predict_weekly.parse_csv 由来 (+ export_weekly_marks と同じrename/history補完済み)。
    戻り値: (c1__ 接頭辞つきの列だけを持つDataFrame, {列名: 見つかったか(bool)})。"""
    typing = typing or load_typing()
    df = normalize_categorical(df)
    out = pd.DataFrame(index=df.index)
    found: dict[str, bool] = {}

    for c in typing["kept_numeric"]:
        if "クラス名_ord" in c:
            continue  # 下で別扱い
        if c in df.columns:
            out[f"c1__{c}"] = pd.to_numeric(df[c], errors="coerce")
            found[c] = True
        else:
            out[f"c1__{c}"] = np.nan
            found[c] = False

    if "クラス名" in df.columns:
        import sys
        sys.path.insert(0, str(BASE))
        from grade_feats import class_name_to_ord
        out["c1__クラス名_ord"] = df["クラス名"].map(class_name_to_ord)
        found["クラス名(→クラス名_ord)"] = True
    else:
        out["c1__クラス名_ord"] = np.nan
        found["クラス名(→クラス名_ord)"] = False

    for c, cats in typing["kept_onehot"].items():
        if c in df.columns:
            s = df[c].astype(str)
            for cat in cats[1:]:  # 先頭カテゴリは学習時と同じ基準水準として落とす
                out[f"c1__{c}__{cat}"] = (s == cat).astype(float)
            found[c] = True
        else:
            for cat in cats[1:]:
                out[f"c1__{c}__{cat}"] = np.nan
            found[c] = False

    return out, found
