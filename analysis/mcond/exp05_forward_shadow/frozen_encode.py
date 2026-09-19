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

# 2026-09-19発見: predict_weekly.parse_csv (週次CSV由来) の生カテゴリ文字列が、
# master_v2.csv (学習時) の表記と食い違う列がある。本番 export_weekly_marks.py の
# apply_encoders() も同じ生カテゴリをそのまま encs (LabelEncoder, master由来のclasses_)
# に渡しており、正規化されないまま "ダート" 等が未知カテゴリ→__NaN__ に落ちている
# (別途ユーザーへ報告済み、本番コードはここでは変更しない)。EXP05-Fの特徴再現では
# 正しい方 (master_v2の表記) に正規化してから one-hot に通す。
CATEGORY_NORMALIZE: dict[str, dict[str, str]] = {
    "芝・ダ": {"ダート": "ダ"},
    "前芝・ダ": {"ダート": "ダ", "障ダ": "ダ", "障芝": "芝"},
    "芝(内・外)": {"内": " 内", "外": " 外", "": " "},
    "馬場状態": {"良(暫定)": "良", "稍重(暫定)": "稍", "重(暫定)": "重", "不良(暫定)": "不"},
    "前走馬場状態": {"良(暫定)": "良", "稍重(暫定)": "稍", "重(暫定)": "重", "不良(暫定)": "不"},
    "天気": {"曇(暫定)": " 曇 ", "晴(暫定)": " 晴 ", "雨(暫定)": " 雨 ", "雪(暫定)": " 雪 "},
}


def normalize_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """CATEGORY_NORMALIZEに定義した列だけ、既知の表記ゆれをmaster_v2表記へ寄せる。
    それ以外の未知値はそのまま(=one-hot側で自動的に0扱い、正しい「未知」)。"""
    df = df.copy()
    for col, mapping in CATEGORY_NORMALIZE.items():
        if col in df.columns:
            df[col] = df[col].astype(str).replace(mapping)
    return df


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
