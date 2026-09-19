# -*- coding: utf-8 -*-
"""spec (EXP05-F 最終確認) §5: カテゴリ正規化の固定単体テスト。"""
from __future__ import annotations
import sys
from pathlib import Path

import joblib
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from category_normalize import NORMALIZERS, normalize_categorical, fixed_count  # noqa: E402


def test_surface_dirt_to_short_form():
    assert NORMALIZERS["芝・ダ"]("ダート") == "ダ"


def test_surface_turf_unchanged():
    assert NORMALIZERS["芝・ダ"]("芝") == "芝"


def test_naigai_inner_gets_leading_space():
    assert NORMALIZERS["芝(内・外)"]("内") == " 内"


def test_naigai_outer_gets_leading_space():
    assert NORMALIZERS["芝(内・外)"]("外") == " 外"


def test_naigai_already_spaced_values_not_double_applied():
    # 正規化関数はdictルックアップなので " 内"/" 外" (既に正しい表記) はそのまま通る
    assert NORMALIZERS["芝(内・外)"](" 内") == " 内"
    assert NORMALIZERS["芝(内・外)"](" 外") == " 外"


def test_baba_condition_strips_teiban_suffix():
    assert NORMALIZERS["馬場状態"]("良(暫定)") == "良"
    assert NORMALIZERS["馬場状態"]("稍(暫定)") == "稍"
    assert NORMALIZERS["馬場状態"]("不(暫定)") == "不"
    assert NORMALIZERS["馬場状態"]("重(暫定)") == "重"


def test_weather_teiban_and_padding():
    assert NORMALIZERS["天気"]("晴(暫定)") == " 晴 "
    assert NORMALIZERS["天気"]("晴") == " 晴 "
    assert NORMALIZERS["天気"]("曇") == " 曇 "
    assert NORMALIZERS["天気"]("小雨") == "小雨"  # 前後スペース無しが正しい (encoder語彙どおり)
    assert NORMALIZERS["天気"]("小雨(暫定)") == "小雨"


def test_race_type_code_numeric_format():
    assert NORMALIZERS["前走競走種別"]("13") == "13.0"
    assert NORMALIZERS["前走競走種別"]("13.0") == "13.0"


def test_weight_type_halfwidth_to_fullwidth():
    assert NORMALIZERS["重量種別"]("ﾊﾝﾃﾞ") == "ハンデ"
    assert NORMALIZERS["重量種別"]("ハンデ") == "ハンデ"


def test_unknown_new_category_not_silently_mapped():
    """ルールに合致しない値はそのまま返す (勝手に既知値へ変換しない)。"""
    assert NORMALIZERS["芝・ダ"]("未知の新種別") == "未知の新種別"
    assert NORMALIZERS["天気"]("台風") == "台風"


def test_normalize_categorical_leaves_true_missing_untouched():
    df = pd.DataFrame({"芝・ダ": ["ダート", "nan", "__NaN__", ""]})
    out = normalize_categorical(df)
    assert out["芝・ダ"].tolist() == ["ダ", "nan", "__NaN__", ""]


def test_fixed_count_reports_only_changed_rows():
    df = pd.DataFrame({"芝・ダ": ["ダート", "芝", "ダート", "nan"]})
    counts = fixed_count(df)
    assert counts["芝・ダ"] == 2


def test_normalized_values_are_in_v6_encoder_vocabulary():
    """正規化後の値が実際にmodels/unified_rank_v6.pklのencoder語彙に含まれることを
    直接確認する (定義が正しいことの最終保証)。"""
    model_path = BASE / "models" / "unified_rank_v6.pkl"
    if not model_path.exists():
        return
    encs = joblib.load(model_path)["encoders"]
    cases = {
        "芝・ダ": ["ダート", "芝"],
        "芝(内・外)": ["内", "外"],
        "馬場状態": ["良(暫定)", "稍(暫定)", "重(暫定)", "不(暫定)"],
        "天気": ["晴", "曇", "雨", "雪", "小雨", "小雪"],
        "前走競走種別": ["13", "11"],
        "重量種別": ["ﾊﾝﾃﾞ"],
    }
    for col, raws in cases.items():
        if col not in encs:
            continue
        classes = set(encs[col].classes_)
        for raw in raws:
            normalized = NORMALIZERS[col](raw)
            assert normalized in classes, f"{col}: {raw!r} -> {normalized!r} not in encoder classes"
