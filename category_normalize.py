# -*- coding: utf-8 -*-
"""
category_normalize.py — 週次CSV由来カテゴリ表記を学習時(master_v2.csv)表記へ正規化する
==========================================================================================
2026-09-19発見: predict_weekly.parse_csv (週次CSV由来) の生カテゴリ文字列が、
master_v2.csv (学習時) の表記と食い違う列がある (芝・ダ="ダート" vs "ダ" 等)。
models/unified_rank_v6.pkl の encoders[col] (LabelEncoder, master由来のclasses_) は
正規化前の値を未知カテゴリとして __NaN__ に落とすため、正規化しないと該当列の情報が
serve で失われる。

このマッピングは export_weekly_marks.py (本番) と
analysis/mcond/exp05_forward_shadow/ (研究、frozen_encode.py・category_parity_audit.py) の
両方から参照される唯一の正本。どちらか一方だけを直接編集しないこと
(tests/test_category_normalize.py が2箇所の一致を保証する)。

各列は「同じ意味の値を確認済みの表記へ寄せる」変換関数として定義する。未知値を意味確認
なしに既知カテゴリへ近似する目的では使わない。ルールに合致しない値はそのまま返す
(=正規化されない、encoder側で正しく「未知」として扱われる)。

2026-09-19 全カテゴリ監査 (analysis/mcond/exp05_forward_shadow/CATEGORY_PARITY.csv) で
追加確認した表記ゆれ:
  - 天気/馬場状態/前走馬場状態: 単純な"(暫定)"接尾辞だけでなく、接尾辞なしの生値
    ("晴"等)自体が学習時の前後スペース付き表記(" 晴 "等)と食い違っていた
    (単純dictでは"晴(暫定)"のような組合せを網羅しきれず不十分だった)
  - 前走競走種別: 生値が整数文字列("13")、学習時はfloat文字列("13.0")
  - 重量種別: 生値が半角カナ("ﾊﾝﾃﾞ")、学習時は全角("ハンデ") — unicodedata NFKCで解消
"""
from __future__ import annotations

import unicodedata
from typing import Callable

import pandas as pd

EPS_WEATHER_SINGLE = {"晴", "曇", "雨", "雪"}
BABA_VALID = {"良", "稍", "重", "不"}


def _strip_teiban(v: str) -> str:
    return v.replace("(暫定)", "")


def _norm_surface(v: str) -> str:
    return {"ダート": "ダ"}.get(v, v)


def _norm_prev_surface(v: str) -> str:
    return {"ダート": "ダ", "障ダ": "ダ", "障芝": "芝"}.get(v, v)


def _norm_naigai(v: str) -> str:
    return {"内": " 内", "外": " 外", "": " "}.get(v, v)


def _norm_baba_condition(v: str) -> str:
    v = _strip_teiban(v)
    return v if v in BABA_VALID else v  # 未対応の値はそのまま (=既知の"未知"に委ねる)


def _norm_weather(v: str) -> str:
    v = _strip_teiban(v)
    if v in EPS_WEATHER_SINGLE:
        return f" {v} "
    return v  # 小雨/小雪 はそのまま encoder 語彙と一致 (前後スペース無し)


def _norm_race_type_code(v: str) -> str:
    """前走競走種別: 生値が整数文字列 ("13")、学習時は float 文字列 ("13.0")。
    数値変換できる場合のみ float 文字列化する (意味を変えない、桁の書式だけ揃える)。"""
    try:
        return str(float(v))
    except (ValueError, TypeError):
        return v


def _norm_weight_type(v: str) -> str:
    """重量種別: 生値が半角カナ("ﾊﾝﾃﾞ")のことがある。学習時は全角("ハンデ")。
    unicodedata.normalize("NFKC") は半角カナ+濁点を正しい全角へ合成する標準変換。"""
    return unicodedata.normalize("NFKC", v)


# 列名 -> 正規化関数。関数は「1つの生文字列 -> 正規化後の文字列」。
# ルールに合致しない入力はそのまま返す (副作用的に "未知" のまま扱われる)。
NORMALIZERS: dict[str, Callable[[str], str]] = {
    "芝・ダ": _norm_surface,
    "前芝・ダ": _norm_prev_surface,
    "芝(内・外)": _norm_naigai,
    "馬場状態": _norm_baba_condition,
    "前走馬場状態": _norm_baba_condition,
    "天気": _norm_weather,
    "前走競走種別": _norm_race_type_code,
    "重量種別": _norm_weight_type,
}

# 実欠損として扱い、正規化対象にしない値 (encoder側の __NaN__ に委ねる)。
_MISSING_TOKENS = {"nan", "__NaN__", ""}


def normalize_categorical(df: pd.DataFrame, normalizers: dict | None = None) -> pd.DataFrame:
    """NORMALIZERSに定義した列だけ正規化する。実欠損トークンは対象外 (そのまま)。"""
    normalizers = normalizers if normalizers is not None else NORMALIZERS
    df = df.copy()
    for col, fn in normalizers.items():
        if col not in df.columns:
            continue
        s = df[col].astype(str)
        mask = ~s.isin(_MISSING_TOKENS)
        df.loc[mask, col] = s[mask].map(fn)
    return df


def fixed_count(df: pd.DataFrame, normalizers: dict | None = None) -> dict[str, int]:
    """列ごとに正規化で値が変わった件数 (ログ用)。"""
    normalizers = normalizers if normalizers is not None else NORMALIZERS
    out = {}
    for col, fn in normalizers.items():
        if col not in df.columns:
            continue
        s = df[col].astype(str)
        mask = ~s.isin(_MISSING_TOKENS)
        if mask.any():
            changed = (s[mask].map(fn) != s[mask]).sum()
            if changed:
                out[col] = int(changed)
    return out
