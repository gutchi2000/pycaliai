"""tests/test_utils.py — parse_time_str のユニットテスト"""
import math
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import parse_time_str


class TestParseTimeStr:
    def test_standard_3part(self):
        """通常の M.SS.T 形式"""
        s = pd.Series(["1.34.5"])
        result = parse_time_str(s)
        assert math.isclose(result[0], 94.5)

    def test_zero_minutes(self):
        """0分台（短距離）"""
        s = pd.Series(["0.58.3"])
        result = parse_time_str(s)
        assert math.isclose(result[0], 58.3)

    def test_zero_time(self):
        """0.00.0"""
        s = pd.Series(["0.00.0"])
        result = parse_time_str(s)
        assert math.isclose(result[0], 0.0)

    def test_fallback_float_string(self):
        """2パーツの場合はfloat変換にフォールバック"""
        s = pd.Series(["94.5"])
        result = parse_time_str(s)
        assert math.isclose(result[0], 94.5)

    def test_nan_string(self):
        """変換不能な文字列はNoneを返す"""
        s = pd.Series(["N/A"])
        result = parse_time_str(s)
        assert result[0] is None

    def test_none_value(self):
        """None値はNoneを返す"""
        s = pd.Series([None])
        result = parse_time_str(s)
        assert result[0] is None

    def test_series_multiple(self):
        """複数要素のSeriesを一括変換"""
        s = pd.Series(["1.34.5", "1.00.0", "2.00.0"])
        result = parse_time_str(s)
        assert math.isclose(result[0], 94.5)
        assert math.isclose(result[1], 60.0)
        assert math.isclose(result[2], 120.0)

    def test_mixed_valid_invalid(self):
        """有効値と無効値が混在するSeries"""
        s = pd.Series(["1.34.5", "bad", None, "0.58.0"])
        result = parse_time_str(s)
        assert math.isclose(result[0], 94.5)
        assert result[1] is None
        assert result[2] is None
        assert math.isclose(result[3], 58.0)

    def test_whitespace_stripped(self):
        """前後の空白が除去される"""
        s = pd.Series([" 1.34.5 "])
        result = parse_time_str(s)
        assert math.isclose(result[0], 94.5)
