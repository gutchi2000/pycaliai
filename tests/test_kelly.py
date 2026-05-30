"""tests/test_kelly.py — kelly_fraction のユニットテスト"""
import math
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kelly import kelly_fraction


class TestKellyFraction:
    def test_basic_positive(self):
        """期待値プラスのケース: f* > 0"""
        # p=0.4, b=2.0 → f* = (0.4*2 - 0.6)/2 = (0.8-0.6)/2 = 0.1
        f = kelly_fraction(0.4, 2.0)
        assert math.isclose(f, 0.1)

    def test_breakeven(self):
        """期待値±0: f*=0"""
        # p=0.5, b=1.0 → f* = (0.5*1 - 0.5)/1 = 0.0
        f = kelly_fraction(0.5, 1.0)
        assert math.isclose(f, 0.0)

    def test_negative_clipped_to_zero(self):
        """期待値マイナスはクリップされて0"""
        # p=0.2, b=1.0 → f* = (0.2 - 0.8)/1 = -0.6 → 0
        f = kelly_fraction(0.2, 1.0)
        assert f == 0.0

    def test_high_odds_low_prob(self):
        """高オッズ・低確率のケース"""
        # p=0.1, b=10.0 → f* = (0.1*10 - 0.9)/10 = (1-0.9)/10 = 0.01
        f = kelly_fraction(0.1, 10.0)
        assert math.isclose(f, 0.01)

    def test_prob_one(self):
        """的中確率1.0 → f*=1.0"""
        f = kelly_fraction(1.0, 2.0)
        assert math.isclose(f, 1.0)

    def test_result_bounded_zero_to_one(self):
        """返値は常に0以上"""
        for p in [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]:
            for b in [0.5, 1.0, 2.0, 5.0]:
                f = kelly_fraction(p, b)
                assert f >= 0.0

    def test_half_kelly(self):
        """Half-Kelly = kelly_fraction / 2"""
        f = kelly_fraction(0.4, 2.0)
        half = f / 2
        assert math.isclose(half, 0.05)
