"""tests/test_backtest.py — backtest ユーティリティ関数のユニットテスト"""
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest import floor_to_unit, get_actual_payout


class TestFloorToUnit:
    def test_exact_multiple(self):
        assert floor_to_unit(500, 100) == 500

    def test_rounds_down(self):
        assert floor_to_unit(550, 100) == 500

    def test_below_unit(self):
        """単位未満は0になる"""
        assert floor_to_unit(99, 100) == 0

    def test_large_amount(self):
        assert floor_to_unit(12345, 100) == 12300

    def test_default_unit(self):
        """デフォルト単位=MIN_UNIT=100"""
        assert floor_to_unit(999) == 900

    def test_zero(self):
        assert floor_to_unit(0, 100) == 0


class TestGetActualPayout:
    def _entry(self):
        return {
            "複勝": {3: 180, 7: 120, 11: 200},
            "馬連": {"3-7": 1500},
            "馬単": {"7-3": 2200},
            "三連複": {"3-7-11": 5000},
            "三連単": {"7-3-11": 18000},
        }

    # 複勝
    def test_fukusho_hit(self):
        pay = get_actual_payout([3], False, "複勝", self._entry())
        assert pay == 180

    def test_fukusho_miss(self):
        pay = get_actual_payout([5], False, "複勝", self._entry())
        assert pay == 0

    # 馬連
    def test_umaren_hit(self):
        pay = get_actual_payout([7, 3], False, "馬連", self._entry())
        assert pay == 1500

    def test_umaren_hit_reversed(self):
        """順序不問（sortしてkeyを作る）"""
        pay = get_actual_payout([3, 7], False, "馬連", self._entry())
        assert pay == 1500

    def test_umaren_miss(self):
        pay = get_actual_payout([1, 2], False, "馬連", self._entry())
        assert pay == 0

    # 馬単
    def test_umatan_hit(self):
        pay = get_actual_payout([7, 3], True, "馬単", self._entry())
        assert pay == 2200

    def test_umatan_miss_reversed(self):
        """馬単は順序あり: 逆順はミス"""
        pay = get_actual_payout([3, 7], True, "馬単", self._entry())
        assert pay == 0

    # 三連複
    def test_sanrenfuku_hit(self):
        pay = get_actual_payout([11, 3, 7], False, "三連複", self._entry())
        assert pay == 5000

    def test_sanrenfuku_miss(self):
        pay = get_actual_payout([1, 2, 3], False, "三連複", self._entry())
        assert pay == 0

    # 三連単
    def test_sanrentan_hit(self):
        pay = get_actual_payout([7, 3, 11], True, "三連単", self._entry())
        assert pay == 18000

    def test_sanrentan_miss_order(self):
        pay = get_actual_payout([3, 7, 11], True, "三連単", self._entry())
        assert pay == 0
