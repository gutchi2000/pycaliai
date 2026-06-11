# -*- coding: utf-8 -*-
"""
本番ライン (v6 marks stack → compute_bets → validate → generate_results) の
純関数ゴールデンテスト。audit 2026-06-11「本番ラインのテスト0」対策。

実行: venv311\\Scripts\\python.exe -m pytest tests/test_production_line.py -q
データファイル非依存 (合成入力のみ)。
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compute_bets import amount_for_ev, allocate, MIN_BET, MAX_BET, BUDGET
from validate_cowork_bets import content_issues, skip_reasons
from generate_results import get_cancelled, _bet_cis


# ============================================================
# compute_bets: 配分系
# ============================================================
class TestAmountForEv:
    def test_monotone_increasing(self):
        """EV が高いほど金額は単調非減少。"""
        evs = [0.5, 0.85, 1.0, 1.2, 1.5, 2.0]
        amts = [amount_for_ev(e) for e in evs]
        assert amts == sorted(amts)

    def test_bands(self):
        assert amount_for_ev(1.50) == 5500
        assert amount_for_ev(1.20) == 3750
        assert amount_for_ev(1.00) == 2250
        assert amount_for_ev(0.85) == 1500
        assert amount_for_ev(0.10) == 900


class TestAllocate:
    def test_sums_to_budget(self):
        amts = allocate([1.0, 2.0, 3.0])
        assert sum(amts) == BUDGET

    def test_respects_min_max(self):
        amts = allocate([0.01, 100.0])  # 極端な重み差でも min/max 厳守
        assert all(MIN_BET <= a <= MAX_BET for a in amts)

    def test_100yen_units(self):
        amts = allocate([1.3, 2.7, 0.9, 1.1])
        assert all(a % 100 == 0 for a in amts)

    def test_empty(self):
        assert allocate([]) == []

    def test_single(self):
        amts = allocate([1.0])
        assert len(amts) == 1 and MIN_BET <= amts[0] <= MAX_BET


# ============================================================
# validate_cowork_bets: 内容バリデーション
# ============================================================
class TestContentIssues:
    VALID = {1, 2, 3, 7}

    def test_ok_tansho(self):
        assert content_issues({"馬券種": "単勝", "買い目": "7", "購入額": 1000},
                              self.VALID) == []

    def test_ok_wide(self):
        assert content_issues({"馬券種": "ワイド", "買い目": "3-7", "購入額": 500},
                              self.VALID) == []

    def test_sanrentan_rejected(self):
        iss = content_issues({"馬券種": "三連単", "買い目": "1-2-3", "購入額": 1000},
                             self.VALID)
        assert any("廃止券種" in s for s in iss)

    def test_unknown_kind(self):
        iss = content_issues({"馬券種": "枠連", "買い目": "1-2", "購入額": 1000},
                             self.VALID)
        assert any("未知券種" in s for s in iss)

    def test_nonexistent_umaban(self):
        iss = content_issues({"馬券種": "ワイド", "買い目": "7-18", "購入額": 1000},
                             self.VALID)
        assert any("存在しない馬番" in s for s in iss)

    def test_bad_amount_unit(self):
        iss = content_issues({"馬券種": "単勝", "買い目": "7", "購入額": 1050},
                             self.VALID)
        assert any("100円単位" in s for s in iss)

    def test_amount_over_cap(self):
        iss = content_issues({"馬券種": "単勝", "買い目": "7", "購入額": 20000},
                             self.VALID)
        assert any("上限超" in s for s in iss)

    def test_zero_amount(self):
        iss = content_issues({"馬券種": "複勝", "買い目": "3", "購入額": 0},
                             self.VALID)
        assert any("非正" in s for s in iss)


class TestSkipReasons:
    def test_chaos_skip(self):
        r = skip_reasons({"field_size": 16}, {"field_chaos_score": 0.95},
                         {"tansho_odds": 3.0, "p_win": 0.3})
        assert any("chaos" in s for s in r)

    def test_field_size_skip(self):
        r = skip_reasons({"field_size": 7}, {"field_chaos_score": 0.5},
                         {"tansho_odds": 3.0, "p_win": 0.3})
        assert any("field_size" in s for s in r)

    def test_no_skip_when_healthy(self):
        r = skip_reasons({"field_size": 16}, {"field_chaos_score": 0.5},
                         {"tansho_odds": 3.0, "p_win": 0.3})
        assert r == []

    def test_low_pwin_skip(self):
        r = skip_reasons({"field_size": 16}, {"field_chaos_score": 0.5},
                         {"tansho_odds": 50.0, "p_win": 0.03})
        assert any("p_win" in s for s in r)


# ============================================================
# generate_results: 取消検出・返還・CI
# ============================================================
def _kekka_frame(rows):
    """kekka CSV と同じ列位置 (iloc[:,4]=馬番, iloc[:,6]=着順) の合成 DF。"""
    return pd.DataFrame(
        [["rid", "東京", "1", "x", ban, f"馬{ban}", jyun] for ban, jyun in rows],
        columns=["c0", "場所", "R", "c3", "馬番", "馬名", "着順"],
    )


class TestGetCancelled:
    def test_rank_zero_is_cancelled(self):
        kk = _kekka_frame([(1, "1"), (2, "2"), (5, "0"), (7, "3")])
        assert get_cancelled(kk) == {5}

    def test_no_cancelled(self):
        kk = _kekka_frame([(1, "1"), (2, "2"), (3, "3")])
        assert get_cancelled(kk) == set()


class TestBetCis:
    def _settled(self, n, hit_every, pay=200.0, cost=1000.0):
        rows = []
        for i in range(n):
            hit = 1 if i % hit_every == 0 else 0
            rows.append({"購入額": cost, "返還": 0.0,
                         "払戻": cost * pay / 100.0 if hit else 0.0,
                         "的中": hit})
        return pd.DataFrame(rows)

    def test_keys_and_ranges(self):
        out = _bet_cis(self._settled(100, 2))
        assert set(out) == {"hit_ci95", "roi_ci95", "roi_verdict"}
        lo, hi = out["roi_ci95"]
        assert lo <= hi
        h_lo, h_hi = out["hit_ci95"]
        assert 0 <= h_lo <= h_hi <= 100

    def test_deterministic(self):
        """seed 固定なので同一入力 → 同一 CI (週次再実行で数値が揺れない)。"""
        df = self._settled(60, 3)
        assert _bet_cis(df) == _bet_cis(df)

    def test_verdict_above_takeout(self):
        # 全 bet 的中・配当200 → ROI 200% で CI 下限が 80 を超える
        out = _bet_cis(self._settled(50, 1, pay=200.0))
        assert out["roi_verdict"] == "above_takeout"

    def test_verdict_below_takeout(self):
        # 全 bet 不的中 → ROI 0% で CI 上限が 80 未満
        out = _bet_cis(self._settled(50, 10**9))
        assert out["roi_verdict"] == "below_takeout"
