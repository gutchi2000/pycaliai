# -*- coding: utf-8 -*-
"""
本番ライン (v6 marks stack → compute_bets → validate → generate_results) の
純関数ゴールデンテスト。audit 2026-06-11「本番ラインのテスト0」対策。

実行: venv311\\Scripts\\python.exe -m pytest tests/test_production_line.py -q
データファイル非依存 (合成入力のみ)。
"""
import sys
import json
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compute_bets import amount_for_ev, allocate, MIN_BET, MAX_BET, BUDGET
import generate_results as gr
import validate_cowork_bets as vcb
from compute_bets import compute_race_bets
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

    def test_infeasible_minimum_raises(self):
        with pytest.raises(ValueError, match="minimum"):
            allocate([1.0, 1.0, 1.0], budget=1000)

    def test_budget_is_floored_to_100yen(self):
        assert allocate([1.0], budget=1050) == [1000]


class TestTopdownBudget:
    def test_low_budget_never_overspends(self):
        marks = {1: "◎", 2: "〇", 3: "▲", 4: "△"}
        horses = []
        for ban in range(1, 9):
            horses.append({
                "umaban": ban,
                "mark": marks.get(ban, ""),
                "p_win": 0.30 if ban == 1 else 0.10,
                "p_sho": 0.70 if ban == 1 else 0.25,
                "tansho_odds": 4.0 + ban,
                "fuku_odds_low": 1.5 + ban / 10,
                "fuku_odds_high": 1.7 + ban / 10,
            })
        race = {
            "race_id": "2099010101010101",
            "race_meta": {
                "race_id": "2099010101010101",
                "place": "東京",
                "field_size": 8,
            },
            "race_confidence": {
                "field_chaos_score": 0.50,
                "top1_dominance": 0.0,
                "top2_concentration": 0.0,
                "ai_market_agreement": 0.5,
            },
            "buy_judgment": {},
            "horses": horses,
            "umaren_matrix": {
                f"{i}-{j}": 20.0 for i in range(1, 9) for j in range(i + 1, 9)
            },
        }

        out = compute_race_bets(
            race, budget=1000, force_floor=True, engine="topdown")

        assert out["race_nature"] == "topdown"
        assert sum(b["購入額"] for b in out["bets"]) <= 1000
        assert all(b["購入額"] >= MIN_BET for b in out["bets"])


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

    def test_umatan_rejected(self):
        iss = content_issues({"馬券種": "馬単", "買い目": "1-2", "購入額": 1000},
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


class TestValidatorFailClosed:
    def test_bundle_missing_race_is_forced_to_skip(self, monkeypatch, tmp_path):
        bundle = tmp_path / "20990101_bundle.json"
        bets = tmp_path / "20990101_bets.json"
        bundle.write_text(json.dumps({"races": []}), encoding="utf-8")
        bets.write_text(json.dumps({"bets": [{
            "race_id": "2099010101010101",
            "race_label": "unknown",
            "race_nature": "test",
            "race_reason": "manual",
            "bets": [{"馬券種": "単勝", "買い目": "1", "購入額": 1000}],
        }]}), encoding="utf-8")
        monkeypatch.setattr(
            sys, "argv",
            ["validate_cowork_bets.py", "--date", "20990101",
             "--bets", str(bets), "--bundle", str(bundle), "--apply"],
        )

        assert vcb.main() == 0
        saved = json.loads(bets.read_text(encoding="utf-8"))["bets"][0]
        assert saved["bets"] == []
        assert saved["race_nature"] == "見送り"
        assert "bundle に race_id 不在" in saved["race_reason"]


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
    def test_roi_bootstrap_clusters_by_race(self):
        """同一レースの複数券を独立標本として扱わない。"""
        df = pd.DataFrame({
            "race_id": ["A"] * 10 + ["B"] * 10,
            "購入額": [1000.0] * 20,
            "返還": [0.0] * 20,
            "払戻": [0.0] * 10 + [2000.0] * 10,
            "的中": [0] * 10 + [1] * 10,
        })
        out = _bet_cis(df, n_boot=4000)
        assert out["roi_ci95"] == [0.0, 200.0]


    def test_verdict_above_takeout(self):
        # 全 bet 的中・配当200 → ROI 200% で CI 下限が 80 を超える
        out = _bet_cis(self._settled(50, 1, pay=200.0))
        assert out["roi_verdict"] == "above_takeout"

    def test_verdict_below_takeout(self):
        # 全 bet 不的中 → ROI 0% で CI 上限が 80 未満
        out = _bet_cis(self._settled(50, 10**9))
        assert out["roi_verdict"] == "below_takeout"

class TestCoworkUnsettled:
    @staticmethod
    def _race():
        return {
            "race_id": "2099010101010101",
            "race_label": "test",
            "bets": [{"馬券種": "単勝", "買い目": "1", "購入額": 1000}],
        }

    @staticmethod
    def _patch_source(monkeypatch, tmp_path, race):
        bets_dir = tmp_path / "cowork_bets"
        bets_dir.mkdir()
        monkeypatch.setattr(gr, "COWORK_BETS_DIR", bets_dir)
        monkeypatch.setattr(gr, "COWORK_OUTPUT_DIR", tmp_path / "cowork_output")
        monkeypatch.setattr(
            gr, "_iter_cowork_race_dicts",
            lambda: iter([("20990101", race, "test")]),
        )

    def test_unstarted_bet_is_not_counted_as_loss(self, monkeypatch, tmp_path):
        race = self._race()
        self._patch_source(monkeypatch, tmp_path, race)
        monkeypatch.setattr(gr, "get_race_kk", lambda *a, **k: pd.DataFrame())

        out = gr.aggregate_cowork_bets({})

        assert out["total"]["bet"] == 0
        assert out["total"]["bet_count"] == 0
        assert out["by_type"]["単勝"]["bet"] == 0
        assert out["bets"][0]["決着"] == "未開催"

    def test_match_error_is_not_counted_as_loss(self, monkeypatch, tmp_path):
        race = self._race()
        self._patch_source(monkeypatch, tmp_path, race)
        kk = _kekka_frame([(1, "1"), (2, "2"), (3, "3")])
        monkeypatch.setattr(gr, "get_race_kk", lambda *a, **k: kk)

        def fail_match(*args, **kwargs):
            raise ValueError("bad payout")

        monkeypatch.setattr(gr, "match_cowork_bet", fail_match)
        out = gr.aggregate_cowork_bets({})

        assert out["total"]["bet"] == 0
        assert out["total"]["bet_count"] == 0
        assert out["bets"][0]["決着"] == "照合失敗"


# ============================================================
# umami: 美味しさスコア (補正後期待回収率 + 来ない馬ゲート)
# ============================================================
from umami import umami


class TestUmami:
    def test_longshot_is_trap(self):
        """例の EV7.31 大穴 (107倍): 生EVが最大でも罠判定でゲート。"""
        u = umami("tansho", 0.068, 107.5)
        assert u["gated"] and u["grade"] == "罠"
        assert "大穴帯" in u["gate_reason"]

    def test_low_psho_is_trap(self):
        u = umami("fukusho", 0.05, 3.0, tansho_odds=40)
        assert u["gated"] and "来る見込み薄" in u["gate_reason"]

    def test_mid_odds_value_passes(self):
        u = umami("tansho", 0.25, 5.0)
        assert not u["gated"]
        assert u["xroi"] is not None and 0.5 < u["xroi"] < 1.2

    def test_xroi_not_monotone_in_ev(self):
        """生EVが大きいほど xroi が高い、にはならない (EV逆転の織込み確認)。"""
        low_ev = umami("tansho", 0.30, 4.0)    # EV 1.2 中人気
        high_ev = umami("tansho", 0.06, 45.0)  # EV 2.7 大穴ぎりぎり
        assert low_ev["xroi"] >= high_ev["xroi"]

    def test_missing_inputs_gated(self):
        assert umami("tansho", None, None)["gated"]


class TestValueHorsesUmamiGate:
    def test_trap_horse_excluded(self):
        """生EVが巨大でも罠馬 (大穴) は妙味馬リストに入らない。"""
        from betting_judgment import extract_value_horses
        horses = [
            {"umaban": 13, "horse_name": "罠馬", "p_win": 0.068,
             "tansho_odds": 107.5, "p_sho": 0.206,
             "fuku_odds_low": 15.0, "fuku_odds_high": 19.0,
             "ai_vs_market": "under"},
            {"umaban": 4, "horse_name": "妙味馬", "p_win": 0.25,
             "tansho_odds": 5.0, "p_sho": 0.60,
             "fuku_odds_low": 1.8, "fuku_odds_high": 2.2,
             "ai_vs_market": "under"},
        ]
        out = extract_value_horses(horses)
        names = [v["horse_name"] for v in out]
        assert "妙味馬" in names
        assert "罠馬" not in names
