"""
ev_filter.py
============
「防御力 3/10 → 8/10」実装モジュール。

機能:
1. BetFilter  ─ 分析で確認した「地獄のセグメント」を除外するフィルタ
2. EVCalibrator ─ EV の逆転現象（高 EV ほど ROI が低い）を補正する
                   アイソトニック回帰ベースのキャリブレーター

分析根拠（valid 2023 / test 2024 合計 6,910R）:
  ─────────────────────────────────────────────
  HELL セグメント（ROI < 60%）
    阪神 全体                  57.9%
    京都 16頭                  41.3%
    新潟 16頭                  26.4%
    阪神 16頭                  51.7%
    東京 16頭 昇級              40.4%
    京都 16頭 昇級              44.0%
    重馬場 × 16頭              50.5%  ← 2026-03-25 削除（3年n=376でROI73.1%、除外閾値未達）
    EV >= 3.0 (valid)          59.1%  ← valid で顕著
  ─────────────────────────────────────────────
  EV 逆転現象（EV が高いほど ROI が落ちる）
    EV 0.8-1.0  → ROI 83.2%
    EV 1.0-1.2  → ROI 84.3%
    EV 1.5-2.0  → ROI 78.0%
    EV 2.0-2.5  → ROI 77.7%
    EV 2.5+     → ROI 72.1%
  ─────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# =========================================================
# 設定定数（ここを変えるだけで戦略を切り替えられる）
# =========================================================

# 全面除外する競馬場（文字列は predict_weekly.py の「場所」カラムと一致）
# 京都: 全体ROI 12.9%（年間-¥348k）→ 16頭のみ除外では漏れが大きすぎた
SKIP_VENUES: set[str] = {"阪神", "京都"}

# 16頭立てのみ除外する競馬場（阪神・京都は全面除外に移動済み）
SKIP_16H_VENUES: set[str] = {"新潟"}

# 16頭 × 昇級戦を除外する競馬場（京都は全面除外に移動済み）
SKIP_16H_UPGRADE_VENUES: set[str] = {"東京", "新潟"}

# EV の上限（これ以上は過信と判定してスキップ）― 生 ev_score 用
EV_UPPER_LIMIT: float = 3.0

# EV 補正スコア（ev_cal）の下限。期待 ROI がこれ以下なら過信 or 低収益帯
# ev_cal = 0.78 → 期待 ROI 78%（EV 1.5-2.0 帯の実績相当）
EV_CAL_LOWER_LIMIT: float = 0.78

# 重馬場コード（馬場状態カラムの値）
HEAVY_TRACK_CODES: set[str] = {"重"}

# 不良馬場 × 16頭は逆に「爆発セグメント」（139%）なので除外しない
# FURYO_16H は買う（フラグで管理）

# クラスコードの基準（JRA 内部コード: 未勝利=10, 1勝=16, 2勝=23, 3勝=24 …）
# ※ master CSV の「クラス区分」カラムの数値
# ─ このコードは predict_weekly.py の前走クラス情報から判定

# キャリブレーターが学習したデータ（観測 ROI 表）
# 観測 EV 中央値とその実績 ROI から isotonic 補間用のアンカーを定義
_EV_ROI_ANCHORS = [
    # (ev_mid, observed_roi_ratio)   ← roi_ratio = 実収率 / 100
    (0.30,  0.950),   # EV 0.0-0.5 帯の実績（ROI 約 95% ← 0.5未満は少サンプル）
    (0.65,  0.798),   # EV 0.5-0.8
    (0.90,  0.832),   # EV 0.8-1.0
    (1.10,  0.843),   # EV 1.0-1.2
    (1.35,  0.833),   # EV 1.2-1.5
    (1.75,  0.780),   # EV 1.5-2.0
    (2.25,  0.777),   # EV 2.0-2.5
    (3.50,  0.721),   # EV 2.5+
]


# =========================================================
# 1. BetFilter ─ 地獄のセグメント除外フィルタ
# =========================================================

@dataclass
class FilterResult:
    should_skip: bool
    reason: str = ""


class BetFilter:
    """
    分析で特定した HELL セグメントをルールベースで除外する。

    Parameters
    ----------
    skip_venues : セット
        全面スキップする競馬場名
    skip_16h_venues : セット
        16頭立てのみスキップする競馬場名
    skip_16h_upgrade_venues : セット
        16頭 × 昇級戦のみスキップする競馬場名
    ev_upper : float
        この EV 以上は過信と判定してスキップ（デフォルト 3.0）
    """

    def __init__(
        self,
        skip_venues: set[str]           = SKIP_VENUES,
        skip_16h_venues: set[str]       = SKIP_16H_VENUES,
        skip_16h_upgrade_venues: set[str] = SKIP_16H_UPGRADE_VENUES,
        ev_upper: float                 = EV_UPPER_LIMIT,
        ev_cal_lower: float             = EV_CAL_LOWER_LIMIT,
    ):
        self.skip_venues            = skip_venues
        self.skip_16h_venues        = skip_16h_venues
        self.skip_16h_upgrade_venues = skip_16h_upgrade_venues
        self.ev_upper               = ev_upper
        self.ev_cal_lower           = ev_cal_lower

    def check(
        self,
        place: str,
        n_horses: int,
        baba: str,
        ev: float,
        is_upgrade: bool = False,
        ev_cal: float | None = None,
    ) -> FilterResult:
        """
        Parameters
        ----------
        place : str
            競馬場名（例: "阪神", "京都"）
        n_horses : int
            出走頭数
        baba : str
            馬場状態（例: "良", "稍重", "重", "不良"）
        ev : float
            ◎ の 生 EV スコア（model_prob × tansho_odds / 0.80）
        is_upgrade : bool
            ◎ が昇級戦かどうか（前走クラス < 今走クラス）
        ev_cal : float | None
            ◎ の 補正済み EV スコア（期待 ROI 比率, 0.72-0.95）
            None の場合は生 EV のみで判定

        Returns
        -------
        FilterResult
            should_skip=True のとき、このレースへの賭けをスキップ
        """
        # ── Rule 1: 全面除外競馬場 ──────────────────────────────
        if place in self.skip_venues:
            return FilterResult(True, f"{place}全面除外(ROI<60%)")

        # ── Rule 2: 16頭 × 除外競馬場 ────────────────────────────
        if n_horses >= 16 and place in self.skip_16h_venues:
            return FilterResult(True, f"{place} 16頭除外(ROI<52%)")

        # ── Rule 3: 16頭 × 昇級 × 東京/新潟 ────────────────────
        if n_horses >= 16 and is_upgrade and place in self.skip_16h_upgrade_venues:
            return FilterResult(True, f"{place} 16頭昇級除外(ROI<45%)")

        # ── Rule 4: 重馬場 × 16頭 ────────────────────────────────
        #   2026-03-25 削除: 元根拠 n=130（valid 2023 単年）、3年合計 n=376 で ROI73.1%
        #   除外閾値（<60%）未達のためルール廃止。n=500以上で再検証。
        #   → hypothesis_registry.md「ルール廃止記録」参照

        # ── Rule 5: EV 補正スコアによるフィルタ ──────────────────
        # ev_cal（期待ROI比率）が閾値以下 → 収益性が低い
        # 生 ev_score のフォールバック: ev >= 3.0 は旧ルール互換
        if ev_cal is not None and ev_cal > 0:
            if ev_cal < self.ev_cal_lower:
                return FilterResult(
                    True,
                    f"ev_cal={ev_cal:.3f}<{self.ev_cal_lower}(期待ROI{ev_cal*100:.1f}%)"
                )
        elif ev >= self.ev_upper:
            return FilterResult(
                True,
                f"EV={ev:.2f}>={self.ev_upper}上限除外(valid=59%)"
            )

        return FilterResult(False, "OK")

    def explain(self) -> str:
        """現在の設定を人間が読める形式で出力"""
        lines = [
            "── BetFilter 設定 ──────────────────",
            f"  全面除外:        {self.skip_venues}",
            f"  16頭除外:        {self.skip_16h_venues}",
            f"  16頭昇級除外:    {self.skip_16h_upgrade_venues}",
            f"  EV 上限(生):     {self.ev_upper}",
            f"  EV_cal 下限:     {self.ev_cal_lower} (期待ROI{self.ev_cal_lower*100:.0f}%未満で除外)",
            "────────────────────────────────────",
        ]
        return "\n".join(lines)


# =========================================================
# 2. EVCalibrator ─ EV 逆転現象の補正
# =========================================================

class EVCalibrator:
    """
    生 EV（model_prob × odds / return_rate）の逆転現象を補正する。

    観測データ:
      EV 0.8-1.0 → ROI 83.2%（高）
      EV 2.5+    → ROI 72.1%（低）  ← 逆転

    アプローチ:
      1. 観測 EV vs 実績 ROI からアイソトニック回帰を学習
      2. 生 EV を「期待 ROI」スコアに変換（高いほど良い = 修正済み）
      3. 買い推奨はこの calibrated_score をしきい値に使う

    Note: この calibration は「ランキング精度の向上」を目的とするもので、
          絶対的な ROI 予測値ではない点に注意。
    """

    def __init__(self):
        self._fitted = False
        self._ev_anchors  = np.array([a[0] for a in _EV_ROI_ANCHORS])
        self._roi_anchors = np.array([a[1] for a in _EV_ROI_ANCHORS])

    def fit_from_analysis(self, ev_series: pd.Series, roi_series: pd.Series) -> "EVCalibrator":
        """
        ev_analysis CSV から直接学習する（オプション）。

        Parameters
        ----------
        ev_series : pd.Series
            各レースの ◎ EV スコア
        roi_series : pd.Series
            各レースの実収率（0-2.0 スケール: 1.0=100%）
        """
        from sklearn.isotonic import IsotonicRegression
        # EV でソートして isotonic 回帰
        sorted_idx = np.argsort(ev_series.values)
        ev_sorted  = ev_series.values[sorted_idx]
        roi_sorted = roi_series.values[sorted_idx]
        self._iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
        self._iso.fit(ev_sorted, roi_sorted)
        self._fitted = True
        return self

    def transform(self, ev: float | np.ndarray) -> float | np.ndarray:
        """
        生 EV → calibrated_score（期待 ROI ベースのスコア）を返す。

        calibrated_score が高いほど「実際に回収できる可能性が高い」レース。
        デフォルトはアンカーポイントの線形補間を使用。
        """
        scalar = np.isscalar(ev)
        arr = np.atleast_1d(np.asarray(ev, dtype=float))

        if self._fitted:
            result = self._iso.predict(arr)
        else:
            # アンカーポイントで線形補間
            result = np.interp(arr, self._ev_anchors, self._roi_anchors)

        return float(result[0]) if scalar else result

    def recommend(
        self,
        ev: float,
        min_roi_threshold: float = 0.80,
    ) -> tuple[bool, str]:
        """
        EV と calibrated_score から「買い推奨」かどうかを判定。

        Parameters
        ----------
        ev : float
            生 EV スコア
        min_roi_threshold : float
            期待 ROI の最低ライン（デフォルト 0.80 = 80%）

        Returns
        -------
        (buy: bool, reason: str)
        """
        expected_roi = self.transform(ev)
        buy = bool(expected_roi >= min_roi_threshold)
        reason = (
            f"EV={ev:.2f} → 期待ROI={expected_roi*100:.1f}% "
            f"({'買推奨' if buy else 'ケン'})"
        )
        return buy, reason

    def calibrated_ev_label(self, ev: float) -> str:
        """EV に補正ラベルを付けて返す（表示用）"""
        score = self.transform(ev)
        stars = "★" * min(5, max(1, int(score * 6)))  # 0.80→4★, 0.85→5★ 程度
        return f"EV{ev:.2f}({stars})"


# =========================================================
# 3. is_upgrade_race ─ 昇級戦判定ヘルパー
# =========================================================

def is_upgrade_race(cls_now: float | None, cls_prev: float | None) -> bool:
    """
    現クラスコード > 前走クラスコードのとき昇級戦と判定。

    JRA クラスコード（参考）:
      未勝利=10, 1勝=16, 2勝=23, 3勝=24, OP=28 …
    """
    try:
        return float(cls_now) > float(cls_prev)
    except (TypeError, ValueError):
        return False


# =========================================================
# 4. make_filter_report ─ フィルタ効果のサマリー出力
# =========================================================

def make_filter_report(
    results: list[dict],
    filter_obj: BetFilter,
) -> str:
    """
    予測結果リストからフィルタ効果レポートを生成する。

    Parameters
    ----------
    results : list[dict]
        predict_weekly.main() が生成する行リスト
    filter_obj : BetFilter
    """
    total = len(set(r.get("レースID","") for r in results))
    skipped = sum(1 for r in results if r.get("フィルタ除外","") != "")
    passed  = total - skipped
    lines = [
        "── フィルタ適用結果 ─────────────────",
        f"  対象レース数:   {total}R",
        f"  通過（買い対象): {passed}R ({passed/max(total,1)*100:.1f}%)",
        f"  除外（ケン):     {skipped}R ({skipped/max(total,1)*100:.1f}%)",
        filter_obj.explain(),
    ]
    return "\n".join(lines)
