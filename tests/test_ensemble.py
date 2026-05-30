"""tests/test_ensemble.py — ensemble ユーティリティ関数のユニットテスト"""
import math
import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ensemble import assign_marks, ensemble_predict, MODEL_WEIGHTS


COL_RACE_ID = "レースID(新/馬番無)"


class TestEnsemblePredict:
    def _make_proba(self, n=10):
        np.random.seed(42)
        return np.random.rand(n)

    def test_output_shape(self):
        n = 10
        p = self._make_proba(n)
        result = ensemble_predict(p, p, p)
        assert result.shape == (n,)

    def test_weighted_average(self):
        """3モデル同値のとき、出力もその値になる"""
        p = np.array([0.5, 0.3, 0.8])
        result = ensemble_predict(p, p, p)
        np.testing.assert_allclose(result, p)

    def test_output_range(self):
        """確率値は0〜1の範囲内"""
        p1 = np.array([0.1, 0.9, 0.5])
        p2 = np.array([0.2, 0.8, 0.4])
        p3 = np.array([0.3, 0.7, 0.6])
        result = ensemble_predict(p1, p2, p3)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_weights_applied(self):
        """重みが正しく適用されているか検証"""
        w = MODEL_WEIGHTS
        total = sum(w.values())
        p1 = np.array([1.0])
        p2 = np.array([0.0])
        p3 = np.array([0.0])
        result = ensemble_predict(p1, p2, p3)
        expected = w["lgbm"] / total
        assert math.isclose(result[0], expected, rel_tol=1e-6)


class TestAssignMarks:
    def _make_df(self, race_id: str, n_horses: int) -> pd.DataFrame:
        return pd.DataFrame({
            COL_RACE_ID: [race_id] * n_horses,
            "馬番": list(range(1, n_horses + 1)),
            "fukusho_flag": [0] * n_horses,
        })

    def test_marks_assigned(self):
        """印が付与されることを確認"""
        df = self._make_df("R001", 8)
        proba = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4])
        result = assign_marks(df, proba)
        assert "mark" in result.columns
        assert "ensemble_prob" in result.columns

    def test_top_mark_is_hon_meiyo(self):
        """最高確率の馬が◎を得る"""
        df = self._make_df("R001", 5)
        proba = np.array([0.1, 0.9, 0.3, 0.7, 0.5])
        result = assign_marks(df, proba)
        top_idx = proba.argmax()
        assert result.loc[top_idx, "mark"] == "◎"

    def test_mark_order(self):
        """◎→◯→▲→△→× の順序で付与される（8頭立て）"""
        df = self._make_df("R001", 8)
        proba = np.linspace(0.1, 0.9, 8)  # 昇順
        result = assign_marks(df, proba)
        # 最大確率のインデックス（7番目）が◎
        assert result.loc[7, "mark"] == "◎"
        assert result.loc[6, "mark"] == "◯"
        assert result.loc[5, "mark"] == "▲"
        assert result.loc[4, "mark"] == "△"
        assert result.loc[3, "mark"] == "×"
        # 5位以下は印なし
        assert result.loc[0, "mark"] == ""

    def test_multiple_races(self):
        """複数レースが独立して印付与される"""
        df1 = self._make_df("R001", 4)
        df2 = self._make_df("R002", 4)
        df = pd.concat([df1, df2], ignore_index=True)
        proba = np.array([0.9, 0.5, 0.3, 0.1,   # R001
                          0.2, 0.8, 0.4, 0.6])   # R002
        result = assign_marks(df, proba)
        # R001の◎はindex 0（確率0.9）
        assert result.loc[0, "mark"] == "◎"
        # R002の◎はindex 5（確率0.8）
        assert result.loc[5, "mark"] == "◎"

    def test_original_df_not_modified(self):
        """元のDataFrameを変更しない"""
        df = self._make_df("R001", 5)
        df_copy = df.copy()
        proba = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
        assign_marks(df, proba)
        pd.testing.assert_frame_equal(df, df_copy)
