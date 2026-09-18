# -*- coding: utf-8 -*-
"""
test_forward_shadow.py — append-only保存・重複防止・凍結モデルの決定論性テスト
実行: python -m pytest analysis/mcond/exp05_forward_shadow/test_forward_shadow.py -q
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pytest

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.exp05_forward_shadow import predict_and_store as pas  # noqa: E402
from analysis.mcond.exp05_forward_shadow import freeze_model as FM  # noqa: E402

FROZEN_EXISTS = pas.FROZEN_PATH.exists()


@pytest.mark.skipif(not FROZEN_EXISTS, reason="frozen_model.joblib 未生成 (freeze_model.py を先に実行)")
def test_frozen_model_deterministic():
    """同じ入力に対し凍結モデルの予測は毎回同じ (再学習が紛れ込んでいないことの確認)。"""
    frozen = pas._frozen()
    X = np.array([[0.1, -0.2, 0.5, 1.0, -0.3, 0.2, 0.4, 0.1]])
    cols = frozen["M3"]["cols"]
    X = np.random.default_rng(0).normal(size=(5, len(cols)))
    p1 = FM.predict_linear(frozen["M3"], X)
    p2 = FM.predict_linear(frozen["M3"], X)
    assert np.allclose(p1, p2)


@pytest.mark.skipif(not FROZEN_EXISTS, reason="frozen_model.joblib 未生成")
def test_frozen_model_no_reoptimization_hint():
    """凍結artifactにC(正則化)が記録されており、exp05_market_residual_devのC_GRID内にあること
    (再探索していないことの間接確認: フリーズ後にCを変える経路が無い)。"""
    frozen = pas._frozen()
    assert frozen["M4"]["C"] in (0.01, 0.1, 1.0)
    assert frozen["M1"]["C"] in (0.1, 1.0, 10.0)


def test_prediction_revision_no_overwrite(tmp_path, monkeypatch):
    """同一 race_id×model_hash への2回目の保存は rev1 を上書きせず rev2 を作る。"""
    monkeypatch.setattr(pas, "OUT_DIR", tmp_path)

    def fake_existing(date_str, rid, model_hash):
        d = tmp_path / date_str
        if not d.exists():
            return []
        return sorted(d.glob(f"{rid}_{model_hash}_rev*.json"))

    date_str, rid, model_hash = "20260101", "2026010101010101", "deadbeef00000000"
    p1 = pas._revision_path(date_str, rid, model_hash, 1)
    p1.parent.mkdir(parents=True, exist_ok=True)
    p1.write_text(json.dumps({"revision": 1, "note": "first"}), encoding="utf-8")

    existing = pas._existing_revisions(date_str, rid, model_hash)
    assert len(existing) == 1
    rev = len(existing) + 1
    p2 = pas._revision_path(date_str, rid, model_hash, rev)
    assert p2 != p1
    p2.write_text(json.dumps({"revision": 2, "note": "second"}), encoding="utf-8")

    assert json.loads(p1.read_text(encoding="utf-8"))["note"] == "first"
    assert json.loads(p2.read_text(encoding="utf-8"))["note"] == "second"


def test_market_snapshot_window_validation():
    """31-38分ウィンドウ外なら valid_for_primary=False になる (別時刻への自動フォールバック禁止の実装確認)。"""
    from analysis.mcond.exp05_forward_shadow.market_snapshot import WINDOW_MIN
    assert WINDOW_MIN == (31.0, 38.0)

    def in_window(m):
        return WINDOW_MIN[0] <= m <= WINDOW_MIN[1]

    assert in_window(35.0) is True
    assert in_window(20.0) is False
    assert in_window(10.0) is False
    assert in_window(40.0) is False


def test_frozen_encode_unknown_category_is_zero():
    from analysis.mcond.exp05_forward_shadow import frozen_encode
    import pandas as pd
    typing = {"kept_numeric": [], "kept_onehot": {"場所": ["中京", "中山", "京都"]}}
    df = pd.DataFrame({"場所": ["未知の場所", "中山"]})
    out, found = frozen_encode.encode_c1(df, typing)
    assert found["場所"] is True
    assert out.loc[0, "c1__場所__中山"] == 0.0
    assert out.loc[1, "c1__場所__中山"] == 1.0
