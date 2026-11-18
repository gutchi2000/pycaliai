# -*- coding: utf-8 -*-
"""公開勝率・複勝率の下限 (export_marks_json.PUBLISH_FLOOR_*) のテスト (2026-11-19)。

配信用 isotonic 較正器は最低スコア帯を 0 に写す。下限を入れた後は、公開される p_win / p_sho が
0 になる馬がいないこと、0 でない値は変わらないこと、混戦度 (race_confidence) は下限前の値で
計算されることを確かめる。
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

import export_marks_json as emj
from backtest_pl_ev import COL_BAN, COL_RID


class _Model:
    def __init__(self, scores):
        self.scores = np.asarray(scores, float)

    def predict(self, X, **_):
        return self.scores[: len(X)]


class _ZeroBelow:
    """閾値未満を 0 に、それ以上は恒等に写す較正器 (配信用 isotonic の最下段と同じ形)。"""

    def __init__(self, thr):
        self.thr = thr

    def predict(self, x):
        x = np.asarray(x, float)
        return np.where(x < self.thr, 0.0, x)


def _race(n=12):
    return pd.DataFrame({
        COL_RID: ["2099010101010101"] * n, COL_BAN: list(range(1, n + 1)),
        "馬名": [f"テスト{i}" for i in range(n)], "クラス名": "1勝", "日付": "20990101",
        "場所": "東京", "芝・ダ": "芝", "距離": 1600, "レース名": "", "f": np.arange(n, dtype=float),
    })


def _export(scores, cal):
    g = _race(len(scores))
    return emj.export_race("2099010101010101", g, _Model(scores), ["f"], {}, {}, {}, calibrators=cal)


def test_no_zero_published_probability_when_calibrator_outputs_zero():
    # 大差のスコア → 下位の PL 勝率・複勝率が閾値未満になり、較正器が 0 を返す
    scores = [5, 4, 3, 2, 1, 0, -2, -4, -6, -8, -10, -12]
    cal = {"tansho": _ZeroBelow(0.01), "fukusho": _ZeroBelow(0.05)}
    horses = _export(scores, cal)["horses"]
    assert all(h["p_win"] > 0 for h in horses)
    assert all(h["p_sho"] > 0 for h in horses)
    assert min(h["p_win"] for h in horses) == pytest.approx(emj.PUBLISH_FLOOR_WIN)
    assert min(h["p_sho"] for h in horses) == pytest.approx(emj.PUBLISH_FLOOR_SHO)


def test_nonzero_values_are_unchanged_and_chaos_uses_prefloor_vector():
    scores = [5, 4, 3, 2, 1, 0, -2, -4, -6, -8, -10, -12]
    cal = {"tansho": _ZeroBelow(0.01), "fukusho": _ZeroBelow(0.05)}
    out = _export(scores, cal)
    # 下限が無かった場合の値を同じ式で作って比べる
    import pl_probs as PL
    w = PL.pl_weights(np.asarray(scores, float))
    raw = cal["tansho"].predict(PL.all_tansho(w))
    by_ban = {h["umaban"]: h["p_win"] for h in out["horses"]}
    for i, v in enumerate(raw):
        if v > 0:
            assert by_ban[i + 1] == round(float(v), 4)
    # 混戦度は下限前の (0 を含む) ベクトルから
    conf = emj.race_confidence(raw, np.zeros(len(raw)), np.arange(1, len(raw) + 1))
    assert out["race_confidence"]["field_chaos_score"] == conf["field_chaos_score"]


def test_floor_is_below_every_production_calibrator_nonzero_output():
    """下限が既存の 0 でない公開値を動かさないこと (全較正器の 0 でない最小出力より小さい)。"""
    from pathlib import Path
    base = Path(emj.__file__).resolve().parent / "models"
    found = 0
    for p in sorted(base.glob("pl_calibrators_v*.pkl")):
        cals = joblib.load(p)
        cals = cals.get("calibrators", cals)
        for key, floor in (("tansho", emj.PUBLISH_FLOOR_WIN), ("fukusho", emj.PUBLISH_FLOOR_SHO)):
            iso = cals.get(key)
            if iso is None or not hasattr(iso, "y_thresholds_"):
                continue
            y = np.asarray(iso.y_thresholds_)
            pos = y[y > 0]
            if len(pos):
                found += 1
                assert floor < pos.min(), f"{p.name} {key}: floor {floor} >= min nonzero {pos.min()}"
    if not found:
        pytest.skip("models/ に較正器が無い (このチェックアウトには配置されていない)")
