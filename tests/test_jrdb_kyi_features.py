# -*- coding: utf-8 -*-
"""jrdb_kyi_features (v8 候補: JRDB 前日 KYI + 外厩) の単体テスト。JRDB の実データは使わない。"""
import numpy as np
import pandas as pd

import jrdb_kyi_features as kf


def test_rid16_to_racekey_hex_day():
    # YYYYMMDD 場2 回2 日2 R2 → 場2 年2 回1 日1(16 進) R2。10 日目は "a"
    rid = pd.Series(["2026111405041012", "2024010606010101"])
    assert kf.rid16_to_racekey(rid).tolist() == ["05264a12", "06241101"]


def test_gaikyu_comment_parse_and_days():
    rows = kf._gk_rows("YC012411.DAT", "0124110102,西山牧場\u300007/03\u3000中3週\n0124110105,社台Ｆ\u300007/10\u3000中週\n")
    assert rows[0] == ("01241101", 2, "西山牧場", "07/03", 3.0)
    assert rows[1][1] == 5 and np.isnan(rows[1][4])
    g = pd.DataFrame(rows, columns=["racekey", "ban", "gk_fac", "gk_mmdd", "gk_weeks"])
    k = pd.DataFrame({"racekey": ["01241101"] * 3, "ban": [1, 2, 5]})
    for c in kf.KYI_NUM:
        k[c] = 1.0
    k["d_jun"] = k["a_jun"] = k["g_jun"] = [1.0, 2.0, 3.0]
    keys = pd.DataFrame({"racekey": ["01241101"] * 3, "ban": [1, 2, 5]})
    f = kf.make_features(keys, pd.Series([20240720] * 3), k, g, {"西山牧場": 1})
    assert list(f.columns) == kf.FEATURES
    assert f["k8_gk_has"].tolist() == [0.0, 1.0, 1.0]
    assert f["k8_gk_fac"].tolist()[1:] == [1.0, 99.0]          # 上位表に無い施設は 99
    assert f["k8_gk_days"].tolist()[1:] == [17.0, 10.0]          # 7/03 → 7/20、7/10 → 7/20
    assert f["k8_d_jun_rel"].tolist() == [0.0, 0.5, 1.0]


def test_gaikyu_days_wraps_year():
    g = pd.DataFrame([("01250101", 1, "A", "12/20", 4.0)], columns=["racekey", "ban", "gk_fac", "gk_mmdd", "gk_weeks"])
    k = pd.DataFrame({"racekey": ["01250101"], "ban": [1]})
    for c in kf.KYI_NUM:
        k[c] = 1.0
    f = kf.make_features(pd.DataFrame({"racekey": ["01250101"], "ban": [1]}), pd.Series([20250105]), k, g, {"A": 1})
    assert f["k8_gk_days"].iloc[0] == 16.0


def test_private_prefixes_hide_k8_from_bundle():
    import marks_shap
    assert "k8_" in marks_shap._PRIVATE_PREFIXES and "j1_" in marks_shap._PRIVATE_PREFIXES
    assert all(c.startswith(kf.PREFIX) for c in kf.FEATURES)


def test_no_market_columns_in_features():
    # 基準オッズ・基準人気・人気指数は学習に入れない (オッズ系は v6/v7 と同じく除外)
    bad = ("odds", "ninki")
    assert not any(b in c for c in kf.FEATURES for b in bad)
