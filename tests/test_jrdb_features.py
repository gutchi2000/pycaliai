# -*- coding: utf-8 -*-
"""jrdb_features: 過去走特徴の as-of・公開禁止・名前での引き直しの契約 (v7 候補、2026-11-13)。実データは使わない。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import jrdb_features as jf


def _sed(tmp_path, rows):
    cols = ["ped", "ymd", "name", "idm", "soten", "furi", "mae_furi", "naka_furi", "ato_furi", "deokure", "ichidori", "course_dori",
            "yon_course", "joushou", "ten_idx", "agari_idx", "pace_idx", "racep_idx", "batai_cd", "kehai_cd", "uma_pace", "race_pace"]
    df = pd.DataFrame([{**{c: np.nan for c in cols}, **r} for r in rows])
    df["batai_cd"] = df["batai_cd"].astype(object); df["kehai_cd"] = df["kehai_cd"].astype(object)
    p = tmp_path / "sed.parquet"
    df.to_parquet(p, index=False)
    return jf.load_history(p)


@pytest.fixture
def hist(tmp_path):
    rows = [dict(ped="10100001", ymd=20260101, name="アルファ", idm=50.0, agari_idx=1.0, uma_pace="H", batai_cd="3"),
            dict(ped="10100001", ymd=20260201, name="アルファ", idm=55.0, agari_idx=2.0, uma_pace="M"),
            dict(ped="10100001", ymd=20260301, name="アルファ", idm=60.0, agari_idx=3.0, uma_pace="S"),
            dict(ped="10100001", ymd=20260307, name="アルファ", idm=65.0, agari_idx=4.0),
            dict(ped="10100002", ymd=20260215, name="ベータ", idm=40.0),
            dict(ped="15000009", ymd=20200101, name="ベータ", idm=99.0)]      # 同名の古い別馬
    return _sed(tmp_path, rows)


def test_feature_list_is_prefixed_and_complete():
    assert len(jf.FEATURES) == 59
    assert all(f.startswith(jf.JRDB_FEATURE_PREFIX) for f in jf.FEATURES)


def test_lags_use_only_runs_at_least_five_days_old(hist):
    ped = pd.Series([2010100001.0, 2010100001.0, 2010100001.0])     # 10 桁でも下 8 桁で引ける
    ymd = pd.Series([20260310, 20260312, 20260301])
    f = jf.past_run_features(ped, ymd, hist)
    # 3/10: 3/07 の走は 3 日前 → 使えない。前走は 3/01
    assert f.j1_idm_1.iloc[0] == 60.0 and f.j1_idm_2.iloc[0] == 55.0 and f.j1_idm_3.iloc[0] == 50.0
    # 3/12: 3/07 の走は 5 日前 → 使える
    assert f.j1_idm_1.iloc[1] == 65.0 and f.j1_idm_2.iloc[1] == 60.0
    # 3/01 当日の走は自分自身 → 使わない
    assert f.j1_idm_1.iloc[2] == 55.0
    assert f.j1_idm_mean5.iloc[1] == pytest.approx(np.mean([65, 60, 55, 50]))
    assert f.j1_idm_max5.iloc[1] == 65.0
    assert f.j1_uma_pace_n_1.iloc[0] == 3 and f.j1_uma_pace_n_3.iloc[0] == 1 and f.j1_batai_3.iloc[0] == 3


def test_unknown_horse_and_bad_date_are_all_nan(hist):
    f = jf.past_run_features(pd.Series([99999999.0, np.nan, 2010100001.0]), pd.Series([20260310, 20260310, np.nan]), hist)
    assert f.isna().all().all()


def test_never_uses_runs_after_the_target_date(hist):
    f = jf.past_run_features(pd.Series([2010100001.0]), pd.Series([20260110]), hist)
    assert f.j1_idm_1.iloc[0] == 50.0 and np.isnan(f.j1_idm_2.iloc[0])


def test_name_fallback_only_when_asked_and_picks_recent_horse(hist):
    synth = pd.Series([-123456.0, -123456.0])                       # 配信側の仮番号 (SED に無い)
    ymd = pd.Series([20260310, 20260310])
    assert jf.past_run_features(synth, ymd, hist).isna().all().all()            # 名前を渡さなければ引かない
    f = jf.past_run_features(synth, ymd, hist, name=pd.Series(["ベータ", "　アルファ "]))
    assert f.j1_idm_1.iloc[0] == 40.0       # 2020 年の同名馬 (99) ではなく、直近に走っている馬
    assert f.j1_idm_1.iloc[1] == 60.0       # 全角空白つきでも一致
    # 名前が None (同定が hit でない行) は引かない
    g = jf.past_run_features(synth, ymd, hist, name=pd.Series([None, None], dtype=object))
    assert g.isna().all().all()


def test_name_fallback_does_not_reach_into_the_future(hist):
    # 2026-01-15 時点では「ベータ」(初出走 2/15) はまだ走っていない。2020 年の同名馬は 1100 日より古い → 引かない
    f = jf.past_run_features(pd.Series([-1.0]), pd.Series([20260115]), hist, name=pd.Series(["ベータ"]))
    assert f.isna().all().all()


def test_add_jrdb_feats_serve_equals_training(hist):
    df = pd.DataFrame({"血統登録番号": [2010100001], "日付": [20260312]})
    a = jf.add_jrdb_feats(df, history=hist)
    b = jf.add_jrdb_feats(df.drop(columns=["日付"]), date_value="20260312", history=hist)
    assert np.allclose(a[jf.FEATURES].values, b[jf.FEATURES].values, equal_nan=True)
    with pytest.raises(ValueError):
        jf.add_jrdb_feats(df, mode="unknown", history=hist)


def test_staleness_days(hist):
    assert jf.staleness_days(20260314, hist) == 7
    assert jf.staleness_days(20260321, hist) == 14


def test_shap_reasons_never_expose_jrdb_features():
    import marks_shap as ms

    class _Exp:
        def shap_values(self, X):
            return np.array([[5.0, 0.1, 3.0]])
    g = pd.DataFrame({"j1_idm_1": [60.0], "kako5_avg_pos": [3.0], "j1_agari_idx_1": [2.0]})
    out = ms.race_contribs(_Exp(), np.zeros((1, 3)), ["j1_idm_1", "kako5_avg_pos", "j1_agari_idx_1"], g, topk=3)
    assert [w["feat"] for w in out[0]] == ["kako5_avg_pos"]


def test_site_scrub_drops_jrdb_reasons():
    import inspect
    import build_site
    src = inspect.getsource(build_site)
    assert 'startswith("j1_")' in src
