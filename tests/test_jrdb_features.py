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


# ---- 配信用の 1 関数と、カバレッジの測り方 (v7 本番化、2026-11-13) ----
def test_coverage_column_maps_sparse_columns_to_run_presence():
    assert jf.coverage_column("j1_furi_1") == "j1_idm_1"
    assert jf.coverage_column("j1_deokure_3") == "j1_idm_3"
    assert jf.coverage_column("j1_idm_2") == "j1_idm_2"
    assert jf.coverage_column("j1_idm_mean5") == "j1_idm_mean5"
    assert jf.coverage_column("kako5_avg_pos") == "kako5_avg_pos"


def test_export_coverage_series_uses_run_presence_for_jrdb_columns():
    from export_weekly_marks import coverage_series, feature_coverage
    df = pd.DataFrame({"j1_idm_1": [50.0, 60.0, np.nan, 55.0], "j1_furi_1": [np.nan, 2.0, np.nan, np.nan], "x": [1.0, 2.0, np.nan, np.nan]})
    assert feature_coverage(coverage_series(df, "j1_furi_1")) == pytest.approx(0.75)     # 不利の記録は 1 頭だけだが、走は 3 頭ある
    assert feature_coverage(coverage_series(df, "x")) == pytest.approx(0.5)


def test_add_serve_feats_flags_missing_identity_and_stale_history(hist):
    from serve_history_feats import IDENTITY_STATUS_COL, PED_ID_COL
    df = pd.DataFrame({"馬名": ["アルファ"]})
    assert "馬同定" in jf.add_serve_feats(df, "20260312", hist)["error"]
    df = pd.DataFrame({"馬名": ["アルファ", "ベータ"], PED_ID_COL: [2010100001.0, -5.0], IDENTITY_STATUS_COL: ["hit", "hit"]})
    ok = jf.add_serve_feats(df, "20260312", hist)
    assert ok["error"] is None and ok["stale_days"] == 5 and ok["coverage"] == 1.0
    assert df.j1_idm_1.tolist() == [65.0, 40.0]                      # 2 頭目は仮番号 → 馬名で引き直し
    stale = jf.add_serve_feats(df, "20260328", hist)                 # 成績の最終 3/07 から 21 日
    assert stale["stale_days"] == 21 and "古い" in stale["error"]


def test_add_serve_feats_does_not_use_name_for_unidentified_rows(hist):
    from serve_history_feats import IDENTITY_STATUS_COL, PED_ID_COL
    df = pd.DataFrame({"馬名": ["ベータ", "ベータ"], PED_ID_COL: [np.nan, np.nan], IDENTITY_STATUS_COL: ["ambiguous", "new"]})
    r = jf.add_serve_feats(df, "20260312", hist)
    assert df.j1_idm_1.isna().all() and "埋まり" in r["error"]


def test_production_model_tag_and_switch_table_are_consistent():
    import json
    from pathlib import Path
    import production_policy as pp
    sys_path = Path(pp.__file__).resolve().parent
    import importlib.util
    spec = importlib.util.spec_from_file_location("switch_production_model", sys_path / "scripts" / "switch_production_model.py")
    sw = importlib.util.module_from_spec(spec); spec.loader.exec_module(sw)
    pp.clear_policy_caches()
    pol = pp.load_policy()
    tag = pp.production_model_tag()
    assert tag in sw.MODELS
    assert pol["artifacts"]["rank_model"] == sw.MODELS[tag]["rank_model"]
    assert pol["artifacts"]["serve_calibrator"] == sw.MODELS[tag]["serve_calibrator"]
    assert pp.artifact_path("rank_model").name == f"unified_rank_{tag}.pkl"
    pins = json.loads((sys_path / "data" / "artifact_pins.json").read_text(encoding="utf-8"))["pins"]
    for rel in (pol["artifacts"]["rank_model"], pol["artifacts"]["serve_calibrator"]):
        assert rel in pins, f"本番成果物 {rel} が artifact_pins に無い"
    # 混戦度の基準分布は方策が宣言する版と一致していること
    q = json.loads((sys_path / "data" / "chaos_quantiles.json").read_text(encoding="utf-8"))
    assert q["reference_id"] == pol["chaos_reference"]["reference_id"]
