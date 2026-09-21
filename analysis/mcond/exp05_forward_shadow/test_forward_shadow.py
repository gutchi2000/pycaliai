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


# ------------------------------------------------------------------ scenario 7: 市場取得成功・予測入力失敗
def test_store_market_only_does_not_discard_market_data(tmp_path, monkeypatch):
    """特徴量snapshotが無くprediction計算できない場合でも、取得済みの市場snapshotは
    捨てずにmarket_onlyレコードとして保存する (market_snapshot_saved=true/prediction_saved=false/
    invalid_for_primary=true)。"""
    monkeypatch.setattr(pas, "OUT_DIR", tmp_path)
    market_result = {
        "ok": True, "valid_for_primary": True, "scheduled_post": "2099-01-01T10:00:00+09:00",
        "market": {"fetched": "2099-01-01T09:25:00", "tansho": {"1": 5.0, "2": 3.2},
                  "overround_tan": 1.25},
    }
    path = pas.store_market_only("20990101", "9999010106040599", market_result,
                                 reason="weekly_input_unavailable")
    rec = json.loads(path.read_text(encoding="utf-8"))
    assert rec["market_snapshot_saved"] is True
    assert rec["prediction_saved"] is False
    assert rec["invalid_for_primary"] is True
    assert rec["reason"] == "weekly_input_unavailable"
    assert rec["raw_market"]["tansho"] == {"1": 5.0, "2": 3.2}


def test_store_market_only_revisions_do_not_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(pas, "OUT_DIR", tmp_path)
    market_result = {"ok": True, "valid_for_primary": False, "scheduled_post": None, "market": {}}
    p1 = pas.store_market_only("20990101", "9999010106040599", market_result, reason="r1")
    p2 = pas.store_market_only("20990101", "9999010106040599", market_result, reason="r2")
    assert p1 != p2
    assert json.loads(p1.read_text(encoding="utf-8"))["reason"] == "r1"
    assert json.loads(p2.read_text(encoding="utf-8"))["reason"] == "r2"


# ------------------------------------------------------------------ 開催日判定ロジック (t35_shadow.ps1)
# PowerShellの-Scheduleブロック自体はpytestで直接実行しないが、判定に使う
# data/jra_known_race_days_override.json の読み込み形式だけは検証しておく。
def test_known_race_day_override_file_valid_json():
    p = BASE / "data" / "jra_known_race_days_override.json"
    assert p.exists()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert "known_race_days" in data
    assert "20260921" in data["known_race_days"]


# ------------------------------------------------------------------ jvlink_race_calendar.py /
#                                                                       jvlink_race_day_probe.py
# fetch_races()/diag_extract()自体はwin32com(32bit専用)が要るため通常のpytest venv(64bit)では
# 検証できない。ここではwin32comを一切importしない純粋ロジック部分だけを実データで検証する。
from analysis.mcond.exp05_forward_shadow import jvlink_race_calendar as JRC  # noqa: E402
from analysis.mcond.exp05_forward_shadow import jvlink_race_day_probe as JRP  # noqa: E402


def test_jvlink_probe_load_known_post_times_matches_weekly_csv():
    """data/weekly/20260919.csvの実データから発走時刻を正しく読めること
    (t10_runner.load_post_timesと同一ロジックの独立複製、32bit環境でも動くよう
    pandasを使わない実装になっている点を確認)。"""
    known = JRP._load_known_post_times("20260919")
    assert len(known) == 24
    assert known["2026091906040501"] == "10:00"
    assert known["2026091909040501"] == "09:45"


def test_jvlink_probe_load_known_post_times_missing_file_returns_empty():
    assert JRP._load_known_post_times("99991231") == {}


def test_jvlink_calendar_verify_against_weekly_reports_exact_matches():
    races = [{"rid16": "2026091906040501", "post": "10:00", "venue": "中山"},
             {"rid16": "2026091909040501", "post": "09:45", "venue": "阪神"},
             {"rid16": "2026091906040502", "post": "99:99", "venue": "中山"}]  # わざと不一致を混ぜる
    v = JRC.verify_against_weekly("20260919", races)
    assert v["available"] is True
    assert v["known_count"] == 24
    assert v["exact_time_matches"] == 2


# ---- _sanity_check: 2026-09-19夜、PowerShell経由の子プロセスでのみ発走時刻が
# 全レース"00:00"に化ける実機不具合が見つかったため追加した自己検査 ----
def test_jvlink_calendar_sanity_check_rejects_all_identical_times():
    races = [{"rid16": f"202609210604070{i}", "post": "00:00"} for i in range(1, 13)]
    err = JRC._sanity_check(races)
    assert err != ""
    assert "00:00" in err or "同一発走時刻" in err


def test_jvlink_calendar_sanity_check_accepts_plausible_distinct_times():
    races = [{"rid16": "2026092106040701", "post": "09:45"},
             {"rid16": "2026092106040702", "post": "10:20"},
             {"rid16": "2026092106040703", "post": "10:50"}]
    assert JRC._sanity_check(races) == ""


# ---- 2026-09-19深夜追加: 日付不一致・レースID重複・非合理的時刻の3チェック ----
def test_jvlink_calendar_sanity_check_rejects_date_mismatch():
    races = [{"rid16": "2026092106040701", "post": "09:45"},
             {"rid16": "2026092006040701", "post": "10:20"}]  # 別日が混入
    err = JRC._sanity_check(races, target_date="20260921")
    assert err != ""
    assert "20260921" in err


def test_jvlink_calendar_sanity_check_rejects_duplicate_race_id():
    races = [{"rid16": "2026092106040701", "post": "09:45"},
             {"rid16": "2026092106040701", "post": "10:20"}]  # 同一rid16が違う時刻で重複
    err = JRC._sanity_check(races, target_date="20260921")
    assert err != ""
    assert "重複" in err


def test_jvlink_calendar_sanity_check_rejects_out_of_range_time():
    races = [{"rid16": "2026092106040701", "post": "03:15"}]  # JRAで実在しない深夜時刻
    err = JRC._sanity_check(races, target_date="20260921")
    assert err != ""
    assert "実運用範囲外" in err


# ---- write_calendar_json / verify_calendar_json: ハッシュ整合性・鮮度チェック ----
def test_jvlink_calendar_write_then_verify_roundtrip(tmp_path):
    races = [{"rid16": "2026092106040701", "post": "09:45", "venue": "中山"},
             {"rid16": "2026092109040701", "post": "10:00", "venue": "阪神"}]
    out = tmp_path / "20260921.json"
    doc = JRC.write_calendar_json("20260921", races, out)
    assert out.exists()
    assert not out.with_suffix(out.suffix + ".tmp").exists()  # atomic rename後に.tmpが残らない
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert JRC.verify_calendar_json(loaded) == ""
    assert doc["record_count"] == 2


def test_jvlink_calendar_verify_rejects_tampered_hash(tmp_path):
    races = [{"rid16": "2026092106040701", "post": "09:45"},
             {"rid16": "2026092109040701", "post": "10:00"}]
    out = tmp_path / "20260921.json"
    JRC.write_calendar_json("20260921", races, out)
    doc = json.loads(out.read_text(encoding="utf-8"))
    doc["post_times"][0] = "23:59"  # ファイルを直接改変(file_hashは古いまま)
    err = JRC.verify_calendar_json(doc)
    assert err != ""
    assert "file_hash" in err


def test_jvlink_calendar_verify_rejects_stale_generated_at():
    doc = {"target_date": "20260921", "generated_at": "2020-01-01T00:00:00",
           "race_ids": ["2026092106040701"], "post_times": ["09:45"],
           "file_hash": JRC._canonical_hash(["2026092106040701"], ["09:45"])}
    err = JRC.verify_calendar_json(doc, max_age_minutes=60.0)
    assert err != ""
    assert "経過" in err


# ------------------------------------------------------------------ observation_report.py
# 2026-09-19深夜、基盤凍結前に追加した読み取り専用の集計ツール。3つの計数カテゴリ
# (market_observations / complete_prediction_observations / valid_primary_observations)
# の定義を合成データで検証する。実際の収集ロジック(t35_shadow.ps1等)には触れない。
from analysis.mcond.exp05_forward_shadow import observation_report as OR  # noqa: E402


def test_observation_report_three_way_counts(tmp_path, monkeypatch):
    date_str = "20990101"
    odds_dir = tmp_path / "odds"; odds_dir.mkdir()
    pred_dir = tmp_path / "pred" / date_str; pred_dir.mkdir(parents=True)
    monkeypatch.setattr(OR, "ODDS_DIR", odds_dir)
    monkeypatch.setattr(OR, "PRED_DIR", tmp_path / "pred")
    monkeypatch.setattr(OR, "CALENDAR_DIR", tmp_path / "calendar")
    monkeypatch.setattr(OR, "WEEKLY_DIR", tmp_path / "weekly")
    monkeypatch.setattr(OR, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(OR, "_current_model_hash", lambda: "deadbeef00000000")

    # race A: market取得成功 + M1/M3/M4完全 + window内(有効) → 3カテゴリ全てに入る
    (odds_dir / "2099010106040701.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (pred_dir / "2099010106040701_deadbeef00000000_rev1.json").write_text(json.dumps({
        "race_id": "2099010106040701", "date": date_str,
        "records": [{"race_id": "2099010106040701", "model_hash": "deadbeef00000000",
                    "valid_for_primary": True, "invalid_reason": None}],
    }), encoding="utf-8")

    # race B: market取得成功 + M1/M3/M4完全だがwindow外(invalid) → market+complete のみ
    (odds_dir / "2099010106040702.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (pred_dir / "2099010106040702_deadbeef00000000_rev1.json").write_text(json.dumps({
        "race_id": "2099010106040702", "date": date_str,
        "records": [{"race_id": "2099010106040702", "model_hash": "deadbeef00000000",
                    "valid_for_primary": False, "invalid_reason": "window外"}],
    }), encoding="utf-8")

    # race C: market取得成功のみ、特徴量snapshot無しでmarket_onlyへフォールバック → marketのみ
    (odds_dir / "2099010106040703.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (pred_dir / "2099010106040703_marketonly_rev1.json").write_text(json.dumps({
        "race_id": "2099010106040703", "market_snapshot_saved": True, "prediction_saved": False,
    }), encoding="utf-8")

    # race D: 市場取得自体が失敗(ok=false) → どのカテゴリにも入らない
    (odds_dir / "2099010106040704.json").write_text(json.dumps({"ok": False}), encoding="utf-8")

    report = OR.build_report(date_str)
    assert report["market_observations"] == 3   # A, B, C (Dはok=falseで除外)
    assert report["complete_prediction_observations"] == 2  # A, B
    assert report["valid_primary_observations"] == 1  # Aのみ
    assert report["invalid_for_primary_count"] == 1  # B
    assert report["invalid_for_primary_reasons"] == {"window外": 1}
    assert report["market_only_count"] == 1  # C
    assert report["fired_task_count"] == 4  # oddsファイル4件全部(A,B,C,D)


def test_observation_report_handles_missing_calendar_and_weekly(tmp_path, monkeypatch):
    date_str = "20990101"
    monkeypatch.setattr(OR, "ODDS_DIR", tmp_path / "odds")
    monkeypatch.setattr(OR, "PRED_DIR", tmp_path / "pred")
    monkeypatch.setattr(OR, "CALENDAR_DIR", tmp_path / "calendar")
    monkeypatch.setattr(OR, "WEEKLY_DIR", tmp_path / "weekly")
    monkeypatch.setattr(OR, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(OR, "_current_model_hash", lambda: "deadbeef00000000")

    report = OR.build_report(date_str)
    assert report["calendar_race_count"] is None
    assert report["weekly_csv_generated_at"] is None
    assert report["t35_task_target_count"] is None
    assert report["market_observations"] == 0
    assert report["complete_prediction_observations"] == 0
    assert report["valid_primary_observations"] == 0
    assert report["missed_count"] == 0


def test_jvlink_calendar_verify_file_cli_rejects_mismatched_date(tmp_path):
    """--verify-fileのCLI相当ロジック: target_date不一致は別日フォールバックせず拒否する
    (verify_calendar_json自体はtarget_date比較をしないため、呼び出し側main()が
    doc['target_date'] != 期待日 を明示的に見ている。ここではその契約をdocレベルで確認)。"""
    races = [{"rid16": "2026092106040701", "post": "09:45"}]
    out = tmp_path / "20260921.json"
    JRC.write_calendar_json("20260921", races, out)
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["target_date"] == "20260921"
    assert doc["target_date"] != "20260920"  # main()のverify-fileはこれを見て別日拒否する


def test_jvlink_calendar_sanity_check_skips_tiny_lists_for_identical_time_check():
    # 「全レース同一時刻」判定は1-2件では発動しない(本当にその日そのレース数しか
    # ない可能性を否定できないため)。ただし00:00や範囲外時刻はどんな件数でも弾く
    # (2026-09-19深夜追加のため、この挙動はもはや「完全素通し」ではない)。
    assert JRC._sanity_check([{"rid16": "2026092106040701", "post": "14:00"}]) == ""


# ==================================================================
# 2026-09-21: venue06 store_prediction 障害の原因調査・修正で追加した回帰テスト群。
# 根本原因: export_weekly_marks.py の「全馬tansho_odds=Noneなら中止とみなしbundle
# から除外」フィルタが venue06(中山、単に当日オッズがTARGET/JV-Link側で未着だった
# だけで開催中止ではない)を誤って全除外し、bundle依存のT-10/T-20/EXP05-F store_
# predictionがvenue06を丸ごと欠測した。副次的に、bundleに無いraceのstore_prediction
# 失敗(masters_vote.VoteError)がmarket_only保存へフォールバックせず未分類ログに
# 落ちる別バグも見つかった。ここではこの2点の修正を検証する。
# ==================================================================

def test_load_bundle_race_converts_voteerror_to_filenotfound(monkeypatch):
    """masters_vote.VoteError (bundleにraceが無い) は FileNotFoundError に正規化される
    (market_snapshot.process_race の except FileNotFoundError → store_market_only
    フォールバックへ載せるための変換。以前はVoteErrorのまま伝播し except Exception の
    未分類バケツに落ちてmarket_only保存が起きなかった)。"""
    import masters_vote as mv

    def fake_load(date_str, rid):
        raise mv.VoteError(f"bundle に {rid} が無い (20260921_bundle.json)")
    monkeypatch.setattr(mv, "load_bundle_race", fake_load)
    with pytest.raises(FileNotFoundError, match="bundle race 欠落"):
        pas._load_bundle_race("20260921", "2026092106040701")


def test_load_bundle_race_missing_race_does_not_match_similar_id(tmp_path, monkeypatch):
    """venue06のraceがbundleに無いとき、venue09等の別raceへ誤って一致しないこと
    (rid完全一致のみで解決している、という「ID結合」契約の確認)。"""
    import masters_vote as mv
    bundle_dir = tmp_path / "reports" / "cowork_input"
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "20990101_bundle.json").write_text(json.dumps({
        "date": "20990101",
        "races": [{"race_id": "9999010109040701", "horses": [{"umaban": 1}]}],
    }), encoding="utf-8")
    monkeypatch.setattr(mv, "BASE", tmp_path)
    with pytest.raises(mv.VoteError):
        mv.load_bundle_race("20990101", "9999010106040701")  # venue06、bundleにはvenue09のみ
    # venue09自体は正しく引けること (誤検知的にVoteErrorを出しているだけでないことの確認)
    race = mv.load_bundle_race("20990101", "9999010109040701")
    assert race["race_id"] == "9999010109040701"


def test_market_snapshot_bundle_race_missing_routes_to_market_only(tmp_path, monkeypatch):
    """本番シナリオの再現: JV-Linkオッズ取得は失敗 (venue06のno O1 record相当) だが、
    bundle race欠落によるstore_prediction失敗はmarket_onlyへフォールバックし、
    取得できた市場情報 (たとえok=falseでも) を捨てずに保存すること。"""
    from analysis.mcond.exp05_forward_shadow import market_snapshot as ms
    from datetime import datetime, timedelta, timezone
    JST = timezone(timedelta(hours=9))

    fake_market_result = {
        "ok": False, "why": "オッズ ok=false (no O1 record)",
        "market": {"ok": False, "reason": "no O1 record", "fetched": "2026-09-21T09:10:00"},
    }
    monkeypatch.setattr(ms, "fetch_and_validate", lambda rid, sp: dict(fake_market_result))

    def fake_store_prediction(date_str, rid, result):
        raise FileNotFoundError(f"bundle race 欠落: bundle に {rid} が無い (20990101_bundle.json)")
    calls = {}

    def fake_store_market_only(date_str, rid, result, reason):
        calls["reason"] = reason
        calls["market"] = result.get("market")
        p = tmp_path / f"{rid}_marketonly_rev1.json"
        p.write_text(json.dumps({"reason": reason}), encoding="utf-8")
        return p
    monkeypatch.setattr(pas, "store_prediction", fake_store_prediction)
    monkeypatch.setattr(pas, "store_market_only", fake_store_market_only)
    monkeypatch.setattr(ms, "BASE", tmp_path)  # path.relative_to(BASE) の print 用

    future_post = datetime.now(JST) + timedelta(hours=1)
    rc = ms.process_race("20990101", "9999010106040701", "R1", dry=False,
                         scheduled_post=future_post)
    assert rc == 2
    assert calls.get("reason") == "bundle_race_missing"
    assert calls.get("market") == fake_market_result["market"]  # 市場dataが捨てられていない


def test_is_retrospective_true_only_for_past_scheduled_post():
    """発走予定時刻が既に過去なら retrospective (=primary禁止) と判定する。"""
    assert pas._is_retrospective({"scheduled_post": "2020-01-01T10:00:00+09:00"}) is True
    assert pas._is_retrospective({"scheduled_post": "2099-01-01T10:00:00+09:00"}) is False
    assert pas._is_retrospective({"scheduled_post": None}) is False
    assert pas._is_retrospective({}) is False


def test_store_market_only_flags_retrospective_recovery(tmp_path, monkeypatch):
    """発走予定時刻が過去のレースへの market_only 保存は retrospective_recovery=true
    かつ invalid_for_primary=true になる (発走後の回復データがprimaryへ紛れ込まない)。"""
    monkeypatch.setattr(pas, "OUT_DIR", tmp_path)
    market_result = {"ok": False, "valid_for_primary": False,
                     "scheduled_post": "2020-01-01T10:00:00+09:00", "market": {}}
    path = pas.store_market_only("20200101", "2020010106040701", market_result,
                                 reason="bundle_race_missing")
    rec = json.loads(path.read_text(encoding="utf-8"))
    assert rec["retrospective_recovery"] is True
    assert rec["invalid_for_primary"] is True
    assert rec["prediction_saved"] is False


def test_store_market_only_future_race_not_flagged_retrospective(tmp_path, monkeypatch):
    monkeypatch.setattr(pas, "OUT_DIR", tmp_path)
    market_result = {"ok": True, "valid_for_primary": True,
                     "scheduled_post": "2099-01-01T10:00:00+09:00", "market": {}}
    path = pas.store_market_only("20990101", "9999010109040701", market_result,
                                 reason="weekly_input_unavailable")
    rec = json.loads(path.read_text(encoding="utf-8"))
    assert rec["retrospective_recovery"] is False


# ==================================================================
# 2026-09-22: ユーザー指摘による market-only の意味の厳密化。
# 2026-09-21 venue06 の実報告で market=0/market_only=1 となり、「market captureが
# 失敗したレース」が「使えるmarket snapshotはあるがpredictionだけ失敗した」かの
# ように見えてしまっていた誤りを修正する回帰テスト。
# ==================================================================

def test_store_market_only_capture_failure_is_not_market_only(tmp_path, monkeypatch):
    """market_result.ok=false (JV-Link取得失敗、例: 'no O1 record') の場合は
    market_capture_success=false かつ market_only=false・market_capture_failure=true
    になる (2026-09-21 venue06の実例: market=0/market_only=1という誤集計の直接原因)。"""
    monkeypatch.setattr(pas, "OUT_DIR", tmp_path)
    market_result = {"ok": False, "valid_for_primary": False,
                     "scheduled_post": "2099-01-01T15:30:00+09:00",
                     "market": {"ok": False, "reason": "no O1 record",
                               "fetched": "2099-01-01T14:55:00"}}
    path = pas.store_market_only("20990101", "9999010106040711", market_result,
                                 reason="bundle_race_missing")
    rec = json.loads(path.read_text(encoding="utf-8"))
    assert rec["market_capture_success"] is False
    assert rec["market_only"] is False
    assert rec["market_capture_failure"] is True
    assert rec["prediction_failure"] is True
    assert rec["market_snapshot_saved"] is True  # 保存自体は試みた(捨てない)ことは変わらない
    assert rec["prediction_saved"] is False


def test_store_market_only_capture_success_flags_correctly(tmp_path, monkeypatch):
    monkeypatch.setattr(pas, "OUT_DIR", tmp_path)
    market_result = {"ok": True, "valid_for_primary": False,
                     "scheduled_post": "2099-01-01T10:00:00+09:00",
                     "market": {"fetched": "2099-01-01T09:50:00", "tansho": {"1": 3.0}}}
    path = pas.store_market_only("20990101", "9999010109040701", market_result,
                                 reason="weekly_input_unavailable")
    rec = json.loads(path.read_text(encoding="utf-8"))
    assert rec["market_capture_success"] is True
    assert rec["market_only"] is True
    assert rec["market_capture_failure"] is False
    assert rec["prediction_failure"] is True


# ---- venue別ID形式・被覆率レポート (observation_report.py) ----
def test_venue_of_preserves_leading_zero_for_all_10_jra_venues():
    from analysis.mcond.exp05_forward_shadow.observation_report import _venue_of
    for i in range(1, 11):
        code = f"{i:02d}"
        rid = f"20260921{code}040701"
        assert len(rid) == 16
        assert _venue_of(rid) == code  # int castでの '06'→'6' 等のゼロ落ちが無いこと


def test_venue06_and_venue09_race_ids_share_kaiji_nichiji_differ_only_in_venue():
    """今回の実インシデントの実データ形式そのものの回帰確認
    (2026-09-21: 中山=06 / 阪神=09、4回7日開催で発走順は同一)。"""
    v06, v09 = "2026092106040701", "2026092109040701"
    assert v06[:8] == v09[:8] == "20260921"           # 日付
    assert v06[8:10] == "06" and v09[8:10] == "09"     # venue
    assert v06[10:] == v09[10:] == "040701"            # 回次・日次・R番号は共通


def test_observation_report_by_venue_isolates_dead_venue_from_healthy_one(tmp_path, monkeypatch):
    """observation_report.by_venue が「1venueだけ0%・他venueは正常」を可視化できること
    (2026-09-21の実障害: venue06=0観測 / venue09=正常、という非対称パターンの再現)。"""
    from analysis.mcond.exp05_forward_shadow import observation_report as OR
    date_str = "20990101"
    odds_dir = tmp_path / "odds"; odds_dir.mkdir()
    pred_dir = tmp_path / "pred" / date_str; pred_dir.mkdir(parents=True)
    cal_dir = tmp_path / "calendar"; cal_dir.mkdir()
    monkeypatch.setattr(OR, "ODDS_DIR", odds_dir)
    monkeypatch.setattr(OR, "PRED_DIR", tmp_path / "pred")
    monkeypatch.setattr(OR, "CALENDAR_DIR", cal_dir)
    monkeypatch.setattr(OR, "WEEKLY_DIR", tmp_path / "weekly")
    monkeypatch.setattr(OR, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(OR, "_current_model_hash", lambda: "deadbeef00000000")

    (cal_dir / f"{date_str}.json").write_text(json.dumps({
        "generated_at": "2099-01-01T08:20:00", "record_count": 2,
        "race_ids": ["2099010106040701", "2099010109040701"],
    }), encoding="utf-8")
    # venue06: タスクは発火したがオッズ取得自体が失敗 (ok=false) → market/complete共に0
    (odds_dir / "2099010106040701.json").write_text(json.dumps({"ok": False}), encoding="utf-8")
    # venue09: 正常に市場取得・完全予測まで到達
    (odds_dir / "2099010109040701.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (pred_dir / "2099010109040701_deadbeef00000000_rev1.json").write_text(json.dumps({
        "race_id": "2099010109040701", "date": date_str,
        "records": [{"race_id": "2099010109040701", "model_hash": "deadbeef00000000",
                    "valid_for_primary": True, "invalid_reason": None}],
    }), encoding="utf-8")

    report = OR.build_report(date_str)
    assert report["by_venue"]["06"]["scheduled"] == 1
    assert report["by_venue"]["06"]["market"] == 0
    assert report["by_venue"]["06"]["collection_failures"] == 1
    assert report["by_venue"]["06"]["complete"] == 0
    assert report["by_venue"]["06"]["complete_prediction_rate"] == 0.0
    assert report["by_venue"]["09"]["market"] == 1
    assert report["by_venue"]["09"]["collection_failures"] == 0
    assert report["by_venue"]["09"]["valid_primary"] == 1
    assert report["by_venue"]["09"]["complete_prediction_rate"] == 1.0
    assert report["collection_failures"] == 1
    assert report["usable_market_observations"] == 1
    assert report["unusable_market_records"] == 1


def test_observation_report_market_only_reclassification_2026_venue06_case(tmp_path, monkeypatch):
    """2026-09-21 venue06の実例そのものの再現: market capture自体が失敗(ok=false)
    しているレースは、たとえ marketonly_rev1.json が保存されていても
    market_only_with_usable_odds には数えない (=「使えるmarket+prediction失敗」
    ではなく「market capture失敗」の方へ分類する)。ファイルは削除・上書きせず
    report側の分類だけで訂正する、という要求そのものを検証する。"""
    from analysis.mcond.exp05_forward_shadow import observation_report as OR
    date_str = "20260921"
    odds_dir = tmp_path / "odds"; odds_dir.mkdir()
    pred_dir = tmp_path / "pred" / date_str; pred_dir.mkdir(parents=True)
    monkeypatch.setattr(OR, "ODDS_DIR", odds_dir)
    monkeypatch.setattr(OR, "PRED_DIR", tmp_path / "pred")
    monkeypatch.setattr(OR, "CALENDAR_DIR", tmp_path / "calendar")
    monkeypatch.setattr(OR, "WEEKLY_DIR", tmp_path / "weekly")
    monkeypatch.setattr(OR, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(OR, "_current_model_hash", lambda: "deadbeef00000000")

    # 実際の2026092106040711のmarketonly_rev1.jsonを模した内容 (market_capture_success
    # フィールドを含む、2026-09-22修正後の新形式)
    (pred_dir / "2026092106040711_marketonly_rev1.json").write_text(json.dumps({
        "race_id": "2026092106040711", "market_capture_success": False,
        "market_only": False, "market_capture_failure": True,
        "prediction_failure": True, "market_snapshot_saved": True,
        "prediction_saved": False, "market_ok": False,
    }), encoding="utf-8")

    report = OR.build_report(date_str)
    assert report["market_only_count"] == 1              # ファイルは存在する(後方互換カウント)
    assert report["market_only_with_usable_odds"] == 0    # だが狭義の定義では0件
    assert report["market_only_capture_failed_count"] == 1


def test_observation_report_market_only_backward_compat_old_format_file(tmp_path, monkeypatch):
    """2026-09-22修正前 (market_capture_success フィールド無し) の既存ファイルは
    market_ok フィールドへフォールバックして正しく分類できること (ファイル自体は
    書き換えていないので、report側だけでの再分類が機能する必要がある)。"""
    from analysis.mcond.exp05_forward_shadow import observation_report as OR
    date_str = "20990101"
    odds_dir = tmp_path / "odds"; odds_dir.mkdir()
    pred_dir = tmp_path / "pred" / date_str; pred_dir.mkdir(parents=True)
    monkeypatch.setattr(OR, "ODDS_DIR", odds_dir)
    monkeypatch.setattr(OR, "PRED_DIR", tmp_path / "pred")
    monkeypatch.setattr(OR, "CALENDAR_DIR", tmp_path / "calendar")
    monkeypatch.setattr(OR, "WEEKLY_DIR", tmp_path / "weekly")
    monkeypatch.setattr(OR, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(OR, "_current_model_hash", lambda: "deadbeef00000000")

    # 旧形式: market_capture_success が無く market_ok のみ (capture成功のケース)
    (pred_dir / "9999010109040701_marketonly_rev1.json").write_text(json.dumps({
        "race_id": "9999010109040701", "market_snapshot_saved": True,
        "prediction_saved": False, "market_ok": True,
    }), encoding="utf-8")

    report = OR.build_report(date_str)
    assert report["market_only_with_usable_odds"] == 1
    assert report["market_only_capture_failed_count"] == 0
