# -*- coding: utf-8 -*-
"""
test_jump_race_p0_gate.py — 障害レース除外 P0 hard gate の回帰テスト
====================================================================
実際に本番 bundle へ混入し v6 に採点・印付けされていた 3 レースを
fixture として固定し、全購入経路で拒否されることを検査する。
通常の芝・ダートレースが従来どおり通る positive control も必須。

実行:
  venv311/Scripts/python.exe -m pytest tests/test_jump_race_p0_gate.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from race_eligibility import (  # noqa: E402
    evaluate_race, assert_bettable, filter_race_ids,
    JumpRaceBettingError, JUMP_TRACK_MIN, JUMP_TRACK_MAX,
)

# 本番 bundle へ実際に混入していた 3 レース (2026-09-22 実測)
KNOWN_JUMP_RIDS = [
    "2026091306040401",   # 20260913 中山 R01 障害
    "2026091909040504",   # 20260919 阪神 R04 障害
    "2026092009040601",   # 20260920 中山 R01 障害
]

# positive control: 同じ開催日の通常平地レース
FLAT_RIDS = [
    "2026091306040402",
    "2026091306040403",
    "2026091909040505",
    "2026092009040602",
]

SAMPLE_BETS = [{"馬券種": "複勝", "買い目": "3", "購入額": 1000}]


# ---------------- 判定 ----------------

@pytest.mark.parametrize("rid", KNOWN_JUMP_RIDS)
def test_known_jump_is_detected(rid):
    el = evaluate_race(rid)
    assert el["is_jump"] is True
    assert el["determination"] == "authoritative"
    assert el["reason"] == "jump_race_excluded"
    tc = el["raw_fields"]["track_code"]
    assert tc is not None and JUMP_TRACK_MIN <= tc <= JUMP_TRACK_MAX


@pytest.mark.parametrize("rid", KNOWN_JUMP_RIDS)
def test_known_jump_not_prediction_eligible(rid):
    el = evaluate_race(rid)
    assert el["prediction_eligible"] is False


@pytest.mark.parametrize("rid", KNOWN_JUMP_RIDS)
def test_known_jump_not_task_registration_eligible(rid):
    assert evaluate_race(rid)["task_registration_eligible"] is False


@pytest.mark.parametrize("rid", KNOWN_JUMP_RIDS)
def test_known_jump_history_only_eligible(rid):
    """完全に捨てず、history-only には保存可能であること。"""
    assert evaluate_race(rid)["history_only_eligible"] is True


@pytest.mark.parametrize("rid", KNOWN_JUMP_RIDS)
def test_known_jump_has_raw_fields(rid):
    raw = evaluate_race(rid)["raw_fields"]
    for k in ("race_id", "date", "track_code", "hira_shogai",
              "in_jump_history_store"):
        assert k in raw


# ---------------- 層1: bundle / race list ----------------

def test_layer1_filter_drops_jump_keeps_flat():
    keep, dropped = filter_race_ids(KNOWN_JUMP_RIDS + FLAT_RIDS,
                                    layer="test_bundle")
    assert set(keep) == set(FLAT_RIDS)
    assert {d["race_id"] for d in dropped} == set(KNOWN_JUMP_RIDS)


# ---------------- 層2: task 登録 ----------------

def test_layer2_build_schedule_excludes_jump():
    from t10_runner import build_schedule
    races = [{"race_id": r, "race_meta": {"place": "X"}}
             for r in KNOWN_JUMP_RIDS + FLAT_RIDS]
    sched, _missing = build_schedule("20260913", races, lead_min=10)
    sched_rids = {rid for _dt, rid, _lab in sched}
    assert not (sched_rids & set(KNOWN_JUMP_RIDS)), \
        "障害レースがタスク登録スケジュールに残っている"


# ---------------- 層3: compute_bets ----------------

@pytest.mark.parametrize("rid", KNOWN_JUMP_RIDS)
def test_layer3_compute_bets_returns_empty(rid):
    from compute_bets import compute_race_bets
    race = {
        "race_id": rid,
        "race_meta": {"race_id": rid, "place": "中山", "field_size": 12},
        "race_confidence": {"field_chaos_score": 0.1},
        "horses": [{"umaban": i, "mark": "◎" if i == 1 else "",
                    "p_win": 0.2, "tansho_odds": 3.0} for i in range(1, 13)],
    }
    out = compute_race_bets(race)
    assert out["bets"] == [], "障害レースに買い目が出ている"
    assert out.get("is_jump") is True
    assert out.get("excluded_reason") == "jump_race_excluded"


def test_layer3_flat_race_still_processed():
    """positive control: 平地レースは従来どおり compute_bets を通る
    (買い目が出るかは市況次第なので、'障害として弾かれない' ことを見る)。"""
    from compute_bets import compute_race_bets
    rid = FLAT_RIDS[0]
    race = {
        "race_id": rid,
        "race_meta": {"race_id": rid, "place": "中山", "field_size": 12},
        "race_confidence": {"field_chaos_score": 0.1},
        "horses": [{"umaban": i, "mark": "◎" if i == 1 else "",
                    "p_win": 0.2, "tansho_odds": 3.0} for i in range(1, 13)],
    }
    out = compute_race_bets(race)
    assert out.get("is_jump") is not True
    assert out.get("excluded_reason") != "jump_race_excluded"


# ---------------- 層4: validate_cowork_bets ----------------

@pytest.mark.parametrize("rid", KNOWN_JUMP_RIDS)
def test_layer4_validate_rejects_nonempty_bets(rid, tmp_path, monkeypatch):
    import validate_cowork_bets as v

    date = rid[:8]
    bets = [{"race_id": rid, "race_label": "障害", "bets": SAMPLE_BETS}]
    bundle = {"races": [{"race_id": rid, "race_meta": {"field_size": 12},
                         "race_confidence": {"field_chaos_score": 0.1},
                         "horses": [{"umaban": 3, "mark": "◎",
                                     "p_win": 0.3, "tansho_odds": 3.0}]}]}
    bp = tmp_path / f"{date}_bets.json"
    bup = tmp_path / f"{date}_bundle.json"
    bp.write_text(json.dumps(bets, ensure_ascii=False), encoding="utf-8")
    bup.write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr(sys, "argv",
                        ["validate_cowork_bets.py", "--bets", str(bp),
                         "--bundle", str(bup), "--date", date])
    rc = v.main()
    assert rc != 0, "障害レースの買い目が違反として検出されていない"


@pytest.mark.parametrize("rid", KNOWN_JUMP_RIDS)
def test_layer4_validate_apply_forces_empty(rid, tmp_path, monkeypatch):
    import validate_cowork_bets as v

    date = rid[:8]
    bets = [{"race_id": rid, "race_label": "障害", "bets": SAMPLE_BETS}]
    bundle = {"races": [{"race_id": rid, "race_meta": {"field_size": 12},
                         "race_confidence": {"field_chaos_score": 0.1},
                         "horses": [{"umaban": 3, "mark": "◎",
                                     "p_win": 0.3, "tansho_odds": 3.0}]}]}
    bp = tmp_path / f"{date}_bets.json"
    bup = tmp_path / f"{date}_bundle.json"
    bp.write_text(json.dumps(bets, ensure_ascii=False), encoding="utf-8")
    bup.write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr(sys, "argv",
                        ["validate_cowork_bets.py", "--bets", str(bp),
                         "--bundle", str(bup), "--date", date, "--apply"])
    v.main()
    after = json.loads(bp.read_text(encoding="utf-8"))
    races = after["bets"] if isinstance(after, dict) else after
    assert races[0]["bets"] == [], "--apply で障害レースの買い目が消えていない"


# ---------------- 層5: 購入・送信直前 ----------------

@pytest.mark.parametrize("rid", KNOWN_JUMP_RIDS)
def test_layer5_assert_bettable_raises(rid):
    with pytest.raises(JumpRaceBettingError):
        assert_bettable(rid, SAMPLE_BETS, layer="test_final")


@pytest.mark.parametrize("rid", KNOWN_JUMP_RIDS)
def test_layer5_empty_bets_do_not_raise(rid):
    assert_bettable(rid, [], layer="test_final")
    assert_bettable(rid, None, layer="test_final")


@pytest.mark.parametrize("rid", FLAT_RIDS)
def test_layer5_flat_race_passes(rid):
    """positive control: 平地は買い目があっても通る。"""
    assert_bettable(rid, SAMPLE_BETS, layer="test_final")


@pytest.mark.parametrize("rid", KNOWN_JUMP_RIDS)
def test_layer5_masters_vote_submit_refuses(rid, monkeypatch):
    """実送信関数 submit() が payload 段階で拒否すること。"""
    import masters_vote as mv
    payload = {"race_id": rid, "bet_data": [{"kind": "fuku", "umaban": 3}]}
    with pytest.raises(JumpRaceBettingError):
        mv.submit(payload, {"check_delay_sec": 1})


def test_layer5_masters_vote_submit_allows_flat(monkeypatch):
    """positive control: 平地は gate を通過して本処理へ進む
    (netkeiba_api の import で止まる = gate は通った)。"""
    import masters_vote as mv
    payload = {"race_id": FLAT_RIDS[0],
               "bet_data": [{"kind": "fuku", "umaban": 3}]}
    try:
        mv.submit(payload, {"check_delay_sec": 1})
    except JumpRaceBettingError:
        pytest.fail("平地レースが障害として拒否された")
    except Exception:
        pass   # ログイン等で失敗するのは想定内 (gate は通過している)


# ---------------- positive control: 平地 ----------------

@pytest.mark.parametrize("rid", FLAT_RIDS)
def test_flat_race_fully_eligible(rid):
    el = evaluate_race(rid)
    assert el["is_jump"] is False
    assert el["prediction_eligible"] is True
    assert el["bet_eligible"] is True
    assert el["task_registration_eligible"] is True
    assert el["history_only_eligible"] is False
    assert el["reason"] == "flat_race_ok"


# ---------------- 判定ロジックの性質 ----------------

def test_conflict_is_fail_closed(monkeypatch):
    """トラックコードと平・障が食い違ったら障害扱い (fail-closed)。"""
    import race_eligibility as re_mod
    re_mod.clear_cache()
    monkeypatch.setattr(re_mod, "_bunseki_track_codes",
                        lambda d: {"2026091306040402": 23})   # 平地
    monkeypatch.setattr(re_mod, "_bias_hira_shogai",
                        lambda d: {"2026091306040402": "1"})  # 障害
    monkeypatch.setattr(re_mod, "_collected_jump_rids", lambda d: frozenset())
    el = re_mod.evaluate_race("2026091306040402")
    assert el["is_jump"] is True
    assert el["reason"] == "jump_detection_conflict"
    assert el["bet_eligible"] is False
    re_mod.clear_cache()


def test_unknown_determination_is_recorded(monkeypatch):
    """判定材料が無い場合は unknown として記録される (通すが黙らない)。"""
    import race_eligibility as re_mod
    re_mod.clear_cache()
    monkeypatch.setattr(re_mod, "_bunseki_track_codes", lambda d: {})
    monkeypatch.setattr(re_mod, "_bias_hira_shogai", lambda d: {})
    monkeypatch.setattr(re_mod, "_collected_jump_rids", lambda d: frozenset())
    el = re_mod.evaluate_race("2099010106040401")
    assert el["determination"] == "unknown"
    assert el["reason"] == "jump_undetermined"
    re_mod.clear_cache()


def test_explicit_track_code_argument_wins():
    el = evaluate_race("2026091306040402", track_code=52)
    assert el["is_jump"] is True
    assert el["raw_fields"]["track_code_source"] == "argument"


if __name__ == "__main__":
    sys.exit(pytest.main([str(Path(__file__)), "-q"]))
