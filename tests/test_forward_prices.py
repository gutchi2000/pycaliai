from pathlib import Path

import pytest

from forward_prices import (
    archive_decision_record,
    archive_market_snapshot,
    build_decision_record,
    fair_win_probabilities,
    read_snapshot,
)
import forward_prices as fp


RID = "2026082905010101"
STAMP = {"policy_id": "test-policy", "policy_sha256": "a" * 64}


def _market():
    return {
        "race_id": RID,
        "fetched": "2026-08-29T10:00:00.123",
        "tansho": {"1": 2.0, "2": 4.0, "3": 8.0},
        "fukusho": {"1": [1.2, 1.4], "2": [1.8, 2.1]},
        "wide": {"1-2": [2.5, 2.9]},
        "overround_tan": 0.875,
    }


def _race():
    return {
        "race_id": RID,
        "horses": [
            {"umaban": 1, "mark": "◎", "p_win": 0.60, "p_sho": 0.85},
            {"umaban": 2, "mark": "〇", "p_win": 0.25, "p_sho": 0.60},
            {"umaban": 3, "mark": "▲", "p_win": 0.15, "p_sho": 0.40},
        ],
    }


def test_market_snapshots_are_immutable_and_readable(tmp_path: Path):
    market = _market()
    first = archive_market_snapshot(market, "t10", stamp=STAMP, root=tmp_path)
    second = archive_market_snapshot(market, "t10", stamp=STAMP, root=tmp_path)
    assert first == second
    assert len(list(tmp_path.rglob("*.json.gz"))) == 1
    saved = read_snapshot(first)
    assert saved["stage"] == "t10"
    assert saved["market"] == market
    assert saved["policy"]["policy_id"] == "test-policy"


def test_market_probability_is_devigged():
    fair = fair_win_probabilities(_market()["tansho"])
    assert sum(fair.values()) == pytest.approx(1.0)
    assert fair[1] > fair[2] > fair[3]


def test_decision_contains_market_residual_and_engine_pair(tmp_path: Path):
    primary = {"race_id": RID, "race_nature": "topdown", "bets": []}
    shadow = {"race_id": RID, "race_nature": "本命勝負", "bets": []}
    record = build_decision_record(
        _race(), _market(), primary, shadow, mode="default",
        residual_shadow={"race_id": RID, "triggered": True},
        model_umaren={(1, 2): 0.30}, model_wide={(1, 2): 0.55}, stamp=STAMP)
    path = archive_decision_record(record, root=tmp_path)
    saved = read_snapshot(path)
    hon = next(x for x in saved["horses"] if x["umaban"] == 1)
    assert hon["win_market_residual"] == pytest.approx(
        hon["p_win_model"] - hon["p_win_market_devig"])
    assert saved["primary"] == primary
    assert saved["shadow"] == shadow
    assert saved["wide_residual_shadow"]["triggered"] is True
    assert saved["pairs"][0]["p_wide_model"] == pytest.approx(0.55)


# ---------------------------------------------------------------- 2026-09-21:
# EXP05-F venue06 保存障害の原因調査で追加した回帰テスト群。
def test_archive_market_snapshot_rejects_collision_with_different_content(tmp_path: Path):
    """同一 race_id/stage/観測秒に内容の異なる2回目の書込が来たら上書きせず例外
    (append-onlyの破壊防止。同一内容の再実行は idempotent に許容する既存挙動と対)。"""
    market1 = _market()
    market2 = dict(_market(), overround_tan=1.10)  # 内容だけ変える
    fixed_time = "2026-08-29T10:00:00.123"
    market1["fetched"] = fixed_time
    market2["fetched"] = fixed_time
    first = archive_market_snapshot(market1, "t10", stamp=STAMP, root=tmp_path)
    with pytest.raises(FileExistsError):
        fp._write_gzip_atomic(first, {"market": market2, "record_sha256": "different"})


@pytest.mark.parametrize("venue", [f"{i:02d}" for i in range(1, 11)])
def test_archive_market_snapshot_preserves_all_10_jra_venue_codes(tmp_path: Path, venue: str):
    """JRA 全10場コード (01=札幌...10=小倉) がゼロ埋め2桁のまま race_id に残ること
    (int cast等でのゼロ落ち (例 '06'→'6') が起きていないことの確認)。"""
    rid = f"20260921{venue}040701"
    assert len(rid) == 16
    market = dict(_market(), race_id=rid, fetched=f"2026-09-21T09:10:00.{venue}00")
    path = archive_market_snapshot(market, "exp05fs_t35", root=tmp_path)
    saved = read_snapshot(path)
    assert saved["race_id"] == rid
    assert saved["race_id"][8:10] == venue


def test_archive_market_snapshot_rejects_malformed_race_id_length():
    """rid16が16桁に届かない (venue桁欠落等) 場合は保存前に拒否する。"""
    with pytest.raises(ValueError):
        archive_market_snapshot({"race_id": "202609216040701"}, "t10")  # 15桁(venue1桁欠け相当)


def test_archive_market_snapshot_warns_but_still_saves_on_timing_anomaly(tmp_path: Path, capsys):
    """発走予定から大きく外れたタイミングでの保存 (2026-09-21発見の-5,228分異常の
    ようなケース) は stderr へ警告するが、保存自体は失敗させない (収集失敗として
    レースを丸ごと落とすより、警告付きで保存し後段の timing canary で隔離する方が
    データを失わない)。"""
    market = dict(_market(), race_id="2026090601020601",
                 fetched="2026-09-10T00:58:35.466")
    path = archive_market_snapshot(
        market, "t10", scheduled_post="2026-09-06T09:50:00", root=tmp_path)
    assert path.exists()
    err = capsys.readouterr().err
    assert "WARN" in err
    assert "2026090601020601" in err


def test_archive_market_snapshot_no_warning_for_normal_timing(tmp_path: Path, capsys):
    market = dict(_market(), race_id="2026090109040701",
                 fetched="2026-09-01T09:40:00.100")
    archive_market_snapshot(
        market, "t10", scheduled_post="2026-09-01T09:50:00", root=tmp_path)
    err = capsys.readouterr().err
    assert "WARN" not in err
