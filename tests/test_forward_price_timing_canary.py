"""
test_forward_price_timing_canary.py
=====================================
2026-09-21、T-10の-5,228分異常調査で追加した回帰テスト。
実行: python -m pytest tests/test_forward_price_timing_canary.py -q
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from analysis.forward_price_timing_canary import (
    classify_timing,
    minutes_to_post,
    scan,
    load_timing_valid_records,
    require_timing_gate,
)


def test_minutes_to_post_normal_t10():
    m = minutes_to_post("2026-09-06T09:50:00", "2026-09-06T09:40:01.247")
    assert m == pytest.approx(9.979, abs=0.01)


def test_minutes_to_post_the_actual_5228_anomaly():
    """2026-09-21発見の実インシデントそのものの再現。"""
    m = minutes_to_post("2026-09-06T09:50:00", "2026-09-10T00:58:35.466")
    assert m == pytest.approx(-5228.59, abs=0.1)


def test_minutes_to_post_missing_fields_returns_none():
    assert minutes_to_post(None, "2026-09-06T09:40:00") is None
    assert minutes_to_post("2026-09-06T09:50:00", None) is None


def test_classify_timing_t10_normal_is_valid():
    valid, reason, m = classify_timing("t10", "2026-09-06T09:50:00", "2026-09-06T09:40:01.247")
    assert valid is True
    assert reason == ""


def test_classify_timing_t10_5228_anomaly_is_invalid():
    valid, reason, m = classify_timing("t10", "2026-09-06T09:50:00", "2026-09-10T00:58:35.466")
    assert valid is False
    assert "ウィンドウ外" in reason
    assert m == pytest.approx(-5228.59, abs=0.1)


def test_classify_timing_close_normal_is_valid():
    valid, reason, m = classify_timing("close", "2026-09-06T09:50:00", "2026-09-06T09:51:00.5")
    assert valid is True


def test_classify_timing_close_before_post_is_invalid():
    """closeは発走後+60秒が仕様。発走"前"に取得されたcloseは異常。"""
    valid, reason, m = classify_timing("close", "2026-09-06T09:50:00", "2026-09-06T09:30:00")
    assert valid is False


def test_classify_timing_missing_scheduled_post_fails_closed():
    """scheduled_postが無いレコードは「有効」ではなく「判定不能」として無効扱い
    (不明を安全側=無効に倒す。T-20の現状の構造的制約 (scheduled_post未埋込) は
    こうして正しく「除外」される)。"""
    valid, reason, m = classify_timing("t20", None, "2026-09-06T09:40:00")
    assert valid is False
    assert "欠落" in reason


def test_classify_timing_unknown_stage_fails_closed():
    valid, reason, m = classify_timing("unknown_stage", "2026-09-06T09:50:00",
                                       "2026-09-06T09:40:00")
    assert valid is False
    assert "未知のstage" in reason


def _write_snapshot(root: Path, rid: str, stage: str, scheduled_post, observed_at,
                    subdir: str | None = None) -> Path:
    d = subdir or rid[:8]
    p = root / d / f"{rid}_{stage}_test.json.gz"
    p.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "record_type": "market_snapshot", "stage": stage, "race_id": rid,
        "observed_at": observed_at, "scheduled_post": scheduled_post, "market": {},
    }
    with gzip.open(p, "wt", encoding="utf-8") as f:
        json.dump(body, f)
    return p


def test_scan_quarantines_anomaly_without_touching_originals(tmp_path: Path):
    """異常レコードを検出してレポートに載せるが、元ファイルの中身は一切変更しない
    (削除・上書き禁止の確認)。"""
    normal = _write_snapshot(tmp_path, "2026090109040701", "t10",
                             "2026-09-01T09:50:00", "2026-09-01T09:40:00.100")
    anomaly = _write_snapshot(tmp_path, "2026090601020601", "t10",
                              "2026-09-06T09:50:00", "2026-09-10T00:58:35.466")
    before_normal = normal.read_bytes()
    before_anomaly = anomaly.read_bytes()

    report = scan(root=tmp_path)

    assert normal.read_bytes() == before_normal   # 元ファイル不変
    assert anomaly.read_bytes() == before_anomaly  # 元ファイル不変 (異常でも削除・上書きしない)
    assert report["anomaly_count"] == 1
    assert report["anomalies"][0]["race_id"] == "2026090601020601"
    assert report["by_stage"]["t10"]["n"] == 2
    assert report["by_stage"]["t10"]["timing_valid"] == 1
    assert report["by_stage"]["t10"]["timing_invalid"] == 1


def test_scan_detects_duplicate_race_stage_pairs(tmp_path: Path):
    _write_snapshot(tmp_path, "2026090601020601", "t10",
                    "2026-09-06T09:50:00", "2026-09-06T09:40:00.100")
    p2 = tmp_path / "20260906" / "2026090601020601_t10_test2.json.gz"
    p2.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p2, "wt", encoding="utf-8") as f:
        json.dump({"record_type": "market_snapshot", "stage": "t10",
                   "race_id": "2026090601020601", "observed_at": "2026-09-10T00:58:35.466",
                   "scheduled_post": "2026-09-06T09:50:00", "market": {}}, f)
    report = scan(root=tmp_path)
    assert report["duplicate_race_stage_count"] == 1
    assert report["duplicates"][0]["race_id"] == "2026090601020601"
    assert len(report["duplicates"][0]["files"]) == 2


def test_load_timing_valid_records_excludes_out_of_window(tmp_path: Path):
    """将来の価格形成モデル学習データ取り込み口が、window外レコードを
    確実に除外すること。"""
    _write_snapshot(tmp_path, "2026090109040701", "t10",
                    "2026-09-01T09:50:00", "2026-09-01T09:40:00.100")
    _write_snapshot(tmp_path, "2026090601020601", "t10",
                    "2026-09-06T09:50:00", "2026-09-10T00:58:35.466")
    records = load_timing_valid_records("t10", root=tmp_path)
    assert len(records) == 1
    assert records[0]["race_id"] == "2026090109040701"


def test_scan_does_not_flag_decision_records(tmp_path: Path):
    """record_type=decision_snapshot (build_decision_record由来) はmarket_snapshotの
    timing判定対象外 (scheduled_postを持たない別スキーマのため誤検知しない)。"""
    p = tmp_path / "20260906" / "2026090601020601_decision_test.json.gz"
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, "wt", encoding="utf-8") as f:
        json.dump({"record_type": "decision_snapshot", "race_id": "2026090601020601"}, f)
    report = scan(root=tmp_path)
    assert report["anomaly_count"] == 0
    assert report["by_stage"] == {}


# ------------------------------------------------------------------
# 2026-09-22: ユーザー指摘対応。「load_timing_valid_records()を将来の推奨関数と
# するだけでなく、価格形成研究の入力コードが生のarchiveを直接読めないことを
# テストまたは明示的なGateで保証してください」への回答としての require_timing_gate()。
# ------------------------------------------------------------------

def test_require_timing_gate_accepts_output_of_load_timing_valid_records(tmp_path: Path):
    _write_snapshot(tmp_path, "2026090109040701", "t10",
                    "2026-09-01T09:50:00", "2026-09-01T09:40:00.100")
    records = load_timing_valid_records("t10", root=tmp_path)
    require_timing_gate(records)  # 例外が出なければOK


def test_require_timing_gate_rejects_raw_dict_not_passed_through_gate():
    """load_timing_valid_records() を経由しない生dict (=生archiveを直接globして
    読んだ場合の典型形) は、gate未通過マーカーが無いため必ず弾かれる。将来の
    価格形成研究コードがこのgateを呼び忘れる/迂回することへの最後の防波堤。"""
    raw_direct_read = {"race_id": "2026090601020601", "stage": "t10",
                       "scheduled_post": "2026-09-06T09:50:00",
                       "observed_at": "2026-09-10T00:58:35.466"}  # 実際の-5,228分異常そのもの
    with pytest.raises(ValueError, match="timing gate"):
        require_timing_gate([raw_direct_read])


def test_require_timing_gate_rejects_mixed_batch_even_if_only_one_record_bypassed():
    good = {"race_id": "2026090109040701", "_timing_gate_passed": True}
    bypassed = {"race_id": "2026090601020601"}  # gateを経由していない
    with pytest.raises(ValueError):
        require_timing_gate([good, bypassed])


def test_require_timing_gate_would_have_caught_the_actual_5228_anomaly_if_read_raw(
        tmp_path: Path):
    """回帰の核心: もし将来の研究コードが load_timing_valid_records() を使わず
    forward_prices.read_snapshot() 等で生archiveを直接読んで組み立てたバッチを
    そのままモデルへ渡そうとしても、require_timing_gate() がgate未通過として
    必ず止める (=2026090601020601の-5,228分異常データが学習に混入する事故を
    構造的に防げる)。"""
    import forward_prices as fp
    path = _write_snapshot(tmp_path, "2026090601020601", "t10",
                           "2026-09-06T09:50:00", "2026-09-10T00:58:35.466")
    raw_batch = [fp.read_snapshot(path)]  # 生archiveの直接読み込みをシミュレート
    with pytest.raises(ValueError, match="timing gate"):
        require_timing_gate(raw_batch)
