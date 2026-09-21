"""
test_export_weekly_marks.py
============================
2026-09-21、venue06(中山) store_prediction 障害の原因調査で追加した回帰テスト。
2026-09-22、ユーザー指摘を受けて venue 別オッズ非対称の扱いを「push を止める
gate_errors」から「morning_odds_unavailable 付与 + 非ブロッキング警告」へ改訂。

根本原因: 「全馬 tansho_odds=None のレースは中止/未確定とみなし bundle から除外する」
フィルタ (2026-09-21朝に一時追加、同日中に誤検知が確定し撤回) が、TARGET/OD CSV側の
オッズ配信タイミング差だけで venue06 の12レース全てを bundle から除外し、bundle依存の
T-10自動馬券登録・T-20サイトプレビュー登録・EXP05-F store_prediction がvenue06を
丸ごと欠測させた。

本ファイルは:
  1. 除外ロジックが完全に撤去されていること (races はフィルタされず全件bundleに残る)
  2. annotate_venue_odds_degradation() が「1venueだけほぼ0%・他venueは正常」を
     正しく検知し、degradedなvenueだけへ morning_odds_unavailable=true を付与し、
     races 自体は一切削除しない (件数・alive venue の中身は不変) こと
を検証する。実行: python -m pytest test_export_weekly_marks.py -q
"""
from __future__ import annotations

from export_weekly_marks import annotate_venue_odds_degradation, venue_odds_coverage


def _race(rid: str, n_horses: int, odds_present: bool) -> dict:
    return {
        "race_id": rid,
        "race_meta": {},
        "horses": [
            {"umaban": i + 1, "tansho_odds": (10.0 + i) if odds_present else None}
            for i in range(n_horses)
        ],
    }


def _day_races(venue06_odds: bool, venue09_odds: bool, n=12) -> list[dict]:
    races = []
    for i in range(1, n + 1):
        rn = f"{i:02d}"
        races.append(_race(f"20260921060407{rn}", 14, venue06_odds))
        races.append(_race(f"20260921090407{rn}", 16, venue09_odds))
    return races


def test_no_races_are_dropped_by_annotate_venue_odds_degradation():
    """races の件数・race_id集合は annotate 前後で一切変わらない
    (2026-09-21以前の「除外してbundleから消す」挙動の撤回を保証する)。"""
    races = _day_races(venue06_odds=False, venue09_odds=True)
    before_ids = sorted(r["race_id"] for r in races)
    annotate_venue_odds_degradation(races)
    after_ids = sorted(r["race_id"] for r in races)
    assert before_ids == after_ids
    assert len(after_ids) == 24  # venue06/09 とも 12 レースずつ、1件も消えない


def test_today_scenario_marks_only_dead_venue_races():
    """2026-09-21の実インシデントの再現: venue06=0%・venue09≈100%。
    venue06のレース/馬にだけ morning_odds_unavailable=true が付き、venue09は
    一切触られない (bet可否には無関係、単なる目印)。"""
    races = _day_races(venue06_odds=False, venue09_odds=True)
    warnings = annotate_venue_odds_degradation(races)
    assert len(warnings) == 1
    assert "06" in warnings[0] and "09" in warnings[0]

    v06 = [r for r in races if r["race_id"][8:10] == "06"]
    v09 = [r for r in races if r["race_id"][8:10] == "09"]
    assert len(v06) == 12 and len(v09) == 12
    for r in v06:
        assert r["race_meta"]["morning_odds_unavailable"] is True
        assert all(h.get("morning_odds_unavailable") is True for h in r["horses"])
    for r in v09:
        assert "morning_odds_unavailable" not in r["race_meta"]
        assert all("morning_odds_unavailable" not in h for h in r["horses"])


def test_both_venues_healthy_no_warning_no_annotation():
    races = _day_races(venue06_odds=True, venue09_odds=True)
    warnings = annotate_venue_odds_degradation(races)
    assert warnings == []
    assert all("morning_odds_unavailable" not in r["race_meta"] for r in races)


def test_full_day_cancellation_all_venues_dead_not_flagged_here():
    """全venueがオッズ0%(真の全面中止シナリオ)は「非対称」ではないのでこの関数は
    何も付与・警告しない — それは呼び出し側の全体 odds_cov<50% ゲートの役割
    (責務分離: このcanaryは「一部venueだけ死んでいる」偏りの検知に特化)。"""
    races = _day_races(venue06_odds=False, venue09_odds=False)
    assert annotate_venue_odds_degradation(races) == []
    assert all("morning_odds_unavailable" not in r["race_meta"] for r in races)


def test_single_venue_day_no_asymmetry_check_applies():
    """開催venueが1つしか無い日はvenue間比較ができないため検査対象外。"""
    races = [_race(f"202609210604070{i}", 14, False) for i in range(1, 5)]
    assert annotate_venue_odds_degradation(races) == []


def test_partial_odds_gap_below_dead_threshold_not_flagged():
    """0%ではなく単に被覆率が低いだけ (例 30%) のvenueは「ほぼ全滅」ではないので
    誤検知しない (閾値 <10% は「実質ゼロ」を狙った保守的な設定)。"""
    races = _day_races(venue06_odds=False, venue09_odds=True)
    for r in races:
        if r["race_id"][8:10] == "06":
            for h in r["horses"][:4]:
                h["tansho_odds"] = 12.3
    warnings = annotate_venue_odds_degradation(races)
    assert warnings == []  # 30% は <10% の "dead" 閾値を割らない
    assert all("morning_odds_unavailable" not in r["race_meta"] for r in races)


def test_venue_extraction_preserves_leading_zero_for_all_10_jra_venues():
    """race_id[8:10] のvenue抽出で '01'/'06' 等の先頭ゼロが落ちないこと
    (int castバグがあれば '06'→'6' となり別venueとしてグルーピングされてしまう)。"""
    races = []
    for i in range(1, 11):
        code = f"{i:02d}"
        races.append(_race(f"20260921{code}040701", 10, odds_present=(i % 2 == 0)))
    # 偶数venueだけオッズあり、奇数venueだけ無し = 明確な非対称のはず
    warnings = annotate_venue_odds_degradation(races)
    assert len(warnings) == 1
    for i in range(1, 11):
        assert f"{i:02d}=" in warnings[0]  # 全venueがゼロ埋め2桁のまま報告文に出ている


def test_venue_odds_coverage_helper_matches_annotate_thresholds():
    races = _day_races(venue06_odds=False, venue09_odds=True)
    cov = venue_odds_coverage(races)
    assert cov["06"] == 0.0
    assert cov["09"] == 1.0
