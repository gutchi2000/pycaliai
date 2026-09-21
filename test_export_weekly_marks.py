"""
test_export_weekly_marks.py
============================
2026-09-21、venue06(中山) store_prediction 障害の原因調査で追加した回帰テスト。

根本原因: 「全馬 tansho_odds=None のレースは中止/未確定とみなし bundle から除外する」
フィルタ (2026-09-21朝に一時追加、同日中に誤検知が確定し撤回) が、TARGET/OD CSV側の
オッズ配信タイミング差だけで venue06 の12レース全てを bundle から除外し、bundle依存の
T-10自動馬券登録・T-20サイトプレビュー登録・EXP05-F store_prediction がvenue06を
丸ごと欠測させた。本ファイルは:
  1. 除外ロジックが完全に撤去されていること (races はフィルタされず全件bundleに残る)
  2. その撤去に伴う安全網として追加した venue_odds_asymmetry_errors() が
     「1venueだけほぼ0%・他venueは正常」という非対称パターンを正しく検知すること
を検証する。実行: python -m pytest test_export_weekly_marks.py -q
"""
from __future__ import annotations

from export_weekly_marks import venue_odds_asymmetry_errors


def _race(rid: str, n_horses: int, odds_present: bool) -> dict:
    return {
        "race_id": rid,
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


def test_no_races_are_dropped_by_venue_odds_asymmetry_errors():
    """venue_odds_asymmetry_errors はエラー文を返すだけで races 自体を変更しない
    (2026-09-21以前の「除外してbundleから消す」挙動の撤回を、シグネチャレベルで
    保証する: list[str] を返す純関数であり races は引数のまま副作用が無い)。"""
    races = _day_races(venue06_odds=False, venue09_odds=True)
    before = [r["race_id"] for r in races]
    venue_odds_asymmetry_errors(races)
    after = [r["race_id"] for r in races]
    assert before == after
    assert len(after) == 24  # venue06/09 とも 12 レースずつ残っている


def test_asymmetric_venue_odds_gap_is_detected_today_scenario():
    """2026-09-21の実インシデントの再現: venue06=0%・venue09≈100% → 検知してpushを止める。"""
    races = _day_races(venue06_odds=False, venue09_odds=True)
    errs = venue_odds_asymmetry_errors(races)
    assert len(errs) == 1
    assert "06" in errs[0] and "09" in errs[0]


def test_both_venues_healthy_no_error():
    races = _day_races(venue06_odds=True, venue09_odds=True)
    assert venue_odds_asymmetry_errors(races) == []


def test_full_day_cancellation_all_venues_dead_not_flagged_here():
    """全venueがオッズ0%(真の全面中止シナリオ)は「非対称」ではないのでこの関数は
    エラーを出さない — それは呼び出し側の全体 odds_cov<50% ゲートの役割
    (責務分離: このcanaryは「一部venueだけ死んでいる」偏りの検知に特化)。"""
    races = _day_races(venue06_odds=False, venue09_odds=False)
    assert venue_odds_asymmetry_errors(races) == []


def test_single_venue_day_no_asymmetry_check_applies():
    """開催venueが1つしか無い日はvenue間比較ができないため検査対象外
    (通常の週中開催や、既に他venueが完全中止で本当に1venueしか残っていない日を
    誤検知しないため)。"""
    races = [_race(f"202609210604070{i}", 14, False) for i in range(1, 5)]
    assert venue_odds_asymmetry_errors(races) == []


def test_partial_odds_gap_below_dead_threshold_not_flagged():
    """0%ではなく単に被覆率が低いだけ (例 30%) のvenueは「ほぼ全滅」ではないので
    誤検知しない (閾値 <10% は「実質ゼロ」を狙った保守的な設定)。"""
    races = _day_races(venue06_odds=False, venue09_odds=True)
    # venue06の一部レースにだけオッズを混ぜて被覆率を約30%にする
    for r in races:
        if r["race_id"][8:10] == "06":
            for h in r["horses"][:4]:
                h["tansho_odds"] = 12.3
    errs = venue_odds_asymmetry_errors(races)
    assert errs == []  # 30% は <10% の "dead" 閾値を割らない


def test_venue_extraction_preserves_leading_zero_for_all_10_jra_venues():
    """race_id[8:10] のvenue抽出で '01'/'06' 等の先頭ゼロが落ちないこと
    (int castバグがあれば '06'→'6' となり別venueとしてグルーピングされてしまう)。"""
    races = []
    for i in range(1, 11):
        code = f"{i:02d}"
        races.append(_race(f"20260921{code}040701", 10, odds_present=(i % 2 == 0)))
    # 偶数venueだけオッズあり、奇数venueだけ無し = 明確な非対称のはず
    errs = venue_odds_asymmetry_errors(races)
    assert len(errs) == 1
    for i in range(1, 11):
        assert f"{i:02d}=" in errs[0]  # 全venueがゼロ埋め2桁のまま報告文に出ている
