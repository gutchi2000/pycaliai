# -*- coding: utf-8 -*-
"""
test_opponent_graph.py
========================
EXP12 Stage1。opponent_graph.pyの合成テスト+削除不変性テスト+IDコリジョン
テスト+external_history_gapテスト。2013-2025年の実データは使わない
(削除不変性の実データ確認はverify_deletion_invariance_realdata.pyで別途)。
"""
import numpy as np
import pandas as pd
import pytest

from opponent_graph import (
    run_day_batched_algorithm, add_external_history_gap, _compute_features,
    LATER_PROVED_STRONG_THRESHOLD,
)

COL_RID = "レースID(新/馬番無)"
COL_BAN = "馬番"
COL_HID = "血統登録番号"
COL_FINISH = "着順"


def _base_row(rid, ban, hid, date_int, finish, gap=0):
    return dict(**{COL_RID: rid, COL_BAN: ban, COL_HID: hid, "date_int": date_int,
                    COL_FINISH: finish, "external_history_gap": gap})


def _ability_row(hid, rid, ban, date_int, ability):
    return dict(hid=hid, **{COL_RID: rid}, ban=ban, date_int=date_int, ability=ability)


# ============================================================
# 1. 中核シナリオ: 「遭遇時は弱かった相手が、その後強いと判明した」
# ============================================================
def test_opponent_current_ability_not_frozen_at_encounter_time():
    """H1がDay1にH2(能力5)を破り、H2がDay2にH3を破って能力が9に上がった場合、
    Day3のH1の対戦相手特徴は「遭遇時の5」ではなく「現在の9」を反映すべき。
    これがscratchコード(opp_best_beaten系)との核心的な違い。"""
    base = pd.DataFrame([
        _base_row("R1", 1, "H1", 20230101, 1),
        _base_row("R1", 2, "H2", 20230101, 2),
        _base_row("R2", 1, "H2", 20230102, 1),
        _base_row("R2", 2, "H3", 20230102, 2),
        _base_row("R3", 1, "H1", 20230103, 1),  # 予測対象行
        _base_row("R3", 2, "H3", 20230103, 2),
    ])
    ability = pd.DataFrame([
        _ability_row("H1", "R1", 1, 20230101, 10.0),
        _ability_row("H2", "R1", 2, 20230101, 5.0),   # H2のDay1時点能力(遭遇時)
        _ability_row("H2", "R2", 1, 20230102, 9.0),   # H2のDay2時点能力(その後判明)
        _ability_row("H3", "R2", 2, 20230102, 3.0),
        _ability_row("H1", "R3", 1, 20230103, 10.0),
        _ability_row("H3", "R3", 2, 20230103, 3.0),
    ])
    out = run_day_batched_algorithm(base, ability, output_year_set={2023})
    row = out[(out["hid"] == "H1") & (out["rid16"] == "R3")].iloc[0]

    assert row["unique_opponent_count"] == 1
    # 核心アサーション: 5(遭遇時)ではなく9(現在)であること
    assert row["opponent_current_strength_mean"] == pytest.approx(9.0)
    assert row["opponent_current_strength_max"] == pytest.approx(9.0)
    assert row["beaten_opponent_strength_mean"] == pytest.approx(9.0)
    assert row["strongest_beaten_opponent"] == pytest.approx(9.0)
    assert np.isnan(row["lost_to_opponent_strength_mean"])
    assert row["opponent_strength_dispersion"] == pytest.approx(0.0)
    # 9-5=4 > LATER_PROVED_STRONG_THRESHOLD(3.0) なので1件カウントされるはず
    assert row["opponent_later_proved_strong_count"] == 1


def test_target_race_coparticipants_are_not_edges():
    """Day3のR3でH1とH3が同走するが、H1にとってH3はまだ「過去の対戦相手」
    ではない(このレース自体が予測対象)。H1のunique_opponent_countは
    H3を含まず1のままであるべき(H2のみ)。"""
    base = pd.DataFrame([
        _base_row("R1", 1, "H1", 20230101, 1),
        _base_row("R1", 2, "H2", 20230101, 2),
        _base_row("R3", 1, "H1", 20230103, 1),
        _base_row("R3", 2, "H3", 20230103, 2),  # H1とH3は今日初めて同走
    ])
    ability = pd.DataFrame([
        _ability_row("H1", "R1", 1, 20230101, 10.0),
        _ability_row("H2", "R1", 2, 20230101, 5.0),
        _ability_row("H1", "R3", 1, 20230103, 10.0),
        _ability_row("H3", "R3", 2, 20230103, 3.0),
    ])
    out = run_day_batched_algorithm(base, ability, output_year_set={2023})
    row = out[(out["hid"] == "H1") & (out["rid16"] == "R3")].iloc[0]
    assert row["unique_opponent_count"] == 1  # H3は含まれない


def test_same_day_earlier_race_not_used():
    """同日内で先に行われたレースの結果も、同日の別レースの特徴には
    使わない(日初時点固定)。R1a(午前)でH1がH2を破っても、同日R1b(午後)の
    H1の対戦相手特徴にはまだ反映されないはず。"""
    base = pd.DataFrame([
        _base_row("R1a", 1, "H1", 20230101, 1),
        _base_row("R1a", 2, "H2", 20230101, 2),
        _base_row("R1b", 1, "H1", 20230101, 1),  # 同日の別レース
        _base_row("R1b", 2, "H3", 20230101, 2),
    ])
    ability = pd.DataFrame([
        _ability_row("H1", "R1a", 1, 20230101, 10.0),
        _ability_row("H2", "R1a", 2, 20230101, 5.0),
        _ability_row("H1", "R1b", 1, 20230101, 10.0),
        _ability_row("H3", "R1b", 2, 20230101, 3.0),
    ])
    out = run_day_batched_algorithm(base, ability, output_year_set={2023})
    row_a = out[(out["hid"] == "H1") & (out["rid16"] == "R1a")].iloc[0]
    row_b = out[(out["hid"] == "H1") & (out["rid16"] == "R1b")].iloc[0]
    assert row_a["unique_opponent_count"] == 0  # R1a時点でH1に過去対戦相手はいない
    assert row_b["unique_opponent_count"] == 0  # R1bも同日なのでR1aの結果は未反映


def test_no_history_gives_nan_and_zero():
    base = pd.DataFrame([_base_row("R1", 1, "H1", 20230101, 1)])
    ability = pd.DataFrame([_ability_row("H1", "R1", 1, 20230101, 10.0)])
    out = run_day_batched_algorithm(base, ability, output_year_set={2023})
    row = out.iloc[0]
    assert row["unique_opponent_count"] == 0
    assert np.isnan(row["opponent_current_strength_mean"])
    assert row["opponent_later_proved_strong_count"] == 0


def test_beaten_and_lost_to_are_distinguished():
    """H1がDay1にH2に負け、Day2にH3に勝つ。Day3の予測でbeaten/lost_toが
    正しく分離されること。"""
    base = pd.DataFrame([
        _base_row("R1", 1, "H2", 20230101, 1),
        _base_row("R1", 2, "H1", 20230101, 2),  # H1負け
        _base_row("R2", 1, "H1", 20230102, 1),
        _base_row("R2", 2, "H3", 20230102, 2),  # H1勝ち
        _base_row("R3", 1, "H1", 20230103, 1),
    ])
    ability = pd.DataFrame([
        _ability_row("H2", "R1", 1, 20230101, 20.0),
        _ability_row("H1", "R1", 2, 20230101, 10.0),
        _ability_row("H1", "R2", 1, 20230102, 10.0),
        _ability_row("H3", "R2", 2, 20230102, 6.0),
    ])
    # H2/H3はR1/R2以降レースに出ないため、20230103時点でのcurrent_abilityは
    # H2=20(R1時点のまま), H3=6(R2時点のまま)
    out = run_day_batched_algorithm(base, ability, output_year_set={2023})
    row = out[(out["hid"] == "H1") & (out["rid16"] == "R3")].iloc[0]
    assert row["unique_opponent_count"] == 2
    assert row["lost_to_opponent_strength_mean"] == pytest.approx(20.0)  # H2
    assert row["beaten_opponent_strength_mean"] == pytest.approx(6.0)    # H3


# ============================================================
# 2. external_history_gap
# ============================================================
def test_external_history_gap_flagged_on_mismatch():
    COL_PREV_DATE = "前走日付"
    df = pd.DataFrame({
        COL_HID: ["H1", "H1", "H1"],
        "date_int": [20230101, 20230201, 20230301],
        COL_PREV_DATE: [np.nan, 230101.0, 230215.0],  # 3走目は前走日付が230215だが、
                                                          # masterの直前行は0201(不一致)
    })
    out = add_external_history_gap(df)
    assert out.iloc[0]["external_history_gap"] == 0  # デビュー戦、前走日付なし
    assert out.iloc[1]["external_history_gap"] == 0  # 0101と一致
    assert out.iloc[2]["external_history_gap"] == 1  # 0215 != 0201、地方/海外等の挟在を示唆


def test_external_history_gap_not_flagged_when_consistent():
    COL_PREV_DATE = "前走日付"
    df = pd.DataFrame({
        COL_HID: ["H1", "H1"],
        "date_int": [20230101, 20230201],
        COL_PREV_DATE: [np.nan, 230101.0],
    })
    out = add_external_history_gap(df)
    assert out.iloc[1]["external_history_gap"] == 0


# ============================================================
# 3. IDコリジョン(文字列保持)テスト
# ============================================================
def test_hid_kept_as_string_no_leading_zero_loss():
    """血統登録番号は文字列のまま保持され、先頭ゼロが消失しないこと。"""
    base = pd.DataFrame([
        _base_row("R1", 1, "0010105413", 20230101, 1),
        _base_row("R1", 2, "0010105526", 20230101, 2),
    ])
    ability = pd.DataFrame([
        _ability_row("0010105413", "R1", 1, 20230101, 10.0),
        _ability_row("0010105526", "R1", 2, 20230101, 5.0),
    ])
    out = run_day_batched_algorithm(base, ability, output_year_set={2023})
    assert set(out["hid"]) == {"0010105413", "0010105526"}
    assert isinstance(out.iloc[0]["hid"], str)


def test_hid_distinguishes_numerically_equal_but_differently_padded_ids():
    """先頭ゼロの有無で異なる馬として区別されること(数値変換されていたら
    "0010105413"と"10105413"が衝突してしまう)。"""
    base = pd.DataFrame([
        _base_row("R1", 1, "0010105413", 20230101, 1),
        _base_row("R2", 1, "10105413", 20230102, 1),  # 別の(架空の)馬
    ])
    ability = pd.DataFrame([
        _ability_row("0010105413", "R1", 1, 20230101, 10.0),
        _ability_row("10105413", "R2", 1, 20230102, 20.0),
    ])
    out = run_day_batched_algorithm(base, ability, output_year_set={2023})
    # current_abilityが別キーとして扱われているか、2頭のhidが別々に残っているかで確認
    assert len(out) == 2
    assert set(out["hid"]) == {"0010105413", "10105413"}


# ============================================================
# 4. 削除不変性(合成データ)
# ============================================================
def test_deletion_invariance_synthetic():
    """未来のレースを削除しても、過去の予測行の特徴は変化しないはず。"""
    base_full = pd.DataFrame([
        _base_row("R1", 1, "H1", 20230101, 1),
        _base_row("R1", 2, "H2", 20230101, 2),
        _base_row("R2", 1, "H1", 20230201, 1),  # 予測対象(過去側)
        _base_row("R3", 1, "H1", 20230301, 1),  # 未来のレース(削除される)
        _base_row("R3", 2, "H2", 20230301, 2),
    ])
    ability_full = pd.DataFrame([
        _ability_row("H1", "R1", 1, 20230101, 10.0),
        _ability_row("H2", "R1", 2, 20230101, 5.0),
        _ability_row("H1", "R2", 1, 20230201, 10.0),
        _ability_row("H1", "R3", 1, 20230301, 11.0),
        _ability_row("H2", "R3", 2, 20230301, 6.0),
    ])
    out_full = run_day_batched_algorithm(base_full, ability_full, output_year_set={2023})

    base_trunc = base_full[base_full["date_int"] < 20230301]
    ability_trunc = ability_full[ability_full["date_int"] < 20230301]
    out_trunc = run_day_batched_algorithm(base_trunc, ability_trunc, output_year_set={2023})

    row_full = out_full[(out_full["hid"] == "H1") & (out_full["rid16"] == "R2")].iloc[0]
    row_trunc = out_trunc[(out_trunc["hid"] == "H1") & (out_trunc["rid16"] == "R2")].iloc[0]
    for col in ["unique_opponent_count", "opponent_current_strength_mean",
                "beaten_opponent_strength_mean", "opponent_later_proved_strong_count"]:
        a, b = row_full[col], row_trunc[col]
        if pd.isna(a) and pd.isna(b):
            continue
        assert a == pytest.approx(b), f"{col}: full={a} trunc={b}"


# ============================================================
# 5. _compute_features 単体
# ============================================================
def test_compute_features_recency_weighting_favors_recent_opponent():
    """recency_weighted_opponent_strengthは、直近の対戦相手により大きい
    重みを与えるはず。"""
    hist = {
        "OLD": [5.0, 1, 0, 20200101],   # 古い対戦、弱い相手
        "NEW": [5.0, 1, 0, 20230101],   # 最近の対戦、同じ遭遇時強さ
    }
    current_ability = {"OLD": 5.0, "NEW": 20.0}  # OLDは変化なし、NEWは現在強い
    feats = _compute_features(hist, current_ability)
    simple_mean = (5.0 + 20.0) / 2
    assert feats["recency_weighted_opponent_strength"] > simple_mean  # NEWの重みが大きいため平均より高い


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
