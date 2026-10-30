# -*- coding: utf-8 -*-
"""
tests/test_shusse_narrate.py — 出世予測 AI v2 のテンプレート文（narrate.py）の単体テスト
======================================================================================
parquet に依存せず合成行で動かす。欠損（父 NaN・調教師不明・k=1）、確率 0/1 の端、確定/未確定の両方で落ちずに文が出ること。

    ./venv311/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/test_shusse_narrate.py
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from analysis.shusse.narrate import (
    LEVEL_NAME, direction_phrase, load_trainer_names, make_card, narrate, pct, race_display_name,
)

TRAINERS = {1103: "尾関知人"}

LABELS = {"n_wins": "勝利数", "mean_margin": "平均着差(秒)", "margin_1": "1走目の勝ち馬との着差(秒)", "debut_dayage": "デビュー時の日齢",
          "best_pos": "最高着順", "trn_te_lvl": "調教師の過去馬の平均到達レベル", "dist_1": "1走目の距離"}


def _starts(n: int, sire="テスト父", sex="牡", pos=(3, 1, 1), lvl=(0, 1, 2), after: list | None = None, trainer_code=1103) -> pd.DataFrame:
    rows = []
    d0 = pd.Timestamp("2022-09-19")
    for i in range(n):
        rows.append({"血統登録番号": 1, "馬名": "テストホース", "date": d0 + pd.Timedelta(days=60 * i), "場所": "中山", "レース名": ["新馬", "未勝利*", "山吹賞･1勝*"][i % 3],
                     "race_level": lvl[i], "pos": pos[i], "field": 10 + i, "margin": 0.0 if pos[i] == 1 else 0.6, "win": int(pos[i] == 1),
                     "性別": sex, "種牡馬": sire, "調教師コード": trainer_code, "芝・ダ": "芝", "距離": 2000, "k": i + 1})
    for a in after or []:
        rows.append({"血統登録番号": 1, "馬名": "テストホース", "date": d0 + pd.Timedelta(days=60 * (n - 1)) + pd.Timedelta(days=a["days"]), "場所": "京都",
                     "レース名": a.get("name", "菊花賞G1"), "race_level": a.get("lvl", 8), "pos": a["pos"], "field": 17, "margin": 0.0 if a["pos"] == 1 else 1.0,
                     "win": int(a["pos"] == 1), "性別": sex, "種牡馬": sire, "芝・ダ": "芝", "距離": 3000, "k": n + 1})
    return pd.DataFrame(rows)


def _row(k=3, **kw) -> dict:
    base = {"血統登録番号": 1, "馬名": "テストホース", "base_date": pd.Timestamp("2022-09-19") + pd.Timedelta(days=60 * (k - 1)), "debut_year": 2022,
            "p_dropout": 0.02, "p_L0_given_ran": 0.003, "p_L1_given_ran": 0.025, "p_L2_given_ran": 0.10, "p_L3_given_ran": 0.43, "p_L4_given_ran": 0.442,
            "expected_level_given_ran": 3.28, "pred_level_given_ran": 4, "p_graded12": 0.434, "p_graded24": 0.48, "rule_level": 2,
            "top3_A_L4": json.dumps([["n_wins", 1.44], ["mean_margin", 0.99], ["debut_dayage", 0.46]]),
            "top3_A_L0": json.dumps([["n_wins", -2.68], ["best_pos", -0.5]]), "top3_D": json.dumps([["margin_1", 0.3]]), "top3_B": None}
    base.update(kw)
    return base


def _assert_text_ok(text: str, min_sent: int = 3):
    assert isinstance(text, str) and len(text) > 40
    assert "nan" not in text.lower() and "None" not in text
    assert text.count("。") + text.count("）") >= min_sent


def test_full_card_confirmed_hit():
    s = _starts(3, after=[{"days": 200, "pos": 1}])
    r = _row(ran_12m=1, y_level=4.0, y_graded12=1, y_graded24=1.0, h24_complete=True, n_starts_12m=4)
    c = make_card(r, s, LABELS, k=3, cohort=pd.Series([0.01, 0.02, 0.9, 0.434, 0.001]), trainer_names=TRAINERS)
    t = narrate(c)
    _assert_text_ok(t)
    assert "テストホース" in t and "（牡、父テスト父、尾関知人厩舎）" in t and "山吹賞(1勝)" in t
    assert c["trainer"] == "尾関知人" and c["trainer_code"] == 1103
    assert "43%" in t and "2 段上" in t
    assert c["outcome"]["verdict"] == "的中" and "菊花賞" in t and "的中" in t
    assert c["cohort_top_pct"] is not None and "同世代の上位" in t
    assert all(x["phrase"] for x in c["reasons"])
    # 向きが決められない特徴はラベルだけ
    dd = {x["feature"]: x["directed"] for x in c["reasons"]}
    assert dd["n_wins"] is True and dd["debut_dayage"] is None


def test_missing_sire_trainer_k1_unconfirmed():
    s = _starts(1, sire=np.nan, pos=(2,), lvl=(0,))
    r = _row(k=1, p_graded12=0.41, p_dropout=0.02, rule_level=0, pred_level_given_ran=4)
    for key in ("ran_12m", "y_level", "y_graded12", "y_graded24", "h24_complete", "n_starts_12m"):
        r.pop(key, None)
    c = make_card(r, s, LABELS, trainer_names={})      # k を渡さない → 基準日以前の走数で推定、調教師表は空
    assert c["k"] == 1 and c["sire"] is None and c["trainer"] is None
    t = narrate(c)
    _assert_text_ok(t)
    assert "父" not in t.split("。")[0]
    assert "デビュー戦は勝てませんでした" in t
    assert c["outcome"] is None and "結果" not in t
    assert "4 段上" in t


def test_probability_edges_zero_one():
    s = _starts(3, pos=(12, 14, 15), lvl=(0, 1, 1))
    r = _row(p_dropout=1.0, p_L0_given_ran=1.0, p_L1_given_ran=0.0, p_L2_given_ran=0.0, p_L3_given_ran=0.0, p_L4_given_ran=0.0,
             p_graded12=0.0, p_graded24=0.0, pred_level_given_ran=0, rule_level=0, expected_level_given_ran=0.0,
             ran_12m=0, y_level=np.nan, y_graded12=0, y_graded24=0.0, h24_complete=True, n_starts_12m=0)
    c = make_card(r, s, LABELS, k=3)
    t = narrate(c)
    _assert_text_ok(t)
    assert "100%" in t and "0%" in t
    assert c["reason_head"] == "top3_D" and "離脱を疑う材料" in t
    assert c["outcome"]["ran_12m"] == 0 and c["outcome"]["verdict"] == "的中" and "離脱" in t
    assert "見立てと同じ" in t
    # 1% 未満の表記
    assert pct(0.001) == "1% 未満" and pct(0.0) == "0%" and pct(1.0) == "100%" and pct(None) == "—"


def test_dropout_not_predicted_and_near_miss():
    s = _starts(3, after=[{"days": 100, "pos": 1, "name": "ホープフG1", "lvl": 8}])
    # 離脱したのに P(離脱) が低い → 外れ（離脱は読めず）
    r = _row(ran_12m=0, y_level=np.nan, p_dropout=0.04, n_starts_12m=0)
    c = make_card(r, _starts(3), LABELS, k=3)
    assert c["outcome"]["verdict"].startswith("外れ") and "離脱は読めず" in narrate(c)
    # 予測 3 / 実績 4 → 惜しい、レース名の補完
    r2 = _row(ran_12m=1, y_level=4.0, pred_level_given_ran=3, rule_level=1, n_starts_12m=3, y_graded12=1, y_graded24=1.0, h24_complete=True)
    c2 = make_card(r2, s, LABELS, k=3)
    t2 = narrate(c2)
    assert c2["outcome"]["verdict"] == "惜しい" and "ホープフルS(G1)" in t2
    assert "確率を均した予測クラス" in t2   # argmax（重賞掲示板）と期待値丸め（3勝/OP）が違う


def test_robust_to_empty_starts_and_bad_top3():
    r = _row(top3_A_L4="not json", top3_A_L0=None, top3_D=float("nan"), top3_B=[["margin_1", -0.2]], ran_12m=1, y_level=2.0, n_starts_12m=2)
    c = make_card(r, None, LABELS, k=3)
    t = narrate(c)
    _assert_text_ok(t, min_sent=2)
    assert c["runs"] == [] and "走の明細なし" in t
    assert c["reason_head"] == "top3_B" and c["reasons"][0]["phrase"] == "1走目の勝ち馬との着差が大きい"
    assert c["outcome"]["actual_level_name"] == LEVEL_NAME[2]
    # dict 入力でも Series 入力でも同じ
    c2 = make_card(pd.Series(r), pd.DataFrame(), LABELS, k=3)
    assert narrate(c2) == t


def test_helpers():
    assert race_display_name("山吹賞･1勝*", 2) == "山吹賞"
    assert race_display_name("未勝利*", 1) == "" and race_display_name("新馬", 0) == ""
    assert race_display_name("１勝クラス", 2) == ""      # 全角は NFKC で揃えて汎用名として落とす
    assert race_display_name("菊花賞G1", 8) == "菊花賞" and race_display_name("阪神ジュG1", 8) == "阪神JF"
    assert race_display_name(np.nan, 5) == ""
    g, d = direction_phrase("margin_1", 0.5, LABELS, head_good=True)
    assert g == "1走目の勝ち馬との着差が小さい" and d is True
    g, d = direction_phrase("margin_1", 0.5, LABELS, head_good=False)
    assert g == "1走目の勝ち馬との着差が大きい" and d is False
    g, d = direction_phrase("trn_te_lvl", -0.1, LABELS, head_good=True)
    assert g == "調教師の過去馬の到達レベルが低い"
    g, d = direction_phrase("dist_1", 0.3, LABELS, head_good=True)
    assert g == "1走目の距離" and d is None


def test_trainer_code_not_in_map_is_omitted():
    """引退調教師など、コードが serve_code_maps の現役表に無い馬は厩舎名を省く（落ちない）。"""
    s = _starts(3, trainer_code=375)           # 現役表に無いコード
    r = _row(ran_12m=1, y_level=2.0, n_starts_12m=3)
    c = make_card(r, s, LABELS, k=3, trainer_names=TRAINERS)
    t = narrate(c)
    assert c["trainer"] is None and c["trainer_code"] == 375
    assert t.split("。")[0] == "**テストホース**（牡、父テスト父）"
    # コード欠損（NaN）でも落ちない。trainer_debut のフォールバックも効く
    s2 = _starts(3, trainer_code=np.nan)
    c2 = make_card(dict(r, trainer_debut=1103), s2, LABELS, k=3, trainer_names=TRAINERS)
    assert c2["trainer"] == "尾関知人"
    # 実ファイルの逆引き（存在すれば）: 値は int コード → 名前
    real = load_trainer_names()
    assert isinstance(real, dict) and (not real or all(isinstance(k, int) for k in real))
