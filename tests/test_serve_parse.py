# -*- coding: utf-8 -*-
"""serve 入力パーサと as-of 履歴特徴の回帰テスト。"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import predict_weekly as pw
from serve_history_feats import _CACHE, fill_history_features


def _tyaku_line(schema, overrides):
    values = [""] * len(schema)
    for key, value in overrides.items():
        values[schema.index(key)] = str(value)
    return ",".join(values)


def _write_tyaku(tmp_path, schema, overrides, date="20990101"):
    data_dir = tmp_path / "tyaku"
    data_dir.mkdir()
    race = ",".join(["2099010101010101"] + [""] * 18)
    horse = _tyaku_line(schema, overrides)
    (data_dir / f"{date}.csv").write_bytes((race + "\r\n" + horse + "\r\n").encode("cp932"))
    return data_dir


@pytest.mark.parametrize(
    "schema,extra",
    [
        (pw.TYAKU_HORSE_COLS_53, {}),
        (pw.TYAKU_HORSE_COLS_52, {}),
        (pw.TYAKU_HORSE_COLS, {"馬体重": 480, "増減": "+ 2"}),
    ],
)
def test_load_tyaku_supports_52_53_and_55_columns(tmp_path, monkeypatch, schema, extra):
    values = {
        "枠番": 1,
        "馬番": 2,
        "中央平地全:1着": 2,
        "中央平地全:2着": 1,
        "中央平地全:3着": 1,
        "中央平地全:外": 6,
        "同コース:1着": 1,
        "同コース:2着": 0,
        "同コース:3着": 0,
        "同コース:外": 2,
        "同クラス:1着": 0,
        "同クラス:2着": 1,
        "同クラス:3着": 0,
        "同クラス:外": 3,
        **extra,
    }
    monkeypatch.setattr(pw, "TYAKU_DIR", _write_tyaku(tmp_path, schema, values))

    out = pw._load_tyaku("20990101")

    assert out is not None and len(out) == 1
    assert int(out.loc[0, "馬番"]) == 2
    expected = (4 + 1.43) / (10 + 5.0)
    assert out.loc[0, "horse_fuku_career"] == pytest.approx(expected)
    if len(schema) in (52, 53):
        assert "馬体重" not in out.columns
        assert "増減" not in out.columns
    else:
        assert out.loc[0, "馬体重"] == 480
        assert out.loc[0, "増減"] == 2


def test_load_tyaku_warns_on_unsupported_width(tmp_path, monkeypatch, caplog):
    data_dir = tmp_path / "tyaku"
    data_dir.mkdir()
    race = ",".join(["2099010101010101"] + [""] * 18)
    bad_horse = ",".join(["1"] + [""] * 53)
    (data_dir / "20990101.csv").write_bytes((race + "\r\n" + bad_horse).encode("cp932"))
    monkeypatch.setattr(pw, "TYAKU_DIR", data_dir)

    assert pw._load_tyaku("20990101") is None
    assert "未対応列数" in caplog.text
    assert "54" in caplog.text


def test_fill_history_features_restores_asof_rolling_rates(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    hist = pd.DataFrame(
        {
            "ped_id": [1, 1, 1, 1, 1, 2, 1],
            "name": ["A", "A", "A", "A", "A", "B", "A"],
            "sire": ["S", "S", "S", "S", "S", "T", "S"],
            "birth_year": [2020, 2020, 2020, 2020, 2020, 2019, 2020],
            "date": [20240101, 20240102, 20240103, 20240104, 20240105, 20240106, 20240201],
            "place": ["東京"] * 7,
            "surface": ["芝"] * 7,
            "dist": [1600] * 7,
            "pos": [1, 4, 2, 5, 3, 4, 1],
            "jockey_code": [10] * 7,
            "trainer_code": [20] * 7,
            "src": ["test"] * 7,
        }
    )
    hist.to_parquet(data_dir / "_horse_history.parquet", index=False)
    (data_dir / "serve_code_maps.json").write_text(
        json.dumps({"jockey": {"J": 10}, "trainer": {"T": 20}}),
        encoding="utf-8",
    )
    frame = pd.DataFrame(
        {
            "馬名": ["A"],
            "種牡馬": ["S"],
            "年齢": [4],
            "日付": ["20240201"],
            "場所": ["東京"],
            "芝・ダ": ["芝"],
            "距離": [1600],
            "騎手": ["J"],
            "調教師": ["T"],
        }
    )
    _CACHE.clear()

    stats = fill_history_features(frame, base=tmp_path)

    assert stats["hit"] == 1
    assert frame.loc[0, "horse_fuku10"] == pytest.approx(3 / 5)
    assert frame.loc[0, "horse_fuku30"] == pytest.approx(3 / 5)
    # B の1走も主体別 rolling に入る。同日 20240201 の勝利は厳密に除外。
    assert frame.loc[0, "jockey_fuku30"] == pytest.approx(3 / 6)
    assert frame.loc[0, "jockey_fuku90"] == pytest.approx(3 / 6)
    assert frame.loc[0, "trainer_fuku30"] == pytest.approx(3 / 6)
    assert frame.loc[0, "trainer_fuku90"] == pytest.approx(3 / 6)
