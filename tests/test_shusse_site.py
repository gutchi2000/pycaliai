# -*- coding: utf-8 -*-
"""
tests/test_shusse_site.py — 出世予測 AI のサイト用 JSON（analysis/shusse/site_export.py）の検査
=================================================================================================
(a) index.json: 年・件数・検証数値が揃い、years の各年に {year}.json がある
(b) 年ファイル: 各馬の必須キー、確率の範囲、pg12 = (1 − pd) × pl[4]、pl の和 = 1、rank が 1..n の連番、
    未確定（基準日 + 365 日 > asof）だけが入っている
(c) 文章カードは年ごとに top_cards 頭以下で、全部その年の馬、text が空でない
(d) 市場情報（オッズ・人気）のキーが一切無い

実行:
    ./venv311/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/test_shusse_site.py
analysis/shusse/out_v2/live/site/index.json が無ければ skip（predict_live → site_export を先に実行）。
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

BASE = Path(__file__).resolve().parents[1]
SITE_SRC = BASE / "analysis" / "shusse" / "out_v2" / "live" / "site"
pytestmark = pytest.mark.skipif(not (SITE_SRC / "index.json").exists(), reason="run analysis.shusse.site_export first")

REQUIRED = {"id", "name", "dy", "sex", "sire", "trainer", "n", "k", "base", "summary", "pd", "pl", "pg12", "pg24", "pct", "lvl", "rank", "why", "hist"}
FORBIDDEN_SUBSTR = ("odds", "オッズ", "人気", "ninki", "pay")


@pytest.fixture(scope="module")
def index():
    return json.loads((SITE_SRC / "index.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def years(index):
    return {y: json.loads((SITE_SRC / f"{y}.json").read_text(encoding="utf-8")) for y in index["years"]}


def test_index_schema(index):
    assert index["model"] == "shusse_v2"
    pd.Timestamp(index["asof"])
    assert index["years"] == sorted(index["years"], reverse=True)
    assert set(map(str, index["n_by_year"])) >= set(map(str, index["years"]))
    assert sum(index["n_by_year"].values()) == index["n_horses"] > 0
    assert set(index["levels"]) == {"0", "1", "2", "3", "4"}
    for k in ("k1", "k2", "k3", "k5"):
        v = index["validation"][k]
        assert 0 <= v["test2022"]["qwk"] <= 1 and 0.5 <= v["test2022"]["g12_auc"] <= 1
        assert v["forward2023"]["n"] > 0 and 0.5 <= v["forward2023"]["g12_auc"] <= 1
    for y in index["years"]:
        assert (SITE_SRC / f"{y}.json").exists()


def test_year_files(index, years):
    asof = pd.Timestamp(index["asof"])
    for y, d in years.items():
        hs = d["horses"]
        assert d["year"] == y and d["n"] == len(hs) == index["n_by_year"][str(y)]
        assert sorted(h["rank"] for h in hs) == list(range(1, len(hs) + 1))
        ranks_by_p = sorted(hs, key=lambda h: -h["pg12"])
        assert [h["rank"] for h in ranks_by_p][:50] == list(range(1, min(50, len(hs)) + 1))
        for h in hs:
            assert REQUIRED <= set(h), REQUIRED - set(h)
            assert h["dy"] == y and h["k"] in (1, 2, 3, 5) and h["n"] >= h["k"]
            assert len(h["pl"]) == 5 and abs(sum(h["pl"]) - 1) < 0.01
            for p in (h["pd"], h["pg12"], h["pg24"], *h["pl"]):
                assert 0 <= p <= 1
            assert abs(h["pg12"] - (1 - h["pd"]) * h["pl"][4]) < 0.003
            assert 0 <= h["pct"] <= 100 and 0 <= h["lvl"] <= 4
            assert pd.Timestamp(h["base"]) + pd.Timedelta(days=365) > asof, "confirmed horse leaked into live set"
            assert isinstance(h["why"], list) and all(isinstance(w, str) and w for w in h["why"])
            assert h["hist"] and h["hist"][-1][0] == h["k"]
            assert [r[0] for r in h["hist"]] == sorted(r[0] for r in h["hist"])


def test_cards(index, years):
    for y, d in years.items():
        ids = {str(h["id"]) for h in d["horses"]}
        assert 0 < len(d["cards"]) <= index["top_cards"]
        assert set(d["cards"]) <= ids
        top = {str(h["id"]) for h in sorted(d["horses"], key=lambda h: -h["pg12"])[: len(d["cards"])]}
        assert set(d["cards"]) == top
        for c in d["cards"].values():
            assert c["text"].strip() and "%" in c["text"]
            assert c["runs"] and all({"date", "label", "pos"} <= set(r) for r in c["runs"])


def test_no_market_fields(index, years):
    def keys(o, acc):
        if isinstance(o, dict):
            for k, v in o.items():
                acc.add(str(k).lower())
                keys(v, acc)
        elif isinstance(o, list):
            for v in o:
                keys(v, acc)
        return acc
    seen = keys(index, set())
    for d in years.values():
        keys(d, seen)
    bad = [k for k in seen if any(s in k for s in FORBIDDEN_SUBSTR)]
    assert not bad, bad
