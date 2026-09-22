# -*- coding: utf-8 -*-
"""
tests/conftest.py — テスト用の eligibility 前提
==============================================
P0 hard gate（`race_eligibility`）は、障害か否かを authoritative に判定できない
レースを **fail-closed で ineligible** にする。これは本番として正しいが、
合成 race_id（`2099...` の予約日付）を使う既存テストはそのままでは
「判定不能」になってしまう。

そこで **`2099` 始まりの合成日付に限り**「authoritative に平地」と宣言する。
実日付（`20xx` の実在開催日）には一切触れないため、本番相当の判定は
そのまま検査される。個別テストが `_bunseki_track_codes` を monkeypatch した
場合はそちらが優先される（本 fixture は session 単位で先に入るだけ）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

# 合成テスト用の予約日付プレフィクス（実在しない）
SYNTHETIC_DATE_PREFIX = "2099"
FLAT_TRACK_CODE = 23          # 芝・内。**テスト内でのみ authoritative 扱い**


@pytest.fixture(scope="session", autouse=True)
def _synthetic_races_are_flat():
    """合成 race_id を authoritative な平地として登録する。"""
    import race_eligibility as m

    real = m._bunseki_track_codes

    def wrapped(date: str):
        if str(date).startswith(SYNTHETIC_DATE_PREFIX):
            # この日のすべての race を平地として返す（race_id は問わない）
            return _SyntheticFlatMap(), "synthetic-test-source"
        return real(date)

    m._bunseki_track_codes = wrapped          # type: ignore[assignment]
    m.clear_cache()
    yield
    m._bunseki_track_codes = real             # type: ignore[assignment]
    m.clear_cache()


class _SyntheticFlatMap(dict):
    """`rid in m` が常に True、`m[rid]` が平地コードを返す辞書。"""

    def __contains__(self, key) -> bool:      # noqa: D105
        return True

    def __getitem__(self, key) -> int:        # noqa: D105
        return FLAT_TRACK_CODE
