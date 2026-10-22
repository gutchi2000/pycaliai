# -*- coding: utf-8 -*-
"""O1 parser — EXP22 o1_history.py の複製（read-only, fail-closed, token 保持）。

原本: C:/Users/gutch/Documents/Codex/PyCaLiAI-exp24/analysis/mcond/exp22_synchronized_cross_pool_waku_dev/o1_history.py
（リポジトリ外の Codex worktree にあるため、レイアウトと検査をそのまま複製した。変更点は無い）
EXP22 Stage 0 監査で全 75,619 録の slot 照合・token 分類が検証済み（STAGE0_DATA_AUDIT.md）。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

RECORD_LEN = 960
HEADER_REGISTERED = (35, 37)
HEADER_STARTERS = (37, 39)
HEADER_SALE_FLAGS = (39, 42)
HEADER_PLACE_KEY = 42
TAN_SLOT0, TAN_ODDS0, TAN_STRIDE, HORSE_SLOTS = 43, 45, 8, 28
FUK_SLOT0, FUK_ODDS0, FUK_STRIDE = 267, 269, 12
WAKU0, WAKU_STRIDE, WAKU_SLOTS = 603, 9, 36
TOTAL_TAN0, TOTAL_FUK0, TOTAL_WAKU0 = 927, 938, 949


def _integer(text: str) -> int | None:
    value = text.strip()
    return int(value) if value.isdigit() else None


def classify_token(token: str) -> str:
    if token.strip().isdigit():
        return "numeric"
    if token and set(token) == {"*"}:
        return "asterisk"
    if token and set(token) == {"-"}:
        return "hyphen"
    if not token.strip():
        return "blank"
    return "other"


def expected_waku_pairs() -> list[str]:
    return [f"{i}{j}" for i in range(1, 9) for j in range(i, 9)]


def frame_for_horse(horse_number: int, declared_headcount: int) -> int:
    if not 1 <= declared_headcount <= 18:
        raise ValueError(f"unsupported declared_headcount={declared_headcount}")
    if not 1 <= horse_number <= declared_headcount:
        raise ValueError(f"horse_number={horse_number} outside 1..{declared_headcount}")
    sizes = [1] * 8
    remaining = declared_headcount - 8
    while remaining > 0:
        for frame in range(7, -1, -1):
            if remaining == 0:
                break
            sizes[frame] += 1
            remaining -= 1
    cumulative = 0
    for frame, size in enumerate(sizes, 1):
        cumulative += size
        if horse_number <= cumulative:
            return frame
    raise AssertionError("unreachable")


@dataclass(frozen=True)
class O1Snapshot:
    division: int
    created_yyyymmdd: str
    race_id: str
    announced_mmddhhmm: str
    registered_headcount: int
    starter_count: int
    sale_flags: str
    place_pay_key: str
    tansho: dict[int, float]
    fukusho: dict[int, tuple[float, float]]
    wakuren: dict[str, float]
    tansho_tokens: dict[int, str]
    fukusho_tokens: dict[int, tuple[str, str]]
    wakuren_tokens: dict[str, str]
    total_tansho_votes: int | None
    total_fukusho_votes: int | None
    total_wakuren_votes: int | None

    @property
    def headcount(self) -> int:
        return self.registered_headcount

    @property
    def announced_at(self) -> datetime:
        return datetime.strptime(f"{self.race_id[:4]}{self.announced_mmddhhmm}", "%Y%m%d%H%M")

    @property
    def active_horses(self) -> set[int]:
        return set(self.tansho)

    @property
    def cancelled_or_unpriced_horses(self) -> set[int]:
        return set(range(1, self.registered_headcount + 1)) - self.active_horses

    @property
    def offered_wakuren_pairs(self) -> set[str]:
        return {
            pair for pair, token in self.wakuren_tokens.items()
            if classify_token(token) in {"numeric", "asterisk"}
        }


def _assert_slot_number(raw: str, expected: int, block: str) -> None:
    value = _integer(raw)
    if value is not None and value != expected:
        raise ValueError(f"{block} slot horse {value} != expected {expected}")


def parse_o1_record(record: str) -> O1Snapshot:
    record = record.rstrip("\r\n")
    if len(record) != RECORD_LEN:
        raise ValueError(f"O1 record length {len(record)} != {RECORD_LEN}")
    if record[:2] != "O1":
        raise ValueError(f"record type {record[:2]!r} is not O1")

    division = _integer(record[2:3])
    race_id = record[11:27]
    announced = record[27:35]
    registered = _integer(record[slice(*HEADER_REGISTERED)])
    starters = _integer(record[slice(*HEADER_STARTERS)])
    if division is None:
        raise ValueError("missing division")
    if len(race_id) != 16 or not race_id.isdigit():
        raise ValueError(f"invalid race_id {race_id!r}")
    if len(announced) != 8 or not announced.isdigit():
        raise ValueError(f"invalid announced time {announced!r}")
    if registered is None or not 1 <= registered <= 28:
        raise ValueError(f"invalid registered headcount {registered!r}")
    if starters is None or not 0 <= starters <= registered:
        raise ValueError(f"invalid starter count {starters!r}")
    datetime.strptime(f"{race_id[:4]}{announced}", "%Y%m%d%H%M")

    tansho: dict[int, float] = {}
    tan_tokens: dict[int, str] = {}
    for horse in range(1, HORSE_SLOTS + 1):
        slot = TAN_SLOT0 + (horse - 1) * TAN_STRIDE
        _assert_slot_number(record[slot : slot + 2], horse, "TAN")
        start = TAN_ODDS0 + (horse - 1) * TAN_STRIDE
        token = record[start : start + 4]
        tan_tokens[horse] = token
        value = _integer(token)
        if value and value > 0:
            tansho[horse] = value / 10.0

    fukusho: dict[int, tuple[float, float]] = {}
    fuku_tokens: dict[int, tuple[str, str]] = {}
    for horse in range(1, HORSE_SLOTS + 1):
        slot = FUK_SLOT0 + (horse - 1) * FUK_STRIDE
        _assert_slot_number(record[slot : slot + 2], horse, "FUKU")
        start = FUK_ODDS0 + (horse - 1) * FUK_STRIDE
        low_token, high_token = record[start : start + 4], record[start + 4 : start + 8]
        fuku_tokens[horse] = (low_token, high_token)
        low, high = _integer(low_token), _integer(high_token)
        if low and high and 0 < low <= high:
            fukusho[horse] = (low / 10.0, high / 10.0)

    wakuren: dict[str, float] = {}
    waku_tokens: dict[str, str] = {}
    for slot, expected in enumerate(expected_waku_pairs()):
        start = WAKU0 + slot * WAKU_STRIDE
        raw_pair = record[start : start + 2]
        if raw_pair.strip() and raw_pair != expected:
            raise ValueError(f"waku slot {slot} pair {raw_pair!r} != expected {expected!r}")
        pair = f"{expected[0]}-{expected[1]}"
        token = record[start + 2 : start + 7]
        waku_tokens[pair] = token
        value = _integer(token)
        if value and value > 0:
            wakuren[pair] = value / 10.0

    return O1Snapshot(
        division=division,
        created_yyyymmdd=record[3:11],
        race_id=race_id,
        announced_mmddhhmm=announced,
        registered_headcount=registered,
        starter_count=starters,
        sale_flags=record[slice(*HEADER_SALE_FLAGS)],
        place_pay_key=record[HEADER_PLACE_KEY],
        tansho=tansho,
        fukusho=fukusho,
        wakuren=wakuren,
        tansho_tokens=tan_tokens,
        fukusho_tokens=fuku_tokens,
        wakuren_tokens=waku_tokens,
        total_tansho_votes=_integer(record[TOTAL_TAN0 : TOTAL_TAN0 + 11]),
        total_fukusho_votes=_integer(record[TOTAL_FUK0 : TOTAL_FUK0 + 11]),
        total_wakuren_votes=_integer(record[TOTAL_WAKU0 : TOTAL_WAKU0 + 11]),
    )


def iter_o1(path: str | Path, encoding: str = "cp932") -> Iterator[O1Snapshot]:
    with Path(path).open("r", encoding=encoding, errors="strict", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield parse_o1_record(line)
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
