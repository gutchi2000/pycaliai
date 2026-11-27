# -*- coding: utf-8 -*-
"""AV(出走取消) の読み取り。2026-11-28 京都11R ネーロノワールの実録 (0B14, 開催日キー)。"""
from jvlink_changes import fetch_all, parse_av
import jvlink_changes

REC = ("AV12026112820261128080507111128065509ネーロノワール"
       + "　" * 11 + "001\r\n")


def _com_mapped(s: str) -> str:
    """JV-Link を PowerShell から呼んだときの見え方 (cp932 のバイトが 1 バイト 1 文字)。"""
    out = []
    for b in s.encode("cp932"):
        try:
            out.append(bytes([b]).decode("cp1252"))
        except UnicodeDecodeError:
            out.append(chr(b))
    return "".join(out)


def test_parse_av_unicode():
    assert parse_av(REC) == {"umaban": 9, "name": "ネーロノワール", "kind": "出走取消"}


def test_parse_av_com_mapped():
    assert parse_av(_com_mapped(REC)) == {"umaban": 9, "name": "ネーロノワール", "kind": "出走取消"}


def test_fetch_all_uses_day_key_for_0b14(monkeypatch):
    calls = []

    def fake(key, spec, max_rec=200):
        calls.append((key, spec))
        return [REC] if (spec, key) == ("0B14", "20261128") else []

    monkeypatch.setattr(jvlink_changes, "fetch_records", fake)
    b = fetch_all(["2026112808050711", "2026112808050712"], date="20261128")
    assert b == {"2026112808050711": {"AV": [REC]}}
    assert ("20261128", "0B14") in calls
    assert all(spec != "0B14" for key, spec in calls if key != "20261128")
    assert all(spec != "0B16" for _, spec in calls)
