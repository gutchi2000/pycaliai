from datetime import datetime
from pathlib import Path

from jvlink_changes import parse_wh, parse_wh_detail
from analysis.bodyweight_forward.collector import collect_race, due_races


def _slot(ban, name, kg, sign, diff):
    return (f"{ban:02d}".encode() + name.encode("cp932").ljust(36, b" ") +
            kg.encode() + sign.encode() + diff.encode())


def _record(rid="2026092606040801"):
    head = f"WH1{'20260926'}{rid}{'00000000'}".encode()
    slots = [_slot(1, "アルビオンライズ", "488", "+", "002"),
             _slot(2, "トパーズヘッド", "484", " ", "000"),
             _slot(3, "アンミツ", "426", "-", "012"),
             _slot(4, "取消馬", "000", " ", "   "),
             _slot(5, "計測不能馬", "999", " ", "999")]
    slots += [_slot(0, "", "   ", " ", "   ")] * 13
    return (head + b"".join(slots)).decode("cp932")


def test_cp932_fixed_width_and_legacy_view():
    rows = parse_wh_detail(_record())
    assert [(r["umaban"], r["name"]) for r in rows[:3]] == [
        (1, "アルビオンライズ"), (2, "トパーズヘッド"), (3, "アンミツ")]
    assert rows[0]["weight_kg"] == 488 and rows[0]["change_kg"] == 2
    assert rows[1]["change_kg"] == 0
    assert rows[2]["change_kg"] == -12
    assert rows[3]["weight_status"] == "scratched"
    assert rows[4]["weight_status"] == "measurement_unavailable"
    assert parse_wh(_record()) == {"1": [488, "+2"], "2": [484, "±0"], "3": [426, "-12"]}


def test_append_only_and_idempotent_content(tmp_path: Path):
    fetcher = lambda rid, spec: [_record()]
    t1 = datetime.fromisoformat("2026-09-26T08:40:00+09:00")
    t2 = datetime.fromisoformat("2026-09-26T08:45:00+09:00")
    a = collect_race("2026092606040801", store=tmp_path, now=t1, fetcher=fetcher)
    b = collect_race("2026092606040801", store=tmp_path, now=t2, fetcher=fetcher)
    assert a["new_snapshot"] is True and b["new_snapshot"] is False
    assert len(list((tmp_path / "snapshots").rglob("*.json"))) == 1
    assert len(list((tmp_path / "attempts").rglob("*.json"))) == 2


def test_no_record_is_preserved_as_attempt(tmp_path: Path):
    result = collect_race("2026092606040801", store=tmp_path,
                          now=datetime.fromisoformat("2026-09-26T08:00:00+09:00"),
                          fetcher=lambda rid, spec: [])
    assert result["success"] is False and result["reason"] == "no_WH_record"
    assert len(list((tmp_path / "attempts").rglob("*.json"))) == 1


def test_due_window_is_inclusive():
    post = datetime.fromisoformat("2026-09-26T10:00:00+09:00")
    cal = [("r", post)]
    assert due_races(cal, post.replace(hour=8, minute=30))
    assert due_races(cal, post.replace(hour=9, minute=55))
    assert not due_races(cal, post.replace(hour=8, minute=29))
    assert not due_races(cal, post.replace(hour=9, minute=56))


def test_com_cp1252_style_mapping_is_reversible():
    raw = _record().encode("cp932")
    undefined = {0x81, 0x8D, 0x8F, 0x90, 0x9D}
    mapped = "".join(bytes([b]).decode("cp1252") if b not in undefined else chr(b) for b in raw)
    rows = parse_wh_detail(mapped)
    assert rows[0]["name"] == "アルビオンライズ"
    assert rows[0]["weight_kg"] == 488
