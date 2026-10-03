# -*- coding: utf-8 -*-
"""serve_health: 一致率の集計と WARN 判定。"""
import json
import serve_health as sh


def _bundle(path, n_races, n_agree):
    races = []
    for i in range(n_races):
        hs = [dict(umaban=u, p_win=0.5 - 0.05 * u, tansho_odds=2.0 + u) for u in range(1, 9)]
        if i >= n_agree:                      # 市場 1番人気を AI 2位へずらす
            hs[1]["tansho_odds"] = 1.5
        races.append(dict(race_id=f"20260101{i:08d}", horses=hs))
    path.write_text(json.dumps(dict(races=races)), encoding="utf-8")


def _setup(tmp_path, monkeypatch, agree_per_day):
    bd = tmp_path / "b"; bd.mkdir()
    for i, k in enumerate(agree_per_day):
        _bundle(bd / f"2026010{i+1}_bundle.json", 30, k)
    monkeypatch.setattr(sh, "BUNDLE_DIR", bd)
    monkeypatch.setattr(sh, "KEKKA_DIR", tmp_path / "k")
    monkeypatch.setattr(sh, "OUT", tmp_path / "out.json")
    monkeypatch.setattr("sys.argv", ["serve_health.py"])
    return tmp_path / "out.json"


def test_ok(tmp_path, monkeypatch):
    out = _setup(tmp_path, monkeypatch, [16, 16, 16, 16])
    assert sh.main() == 0
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["status"] == "OK" and rec["n_races"] == 120 and abs(rec["agree_rate"] - 16 / 30) < 1e-3


def test_warn(tmp_path, monkeypatch):
    out = _setup(tmp_path, monkeypatch, [10, 10, 10, 10])
    assert sh.main() == 2
    assert json.loads(out.read_text(encoding="utf-8"))["status"] == "WARN"


def test_insufficient_races_is_hold(tmp_path, monkeypatch):
    out = _setup(tmp_path, monkeypatch, [5, 5])
    assert sh.main() == 0 and not out.exists()
