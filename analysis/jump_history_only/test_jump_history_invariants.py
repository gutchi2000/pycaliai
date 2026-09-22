# -*- coding: utf-8 -*-
"""
test_jump_history_invariants.py — P1 forward-only collector の hard test
=======================================================================
全 invariant を機械的に検査する。1 つでも落ちたら collector を本番接続しない。

実行:
  venv311/Scripts/python.exe -m pytest analysis/jump_history_only/test_jump_history_invariants.py -q
  または
  venv311/Scripts/python.exe analysis/jump_history_only/test_jump_history_invariants.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))

from analysis.jump_history_only.jump_history_collector import (  # noqa: E402
    CollisionError, RAW_DIR, SETTLED_DIR, MANIFEST, STORE,
    build_raw_card, build_settled, append_jsonl, read_jsonl,
    content_hash, JUMP_TRACK_MIN, JUMP_TRACK_MAX,
)

FIXTURES = [20260905, 20260906]
FLAGS = {"history_only": True, "prediction_eligible": False,
         "bet_eligible": False, "task_registration_eligible": False}


def _all_raw() -> list[dict]:
    out = []
    for p in sorted(RAW_DIR.glob("*.jsonl")):
        out.extend(read_jsonl(p))
    return out


def _all_settled() -> list[dict]:
    out = []
    for p in sorted(SETTLED_DIR.glob("*.jsonl")):
        out.extend(read_jsonl(p))
    return out


# ---------------- T1: eligibility フラグ ----------------

def test_t1_all_records_carry_history_only_flags():
    recs = _all_raw() + _all_settled()
    assert recs, "収集済みレコードが無い"
    for r in recs:
        for k, v in FLAGS.items():
            assert r.get(k) == v, f"{k} != {v} in {r.get('race_id')}"


# ---------------- T2: 通常 bundle / race list への混入 ----------------

# この collector を作る **前から** 本番 bundle に入っていた障害レース。
# 原因は data/weekly の障害除外が日によって効いたり効かなかったりすること
# (2026-09-22 発見)。collector が作ったものではない。
# 新しい混入を検出するための baseline として固定する。
KNOWN_PREEXISTING_BUNDLE_LEAKS = {
    "2026091306040401",   # 20260913 中山 R01 障害
    "2026091909040504",   # 20260919 阪神 R04 障害
    "2026092009040601",   # 20260920 中山 R01 障害
}


def _bundle_leaks() -> set[str]:
    jump_rids = {r["race_id"] for r in _all_raw()}
    leaked = set()
    for bp in sorted((BASE / "reports" / "cowork_input").glob("*_bundle.json")):
        try:
            j = json.loads(bp.read_text(encoding="utf-8"))
        except Exception:
            continue
        for race in j.get("races", []):
            rid = str(race.get("race_id") or race.get("rid") or "")[:16]
            if rid in jump_rids:
                leaked.add(rid)
    return leaked


def test_t2_no_new_jump_race_leaks_into_bundle():
    """既知の 3 件を超える**新規**混入が無いこと。
    既知分は collector 以前から存在する本番側の状態であり、
    このテストはそれを固定して新規発生を検出する。"""
    assert _all_raw(), "jump race_id が無い"
    new = _bundle_leaks() - KNOWN_PREEXISTING_BUNDLE_LEAKS
    assert not new, f"障害 race_id が新たに bundle へ混入: {sorted(new)}"


def test_t2b_known_leaks_carried_no_actual_bets():
    """既知混入 3 件で実際に買い目が発注されていないこと
    (発注されていたら除外レースの誤購入になる)。"""
    bad = []
    for op in sorted((BASE / "reports" / "cowork_output").glob("*_bets.json")):
        try:
            j = json.loads(op.read_text(encoding="utf-8"))
        except Exception:
            continue
        races = j.get("races", []) if isinstance(j, dict) else j
        for r in (races if isinstance(races, list) else []):
            if not isinstance(r, dict):
                continue
            rid = str(r.get("race_id", ""))[:16]
            if rid in KNOWN_PREEXISTING_BUNDLE_LEAKS and r.get("bets"):
                bad.append((op.name, rid, r.get("bets")))
    assert not bad, f"障害レースに買い目が付いている: {bad}"


def test_t2c_collector_output_never_marked_eligible():
    """collector 自身の出力が予測/購入/タスク登録の対象にならないこと。"""
    for r in _all_raw() + _all_settled():
        assert r["history_only"] is True
        assert r["prediction_eligible"] is False
        assert r["bet_eligible"] is False
        assert r["task_registration_eligible"] is False


# ---------------- T3: scheduler タスク非登録 ----------------

def test_t3_no_scheduler_task_for_jump_races():
    """T-35/T-20/T-10/Vote のタスク名は race_id を含む
    (例 PyCaLiAI_T10R_2026092206040710)。障害 race_id のタスクが無いこと。"""
    import subprocess
    jump_rids = {r["race_id"] for r in _all_raw()}
    try:
        out = subprocess.run(["schtasks", "/query", "/fo", "LIST"],
                             capture_output=True, text=True, timeout=60,
                             errors="replace").stdout
    except Exception as e:
        pytest.skip(f"schtasks を実行できない: {e}")
    hits = [rid for rid in jump_rids if rid in out]
    assert not hits, f"障害 race_id のスケジューラタスクが存在: {hits}"


# ---------------- T4: 決定的な join / 馬名 join 禁止 ----------------

def test_t4_join_is_deterministic_and_nameless():
    raw = _all_raw()
    assert raw
    for r in raw:
        assert r["race_id"] and len(r["race_id"]) == 16
        assert r["ped_id"], "血統登録番号が空 (主キー欠損)"
        assert r["umaban"] is not None
    keys = [(r["race_id"], r["ped_id"]) for r in raw]
    assert len(keys) == len(set(keys)), "race_id+ped_id が一意でない"
    keys2 = [(r["race_id"], r["umaban"]) for r in raw]
    assert len(keys2) == len(set(keys2)), "race_id+馬番 が一意でない"
    # 馬名フィールドを保持していないこと (join に使えないようにする)
    for r in raw:
        assert "horse_name" not in r and "馬名" not in r, "馬名を保持している"


def test_t4b_collector_source_has_no_name_join():
    """コメント・方針文の「馬名」は許容し、**コードとしての馬名参照**だけを禁じる。"""
    src_lines = (Path(__file__).resolve().parent /
                 "jump_history_collector.py").read_text(encoding="utf-8").splitlines()
    forbidden = ['"馬名"', "'馬名'", "horse_name", "_clean_name",
                 'on="馬名"', "merge(.*馬名"]
    hits = []
    for i, line in enumerate(src_lines, 1):
        code = line.split("#", 1)[0]
        # docstring/方針文字列の行を除くため、代入・参照の形だけを見る
        for pat in forbidden:
            if pat in code:
                hits.append((i, line.strip()))
    assert not hits, f"collector が馬名をコード上で参照している: {hits}"


# ---------------- T5: 結果確定前は settled へ入れない ----------------

def test_t5_settled_only_when_result_exists():
    for p in sorted(SETTLED_DIR.glob("*.jsonl")):
        d = int(p.stem)
        assert (BASE / "data" / "kekka" / f"{d}.csv").exists(), \
            f"{d}: kekka が無いのに settled がある"
    for r in _all_settled():
        assert r.get("result_available_at"), "result_available_at が無い"
        assert r.get("result_source_sha256"), "result_source_sha256 が無い"


def test_t5b_dates_without_kekka_have_no_settled():
    for p in sorted(RAW_DIR.glob("*.jsonl")):
        d = int(p.stem)
        if not (BASE / "data" / "kekka" / f"{d}.csv").exists():
            assert not (SETTLED_DIR / f"{d}.jsonl").exists(), \
                f"{d}: 結果未確定なのに settled が存在する"


# ---------------- T6: future race を履歴へ入れない ----------------

def test_t6_no_future_race_in_settled():
    today = int(pd.Timestamp.now().strftime("%Y%m%d"))
    for r in _all_settled():
        assert r["race_date"] <= today, f"未来日の settled: {r['race_date']}"


# ---------------- T7: raw card を結果で上書きしない ----------------

def test_t7_raw_card_carries_no_result_fields():
    forbidden = {"finish_code_raw", "completed", "dnf", "scratched",
                 "started", "result_available_at", "settled_at"}
    for r in _all_raw():
        bad = forbidden & set(r.keys())
        assert not bad, f"raw card が結果項目を持っている: {bad}"


def test_t7b_layers_are_separate_files():
    assert RAW_DIR.resolve() != SETTLED_DIR.resolve()
    for p in RAW_DIR.glob("*.jsonl"):
        assert not p.resolve().is_relative_to(SETTLED_DIR.resolve())


# ---------------- T8: idempotency / collision ----------------

def test_t8_reprocessing_same_source_is_idempotent(tmp_path):
    raw, _ = build_raw_card(FIXTURES[0])
    assert raw
    p = tmp_path / "raw.jsonl"
    r1 = append_jsonl(p, raw, ("race_id", "ped_id"), ("captured_at",), dry=False)
    r2 = append_jsonl(p, raw, ("race_id", "ped_id"), ("captured_at",), dry=False)
    assert r1["added"] == len(raw)
    assert r2["added"] == 0 and r2["skipped_idempotent"] == len(raw)
    assert r2["total_after"] == len(raw)


def test_t9_content_change_raises_collision(tmp_path):
    raw, _ = build_raw_card(FIXTURES[0])
    p = tmp_path / "raw.jsonl"
    append_jsonl(p, raw, ("race_id", "ped_id"), ("captured_at",), dry=False)
    mutated = [dict(r) for r in raw]
    mutated[0]["distance"] = (mutated[0]["distance"] or 0) + 100
    with pytest.raises(CollisionError):
        append_jsonl(p, mutated, ("race_id", "ped_id"), ("captured_at",),
                     dry=False)
    # 衝突時は 1 行も書かれていないこと
    assert len(read_jsonl(p)) == len(raw)


def test_t9b_volatile_field_change_is_not_a_collision(tmp_path):
    raw, _ = build_raw_card(FIXTURES[0])
    p = tmp_path / "raw.jsonl"
    append_jsonl(p, raw, ("race_id", "ped_id"), ("captured_at",), dry=False)
    later = [dict(r, captured_at="2099-01-01T00:00:00+09:00") for r in raw]
    r = append_jsonl(p, later, ("race_id", "ped_id"), ("captured_at",),
                     dry=False)
    assert r["added"] == 0, "captured_at の違いで衝突してはいけない"


# ---------------- T10: jump のみ / track_code ----------------

def test_t10_only_jump_races_collected():
    for r in _all_raw():
        assert r["jump_flag"] is True
        assert JUMP_TRACK_MIN <= r["track_code"] <= JUMP_TRACK_MAX


# ---------------- T11: schema / provenance ----------------

def test_t11_schema_and_provenance_present():
    for r in _all_raw():
        assert r["schema_version"].startswith("jump-history-only/raw_card/")
        assert r["captured_at"] and r["source_file"] and r["source_sha256"]
        assert len(r["source_sha256"]) == 64
    for r in _all_settled():
        assert r["schema_version"].startswith("jump-history-only/settled/")
        assert len(r["result_source_sha256"]) == 64


def test_t11b_source_sha_matches_actual_file():
    for r in _all_raw():
        p = BASE / r["source_file"]
        assert p.exists(), f"source_file が無い: {p}"
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for c in iter(lambda: f.read(1 << 20), b""):
                h.update(c)
        assert h.hexdigest() == r["source_sha256"], \
            f"source_sha256 不一致: {r['source_file']}"


# ---------------- T12: 原本不変 ----------------

def test_t12_collector_never_writes_to_source_dirs():
    src = (Path(__file__).resolve().parent /
           "jump_history_collector.py").read_text(encoding="utf-8")
    # 書込は STORE 配下のみ。BUNSEKI/KEKKA への書込 API を呼ばないこと
    for pat in ["BUNSEKI /", "KEKKA /"]:
        for line in src.splitlines():
            if pat in line and any(w in line for w in
                                   ["to_csv", "open(", "write", "unlink",
                                    "replace", "mkdir"]):
                # 読み取り用の open( は許容する
                assert "encoding=\"cp932\"" in line or "sha256_file" in line, \
                    f"原本ディレクトリへの書込の疑い: {line.strip()}"


def test_t12b_store_is_outside_production_inputs():
    for d in (RAW_DIR, SETTLED_DIR):
        assert not d.resolve().is_relative_to((BASE / "data" / "weekly").resolve())
        assert not d.resolve().is_relative_to((BASE / "data" / "bunseki").resolve())
        assert not d.resolve().is_relative_to((BASE / "data" / "kekka").resolve())


def test_t12c_no_production_code_reads_the_store():
    """production 側が誤ってこのストアを読み始めていないこと。"""
    hits = []
    for p in BASE.glob("*.py"):
        try:
            t = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if "history_only/jump" in t or "history_only\\jump" in t:
            hits.append(p.name)
    assert not hits, f"production スクリプトがストアを参照: {hits}"


# ---------------- T13: manifest / coverage ----------------

def test_t13_manifest_records_coverage_start_and_gap():
    assert MANIFEST.exists(), "manifest.json が無い"
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert m.get("jump_history_coverage_start"), "coverage_start が無い"
    gap = m.get("known_unrecoverable_gap") or {}
    assert gap.get("jump_races") == 47 and gap.get("rows") == 582, \
        "復元不能 gap が固定されていない"
    assert "推測補完しない" in gap.get("policy", "")


# ---------------- T14: fixture ----------------

@pytest.mark.parametrize("d", FIXTURES)
def test_t14_fixture_days_complete(d):
    raw, _ = build_raw_card(d)
    assert raw, f"{d}: 障害レースの raw card が作れない"
    settled, _ = build_settled(d, raw)
    assert settled, f"{d}: settled が作れない"
    assert len(settled) == len(raw)
    assert all(s["started"] or s["scratched"] for s in settled)


if __name__ == "__main__":
    sys.exit(pytest.main([str(Path(__file__)), "-q"]))
