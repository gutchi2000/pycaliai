# -*- coding: utf-8 -*-
"""
race_eligibility.py — レース適格性の単一判定点（P0 hard gate の共通基盤）
========================================================================
障害競走を prediction / betting / task 登録から確実に外すための**唯一の判定関数**。

判定方針（2026-09-22 ユーザー指示）:
  - 主判定は authoritative な `トラックコード(JV) ∈ 51..59`
  - レース名・`芝・ダ` 列は**使わない**（`芝・ダ` は障害を芝/ダへ recode するため）
  - 利用できるなら `平・障` も照合し、**不一致は fail-closed**（障害として扱う）

背景（実測）: `data/weekly` の障害包含は日によって不安定で、実際に 3 レースが
本番 bundle へ入り v6 に採点・印付けされていた。買い目は偶然ゼロだったが、
コードレベルの拒否は存在しなかった。本モジュールがその単一の防波堤になる。

使い方:
    from race_eligibility import evaluate_race, assert_bettable

    el = evaluate_race("2026091306040401")
    if not el["prediction_eligible"]:
        continue                     # bundle へ入れない

    assert_bettable(rid, bets)       # 購入直前。障害 × 非空買い目なら例外
"""
from __future__ import annotations

import csv
import json
import logging
from functools import lru_cache
from pathlib import Path

BASE = Path(__file__).parent
BUNSEKI_DIR = BASE / "data" / "bunseki"
BIAS_DIR = BASE / "data" / "bias"
JUMP_STORE = BASE / "data" / "history_only" / "jump" / "raw_card"

# JRA-VAN トラックコード 51-59 = 障害（確定判定）
JUMP_TRACK_MIN, JUMP_TRACK_MAX = 51, 59

REASON_JUMP = "jump_race_excluded"
REASON_CONFLICT = "jump_detection_conflict"
REASON_FLAT = "flat_race_ok"
REASON_UNKNOWN = "jump_undetermined"

logger = logging.getLogger(__name__)


class JumpRaceBettingError(RuntimeError):
    """障害レースに非空の買い目が渡された。購入経路は必ずこれで止まる。"""


def _rid16(v) -> str:
    return str(v or "").strip()[:16]


def _date_of(rid: str) -> str | None:
    return rid[:8] if len(rid) >= 8 and rid[:8].isdigit() else None


@lru_cache(maxsize=64)
def _bunseki_track_codes(date: str) -> dict[str, int]:
    """{rid16: トラックコード(JV)}。authoritative。"""
    p = BUNSEKI_DIR / f"{date}.csv"
    if not p.exists():
        return {}
    out: dict[str, int] = {}
    try:
        with open(p, encoding="cp932", errors="replace", newline="") as f:
            for row in csv.DictReader(f):
                rid = _rid16(row.get("レースID(新)"))
                tc = str(row.get("トラックコード(JV)", "")).strip()
                if rid and tc.isdigit():
                    out[rid] = int(tc)
    except Exception as e:                                   # pragma: no cover
        logger.warning("bunseki 読込失敗 %s: %s", p.name, e)
        return {}
    return out


@lru_cache(maxsize=64)
def _bias_hira_shogai(date: str) -> dict[str, str]:
    """{rid16: 平・障}。'1' が障害。cross-check 用。"""
    p = BIAS_DIR / f"{date}.csv"
    if not p.exists():
        return {}
    out: dict[str, str] = {}
    try:
        with open(p, encoding="cp932", errors="replace", newline="") as f:
            for row in csv.DictReader(f):
                rid = _rid16(row.get("レースID"))
                v = str(row.get("平・障", "")).strip()
                if rid and v:
                    out[rid] = v
    except Exception as e:                                   # pragma: no cover
        logger.warning("bias 読込失敗 %s: %s", p.name, e)
        return {}
    return out


@lru_cache(maxsize=64)
def _collected_jump_rids(date: str) -> frozenset[str]:
    """history-only collector が既に障害と判定して保存した race_id。"""
    p = JUMP_STORE / f"{date}.jsonl"
    if not p.exists():
        return frozenset()
    rids = set()
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rids.add(_rid16(json.loads(line).get("race_id")))
    except Exception as e:                                   # pragma: no cover
        logger.warning("jump store 読込失敗 %s: %s", p.name, e)
    return frozenset(rids)


def evaluate_race(race_id, *, track_code=None) -> dict:
    """レース 1 本の適格性を返す。

    track_code を明示的に渡した場合はそれを最優先の authoritative 値として使う
    （呼び出し側が既に bunseki 由来の値を持っているケース）。
    """
    rid = _rid16(race_id)
    date = _date_of(rid)
    raw: dict = {"race_id": rid, "date": date}

    # --- 主判定: トラックコード(JV) ---
    tc = None
    if track_code is not None and str(track_code).strip() not in ("", "nan"):
        try:
            tc = int(float(track_code))
            raw["track_code_source"] = "argument"
        except (TypeError, ValueError):
            tc = None
    if tc is None and date:
        m = _bunseki_track_codes(date)
        if rid in m:
            tc = m[rid]
            raw["track_code_source"] = "data/bunseki"
    raw["track_code"] = tc
    tc_jump = (JUMP_TRACK_MIN <= tc <= JUMP_TRACK_MAX) if tc is not None else None

    # --- cross-check: 平・障 ---
    hs = _bias_hira_shogai(date).get(rid) if date else None
    raw["hira_shogai"] = hs
    hs_jump = (hs == "1") if hs is not None else None

    # --- 補強: collector が既に障害として保存済みか ---
    collected = rid in _collected_jump_rids(date) if date else False
    raw["in_jump_history_store"] = collected

    # --- 統合判定（不一致は fail-closed = 障害扱い） ---
    if tc_jump is not None and hs_jump is not None and tc_jump != hs_jump:
        is_jump, determination, reason = True, "conflict", REASON_CONFLICT
    elif tc_jump is True or hs_jump is True or collected:
        is_jump, determination, reason = True, "authoritative", REASON_JUMP
    elif tc_jump is False:
        is_jump, determination, reason = False, "authoritative", REASON_FLAT
    elif hs_jump is False:
        is_jump, determination, reason = False, "authoritative", REASON_FLAT
    else:
        # 判定材料が無い。障害だと断定できないので通すが、必ず記録する。
        is_jump, determination, reason = False, "unknown", REASON_UNKNOWN

    if is_jump:
        return {"race_id": rid, "is_jump": True, "determination": determination,
                "prediction_eligible": False, "bet_eligible": False,
                "task_registration_eligible": False,
                "history_only_eligible": True, "reason": reason,
                "raw_fields": raw}

    return {"race_id": rid, "is_jump": False, "determination": determination,
            "prediction_eligible": True, "bet_eligible": True,
            "task_registration_eligible": True,
            "history_only_eligible": False, "reason": reason,
            "raw_fields": raw}


def is_jump_race(race_id, *, track_code=None) -> bool:
    return evaluate_race(race_id, track_code=track_code)["is_jump"]


def filter_race_ids(race_ids, *, layer: str) -> tuple[list[str], list[dict]]:
    """race_id 列を (通すもの, 除外したもの) に分ける。除外は必ずログへ残す。"""
    keep, dropped = [], []
    for rid in race_ids:
        el = evaluate_race(rid)
        if el["prediction_eligible"]:
            keep.append(_rid16(rid))
        else:
            dropped.append(el)
    log_exclusions(dropped, layer=layer)
    return keep, dropped


def log_exclusions(dropped: list[dict], *, layer: str) -> None:
    if not dropped:
        return
    rids = [d["race_id"] for d in dropped]
    logger.warning("[%s] 障害レース %d 件を除外: %s", layer, len(dropped), rids)
    print(f"  ⚠ [{layer}] 障害レース {len(dropped)} 件を除外: {rids}", flush=True)


def assert_bettable(race_id, bets, *, layer: str = "final") -> None:
    """購入・送信の直前で呼ぶ最終防壁。
    上流で除外済みでも省略しない（defense in depth)。"""
    has_bets = bool(bets)
    if isinstance(bets, dict):
        has_bets = bool(bets.get("bets") or bets.get("tickets") or bets.get("bet_data"))
    if not has_bets:
        return
    el = evaluate_race(race_id)
    if el["is_jump"]:
        raise JumpRaceBettingError(
            f"[{layer}] 障害レース {el['race_id']} に非空の買い目が渡された "
            f"(reason={el['reason']}, determination={el['determination']}, "
            f"raw={el['raw_fields']})。購入経路を停止する。")


def clear_cache() -> None:
    """テスト用: ファイルを差し替えたときにキャッシュを捨てる。
    monkeypatch 済み (= lru_cache でない) 場合は何もしない。"""
    for fn in (_bunseki_track_codes, _bias_hira_shogai, _collected_jump_rids):
        clear = getattr(fn, "cache_clear", None)
        if clear is not None:
            clear()
