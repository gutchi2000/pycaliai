# -*- coding: utf-8 -*-
"""
race_eligibility.py — レース適格性の単一判定点（P0 hard gate の共通基盤）
========================================================================
障害競走を prediction / betting / task 登録から確実に外すための**唯一の判定関数**。

判定方針（2026-09-22 ユーザー指示）:
  - 主判定は authoritative な `トラックコード(JV) ∈ 51..59`
  - レース名・`芝・ダ` 列は**使わない**（`芝・ダ` は障害を芝/ダへ recode するため）
  - 利用できるなら `平・障` も照合し、**不一致は fail-closed**（障害として扱う）

★★ 最重要の hard rule（2026-09-22 追加）★★
  `predict_weekly.parse_csv` は週次 CSV に `トラックコード(JV)` 列が無いとき
  **中央値 23（芝・内）で埋める**。この 23 を JV コードとして扱うと
  **障害レースが平地として通る**。したがって:

    - `track_code_source == "default"` は**判定根拠に使わない**
    - 値が 23 でも source が default なら **unknown**
    - **unknown は prediction / bet / task すべて ineligible**（fail-closed）
    - unknown を通常平地として黙って通さない

  モデル入力用の欠損埋め 23 と、eligibility 判定用の JV track code は
  **完全に別物**として扱う。

使い方:
    from race_eligibility import evaluate_race, assert_bettable

    el = evaluate_race("2026091306040401")
    if not el["prediction_eligible"]:
        continue                     # bundle へ入れない

    assert_bettable(rid, bets)       # 購入直前。障害/unknown × 非空買い目なら例外
"""
from __future__ import annotations

import csv
import hashlib
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

# predict_weekly._MISSING_FEATURE_MEDIANS の埋め値。**証拠にしてはいけない**
DEFAULT_TRACK_CODE_FILL = 23

ELIGIBILITY_SCHEMA_VERSION = "race-eligibility/1"

# provenance ラベル
SRC_RAW_JV = "raw_jv"                 # JV-Link 生レコード由来
SRC_BUNSEKI = "bunseki"               # data/bunseki（TARGET 出走馬分析）
SRC_CALENDAR = "calendar"             # 開催カレンダー由来
SRC_WEEKLY_EXPLICIT = "weekly_explicit"   # 週次 CSV に実在した値
SRC_DEFAULT = "default"               # 欠損埋め。**証拠にしない**
SRC_MISSING = "missing"               # 値そのものが無い

AUTHORITATIVE_SOURCES = {SRC_RAW_JV, SRC_BUNSEKI, SRC_WEEKLY_EXPLICIT, SRC_CALENDAR}

REASON_JUMP = "jump_race_excluded"
REASON_CONFLICT = "jump_detection_conflict"
REASON_FLAT = "flat_race_ok"
REASON_UNKNOWN = "jump_undetermined"

logger = logging.getLogger(__name__)


class JumpRaceBettingError(RuntimeError):
    """障害 or 判定不能なレースに非空の買い目が渡された。購入経路はこれで止まる。"""


class EligibilityMetadataError(RuntimeError):
    """保存された eligibility metadata が欠落・不整合。fail-closed。"""


def _rid16(v) -> str:
    return str(v or "").strip()[:16]


def _date_of(rid: str) -> str | None:
    return rid[:8] if len(rid) >= 8 and rid[:8].isdigit() else None


def _sha256_file(p: Path) -> str | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@lru_cache(maxsize=64)
def _bunseki_track_codes(date: str) -> tuple[dict, str | None]:
    """({rid16: トラックコード(JV)}, source_sha256)。authoritative。"""
    p = BUNSEKI_DIR / f"{date}.csv"
    if not p.exists():
        return {}, None
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
        return {}, None
    return out, _sha256_file(p)


@lru_cache(maxsize=64)
def _bias_hira_shogai(date: str) -> tuple[dict, str | None]:
    """({rid16: 平・障}, source_sha256)。'1' が障害。cross-check 用。"""
    p = BIAS_DIR / f"{date}.csv"
    if not p.exists():
        return {}, None
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
        return {}, None
    return out, _sha256_file(p)


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


def evaluate_race(race_id, *, track_code=None,
                  track_code_source: str | None = None) -> dict:
    """レース 1 本の適格性を返す。

    track_code を渡す場合は **track_code_source を必ず明示**すること。
    source を省略した場合、その値は証拠として採用しない（安全側）。
    `track_code_source="default"` は常に証拠にしない。
    """
    rid = _rid16(race_id)
    date = _date_of(rid)

    ev: dict = {
        "race_id": rid, "date": date,
        "track_code_value": None, "track_code_source": SRC_MISSING,
        "flat_jump_value": None, "flat_jump_source": SRC_MISSING,
        "raw_presence": {"track_code_present": False, "flat_jump_present": False},
        "source_hashes": {},
        "in_jump_history_store": False,
    }

    # --- 呼び出し側から渡された値（source が明示され、default でない場合のみ採用） ---
    tc = None
    if track_code is not None and str(track_code).strip() not in ("", "nan", "None"):
        ev["raw_presence"]["track_code_present"] = True
        try:
            v = int(float(track_code))
        except (TypeError, ValueError):
            v = None
        if v is not None:
            ev["track_code_value"] = v
            if track_code_source == SRC_DEFAULT:
                ev["track_code_source"] = SRC_DEFAULT      # 証拠にしない
            elif track_code_source in AUTHORITATIVE_SOURCES:
                ev["track_code_source"] = track_code_source
                tc = v
            else:
                # source 不明の値は採用しない（安全側）
                ev["track_code_source"] = track_code_source or "unspecified"

    # --- bunseki（authoritative）。上で採用できていなければこちらを使う ---
    if tc is None and date:
        m, sha = _bunseki_track_codes(date)
        if rid in m:
            tc = m[rid]
            ev["track_code_value"] = tc
            ev["track_code_source"] = SRC_BUNSEKI
            ev["raw_presence"]["track_code_present"] = True
            ev["source_hashes"]["bunseki"] = sha

    tc_jump = (JUMP_TRACK_MIN <= tc <= JUMP_TRACK_MAX) if tc is not None else None

    # --- cross-check: 平・障 ---
    hs_jump = None
    if date:
        hm, hsha = _bias_hira_shogai(date)
        if rid in hm:
            ev["flat_jump_value"] = hm[rid]
            ev["flat_jump_source"] = SRC_BUNSEKI if False else "bias"
            ev["raw_presence"]["flat_jump_present"] = True
            ev["source_hashes"]["bias"] = hsha
            hs_jump = (hm[rid] == "1")

    collected = rid in _collected_jump_rids(date) if date else False
    ev["in_jump_history_store"] = collected

    # --- 統合判定 ---
    if tc_jump is not None and hs_jump is not None and tc_jump != hs_jump:
        is_jump, determination, reason = True, "conflict", REASON_CONFLICT
    elif tc_jump is True or hs_jump is True or collected:
        is_jump, determination, reason = True, "authoritative", REASON_JUMP
    elif tc_jump is False or hs_jump is False:
        is_jump, determination, reason = False, "authoritative", REASON_FLAT
    else:
        # ★ 判定材料が無い（または default 埋めしか無い）→ fail-closed
        is_jump, determination, reason = None, "unknown", REASON_UNKNOWN

    if determination == "unknown":
        return {"race_id": rid, "is_jump": None, "determination": "unknown",
                "prediction_eligible": False, "bet_eligible": False,
                "task_registration_eligible": False,
                "history_only_eligible": True, "reason": reason,
                "eligibility_schema_version": ELIGIBILITY_SCHEMA_VERSION,
                "raw_fields": ev}

    if is_jump:
        return {"race_id": rid, "is_jump": True, "determination": determination,
                "prediction_eligible": False, "bet_eligible": False,
                "task_registration_eligible": False,
                "history_only_eligible": True, "reason": reason,
                "eligibility_schema_version": ELIGIBILITY_SCHEMA_VERSION,
                "raw_fields": ev}

    return {"race_id": rid, "is_jump": False, "determination": determination,
            "prediction_eligible": True, "bet_eligible": True,
            "task_registration_eligible": True,
            "history_only_eligible": False, "reason": reason,
            "eligibility_schema_version": ELIGIBILITY_SCHEMA_VERSION,
            "raw_fields": ev}


def is_jump_race(race_id, **kw) -> bool:
    """True = 障害と確定。unknown は False を返さず例外にしないが、
    eligibility 判定には evaluate_race を使うこと。"""
    return evaluate_race(race_id, **kw)["is_jump"] is True


def eligibility_metadata(el: dict) -> dict:
    """bundle 等へ保存する metadata（boolean だけを信用させないための証拠付き）。"""
    ev = el["raw_fields"]
    return {
        "eligibility_schema_version": el["eligibility_schema_version"],
        "race_id": el["race_id"],
        "computed": {
            "is_jump": el["is_jump"],
            "prediction_eligible": el["prediction_eligible"],
            "bet_eligible": el["bet_eligible"],
            "task_registration_eligible": el["task_registration_eligible"],
            "history_only_eligible": el["history_only_eligible"],
            "determination": el["determination"],
            "reason": el["reason"],
        },
        "evidence": {
            "track_code_value": ev["track_code_value"],
            "track_code_source": ev["track_code_source"],
            "flat_jump_value": ev["flat_jump_value"],
            "flat_jump_source": ev["flat_jump_source"],
            "raw_presence": ev["raw_presence"],
            "in_jump_history_store": ev["in_jump_history_store"],
        },
        "source_hashes": ev["source_hashes"],
    }


def verify_metadata(meta: dict | None, race_id) -> dict:
    """下流での検証。boolean を鵜呑みにせず共通関数で再計算して突き合わせる。
    metadata 欠落・schema 不一致・source hash 不一致はすべて fail-closed。"""
    rid = _rid16(race_id)
    fresh = evaluate_race(rid)
    if meta is None:
        # metadata が無い場合も、再計算結果で判断する（黙って通さない）
        if not fresh["prediction_eligible"]:
            raise EligibilityMetadataError(
                f"{rid}: eligibility metadata 欠落 かつ 再計算が "
                f"{fresh['reason']} → fail-closed")
        return fresh
    if meta.get("eligibility_schema_version") != ELIGIBILITY_SCHEMA_VERSION:
        raise EligibilityMetadataError(
            f"{rid}: eligibility schema 不一致 "
            f"({meta.get('eligibility_schema_version')}) → fail-closed")
    saved_hashes = meta.get("source_hashes") or {}
    now_hashes = fresh["raw_fields"]["source_hashes"]
    for k, v in saved_hashes.items():
        if k in now_hashes and now_hashes[k] and v and now_hashes[k] != v:
            raise EligibilityMetadataError(
                f"{rid}: source hash 不一致 ({k}) → fail-closed")
    return fresh


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
    jump = [d for d in dropped if d.get("is_jump") is True]
    unk = [d for d in dropped if d.get("determination") == "unknown"]
    if jump:
        rids = [d["race_id"] for d in jump]
        logger.warning("[%s] 障害レース %d 件を除外: %s", layer, len(jump), rids)
        print(f"  ⚠ [{layer}] 障害レース {len(jump)} 件を除外: {rids}", flush=True)
    if unk:
        rids = [d["race_id"] for d in unk]
        logger.error("[%s] 判定不能(unknown) %d 件を fail-closed で除外: %s "
                     "→ data/bunseki/{date}.csv を配置すること",
                     layer, len(unk), rids)
        print(f"  ✗ [{layer}] 判定不能 {len(unk)} 件を fail-closed で除外: "
              f"{rids}  (bunseki 未配置の疑い)", flush=True)


def assert_bettable(race_id, bets, *, layer: str = "final",
                    metadata: dict | None = None) -> None:
    """購入・送信の直前で呼ぶ最終防壁。
    上流で除外済みでも省略しない（defense in depth）。
    障害だけでなく **unknown も拒否**する。"""
    has_bets = bool(bets)
    if isinstance(bets, dict):
        has_bets = bool(bets.get("bets") or bets.get("tickets") or bets.get("bet_data"))
    if not has_bets:
        return
    el = verify_metadata(metadata, race_id) if metadata is not None \
        else evaluate_race(race_id)
    if not el["bet_eligible"]:
        raise JumpRaceBettingError(
            f"[{layer}] {el['race_id']} に非空の買い目が渡された "
            f"(is_jump={el['is_jump']}, determination={el['determination']}, "
            f"reason={el['reason']}, "
            f"track_code={el['raw_fields']['track_code_value']}"
            f"/{el['raw_fields']['track_code_source']})。購入経路を停止する。")


def clear_cache() -> None:
    """テスト用: ファイルを差し替えたときにキャッシュを捨てる。"""
    for fn in (_bunseki_track_codes, _bias_hira_shogai, _collected_jump_rids):
        clear = getattr(fn, "cache_clear", None)
        if clear is not None:
            clear()
