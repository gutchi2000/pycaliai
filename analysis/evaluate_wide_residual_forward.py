"""Evaluate the preregistered v3 wide-residual shadow cohort.

The candidate population comes *only* from immutable ``*_shadow.json`` files.
Results are joined afterwards and are used only for canonical settlement.  A
race is admitted to Arm A, M1 and M2 together, or to none of them.

Ordinary TARGET result exports do not contain authoritative wide payouts, so
the production loader combines ``data/kekka/{date}.csv`` with
``build_site.parse_wide_kekka()``.  Ambiguous results, incomplete wide payout
maps, invalid candidates, and price-lineage failures are never converted to a
zero return.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from canonical_settlement import (  # noqa: E402
    SettlementError,
    canonical_combo,
    normalize_result,
    settle_bets,
)
from forward_prices import canonical_stage  # noqa: E402


DEFAULT_POLICY = BASE / "data" / "shadow_policies" / "wide_residual_shadow_v3.json"
DEFAULT_SHADOW_DIR = BASE / "reports" / "wide_residual_shadow"
DEFAULT_RESULT_DIR = BASE / "data" / "kekka"
DEFAULT_CHANGES_DIR = BASE / "reports" / "live_changes"
DEFAULT_FORWARD_ROOT = BASE / "data" / "forward_prices"
DEFAULT_OUTPUT = BASE / "reports" / "wide_residual_forward_eval.json"

ARMS = ("arm_a", "control_m1", "control_m2")
REQUIRED_PARENT_ARTIFACTS = frozenset(
    {"rank_model", "serve_calibrator", "serve_feature_baseline"})
RESULT_COLUMNS = (
    "date", "place", "race_no", "frame", "ban", "horse", "jyun",
    "rid_horse", "tansho", "fukusho", "wakuren", "umaren", "umatan",
    "sanrenpuku", "sanrentan",
)
_OUTCOME_KEYS = frozenset({
    "won", "hit", "payout", "return", "total_back", "actual", "result",
    "確定着順", "実払戻額", "的中",
})


ResultLoader = Callable[[str], Mapping[str, Any]]
WideMapLoader = Callable[[], Mapping[Any, Mapping[Any, Any]]]
LineageLoader = Callable[[str], Mapping[str, Any]]


class FormalLookLockError(RuntimeError):
    """An existing formal look cannot safely be replaced or re-evaluated."""


def _decode(raw: bytes) -> str:
    for encoding in ("cp932", "utf-8-sig", "utf-8"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")


def _rid16(value: Any) -> str:
    return re.sub(r"\D", "", str(value or ""))[:16]


def _integer(value: Any) -> int | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not number.is_integer():
        return None
    return int(number)


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _round(value: float | None, digits: int = 4) -> float | None:
    return round(value, digits) if value is not None and math.isfinite(value) else None


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentile(values: Iterable[float], probability: float) -> float | None:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def load_policy(path: Path = DEFAULT_POLICY) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("shadow policy root must be an object")
    required = {
        "policy_id", "effective_from", "status", "real_money_enabled",
        "candidate", "data_quality_checkpoint", "futility_checkpoint",
        "formal_look", "required_stress_reporting",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"shadow policy missing fields: {missing}")
    if value["status"] != "shadow_only" or value["real_money_enabled"] is not False:
        raise ValueError("forward evaluator accepts shadow-only policies")
    return value


def load_raw_results(date: str, result_dir: Path = DEFAULT_RESULT_DIR) -> dict[str, dict]:
    """Load a TARGET daily result export, assigning semantic names by position.

    The first fifteen fields match the payout meanings used by
    ``backtest_pl_ev.load_payouts``; the leading date/race metadata fields are
    retained so that the authoritative wide map can be addressed exactly.
    """

    path = Path(result_dir) / f"{date}.csv"
    if not path.exists():
        return {}
    reader = csv.reader(io.StringIO(_decode(path.read_bytes())))
    next(reader, None)
    races: dict[str, dict] = {}
    for raw in reader:
        if len(raw) < len(RESULT_COLUMNS):
            continue
        named = dict(zip(RESULT_COLUMNS, raw[:len(RESULT_COLUMNS)]))
        rid = _rid16(named["rid_horse"])
        ban, jyun = _integer(named["ban"]), _integer(named["jyun"])
        race_no = _integer(named["race_no"])
        if len(rid) != 16 or ban is None or jyun is None or race_no is None:
            continue
        row = dict(named)
        row.update({
            "馬番": ban,
            "確定着順": jyun,
            "単勝配当": named["tansho"],
            "複勝配当": named["fukusho"],
            "馬連配当": named["umaren"],
            "馬単配当": named["umatan"],
            "三連複配当": named["sanrenpuku"],
            "三連単配当": named["sanrentan"],
        })
        record = races.setdefault(rid, {
            "rows": [],
            "place": str(named["place"]).strip(),
            "race_no": race_no,
            "race_status": "confirmed",
            # The ordinary export cannot distinguish a refund from a DNF.
            # ``classify_zero_positions`` fills these from the JV-Link change
            # feed, falling back to a conservative all-started assumption.
            "refunded_horses": [],
            "non_refunded_zero_horses": [],
            "zero_classification": "none",
        })
        record["rows"].append(row)
    classify_zero_positions(date, races)
    return races


def load_cancelled_horses(date: str,
                          changes_dir: Path = DEFAULT_CHANGES_DIR) -> dict[str, set[int]]:
    """JV-Link 当日変更フィードから {rid16: {取消・除外の馬番}} を読む。

    ``t10.ps1`` が ``changes.ps1`` 経由で ``reports/live_changes/{date}.json`` を
    書く。存在しない開催日は空 dict を返し、呼び出し側が保守的側へ倒す。
    """
    path = Path(changes_dir) / f"{date}.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: dict[str, set[int]] = {}
    for rid, entry in (payload.get("races") or {}).items():
        bans: set[int] = set()
        for cancel in (entry or {}).get("cancels") or []:
            ban = _integer((cancel or {}).get("umaban"))
            if ban is not None:
                bans.add(ban)
        if bans:
            out[_rid16(rid)] = bans
    return out


def classify_zero_positions(date: str, races: dict[str, dict],
                            changes_dir: Path = DEFAULT_CHANGES_DIR) -> None:
    """着順0の馬を「返還(取消・除外)」と「出走して着順なし」へ振り分ける。

    canonical settler は未分類の着順0が1頭でもあると全marketを unresolved にする。
    権威ソース (JV-Link AV レコード) があればそれを使い、無ければ **全て出走扱い**
    に倒す。出走扱いは券を返還でなく外れとして決済するため ROI を過小評価する側
    であり、昇格判定を緩める方向には決して働かない。
    """
    cancels = load_cancelled_horses(date, changes_dir)
    for rid, record in races.items():
        zero: list[int] = []
        for row in record["rows"]:
            jyun = _integer(row.get("確定着順"))
            ban = _integer(row.get("馬番"))
            if ban is not None and (jyun is None or jyun == 0):
                zero.append(ban)
        if not zero:
            record["zero_classification"] = "no_zero_position"
            continue
        refunded = sorted(set(zero) & cancels.get(rid, set()))
        record["refunded_horses"] = refunded
        record["non_refunded_zero_horses"] = sorted(set(zero) - set(refunded))
        record["zero_classification"] = (
            "jv_change_feed" if rid in cancels else "conservative_all_started")


def load_authoritative_wide_map() -> Mapping[Any, Mapping[Any, Any]]:
    # Import locally: build_site has substantial module-level application data.
    from build_site import parse_wide_kekka
    return parse_wide_kekka()


def load_price_lineage(date: str, forward_root: Path = DEFAULT_FORWARD_ROOT) -> dict[str, dict]:
    """Return decision/T-10 hashes and close coverage from append-only ledgers."""

    by_race: dict[str, dict] = defaultdict(
        lambda: {"decision_hashes": set(), "decisions": [],
                 "t10_hashes": set(), "has_close": False})
    directory = Path(forward_root) / date
    if not directory.exists():
        return {}
    for path in sorted(directory.glob("*.json.gz")):
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                record = json.load(handle)
        except (OSError, ValueError, gzip.BadGzipFile):
            continue
        rid = _rid16(record.get("race_id"))
        if len(rid) != 16:
            continue
        market_hash = str(record.get("market_sha256") or "")
        if record.get("record_type") == "decision_snapshot" and market_hash:
            by_race[rid]["decision_hashes"].add(market_hash)
            by_race[rid]["decisions"].append({
                "market_sha256": market_hash,
                "policy": record.get("policy"),
            })
        elif record.get("record_type") == "market_snapshot":
            # schema v2 (観測計画 v2.1): 旧 `close` は `close_late` として保存される。v1 の `close` 録も
            # canonical_stage() で同じ `close_late` に解決して読む。
            stage = canonical_stage(record.get("stage"))
            if stage == "t10" and market_hash:
                by_race[rid]["t10_hashes"].add(market_hash)
            elif stage == "close_late":
                by_race[rid]["has_close"] = True
    return dict(by_race)


def _date_from_path(path: Path) -> str | None:
    match = re.search(r"(?<!\d)(20\d{6})(?!\d)", path.name)
    return match.group(1) if match else None


def _wide_key(date: str, record: Mapping[str, Any]) -> tuple[int, int, int, str, int] | None:
    race_no = _integer(record.get("race_no"))
    place = str(record.get("place") or "").strip()
    if len(date) != 8 or race_no is None or not place:
        return None
    return int(date[:4]), int(date[4:6]), int(date[6:8]), place, race_no


def _valid_sha256(value: Any) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", str(value or "")))


def _validate_parent_policy(value: Any) -> tuple[dict | None, str | None]:
    """Validate the complete production policy/artifact stamp frozen at T-10."""

    if not isinstance(value, dict):
        return None, "parent_policy is missing or is not an object"
    required = {
        "policy_id", "policy_schema_version", "policy_sha256",
        "chaos_reference_id", "chaos_reference_sha256",
        "chaos_skip_percentile", "chaos_skip_raw_equivalent", "artifact_sha256",
    }
    missing = sorted(required - set(value))
    if missing:
        return None, f"parent_policy required field(s) missing: {missing}"
    if not value.get("policy_id") or not value.get("chaos_reference_id"):
        return None, "parent_policy policy/reference id is empty"
    for key in ("policy_sha256", "chaos_reference_sha256"):
        if not _valid_sha256(value.get(key)):
            return None, f"parent_policy {key} is not a SHA-256 digest"
    artifacts = value.get("artifact_sha256")
    if not isinstance(artifacts, dict):
        return None, "parent_policy artifact_sha256 is not an object"
    missing_artifacts = sorted(REQUIRED_PARENT_ARTIFACTS - set(artifacts))
    if missing_artifacts:
        return None, f"parent_policy artifact hash(es) missing: {missing_artifacts}"
    invalid_artifacts = sorted(
        name for name, digest in artifacts.items() if not _valid_sha256(digest))
    if invalid_artifacts:
        return None, f"parent_policy invalid artifact hash(es): {invalid_artifacts}"
    return dict(value), None


def _lineage_status(lineage: Mapping[str, Any] | None, market_hash: str,
                    parent_policy: Mapping[str, Any]) -> dict[str, Any]:
    lineage = lineage or {}
    decisions = {str(value) for value in (lineage.get("decision_hashes") or ())}
    t10s = {str(value) for value in (lineage.get("t10_hashes") or ())}
    decision_match = bool(market_hash and market_hash in decisions)
    t10_match = bool(market_hash and market_hash in t10s)
    matching_decisions = [
        record for record in (lineage.get("decisions") or ())
        if isinstance(record, Mapping)
        and str(record.get("market_sha256") or "") == market_hash
    ]
    parent_policy_match = any(
        record.get("policy") == parent_policy for record in matching_decisions)
    return {
        "decision_match": decision_match,
        "t10_match": t10_match,
        "hash_match": decision_match and t10_match,
        "parent_policy_match": parent_policy_match,
        "full_lineage_match": decision_match and t10_match and parent_policy_match,
        "has_close": bool(lineage.get("has_close")),
    }


def _validate_candidate(row: Any, arm: str, policy: Mapping[str, Any]) -> tuple[dict | None, str | None]:
    if not isinstance(row, dict):
        return None, f"{arm}: candidate is not an object"
    forbidden = sorted(_OUTCOME_KEYS & set(row))
    if forbidden:
        return None, f"{arm}: outcome field(s) present: {forbidden}"
    try:
        combo = canonical_combo("wide", row.get("selection"))
    except SettlementError as exc:
        return None, f"{arm}: {exc}"
    stake = _integer(row.get("virtual_stake_yen"))
    expected_stake = int(policy["candidate"]["stake_per_ticket_yen"])
    if stake != expected_stake or stake <= 0 or stake % 100:
        return None, f"{arm}: invalid virtual stake {stake!r}"
    required_numbers = ("p_model", "p_market_fair", "residual",
                        "odds_t10_low", "odds_t10_high", "odds_t10_mid")
    parsed = {name: _number(row.get(name)) for name in required_numbers}
    if any(value is None for value in parsed.values()):
        return None, f"{arm}: non-finite candidate input"
    if parsed["odds_t10_low"] <= 1 or parsed["odds_t10_high"] < parsed["odds_t10_low"]:
        return None, f"{arm}: invalid T-10 odds range"
    if arm == "arm_a":
        candidate = policy["candidate"]
        if not (float(candidate["residual_lower_inclusive"]) <= parsed["residual"]
                < float(candidate["residual_upper_exclusive"])):
            return None, f"{arm}: residual outside preregistered band"
    return {
        "selection": f"{combo[0]}-{combo[1]}",
        "combo": combo,
        "stake": stake,
        **parsed,
    }, None


def _validate_triggered_race(race: Any, policy: Mapping[str, Any], date: str) -> tuple[dict | None, str | None]:
    if not isinstance(race, dict):
        return None, "race entry is not an object"
    if _OUTCOME_KEYS & set(race):
        return None, "race entry contains outcome field"
    rid = _rid16(race.get("race_id"))
    if len(rid) != 16 or rid[:8] != date:
        return None, f"invalid/date-mismatched race_id {rid!r}"
    if race.get("triggered") is not True:
        return None, "not_triggered"
    if race.get("real_money_enabled") is not False:
        return None, "real_money_enabled must be false"
    if race.get("hard_gate_passed") is not True:
        return None, "triggered race failed hard gate"
    parent_policy, parent_error = _validate_parent_policy(race.get("parent_policy"))
    if parent_error:
        return None, parent_error
    expected_parent = policy.get("parent_production_policy")
    if parent_policy["policy_id"] != expected_parent:
        return None, (f"parent_policy id mismatch: {parent_policy['policy_id']!r} "
                      f"!= {expected_parent!r}")

    arms: dict[str, list[dict]] = {}
    for arm in ARMS:
        raw_rows = race.get(arm)
        if not isinstance(raw_rows, list) or not raw_rows:
            return None, f"{arm}: missing candidates"
        rows: list[dict] = []
        seen: set[tuple[int, int]] = set()
        for raw in raw_rows:
            parsed, error = _validate_candidate(raw, arm, policy)
            if error:
                return None, error
            assert parsed is not None
            if parsed["combo"] in seen:
                return None, f"{arm}: duplicate selection {parsed['selection']}"
            seen.add(parsed["combo"])
            rows.append(parsed)
        arms[arm] = rows

    maximum = int(policy["candidate"]["max_tickets_per_race"])
    top_n = int(policy["candidate"]["take_model_top_n_before_filter"])
    if len(arms["arm_a"]) > maximum:
        return None, "arm_a exceeds max tickets"
    if len(arms["control_m1"]) != top_n or len(arms["control_m2"]) != top_n:
        return None, "controls do not contain preregistered top-N"
    if not {row["combo"] for row in arms["arm_a"]}.issubset(
            {row["combo"] for row in arms["control_m1"]}):
        return None, "arm_a is not a subset of model-top control"
    return {
        "date": date,
        "race_id": rid,
        "observed_at": race.get("observed_at"),
        "market_sha256": str(race.get("market_sha256") or ""),
        "arms": arms,
        "parent_policy": parent_policy,
    }, None


def _coerce_result_record(value: Any) -> dict | None:
    if isinstance(value, list):
        return {"rows": value, "race_status": "confirmed",
                "refunded_horses": [], "non_refunded_zero_horses": []}
    if isinstance(value, dict) and isinstance(value.get("rows"), list):
        return value
    return None


def _settle_race(candidate: Mapping[str, Any], result_record: Mapping[str, Any],
                 wide_payouts: Mapping[Any, Any]) -> tuple[dict | None, str | None]:
    try:
        normalized = normalize_result(
            result_record["rows"],
            refunded_horses=result_record.get("refunded_horses") or (),
            non_refunded_zero_horses=result_record.get("non_refunded_zero_horses") or (),
            race_status=str(result_record.get("race_status") or "confirmed"),
            payout_maps={"wide": wide_payouts},
        )
    except (SettlementError, TypeError, ValueError) as exc:
        return None, f"normalize_error: {exc}"

    arms: dict[str, dict] = {}
    for arm in ARMS:
        bets = {row["combo"]: row["stake"] for row in candidate["arms"][arm]}
        try:
            settled = settle_bets("wide", bets, normalized)
        except (SettlementError, TypeError, ValueError) as exc:
            return None, f"{arm}_settle_error: {exc}"
        if not settled["ok"] or settled["total_back"] is None:
            issues = "; ".join(str(value) for value in settled.get("issues") or ())
            return None, f"{arm}_unresolved: {issues or settled.get('market_status')}"
        tickets = []
        for candidate_row in candidate["arms"][arm]:
            detail = settled["detail"].get(candidate_row["combo"])
            if not detail or detail.get("back") is None:
                return None, f"{arm}_missing_ticket_detail"
            tickets.append({
                "selection": candidate_row["selection"],
                "stake": candidate_row["stake"],
                "back": int(detail["back"]),
                "state": detail["state"],
                "odds_t10_low": candidate_row["odds_t10_low"],
            })
        arms[arm] = {
            "stake": int(settled["stake"]),
            "back": int(settled["total_back"]),
            "n_hit": int(settled["n_hit"]),
            "tickets": tickets,
        }
    return {
        "date": candidate["date"],
        "race_id": candidate["race_id"],
        "arms": arms,
    }, None


def _arm_summary(rows: list[dict], arm: str) -> dict[str, Any]:
    tickets = [ticket for row in rows for ticket in row["arms"][arm]["tickets"]]
    stake = sum(ticket["stake"] for ticket in tickets)
    back = sum(ticket["back"] for ticket in tickets)
    return {
        "races": len(rows),
        "tickets": len(tickets),
        "hits": sum(ticket["state"] == "hit" for ticket in tickets),
        "stake_yen": stake,
        "return_yen": back,
        "net_yen": back - stake,
        "roi_pct": _round(back / stake * 100.0 if stake else None),
    }


def _bootstrap(rows: list[dict], repetitions: int, seed: int) -> dict[str, Any]:
    by_day: dict[str, dict[str, list[int]]] = {}
    for date in sorted({row["date"] for row in rows}):
        day_rows = [row for row in rows if row["date"] == date]
        by_day[date] = {
            arm: [sum(row["arms"][arm][key] for row in day_rows) for key in ("stake", "back")]
            for arm in ARMS
        }
    days = sorted(by_day)
    if not days or repetitions <= 0:
        empty_arm = {"ci95_pct": [None, None], "one_sided_95_upper_pct": None}
        return {
            "blocks": "race_day", "repetitions": repetitions, "seed": seed,
            "arms": {arm: dict(empty_arm) for arm in ARMS},
            "delta_vs_primary": {
                "arm_a_minus_control_m1": {"ci95_pt": [None, None],
                                            "one_sided_95_upper_pt": None},
                "arm_a_minus_control_m2": {"ci95_pt": [None, None],
                                            "one_sided_95_upper_pt": None},
            },
        }
    rng = random.Random(seed)
    samples: dict[str, list[float]] = {arm: [] for arm in ARMS}
    deltas_m1: list[float] = []
    deltas_m2: list[float] = []
    for _ in range(repetitions):
        chosen = [days[rng.randrange(len(days))] for _ in days]
        rois: dict[str, float] = {}
        for arm in ARMS:
            stake = sum(by_day[day][arm][0] for day in chosen)
            back = sum(by_day[day][arm][1] for day in chosen)
            rois[arm] = back / stake * 100.0
            samples[arm].append(rois[arm])
        deltas_m1.append(rois["arm_a"] - rois["control_m1"])
        deltas_m2.append(rois["arm_a"] - rois["control_m2"])

    def interval(values: list[float], *, suffix: str) -> dict[str, Any]:
        return {
            f"ci95_{suffix}": [_round(_percentile(values, .025)),
                               _round(_percentile(values, .975))],
            f"one_sided_95_upper_{suffix}": _round(_percentile(values, .95)),
        }

    return {
        "blocks": "race_day", "repetitions": repetitions, "seed": seed,
        "arms": {arm: interval(values, suffix="pct") for arm, values in samples.items()},
        "delta_vs_primary": {
            "arm_a_minus_control_m1": interval(deltas_m1, suffix="pt"),
            "arm_a_minus_control_m2": interval(deltas_m2, suffix="pt"),
        },
    }


def _ticket_metrics(tickets: list[dict]) -> dict[str, Any]:
    stake = sum(int(ticket["stake"]) for ticket in tickets)
    back = sum(int(ticket["back"]) for ticket in tickets)
    return {
        "tickets": len(tickets), "stake_yen": stake, "return_yen": back,
        "net_yen": back - stake,
        "roi_pct": _round(back / stake * 100.0 if stake else None),
    }


def _stress(rows: list[dict], policy: Mapping[str, Any]) -> dict[str, Any]:
    requested = policy["required_stress_reporting"]
    all_tickets: dict[str, list[dict]] = {
        arm: [dict(ticket, date=row["date"], race_id=row["race_id"])
              for row in rows for ticket in row["arms"][arm]["tickets"]]
        for arm in ARMS
    }
    top_removed: dict[str, dict[str, Any]] = {}
    for arm, tickets in all_tickets.items():
        ranked = sorted(range(len(tickets)), key=lambda index: tickets[index]["back"], reverse=True)
        arm_out: dict[str, Any] = {}
        for count in requested["remove_top_payout_counts"]:
            removed = set(ranked[:int(count)])
            stressed = [dict(ticket, back=0 if index in removed else ticket["back"])
                        for index, ticket in enumerate(tickets)]
            arm_out[str(count)] = _ticket_metrics(stressed)
        top_removed[arm] = arm_out

    odds_bands: dict[str, list[dict]] = {}
    for arm, tickets in all_tickets.items():
        bands = []
        for lower, upper in requested["odds_low_bands"]:
            selected = [ticket for ticket in tickets
                        if float(lower) <= ticket["odds_t10_low"]
                        and (upper is None or ticket["odds_t10_low"] < float(upper))]
            bands.append({"lower_inclusive": lower, "upper_exclusive": upper,
                          **_ticket_metrics(selected)})
        odds_bands[arm] = bands

    monthly: dict[str, list[dict]] = {}
    months = sorted({row["date"][:6] for row in rows})
    for arm, tickets in all_tickets.items():
        monthly[arm] = [
            {"month": month, **_ticket_metrics(
                [ticket for ticket in tickets if ticket["date"].startswith(month)])}
            for month in months
        ]

    days = sorted({row["date"] for row in rows})
    cut = (len(days) + 1) // 2
    halves = {"first_half": set(days[:cut]), "second_half": set(days[cut:])}
    time_halves: dict[str, dict[str, Any]] = {}
    for arm, tickets in all_tickets.items():
        time_halves[arm] = {
            name: _ticket_metrics([ticket for ticket in tickets if ticket["date"] in selected])
            for name, selected in halves.items()
        }

    lodo: dict[str, list[dict]] = {}
    for arm, tickets in all_tickets.items():
        lodo[arm] = [
            {"omitted_date": day, **_ticket_metrics(
                [ticket for ticket in tickets if ticket["date"] != day])}
            for day in days
        ]
    return {
        "remove_top_payouts": top_removed,
        "odds_low_bands": odds_bands,
        "monthly": monthly,
        "time_halves": time_halves,
        "leave_one_race_day_out": lodo,
    }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
                       encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def evaluate(
    *,
    shadow_dir: Path = DEFAULT_SHADOW_DIR,
    policy_path: Path = DEFAULT_POLICY,
    result_dir: Path = DEFAULT_RESULT_DIR,
    forward_root: Path = DEFAULT_FORWARD_ROOT,
    result_loader: ResultLoader | None = None,
    wide_map_loader: WideMapLoader | None = None,
    lineage_loader: LineageLoader | None = None,
    bootstrap_repetitions: int | None = None,
) -> dict[str, Any]:
    """Evaluate all v3 shadow files and return a JSON-serializable report."""

    policy_path = Path(policy_path)
    policy = load_policy(policy_path)
    expected_stamp = {
        "policy_id": policy["policy_id"],
        "policy_sha256": _file_sha256(policy_path),
        "parent_production_policy": policy.get("parent_production_policy"),
        "status": policy["status"],
        "real_money_enabled": False,
    }
    result_loader = result_loader or (lambda date: load_raw_results(date, result_dir))
    wide_map_loader = wide_map_loader or load_authoritative_wide_map
    lineage_loader = lineage_loader or (lambda date: load_price_lineage(date, forward_root))
    wide_map = wide_map_loader()

    diagnostics: Counter[str] = Counter()
    errors: list[dict[str, str]] = []
    candidates: list[dict] = []
    lineage_by_race: dict[str, dict] = {}
    result_cache: dict[str, Mapping[str, Any]] = {}
    lineage_cache: dict[str, Mapping[str, Any]] = {}
    cohort_parent_policy: dict[str, Any] | None = None
    files = sorted(Path(shadow_dir).glob("*_shadow.json"))
    effective = str(policy["effective_from"])
    for path in files:
        date = _date_from_path(path)
        if date is None or date < effective:
            continue
        diagnostics["files_seen"] += 1
        try:
            root = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            diagnostics["invalid_file"] += 1
            errors.append({"scope": path.name, "reason": f"read_error: {exc}"})
            continue
        if not isinstance(root, dict) or root.get("policy") != expected_stamp:
            diagnostics["policy_mismatch_file"] += 1
            errors.append({"scope": path.name, "reason": "root policy stamp mismatch"})
            continue
        if date not in result_cache:
            result_cache[date] = result_loader(date) or {}
            lineage_cache[date] = lineage_loader(date) or {}
        for raw_race in root.get("races") or []:
            diagnostics["races_seen"] += 1
            if not isinstance(raw_race, dict) or raw_race.get("policy") != expected_stamp:
                diagnostics["policy_mismatch_race"] += 1
                errors.append({"scope": str(getattr(raw_race, "get", lambda *_: "?")("race_id")),
                               "reason": "race policy stamp mismatch"})
                continue
            if raw_race.get("triggered") is not True:
                diagnostics["not_triggered"] += 1
                continue
            candidate, error = _validate_triggered_race(raw_race, policy, date)
            if error:
                diagnostics["invalid_candidate"] += 1
                errors.append({"scope": str(raw_race.get("race_id")), "reason": error})
                continue
            assert candidate is not None
            if cohort_parent_policy is None:
                cohort_parent_policy = candidate["parent_policy"]
            elif candidate["parent_policy"] != cohort_parent_policy:
                diagnostics["parent_policy_cohort_mismatch"] += 1
                diagnostics["invalid_candidate"] += 1
                errors.append({"scope": candidate["race_id"],
                               "reason": "parent_policy differs within cohort"})
                continue
            diagnostics["triggered_valid"] += 1
            lineage = _lineage_status(
                lineage_cache[date].get(candidate["race_id"]), candidate["market_sha256"],
                candidate["parent_policy"])
            lineage_by_race[candidate["race_id"]] = lineage
            candidate["lineage"] = lineage
            candidates.append(candidate)

    manifest = [{
        "date": row["date"], "race_id": row["race_id"],
        "observed_at": row["observed_at"], "market_sha256": row["market_sha256"],
        "parent_policy": row["parent_policy"],
        **{arm: [{key: candidate[key] for key in (
            "selection", "stake", "p_model", "p_market_fair", "residual",
            "odds_t10_low", "odds_t10_high", "odds_t10_mid")}
                 for candidate in row["arms"][arm]] for arm in ARMS},
    } for row in candidates]
    manifest_hash = hashlib.sha256(_canonical_json(manifest)).hexdigest()

    settled_rows: list[dict] = []
    for candidate in candidates:
        date, rid = candidate["date"], candidate["race_id"]
        result_record = _coerce_result_record(result_cache.get(date, {}).get(rid))
        if result_record is None:
            diagnostics["result_missing"] += 1
            continue
        wide_key = _wide_key(date, result_record)
        payouts = result_record.get("wide_payouts")
        if payouts is None and wide_key is not None:
            payouts = wide_map.get(wide_key)
        if not isinstance(payouts, Mapping):
            diagnostics["wide_payout_missing"] += 1
            errors.append({"scope": rid, "reason": "authoritative wide payout missing"})
            continue
        settled, error = _settle_race(candidate, result_record, payouts)
        if error:
            diagnostics["settlement_unresolved"] += 1
            errors.append({"scope": rid, "reason": error})
            continue
        assert settled is not None
        settled["lineage"] = candidate["lineage"]
        settled_rows.append(settled)
        diagnostics["settled_races"] += 1

    summaries = {arm: _arm_summary(settled_rows, arm) for arm in ARMS}
    settled_days = sorted({row["date"] for row in settled_rows})
    repetitions = (int(bootstrap_repetitions) if bootstrap_repetitions is not None
                   else int(policy["formal_look"]["bootstrap_repetitions"]))
    bootstrap = _bootstrap(settled_rows, repetitions,
                           int(policy["formal_look"]["bootstrap_seed"]))
    stress = _stress(settled_rows, policy)

    settled_lineage = [row["lineage"] for row in settled_rows]
    hash_matches = sum(row["hash_match"] for row in settled_lineage)
    parent_matches = sum(row["parent_policy_match"] for row in settled_lineage)
    full_matches = sum(row["full_lineage_match"] for row in settled_lineage)
    close_matches = sum(row["has_close"] for row in settled_lineage)
    denominator = len(settled_lineage)
    lineage_summary = {
        "settled_races": denominator,
        "decision_market_hash_matches": hash_matches,
        "decision_market_hash_match_pct": _round(hash_matches / denominator * 100.0
                                                  if denominator else None),
        "parent_policy_matches": parent_matches,
        "parent_policy_match_pct": _round(parent_matches / denominator * 100.0
                                           if denominator else None),
        "full_lineage_matches": full_matches,
        "full_lineage_match_pct": _round(full_matches / denominator * 100.0
                                          if denominator else None),
        "close_price_races": close_matches,
        "close_price_coverage_pct": _round(close_matches / denominator * 100.0
                                            if denominator else None),
        "triggered_valid_races": len(candidates),
        "triggered_hash_matches": sum(row["lineage"]["hash_match"] for row in candidates),
        "cohort_parent_policy_consistent": diagnostics["parent_policy_cohort_mismatch"] == 0,
        "cohort_parent_policy_sha256": (
            hashlib.sha256(_canonical_json(cohort_parent_policy)).hexdigest()
            if cohort_parent_policy is not None else None),
    }

    arm_tickets = summaries["arm_a"]["tickets"]
    n_days = len(settled_days)
    quality_cfg = policy["data_quality_checkpoint"]
    futility_cfg = policy["futility_checkpoint"]
    formal_cfg = policy["formal_look"]
    quality_reached = (arm_tickets >= int(quality_cfg["minimum_settled_tickets"])
                       and n_days >= int(quality_cfg["minimum_distinct_race_days"]))
    futility_reached = arm_tickets >= int(futility_cfg["minimum_settled_tickets"])
    absolute_upper = bootstrap["arms"]["arm_a"]["one_sided_95_upper_pct"]
    delta_upper = bootstrap["delta_vs_primary"]["arm_a_minus_control_m1"][
        "one_sided_95_upper_pt"]
    absolute_futile = (futility_reached and absolute_upper is not None
                       and absolute_upper <= float(
                           futility_cfg["stop_if_absolute_roi_one_sided_95_upper_lte_pct"]))
    delta_futile = (futility_reached and delta_upper is not None
                    and delta_upper <= float(
                        futility_cfg["stop_if_delta_vs_primary_one_sided_95_upper_lte_pt"]))
    formal_reached = (arm_tickets >= int(formal_cfg["minimum_settled_tickets"])
                      and n_days >= int(formal_cfg["minimum_distinct_race_days"]))
    absolute_lower = bootstrap["arms"]["arm_a"]["ci95_pct"][0]
    delta_lower = bootstrap["delta_vs_primary"]["arm_a_minus_control_m1"]["ci95_pt"][0]
    top3_roi = stress["remove_top_payouts"]["arm_a"].get("3", {}).get("roi_pct")
    formal_conditions = {
        "absolute_roi_ci95_lower": (absolute_lower is not None and absolute_lower
                                     > float(formal_cfg["absolute_roi_ci95_lower_gt_pct"])),
        "paired_delta_vs_primary_ci95_lower": (
            delta_lower is not None and delta_lower
            > float(formal_cfg["paired_delta_vs_primary_ci95_lower_gt_pt"])),
        "roi_after_removing_top_3_payouts": (
            top3_roi is not None and top3_roi
            > float(formal_cfg["roi_after_removing_top_3_payouts_gt_pct"])),
        "decision_market_hash_match": (
            lineage_summary["decision_market_hash_match_pct"] is not None
            and lineage_summary["decision_market_hash_match_pct"]
            >= float(formal_cfg["decision_market_hash_match_pct"])),
        "parent_policy_artifact_hash_match": (
            lineage_summary["full_lineage_match_pct"] is not None
            and lineage_summary["full_lineage_match_pct"] >= 100.0
            and lineage_summary["cohort_parent_policy_consistent"]),
        "close_price_coverage": (
            lineage_summary["close_price_coverage_pct"] is not None
            and lineage_summary["close_price_coverage_pct"]
            >= float(formal_cfg["close_price_coverage_gte_pct"])),
        "no_policy_or_candidate_integrity_errors": not (
            diagnostics["policy_mismatch_file"] or diagnostics["policy_mismatch_race"]
            or diagnostics["invalid_candidate"]),
    }
    promotion = bool(formal_reached and not absolute_futile and not delta_futile
                     and all(formal_conditions.values()))
    if formal_reached:
        state = "formal_pass" if promotion else "formal_fail"
    elif futility_reached and (absolute_futile or delta_futile):
        state = "stop_futility"
    elif futility_reached:
        state = "collecting_formal"
    elif quality_reached:
        state = "quality_reached_expected_inconclusive"
    else:
        state = "collecting_quality"

    return {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "policy_id": policy["policy_id"],
        "policy_sha256": expected_stamp["policy_sha256"],
        "semantics": {
            "candidate_source": "frozen v3 shadow JSON only; outcomes never select candidates",
            "settlement": "canonical fail-closed; all three arms share the same settled races",
            "wide_payout_source": "build_site.parse_wide_kekka authoritative map",
            "close_price": "coverage/market reference only; not an execution price",
        },
        "candidate_population": {
            "triggered_valid_races": len(candidates),
            "manifest_sha256": manifest_hash,
            "tickets": {arm: sum(len(row["arms"][arm]) for row in candidates) for arm in ARMS},
        },
        "quality": {**dict(sorted(diagnostics.items())), "errors": errors},
        "same_population": {
            "enforced": True,
            "settled_races": len(settled_rows),
            "distinct_race_days": n_days,
            "race_counts_by_arm": {arm: summaries[arm]["races"] for arm in ARMS},
        },
        "arms": summaries,
        "bootstrap": bootstrap,
        "lineage": lineage_summary,
        "stress": stress,
        "checkpoints": {
            "state": state,
            "quality": {
                "reached": quality_reached,
                "minimum_settled_tickets": int(quality_cfg["minimum_settled_tickets"]),
                "minimum_distinct_race_days": int(quality_cfg["minimum_distinct_race_days"]),
                "promotion_allowed": False,
                "expected_result": quality_cfg.get("expected_result"),
            },
            "futility": {
                "reached": futility_reached,
                "absolute_futile": absolute_futile,
                "delta_vs_primary_futile": delta_futile,
                "stop": bool(absolute_futile or delta_futile),
                "promotion_allowed": False,
            },
            "formal": {
                "reached": formal_reached,
                "minimum_settled_tickets": int(formal_cfg["minimum_settled_tickets"]),
                "minimum_distinct_race_days": int(formal_cfg["minimum_distinct_race_days"]),
                "conditions": formal_conditions,
                "promotion": promotion,
            },
        },
        # This top-level flag is deliberately false before the single formal look.
        "promotion": promotion,
    }


def run_and_write(
    *,
    out: Path = DEFAULT_OUTPUT,
    shadow_dir: Path = DEFAULT_SHADOW_DIR,
    policy_path: Path = DEFAULT_POLICY,
    result_dir: Path = DEFAULT_RESULT_DIR,
    forward_root: Path = DEFAULT_FORWARD_ROOT,
    result_loader: ResultLoader | None = None,
    wide_map_loader: WideMapLoader | None = None,
    lineage_loader: LineageLoader | None = None,
    bootstrap_repetitions: int | None = None,
) -> tuple[dict[str, Any], bool]:
    """Evaluate atomically, or return the already locked single formal look.

    The boolean return value is true only when an existing formal report was
    returned without re-running ``evaluate``.  A formal report from another
    policy is never overwritten.  A valid collecting report may be refreshed.
    """

    out = Path(out)
    policy_path = Path(policy_path)
    policy = load_policy(policy_path)
    formal_looks = _integer(policy["formal_look"].get("number_of_formal_looks"))
    if formal_looks != 1:
        raise FormalLookLockError(
            f"only number_of_formal_looks=1 is supported, got {formal_looks!r}")
    expected_policy_sha256 = _file_sha256(policy_path)
    if out.exists():
        try:
            existing = json.loads(out.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise FormalLookLockError(
                f"existing report is unreadable; refusing overwrite: {out}: {exc}") from exc
        if not isinstance(existing, dict):
            raise FormalLookLockError(
                f"existing report root is not an object; refusing overwrite: {out}")
        formal = ((existing.get("checkpoints") or {}).get("formal") or {})
        reached = formal.get("reached")
        if not isinstance(reached, bool):
            raise FormalLookLockError(
                f"existing report has no boolean formal.reached; refusing overwrite: {out}")
        if reached is True:
            existing_sha256 = existing.get("policy_sha256")
            if existing_sha256 != expected_policy_sha256:
                raise FormalLookLockError(
                    "existing formal report belongs to another policy; refusing second look: "
                    f"{existing_sha256!r} != {expected_policy_sha256!r}")
            manifest = (existing.get("candidate_population") or {}).get("manifest_sha256")
            if not _valid_sha256(manifest):
                raise FormalLookLockError(
                    "existing formal report has no valid candidate manifest; refusing overwrite")
            return existing, True

    report = evaluate(
        shadow_dir=shadow_dir, policy_path=policy_path,
        result_dir=result_dir, forward_root=forward_root,
        result_loader=result_loader, wide_map_loader=wide_map_loader,
        lineage_loader=lineage_loader,
        bootstrap_repetitions=bootstrap_repetitions,
    )
    _atomic_write_json(out, report)
    return report, False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-dir", type=Path, default=DEFAULT_SHADOW_DIR)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--forward-root", type=Path, default=DEFAULT_FORWARD_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report, locked = run_and_write(
        out=args.out, shadow_dir=args.shadow_dir, policy_path=args.policy,
        result_dir=args.result_dir, forward_root=args.forward_root)
    print(json.dumps({
        "out": str(args.out),
        "state": report["checkpoints"]["state"],
        "arm_a_tickets": report["arms"]["arm_a"]["tickets"],
        "promotion": report["promotion"],
        "formal_locked": locked,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
