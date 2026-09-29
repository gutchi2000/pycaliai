# -*- coding: utf-8 -*-
"""Canonical, fail-closed settlement for JRA horse-racing tickets.

This module deliberately separates two operations:

``normalize_result``
    Converts horse-level result rows into an explicit winning-combination to
    payout map for every supported ticket type.

``settle_bets``
    Settles stakes against one normalized market result while retaining hits,
    losses, refunds, and unresolved tickets as separate states.

The result CSV's ``確定着順 == 0`` is *not* a reliable cancellation flag: it
also contains horses which started but did not finish.  Therefore every
zero-position horse must be classified explicitly as either refunded or
non-refunded.  An unclassified zero makes normalization fail closed.

Payouts are expressed as yen returned per 100 yen stake (stake included), as
published by JRA.  Unordered ticket combinations are stored in sorted order;
ordered ticket combinations preserve finishing order.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, permutations
import math
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence


Combo = tuple[int, ...]

BET_ALIASES: dict[str, str] = {
    "win": "win",
    "単勝": "win",
    "place": "place",
    "複勝": "place",
    "wide": "wide",
    "ワイド": "wide",
    "quinella": "quinella",
    "umaren": "quinella",
    "馬連": "quinella",
    "exacta": "exacta",
    "umatan": "exacta",
    "馬単": "exacta",
    "trio": "trio",
    "sanpuku": "trio",
    "三連複": "trio",
    "３連複": "trio",
    "trifecta": "trifecta",
    "sanrentan": "trifecta",
    "三連単": "trifecta",
    "３連単": "trifecta",
}

BET_TYPES = ("win", "place", "wide", "quinella", "exacta", "trio", "trifecta")
UNORDERED_TYPES = frozenset({"wide", "quinella", "trio"})
ARITY = {
    "win": 1,
    "place": 1,
    "wide": 2,
    "quinella": 2,
    "exacta": 2,
    "trio": 3,
    "trifecta": 3,
}

_PAYOUT_COLUMNS: dict[str, tuple[str, ...]] = {
    "win": ("単勝配当", "単勝払戻", "win_payout"),
    "place": ("複勝配当", "複勝払戻", "place_payout"),
    # The ordinary TARGET result export has no wide payout column.  Supply an
    # explicit payout map from wide_kekka or another authoritative source.
    "wide": ("ワイド配当", "ワイド払戻", "wide_payout"),
    "quinella": ("馬連", "馬連配当", "馬連払戻", "quinella_payout"),
    "exacta": ("馬単", "馬単配当", "馬単払戻", "exacta_payout"),
    "trio": ("３連複", "三連複", "三連複配当", "三連複払戻", "trio_payout"),
    "trifecta": ("３連単", "三連単", "三連単配当", "三連単払戻", "trifecta_payout"),
}
_HORSE_COLUMNS = ("馬番", "umaban", "horse_no", "horse_number")
_POSITION_COLUMNS = ("確定着順", "chaku", "position", "finish_position")


class SettlementError(ValueError):
    """Raised when an input violates the canonical settlement contract."""


@dataclass(frozen=True)
class MarketResult:
    """Normalized result for one ticket type.

    ``status`` is one of ``confirmed``, ``void``, or ``unresolved``.  A caller
    must never turn ``unresolved`` into a zero return.
    """

    bet_type: str
    status: str
    winning_combos: tuple[Combo, ...] = ()
    payout_per100: Mapping[Combo, int] = field(default_factory=dict)
    issues: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "payout_per100", MappingProxyType(dict(self.payout_per100)))

    @property
    def ok(self) -> bool:
        return self.status in {"confirmed", "void"}


@dataclass(frozen=True)
class NormalizedRaceResult:
    """All normalized markets plus explicit refund information for one race."""

    status: str
    markets: Mapping[str, MarketResult]
    refunded_horses: frozenset[int] = frozenset()
    non_refunded_zero_horses: frozenset[int] = frozenset()
    zero_position_horses: frozenset[int] = frozenset()
    n_starters: int = 0
    issues: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "markets", MappingProxyType(dict(self.markets)))

    @property
    def ok(self) -> bool:
        return self.status in {"confirmed", "void"} and all(m.ok for m in self.markets.values())

    def market(self, bet_type: str) -> MarketResult:
        return self.markets[canonical_bet_type(bet_type)]


def canonical_bet_type(bet_type: str) -> str:
    """Return the stable English identifier for a Japanese/English alias."""

    try:
        return BET_ALIASES[str(bet_type).strip()]
    except KeyError as exc:
        raise SettlementError(f"unsupported bet type: {bet_type!r}") from exc


def canonical_combo(bet_type: str, combo: Any) -> Combo:
    """Validate and canonicalize one ticket combination.

    Strings may use ``-``, ``>``, commas, or whitespace as separators.
    """

    kind = canonical_bet_type(bet_type)
    if isinstance(combo, int):
        values = (combo,)
    elif isinstance(combo, str):
        parts = [p for p in re.split(r"[-,>\s]+", combo.strip()) if p]
        try:
            values = tuple(int(p) for p in parts)
        except ValueError as exc:
            raise SettlementError(f"invalid {kind} combo: {combo!r}") from exc
    else:
        try:
            values = tuple(int(v) for v in combo)
        except (TypeError, ValueError) as exc:
            raise SettlementError(f"invalid {kind} combo: {combo!r}") from exc

    if len(values) != ARITY[kind]:
        raise SettlementError(f"{kind} requires {ARITY[kind]} horse(s), got {values!r}")
    if any(v <= 0 for v in values) or len(set(values)) != len(values):
        raise SettlementError(f"invalid horse numbers in {kind} combo: {values!r}")
    return tuple(sorted(values)) if kind in UNORDERED_TYPES else values


def _first_present(row: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return None


def _to_int(value: Any, *, label: str) -> int:
    if value is None or isinstance(value, bool):
        raise SettlementError(f"missing {label}")
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise SettlementError(f"invalid {label}: {value!r}") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise SettlementError(f"invalid {label}: {value!r}")
    return int(number)


def _parse_payout(value: Any) -> int | None:
    """Parse a published payout while rejecting parenthesized final odds."""

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.startswith("("):
        return None
    text = text.replace(",", "").replace("¥", "").replace("￥", "").replace("\\", "")
    match = re.fullmatch(r"\s*(\d+(?:\.0+)?)\s*", text)
    if not match:
        return None
    value_f = float(match.group(1))
    if not math.isfinite(value_f) or value_f <= 0 or not value_f.is_integer():
        return None
    return int(value_f)


def _finish_sequences(positions: Mapping[int, int], slots: int) -> tuple[Combo, ...]:
    """Enumerate all ordered finish sequences compatible with dead heats."""

    eligible = [horse for horse, pos in positions.items() if 0 < pos <= slots]
    if len(eligible) < slots:
        return ()
    out: set[Combo] = set()
    for seq in permutations(eligible, slots):
        valid = True
        for i in range(slots):
            if positions[seq[i]] > i + 1:
                valid = False
                break
            for j in range(i + 1, slots):
                if positions[seq[i]] > positions[seq[j]]:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            out.add(tuple(seq))
    return tuple(sorted(out))


def _winning_combos(kind: str, positions: Mapping[int, int], n_starters: int) -> tuple[Combo, ...]:
    if kind == "win":
        return tuple((h,) for h in sorted(h for h, pos in positions.items() if pos == 1))

    place_slots = 3 if n_starters >= 8 else 2
    if kind == "place":
        return tuple((h,) for h in sorted(h for h, pos in positions.items() if 0 < pos <= place_slots))
    if kind == "wide":
        # ワイドは複勝と違い、7頭以下でも3着までの組を払う。
        # 実測 (data/kekka/wide_kekka.csv 2026): 5頭3R・6頭12R・7頭26R すべて3組。
        # place_slots(=2) を流用すると 1組しか当たりにならず unresolved になる。
        placed = sorted(h for h, pos in positions.items() if 0 < pos <= 3)
        # 3着同着で「3着以内」が4頭以上になる場合、同着3着どうしの組は払われない
        # (実測: 16頭立てで5組 = C(4,2)-1)。
        tied_third = {h for h in placed if positions[h] == 3}
        overflow = len(placed) > 3
        return tuple(
            combo for combo in combinations(placed, 2)
            if not (overflow and combo[0] in tied_third and combo[1] in tied_third)
        )

    slots = 2 if kind in {"quinella", "exacta"} else 3
    sequences = _finish_sequences(positions, slots)
    if kind in UNORDERED_TYPES:
        return tuple(sorted({tuple(sorted(seq)) for seq in sequences}))
    return sequences


def _row_payout(row: Mapping[str, Any], kind: str) -> int | None:
    nested = row.get("payouts")
    if isinstance(nested, Mapping):
        for alias, canonical in BET_ALIASES.items():
            if canonical == kind and alias in nested:
                parsed = _parse_payout(nested[alias])
                if parsed is not None:
                    return parsed
    return _parse_payout(_first_present(row, _PAYOUT_COLUMNS[kind]))


def _explicit_map(
    kind: str,
    raw_map: Mapping[Any, Any] | None,
    winners: tuple[Combo, ...],
) -> tuple[dict[Combo, int], list[str]] | None:
    if raw_map is None:
        return None
    parsed: dict[Combo, int] = {}
    issues: list[str] = []
    for raw_combo, raw_payout in raw_map.items():
        try:
            combo = canonical_combo(kind, raw_combo)
        except SettlementError as exc:
            issues.append(str(exc))
            continue
        payout = _parse_payout(raw_payout)
        if payout is None:
            issues.append(f"invalid explicit payout for {kind} {combo}: {raw_payout!r}")
            continue
        if combo in parsed and parsed[combo] != payout:
            issues.append(f"conflicting explicit payouts for {kind} {combo}")
        parsed[combo] = payout
    expected = set(winners)
    supplied = set(parsed)
    if supplied - expected:
        issues.append(f"non-winning explicit {kind} combo(s): {sorted(supplied - expected)}")
    if expected - supplied:
        issues.append(f"missing explicit {kind} payout(s): {sorted(expected - supplied)}")
    return parsed, issues


def _infer_payouts(
    kind: str,
    winners: tuple[Combo, ...],
    rows_by_horse: Mapping[int, Mapping[str, Any]],
) -> tuple[dict[Combo, int], list[str]]:
    """Infer only mappings that are unambiguous from horse-level rows."""

    payouts: dict[Combo, int] = {}
    issues: list[str] = []
    if not winners:
        return payouts, [f"cannot derive winning {kind} combination"]

    if kind in {"win", "place"}:
        for combo in winners:
            payout = _row_payout(rows_by_horse[combo[0]], kind)
            if payout is None:
                issues.append(f"missing {kind} payout for {combo}")
            else:
                payouts[combo] = payout
        return payouts, issues

    if len(winners) == 1:
        values = {
            payout
            for horse in set(winners[0])
            if (payout := _row_payout(rows_by_horse[horse], kind)) is not None
        }
        if len(values) == 1:
            payouts[winners[0]] = values.pop()
        elif not values:
            issues.append(f"missing {kind} payout for {winners[0]}")
        else:
            issues.append(f"conflicting {kind} payouts for {winners[0]}: {sorted(values)}")
        return payouts, issues

    # TARGET attaches each branch's payout to the tied horse which uniquely
    # identifies that winning combination (e.g. each third-place dead heat).
    # Ordered combinations with the same horse set have no such identifier and
    # remain unresolved unless an explicit map is supplied.
    winner_sets = [set(combo) for combo in winners]
    for index, combo in enumerate(winners):
        other_horses: set[int] = set()
        for other_index, other in enumerate(winner_sets):
            if other_index != index:
                other_horses.update(other)
        branch_horses = winner_sets[index] - other_horses
        values = {
            payout
            for horse in branch_horses
            if (payout := _row_payout(rows_by_horse[horse], kind)) is not None
        }
        if len(values) == 1:
            payouts[combo] = values.pop()
        elif not branch_horses:
            issues.append(f"ambiguous {kind} payout mapping for ordered combo {combo}")
        elif not values:
            issues.append(f"missing branch payout for {kind} {combo}")
        else:
            issues.append(f"conflicting branch payouts for {kind} {combo}: {sorted(values)}")
    return payouts, issues


def normalize_result(
    rows: Iterable[Mapping[str, Any]],
    *,
    refunded_horses: Iterable[int] | None = None,
    non_refunded_zero_horses: Iterable[int] | None = None,
    race_status: str = "confirmed",
    payout_maps: Mapping[str, Mapping[Any, Any]] | None = None,
) -> NormalizedRaceResult:
    """Normalize raw horse-level result rows into all supported markets.

    Args:
        rows: One mapping per horse. Japanese TARGET columns and the documented
            English aliases are accepted.
        refunded_horses: Horses explicitly declared cancelled/excluded.  This
            must come from an authoritative refund/status source.
        non_refunded_zero_horses: Position-zero horses explicitly declared to
            have started (e.g. did not finish) and therefore not refunded.
        race_status: ``confirmed`` or ``void``.  A void race refunds all bets.
        payout_maps: Optional authoritative combo-to-payout maps by ticket type.
            This is required for ordinary TARGET wide settlement and resolves
            ordered dead-heat payouts which horse-level rows cannot associate.

    Any zero-position horse absent from both explicit classification sets makes
    every market unresolved.  Position zero is never auto-refunded.
    """

    status = str(race_status).strip().lower()
    if status not in {"confirmed", "void"}:
        raise SettlementError(f"unsupported race status: {race_status!r}")

    raw_rows = list(rows)
    rows_by_horse: dict[int, Mapping[str, Any]] = {}
    positions: dict[int, int] = {}
    issues: list[str] = []
    for row in raw_rows:
        try:
            horse = _to_int(_first_present(row, _HORSE_COLUMNS), label="horse number")
            position = _to_int(_first_present(row, _POSITION_COLUMNS), label=f"position for horse {horse}")
        except SettlementError as exc:
            issues.append(str(exc))
            continue
        if horse <= 0 or position < 0:
            issues.append(f"invalid result row: horse={horse}, position={position}")
            continue
        if horse in rows_by_horse:
            issues.append(f"duplicate horse result row: {horse}")
            continue
        rows_by_horse[horse] = row
        positions[horse] = position

    refunded = frozenset(_to_int(x, label="refunded horse") for x in (refunded_horses or ()))
    non_refunded = frozenset(
        _to_int(x, label="non-refunded zero-position horse")
        for x in (non_refunded_zero_horses or ())
    )
    overlap = refunded & non_refunded
    if overlap:
        issues.append(f"horses classified both refunded and non-refunded: {sorted(overlap)}")
    positively_finished = {h for h, pos in positions.items() if pos > 0}
    if refunded & positively_finished:
        issues.append(f"positive finishers cannot be refunded: {sorted(refunded & positively_finished)}")

    zero_horses = frozenset(h for h, pos in positions.items() if pos == 0)
    unclassified_zero = zero_horses - refunded - non_refunded
    if unclassified_zero:
        issues.append(
            "zero-position horse(s) require explicit refund classification: "
            f"{sorted(unclassified_zero)}"
        )
    if non_refunded - zero_horses:
        issues.append(
            "non-refunded zero-position declaration has no zero-position row: "
            f"{sorted(non_refunded - zero_horses)}"
        )
    if not rows_by_horse:
        issues.append("no valid horse result rows")

    n_starters = len(rows_by_horse) - len(refunded & set(rows_by_horse))
    normalized_explicit: dict[str, Mapping[Any, Any]] = {}
    for alias, mapping in (payout_maps or {}).items():
        kind = canonical_bet_type(alias)
        if kind in normalized_explicit:
            raise SettlementError(f"duplicate payout map alias for {kind}")
        normalized_explicit[kind] = mapping

    markets: dict[str, MarketResult] = {}
    if status == "void":
        for kind in BET_TYPES:
            markets[kind] = MarketResult(kind, "void")
        return NormalizedRaceResult(
            status="void",
            markets=markets,
            refunded_horses=refunded,
            non_refunded_zero_horses=non_refunded,
            zero_position_horses=zero_horses,
            n_starters=n_starters,
            issues=tuple(issues),
        )

    structural_failure = bool(issues)
    for kind in BET_TYPES:
        winners = _winning_combos(kind, positions, n_starters)
        market_issues = list(issues) if structural_failure else []
        explicit = _explicit_map(kind, normalized_explicit.get(kind), winners)
        if explicit is not None:
            payouts, payout_issues = explicit
        else:
            payouts, payout_issues = _infer_payouts(kind, winners, rows_by_horse)
        market_issues.extend(payout_issues)
        missing = set(winners) - set(payouts)
        if missing and not payout_issues:
            market_issues.append(f"missing {kind} payout(s): {sorted(missing)}")
        market_status = "unresolved" if market_issues else "confirmed"
        markets[kind] = MarketResult(
            bet_type=kind,
            status=market_status,
            winning_combos=winners,
            payout_per100=payouts,
            issues=tuple(market_issues),
        )

    race_outcome = "unresolved" if structural_failure else "confirmed"
    return NormalizedRaceResult(
        status=race_outcome,
        markets=markets,
        refunded_horses=refunded,
        non_refunded_zero_horses=non_refunded,
        zero_position_horses=zero_horses,
        n_starters=n_starters,
        issues=tuple(issues),
    )


def settle_bets(
    bet_type: str,
    bets: Mapping[Any, int],
    result: NormalizedRaceResult,
) -> dict[str, Any]:
    """Settle combo-to-stake bets against a normalized race result.

    Stakes must be positive 100-yen increments.  If any purchased winning
    ticket has no authoritative payout, or the requested market itself is
    unresolved, ``ok`` is false and aggregate return/ROI/P&L are ``None``.
    """

    kind = canonical_bet_type(bet_type)
    market = result.market(kind)
    normalized_bets: dict[Combo, int] = {}
    for raw_combo, raw_stake in bets.items():
        combo = canonical_combo(kind, raw_combo)
        stake = _to_int(raw_stake, label=f"stake for {combo}")
        if stake <= 0 or stake % 100:
            raise SettlementError(f"stake must be a positive 100-yen increment: {stake}")
        normalized_bets[combo] = normalized_bets.get(combo, 0) + stake

    total_stake = sum(normalized_bets.values())
    detail: dict[Combo, dict[str, Any]] = {}
    refund = 0
    hit_return = 0
    unresolved: list[Combo] = []
    winners = set(market.winning_combos)

    for combo, stake in normalized_bets.items():
        if result.status == "void" or market.status == "void":
            refund += stake
            detail[combo] = {"stake": stake, "state": "refund", "back": stake}
        elif set(combo) & result.refunded_horses:
            refund += stake
            detail[combo] = {"stake": stake, "state": "refund", "back": stake}
        elif market.status == "unresolved":
            unresolved.append(combo)
            detail[combo] = {"stake": stake, "state": "unresolved", "back": None}
        elif combo in winners:
            payout = market.payout_per100.get(combo)
            if payout is None:
                unresolved.append(combo)
                detail[combo] = {
                    "stake": stake,
                    "state": "hit_unknown_payout",
                    "back": None,
                }
            else:
                back = stake // 100 * payout
                hit_return += back
                detail[combo] = {
                    "stake": stake,
                    "state": "hit",
                    "back": back,
                    "payout_per100": payout,
                }
        else:
            detail[combo] = {"stake": stake, "state": "lose", "back": 0}

    ok = not unresolved
    total_back = refund + hit_return if ok else None
    return {
        "bet_type": kind,
        "stake": total_stake,
        "refund": refund,
        "hit_return": hit_return if ok else None,
        "total_back": total_back,
        "pl": (total_back - total_stake) if total_back is not None else None,
        "roi": (total_back / total_stake) if total_back is not None and total_stake else None,
        "n_hit": sum(1 for item in detail.values() if item["state"] == "hit"),
        "unresolved": unresolved,
        "ok": ok,
        "market_status": market.status,
        "issues": market.issues,
        "detail": detail,
    }


__all__ = [
    "BET_TYPES",
    "Combo",
    "MarketResult",
    "NormalizedRaceResult",
    "SettlementError",
    "canonical_bet_type",
    "canonical_combo",
    "normalize_result",
    "settle_bets",
]
