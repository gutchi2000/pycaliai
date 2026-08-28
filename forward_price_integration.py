"""compute_bets と前向き価格ledgerの薄い統合層。"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

from forward_prices import archive_decision_record, build_decision_record


def _rid16(value) -> str:
    return re.sub(r"\D", "", str(value or ""))[:16]


def archive_compute_decisions(
    races: list[dict], primary: list[dict], shadow: list[dict],
    live_dir: Path, *, mode: str, stamp: dict,
    pair_probability_fn: Callable[[list[dict]], tuple[dict, dict]],
) -> list[Path]:
    """本番apply前に全decisionを保存する。1件でも失敗すれば例外でfail-closed。"""
    race_by_id = {_rid16(r.get("race_id") or r.get("race_meta", {}).get("race_id")): r
                  for r in races}
    shadow_by_id = {_rid16(e.get("race_id")): e for e in shadow}
    archived: list[Path] = []
    for decision in primary:
        rid = _rid16(decision.get("race_id"))
        race = race_by_id.get(rid)
        if race is None:
            raise ValueError(f"decision raceがbundleにない: {rid}")
        market_path = Path(live_dir) / f"{rid}.json"
        try:
            market = json.loads(market_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"T-10 market読込不能: {market_path}: {exc}") from exc
        if _rid16(market.get("race_id")) != rid:
            raise ValueError(f"market race_id不一致: {rid} vs {market.get('race_id')}")
        umaren, wide = pair_probability_fn(race.get("horses", []))
        record = build_decision_record(
            race, market, decision, shadow_by_id.get(rid), mode=mode,
            model_umaren=umaren, model_wide=wide, stamp=stamp)
        archived.append(archive_decision_record(record))
    return archived
