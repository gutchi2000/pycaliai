#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""前向き価格ledgerの完全性・市場残差・締切価格ドリフトを評価する。

判断時点に実際に読んだJV-Link payloadのhashと、追記専用T-10 snapshotを照合する。
締切価格は約定価格ではなく最終市場の参照点として扱い、CLVとは呼ばない。
異なるpolicy_idやhash不一致を検出した場合は集計を失敗させる。
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from forward_prices import FORWARD_ROOT, fair_win_probabilities, read_snapshot  # noqa: E402
from generate_results import get_race_kk, get_winner, load_kekka_all, parse_race_id_16  # noqa: E402
from production_policy import load_policy  # noqa: E402


def _policy_id(record: dict) -> str | None:
    return (record.get("policy") or {}).get("policy_id")


def _observed(record: dict) -> str:
    return str(record.get("observed_at") or "")


def _num(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _metrics(rows: list[dict], field: str) -> dict:
    vals = [(r[field], r["won"]) for r in rows if r.get(field) is not None]
    if not vals:
        return {"n": 0}
    eps = 1e-12
    brier = sum((p - y) ** 2 for p, y in vals) / len(vals)
    logloss = -sum(y * math.log(max(eps, min(1 - eps, p)))
                   + (1 - y) * math.log(max(eps, min(1 - eps, 1 - p)))
                   for p, y in vals) / len(vals)
    return {"n": len(vals), "brier": round(brier, 6), "logloss": round(logloss, 6)}


def _residual_bins(rows: list[dict]) -> list[dict]:
    edges = [-math.inf, -0.05, -0.02, 0.0, 0.02, 0.05, math.inf]
    out = []
    for lo, hi in zip(edges, edges[1:]):
        xs = [r for r in rows if r.get("residual_t10") is not None
              and lo <= r["residual_t10"] < hi]
        if not xs:
            continue
        out.append({
            "lo": None if math.isinf(lo) else lo,
            "hi": None if math.isinf(hi) else hi,
            "n": len(xs),
            "mean_residual_t10": round(sum(x["residual_t10"] for x in xs) / len(xs), 6),
            "realized_win_rate": round(sum(x["won"] for x in xs) / len(xs), 6),
            "mean_model_p": round(sum(x["p_model"] for x in xs) / len(xs), 6),
            "mean_market_t10_p": round(sum(x["p_market_t10"] for x in xs) / len(xs), 6),
        })
    return out


def _bet_horses(primary: dict) -> set[int]:
    out: set[int] = set()
    for bet in primary.get("bets") or []:
        for token in re.findall(r"\d+", str(bet.get("買い目") or "")):
            out.add(int(token))
    return out


def evaluate(root: Path, start: int, expected_policy_id: str) -> tuple[dict, list[str]]:
    decisions: dict[str, list[dict]] = defaultdict(list)
    markets: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    lineage_errors: list[str] = []

    for path in sorted(root.glob("????????/*.json.gz")):
        if int(path.parent.name) < start:
            continue
        try:
            record = read_snapshot(path)
        except Exception as exc:
            lineage_errors.append(f"read failure {path}: {exc}")
            continue
        rid = str(record.get("race_id") or "")
        if len(rid) != 16:
            lineage_errors.append(f"invalid race_id {path}: {rid!r}")
            continue
        if _policy_id(record) != expected_policy_id:
            lineage_errors.append(
                f"policy mismatch {path.name}: {_policy_id(record)!r} != {expected_policy_id!r}")
            continue
        if record.get("record_type") == "decision_snapshot":
            decisions[rid].append(record)
        elif record.get("record_type") == "market_snapshot":
            markets[rid][str(record.get("stage"))].append(record)

    kekka = load_kekka_all()
    horse_rows: list[dict] = []
    race_rows: list[dict] = []
    integrity_errors: list[str] = []
    counts = {"decision_races": len(decisions), "t10_races": 0, "close_races": 0,
              "paired_t10": 0, "paired_close": 0, "settled_races": 0,
              "bet_races": 0, "skip_races": 0}

    for rid, versions in sorted(decisions.items()):
        decision = max(versions, key=_observed)
        t10s = markets[rid].get("t10", [])
        closes = markets[rid].get("close", [])
        counts["t10_races"] += int(bool(t10s))
        counts["close_races"] += int(bool(closes))
        exact_t10 = next((x for x in t10s
                          if x.get("market_sha256") == decision.get("market_sha256")), None)
        if exact_t10 is None:
            integrity_errors.append(f"{rid}: decisionと同一hashのT-10価格なし")
            continue
        counts["paired_t10"] += 1
        close = max(closes, key=_observed) if closes else None
        if close is None:
            integrity_errors.append(f"{rid}: close価格なし")
            continue
        counts["paired_close"] += 1

        primary = decision.get("primary") or {}
        selected = _bet_horses(primary)
        n_bets = len(primary.get("bets") or [])
        counts["bet_races"] += int(n_bets > 0)
        counts["skip_races"] += int(n_bets == 0)

        parsed = parse_race_id_16(rid)
        winner = None
        if parsed:
            race_kk = get_race_kk(kekka, rid[:8], parsed["place_name"], parsed["race_no"])
            if not race_kk.empty:
                winner = get_winner(race_kk)
                counts["settled_races"] += 1

        close_fair = fair_win_probabilities(
            (close or {}).get("market", {}).get("tansho") or {})
        for h in decision.get("horses") or []:
            ban = h.get("umaban")
            p_model = _num(h.get("p_win_model"))
            p_t10 = _num(h.get("p_win_market_devig"))
            p_close = _num(close_fair.get(int(ban))) if ban is not None else None
            if p_model is None or p_t10 is None:
                continue
            horse_rows.append({
                "race_id": rid, "umaban": int(ban), "p_model": p_model,
                "p_market_t10": p_t10, "p_market_close": p_close,
                "residual_t10": p_model - p_t10,
                "residual_close": p_model - p_close if p_close is not None else None,
                "market_move": p_close - p_t10 if p_close is not None else None,
                "selected": int(int(ban) in selected),
                "won": int(winner is not None and int(ban) == int(winner)),
                "settled": winner is not None,
            })
        race_rows.append({"race_id": rid, "decision_at": decision.get("observed_at"),
                          "close_at": close.get("observed_at") if close else None,
                          "n_bets": n_bets})

    settled = [r for r in horse_rows if r["settled"]]
    with_close = [r for r in horse_rows if r.get("p_market_close") is not None]
    selected_close = [r for r in with_close if r["selected"]]
    report = {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "policy_id": expected_policy_id,
        "start_date": start,
        "semantics": "closeは発走後取得の最終市場参照点。JRAパリミュチュエルの約定価格/CLVではない",
        "coverage": counts,
        "integrity_errors": integrity_errors,
        "settled_metrics": {
            "model": _metrics(settled, "p_model"),
            "market_t10": _metrics(settled, "p_market_t10"),
            "market_close": _metrics(settled, "p_market_close"),
        },
        "market_residual_bins": _residual_bins(settled),
        "price_drift": {
            "horse_rows": len(with_close),
            "mean_market_move": (round(sum(r["market_move"] for r in with_close)
                                            / len(with_close), 8) if with_close else None),
            "selected_horse_rows": len(selected_close),
            "selected_mean_market_move": (
                round(sum(r["market_move"] for r in selected_close) / len(selected_close), 8)
                if selected_close else None),
            "selected_backed_share": (
                round(sum(r["market_move"] > 0 for r in selected_close) / len(selected_close), 6)
                if selected_close else None),
        },
        "races": race_rows,
    }
    return report, lineage_errors


def main() -> int:
    policy = load_policy()
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=FORWARD_ROOT)
    ap.add_argument("--start", type=int,
                    default=int(policy["prospective"]["start_date"]))
    ap.add_argument("--policy-id", default=str(policy["policy_id"]))
    ap.add_argument("--out", type=Path,
                    default=BASE / "reports" / "forward_price_eval.json")
    args = ap.parse_args()

    report, lineage_errors = evaluate(args.root, args.start, args.policy_id)
    if lineage_errors or report["integrity_errors"]:
        print("[ERROR] forward price ledgerの来歴/完全性エラー", file=sys.stderr)
        for msg in (lineage_errors + report["integrity_errors"])[:50]:
            print(f"  - {msg}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(args.out)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[saved] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
