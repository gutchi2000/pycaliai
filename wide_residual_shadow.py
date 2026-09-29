"""事前登録済みワイド市場残差ポリシーの前向きshadow実装。

実投票用の買い目には一切触れず、本番と同じT-10観測・hard gateから仮想100円券と
同一発火レース対照を生成する。仕様の単一ソースは
``data/shadow_policies/wide_residual_shadow_v3.json``。
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Callable

from forward_prices import payload_sha256
from production_policy import find_hon, hard_skip_reasons, load_policy, policy_stamp


BASE = Path(__file__).resolve().parent
POLICY_PATH = BASE / "data" / "shadow_policies" / "wide_residual_shadow_v3.json"


class ShadowPolicyError(RuntimeError):
    """shadow policyまたはT-10入力が事前登録仕様を満たさない。"""


def _rid16(value) -> str:
    return re.sub(r"\D", "", str(value or ""))[:16]


def _num(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_shadow_policy(path: Path = POLICY_PATH) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ShadowPolicyError(f"shadow policy読込不能: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ShadowPolicyError("shadow policy rootがobjectでない")
    required = {"policy_id", "status", "real_money_enabled", "parent_production_policy",
                "candidate", "controls", "formal_look"}
    missing = sorted(required - set(value))
    if missing:
        raise ShadowPolicyError(f"shadow policy必須項目欠落: {missing}")
    if value["status"] != "shadow_only" or value["real_money_enabled"] is not False:
        raise ShadowPolicyError("wide residual v3はshadow_only / real_money=falseでなければならない")
    parent = load_policy()["policy_id"]
    if value["parent_production_policy"] != parent:
        raise ShadowPolicyError(
            f"親production policy不一致: {value['parent_production_policy']} != {parent}")
    return value


def shadow_policy_stamp(path: Path = POLICY_PATH) -> dict:
    policy = load_shadow_policy(path)
    return {
        "policy_id": policy["policy_id"],
        "policy_sha256": _sha256_file(path),
        "parent_production_policy": policy["parent_production_policy"],
        "status": policy["status"],
        "real_money_enabled": False,
    }


def market_wide_fair(wide: dict) -> dict[tuple[int, int], float]:
    """ワイド中点オッズをde-vigし、的中ペア3本に合わせて合計3へ正規化する。"""
    raw: dict[tuple[int, int], float] = {}
    for key, value in (wide or {}).items():
        try:
            a, b = (int(x) for x in str(key).split("-"))
        except (TypeError, ValueError):
            continue
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            continue
        lo, hi = _num(value[0]), _num(value[1])
        if lo is None or hi is None or lo <= 1.0 or hi <= 1.0 or hi < lo:
            continue
        raw[(min(a, b), max(a, b))] = 1.0 / ((lo + hi) / 2.0)
    total = sum(raw.values())
    if total <= 0:
        return {}
    # 既存の独立監査実装と一致させる。極端な個別値だけ確率上限1へ丸める。
    return {key: min(1.0, value * 3.0 / total) for key, value in raw.items()}


def compute_shadow(
    race: dict,
    market: dict,
    *,
    pair_probability_fn: Callable[[list[dict]], tuple[dict, dict]],
    policy_path: Path = POLICY_PATH,
) -> dict:
    """1レースの仮想Arm Aと、同じ発火レース用M1/M2候補を生成する。"""
    policy = load_shadow_policy(policy_path)
    cfg = policy["candidate"]
    rid = _rid16(race.get("race_id") or (race.get("race_meta") or {}).get("race_id"))
    if len(rid) != 16 or _rid16(market.get("race_id")) != rid:
        raise ShadowPolicyError(
            f"race_id不一致: bundle={rid!r} market={market.get('race_id')!r}")
    if not market.get("ok"):
        raise ShadowPolicyError(f"T-10 market ok=false: {market.get('reason', '')}")

    horses = [dict(h) for h in (race.get("horses") or [])]
    live_tan = {int(k): _num(v) for k, v in (market.get("tansho") or {}).items()}
    for horse in horses:
        try:
            ban = int(horse["umaban"])
        except (KeyError, TypeError, ValueError):
            continue
        if live_tan.get(ban) is not None:
            horse["tansho_odds"] = live_tan[ban]

    gate = hard_skip_reasons(
        race.get("race_meta") or {}, race.get("race_confidence") or {}, find_hon(horses))
    base = {
        "race_id": rid,
        "observed_at": market.get("fetched"),
        "policy": shadow_policy_stamp(policy_path),
        "parent_policy": policy_stamp(),
        "market_sha256": payload_sha256(market),
        "real_money_enabled": False,
        "hard_gate_passed": not gate,
        "hard_gate_reasons": gate,
        "arm_a": [],
        "control_m1": [],
        "control_m2": [],
        "triggered": False,
    }
    if gate:
        return base

    _, model_wide = pair_probability_fn(horses)
    live_wide = market.get("wide") or {}
    fair = market_wide_fair(live_wide)
    if not fair:
        raise ShadowPolicyError("T-10ワイド市場確率を計算できない")
    active = sorted(
        int(horse["umaban"])
        for horse in horses
        if _num(horse.get("tansho_odds")) is not None
    )
    expected_pairs = {
        (active[i], active[j])
        for i in range(len(active))
        for j in range(i + 1, len(active))
    }
    if set(fair) != expected_pairs:
        missing = len(expected_pairs - set(fair))
        extra = len(set(fair) - expected_pairs)
        raise ShadowPolicyError(
            f"T-10ワイド価格が不完全: expected={len(expected_pairs)} "
            f"actual={len(fair)} missing={missing} extra={extra}")

    priced = []
    max_mid = float(cfg["max_odds_mid"])
    for pair, probability in (model_wide or {}).items():
        key = (min(int(pair[0]), int(pair[1])), max(int(pair[0]), int(pair[1])))
        odds_range = live_wide.get(f"{key[0]}-{key[1]}")
        if not isinstance(odds_range, (list, tuple)) or len(odds_range) < 2:
            continue
        lo, hi = _num(odds_range[0]), _num(odds_range[1])
        p_model, p_market = _num(probability), fair.get(key)
        if (lo is None or hi is None or p_model is None or p_market is None
                or (lo + hi) / 2.0 > max_mid):
            continue
        priced.append({
            "selection": f"{key[0]}-{key[1]}",
            "p_model": p_model,
            "p_market_fair": p_market,
            "residual": p_model - p_market,
            "odds_t10_low": lo,
            "odds_t10_high": hi,
            "odds_t10_mid": (lo + hi) / 2.0,
            "virtual_stake_yen": int(cfg["stake_per_ticket_yen"]),
        })
    if not priced:
        raise ShadowPolicyError("T-10価格付きワイド候補が0件")

    n_top = int(cfg["take_model_top_n_before_filter"])
    model_top = sorted(priced, key=lambda row: (-row["p_model"], row["selection"]))[:n_top]
    market_top = sorted(priced, key=lambda row: (-row["p_market_fair"], row["selection"]))[:n_top]
    lower = float(cfg["residual_lower_inclusive"])
    upper = float(cfg["residual_upper_exclusive"])
    arm = [row for row in model_top if lower <= row["residual"] < upper]
    arm = arm[:int(cfg["max_tickets_per_race"])]

    base["arm_a"] = arm
    base["control_m1"] = model_top if arm else []
    base["control_m2"] = market_top if arm else []
    base["triggered"] = bool(arm)
    return base


def merge_daily_shadow(path: Path, entries: list[dict]) -> Path:
    """t10_runnerのレース単位呼出しに対応する置換型・原子的マージ。"""
    if not entries:
        return path
    stamp = entries[0]["policy"]
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        current = {"policy": stamp, "races": []}
    old = (current.get("policy") or {}).get("policy_id")
    if current.get("races") and old != stamp["policy_id"]:
        raise ShadowPolicyError(f"shadow cohort混在: {old} != {stamp['policy_id']}")
    by_rid = {_rid16(row.get("race_id")): row for row in current.get("races", [])}
    for row in entries:
        if row.get("policy") != stamp:
            raise ShadowPolicyError("同一日ファイル内でshadow policy stampが不一致")
        by_rid[_rid16(row.get("race_id"))] = row
    current["policy"] = stamp
    current["races"] = sorted(by_rid.values(), key=lambda row: row["race_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path
