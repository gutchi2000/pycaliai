"""本番馬券policyの単一ソースと来歴検証。

閾値はモデル出力の生値ではなく、凍結した参照分布上のpercentileで定義する。
policy・分位表・モデル成果物のhashを各本番出力へ刻み、途中で動作点が変わった
データを同じ前向きcohortへ混ぜない。
"""
from __future__ import annotations

import bisect
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent
POLICY_PATH = BASE / "data" / "production_policy.json"


class PolicyError(RuntimeError):
    """本番policyまたは参照成果物が不整合。買い処理はfail-closedにする。"""


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PolicyError(f"JSON読込不能: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PolicyError(f"JSON rootがobjectでない: {path}")
    return value


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise PolicyError(f"policy成果物が存在しない: {path}")
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@lru_cache(maxsize=1)
def load_policy() -> dict:
    p = _read_json(POLICY_PATH)
    required = {"schema_version", "policy_id", "engine", "engine_version",
                "chaos_reference", "hard_skip", "prospective"}
    missing = sorted(required - set(p))
    if missing:
        raise PolicyError(f"production_policy 欠落: {missing}")
    chaos = p["chaos_reference"]
    q = float(chaos.get("skip_percentile", -1))
    if not 0.0 < q < 1.0:
        raise PolicyError(f"chaos skip_percentile不正: {q}")
    return p


def _chaos_path(policy: dict | None = None) -> Path:
    p = policy or load_policy()
    rel = p["chaos_reference"].get("path")
    if not rel:
        raise PolicyError("chaos_reference.path 欠落")
    return BASE / str(rel)


@lru_cache(maxsize=1)
def load_quantiles() -> dict:
    policy = load_policy()
    path = _chaos_path(policy)
    q = _read_json(path)
    expected = policy["chaos_reference"].get("reference_id")
    actual = q.get("reference_id")
    if expected != actual:
        raise PolicyError(
            f"chaos参照版不一致: policy={expected!r}, quantiles={actual!r}")
    tables = q.get("quantiles")
    if not isinstance(tables, dict):
        raise PolicyError("chaos_quantiles.quantiles 欠落")
    for key in ("field_chaos_score", "top1_dominance", "top2_concentration"):
        vals = tables.get(key)
        if not isinstance(vals, list) or len(vals) != 101:
            raise PolicyError(f"分位表 {key} は101点必要")
        if any(not isinstance(v, (int, float)) or not math.isfinite(float(v)) for v in vals):
            raise PolicyError(f"分位表 {key} に非数値")
        if any(float(a) > float(b) for a, b in zip(vals, vals[1:])):
            raise PolicyError(f"分位表 {key} が単調でない")
    return q


def percentile(raw: Any, key: str) -> float | None:
    """生値を凍結参照分布上のpercentileへ線形補間する。欠損はNone。"""
    if raw is None or isinstance(raw, bool):
        return None
    try:
        x = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    vals = [float(v) for v in load_quantiles()["quantiles"].get(key, [])]
    if len(vals) != 101:
        return None
    if x <= vals[0]:
        return 0.0
    if x >= vals[-1]:
        return 1.0
    i = bisect.bisect_right(vals, x)
    lo, hi = vals[i - 1], vals[i]
    frac = 0.0 if hi == lo else (x - lo) / (hi - lo)
    return ((i - 1) + frac) / 100.0


def raw_at_percentile(q: float, key: str = "field_chaos_score") -> float:
    vals = [float(v) for v in load_quantiles()["quantiles"][key]]
    pos = min(1.0, max(0.0, float(q))) * 100.0
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def find_hon(horses: list[dict]) -> dict | None:
    return next((h for h in horses if h.get("mark") == "◎"), None)


def hard_skip_reasons(race_meta: dict, race_conf: dict,
                      hon: dict | None) -> list[str]:
    """本番共通のhard見送り理由。欠損はpolicy設定どおりfail-closed。"""
    p = load_policy()
    cfg = p["hard_skip"]
    fail_closed = bool(cfg.get("fail_closed_on_missing", True))
    reasons: list[str] = []

    raw = race_conf.get("field_chaos_score")
    chaos_pct = percentile(raw, "field_chaos_score")
    chaos_limit = float(p["chaos_reference"]["skip_percentile"])
    if chaos_pct is None:
        if fail_closed:
            reasons.append("chaos欠損/変換不能")
    elif chaos_pct + 1e-12 >= chaos_limit:
        reasons.append(
            f"chaos raw={float(raw):.3f} pct={chaos_pct:.3f}>={chaos_limit:.3f}")

    fs = race_meta.get("field_size")
    try:
        fs_num = int(float(fs)) if fs is not None else None
    except (TypeError, ValueError):
        fs_num = None
    if fs_num is None:
        if fail_closed:
            reasons.append("field_size欠損")
    elif fs_num <= int(cfg["field_size_max"]):
        reasons.append(f"field_size {fs_num}<={int(cfg['field_size_max'])}")

    if hon is None:
        reasons.append("◎なし")
        return reasons
    if cfg.get("require_hon_tansho_odds", True) and hon.get("tansho_odds") is None:
        reasons.append("◎ tansho_odds欠損")
    pw = hon.get("p_win")
    try:
        pw_num = float(pw) if pw is not None else None
    except (TypeError, ValueError):
        pw_num = None
    if pw_num is None or not math.isfinite(pw_num):
        if fail_closed:
            reasons.append("◎ p_win欠損")
    elif pw_num < float(cfg["hon_p_win_min"]):
        reasons.append(f"◎ p_win {pw_num:.3f}<{float(cfg['hon_p_win_min']):.3f}")
    return reasons


@lru_cache(maxsize=1)
def policy_stamp() -> dict:
    """本番出力に刻む、再現可能なpolicy/artifact来歴。"""
    p = load_policy()
    qpath = _chaos_path(p)
    stamp = {
        "policy_id": p["policy_id"],
        "policy_schema_version": p["schema_version"],
        "policy_sha256": sha256_file(POLICY_PATH),
        "chaos_reference_id": p["chaos_reference"]["reference_id"],
        "chaos_reference_sha256": sha256_file(qpath),
        "chaos_skip_percentile": float(p["chaos_reference"]["skip_percentile"]),
        "chaos_skip_raw_equivalent": round(raw_at_percentile(
            float(p["chaos_reference"]["skip_percentile"])), 6),
    }
    hashes = {}
    for name, rel in (p.get("artifacts") or {}).items():
        hashes[name] = sha256_file(BASE / str(rel))
    stamp["artifact_sha256"] = hashes
    return stamp


def clear_policy_caches() -> None:
    """テスト・再生成ツール用。通常運用では呼ばない。"""
    load_policy.cache_clear()
    load_quantiles.cache_clear()
    policy_stamp.cache_clear()
