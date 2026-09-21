"""前向き価格スナップショットと市場残差shadowの永続化。

`reports/live_odds/{race}.json` は本番計算用のlatest viewで上書きされる。本モジュールは
同じ観測を `data/forward_prices/` へgzip圧縮・追記専用で保存し、T-10判断時点と
締切後価格を後から厳密にpairできるようにする。
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from production_policy import policy_stamp

BASE = Path(__file__).resolve().parent
FORWARD_ROOT = BASE / "data" / "forward_prices"
SCHEMA_VERSION = 1
_JST = timezone(timedelta(hours=9))

# 粗い正常性チェック専用 (2026-09-21、T-10の-5,228分異常の再発防止で追加)。
# 「明らかにおかしい (別日の再取得等)」だけを標準エラーへ警告するためのゆるい
# 目安であり、保存の可否には一切影響しない (bodyにもhashにも含めない — 既存ファイルの
# 冪等再書込判定を壊さないため)。厳密なwindow判定・除外は
# analysis/forward_price_timing_canary.py が別途、読み取り専用で行う。
_TIMING_SANITY_WINDOWS_MIN = {
    "t10": (-30.0, 60.0), "close": (-30.0, 30.0), "t20": (-30.0, 90.0),
    "vote": (-30.0, 60.0), "exp05fs_t35": (-30.0, 90.0),
}


def _warn_if_timing_looks_wrong(stage: str, rid: str, scheduled_post, observed_at: str) -> None:
    window = _TIMING_SANITY_WINDOWS_MIN.get(stage)
    if window is None or not scheduled_post:
        return
    try:
        sp = datetime.fromisoformat(str(scheduled_post))
        oa = datetime.fromisoformat(str(observed_at))
        if sp.tzinfo is None:
            sp = sp.replace(tzinfo=_JST)
        if oa.tzinfo is None:
            oa = oa.replace(tzinfo=_JST)
        m = (sp - oa).total_seconds() / 60.0
    except Exception:
        return
    lo, hi = window
    if not (lo <= m <= hi):
        print(f"[forward_prices][WARN] {stage} race={rid} scheduled_post={scheduled_post} "
              f"observed_at={observed_at} minutes_to_post={m:.1f} "
              f"(想定[{lo},{hi}]分から外れる。手動/デバッグ実行を本番storeへ紛れ込ませて"
              f"いないか確認。保存は正常に続行する。analysis.forward_price_timing_canary で"
              f"事後隔離・除外できる)", file=sys.stderr)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def payload_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _rid16(value: Any) -> str:
    return re.sub(r"\D", "", str(value or ""))[:16]


def _slug_time(value: Any) -> str:
    text = str(value or datetime.now().isoformat(timespec="milliseconds"))
    return re.sub(r"[^0-9]", "", text)[:17].ljust(17, "0")


def _write_gzip_atomic(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = read_snapshot(path)
        if _canonical_bytes(existing) != _canonical_bytes(value):
            raise FileExistsError(f"immutable snapshot collision: {path}")
        return path
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", newline="\n") as fh:
        json.dump(value, fh, ensure_ascii=False, sort_keys=True,
                  separators=(",", ":"))
    tmp.replace(path)
    return path


def read_snapshot(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        value = json.load(fh)
    if not isinstance(value, dict):
        raise ValueError(f"snapshot rootがobjectでない: {path}")
    return value


def archive_market_snapshot(market: dict, stage: str, *,
                            scheduled_post: str | None = None,
                            stamp: dict | None = None,
                            root: Path = FORWARD_ROOT) -> Path:
    """JV-Link観測を不変スナップショットとして保存する。

    stage は `t10` / `t20` / `vote` / `close` / `manual` / `exp05fs_t35`。`vote` は学生大会の投票時点
    (発走 4 分前) の観測で、本番 T-10 の cohort とは別物として保存する。`t20` は
    サイト公開プレビュー (t20_site_bets.py) 専用で、こちらも T-10/vote/close の
    どの cohort とも混ぜない (2026-09-11 追加)。`exp05fs_t35` は
    analysis/mcond/exp05_forward_shadow (発走31-38分前) 専用、他のどの cohort とも
    混ぜない (2026-09-19 追加)。
    同一秒・同一内容の再実行は同じファイルへ
    冪等書込、内容が違えばhash suffixが変わり履歴を失わない。
    """
    if stage not in {"t10", "t20", "vote", "close", "manual", "exp05fs_t35"}:
        raise ValueError(f"未知のprice stage: {stage}")
    rid = _rid16(market.get("race_id"))
    if len(rid) != 16:
        raise ValueError(f"race_id不正: {market.get('race_id')!r}")
    observed = str(market.get("fetched") or datetime.now().isoformat(timespec="milliseconds"))
    body = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "market_snapshot",
        "stage": stage,
        "race_id": rid,
        "observed_at": observed,
        "scheduled_post": scheduled_post,
        "source": "JV-Link:0B31/0B33/0B34",
        "policy": stamp or policy_stamp(),
        "market": market,
    }
    body["market_sha256"] = payload_sha256(market)
    body["record_sha256"] = payload_sha256(body)
    name = (f"{rid}_{stage}_{_slug_time(observed)}_"
            f"{body['record_sha256'][:10]}.json.gz")
    saved = _write_gzip_atomic(Path(root) / rid[:8] / name, body)
    _warn_if_timing_looks_wrong(stage, rid, scheduled_post, observed)
    return saved


def fair_win_probabilities(tansho: dict) -> dict[int, float]:
    """単勝オッズをレース内de-vigし、合計1の市場確率へ変換する。"""
    inv: dict[int, float] = {}
    for key, value in (tansho or {}).items():
        try:
            ban, odds = int(key), float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(odds) and odds > 1.0:
            inv[ban] = 1.0 / odds
    total = sum(inv.values())
    return {ban: value / total for ban, value in inv.items()} if total > 0 else {}


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _pair_key(value: str) -> tuple[int, int] | None:
    nums = [int(x) for x in re.findall(r"\d+", str(value))]
    if len(nums) != 2:
        return None
    return min(nums), max(nums)


def build_decision_record(race: dict, market: dict, primary: dict,
                          shadow: dict | None, *, mode: str,
                          residual_shadow: dict | None = None,
                          model_umaren: dict[tuple[int, int], float] | None = None,
                          model_wide: dict[tuple[int, int], float] | None = None,
                          stamp: dict | None = None) -> dict:
    """同一T-10観測上のmodel-vs-market残差と両engine判断を一体化する。"""
    rid = _rid16(race.get("race_id") or race.get("race_meta", {}).get("race_id"))
    if len(rid) != 16:
        raise ValueError(f"race_id不正: {rid!r}")
    fair = fair_win_probabilities(market.get("tansho") or {})
    tan = {int(k): _num(v) for k, v in (market.get("tansho") or {}).items()}
    fuku = {int(k): v for k, v in (market.get("fukusho") or {}).items()}
    horses = []
    for h in race.get("horses", []):
        try:
            ban = int(h.get("umaban"))
        except (TypeError, ValueError):
            continue
        p_model = _num(h.get("p_win"))
        p_market = fair.get(ban)
        fo = fuku.get(ban)
        horses.append({
            "umaban": ban,
            "mark": h.get("mark") or "",
            "p_win_model": p_model,
            "p_sho_model": _num(h.get("p_sho")),
            "tansho_odds_t10": tan.get(ban),
            "fukusho_odds_t10": fo,
            "p_win_market_devig": p_market,
            "win_market_residual": (
                p_model - p_market if p_model is not None and p_market is not None else None),
        })

    live_wide = market.get("wide") or {}
    pair_rows = []
    keys = set(model_wide or {}) | set(model_umaren or {})
    keys |= {k for raw in live_wide for k in [_pair_key(raw)] if k is not None}
    for key in sorted(keys):
        wide_range = live_wide.get(f"{key[0]}-{key[1]}")
        mid = None
        if isinstance(wide_range, (list, tuple)) and len(wide_range) >= 2:
            lo, hi = _num(wide_range[0]), _num(wide_range[1])
            if lo is not None and hi is not None:
                mid = (lo + hi) / 2.0
        pw = _num((model_wide or {}).get(key))
        pair_rows.append({
            "pair": f"{key[0]}-{key[1]}",
            "p_umaren_model": _num((model_umaren or {}).get(key)),
            "p_wide_model": pw,
            "wide_odds_t10": wide_range,
            "wide_implied_raw_mid": (1.0 / mid if mid and mid > 0 else None),
            "wide_market_residual_raw": (
                pw - 1.0 / mid if pw is not None and mid and mid > 0 else None),
        })

    record = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "decision_snapshot",
        "race_id": rid,
        "observed_at": str(market.get("fetched") or datetime.now().isoformat(timespec="milliseconds")),
        "mode": mode,
        "policy": stamp or policy_stamp(),
        "market_sha256": payload_sha256(market),
        "market_overround_tan": _num(market.get("overround_tan")),
        "horses": horses,
        "pairs": pair_rows,
        "primary": primary,
        "shadow": shadow,
        "wide_residual_shadow": residual_shadow,
    }
    record["record_sha256"] = payload_sha256(record)
    return record


def archive_decision_record(record: dict, *, root: Path = FORWARD_ROOT) -> Path:
    rid = _rid16(record.get("race_id"))
    if len(rid) != 16:
        raise ValueError(f"race_id不正: {record.get('race_id')!r}")
    observed = record.get("observed_at")
    digest = record.get("record_sha256") or payload_sha256(record)
    name = f"{rid}_decision_{_slug_time(observed)}_{str(digest)[:10]}.json.gz"
    return _write_gzip_atomic(Path(root) / rid[:8] / name, record)
