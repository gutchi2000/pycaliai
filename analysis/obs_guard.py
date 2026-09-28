# -*- coding: utf-8 -*-
"""
obs_guard.py — 観測計画 v2.1: 500R 到達前に性能・ROI・帯選択を実行させないための入口 guard
=========================================================================================
§5.3 / §6 / §7 の性能・回収率・帯選択・候補選択の集計スクリプトは、計算の最初に
`assert_performance_allowed(stream, root)` を呼ぶこと。有効 race が 500 未満なら PermissionError で止まる。

有効 race の数え方は label-free な部分だけ（§5.3 の有効 race 定義のうち、発売中・全組完全・
race_key 一致・`t10`（三連複は trio_t10）と final 候補の両方が存在）。同着の除外は結果を要するため
ここでは数えない。したがって本関数の件数は §5.3 の有効 race の上限であり、上限が 500 未満なら
確実に未到達。
"""
from __future__ import annotations

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from forward_prices import FORWARD_ROOT, canonical_stage, read_snapshot  # noqa: E402

MIN_RACES_FOR_PERFORMANCE = 500
DECISION_STAGE = {"trio": "trio_t10", "umatan": "t10", "stake": "t10"}
SPEC = {"trio": "0B35", "umatan": "0B34", "stake": None}
FINAL_STAGES = {"final_rt_candidate", "final_stock_candidate"}


def _ok_capture(rec: dict, spec: str | None) -> bool:
    caps = rec.get("jv_captures") or []
    if spec is None:
        return bool(caps) and all(c.get("ok") for c in caps)
    return any(c.get("spec") == spec and c.get("ok") for c in caps)


def count_label_free_valid_races(stream: str, root: Path = FORWARD_ROOT) -> int:
    if stream not in DECISION_STAGE:
        raise ValueError(f"unknown stream {stream!r}")
    dec, fin = set(), set()
    for p in root.glob("*/*.json.gz"):
        if p.parent.name.startswith("_"):
            continue
        rec = read_snapshot(p)
        st = canonical_stage(rec.get("stage"))
        if st == DECISION_STAGE[stream] and _ok_capture(rec, SPEC[stream]):
            dec.add(rec.get("race_id"))
        elif st in FINAL_STAGES and _ok_capture(rec, SPEC[stream]):
            fin.add(rec.get("race_id"))
    return len(dec & fin)


def assert_performance_allowed(stream: str, root: Path = FORWARD_ROOT, n: int | None = None) -> int:
    n = count_label_free_valid_races(stream, root) if n is None else n
    if n < MIN_RACES_FOR_PERFORMANCE:
        raise PermissionError(
            f"observation plan v2.1: {stream} has {n} label-free valid races (< {MIN_RACES_FOR_PERFORMANCE}). "
            "Performance, ROI, band or candidate selection must not run before 500R.")
    return n
