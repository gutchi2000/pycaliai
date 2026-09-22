# -*- coding: utf-8 -*-
"""
layer_evidence_audit.py — 各防御層で eligibility evidence が再取得できるか検査
=============================================================================
「上流で race を消したから下流は安全」とは数えない。
各層が **race_id から独立に evidence を再取得できる** ことを実測で確認する。

出力: analysis/jump_history_only/out/layer_evidence_audit.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))

from race_eligibility import (  # noqa: E402
    evaluate_race, eligibility_metadata, verify_metadata, assert_bettable,
    JumpRaceBettingError, EligibilityMetadataError, clear_cache,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)

JUMP = "2026091306040401"
FLAT = "2026091306040402"
UNKNOWN = "2099123106040401"     # どのソースにも無い


def log(m):
    print(m, flush=True)


def probe_layer(name: str, fn) -> dict:
    """層 fn(rid) -> (eligible: bool) を 3 種の rid で確認。"""
    out = {}
    for label, rid in (("jump", JUMP), ("flat", FLAT), ("unknown", UNKNOWN)):
        try:
            out[label] = fn(rid)
        except (JumpRaceBettingError, EligibilityMetadataError) as e:
            out[label] = f"REFUSED({type(e).__name__})"
        except Exception as e:                            # pragma: no cover
            out[label] = f"ERROR({type(e).__name__}: {str(e)[:60]})"
    return out


def main() -> int:
    clear_cache()
    log("=" * 78)
    log("各防御層の evidence 再取得可否")
    log("=" * 78)

    rows = []

    # --- 層1: bundle creation ---
    def l1(rid):
        el = evaluate_race(rid)
        return el["prediction_eligible"]
    rows.append({
        "layer": "1. bundle creation (export_weekly_marks)",
        "available_fields": "race_id（df の トラックコード(JV) は **渡さない**)",
        "authoritative_source": "data/bunseki/{date}.csv → トラックコード(JV)"
                                " + data/bias → 平・障",
        "uses_default": "No（df の欠損埋め 23 を一切渡さない）",
        "unknown_behavior": "prediction_eligible=False で bundle から除外し、"
                            "ERROR ログ + 標準出力へ警告",
        "probe": probe_layer("l1", l1),
    })

    # --- 層2: task registration ---
    def l2(rid):
        from t10_runner import build_schedule
        races = [{"race_id": rid, "race_meta": {"place": "X"}}]
        sched, _ = build_schedule(rid[:8], races, lead_min=10)
        return bool(sched)
    rows.append({
        "layer": "2. task registration (t10_runner.build_schedule)",
        "available_fields": "race_id（bundle の race dict から取得）",
        "authoritative_source": "同上（race_id から再取得）",
        "uses_default": "No",
        "unknown_behavior": "task_registration_eligible=False で schedule から除外",
        "probe": probe_layer("l2", l2),
    })

    # --- 層3: compute_bets ---
    def l3(rid):
        from compute_bets import compute_race_bets
        race = {"race_id": rid,
                "race_meta": {"race_id": rid, "place": "中山", "field_size": 12},
                "race_confidence": {"field_chaos_score": 0.1},
                "horses": [{"umaban": i, "mark": "◎" if i == 1 else "",
                            "p_win": 0.2, "tansho_odds": 3.0}
                           for i in range(1, 13)]}
        out = compute_race_bets(race)
        return out.get("excluded_reason") is None
    rows.append({
        "layer": "3. compute_bets",
        "available_fields": "race_id + bundle の eligibility metadata",
        "authoritative_source": "verify_metadata() が race_id から**再計算**し "
                                "metadata と突合（boolean を信用しない）",
        "uses_default": "No",
        "unknown_behavior": "bets=[] / race_nature=見送り / "
                            "eligibility_determination=unknown。"
                            "metadata 欠落・schema 不一致・hash 不一致は "
                            "そのレースだけ fail-closed（バッチは止めない）",
        "probe": probe_layer("l3", l3),
    })

    # --- 層4: validate ---
    def l4(rid):
        el = evaluate_race(rid)
        return el["bet_eligible"]
    rows.append({
        "layer": "4. validate_cowork_bets",
        "available_fields": "race_id（bets.json から）。bundle 不在でも可",
        "authoritative_source": "evaluate_race() で race_id から独立に再取得",
        "uses_default": "No",
        "unknown_behavior": "非空買い目なら違反として計上し、--apply で bets=[] に矯正",
        "probe": probe_layer("l4", l4),
    })

    # --- 層5: final submit ---
    def l5(rid):
        assert_bettable(rid, [{"馬券種": "複勝", "買い目": "3", "購入額": 100}],
                        layer="audit")
        return True
    rows.append({
        "layer": "5. final submit (masters_vote.submit)",
        "available_fields": "payload の race_id（+ 任意で metadata）",
        "authoritative_source": "assert_bettable() → verify_metadata/"
                                "evaluate_race で再取得",
        "uses_default": "No",
        "unknown_behavior": "非空買い目なら JumpRaceBettingError を送出して停止",
        "probe": probe_layer("l5", l5),
    })

    log(f"\n{'layer':<46} {'jump':<26} {'flat':<8} {'unknown':<26}")
    for r in rows:
        p = r["probe"]
        log(f"{r['layer']:<46} {str(p['jump']):<26} {str(p['flat']):<8} "
            f"{str(p['unknown']):<26}")

    # 期待: jump/unknown は必ず「通らない」、flat は必ず「通る」
    ok = True
    for r in rows:
        p = r["probe"]
        if p["jump"] is True or p["unknown"] is True or p["flat"] is not True:
            ok = False
    log(f"\n全層で jump/unknown が遮断され flat が通る: {'PASS' if ok else 'FAIL'}")

    payload = {"generated_at": datetime.now().isoformat(),
               "probe_race_ids": {"jump": JUMP, "flat": FLAT,
                                  "unknown": UNKNOWN},
               "layers": rows, "all_layers_ok": ok}
    (OUT / "layer_evidence_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    log(f"保存: {OUT / 'layer_evidence_audit.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
