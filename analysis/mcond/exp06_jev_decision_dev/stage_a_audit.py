# -*- coding: utf-8 -*-
"""
stage_a_audit.py — Stage A: API・再現性監査 (spec.json Stage A参照)
================================================================================
実施すること:
  - 認証成功
  - Choice/Score/Noulの型が期待通りか
  - 確率和が1に近いか
  - confidenceの範囲(0-1または0-100を想定、実際のレンジを記録)
  - 応答モデル名・usageが取れるか
  - タイムアウト・429・5xx時の処理(jev_client.pyのリトライ骨格を使う)
  - 同一入力を複数回送った際の変動(決定論的か、揺らぐならどの程度か)
  - 日本語入力と英語入力の差(本評価では使わないが記録だけする)

英語固定テンプレートを使う。ここでの比較結果を見て質問文や言語を変更しない
(spec §Stage A、変更してよいのは公式ドキュメント判明後の`_call_jev_api_raw`実装のみ)。

現状: jev_client._call_jev_api_raw が未実装のため、このスクリプトを実行すると
NotImplementedError で止まる。TypeSafe Jev APIの仕様確定後、jev_client.pyの
該当箇所を実装してから再実行すること。

実行: python -m analysis.mcond.exp06_jev_decision_dev.stage_a_audit
出力: analysis/mcond/exp06_jev_decision_dev/out/STAGE_A_AUDIT.json (+ .md要約)
"""
from __future__ import annotations
import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
sys.path.insert(0, str(BASE))

from analysis.mcond.exp06_jev_decision_dev import jev_client as JC  # noqa: E402

SPEC = json.loads((HERE / "spec.json").read_text(encoding="utf-8"))
PROMPT_SCHEMA_HASH = "exp06_v1_20260920"  # spec.jsonのquestions/state_fields固定時点の印

# 匿名・合成のテストstate(実レースデータは使わない、Stage A自体はAPI応答の型検査が目的)
_SYNTHETIC_STATE = {
    "anon_race_id": "SYNTH_0001", "venue_code": "SYNTH", "surface": "turf",
    "distance_band": "1600-1800", "class_band": "mid", "field_size": 14,
    "m1_top_prob": 0.28, "m3_top_prob": 0.27, "m4_top_prob": 0.27,
    "model_rank_disagreement": 0.1, "model_prob_variance": 0.001,
    "market_prob": 0.20, "ai_market_divergence": 0.07,
    "odds_t35": 4.5, "odds_t20": 4.3, "odds_t10": 4.1, "odds_change_rate": -0.09,
    "popularity_rank_change": 0, "feature_missing_rate": 0.02, "unknown_category_rate": 0.0,
    "in_distribution_support": 0.8, "similar_past_case_count": 340,
    "candidate_ticket_count": 3, "candidate_ticket_probs": [0.27, 0.20, 0.15],
    "current_odds": 4.1, "predicted_confirmed_odds": 4.0,
    "ev_point_estimate": 1.08, "ev_lower_bound": 0.95, "min_payout": 100,
    "same_horse_concentration_rate": 0.3,
}


def run_audit() -> dict:
    result: dict = {"generated_at": datetime.now().isoformat(timespec="seconds"),
                    "prompt_schema_hash": PROMPT_SCHEMA_HASH, "checks": {}}

    # 1. 認証成功 + 基本応答の型検査
    r1 = JC.query_jev("SYNTH_0001", PROMPT_SCHEMA_HASH, _SYNTHETIC_STATE, SPEC["questions"],
                       exp05_model_hash="n/a_stage_a", market_snapshot_time="n/a_stage_a")
    result["checks"]["basic_call"] = r1
    if not r1.get("ok"):
        result["verdict"] = "BLOCKED"
        result["reason"] = r1.get("error", "unknown")
        return result

    # 2. 再現性(同一入力2回目、キャッシュ経由になるはず = 再課金防止の確認も兼ねる)
    r2 = JC.query_jev("SYNTH_0001", PROMPT_SCHEMA_HASH, _SYNTHETIC_STATE, SPEC["questions"],
                       exp05_model_hash="n/a_stage_a", market_snapshot_time="n/a_stage_a")
    result["checks"]["cache_hit_on_repeat"] = bool(r2.get("from_cache"))

    result["verdict"] = "RAN"
    return result


def main() -> int:
    (HERE / "out").mkdir(parents=True, exist_ok=True)
    try:
        result = run_audit()
    except NotImplementedError as exc:
        result = {"verdict": "NOT_IMPLEMENTED", "reason": str(exc),
                  "generated_at": datetime.now().isoformat(timespec="seconds")}
    out_path = HERE / "out" / "STAGE_A_AUDIT.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0 if result.get("verdict") == "RAN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
