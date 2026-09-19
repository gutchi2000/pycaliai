# -*- coding: utf-8 -*-
"""
stage_a_audit.py — Stage A: API・再現性監査 (spec.json Stage A参照)
================================================================================
実施すること:
  1. API疎通(認証成功)
  2. returned_model
  3. Choice/Score/Noulの型が期待通りか(schema検証)
  4. 確率和が1に近いか(質問ごと)・confidenceの範囲(質問ごと、Noulはconfidenceフィールド
     自体を持たないためNone、それ以外は必須)
  5. 同一入力を複数回送った際の変動(再現性。query_jevのキャッシュを経由せず
     _call_jev_api_rawを直接複数回叩く。回数は課金を抑えるため最小限=3回に限定)
  6. 日本語入力と英語入力の差(記録のみ、本評価の言語は英語固定のまま変更しない)
  7. latencyとusage
  8. キャッシュ再利用の確認(query_jev経由で2回目がfrom_cache=Trueになるか)

英語固定テンプレートを使う。ここでの比較結果を見て質問文や言語を変更しない
(spec §Stage A、変更してよいのは公式ドキュメント判明後の`_call_jev_api_raw`実装のみ)。

2026-09-20、ユーザー提示の実成功fixture(HTTP 200確認済み)に基づき
_questions_to_api_payload()の出力形式を確定させた後に実行:
  questions は id をキーとする dict、各値は {type, instructions, criteria} の
  フラット構造(id/name/scale/options等の余剰キーは送らない)。

実行: python analysis/mcond/exp06_jev_decision_dev/stage_a_audit.py
      (TYPESAFE_API_KEYが呼び出し元プロセスの環境変数に設定されている前提)
出力: analysis/mcond/exp06_jev_decision_dev/out/STAGE_A_AUDIT.json
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
PROMPT_SCHEMA_HASH = "exp06_v2_20260920"  # v1→v2: instructions/criteria追加でペイロード形状が変わったため印を更新

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

# 日本語版questions(spec.jsonの英語版と1:1対応、Stage Aの言語差記録専用。
# 本評価には使わない=結果を見て採用しない)。2026-09-20確定の実スキーマ
# (instructions/criteria)に合わせてある。
_JA_QUESTIONS = [
    {"id": "Q1", "name": "evidence_consistency", "type": "Noul",
     "instructions": "独立した予測シグナルと市場シグナルは、自動的な購入判断を支持するのに"
                     "十分整合している。",
     "criteria": {"true": "シグナルは自動判断を支持するのに十分整合している。",
                 "false": "シグナルは自動判断を支持するのに十分整合していない。"}},
    {"id": "Q2", "name": "out_of_distribution_risk", "type": "Score",
     "instructions": "このケースが類似の過去事例にどの程度支持されているかを評価する。",
     "levels": ["類似の過去事例に明確に支持されている", "概ね支持されている", "境界的",
               "過去事例による支持が弱い", "過去事例の支持範囲外"]},
    {"id": "Q3", "name": "market_price_risk", "type": "Score",
     "instructions": "現在の市場価格が判断実行前に不利に変動する、または信頼できないリスクを評価する。",
     "levels": ["価格は安定しているように見える", "通常範囲の小さな変動", "中程度の不確実性",
               "不利な変動リスクが高い", "現在の価格は信頼できない"]},
    {"id": "Q4", "name": "trust_source", "type": "Choice",
     "instructions": "この判断でどの情報源を信頼すべきか選択する。",
     "option_descriptions": {"AI": "予測モデルの推定値を市場価格より信頼する。",
                             "MARKET": "市場価格を予測モデルの推定値より信頼する。",
                             "BLEND": "予測モデルの推定値と市場価格を組み合わせる。",
                             "ABSTAIN": "どちらの情報源も判断根拠として信頼できない。"}},
    {"id": "Q5", "name": "participation", "type": "Choice",
     "instructions": "このレースに参加するかどうか、参加しない場合はその理由を決定する。",
     "option_descriptions": {"BET": "自動的にこのレースへ参加する。",
                             "PASS_NO_EDGE": "市場に対する優位性が無いため参加しない。",
                             "PASS_UNCERTAIN": "根拠が不確実すぎるため参加しない。",
                             "PASS_OOD": "過去事例の支持範囲外のため参加しない。",
                             "PASS_PRICE_RISK": "市場価格が不利に変動するリスクが高いため参加しない。"}},
    {"id": "Q6", "name": "ticket_policy", "type": "Choice",
     "instructions": "このレースの馬券方針を選択する。",
     "option_descriptions": {"NO_BET": "このレースには一切賭けない。",
                             "PLACE_SINGLE": "単一の馬の複勝のみ賭ける。",
                             "WIN_SINGLE": "単一の馬の単勝のみ賭ける。",
                             "WIDE_ONE": "ワイド1点を賭ける。",
                             "WIDE_TWO": "ワイド2点を賭ける。",
                             "CURRENT_TOPDOWN": "現行の本番topdown馬券方針に従う。"}},
]


def _validate_schema(answers: dict | None, probabilities: dict | None, confidence: dict | None) -> dict:
    """Choice/Score/Noulの型・質問ごとの確率和・confidence範囲を検証する。
    (2026-09-20訂正: probabilities/confidenceは質問ごとのdict、トップレベル単一値ではない)"""
    checks: dict = {"per_question": {}, "overall_ok": True}
    by_id = {q["id"]: q for q in SPEC["questions"]}
    probs = probabilities or {}
    confs = confidence or {}
    for qid, q in by_id.items():
        entry = {"type": q["type"]}
        p = probs.get(qid)
        if p is None:
            entry["probability_present"] = False
            entry["ok"] = False
            checks["overall_ok"] = False
        else:
            entry["probability_present"] = True
            if isinstance(p, dict):
                total = sum(v for v in p.values() if isinstance(v, (int, float)))
                entry["probability_sum"] = total
                entry["probability_sum_near_1"] = abs(total - 1.0) < 0.05
                entry["all_in_0_1"] = all(0.0 <= v <= 1.0 for v in p.values()
                                          if isinstance(v, (int, float)))
                entry["ok"] = entry["probability_sum_near_1"] and entry["all_in_0_1"]
            else:
                entry["ok"] = False
            if not entry.get("ok", False):
                checks["overall_ok"] = False

        c = confs.get(qid)
        if q["type"] == "Noul":
            entry["confidence_expected"] = False
            entry["confidence_value"] = c
        else:
            entry["confidence_expected"] = True
            entry["confidence_value"] = c
            entry["confidence_in_0_1"] = isinstance(c, (int, float)) and 0.0 <= c <= 1.0
            if not entry["confidence_in_0_1"]:
                checks["overall_ok"] = False
        checks["per_question"][qid] = entry
    return checks


def run_audit() -> dict:
    result: dict = {"generated_at": datetime.now().isoformat(timespec="seconds"),
                    "prompt_schema_hash": PROMPT_SCHEMA_HASH, "checks": {}}

    # 1+2+7+8. API疎通・returned_model・latency/usage・キャッシュ再利用
    r1 = JC.query_jev("SYNTH_0001", PROMPT_SCHEMA_HASH, _SYNTHETIC_STATE, SPEC["questions"],
                       exp05_model_hash="n/a_stage_a", market_snapshot_time="n/a_stage_a")
    result["checks"]["basic_call"] = {k: v for k, v in r1.items() if k != "all_answers"}
    result["checks"]["basic_call"]["answers_present"] = r1.get("all_answers") is not None
    if not r1.get("ok"):
        result["verdict"] = "FAIL"
        result["reason"] = r1.get("error", "unknown")
        return result

    r2 = JC.query_jev("SYNTH_0001", PROMPT_SCHEMA_HASH, _SYNTHETIC_STATE, SPEC["questions"],
                       exp05_model_hash="n/a_stage_a", market_snapshot_time="n/a_stage_a")
    result["checks"]["cache_hit_on_repeat"] = bool(r2.get("from_cache"))

    # 3+4. schema検証・確率和・confidence範囲(質問ごと)
    result["checks"]["schema_validation"] = _validate_schema(
        r1.get("all_answers"), r1.get("all_probabilities"), r1.get("confidence"))

    # 5. 再現性(cacheを経由せずrawを複数回、最小限=3回。課金を抑えるためこの回数に限定)
    repro_runs = []
    for _ in range(3):
        try:
            raw = JC._call_jev_api_raw(PROMPT_SCHEMA_HASH, _SYNTHETIC_STATE, SPEC["questions"])
            repro_runs.append({"answers": raw.get("answers")})
        except Exception as exc:
            repro_runs.append({"error": str(exc)})
    result["checks"]["reproducibility_raw_runs"] = repro_runs
    answer_reprs = [json.dumps(r.get("answers"), sort_keys=True, ensure_ascii=False)
                    for r in repro_runs if "answers" in r]
    result["checks"]["reproducibility_identical_across_runs"] = (
        len(set(answer_reprs)) <= 1 if answer_reprs else None)

    # 6. 日本語/英語差(記録のみ)
    try:
        raw_ja = JC._call_jev_api_raw(PROMPT_SCHEMA_HASH + "_ja_diag", _SYNTHETIC_STATE, _JA_QUESTIONS)
        result["checks"]["ja_vs_en"] = {
            "en_answers": r1.get("all_answers"), "ja_answers": raw_ja.get("answers"),
            "identical": json.dumps(r1.get("all_answers"), sort_keys=True, ensure_ascii=False)
                        == json.dumps(raw_ja.get("answers"), sort_keys=True, ensure_ascii=False),
        }
    except Exception as exc:
        result["checks"]["ja_vs_en"] = {"error": str(exc)}

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
