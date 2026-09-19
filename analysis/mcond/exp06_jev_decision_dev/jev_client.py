# -*- coding: utf-8 -*-
"""
jev_client.py — TypeSafe AI Jev API向けの低レベルクライアント + append-onlyキャッシュ
================================================================================
spec.json の absolute_conditions を機械的に強制する層。呼び出しロジック自体
(質問文・state組み立て・EXP05との統合)はここには置かない(別ファイルの責務)。

【API仕様(2026-09-20、ユーザー提示の公式資料に基づく。推測実装ではない)】
  Endpoint: POST https://api.typesafe.ai/v1/systemone
  Authorization: Bearer $TYPESAFE_API_KEY
  Content-Type: application/json
  model: jev-latest (ユーザーの実機疎通確認では返却モデル jev-1.13.0)
  state: string/object/array (ここではobjectとしてstateをそのまま渡す)
  questions: noul/choice/score (spec.jsonのquestions[].typeから小文字マップ)
  429/529: 指数バックオフで再試行 (SDKの既定リトライに準ずる、下記参照)
参照: https://docs.typesafe.ai/api / https://docs.typesafe.ai/introduction/quickstart /
      https://docs.typesafe.ai/sdk

このファイルが保証すること:
  - APIキーは環境変数 TYPESAFE_API_KEY からのみ読む。読んだ値をログ・例外メッセージ・
    保存JSON・戻り値のいずれにも含めない。
  - 同一 input_hash への再問い合わせを避ける(ディスクキャッシュ、課金防止)。
  - 応答は append-only で保存する(同じキーへの2回目の呼び出しはキャッシュを返すだけで
    再送しない。ログファイル自体も追記のみ、上書きしない)。
  - API失敗(タイムアウト・429・5xx・例外)は例外を外へ伝播させず、呼び出し側が
    「この1レースはスキップ」と判断できる形(ok=False)で返す設計にする
    (EXP05-Fの非干渉設計と同じ思想)。429/529のみ指数バックオフで再試行、
    それ以外の失敗は線形バックオフで再試行する。

実行: このファイルは単体実行を想定しない。exp06本体のドライバから import して使う。
"""
from __future__ import annotations
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "out" / "jev_cache"
RESPONSES_DIR = HERE / "out" / "jev_responses"

TIMEOUT_S = 30.0
MAX_RETRIES = 2
RETRY_BACKOFF_S = 2.0


class JevConfigError(Exception):
    """APIキー未設定など、呼び出し前提の設定エラー(APIキー自体は含めない)。即時FAIL、リトライしない。"""


class JevFatalError(Exception):
    """401/403/422、およびそれ以外の4xx。設定またはリクエスト自体の問題なので
    即時FAIL、リトライしない(2026-09-20ユーザー指定のHTTP規則)。"""


class JevRateLimitError(Exception):
    """429/529 (rate limit / overload)。query_jev側で指数バックオフの対象にする(有限回)。"""


class JevTransientError(Exception):
    """timeout・接続エラー・一時的な5xx(429/529以外)。query_jev側で指数バックオフの
    対象にする(有限回、2026-09-20ユーザー指定のHTTP規則)。"""


def _get_api_key() -> str:
    key = os.environ.get("TYPESAFE_API_KEY", "")
    if not key:
        raise JevConfigError("TYPESAFE_API_KEY が環境変数に設定されていない。"
                             "コード・spec.json・ログにはキーを一切書かないこと。")
    return key


def compute_input_hash(prompt_schema_hash: str, state: dict) -> str:
    """state(匿名化済み診断量)+prompt_schema_hash から決定論的なハッシュを作る。
    同一入力への再課金防止のキーとして使う。"""
    canon = json.dumps({"prompt_schema_hash": prompt_schema_hash, "state": state},
                       ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


def _cache_path(input_hash: str) -> Path:
    return CACHE_DIR / f"{input_hash}.json"


def load_cached(input_hash: str) -> dict | None:
    p = _cache_path(input_hash)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"

# spec.json の questions[].type ("Noul"/"Choice"/"Score") → TypeSafe API の
# 期待する小文字表記への対応 (2026-09-20、ユーザー提示の公式仕様より)
_TYPE_MAP = {"Noul": "noul", "Choice": "choice", "Score": "score"}


def _questions_to_api_payload(questions: list[dict]) -> dict:
    """spec.json の questions 構造(リスト、このプロジェクトのドキュメント用の形)を
    TypeSafe API が期待する questions ペイロード(id をキーとする dict、各値は
    {type, instructions, criteria} のフラット構造。入れ子ラッパーやscale/options等の
    余剰キーは含めない)へ変換する。

    2026-09-20、ユーザー提示の実成功fixture(HTTP 200確認済み)に基づく正しい形:
      "Q1": {"type": "noul", "instructions": "...", "criteria": {"true": "...", "false": "..."}}
      "Q2": {"type": "score", "instructions": "...", "criteria": ["level0", ..., "level4"]}
      "Q3": {"type": "choice", "instructions": "...", "criteria": {"OPT": "desc", ...}}
    id/name/scale/options等はAPIに送らない(criteriaのlist長がscaleを兼ねる、
    optionsはcriteria dictのキーがそのまま選択肢になる)。"""
    out: dict = {}
    for q in questions:
        item = {"type": _TYPE_MAP[q["type"]], "instructions": q["instructions"]}
        if q["type"] == "Noul":
            item["criteria"] = q["criteria"]
        elif q["type"] == "Score":
            item["criteria"] = q["levels"]
        elif q["type"] == "Choice":
            item["criteria"] = q["option_descriptions"]
        out[q["id"]] = item
    return out


# HTTPステータス分類ルール (2026-09-20、ユーザー指定):
#   401/403/422           : JevFatalError (即時FAIL、リトライしない)
#   429/529                : JevRateLimitError (指数バックオフ、有限回)
#   timeout/接続エラー/その他5xx : JevTransientError (指数バックオフ、有限回)
#   その他4xx               : JevFatalError (即時FAIL、リトライしない)
_FATAL_STATUS = {401, 403, 422}


def _call_jev_api_raw(prompt_schema_hash: str, state: dict, questions: list[dict]) -> dict:
    """TypeSafe Jev API (POST /v1/systemone) への実際のHTTPコール。
    2026-09-20、ユーザー提示の公式仕様に基づく実装 (推測実装ではない):
      Endpoint: POST https://api.typesafe.ai/v1/systemone
      Authorization: Bearer $TYPESAFE_API_KEY
      Content-Type: application/json
      model: jev-latest
      state: string/object/array (ここではobject=stateそのものを渡す)
      questions: noul/choice/score (種別ごとに構造が異なる、_questions_to_api_payload参照)
    ステータスコード別の分類は上記 _FATAL_STATUS / JevRateLimitError / JevTransientError
    参照。例外メッセージにはステータスコードのみを含め、キー・Authorizationヘッダー・
    リクエストヘッダーは一切含めない(2026-09-20ユーザー指定)。
    レスポンスの正確なJSONキー名はstage_a_auditの初回実行結果で検証・記録する方針
    (このrawレスポンスをそのまま`_raw`として返し、query_jev側で緩く読む)。"""
    import requests  # 遅延import: このファイル自体はrequests無しでも読み込める設計を保つ

    api_key = _get_api_key()
    payload = {
        "model": JEV_MODEL,
        "state": state,
        "questions": _questions_to_api_payload(questions),
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(JEV_ENDPOINT, json=payload, headers=headers, timeout=TIMEOUT_S)
    except requests.exceptions.Timeout:
        raise JevTransientError("jev api timeout") from None
    except requests.exceptions.ConnectionError:
        raise JevTransientError("jev api connection error") from None

    sc = resp.status_code
    if sc in (429, 529):
        raise JevRateLimitError(f"jev api rate/overload status={sc}")
    if sc in _FATAL_STATUS:
        raise JevFatalError(f"jev api fatal status={sc}")
    if 500 <= sc < 600:
        raise JevTransientError(f"jev api transient server error status={sc}")
    if 400 <= sc < 500:
        raise JevFatalError(f"jev api fatal status={sc}")
    resp.raise_for_status()  # 想定外の非2xxが残っていた場合の最終防波堤
    body = resp.json()
    answers = body.get("answers") or body.get("results") or {}
    probabilities, confidence = _extract_probabilities_and_confidence(answers)
    return {
        "model": body.get("model", JEV_MODEL),
        "answers": answers,
        "probabilities": probabilities,
        "confidence": confidence,
        "usage": body.get("usage"),
        "http_status": sc,
        "_raw": body,  # stage_a_auditが未知フィールドの有無を検査できるよう生も残す
    }


def _extract_probabilities_and_confidence(answers: dict) -> tuple[dict, dict]:
    """実レスポンスでは probabilities/confidence は各質問の answers[qid] 配下に
    ネストされている(トップレベルには無い、2026-09-20実機確認済み)。qid をキーとする
    dictへ集約する。noulタイプはprobabilities/confidenceフィールド自体を持たないため、
    "noul"値をtrue/false二値の擬似probabilitiesとして構成する(confidenceは無し=None)。"""
    probs: dict = {}
    conf: dict = {}
    for qid, ans in (answers or {}).items():
        if not isinstance(ans, dict):
            continue
        atype = ans.get("type")
        if atype == "noul":
            v = ans.get("noul")
            if isinstance(v, (int, float)):
                probs[qid] = {"true": v, "false": 1.0 - v}
            conf[qid] = None
        else:
            probs[qid] = ans.get("probabilities")
            conf[qid] = ans.get("confidence")
    return probs, conf


def query_jev(race_id: str, prompt_schema_hash: str, state: dict, questions: list[dict],
             exp05_model_hash: str, market_snapshot_time: str) -> dict:
    """キャッシュ確認 → (無ければ)API呼び出し → append-only保存、を行う。
    例外は投げない(ok=Falseで返す、EXP05-F非干渉設計と同じ思想)。APIキー自体は
    戻り値のいかなるフィールドにも含めない。"""
    input_hash = compute_input_hash(prompt_schema_hash, state)
    cached = load_cached(input_hash)
    if cached is not None:
        return {**cached, "from_cache": True}

    decision_time = datetime.now().isoformat(timespec="seconds")
    try:
        _get_api_key()  # 存在確認のみ、値はここで捨てる(呼び出し先の関数が別途読む)
    except JevConfigError as exc:
        return {"ok": False, "error": str(exc), "race_id": race_id, "input_hash": input_hash,
               "from_cache": False}

    t0 = time.time()
    last_exc = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            raw = _call_jev_api_raw(prompt_schema_hash, state, questions)
            latency_ms = int((time.time() - t0) * 1000)
            record = {
                "ok": True,
                "timestamp": decision_time,
                "race_id": race_id,
                "input_hash": input_hash,
                "prompt_schema_hash": prompt_schema_hash,
                "returned_model": raw.get("model"),
                "all_answers": raw.get("answers"),
                "all_probabilities": raw.get("probabilities"),
                "confidence": raw.get("confidence"),
                "usage": raw.get("usage"),
                "latency_ms": latency_ms,
                "http_status": raw.get("http_status"),
                "retry_count": attempt,
                "exp05_model_hash": exp05_model_hash,
                "market_snapshot_time": market_snapshot_time,
                "decision_time": decision_time,
                "joined_result_and_payout": None,
                "from_cache": False,
            }
            _save_append_only(input_hash, record)
            return record
        except NotImplementedError:
            raise  # 未実装は隠さずそのまま伝播(呼び出し側に気づかせる)
        except (JevConfigError, JevFatalError) as exc:
            # 401/403/422・その他4xx・設定エラー: 即時FAIL、リトライしない
            # (2026-09-20ユーザー指定。設定/リクエスト自体の問題を何度叩いても直らない)
            return {"ok": False, "error": str(exc), "race_id": race_id, "input_hash": input_hash,
                   "retry_count": attempt, "from_cache": False}
        except (JevRateLimitError, JevTransientError) as exc:
            # 429/529・timeout・接続エラー・一時的5xx: 有限回の指数バックオフ
            last_exc = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_S * (2 ** attempt))
                continue
        except Exception as exc:
            # 分類されない想定外の失敗も安全側(即時FAIL・リトライしない)で扱う
            return {"ok": False, "error": str(exc), "race_id": race_id, "input_hash": input_hash,
                   "retry_count": attempt, "from_cache": False}
    return {"ok": False, "error": str(last_exc), "race_id": race_id, "input_hash": input_hash,
           "retry_count": MAX_RETRIES, "from_cache": False}


def _save_append_only(input_hash: str, record: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
    # キャッシュ(再課金防止用、input_hash単位で1件)
    _cache_path(input_hash).write_text(json.dumps(record, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    # append-onlyの応答ログ(監査用、日付ごとに追記のみ、上書きしない)
    day = datetime.now().strftime("%Y%m%d")
    log_path = RESPONSES_DIR / f"{day}.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
