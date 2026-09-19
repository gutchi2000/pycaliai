# -*- coding: utf-8 -*-
"""
jev_client.py — TypeSafe AI Jev API向けの低レベルクライアント + append-onlyキャッシュ
================================================================================
spec.json の absolute_conditions を機械的に強制する層。呼び出しロジック自体
(質問文・state組み立て・EXP05との統合)はここには置かない(別ファイルの責務)。

【現状: HTTP呼び出し部分は未実装(要API仕様)】
`_call_jev_api_raw()` はTypeSafe Jev APIの実際のエンドポイント・認証ヘッダ形式・
リクエスト/レスポンススキーマが不明なため、意図的に未実装(NotImplementedError)の
ままにしてある。推測で実装すると「一見動くが実際には壊れている」コードになり、
ユーザーの実APIコール(課金対象)を無駄にするリスクが高いため、公式ドキュメントか
サンプルレスポンスの提示を待ってから実装する方針(このプロジェクトの既定方針:
外部APIは1回の否定的結果や推測ではなく実データで検証する)。

このファイル自体が保証すること(API仕様が未確定でも先に固められる部分):
  - APIキーは環境変数 TYPESAFE_API_KEY からのみ読む。読んだ値をログ・例外メッセージ・
    保存JSON・戻り値のいずれにも含めない。
  - 同一 input_hash への再問い合わせを避ける(ディスクキャッシュ、課金防止)。
  - 応答は append-only で保存する(同じキーへの2回目の呼び出しは新しいrevisionを作る、
    上書きしない)。
  - API失敗(タイムアウト・429・5xx・例外)は例外を外へ伝播させず、呼び出し側が
    「この1レースはスキップ」と判断できる形(ok=False)で返す設計にする
    (EXP05-Fの非干渉設計と同じ思想)。

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
    """APIキー未設定など、呼び出し前提の設定エラー(APIキー自体は含めない)。"""


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


def _call_jev_api_raw(prompt_schema_hash: str, state: dict, questions: list[dict]) -> dict:
    """実際のHTTPコール。未実装 — TypeSafe Jev APIの仕様(エンドポイント・
    認証ヘッダ形式・リクエストボディ・Choice/Score/Noulのレスポンス表現)が
    判明次第ここに実装する。それまでこの関数を呼ぶと明示的にNotImplementedError。"""
    raise NotImplementedError(
        "TypeSafe Jev APIの仕様が未確定のため未実装。エンドポイントURL・認証方式・"
        "リクエスト/レスポンスのサンプル(できれば実際の1回分のレスポンスJSON)を"
        "確認してから実装すること。推測実装はしない方針(既存の教訓: 外部APIの"
        "挙動は1回の失敗/推測で結論づけず実データで検証する)。")


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
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_S * (attempt + 1))
                continue
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
