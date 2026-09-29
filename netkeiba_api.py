# -*- coding: utf-8 -*-
"""netkeiba_api.py — AI競馬予想マスターズ2026 学生大会 投票API クライアント

公式仕様 (参加マニュアル / API説明):
  login   POST https://masters.netkeiba.com/ai2026_student/api/login
          params login_id / password → data.access_token (有効 300 秒)
  bet     POST https://masters.netkeiba.com/ai2026_student/api/bet
          header Authorization: Bearer <token> / param (array) bet_data
          ★エラー時は 1 件も処理されない (status=NG なら全件未処理)
  check   GET  https://masters.netkeiba.com/ai2026_student/api/bet
          param (array) race_id  ★投票直後は非同期のため 1 分ほど空けて呼ぶ
  logout  POST https://masters.netkeiba.com/ai2026_student/api/logout

本モジュールは HTTP と資格情報だけを扱う。買い目の組み立て・検証は masters_vote.py。

資格情報 (このファイルは .gitignore 済み。人間が自分で書く):
  data/netkeiba_api.json = {"login_id": "...", "password": "...", "enabled": true}
  環境変数 NETKEIBA_LOGIN_ID / NETKEIBA_PASSWORD が優先。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import requests

BASE = Path(__file__).resolve().parent
CONFIG_PATH = BASE / "data" / "netkeiba_api.json"

LOGIN_URL = "https://masters.netkeiba.com/ai2026_student/api/login"
BET_URL = "https://masters.netkeiba.com/ai2026_student/api/bet"
LOGOUT_URL = "https://masters.netkeiba.com/ai2026_student/api/logout"

TIMEOUT_SEC = 30
TOKEN_TTL_SEC = 300


class NetkeibaApiError(RuntimeError):
    """API 呼び出しが status=OK を返さなかった / 通信に失敗した。"""


# ---------------------------------------------------------------- 資格情報
def load_config() -> dict:
    """資格情報と運用フラグを返す。存在しない項目は空文字。"""
    cfg: dict = {}
    try:
        loaded = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            cfg = loaded
    except (OSError, ValueError):
        cfg = {}
    login_id = os.environ.get("NETKEIBA_LOGIN_ID") or str(cfg.get("login_id") or "")
    password = os.environ.get("NETKEIBA_PASSWORD") or str(cfg.get("password") or "")
    cfg["login_id"] = login_id.strip()
    cfg["password"] = password
    cfg.setdefault("enabled", False)
    return cfg


def credentials_ready() -> bool:
    cfg = load_config()
    return bool(cfg["login_id"] and cfg["password"])


# ---------------------------------------------------------------- 共通処理
def _response_data(r: requests.Response, what: str) -> Any:
    """公式サンプル getResponseData と同じ契約。status!=OK は例外にする。"""
    try:
        body = r.json()
    except ValueError as exc:
        raise NetkeibaApiError(
            f"{what}: JSON でない応答 (HTTP {r.status_code}) {r.text[:300]!r}") from exc
    if not isinstance(body, dict) or body.get("status") != "OK":
        raise NetkeibaApiError(f"{what}: status!=OK (HTTP {r.status_code}) "
                               f"{json.dumps(body, ensure_ascii=False)[:500]}")
    if "data" not in body:
        raise NetkeibaApiError(f"{what}: data 欠落 "
                               f"{json.dumps(body, ensure_ascii=False)[:300]}")
    return body


def _request(method: str, url: str, *, what: str, retries: int = 2,
             retry_delay_sec: float = 2.0, **kwargs) -> dict:
    """通信エラーのみ再試行する。status=NG は再試行しない (仕様上、未処理確定)。"""
    last: Exception | None = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            r = requests.request(method, url, timeout=TIMEOUT_SEC, **kwargs)
            return _response_data(r, what)
        except NetkeibaApiError:
            raise
        except requests.RequestException as exc:
            last = exc
            if attempt < retries:
                time.sleep(retry_delay_sec)
    raise NetkeibaApiError(f"{what}: 通信失敗 {last}")


# ---------------------------------------------------------------- 各 API
def login(login_id: str | None = None, password: str | None = None) -> str:
    """アクセストークン (有効 300 秒) を取得する。"""
    if login_id is None or password is None:
        cfg = load_config()
        login_id = login_id if login_id is not None else cfg["login_id"]
        password = password if password is not None else cfg["password"]
    if not login_id or not password:
        raise NetkeibaApiError(
            "login_id/password 未設定 (data/netkeiba_api.json か "
            "環境変数 NETKEIBA_LOGIN_ID / NETKEIBA_PASSWORD)")
    body = _request("POST", LOGIN_URL, what="login",
                    json={"login_id": login_id, "password": password},
                    headers={"Content-Type": "application/json"})
    token = (body.get("data") or {}).get("access_token")
    if not token:
        raise NetkeibaApiError("login: access_token が空")
    return str(token)


def post_bets(token: str, bet_data: list[dict]) -> dict:
    """投票する。エラー時は 1 件も処理されない (公式仕様)。

    戻り値は data 部 + status。error_count>0 の判定は呼び出し側で行う
    (status=OK でも個別レースが list_error に落ちることがあるため)。
    """
    if not bet_data:
        raise NetkeibaApiError("post_bets: bet_data が空")
    body = _request("POST", BET_URL, what="bet",
                    json={"bet_data": bet_data},
                    headers={"Content-Type": "application/json",
                             "Authorization": f"Bearer {token}"})
    return body


def check_bets(token: str, race_ids: list[str]) -> list[dict]:
    """投票内容を確認する。★投票直後は非同期のため 1 分ほど空けて呼ぶこと。"""
    params = {f"race_id[{i}]": rid for i, rid in enumerate(race_ids)}
    body = _request("GET", BET_URL, what="bet(check)", params=params,
                    headers={"Authorization": f"Bearer {token}"})
    data = body.get("data")
    return data if isinstance(data, list) else []


def logout(token: str) -> bool:
    """明示ログアウト。失敗しても 5 分で自動失効するため非致命。"""
    try:
        _request("POST", LOGOUT_URL, what="logout", retries=1,
                 headers={"Authorization": f"Bearer {token}"})
        return True
    except NetkeibaApiError:
        return False
