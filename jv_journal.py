# -*- coding: utf-8 -*-
"""
jv_journal.py — JV-Link 取得ジャーナル（観測計画 v2.1 §3 Stage 0 (6) 並走成功率の一次記録）
======================================================================================
JV-Link を叩くすべての取得（本番 t10/t20/close_late/exp05fs_t35、T−2 候補、三連複 shadow、
final 候補）について、開始・終了時刻（ms, JST）、rc、返却録数を 1 取得 1 ファイルで追記保存する。
複数プロセスが同時に書くため 1 ファイルに追記せず、ファイル名に開始時刻・pid・spec を入れて衝突させない。

書込は fail-open: 失敗しても呼び出し元（本番の価格取得）の挙動・終了コードを一切変えない。
成否（race_key 一致・全組被覆）は取得後の構造化結果（forward_prices の jv_captures）と
開始時刻で突き合わせる（analysis/obs_stage0_audit.py）。
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
JOURNAL_ROOT = BASE / "data" / "jvlink_fetch_journal"
DRY_JOURNAL_ROOT = BASE / "data" / "jvlink_fetch_journal_dry"
_CONTEXT: dict = {"process": "unknown", "stage": None, "race_id": None, "dry": False}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def set_context(**kw) -> None:
    _CONTEXT.update(kw)


def journal_root() -> Path:
    env = os.environ.get("PYCALIAI_JV_JOURNAL_ROOT")
    if env:
        return Path(env)
    return DRY_JOURNAL_ROOT if _CONTEXT.get("dry") else JOURNAL_ROOT


def default_process() -> str:
    """set_context で名前が付いていない取得の process 名 = 実行中スクリプトの repo 相対パス（拡張子なし）。
    例: analysis/bodyweight_forward/collector。`unknown` のまま記録しないための fallback。"""
    try:
        argv0 = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] not in ("", "-c", "-") else None
        if argv0 is None:
            return "python"
        try:
            return argv0.relative_to(BASE).with_suffix("").as_posix()
        except ValueError:
            return argv0.stem or "python"
    except Exception:
        return "python"


def write_event(race_key: str, spec: str, meta: dict) -> Path | None:
    try:
        started = str(meta.get("fetch_started_at") or now_iso())
        compact = "".join(ch for ch in started[:23] if ch.isdigit())
        day = compact[:8] or datetime.now().strftime("%Y%m%d")
        event = {**_CONTEXT, "race_id": race_key or _CONTEXT.get("race_id"), "spec": spec,
                 "pid": os.getpid(), **{k: meta.get(k) for k in (
                     "fetch_started_at", "fetch_finished_at", "rc_init", "rc_open",
                     "n_records_returned", "error", "stream")}}
        if event.get("process") in (None, "", "unknown"):
            event["process"], event["process_source"] = default_process(), "argv"
        else:
            event["process_source"] = "context"
        root = journal_root() / day
        root.mkdir(parents=True, exist_ok=True)
        safe_proc = str(event.get("process")).replace("/", ".")
        name = f"{compact}_{os.getpid()}_{safe_proc}_{spec}_{race_key}.json"
        path = root / name
        n = 1
        while path.exists():
            path = root / f"{name[:-5]}__{n}.json"
            n += 1
        path.write_text(json.dumps(event, ensure_ascii=False), encoding="utf-8")
        return path
    except Exception:
        return None
