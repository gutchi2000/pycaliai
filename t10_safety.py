"""T-10 runnerのfail-closed書込と締切価格取得。"""
from __future__ import annotations

import time
from typing import Callable


def force_skip(date_str: str, rid16: str, label: str, reason: str,
               notify_fn: Callable[[str], bool] | None = None) -> None:
    """取得・計算・validatorのどこで失敗しても、古い買い目を空に置換する。"""
    from compute_bets import ENGINE_VERSION, apply_to_bets_json
    apply_to_bets_json(date_str, [{
        "race_id": rid16,
        "race_label": label,
        "race_nature": "見送り",
        "race_reason": reason + " (fail-closed)",
        "bets": [],
    }], stamp={"model": "v6", "engine": "t10_fail_closed",
              "engine_version": ENGINE_VERSION})
    if notify_fn:
        notify_fn(f"⚠ {label} **見送り** — {reason}")


def capture_close_price(rid16: str, scheduled_post: str,
                        py32: list[str], run_cmd: Callable,
                        notify_fn: Callable[[str], bool] | None = None,
                        retries: int = 3, retry_delay_sec: int = 10) -> bool:
    """締切後のJV-Link価格を取得し、forward ledgerへ保存する。"""
    cmd = [*py32, "jvlink_odds.py", "--race", rid16,
           "--stage", "close", "--scheduled-post", scheduled_post]
    last_rc = -1
    for attempt in range(1, max(1, retries) + 1):
        last_rc, out = run_cmd(cmd)
        line = next((x for x in out.splitlines() if "[jvlink_odds]" in x),
                    out.strip()[-300:])
        print(f"  [close] jvlink_odds attempt={attempt}/{retries} "
              f"(exit {last_rc}) {line}")
        if last_rc == 0:
            return True
        if attempt < retries:
            time.sleep(max(0, retry_delay_sec))
    if notify_fn:
        notify_fn(f"⚠ {rid16} 締切価格の保存失敗 (exit {last_rc})")
    return False
