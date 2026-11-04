# -*- coding: utf-8 -*-
"""
input_quality.py — 予測入力の品質ゲート (馬同定) の単一ソース
================================================================
仕様: docs/research/SERVE_REPRODUCIBILITY_FIX_SPEC_20261004.md §3.3 (2026-10-04 Fable 決定で確定)

  レース単位: 馬同定が判別不能 (ambiguous) の馬を 1 頭でも含むレースは見送り。
              export_weekly_marks が bundle の各レースに input_quality を書き、
              compute_bets (T-10)・t20_site_bets (T-20 速報)・shadow_a95 (紙上台帳) が読む。
  開催日単位: 判別不能が出走馬の 2% を超えた開催日は全体を品質不合格 (bundle を公開しない)。

予測経路の版: policy が serve_path_version を宣言しているとき、レースの版
(input_quality.serve_path_version、記録が無ければ sp-20261004-0) が違えば見送る。
旧版と新版の記録を同じ cohort に混ぜないため。
"""
from __future__ import annotations

import re

GATE_ID = "identity_ambiguous"
IDENTITY_AMBIGUOUS_DAY_MAX = 0.02      # 開催日の判別不能率の上限 (これを超えたら不合格)
IDENTITY_STATUSES = ("hit", "new", "ambiguous")


def _rid16(x) -> str:
    return re.sub(r"\D", "", str(x or ""))[:16]


def identity_map(df, rid_col: str, ban_col: str, status_col: str) -> dict[tuple[str, int], str]:
    """出走表 frame の馬ごとの同定状態 → {(rid16, 馬番): status}。"""
    out: dict[tuple[str, int], str] = {}
    if status_col not in df.columns:
        return out
    for rid, ban, st in zip(df[rid_col], df[ban_col], df[status_col]):
        try:
            out[(_rid16(rid), int(ban))] = str(st)
        except (TypeError, ValueError):
            continue
    return out


def annotate_race(payload: dict, ident: dict[tuple[str, int], str], serve_path_version: str) -> dict:
    """bundle の 1 レースに馬ごとの identity_status とレースの input_quality を書く。

    同定状態が分からない馬 (frame に無い) は unknown とし、判別不能と同じく見送りにする。
    """
    rid = _rid16(payload.get("race_id"))
    counts = {k: 0 for k in (*IDENTITY_STATUSES, "unknown")}
    bad = []
    for h in payload.get("horses", []):
        try:
            st = ident.get((rid, int(h.get("umaban"))), "unknown")
        except (TypeError, ValueError):
            st = "unknown"
        st = st if st in IDENTITY_STATUSES else "unknown"
        h["identity_status"] = st
        counts[st] += 1
        if st in ("ambiguous", "unknown"):
            bad.append({"umaban": h.get("umaban"), "horse_name": h.get("horse_name"), "status": st})
    iq = {"serve_path_version": serve_path_version, "identity": counts,
          "bet_eligible": not bad, "gate": GATE_ID}
    if bad:
        iq["reasons"] = ["identity_ambiguous"]
        iq["ambiguous_horses"] = bad
    payload["input_quality"] = iq
    return iq


def day_summary(races: list[dict], serve_path_version: str) -> dict:
    """開催日の判別不能率と見送りレース。rate > IDENTITY_AMBIGUOUS_DAY_MAX なら day_pass=False。"""
    n = n_bad = 0
    skipped = []
    for r in races:
        iq = r.get("input_quality") or {}
        c = iq.get("identity") or {}
        n += sum(int(c.get(k, 0)) for k in (*IDENTITY_STATUSES, "unknown"))
        n_bad += int(c.get("ambiguous", 0)) + int(c.get("unknown", 0))
        if iq.get("bet_eligible") is False:
            skipped.append({"race_id": r.get("race_id"), "ambiguous_horses": iq.get("ambiguous_horses", [])})
    rate = (n_bad / n) if n else 0.0
    return {"gate": GATE_ID, "serve_path_version": serve_path_version,
            "n_horses": n, "n_ambiguous": n_bad, "rate": round(rate, 6),
            "day_max": IDENTITY_AMBIGUOUS_DAY_MAX, "day_pass": rate <= IDENTITY_AMBIGUOUS_DAY_MAX,
            "skipped_races": skipped}


def race_skip_reason(race: dict, policy_serve_path_version: str | None = None) -> str | None:
    """見送り理由 (無ければ None)。compute_bets / t20_site_bets / shadow_a95 が共通で使う。"""
    from serve_path import LEGACY_VERSION
    iq = race.get("input_quality") or {}
    if policy_serve_path_version:
        ver = iq.get("serve_path_version") or LEGACY_VERSION
        if ver != policy_serve_path_version:
            return f"予測経路の版不一致 (bundle={ver}, policy={policy_serve_path_version})"
    if iq.get("bet_eligible") is False:
        bans = ",".join(str(x.get("umaban")) for x in iq.get("ambiguous_horses", []))
        return f"予測入力の品質ゲート: 馬同定が判別不能 (馬番 {bans})"
    return None
