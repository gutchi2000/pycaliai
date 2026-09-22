# -*- coding: utf-8 -*-
"""masters_vote.py — AI競馬予想マスターズ2026 学生大会 自動投票

============================ 締切と時刻設計 ============================
公式マニュアル: **投票は枠番発表後から発走時刻 3 分前まで**。再投票は上書き。
したがって「2 分前オッズ」は構造的に使えない (その時刻には投票が閉じている)。
本モジュールは以下で回す:

  T-4:00  JV-Link で当該レースのオッズを取得 (stage=vote, 本番 T-10 とは別ファイル)
  T-3:5x  買い目を組み立てて POST  (hard deadline = 発走 3 分前 - safety)
  T-2:5x  check API で「実際に登録されたか」を確認 (公式推奨: 投票後 1 分空ける)
          → ログアウト

本番 T-10 ライン (t10_runner.py) には一切干渉しない。事前登録 shadow
(wide_residual_shadow_v3) の台帳も書き換えない。本モジュールは同じ policy 演算を
**T-4 の価格で**独立に評価し、その結果を大会 API に流すだけ。

============================ 買い目 ============================
arm = "residual" (既定): ワイド残差 2 点 = wide_residual_shadow_v3 の Arm A。
  model 上位 2 ペアのうち residual = p_model - p_market_fair が [0, 0.05) のものを
  最大 2 点。hard gate (chaos percentile / field_size / ◎p_win) を通ったレースのみ。
filler: Arm A が不発のレースを埋める補助アーム。既定 null (=投票しない)。
  "model_top2" = 同じ hard gate を通ったレースで model 上位 2 ペアを買う (M1 対照)。

arm = "aite_switch": ◎オッズ×相手オッズで券種を切り替える (2026-09-05 実測配線)。
  相手 = model 本命馬連ペアのうち ◎ でない方の単勝オッズ。
    相手オッズ >= 10倍           → ワイド上位2点 (馬連は的中0%まで崩壊する帯)
    相手オッズ < 10倍 かつ ◎ < 2倍 → ワイド上位2点 + 馬連本命1点 (両方、n=10だが両方黒字圏)
    それ以外 (相手 < 10倍, ◎ >= 2倍) → 馬連本命1点 (ROI最良帯)
  hard gate は "residual" と共通 (production_policy.hard_skip_reasons)。
  ★2026-09-05 時点で 107R (10開催日) の後ろ向き検証のみ。前向き検証は未実施。

★実測 (過去 673R の実 T-10 データ, "residual" 単独): 発火率は 9.4% ≈ 3R/日。
  9 日間で ≈27R にしかならず、大会の最低投票要件 (96R) には届かない。
  filler="model_top2" で ≈8.9R/日 (9 日 ≈80R)、gate も外すと ≈14R/日 (9 日 ≈127R)。
  設定は data/masters_vote.json で切り替える。

実行:
  python masters_vote.py --race 2026082905030201 --date 20260829
  python masters_vote.py --race ... --dry          # 送信せずペイロードだけ表示
  python masters_vote.py --date 20260829 --verify  # 送信済みを check API で再確認
  python masters_vote.py --summary                 # 96R / 50万pt 進捗
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

BASE = Path(__file__).resolve().parent
PY32 = ["py", "-3.12-32"]                      # JV-Link は 32-bit COM
VOTE_ODDS_DIR = BASE / "reports" / "vote_odds"  # 本番 T-10 の live_odds とは別
LEDGER_DIR = BASE / "reports" / "masters_vote"
CONFIG_PATH = BASE / "data" / "masters_vote.json"

MARK_CODE = {"◎": 1, "〇": 2, "○": 2, "▲": 3, "△": 4}
SINGLE_MARKS = (1, 2, 3)                        # ◎〇▲ は 1 頭まで
BET_WIDE_BARA = "b5_c0_{a}_{b}"                 # ワイド・バラ買い (識別5)
BET_UMAREN_BARA = "b4_c0_{a}_{b}"               # 馬連・バラ買い (識別4)
AITE_ODDS_LONGSHOT = 10.0                       # 相手オッズがこれ以上→ワイドへ退避
HON_ODDS_CHALK = 2.0                            # ◎オッズがこれ未満→両方乗せ
# bet_data 要素の買い目キー。公式「投票」サンプルが手に入らなかったため、
# 2026-08-29 に実 API へ候補を投げて確定させた ("bet_list" は「書式が不正
# (bet項目不明)」で拒否、"bet" + money=int が受理されチェック API で一致)。
BET_KEY = "bet"

DEFAULT_CONFIG = {
    "enabled": False,
    "arm": "residual",                          # "residual" | "aite_switch"
    "arm_before": None,                         # arm_effective_from より前の日だけ使う arm (None=常にarm)
    "arm_effective_from": None,                 # "YYYYMMDD"。この日より前は arm_before を使う
    "filler": None,
    "ignore_hard_gate": False,
    "use_official_timetable": True,
    "notify_skips": False,
    "stake_per_ticket_yen": 5000,
    "fetch_lead_min": 4.5,
    "submit_safety_sec": 20,
    "check_delay_sec": 60,
    "max_points_per_race": 200,
    "fallback_wide_min100": False,
    "required_races": 96,
    "required_total_yen": 500000,
    "race_days": ['20260829', '20260830', '20260905', '20260906', '20260912', '20260913', '20260919', '20260920', '20260921'],
}

sys.stdout.reconfigure(encoding="utf-8")


class VoteError(RuntimeError):
    """投票を中止すべき状態。実行は fail-closed (投票しない) で終える。"""


# ------------------------------------------------------------------ 設定
def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    try:
        loaded = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            cfg.update(loaded)
    except (OSError, ValueError):
        pass
    return cfg


# ------------------------------------------------------------------ 変換
def rid16(value) -> str:
    return re.sub(r"\D", "", str(value or ""))[:16]


def netkeiba_race_id(rid: str) -> str:
    """TARGET 16 桁 → netkeiba 12 桁。

    16 桁 = 年(4) 月日(4) 場(2) 回(2) 日(2) R(2)
    12 桁 = 年(4) 場(2) 回(2) 日(2) R(2)   (月日を落とす)

    すでに 12 桁 (netkeiba の race_id をそのまま貼った場合) はそのまま通す。
    """
    rid = re.sub(r"\D", "", str(rid or ""))
    if len(rid) == 12:
        return rid
    rid = rid[:16]
    if len(rid) != 16:
        raise VoteError(f"race_id が 12 桁でも 16 桁でもない: {rid!r}")
    return rid[:4] + rid[8:16]


def parse_hhmm(s: str) -> tuple[int, int] | None:
    s = str(s or "").strip()
    m = re.match(r"^(\d{1,2})[:時](\d{2})", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.match(r"^(\d{3,4})$", s)
    if m:
        return int(m.group(1)) // 100, int(m.group(1)) % 100
    return None


def post_datetime(date_str: str, rid: str, override: str | None) -> datetime:
    """発走予定時刻。--post > 大会 当日データAPI の公式 timetable > weekly CSV。

    大会サーバはこの公式時刻の 4 分 30 秒前にオッズを取る。当日の発走時刻変更は
    weekly CSV (朝のスナップショット) に載らないので、公式側を先に見る。
    """
    hm = parse_hhmm(override) if override else None
    if hm is None and load_config().get("use_official_timetable", True):
        try:
            from masters_dayapi import DayApiError, post_times
            hm = parse_hhmm(post_times(date_str).get(rid, ""))
            if hm:
                print("  発走時刻: 大会 当日データAPI の公式 timetable を使用")
        except (ImportError, DayApiError) as exc:
            print(f"  [info] 当日データAPI 使用不可 ({exc}) → weekly CSV にフォールバック")
    if hm is None:
        from t10_runner import load_post_times
        hm = parse_hhmm(load_post_times(date_str).get(rid, ""))
    if hm is None:
        raise VoteError(f"発走時刻不明 (当日データAPI にも data/weekly/{date_str}.csv にも無い): {rid}")
    day = datetime.strptime(date_str, "%Y%m%d")
    return day.replace(hour=hm[0], minute=hm[1], second=0, microsecond=0)


# ------------------------------------------------------------------ 印
def build_marks(horses: list[dict], active: set[int]) -> dict[str, int]:
    """bundle の印 → {"馬番": 印番号}。◎必須・◎〇▲は各1頭・△は複数可。

    取消等で ◎〇▲ が不在になった場合は ai_rank 順の現存馬で埋める
    (◎ が無いと大会側で登録できないため)。
    """
    ranked = sorted(
        (h for h in horses if int(h.get("umaban", 0)) in active),
        key=lambda h: (h.get("ai_rank") if isinstance(h.get("ai_rank"), int) else 999,
                       int(h.get("umaban", 0))))
    if not ranked:
        raise VoteError("現存馬が 0 頭 (印を作れない)")

    marks: dict[int, int] = {}
    for h in ranked:
        code = MARK_CODE.get(str(h.get("mark") or "").strip())
        if code is None:
            continue
        if code in SINGLE_MARKS and code in marks.values():
            continue                                  # ◎〇▲ の重複は捨てる
        marks[int(h["umaban"])] = code

    # 取消等で ◎〇▲ が欠けたら、格上げで詰める (〇→◎ → ▲→〇 → 印なし→▲)。
    # 印なし馬をいきなり ◎ にすると ◎勝率 (順位タイブレーク) を無駄に落とす。
    for code in SINGLE_MARKS:
        if code in marks.values():
            continue
        for h in ranked:
            ban = int(h["umaban"])
            if marks.get(ban, 99) > code:
                marks[ban] = code
                break
    if 1 not in marks.values():
        raise VoteError("◎ を決められない (現存馬不足)")
    return {str(k): v for k, v in sorted(marks.items())}


# ------------------------------------------------------------------ 買い目
def wide_bet_id(selection: str) -> str:
    """'3-7' → 'b5_c0_3_7'。馬番は小さい方を前に (公式の並び順ルール)。"""
    try:
        a, b = (int(x) for x in str(selection).split("-"))
    except (TypeError, ValueError) as exc:
        raise VoteError(f"ワイド買い目が不正: {selection!r}") from exc
    if a == b:
        raise VoteError(f"同一馬番のワイド: {selection!r}")
    return BET_WIDE_BARA.format(a=min(a, b), b=max(a, b))


def umaren_bet_id(selection: str) -> str:
    """'3-7' → 'b4_c0_3_7'。馬番は小さい方を前に (公式の並び順ルール)。"""
    try:
        a, b = (int(x) for x in str(selection).split("-"))
    except (TypeError, ValueError) as exc:
        raise VoteError(f"馬連買い目が不正: {selection!r}") from exc
    if a == b:
        raise VoteError(f"同一馬番の馬連: {selection!r}")
    return BET_UMAREN_BARA.format(a=min(a, b), b=max(a, b))


def bet_id_for(ticket: dict) -> str:
    """ticket の kind (既定 wide) に応じた bet_id を返す。"""
    sel = str(ticket["selection"])
    return umaren_bet_id(sel) if ticket.get("kind") == "umaren" else wide_bet_id(sel)


def aite_switch_tickets(race: dict, market: dict, cfg: dict | None = None
                        ) -> tuple[list[dict], str]:
    """◎オッズ×相手オッズ(2026-09-05 実測)で券種を切り替える。

    相手 = model 上位馬連ペア(all_umaren argmax)のうち ◎ でない方。107R の
    後ろ向き検証 (analysis/aite_odds_switch.py 系の場当たり検証を本配線用に再実装)
    では相手オッズ帯が支配的で、◎オッズは「相手が現実的な帯での両方乗せ」補正にしか
    効かなかった。3 分岐とも判定できない場合は fail-closed で見送る。

    cfg["aite_force_both"] = True の場合、オッズ帯による3分岐を無視し、
    hard gate 通過・オッズ取得済みの全レースで常に 馬連本命1点+ワイド上位2点を
    セットで買う (2026-09-22 大会最終日、選別を捨てて出走機会を最大化する設定)。
    """
    force_both = bool((cfg or {}).get("aite_force_both"))
    import numpy as np
    import pl_probs as PL
    from compute_bets import pl_pair_probs
    from production_policy import find_hon, hard_skip_reasons

    horses = [dict(h) for h in (race.get("horses") or [])
              if isinstance(h.get("ai_score"), (int, float))]
    if len(horses) < 6:
        return [], "頭数不足(<6)"
    live_tan = {int(k): v for k, v in (market.get("tansho") or {}).items()}
    for h in horses:
        ban = int(h["umaban"])
        if live_tan.get(ban) is not None:
            h["tansho_odds"] = live_tan[ban]

    hon = find_hon(horses)
    gate_reasons = hard_skip_reasons(race.get("race_meta") or {},
                                     race.get("race_confidence") or {}, hon)
    if gate_reasons:
        return [], "hard_gate:" + " / ".join(gate_reasons)

    uma = [int(h["umaban"]) for h in horses]
    w = PL.pl_weights(np.array([float(h["ai_score"]) for h in horses]))
    (i, j), p_uma = max(PL.all_umaren(w).items(), key=lambda kv: kv[1])
    sel = tuple(sorted((uma[i], uma[j])))

    hon_umaban = int(hon["umaban"])
    hon_odds = hon.get("tansho_odds")
    if hon_umaban not in sel or hon_odds is None:
        return [], "◎が model 本命馬連ペア外、または◎単勝オッズ欠損"
    partner_umaban = sel[0] if sel[1] == hon_umaban else sel[1]
    partner_h = next((h for h in horses if int(h["umaban"]) == partner_umaban), None)
    aite_odds = partner_h.get("tansho_odds") if partner_h else None
    if aite_odds is None:
        return [], "相手単勝オッズ欠損"

    live_wide = market.get("wide") or {}
    _, model_wide = pl_pair_probs(horses)
    wide_tickets = []
    for (a, b), p in sorted(model_wide.items(), key=lambda kv: -kv[1])[:2]:
        key = (min(int(a), int(b)), max(int(a), int(b)))
        rng = live_wide.get(f"{key[0]}-{key[1]}")
        t = {"selection": f"{key[0]}-{key[1]}", "kind": "wide", "p_model": p}
        if isinstance(rng, (list, tuple)) and len(rng) >= 2:
            t["odds_t10_low"], t["odds_t10_high"] = float(rng[0]), float(rng[1])
        wide_tickets.append(t)
    uma_ticket_t = {"selection": f"{sel[0]}-{sel[1]}", "kind": "umaren", "p_model": p_uma,
                   "hon_odds": float(hon_odds), "aite_odds": float(aite_odds)}
    live_umaren = market.get("umaren") or {}
    market_umaren_odds = live_umaren.get(f"{sel[0]}-{sel[1]}")
    if market_umaren_odds is not None:
        uma_ticket_t["odds"] = float(market_umaren_odds)
    uma_ticket = [uma_ticket_t]

    if force_both:
        return (wide_tickets + uma_ticket,
                f"aite_switch:両方[force](◎{hon_odds:.1f}倍/相手{aite_odds:.1f}倍)")
    if aite_odds >= AITE_ODDS_LONGSHOT:
        return wide_tickets, f"aite_switch:ワイド(相手{aite_odds:.1f}倍)"
    if hon_odds < HON_ODDS_CHALK:
        return wide_tickets + uma_ticket, f"aite_switch:両方(◎{hon_odds:.1f}倍/相手{aite_odds:.1f}倍)"
    return uma_ticket, f"aite_switch:馬連(◎{hon_odds:.1f}倍/相手{aite_odds:.1f}倍)"


def fallback_wide_ticket(race: dict, market: dict) -> list[dict]:
    """本アーム (residual/aite_switch) が見送りになった時の最終フォールバック。

    hard gate 不通過・シグナル無しでも、model 最上位ワイド1点だけを最小額 (100円)
    で投票する。目的は的中ではなく大会の最低投票要件 (96R) を金額をほぼかけずに
    稼ぐこと。通常アームの判断 (見送りガード含む) には一切割り込まない —
    tickets が空になった後にだけ呼ばれる。
    """
    from compute_bets import pl_pair_probs

    live_tan = {int(k): v for k, v in (market.get("tansho") or {}).items()}
    horses = [dict(h) for h in (race.get("horses") or [])
              if isinstance(h.get("ai_score"), (int, float))
              and int(h.get("umaban", -1)) in live_tan]
    if len(horses) < 3:
        return []
    _, model_wide = pl_pair_probs(horses)
    if not model_wide:
        return []
    (a, b), p = max(model_wide.items(), key=lambda kv: kv[1])
    key = (min(int(a), int(b)), max(int(a), int(b)))
    return [{"selection": f"{key[0]}-{key[1]}", "kind": "wide", "p_model": p}]


def select_tickets(shadow: dict, cfg: dict) -> tuple[list[dict], str]:
    """(採用する ticket 行, 採用アーム名) を返す。買わない場合は ([], 理由)。

    ignore_hard_gate=True は本番の chaos / field_size / ◎p_win ハード見送りを外し、
    残差帯だけで拾う。実弾ではなく仮想ポイントの大会で、最低投票要件 (96R) を
    満たすために母数を広げる用途。
    """
    if not shadow.get("hard_gate_passed"):
        if not cfg.get("ignore_hard_gate"):
            return [], "hard_gate:" + " / ".join(shadow.get("hard_gate_reasons") or [])
        band = cfg.get("_no_gate_candidates") or []
        return (list(band), "residual(no-gate)") if band else ([], "残差0〜+0.05に該当なし(gate外)")
    if shadow.get("triggered"):
        return list(shadow["arm_a"]), "residual"
    filler = cfg.get("filler")
    if filler == "model_top2" and shadow.get("control_m1"):
        return list(shadow["control_m1"]), "filler:model_top2"
    return [], "残差0〜+0.05に該当なし"


def residual_candidates(race: dict, market: dict) -> list[dict]:
    """hard gate を見ずに、残差帯に入る model 上位ペアを出す。

    数値は `data/shadow_policies/wide_residual_shadow_v3.json` を単一ソースとして読む
    (Arm A と同じ演算)。gate が通るレースでは compute_shadow の arm_a と一致する。
    """
    from compute_bets import pl_pair_probs
    from wide_residual_shadow import (ShadowPolicyError, load_shadow_policy,
                                      market_wide_fair)

    cfg = load_shadow_policy()["candidate"]
    horses = [dict(h) for h in (race.get("horses") or [])]
    live_tan = {int(k): v for k, v in (market.get("tansho") or {}).items()}
    for h in horses:
        try:
            ban = int(h["umaban"])
        except (KeyError, TypeError, ValueError):
            continue
        if live_tan.get(ban) is not None:
            h["tansho_odds"] = live_tan[ban]

    live_wide = market.get("wide") or {}
    fair = market_wide_fair(live_wide)
    if not fair:
        raise ShadowPolicyError("ワイド市場確率を計算できない")
    active = sorted(live_tan)
    expected = {(active[i], active[j])
                for i in range(len(active)) for j in range(i + 1, len(active))}
    if cfg.get("require_complete_active_pair_market", True) and set(fair) != expected:
        raise ShadowPolicyError(
            f"ワイド価格が不完全: expected={len(expected)} actual={len(fair)}")

    _, model_wide = pl_pair_probs(horses)
    max_mid = float(cfg["max_odds_mid"])
    priced = []
    for pair, prob in (model_wide or {}).items():
        key = (min(int(pair[0]), int(pair[1])), max(int(pair[0]), int(pair[1])))
        rng = live_wide.get(f"{key[0]}-{key[1]}")
        if not isinstance(rng, (list, tuple)) or len(rng) < 2:
            continue
        lo, hi = float(rng[0]), float(rng[1])
        p_market = fair.get(key)
        if p_market is None or (lo + hi) / 2.0 > max_mid:
            continue
        priced.append({"selection": f"{key[0]}-{key[1]}", "p_model": float(prob),
                       "p_market_fair": p_market, "residual": float(prob) - p_market,
                       "odds_t10_low": lo, "odds_t10_high": hi,
                       "odds_t10_mid": (lo + hi) / 2.0})
    if not priced:
        raise ShadowPolicyError("価格付きワイド候補が0件")

    top = sorted(priced, key=lambda r: (-r["p_model"], r["selection"])
                 )[:int(cfg["take_model_top_n_before_filter"])]
    lower = float(cfg["residual_lower_inclusive"])
    upper = float(cfg["residual_upper_exclusive"])
    band = [r for r in top if lower <= r["residual"] < upper]
    return band[:int(cfg["max_tickets_per_race"])]


def _stake_for(ticket: dict, stake) -> int:
    """stake が dict (kind別単価) なら ticket['kind'] で引く。int ならそのまま。"""
    if isinstance(stake, dict):
        kind = ticket.get("kind", "wide")
        amount = stake.get(kind, stake.get("_default"))
        if amount is None:
            raise VoteError(f"stake_per_ticket_yen_by_kind に kind={kind!r} の設定が無い")
    else:
        amount = stake
    amount = int(amount)
    if amount < 100 or amount % 100 != 0:
        raise VoteError(f"1点金額は 100 円単位・100 円以上: {amount}")
    return amount


def build_bet_list(tickets: list[dict], stake, active: set[int],
                   max_points: int) -> list[dict]:
    """stake は一律の int、または {"umaren": 15000, "wide": 5000} のような
    kind別単価 dict。dict の場合 ticket['kind'] (既定 wide) で単価を引く。
    """
    bets, seen = [], set()
    for t in tickets:
        sel = str(t["selection"])
        a, b = (int(x) for x in sel.split("-"))
        if a not in active or b not in active:
            raise VoteError(f"取消/不在の馬番を含む買い目: {sel} (現存 {sorted(active)})")
        bet_id = bet_id_for(t)
        if bet_id in seen:
            continue
        seen.add(bet_id)
        bets.append({"bet_id": bet_id, "money": _stake_for(t, stake)})
    if len(bets) > max_points:
        raise VoteError(f"買い目 {len(bets)} 点 > 上限 {max_points} 点")
    return bets


def assemble_payload(race: dict, market: dict, tickets: list[dict], why: str, cfg: dict,
                     stake_override: int | None = None
                     ) -> tuple[dict | None, str, list[dict]]:
    """tickets が決まった後の共通処理 (印付け・bet_list 化・payload 組立)。

    stake_override はフォールバック (最小額固定) 専用。通常時は None のまま
    cfg の券種別単価 (stake_per_ticket_yen_by_kind) を使う。
    """
    if not tickets:
        return None, why, []
    active = {int(k) for k in (market.get("tansho") or {})}
    if not active:
        raise VoteError("T-4 単勝オッズが空 (現存馬を確定できない)")
    stake_cfg = (stake_override if stake_override is not None else
                (cfg.get("stake_per_ticket_yen_by_kind") or cfg["stake_per_ticket_yen"]))
    bet_list = build_bet_list(tickets, stake_cfg, active,
                              int(cfg["max_points_per_race"]))
    payload = {
        "race_id": netkeiba_race_id(race["race_id"]),
        "mark": build_marks(race.get("horses") or [], active),
        BET_KEY: bet_list,
    }
    return payload, why, tickets


def build_payload(race: dict, market: dict, shadow: dict, cfg: dict
                  ) -> tuple[dict | None, str, list[dict]]:
    """(bet_data 要素, 採用アーム/見送り理由, ticket 明細) を返す。arm="residual" 用。"""
    if cfg.get("ignore_hard_gate") and not shadow.get("hard_gate_passed"):
        cfg = dict(cfg, _no_gate_candidates=residual_candidates(race, market))
    tickets, why = select_tickets(shadow, cfg)
    return assemble_payload(race, market, tickets, why, cfg)


def build_payload_aite_switch(race: dict, market: dict, cfg: dict
                              ) -> tuple[dict | None, str, list[dict]]:
    """(bet_data 要素, 採用アーム/見送り理由, ticket 明細) を返す。arm="aite_switch" 用。"""
    tickets, why = aite_switch_tickets(race, market, cfg)
    return assemble_payload(race, market, tickets, why, cfg)


# ------------------------------------------------------------------ 台帳
def ledger_path(date_str: str) -> Path:
    return LEDGER_DIR / f"{date_str}.json"


def load_ledger(date_str: str) -> dict:
    try:
        doc = json.loads(ledger_path(date_str).read_text(encoding="utf-8"))
        if isinstance(doc, dict) and isinstance(doc.get("races"), list):
            return doc
    except (OSError, ValueError):
        pass
    return {"date": date_str, "races": []}


def save_ledger(date_str: str, entry: dict) -> None:
    doc = load_ledger(date_str)
    rows = {rid16(r.get("race_id")): r for r in doc["races"]}
    rows[rid16(entry["race_id"])] = entry
    doc["races"] = sorted(rows.values(), key=lambda r: str(r.get("race_id")))
    p = ledger_path(date_str)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


# ------------------------------------------------------------------ 取得
def fetch_vote_odds(rid: str, scheduled_post: datetime) -> dict:
    """T-4 の JV-Link 価格を取得する (本番 T-10 の live_odds を上書きしない)。"""
    VOTE_ODDS_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [*PY32, "jvlink_odds.py", "--race", rid, "--stage", "vote",
           "--out-dir", str(VOTE_ODDS_DIR),
           "--scheduled-post", scheduled_post.isoformat()]
    r = subprocess.run(cmd, cwd=str(BASE), env=dict(os.environ, PYTHONUTF8="1"),
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=120)
    out = (r.stdout or "") + (r.stderr or "")
    line = next((x for x in out.splitlines() if "[jvlink_odds]" in x), out.strip()[-200:])
    print(f"  [1/4] jvlink_odds --stage vote (exit {r.returncode}) {line}")
    if r.returncode != 0:
        raise VoteError(f"T-4 価格取得失敗 (exit {r.returncode})")
    market = json.loads((VOTE_ODDS_DIR / f"{rid}.json").read_text(encoding="utf-8"))
    if not market.get("ok"):
        raise VoteError(f"T-4 価格 ok=false: {market.get('reason', '')}")
    return market


def load_bundle_race(date_str: str, rid: str) -> dict:
    path = BASE / "reports" / "cowork_input" / f"{date_str}_bundle.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    race = next((r for r in doc.get("races", []) if rid16(r.get("race_id")) == rid), None)
    if race is None:
        raise VoteError(f"bundle に {rid} が無い ({path.name})")
    return race


# ------------------------------------------------------------------ 投票
def submit(payload: dict, cfg: dict, notify=None) -> dict:
    """login → POST → (1分待機) → check → logout。戻り値は台帳用の結果 dict。"""
    import netkeiba_api as api
    # --- P0 hard gate (層5/5): 実送信の直前。上流で除外済みでも省略しない。
    from race_eligibility import assert_bettable
    _rid = rid16(payload.get("race_id") or payload.get("raceId") or "")
    assert_bettable(_rid, payload.get("bet_data") or payload.get("bets"),
                    layer="masters_vote.submit")

    result: dict = {"sent_at": datetime.now().isoformat(timespec="seconds"),
                    "verified": False}
    token = api.login()
    try:
        body = api.post_bets(token, [payload])
        data = body.get("data") or {}
        result["bet_response"] = data
        result["remaining_money"] = body.get("remaining_money")
        errors = data.get("list_error") or []
        if int(data.get("error_count") or 0) or errors:
            result["error"] = f"error_count={data.get('error_count')} {errors}"
            print(f"  [2/4] bet: ★エラー {result['error']} (公式仕様上、1件も未処理)")
            return result
        print(f"  [2/4] bet: success_count={data.get('success_count')} "
              f"{data.get('list_bet_race')}")

        delay = int(cfg["check_delay_sec"])
        print(f"  [3/4] check API まで {delay}s 待機 (投票直後は非同期のため)")
        time.sleep(delay)
        # ★確認の失敗で投票の記録を落とさない。未反映レースは status=NG が返る仕様
        #   (公式サンプル 04 の注記) なので、ここで例外にすると成功した投票が
        #   台帳から消える。確認できなかったことだけを残して続行する。
        try:
            rows = api.check_bets(token, [payload["race_id"]])
            result["check_response"] = rows
            result["verified"] = verify(payload, rows)
            print(f"  [3/4] check: {'✅ 登録確認 OK' if result['verified'] else '⚠ 不一致'}")
        except api.NetkeibaApiError as exc:
            result["check_error"] = str(exc)
            print(f"  [3/4] check: ⚠ 確認できず ({exc}) — 投票自体は受理済み")
    finally:
        api.logout(token)
        print("  [4/4] logout")
    return result


def verify(payload: dict, rows: list[dict]) -> bool:
    """check API の応答が、送ったのと同じ bet_id / 金額になっているか。"""
    row = next((r for r in rows
                if str(r.get("race_id")) == str(payload["race_id"])), None)
    if row is None:
        return False
    got = {str(b.get("bet_id")): int(float(b.get("money") or 0))
           for b in (row.get("bet") or [])}
    want = {b["bet_id"]: int(b["money"]) for b in payload[BET_KEY]}
    return got == want


def effective_arm(cfg: dict, date_str: str) -> str:
    """arm_effective_from 前は arm_before を使う日付切替込みの実効 arm。"""
    eff_from = cfg.get("arm_effective_from")
    if eff_from and str(date_str) < str(eff_from):
        return cfg.get("arm_before") or "residual"
    return cfg.get("arm", "residual")


KIND_LABEL = {"wide": "ワイド", "umaren": "馬連"}


def ticket_odds_note(t: dict) -> str:
    """ticket の券種/データに応じたオッズ注記。表示用・投票判断には使わない。

    residual アームはワイドの市場残差、aite_switch のワイドは市場レンジ、
    aite_switch の馬連は「馬連自体の市場オッズが JV-Link 未取得(0B31/33/34のみ)
    のため」◎・相手の単勝オッズで代用する。
    """
    if "residual" in t:
        return (f"T-4 {t['odds_t10_low']:.1f}-{t['odds_t10_high']:.1f}倍 "
                f"model {t['p_model']*100:.1f}% / 市場 {t['p_market_fair']*100:.1f}% "
                f"(残差 {t['residual']:+.3f})")
    if t.get("kind") == "wide" and t.get("odds_t10_low") is not None:
        return f"市場 {t['odds_t10_low']:.1f}-{t['odds_t10_high']:.1f}倍"
    if t.get("kind") == "umaren" and t.get("odds") is not None:
        return f"市場 {t['odds']:.1f}倍"
    if t.get("kind") == "umaren" and t.get("hon_odds") is not None:
        return f"◎単勝{t['hon_odds']:.1f}倍・相手単勝{t['aite_odds']:.1f}倍（馬連オッズ欠損の代替表示）"
    p = t.get("p_model")
    return f"model {p*100:.1f}%" if isinstance(p, (int, float)) else ""


# ------------------------------------------------------------------ 1レース
def vote_race(date_str: str, rid: str, *, post_override: str | None = None,
              dry: bool = False, force: bool = False, notify=None,
              odds_json: str | None = None,
              stake_override: int | None = None) -> int:
    cfg = load_config()
    cfg = dict(cfg, arm=effective_arm(cfg, date_str))
    if stake_override:
        cfg = dict(cfg, stake_per_ticket_yen=int(stake_override),
                  stake_per_ticket_yen_by_kind=None)
    rid = rid16(rid)
    label = rid
    print(f"\n[{datetime.now():%H:%M:%S}] ▶ 大会投票 {rid} (date={date_str}, arm={cfg.get('arm')})")

    # --- P0 hard gate (層5/5, 早期): 障害レースは投票経路に入れない ---
    from race_eligibility import evaluate_race, log_exclusions
    _el = evaluate_race(rid)
    if not _el["bet_eligible"]:
        log_exclusions([_el], layer="masters_vote")
        save_ledger(date_str, {"race_id": rid, "label": label, "voted": False,
                               "reason": ("障害競走のため対象外 (P0 hard gate)"
                                          if _el["is_jump"] else
                                          "eligibility 判定不能のため対象外 "
                                          "(P0 hard gate, fail-closed)"),
                               "arm": None,
                               "at": datetime.now().isoformat(timespec="seconds")})
        if notify:
            notify(f"⚠ 大会投票 {rid}: 障害競走のため見送り (P0 hard gate)")
        return 0

    post = post_datetime(date_str, rid, post_override)
    deadline = post - timedelta(minutes=3, seconds=int(cfg["submit_safety_sec"]))
    now = datetime.now()
    print(f"  発走 {post:%H:%M} / 投票締切 {post - timedelta(minutes=3):%H:%M:%S} "
          f"/ 送信期限 {deadline:%H:%M:%S} (safety {cfg['submit_safety_sec']}s)")
    if not dry and now >= deadline:
        print(f"  ✗ 送信期限を過ぎている ({now:%H:%M:%S}) → 投票しない")
        save_ledger(date_str, {"race_id": rid, "voted": False,
                               "reason": "送信期限超過", "arm": None,
                               "at": now.isoformat(timespec="seconds")})
        if notify:
            notify(f"⚠ 大会投票 {rid}: 送信期限超過で見送り")
        return 2

    prev = next((r for r in load_ledger(date_str)["races"]
                 if rid16(r.get("race_id")) == rid), None)
    if prev and prev.get("verified") and not force and not dry:
        print("  = 送信済み・確認済みのためスキップ (--force で再送=上書き)")
        return 0

    if odds_json:                      # テスト用: JV-Link を叩かず保存済み価格で通す
        market = json.loads(Path(odds_json).read_text(encoding="utf-8"))
        print(f"  [1/4] --odds-json {odds_json} を使用 (JV-Link は叩かない)")
        if not market.get("ok"):
            raise VoteError(f"価格 ok=false: {market.get('reason', '')}")
    else:
        market = fetch_vote_odds(rid, post)
    race = load_bundle_race(date_str, rid)
    from compute_bets import race_label
    label = race_label(rid, race.get("race_meta") or {}) or rid

    if cfg.get("arm") == "aite_switch":
        payload, why, tickets = build_payload_aite_switch(race, market, cfg)
    else:
        from compute_bets import pl_pair_probs
        from wide_residual_shadow import compute_shadow
        shadow = compute_shadow(race, market, pair_probability_fn=pl_pair_probs)
        payload, why, tickets = build_payload(race, market, shadow, cfg)

    if payload is None and cfg.get("fallback_wide_min100"):
        fb_tickets = fallback_wide_ticket(race, market)
        if fb_tickets:
            payload, why, tickets = assemble_payload(
                race, market, fb_tickets, f"fallback_wide100(見送り理由: {why})",
                cfg, stake_override=100)

    if payload is None:
        print(f"  — 見送り: {why}")
        save_ledger(date_str, {"race_id": rid, "label": label, "voted": False,
                               "reason": why, "arm": None,
                               "at": datetime.now().isoformat(timespec="seconds")})
        if notify and cfg.get("notify_skips"):
            notify(f"🎓 大会 {label} **見送り** — {why}")
        return 0

    total = sum(b["money"] for b in payload[BET_KEY])
    print(f"  買い目 [{why}] {len(payload[BET_KEY])}点 ¥{total:,} "
          f"/ netkeiba race_id={payload['race_id']}")
    for b, t in zip(payload[BET_KEY], tickets):
        label_kind = KIND_LABEL.get(t.get("kind", "wide"), t.get("kind", "wide"))
        note = ticket_odds_note(t)
        print(f"     {label_kind} {t['selection']:6s} {b['bet_id']:14s} ¥{b['money']:,}"
              + (f"  {note}" if note else ""))
    print(f"     印 {payload['mark']}")

    if dry:
        print("\n  [dry] 送信しない。実際に POST する JSON:")
        print(json.dumps({"bet_data": [payload]}, ensure_ascii=False, indent=2))
        return 0
    if not cfg.get("enabled"):
        # ★2026-08-29 の事故: 16R 分の買い目が出ていたのに enabled=false で
        #   黙って捨てられ、丸一日投票ゼロになった。設定ミスは必ず可視化する。
        print("  ✗ data/masters_vote.json の enabled=false → 送信しない")
        save_ledger(date_str, {
            "race_id": rid, "label": label, "voted": False,
            "reason": "enabled=false のため未送信 (設定ミス)", "arm": why,
            "would_have_bet": payload, "total_yen": total,
            "at": datetime.now().isoformat(timespec="seconds")})
        if notify:
            notify(f"❗ 大会 {label}: 買い目 {len(payload[BET_KEY])}点 ¥{total:,} を"
                   f"作ったが **enabled=false のため未送信**。"
                   f"data/masters_vote.json を確認すること")
        return 0

    now = datetime.now()
    if now >= deadline:
        print(f"  ✗ 準備中に送信期限を過ぎた ({now:%H:%M:%S}) → 投票しない")
        save_ledger(date_str, {"race_id": rid, "label": label, "voted": False,
                               "reason": "準備中に送信期限超過", "arm": why,
                               "at": now.isoformat(timespec="seconds")})
        return 2

    result = submit(payload, cfg, notify=notify)
    entry = {"race_id": rid, "netkeiba_race_id": payload["race_id"], "label": label,
             "voted": "error" not in result, "arm": why, "payload": payload,
             "total_yen": total, "market_observed_at": market.get("fetched"),
             **result}
    save_ledger(date_str, entry)
    if notify:
        mark = "✅" if entry["voted"] and result.get("verified") else (
            "⚠" if entry["voted"] else "❌")
        SEP = "──────────"
        lines = [f"{mark} 大会投票　{label}　発走{post:%H:%M}", SEP]
        for b, t in zip(payload[BET_KEY], tickets):
            label_kind = KIND_LABEL.get(t.get("kind", "wide"), t.get("kind", "wide"))
            note = ticket_odds_note(t)
            lines.append(f"{label_kind} `{t['selection']}` ¥{b['money']:,}"
                        + (f"（{note}）" if note else ""))
        lines.append(SEP)
        lines.append(f"合計 ¥{total:,}")
        if result.get("remaining_money"):
            lines.append(f"残高 {int(result['remaining_money']):,}pt")
        if result.get("error"):
            lines.append(f"エラー: {result['error']}")
        elif not result.get("verified"):
            lines.append("※ check API で登録確認できず — 手動確認を推奨")
        notify("\n".join(lines))
    return 0 if entry["voted"] else 1


# ------------------------------------------------------------------ 補助
def verify_day(date_str: str) -> int:
    """送信済みレースを check API でまとめて再確認する。"""
    import netkeiba_api as api
    doc = load_ledger(date_str)
    sent = [r for r in doc["races"] if r.get("voted") and r.get("payload")]
    if not sent:
        print(f"{date_str}: 送信済みレース無し")
        return 0
    token = api.login()
    try:
        try:
            rows = api.check_bets(token, [r["payload"]["race_id"] for r in sent])
        except api.NetkeibaApiError as exc:
            # 1 レースでも未投票が混じると status=NG になる (公式サンプル 04 の注記)。
            # まとめて聞けないので 1 レースずつ聞き直す。
            print(f"  [warn] 一括確認が NG ({exc}) → 1 レースずつ確認する")
            rows = []
            for r in sent:
                try:
                    rows += api.check_bets(token, [r["payload"]["race_id"]])
                except api.NetkeibaApiError:
                    pass
    finally:
        api.logout(token)
    ng = 0
    for r in sent:
        ok = verify(r["payload"], rows)
        r["verified"] = ok
        r["check_response"] = rows
        save_ledger(date_str, r)
        print(f"  {'✅' if ok else '❌'} {r.get('label', r['race_id'])} "
              f"{r['payload']['race_id']}")
        ng += 0 if ok else 1
    print(f"{date_str}: {len(sent)-ng}/{len(sent)} 確認 OK")
    return 1 if ng else 0


def format_test(date_str: str, rid: str, stake: int = 100) -> int:
    """★本番前の一発検証: race_id / bet_data 構造 / bet_id 書式を実 API で確かめる。

    ◎-〇 のワイド 1 点を最小額で実際に投票し、check API で読み戻して一致を見る。
    未検証だったのは (a) TARGET 16 桁 → netkeiba 12 桁の race_id 変換、
    (b) 公式「投票」サンプル未入手のまま組んだ bet_data 配列構造 の 2 点。
    ここが通れば当日の自動投票はこの 2 つで落ちない。

    ※このレースは「投票済み」になる (的中率の分母に入る)。T-4 の本投票が
      発火すれば上書きされる。
    """
    import netkeiba_api as api

    nk_id = netkeiba_race_id(rid)
    try:
        race = load_bundle_race(date_str, rid16(rid))
        horses = race.get("horses") or []
        active = {int(h["umaban"]) for h in horses if h.get("umaban") is not None}
        marks = build_marks(horses, active)
        hon = next(int(b) for b, c in marks.items() if c == 1)
        tai = next(int(b) for b, c in marks.items() if c == 2)
    except (VoteError, OSError, ValueError, StopIteration):
        # bundle が無くても書式検証はできる。1・2・3 番は 3 頭立て以上なら必ず存在する。
        print("[info] bundle が無いので最小構成で検証する (印 ◎1 〇2 ▲3 / ワイド 1-2)")
        marks, hon, tai = {"1": 1, "2": 2, "3": 3}, 1, 2
    bet_id = wide_bet_id(f"{hon}-{tai}")

    # 公式「投票」サンプル未入手のため、bet_data 要素の形は API に判定させる。
    # エラー時は 1 件も登録されない仕様なので、候補を順に試すのは安全。
    shapes = [
        ("bet", "int", {"race_id": nk_id, "mark": marks,
                        "bet": [{"bet_id": bet_id, "money": int(stake)}]}),
        ("bet", "str", {"race_id": nk_id, "mark": marks,
                        "bet": [{"bet_id": bet_id, "money": str(int(stake))}]}),
        ("bet_list", "str", {"race_id": nk_id, "mark": marks,
                             "bet_list": [{"bet_id": bet_id, "money": str(int(stake))}]}),
        ("bet_list", "int", {"race_id": nk_id, "mark": marks,
                             "bet_list": [{"bet_id": bet_id, "money": int(stake)}]}),
    ]

    token = api.login()
    try:
        payload = data = None
        for key, money_type, candidate in shapes:
            print(f"\n--- 試行: 買い目キー={key!r} / money={money_type} ---")
            print(json.dumps({"bet_data": [candidate]}, ensure_ascii=False))
            try:
                body = api.post_bets(token, [candidate])
            except api.NetkeibaApiError as exc:
                print(f"  ✗ {exc}")
                continue
            data = body.get("data") or {}
            errs = data.get("list_error") or []
            if int(data.get("error_count") or 0) or errs:
                print(f"  ✗ error_count={data.get('error_count')} {errs}")
                continue
            payload = candidate
            print(f"  ✅ 受理: {data.get('list_bet_race')}")
            print(f"\n★ 正しい形式は 買い目キー={key!r} / money={money_type} "
                  f"— masters_vote.build_payload をこれに合わせること。")
            break
        if payload is None:
            print("\n❌ どの形式も通らなかった。上のエラー文言を見て候補を足す。")
            return 1
        print("[check] 60 秒待って読み戻す…")
        time.sleep(60)
        rows = api.check_bets(token, [payload["race_id"]])
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        if verify(payload, rows):
            print("\n✅ race_id 変換・bet_data 構造・bet_id 書式すべて正しい。"
                  "data/masters_vote.json の enabled を true にして本番投入可。")
            return 0
        print("\n⚠ 投票は通ったが check API の内容が一致しない。中身を目視で確認。")
        return 1
    finally:
        api.logout(token)


def check_race(rid: str) -> int:
    """指定レースの登録内容を確認 API で読み戻して表示する (読み取り専用)。"""
    import netkeiba_api as api

    nk_id = netkeiba_race_id(rid)
    token = api.login()
    try:
        try:
            rows = api.check_bets(token, [nk_id])
        except api.NetkeibaApiError as exc:
            # 未投票の race_id を含めると status=NG (公式サンプル 04 の注記)
            print(f"race_id={nk_id}: 登録なし、または確認できず\n  {exc}")
            return 1
    finally:
        api.logout(token)

    row = next((r for r in rows if str(r.get("race_id")) == nk_id), None)
    if row is None:
        print(f"race_id={nk_id}: 応答に該当レースなし")
        return 1
    mark_name = {1: "◎", 2: "〇", 3: "▲", 4: "△"}
    marks = sorted(((int(v), int(b)) for b, v in (row.get("mark") or {}).items()))
    print(f"race_id = {nk_id}")
    print("  印   " + " ".join(f"{mark_name.get(code, code)}{ban}" for code, ban in marks))
    bets = row.get("bet") or []
    total = sum(int(float(b.get("money") or 0)) for b in bets)
    print(f"  買い目 {len(bets)}点 合計 ¥{total:,}")
    for b in bets:
        print(f"    {b.get('bet_id')}  ¥{int(float(b.get('money') or 0)):,}")
    return 0


def login_test() -> int:
    """資格情報の疎通確認。トークンそのものは表示しない。"""
    import netkeiba_api as api

    cfg = api.load_config()
    if not cfg["login_id"] or not cfg["password"]:
        print("✗ 資格情報が未入力。data/netkeiba_api.json の login_id / password を"
              "埋めるか、環境変数 NETKEIBA_LOGIN_ID / NETKEIBA_PASSWORD を設定。")
        return 1
    try:
        token = api.login()
    except api.NetkeibaApiError as exc:
        print(f"✗ ログイン失敗: {exc}")
        return 1
    print(f"✅ ログイン成功 (access_token {len(token)} 文字を取得。有効 300 秒)")
    print(f"   ログアウト: {'OK' if api.logout(token) else '失敗 (5分で自動失効)'}")
    vote_cfg = load_config()
    by_kind = vote_cfg.get("stake_per_ticket_yen_by_kind")
    stake_disp = (", ".join(f"{k}=¥{v:,}" for k, v in by_kind.items())
                  if by_kind else f"¥{vote_cfg['stake_per_ticket_yen']:,}")
    print(f"   masters_vote.json: enabled={vote_cfg['enabled']} "
          f"ignore_hard_gate={vote_cfg.get('ignore_hard_gate')} "
          f"1点={stake_disp}")
    if not vote_cfg["enabled"]:
        print("   ⚠ enabled=false のため、実際の投票はまだ送信されません。")
    return 0


def summary() -> int:
    """最低投票要件 (96R / 50万pt) の進捗とペース判定。

    ★実測 (過去 22 開催日) では中央値 14R/日 = 9 日で 126R と余裕があるが、
      実測最小の 11R/日 が続くと 99R で margin は 3R しかない。足りない兆候が
      出たら filler="model_top2" を入れる (母数が 8.9R/日 増える) 判断のために、
      残り日数から必要ペースを出す。
    """
    cfg = load_config()
    races = total = 0
    done_days: set[str] = set()
    for p in sorted(LEDGER_DIR.glob("*.json")):
        for r in json.loads(p.read_text(encoding="utf-8")).get("races", []):
            if r.get("voted"):
                races += 1
                total += int(r.get("total_yen") or 0)
                done_days.add(p.stem)
    need_r, need_y = int(cfg["required_races"]), int(cfg["required_total_yen"])
    print(f"投票レース数 {races} / {need_r}  ({'OK' if races >= need_r else '不足'})")
    print(f"累計投票額   ¥{total:,} / ¥{need_y:,}  "
          f"({'OK' if total >= need_y else '不足'})")

    all_days = [str(d) for d in (cfg.get("race_days") or [])]
    if not all_days:
        return 0
    today = datetime.now().strftime("%Y%m%d")
    left = [d for d in all_days if d > today and d not in done_days]
    print(f"\n開催日 {len(done_days)}/{len(all_days)} 消化、残り {len(left)} 日")
    if not left:
        ok = races >= need_r and total >= need_y
        print("残り日なし → " + ("要件クリア" if ok else "★要件未達"))
        return 0 if ok else 1

    need_r_pace = max(0, need_r - races) / len(left)
    need_y_pace = max(0, need_y - total) / len(left)
    print(f"必要ペース: {need_r_pace:.1f} R/日 ・ ¥{need_y_pace:,.0f}/日")
    # 実測 (本番設定): 中央値 14R/日、最悪の日で 11R/日
    if need_r_pace > 11:
        print("  ★ 実測の最悪日 (11R/日) でも届かないペース。"
              'filler="model_top2" を有効化して母数を増やすこと (+8.9R/日)。')
    elif need_r_pace > 8:
        print("  ⚠ 余裕が薄い。次の開催日の結果を見て filler の投入を判断。")
    else:
        print("  実測ペース (中央値 14R/日) なら余裕あり。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="学生大会 自動投票 (締切=発走3分前)")
    ap.add_argument("--race", help="16 桁 race_id")
    ap.add_argument("--date", default=None, help="YYYYMMDD (既定=今日)")
    ap.add_argument("--post", default=None, help="発走時刻 HH:MM (省略時 weekly CSV)")
    ap.add_argument("--dry", action="store_true", help="送信せずペイロードを表示")
    ap.add_argument("--force", action="store_true", help="確認済みでも再送 (上書き)")
    ap.add_argument("--verify", action="store_true", help="その日の送信済みを再確認")
    ap.add_argument("--summary", action="store_true", help="96R / 50万pt 進捗")
    ap.add_argument("--odds-json", default=None,
                    help="テスト用: JV-Link を叩かず保存済み価格 JSON で通す")
    ap.add_argument("--login-test", action="store_true",
                    help="資格情報の疎通確認だけして終了 (投票しない)")
    ap.add_argument("--check-race", default=None,
                    help="指定レースの登録内容を確認 API で読み戻す (12/16 桁, 読み取り専用)")
    ap.add_argument("--format-test", action="store_true",
                    help="★本番前検証: --race に ◎-〇 ワイド 1 点を最小額で実投票し"
                         " check API で読み戻す。--yes が要る")
    ap.add_argument("--stake", type=int, default=None, help="1 点金額を上書き")
    ap.add_argument("--yes", action="store_true", help="--format-test の実送信を許可")
    args = ap.parse_args()

    if args.check_race:
        return check_race(args.check_race)
    if args.login_test:
        return login_test()
    if args.format_test:
        if not args.race:
            ap.error("--format-test には --race が要る")
        if not args.yes:
            print("--format-test は実際に投票します (最小額)。実行するなら --yes を付けてください。")
            return 1
        return format_test(args.date or datetime.now().strftime("%Y%m%d"),
                           args.race, args.stake or 100)
    if args.summary:
        return summary()
    date_str = args.date or datetime.now().strftime("%Y%m%d")
    if args.verify:
        return verify_day(date_str)
    if not args.race:
        ap.error("--race か --verify / --summary が必要")

    try:
        from t10_runner import notify
    except Exception:
        notify = None
    try:
        return vote_race(date_str, args.race, post_override=args.post,
                         dry=args.dry, force=args.force, notify=notify,
                         odds_json=args.odds_json, stake_override=args.stake)
    except Exception as exc:
        print(f"  ✗ 投票中止 (fail-closed): {exc}")
        save_ledger(date_str, {"race_id": rid16(args.race), "voted": False,
                               "reason": f"例外: {exc}", "arm": None,
                               "at": datetime.now().isoformat(timespec="seconds")})
        if notify:
            notify(f"❌ 大会投票 {rid16(args.race)} 中止: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
