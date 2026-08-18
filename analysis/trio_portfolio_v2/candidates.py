# -*- coding: utf-8 -*-
"""candidates.py — 設計書 §2「候補馬と主腕（完全固定）」

blend 順位は **候補プールの中心を選ぶためだけ** に使う（§1）。確率にも購入額にも
一切入れない。

blend 順位用 score（設計書 §1、本番 compute_bets.hosei_marks と同式）:
    u_i = log(p_win_i) + λ * log(π_i)          λ = 1.5 (data/t10_blend.json)
    π_i = de-vig 市場単勝確率 = (1/odds_i) / Σ(1/odds_j)

腕:
    U0 = blend上位4            (最大  4組)
    U1 = blend上位4 + R(≤1)    (最大 10組)
    U2 = blend上位5            (最大 10組)
    U3 = blend上位5 + R(≤1)    (最大 20組)   ← 主腕。U0-U2 は診断専用

R（AI独自性枠）: 元AI順位 1〜3 かつ blend順位 5位以下、複数なら元AI順位が最上位、
同順位は馬番昇順。Core に既に居る馬は U = Core ∪ R で吸収されるため除く
（U3 では実質 blend 6位以下が R になる = 設計書の「最大6頭・20組」と整合）。

取消馬は AI・blend・市場の全順位から除外してから再順位化する（§2.2）。
"""
from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
BLEND_JSON = BASE / "data" / "t10_blend.json"

ARMS = {"U0": (4, False), "U1": (4, True), "U2": (5, False), "U3": (5, True)}
MAIN_ARM = "U3"


def blend_lambda(default: float = 1.5) -> float:
    try:
        v = float(json.loads(BLEND_JSON.read_text(encoding="utf-8"))["lambda"])
        return v if v > 0 else default
    except Exception:
        return default


def _num(x):
    try:
        v = float(x)
        return v if v == v else None
    except Exception:
        return None


def rank_horses(horses, tansho_odds: dict, scratched=(), lam: float | None = None) -> dict:
    """出走馬の AI順位 / blend順位 / 市場順位 をまとめて作る。

    horses      : bundle の [{umaban, ai_rank, ai_score, p_win}, ...]
    tansho_odds : {umaban(int): T-10 単勝オッズ}  ← 決定時点で存在する情報のみ
    scratched   : 取消・除外の馬番集合
    返り値 {ai_rank, blend_rank, mkt_rank, u, pi, used, dropped}
      *_rank は {umaban: 1..n}。u は blend score。
    """
    lam = blend_lambda() if lam is None else lam
    sc = {int(x) for x in scratched}
    rows = []
    for h in horses:
        b = h.get("umaban")
        if b is None or int(b) in sc:
            continue
        b = int(b)
        s, p, o = _num(h.get("ai_score")), _num(h.get("p_win")), _num(tansho_odds.get(b))
        if s is None or p is None or o is None or p <= 0 or o <= 0:
            continue
        rows.append({"umaban": b, "ai_score": s, "p_win": p, "odds": o})
    dropped = [int(h["umaban"]) for h in horses
               if h.get("umaban") is not None and int(h["umaban"]) not in sc
               and int(h["umaban"]) not in {r["umaban"] for r in rows}]
    if not rows:
        return {"ai_rank": {}, "blend_rank": {}, "mkt_rank": {}, "u": {}, "pi": {},
                "used": [], "dropped": dropped, "lambda": lam}
    s_inv = sum(1.0 / r["odds"] for r in rows)          # de-vig 正規化
    for r in rows:
        r["pi"] = (1.0 / r["odds"]) / s_inv
        r["u"] = math.log(r["p_win"]) + lam * math.log(r["pi"])

    def ranked(key):
        srt = sorted(rows, key=lambda r: (-r[key], r["umaban"]))
        return {r["umaban"]: i + 1 for i, r in enumerate(srt)}

    return {"ai_rank": ranked("ai_score"), "blend_rank": ranked("u"),
            "mkt_rank": ranked("pi"), "u": {r["umaban"]: r["u"] for r in rows},
            "pi": {r["umaban"]: r["pi"] for r in rows},
            "used": sorted(r["umaban"] for r in rows), "dropped": sorted(dropped),
            "lambda": lam}


def pick_rescue(ranks: dict, core: list) -> int | None:
    """§2.2 の AI独自性枠 R。該当なしなら None。"""
    ai, bl = ranks["ai_rank"], ranks["blend_rank"]
    cand = [b for b in ai
            if ai[b] <= 3 and bl.get(b, 99) >= 5 and b not in core]
    if not cand:
        return None
    cand.sort(key=lambda b: (ai[b], b))       # 元AI順位が最上位 → 同順位は馬番昇順
    return cand[0]


def build_arm(ranks: dict, arm: str) -> dict:
    """腕名 → {core, rescue, horses, combos}。combos は (a,b,c) 昇順タプル列。"""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm}")
    top_n, use_r = ARMS[arm]
    bl = ranks["blend_rank"]
    core = [b for b, r in sorted(bl.items(), key=lambda kv: (kv[1], kv[0])) if r <= top_n]
    r = pick_rescue(ranks, core) if use_r else None
    horses = sorted(core + ([r] if r is not None else []))
    combos = [tuple(sorted(c)) for c in combinations(horses, 3)]
    if r is not None:
        # §2.1「R を含む組は必ず Core の馬2頭と組む」— R は1頭なので構造的に成立。
        assert all(sum(1 for x in c if x in core) == 2 for c in combos if r in c)
    return {"arm": arm, "core": core, "rescue": r, "horses": horses,
            "combos": combos, "n_combos": len(combos)}


def build_all_arms(horses, tansho_odds: dict, scratched=()) -> dict:
    ranks = rank_horses(horses, tansho_odds, scratched)
    return {"ranks": ranks, "arms": {a: build_arm(ranks, a) for a in ARMS},
            "main_arm": MAIN_ARM}
