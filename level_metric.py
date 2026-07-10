# -*- coding: utf-8 -*-
"""
level_metric.py — 「馬レベル」を公開レース事実だけから算出する純関数群
================================================================================
ZI/補正タイム等の外部指数(TARGET/JRA-VAN の独自指数=Web再掲は控えるべき)には依存せず、
history.runs の **公開事実**(着順 pos / 人気 ninki, JRAが公式発表する値)だけで
「近走の成績レベル」をスコア化する。build_site.py と build_level_norms.py の両方が使う。

設計:
  run_rating(pos, ninki) : 1走の出来を 0〜118 でスコア化
    ・着順の質 (1着=100 … 逓減)
    ・人気を上回ったか (ninki-pos>0 で加点=地力の公開シグナル)
  horse_level_raw(history): 近5走を直近重み付き平均 0.7 + 直近3走ベスト 0.3 で合成
    ・出走歴ゼロ(新馬/初出走)は None(=レベル判定不能, graceful)

生スコアは build_level_norms.py が母集団分位で 0〜100 に正規化し S〜D にタイア分けする。
"""
from __future__ import annotations

# 着順 → 質スコア (field-relative だが全て JRA 公表の公開事実)
_POS_SCORE = {1: 100.0, 2: 86.0, 3: 75.0, 4: 66.0, 5: 59.0}
# 直近ほど重い減衰重み (n_ago 昇順)
_RECENCY_W = [1.0, 0.72, 0.52, 0.38, 0.28]


def class_group(klass: str):
    """現レースの klass を メンバーレベル norm 用グループへ正規化。障害/新馬は None。"""
    k = str(klass or "")
    if "新馬" in k or "障" in k or k.startswith("J"):
        return None
    if "未勝利" in k:
        return "未勝利"
    if "1勝" in k or "500" in k:
        return "1勝"
    if "2勝" in k or "1000" in k:
        return "2勝"
    if "3勝" in k or "1600" in k:
        return "3勝"
    if k.startswith("G") or k.startswith("Ｇ") or "重賞" in k:
        return "重賞"
    if "オープン" in k or "OP" in k or "(L)" in k or "Ｌ" in k:
        return "オープン"
    return "その他"


def run_rating(pos, ninki) -> float | None:
    """1走の出来を 0〜118 で。pos 不明は None。"""
    try:
        pos = int(pos)
    except (TypeError, ValueError):
        return None
    if pos <= 0:
        return None
    base = _POS_SCORE.get(pos, max(12.0, 59.0 - (pos - 5) * 6.0))
    beat = 0.0
    try:
        nk = int(ninki)
        if nk > 0:
            beat = max(-7, min(7, nk - pos)) * 2.2   # 人気を上回れば地力加点 ±~15
    except (TypeError, ValueError):
        pass
    return max(0.0, min(118.0, base + beat))


def horse_level_raw(history) -> float | None:
    """近走の成績レベル生スコア。出走歴なしは None。"""
    runs = (history or {}).get("runs") or []
    rated = []
    for u in runs:
        r = run_rating(u.get("pos"), u.get("ninki"))
        if r is not None:
            rated.append((u.get("n_ago") or 9, r))
    if not rated:
        return None
    rated.sort(key=lambda x: x[0])               # 直近を先頭に
    num = den = 0.0
    for i, (_, r) in enumerate(rated[:5]):
        w = _RECENCY_W[i] if i < len(_RECENCY_W) else 0.2
        num += w * r
        den += w
    wmean = num / den if den else 0.0
    best3 = max(r for _, r in rated[:3])         # 直近3走のベスト
    return 0.7 * wmean + 0.3 * best3
