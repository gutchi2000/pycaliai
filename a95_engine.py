# -*- coding: utf-8 -*-
"""
a95_engine.py — a95_equal 方策の買い目生成 (単一ソース)
========================================================
方策の正本は data/shadow_policies/a95_equal_v1.json (sha256 自己検査つき)。
  - 順序 : p_win 降順 (生の v6 較正確率。補正印・オッズ blend は使わない)
  - 帯   : AI1位の単勝オッズ (呼び出し側が渡した時点の値。T-10/T-20 ならライブ値)
  - 買い目: 帯ごとの採用券種を AI 上位馬で 1 点ずつ、予算を均等 (100 円格子)

呼び出し元: compute_bets.py (T-10 本線) / t20_site_bets.py (T-20 サイト速報) /
            shadow_a95.py (bundle 時点の紙上台帳)。
ここは買い目を組むだけで、見送り判定 (hard gate / 障害) は呼び出し側の責務。
"""
from __future__ import annotations
import math

from shadow_a95 import band_of, equal_stakes, load_policy, waku_of

KIND_JP = dict(tan="単勝", fuku="複勝", umaren="馬連", umatan="馬単", wide="ワイド",
               wakuren="枠連", sanpuku="三連複", sanrentan="三連単")
# 公開・通知用の帯コード (オッズ生値を文面に出さないため。A=最も低オッズ帯)
BAND_CODE = {"<2倍": "A", "2-3倍": "B", "3-5倍": "C", "5-8倍": "D", "8-15倍": "E", "15倍〜": "F"}


class A95Error(ValueError):
    """買い目を組めない (頭数不足・AI1位のオッズ欠損など)。呼び出し側は見送りにする。"""


def _f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def selection_for(kind: str, a1: int, a2: int, a3: int, w1: int, w2: int) -> str:
    if kind in ("tan", "fuku"):
        return str(a1)
    if kind in ("umaren", "wide"):
        return f"{min(a1, a2)}-{max(a1, a2)}"
    if kind == "umatan":
        return f"{a1}→{a2}"
    if kind == "wakuren":
        return f"{min(w1, w2)}-{max(w1, w2)}"
    if kind == "sanpuku":
        return "-".join(str(x) for x in sorted((a1, a2, a3)))
    if kind == "sanrentan":
        return f"{a1}→{a2}→{a3}"
    raise A95Error(f"未知の券種 {kind}")


def build(horses: list[dict], budget: int | None = None, policy: dict | None = None) -> dict:
    """horses: [{umaban, p_win, tansho_odds}, ...] → a95 の買い目。

    戻り値: {band, band_code, a1, a2, a3, o1, waku1, waku2, n,
             tickets: [{kind, kind_jp, selection, stake}], policy_id, policy_sha256}
    """
    pol = policy or load_policy()
    hs = [h for h in horses if _f(h.get("p_win")) is not None and h.get("umaban") is not None]
    if len(hs) < 5:
        raise A95Error(f"頭数不足 (<5, 現在{len(hs)})")
    order = sorted(hs, key=lambda h: -float(h["p_win"]))
    a1, a2, a3 = (int(order[i]["umaban"]) for i in range(3))
    o1 = _f(order[0].get("tansho_odds"))
    if o1 is None or o1 <= 0:
        raise A95Error("AI1位の単勝オッズ欠損")
    band = band_of(o1, pol)
    n = len(hs)
    w1, w2 = waku_of(a1, n), waku_of(a2, n)
    cap = int(budget) if budget else int(pol["cap_per_race_yen"])
    unit = int(pol["unit_yen"])
    cap = cap // unit * unit
    kinds = list(pol["plan"][band])
    while kinds and cap // len(kinds) < unit:      # 低予算: 後ろの券種から落とす
        kinds.pop()
    if not kinds:
        raise A95Error(f"予算¥{cap:,}では1点も買えない")
    stakes = equal_stakes(kinds, cap, unit)
    tickets = [dict(kind=k, kind_jp=KIND_JP[k], selection=selection_for(k, a1, a2, a3, w1, w2),
                    stake=int(stakes[k])) for k in kinds]
    return dict(band=band, band_code=BAND_CODE.get(band, "?"), a1=a1, a2=a2, a3=a3, o1=o1,
                waku1=w1, waku2=w2, n=n, tickets=tickets,
                policy_id=pol["policy_id"], policy_sha256=pol["policy_sha256"])
