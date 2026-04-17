"""
kelly_allocator.py
==================
Fractional Kelly 基準で馬券買い目に予算を配分するモジュール。

## 原理
Kelly 基準（最適賭け比率）:
    f* = (p × b - (1-p)) / b        (b = 配当-1 = オッズ-1)
Fractional Kelly (1/4):
    f = f* × kelly_fraction          # デフォルト 0.25

## 使い方
    from kelly_allocator import kelly_allocate, kelly_allocate_trifecta

    # 汎用: {bet_id: (p, payout_estimate)} から金額配分を返す
    allocs = kelly_allocate(
        bets   = {1: (0.35, 5.5), 2: (0.20, 8.0)},
        budget = 10000,
        fraction = 0.25,
    )
    # => {1: 4000, 2: 2000, ...} 100円単位整数

    # 三連単: [(combo, p, payout_estimate), ...] から配分
    allocs = kelly_allocate_trifecta(
        combos  = [((1,2,3), 0.05, 1200), ((1,3,2), 0.02, 2500)],
        budget  = 9600,
        fraction = 0.25,
        min_bet  = 100,
        top_n    = 9,
    )
"""
from __future__ import annotations

import math
from typing import Any


# ============================================================
# コア計算
# ============================================================

def kelly_fraction(p: float, payout: float) -> float:
    """Kelly 基準の推奨賭け比率 (f*) を返す。

    payout 規約: payout_table.parquet と同じ **¥/¥100** 単位。
        例: 単勝 4 倍 → payout=400, 三連単 2990 円 → payout=2990

    EV 計算式: EV = p × payout / 100
    Kelly 式:  f* = (p × b - (1-p)) / b, b = payout/100 - 1

    Args:
        p:      推定的中確率  (0–1)
        payout: 推定配当 ¥/¥100 (例: 400 = 4倍, 2990 = 29.9倍)

    Returns:
        f*: Kelly 比率 (0.0 以上 1.0 以下)
    """
    if payout <= 100.0 or p <= 0.0 or p >= 1.0:
        return 0.0
    b = payout / 100.0 - 1.0   # 純利益倍率
    f = (p * b - (1.0 - p)) / b
    return max(0.0, min(1.0, f))


def kelly_allocate(
    bets:     dict[Any, tuple[float, float]],
    budget:   int,
    fraction: float = 0.25,
    min_bet:  int   = 100,
    unit:     int   = 100,
) -> dict[Any, int]:
    """各買い目に Kelly × fraction で予算を配分する。

    Args:
        bets:     {bet_id: (p, payout)}
                  payout は **¥/¥100 単位** (ex. 400=4倍, 2990=29.9倍)
        budget:   総予算 (円)
        fraction: Kelly 係数 (0<fraction≤1, デフォルト 0.25)
        min_bet:  最小賭け金 (円, デフォルト 100)
        unit:     丸め単位 (円, デフォルト 100)

    Returns:
        {bet_id: 賭け金} 合計は budget を超えない範囲で最大化。
        Kelly f*=0 の買い目は除外される。
    """
    if not bets:
        return {}

    # f* 計算
    raw: dict[Any, float] = {
        bid: kelly_fraction(p, payout) * fraction
        for bid, (p, payout) in bets.items()
    }

    total_f = sum(raw.values())
    if total_f <= 0.0:
        # 全て f=0 → 均等配分
        n = len(bets)
        each = _floor_unit(budget // n, unit)
        return {bid: each for bid in bets} if each >= min_bet else {}

    # Kelly 比率を正規化して budget に適用
    result: dict[Any, int] = {}
    for bid, f in raw.items():
        amt = _floor_unit(int(budget * f / total_f), unit)
        if amt >= min_bet:
            result[bid] = amt

    return result


def kelly_allocate_trifecta(
    combos:   list[tuple[tuple[int, int, int], float, float]],
    budget:   int   = 9600,
    fraction: float = 0.25,
    min_bet:  int   = 100,
    unit:     int   = 100,
    top_n:    int   = 9,
    ev_gate:  float = 0.95,
) -> list[dict]:
    """三連単コンボリストに Kelly 配分を適用する。

    Args:
        combos:   [(combo_tuple, p, payout_estimate), ...]
                  combo_tuple = (first, second, third) 馬番
                  p = 推定確率, payout_estimate = 推定配当 (円/100円)
        budget:   総予算 (円)
        fraction: Kelly 係数
        min_bet:  最小賭け金 (円)
        unit:     丸め単位 (円)
        top_n:    上位 N コンボのみ対象 (Kelly f=0 を除外した後さらに絞る)
        ev_gate:  EV ≥ この値のコンボのみ対象 (EV = p × payout)

    Returns:
        [{"combo": (f,s,t), "bet": 金額, "p": 確率, "payout_est": 配当, "ev": 期待値, "kelly_f": f}]
        bet合計は budget 以下。
    """
    if not combos:
        return []

    # EV フィルタ (ev = p × payout/100)
    filtered = [
        (combo, p, pay)
        for (combo, p, pay) in combos
        if p > 0 and pay > 100 and p * (pay / 100.0) >= ev_gate
    ]
    if not filtered:
        return []

    # Kelly f* でソートして top_n を選択
    scored = []
    for combo, p, pay in filtered:
        f = kelly_fraction(p, pay) * fraction
        ev = p * (pay / 100.0)
        scored.append((combo, p, pay, f, ev))
    scored.sort(key=lambda x: x[3], reverse=True)   # f 降順
    scored = scored[:top_n]

    # 選択コンボに予算配分
    total_f = sum(x[3] for x in scored)
    if total_f <= 0.0:
        # フォールバック: 均等
        each = _floor_unit(budget // len(scored), unit) if scored else 0
        return [
            {"combo": c, "bet": each, "p": p, "payout_est": pay,
             "ev": round(p * pay / 100.0, 4), "kelly_f": 0.0}
            for c, p, pay, _, _ in scored
        ] if each >= min_bet else []

    result = []
    total_allocated = 0
    for combo, p, pay, f, ev in scored:
        amt = _floor_unit(int(budget * f / total_f), unit)
        if amt < min_bet:
            amt = min_bet
        result.append({
            "combo":       combo,
            "bet":         amt,
            "p":           round(p, 6),
            "payout_est":  round(pay, 1),
            "ev":          round(ev, 4),
            "kelly_f":     round(f, 4),
        })
        total_allocated += amt

    # 予算オーバーしている場合は最大コンボから削る
    if total_allocated > budget:
        result.sort(key=lambda x: x["kelly_f"])  # f 小さい順に削る
        for item in result:
            if total_allocated <= budget:
                break
            excess = total_allocated - budget
            cut = min(item["bet"] - min_bet, _ceil_unit(excess, unit))
            if cut > 0:
                item["bet"] -= cut
                total_allocated -= cut

    return result


# ============================================================
# ユーティリティ
# ============================================================

def _floor_unit(amount: int, unit: int) -> int:
    """unit の倍数に切り捨て。"""
    return (amount // unit) * unit


def _ceil_unit(amount: int, unit: int) -> int:
    """unit の倍数に切り上げ。"""
    return math.ceil(amount / unit) * unit


def uniform_allocate(
    n_combos: int,
    budget:   int,
    unit:     int = 100,
    min_bet:  int = 100,
) -> int:
    """均等割り配分: 1 コンボあたりの賭け金を返す。

    HALO の現行ロジックの互換実装（Kelly 導入前の フォールバック用）。
    """
    if n_combos <= 0:
        return 0
    each = (budget // n_combos // unit) * unit
    return max(min_bet, each)


# ============================================================
# self test
# ============================================================
if __name__ == "__main__":
    import io, sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print("=== kelly_fraction テスト (payout=¥/¥100) ===")
    cases = [
        (0.35,  400,  "単勝 4倍    "),
        (0.08, 2000,  "馬連 20倍   "),
        (0.02, 5000,  "三連単 50倍  "),
        (0.01,  500,  "期待値割れ   "),
    ]
    for p, pay, label in cases:
        f = kelly_fraction(p, pay)
        ev = p * pay / 100
        print(f"  {label}: p={p:.2f} payout={pay}  f*={f:.4f}  EV={ev:.3f}")

    print()
    print("=== kelly_allocate テスト ===")
    bets = {
        "単勝1": (0.35,  400),   # 4倍
        "単勝2": (0.15,  800),   # 8倍
        "馬連":  (0.08, 1500),   # 15倍
    }
    allocs = kelly_allocate(bets, budget=10000, fraction=0.25)
    total = sum(allocs.values())
    for bid, amt in allocs.items():
        print(f"  {bid}: ¥{amt:,}")
    print(f"  合計: ¥{total:,} / ¥10,000")

    print()
    print("=== kelly_allocate_trifecta テスト ===")
    # payout は ¥/¥100 (三連単の典型的な払戻額)
    # EV>1 のケース (p=0.08, pay=2000 → EV=1.60) を含める
    combos = [
        ((1, 2, 3), 0.08, 2000),   # 20倍  EV=1.60  ← Kelly f>0
        ((1, 3, 2), 0.05, 3000),   # 30倍  EV=1.50  ← Kelly f>0
        ((2, 1, 3), 0.02, 5000),   # 50倍  EV=1.00  ← Kelly f=0 (borderline)
        ((1, 2, 4), 0.01, 3200),   # 32倍  EV=0.32  ← EV gate で除外
    ]
    allocs2 = kelly_allocate_trifecta(combos, budget=9600, fraction=0.25, top_n=4, ev_gate=0.95)
    total2 = sum(x["bet"] for x in allocs2)
    for item in allocs2:
        print(f"  {item['combo']}: ¥{item['bet']:,}  "
              f"p={item['p']:.4f}  EV={item['ev']:.3f}  kelly_f={item['kelly_f']:.4f}")
    print(f"  合計: ¥{total2:,} / ¥9,600")
