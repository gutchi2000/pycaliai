# -*- coding: utf-8 -*-
"""allocate.py — 設計書 §6「整数配分・75%下限・予算」

制約（すべて保守価格 O_low・保守確率 q_low の上で判定する）:
    B = Σ b_c,  b_c は 100円単位かつ b_c >= 100,  B <= 10,000
    ∀c∈S: b_c * O_low(c) >= 0.75 * B                     ← トリガミ床（§0.1-1）
    E(b)  = Σ q_low(c)*b_c*O_low(c) / B  > 1.00          ← 保守EV（§0.1-2）
    B    <= floor_to_100( 10000 * min(1, max(0,(E-1)/0.10)) )
採用は 期待利益 B*(E-1) 最大。同値は 組数少 → 総投資少 → 保守払戻のばらつき小。

■ 実装上の定理（P7 で brute force と一致を確認済み）
    期待利益 = Σ b_c*(q_low_c*O_low_c - 1) = Σ b_c*e_c は b について線形。
    B を固定すると床 m_c は S と B だけで決まるので、最適解は
      「S の各組に床 m_c を置き、残差を e_c 最大の組に全部乗せる」。
    さらに e_c が最大の組だけを S にした singleton は、
      (a) 期待利益、(b) E、(c) それゆえ B_cap の全てを同時に最大化する。
    ⇒ 設計書の目的関数のままだと **最適解は常に1点買い** になり、
      75%下限は決して有効制約にならない（O_low>=0.75 は常に真）。
    この構造的帰結は preflight レポートで明示する（§0.2 により式は変更しない）。
"""
from __future__ import annotations

import math
from itertools import combinations

UNIT = 100
BUDGET_MAX = 10_000
FLOOR_RATIO = 0.75
EV_MIN = 1.00
CAP_SLOPE = 0.10


def _ceil_unit(x: float, unit: int = UNIT) -> int:
    return int(math.ceil(x / unit - 1e-9)) * unit


def _floor_unit(x: float, unit: int = UNIT) -> int:
    return int(math.floor(x / unit + 1e-9)) * unit


def budget_cap(E: float, budget_max: int = BUDGET_MAX,
               slope: float = CAP_SLOPE, ev_min: float = EV_MIN) -> int:
    """§6.2 B_cap(E) を 100円格子に落とした値。E=1.01→1,000 / 1.05→5,000 / >=1.10→10,000。"""
    frac = min(1.0, max(0.0, (E - ev_min) / slope))
    return _floor_unit(budget_max * frac)


def eval_alloc(cands: list, bets: dict, floor_ratio: float = FLOOR_RATIO,
               budget_max: int = BUDGET_MAX, unit: int = UNIT,
               ev_min: float = EV_MIN) -> dict:
    """与えられた配分の全制約チェック + 指標。bets={key: 円}。"""
    idx = {c["key"]: c for c in cands}
    B = sum(bets.values())
    ret = {"bets": dict(bets), "B": B, "n": len(bets)}
    viol = []
    if B <= 0:
        viol.append("empty")
    if B > budget_max:
        viol.append(f"B>{budget_max}")
    for k, b in bets.items():
        if k not in idx:
            viol.append(f"unknown combo {k}")
            continue
        if b < unit or b % unit:
            viol.append(f"{k}: b={b} が100円格子/最低100円に反する")
        if b * idx[k]["o_low"] < floor_ratio * B - 1e-9:
            viol.append(f"{k}: 払戻 {b*idx[k]['o_low']:.0f} < 75%下限 {floor_ratio*B:.0f}")
    gross = sum(idx[k]["q_low"] * b * idx[k]["o_low"] for k, b in bets.items() if k in idx)
    E = gross / B if B else 0.0
    cap = budget_cap(E, budget_max, ev_min=ev_min)
    if E <= ev_min:
        viol.append(f"E={E:.4f} <= {ev_min}")
    if B > cap:
        viol.append(f"B={B} > B_cap={cap}")
    pay = [b * idx[k]["o_low"] for k, b in bets.items() if k in idx]
    mu = sum(pay) / len(pay) if pay else 0.0
    ret.update({"E": E, "expected_profit": B * (E - 1.0), "B_cap": cap,
                "payout_var": (sum((p - mu) ** 2 for p in pay) / len(pay)) if pay else 0.0,
                "min_payout_ratio": min((p / B for p in pay), default=0.0),
                "violations": viol, "feasible": not viol})
    return ret


def _best_for_subset_B(cands: list, subset: tuple, B: int,
                       floor_ratio: float, unit: int) -> dict | None:
    """S と B を固定したときの最適整数配分（床を置いて残差を e 最大へ）。"""
    idx = {c["key"]: c for c in cands}
    mins = {}
    for k in subset:
        c = idx[k]
        m = max(unit, _ceil_unit(floor_ratio * B / c["o_low"], unit))
        mins[k] = m
    tot = sum(mins.values())
    if tot > B:
        return None
    best = max(subset, key=lambda k: (idx[k]["q_low"] * idx[k]["o_low"] - 1.0, -idx[k]["o_low"]))
    bets = dict(mins)
    bets[best] += B - tot
    return bets


def optimize(cands: list, budget_max: int = BUDGET_MAX, unit: int = UNIT,
             floor_ratio: float = FLOOR_RATIO, ev_min: float = EV_MIN,
             mode: str = "proof", max_subset: int = 12) -> dict:
    """設計書 §6.2 の探索。

    cands: [{"key": "1-2-3", "q_low": float, "o_low": float}, ...]
    mode : "proof"      … 定理に基づく閉形式（B ごとに singleton を評価）
           "exhaustive" … 全非空部分集合 × 全 B を総当たり（テスト・検証用、n<=max_subset）
    返り値 {"decision": "bet"/"skip", ...}
    """
    cands = [c for c in cands if c.get("o_low") and c["o_low"] > 0
             and c.get("q_low") is not None and c["q_low"] >= 0]
    if not cands:
        return {"decision": "skip", "reason": "NO_CANDIDATES"}
    keys = [c["key"] for c in cands]
    if mode == "exhaustive":
        if len(keys) > max_subset:
            raise ValueError(f"exhaustive は n<={max_subset} まで (n={len(keys)})")
        subsets = [s for r in range(1, len(keys) + 1) for s in combinations(keys, r)]
    else:
        subsets = [(k,) for k in keys]

    sols = []
    for B in range(unit, budget_max + 1, unit):
        for S in subsets:
            bets = _best_for_subset_B(cands, S, B, floor_ratio, unit)
            if bets is None:
                continue
            r = eval_alloc(cands, bets, floor_ratio, budget_max, unit, ev_min)
            if r["feasible"]:
                sols.append(r)
    if not sols:
        return {"decision": "skip", "reason": "NO_FEASIBLE_PORTFOLIO",
                "best_edge": max(c["q_low"] * c["o_low"] - 1.0 for c in cands),
                "n_candidates": len(cands)}
    sols.sort(key=lambda r: (-r["expected_profit"], r["n"], r["B"], r["payout_var"]))
    best = sols[0]
    return {"decision": "bet", "bets": best["bets"], "B": best["B"], "E": best["E"],
            "expected_profit": best["expected_profit"], "B_cap": best["B_cap"],
            "n_combos": best["n"], "min_payout_ratio": best["min_payout_ratio"],
            "payout_var": best["payout_var"], "n_feasible": len(sols),
            "mode": mode}


# ============================================================
# 単体テスト（設計書 P7: 人工ケースで 100円・1万円・75%・見送りを全件検証）
# ============================================================
def _c(key, q, o):
    return {"key": key, "q_low": q, "o_low": o}


def _t_skip_when_no_edge():
    cands = [_c("1-2-3", 0.05, 15.0), _c("1-2-4", 0.03, 20.0)]   # e = -0.25, -0.40
    r = optimize(cands)
    assert r["decision"] == "skip" and r["reason"] == "NO_FEASIBLE_PORTFOLIO", r
    return "EV<=1 の候補しかなければ見送り (NO_FEASIBLE_PORTFOLIO)"


def _t_min_100_when_thin_edge():
    # E=1.0005 → B_cap = floor100(10000*0.005) = 0 → 見送り（100円未満しか作れない）
    r = optimize([_c("1-2-3", 0.06670, 15.0)])
    assert r["decision"] == "skip", r
    # E=1.02 → B_cap = 2,000
    r2 = optimize([_c("1-2-3", 0.0680, 15.0)])
    assert r2["decision"] == "bet" and r2["B"] == 2000, r2
    return "薄い優位は B_cap<100 で見送り / E=1.02 なら B=2,000 に自動縮小"


def _t_full_budget_at_E110():
    r = optimize([_c("1-2-3", 0.10, 12.0)])          # E = 1.20 → cap 10,000
    assert r["decision"] == "bet" and r["B"] == 10_000, r
    assert abs(r["E"] - 1.20) < 1e-9
    return "E>=1.10 で上限 1万円まで張る (E=1.20→B=10,000)"


def _t_floor75_enforced():
    # 2点強制ケース: 75%下限を満たす配分だけが feasible
    cands = [_c("1-2-3", 0.20, 6.0), _c("1-2-4", 0.20, 6.0)]
    r = eval_alloc(cands, {"1-2-3": 5000, "1-2-4": 5000})
    assert r["feasible"], r                                     # 5000*6=30000 >= 7500
    bad = eval_alloc(cands, {"1-2-3": 9900, "1-2-4": 100})
    assert not bad["feasible"] and any("75%" in v for v in bad["violations"]), bad
    #   100*6=600 < 0.75*10000=7500 → 床違反を確実に検出
    return "75%下限は eval_alloc で厳密に検出（違反配分を feasible にしない）"


def _t_grid_and_bounds():
    cands = [_c("1-2-3", 0.10, 12.0)]
    assert not eval_alloc(cands, {"1-2-3": 150})["feasible"]     # 100円格子違反
    assert not eval_alloc(cands, {"1-2-3": 50})["feasible"]      # 最低100円違反
    assert not eval_alloc(cands, {"1-2-3": 10_100})["feasible"]  # 予算上限違反
    r = optimize(cands)
    assert r["B"] % 100 == 0 and 100 <= r["B"] <= 10_000
    assert all(b % 100 == 0 and b >= 100 for b in r["bets"].values())
    return "100円格子・最低100円・上限1万円を全経路で強制"


def _t_exhaustive_matches_closed_form():
    """定理の検証: 全部分集合 × 全B の総当たり最適解と閉形式が一致する。"""
    import random
    random.seed(3)
    for _ in range(20):
        cands = [_c(f"c{i}", random.uniform(0.01, 0.25), random.uniform(3.0, 60.0))
                 for i in range(5)]
        a = optimize(cands, mode="proof")
        b = optimize(cands, mode="exhaustive")
        assert a["decision"] == b["decision"], (a, b)
        if a["decision"] == "bet":
            assert abs(a["expected_profit"] - b["expected_profit"]) < 1e-6, (a, b)
            assert a["B"] == b["B"] and a["n_combos"] == b["n_combos"] == 1, (a, b)
    return "総当たり(2^5×100通り)と閉形式が全20ケースで一致。最適は常に1点＝75%床は不活性"


def _t_scale_of_bcap():
    for q, o, want in [(0.101, 10.0, 1000), (0.105, 10.0, 5000), (0.110, 10.0, 10000)]:
        r = optimize([_c("1-2-3", q, o)])
        assert r["decision"] == "bet" and r["B"] == want, (q, o, r)
    return "E=1.01→1,000 / 1.05→5,000 / 1.10→10,000 の B_cap 段階が設計書どおり"


TESTS = [_t_skip_when_no_edge, _t_min_100_when_thin_edge, _t_full_budget_at_E110,
         _t_floor75_enforced, _t_grid_and_bounds, _t_exhaustive_matches_closed_form,
         _t_scale_of_bcap]


def run_tests() -> list[dict]:
    out = []
    for t in TESTS:
        try:
            out.append({"test": t.__name__, "ok": True, "detail": t()})
        except Exception as e:
            out.append({"test": t.__name__, "ok": False,
                        "detail": f"{type(e).__name__}: {e}"})
    return out


if __name__ == "__main__":
    for r in run_tests():
        print(("  OK  " if r["ok"] else "  NG  ") + r["test"] + " — " + r["detail"])
