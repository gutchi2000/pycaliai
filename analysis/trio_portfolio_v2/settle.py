# -*- coding: utf-8 -*-
"""settle.py — 設計書 P10「三連複払戻・不成立・返還の決済」

決定後にだけ触る層。決定側 (candidates / allocate / collector) からは絶対に
import しない（P8 未来遮断）。

result 契約:
    {"status": "confirmed" | "void",        # void = レース不成立 → 全額返還
     "winning_combos": ["a-b-c", ...],      # 3着同着なら複数
     "payout_per100": {"a-b-c": 円},        # 100円あたり払戻
     "refunded_horses": [馬番, ...]}        # 取消・除外 → 当該馬を含む組は返還
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]


def combo_horses(key: str) -> tuple:
    return tuple(int(x) for x in key.split("-"))


def settle(bets: dict, result: dict) -> dict:
    """{combo: 購入額} × 結果 → 決済。返り値に返還・的中・回収を分けて残す。"""
    status = result.get("status", "confirmed")
    refunded_h = {int(x) for x in result.get("refunded_horses", [])}
    win = set(result.get("winning_combos", []))
    pay100 = result.get("payout_per100", {})
    stake = sum(bets.values())
    refund = hit_return = 0
    detail = {}
    for k, b in bets.items():
        hs = combo_horses(k)
        if status == "void" or (refunded_h & set(hs)):
            refund += b
            detail[k] = {"stake": b, "state": "refund", "back": b}
            continue
        if k in win:
            p = pay100.get(k)
            if p is None:
                detail[k] = {"stake": b, "state": "hit_unknown_payout", "back": None}
                continue
            v = int(round(b / 100.0 * float(p)))
            hit_return += v
            detail[k] = {"stake": b, "state": "hit", "back": v,
                         "payout_per100": float(p)}
        else:
            detail[k] = {"stake": b, "state": "lose", "back": 0}
    unknown = [k for k, d in detail.items() if d["state"] == "hit_unknown_payout"]
    total_back = refund + hit_return
    return {"stake": stake, "refund": refund, "hit_return": hit_return,
            "total_back": total_back, "pl": total_back - stake,
            "roi": (total_back / stake) if stake else None,
            "n_hit": sum(1 for d in detail.values() if d["state"] == "hit"),
            "unresolved": unknown, "ok": not unknown, "detail": detail}


# ============================================================
# kekka CSV → result 契約（決済専用の読み取り）
# ============================================================
def load_results(date_str: str) -> dict:
    """data/kekka/{date}.csv → {rid16: result契約}。

    確定着順 0 = 取消/除外/中止 → refunded_horses。
    1着同着でも上位3頭が一意なら winning_combos は1組。
    3着同着など上位が4頭以上になる場合は組が複数になり、kekka の払戻列だけでは
    組↔払戻の対応が確定しないので status に AMBIGUOUS を立てて fail-closed。
    """
    p = BASE / "data" / "kekka" / f"{date_str}.csv"
    if not p.exists():
        return {}
    rows = defaultdict(list)
    with open(p, encoding="cp932", errors="replace", newline="") as f:
        for r in csv.DictReader(f):
            rid_raw = str(r.get("レースID(新)", ""))
            if len(rid_raw) < 18:
                continue
            rows[rid_raw[:16]].append(
                {"umaban": int(rid_raw[16:18]),
                 "chaku": int(float(r.get("確定着順") or 0)),
                 "trio": (r.get("３連複") or "").strip()})
    out = {}
    for rid, hs in rows.items():
        refunded = [h["umaban"] for h in hs if h["chaku"] <= 0]
        run = [h for h in hs if h["chaku"] > 0]
        top = sorted(run, key=lambda h: h["chaku"])
        pays = sorted({float(h["trio"]) for h in hs if h["trio"]})
        if not top:
            out[rid] = {"status": "void", "winning_combos": [], "payout_per100": {},
                        "refunded_horses": refunded, "n_runners": len(run)}
            continue
        third = sorted(h["chaku"] for h in run)[2] if len(run) >= 3 else None
        inner = [h["umaban"] for h in run if third and h["chaku"] < third]
        tied = [h["umaban"] for h in run if third and h["chaku"] == third]
        combos = ["-".join(str(x) for x in sorted(inner + list(c)))
                  for c in combinations(sorted(tied), 3 - len(inner))] if third else []
        status = "confirmed"
        payout = {}
        if len(combos) == 1 and len(pays) == 1:
            payout = {combos[0]: pays[0]}
        elif len(combos) == len(pays) == 0:
            status = "void"
        else:
            status = "ambiguous_deadheat"      # 組↔払戻の対応が確定できない
        out[rid] = {"status": status, "winning_combos": combos,
                    "payout_per100": payout, "refunded_horses": refunded,
                    "n_runners": len(run), "n_payouts": len(pays)}
    return out


# ============================================================
# 単体テスト（P10）
# ============================================================
def _t_hit_refund_lose():
    res = {"status": "confirmed", "winning_combos": ["1-2-3"],
           "payout_per100": {"1-2-3": 1490.0}, "refunded_horses": [8]}
    r = settle({"1-2-3": 500, "1-2-4": 300, "1-8-9": 200}, res)
    assert r["detail"]["1-2-3"]["back"] == 7450          # 500/100*1490
    assert r["detail"]["1-2-4"]["state"] == "lose"
    assert r["detail"]["1-8-9"]["state"] == "refund" and r["refund"] == 200
    assert r["stake"] == 1000 and r["total_back"] == 7650 and r["pl"] == 6650
    return "的中(按分)/取消返還/ハズレ を分けて決済 (500円→7,450円 等)"


def _t_void_refunds_all():
    r = settle({"1-2-3": 400, "2-3-4": 600},
               {"status": "void", "winning_combos": [], "payout_per100": {}})
    assert r["refund"] == 1000 and r["pl"] == 0 and r["roi"] == 1.0
    return "レース不成立は全額返還 (ROI=1.0、損益0)"


def _t_unresolved_is_failclosed():
    r = settle({"1-2-3": 100}, {"status": "confirmed", "winning_combos": ["1-2-3"],
                                "payout_per100": {}})
    assert not r["ok"] and r["unresolved"] == ["1-2-3"]
    return "払戻不明の的中は ok=False で fail-closed（0円扱いにしない）"


def _t_real_kekka_roundtrip():
    """実 kekka で「的中組に100円」を決済すると払戻と一致するか（直近8開催）。"""
    import glob
    files = sorted(glob.glob(str(BASE / "data" / "kekka" / "2026*.csv")))[-8:]
    n = ok = amb = 0
    for f in files:
        for rid, res in load_results(Path(f).stem).items():
            if res["status"] != "confirmed" or not res["winning_combos"]:
                amb += res["status"] == "ambiguous_deadheat"
                continue
            k = res["winning_combos"][0]
            r = settle({k: 100}, res)
            n += 1
            ok += (r["detail"][k]["back"] == int(round(res["payout_per100"][k])))
            miss = settle({"1-2-3" if k != "1-2-3" else "1-2-4": 100}, res)
            ok += (list(miss["detail"].values())[0]["state"] in ("lose", "refund"))
    assert n > 0 and ok == 2 * n, (n, ok)
    return f"実 kekka {n}R で的中決済＝公表払戻に一致、非的中は lose/refund (曖昧同着 {amb}R は除外)"


TESTS = [_t_hit_refund_lose, _t_void_refunds_all, _t_unresolved_is_failclosed,
         _t_real_kekka_roundtrip]


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
