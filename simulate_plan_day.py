# -*- coding: utf-8 -*-
"""
simulate_plan_day.py — 枠プランの買い目を当日結果で決済 (「{date}ならどうだったか」)
================================================================================
build_bet_plan の枠 → compute_bets(force_floor) で買い目生成 → data/kekka/{date}.csv +
wide_kekka.csv の実配当で決済。勝負/準勝負/消化 枠別 ROI と日合計を出す。

実行: PYTHONUTF8=1 ./venv311/Scripts/python.exe simulate_plan_day.py 20260614
注: 過去日の振り返り。選択は bundle T-10 オッズ、採点は確定配当(単位100円)。
"""
from __future__ import annotations
import io, json, re, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import pandas as pd
import compute_bets as CB

BASE = Path(__file__).parent
def rid16(x): return re.sub(r"\D", "", str(x))[:16]
def _n(x):
    try: return float(str(x).replace(",", "")) if str(x).strip() not in ("", "nan") else 0.0
    except Exception: return 0.0


def load_plan(date):
    p = json.loads((BASE / "reports" / "bet_plan" / f"{date}.json").read_text(encoding="utf-8"))
    out = {}
    for tier, rows in p["tiers"].items():
        if tier in ("見送り",): continue
        for r in rows:
            if r.get("budget"):
                out[rid16(r["rid"])] = (tier, int(r["budget"]))
    return out


def load_kekka(date):
    df = pd.read_csv(BASE / "data" / "kekka" / f"{date}.csv", encoding="cp932", low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    df["rid16"] = df["レースID(新)"].map(rid16)
    df["着"] = pd.to_numeric(df["確定着順"], errors="coerce")
    df["馬番"] = pd.to_numeric(df["馬番"], errors="coerce")
    out = {}
    for rid, g in df.groupby("rid16"):
        g1 = g[g["着"] == 1]; g2 = g[g["着"] == 2]; g3 = g[g["着"] == 3]
        if len(g1) == 0 or len(g2) == 0 or len(g3) == 0: continue
        w, p, s = int(g1.iloc[0]["馬番"]), int(g2.iloc[0]["馬番"]), int(g3.iloc[0]["馬番"])
        fuku = {int(r["馬番"]): _n(r["複勝配当"]) for _, r in g.iterrows() if r["着"] in (1, 2, 3)}
        out[rid] = {"win": w, "plc": p, "sho": s, "place": g1.iloc[0]["場所"], "R": int(g1.iloc[0]["Ｒ"]),
                    "tansho": _n(g1.iloc[0]["単勝配当"]), "fuku": fuku,
                    "umaren": _n(g1.iloc[0]["馬連"])}
    return out


def load_wide(date):
    """wide_kekka.csv (ヘッダ無) → {(場所,R): {frozenset(i,j): pay}} for the date。"""
    y, m, d = int(date[:4]), int(date[4:6]), int(date[6:8])
    df = pd.read_csv(BASE / "data" / "kekka" / "wide_kekka.csv", encoding="cp932", header=None, low_memory=False)
    out = {}
    for _, r in df.iterrows():
        try:
            if int(r[0]) != y or int(r[1]) != m or int(r[2]) != d: continue
        except Exception:
            continue
        place = str(r[3]); R = int(r[4]); pairs = {}
        for part in str(r[9]).split("/"):
            mm = re.search(r"(\d+)\s*-\s*(\d+)\s*\\?\s*([\d,]+)", part)
            if mm:
                pairs[frozenset((int(mm.group(1)), int(mm.group(2))))] = _n(mm.group(3))
        if pairs: out[(place, R)] = pairs
    return out


def settle(bet, ko, wide_pairs):
    """1点の払戻(¥)を返す。bet: {馬券種,買い目,購入額}。"""
    kind = bet["馬券種"]; sel = bet["買い目"]; amt = bet["購入額"]; unit = amt / 100.0
    if kind == "単勝":
        return ko["tansho"] * unit if int(sel) == ko["win"] else 0.0
    if kind == "複勝":
        b = int(sel)
        return ko["fuku"].get(b, 0.0) * unit if b in (ko["win"], ko["plc"], ko["sho"]) else 0.0
    if kind == "馬連":
        i, j = (int(x) for x in sel.split("-"))
        return ko["umaren"] * unit if {i, j} == {ko["win"], ko["plc"]} else 0.0
    if kind == "ワイド":
        i, j = (int(x) for x in sel.split("-"))
        pay = (wide_pairs or {}).get(frozenset((i, j)), 0.0)
        return pay * unit
    return 0.0


def main():
    date = sys.argv[1] if len(sys.argv) > 1 else "20260614"
    plan = load_plan(date)
    kekka = load_kekka(date)
    wide = load_wide(date)
    bundle = json.loads((BASE / "reports" / "cowork_input" / f"{date}_bundle.json").read_text(encoding="utf-8"))
    races = {rid16(r.get("race_id")): r for r in bundle["races"]}

    tiers = {"勝負": [0, 0.0, 0, 0], "準勝負": [0, 0.0, 0, 0], "消化": [0, 0.0, 0, 0]}  # stake,ret,bets,hits
    print("=" * 90)
    print(f"枠プラン 6/14 実結果決済  [選択=bundle T-10オッズ / 採点=確定配当]")
    print("=" * 90)
    day_stake = day_ret = 0

    # 枠順で並べる
    order = {"勝負": 0, "準勝負": 1, "消化": 2}
    items = sorted(plan.items(), key=lambda kv: (order[kv[1][0]], -_plan_conf(kv[0], races)))
    cur_tier = None
    for rid, (tier, budget) in items:
        r = races.get(rid)
        if r is None: continue
        e = CB.compute_race_bets(r, live_dir=None, budget=budget, force_floor=True)
        ko = kekka.get(rid)
        wp = wide.get((ko["place"], ko["R"])) if ko else None
        rm = r.get("race_meta", {})
        if tier != cur_tier:
            print(f"\n{'━'*4} 【{tier}枠】 {'━'*40}")
            cur_tier = tier
        # marks
        hs = sorted(r.get("horses", []), key=lambda h: h.get("ai_rank") or 99)
        marks = "  ".join(f"{h.get('mark')}{h.get('horse_name')}" for h in hs[:5] if h.get("mark"))
        stake = sum(b["購入額"] for b in e["bets"]); ret = 0.0
        print(f"\n● {rm.get('place','')}{rm.get('course','')} {rm.get('race_name','') or rm.get('class','')} "
              f"[{e['race_nature']}] {len(e['bets'])}点 ¥{stake:,}")
        if ko:
            print(f"   結果: {ko['win']}-{ko['plc']}-{ko['sho']}着  | 印 {marks}")
        for b in e["bets"]:
            pay = settle(b, ko, wp) if ko else 0.0
            ret += pay
            hit = f"→的中 ¥{int(pay):,}" if pay > 0 else "→外"
            print(f"   {b['馬券種']:3s} {b['買い目']:7s} ¥{b['購入額']:>5,}  {b['理由']}  {hit}")
            t = tiers[tier]; t[2] += 1; t[3] += 1 if pay > 0 else 0
        t = tiers[tier]; t[0] += stake; t[1] += ret
        day_stake += stake; day_ret += ret
        print(f"   レース収支: ¥{int(ret-stake):+,} (回収 {ret/stake*100 if stake else 0:.0f}%)")

    print("\n" + "=" * 90)
    print(f"{'枠':6s} {'R数':>4s} {'投資':>10s} {'払戻':>10s} {'収支':>11s} {'回収率':>7s} {'的中':>8s}")
    nbet_tot = 0
    for tier in ("勝負", "準勝負", "消化"):
        st, rt, nb, nh = tiers[tier]
        nbet_tot += nb
        nr = sum(1 for _, (tr, _b) in plan.items() if tr == tier and rid16 in races)
        print(f"{tier:6s} {sum(1 for k,(tr,_b) in plan.items() if tr==tier):>4d} ¥{int(st):>9,} ¥{int(rt):>9,} "
              f"¥{int(rt-st):>+10,} {rt/st*100 if st else 0:>6.1f}% {nh}/{nb}")
    print("-" * 90)
    n_races = len({k for k in plan})
    print(f"{'合計':6s} {n_races:>4d} ¥{int(day_stake):>9,} ¥{int(day_ret):>9,} ¥{int(day_ret-day_stake):>+10,} "
          f"{day_ret/day_stake*100 if day_stake else 0:>6.1f}%  ({nbet_tot}点)")
    print(f"\nプロ条件: {n_races}R(≥10) ∧ ¥{int(day_stake):,}(≥¥100,000) → "
          f"{'✅ 達成' if n_races>=10 and day_stake>=100000 else '❌ 未達'}")
    print("=" * 90)


def _plan_conf(rid, races):
    r = races.get(rid)
    if not r: return 0
    rc = r.get("race_confidence", {})
    return (rc.get("top2_concentration") or 0)


if __name__ == "__main__":
    main()
