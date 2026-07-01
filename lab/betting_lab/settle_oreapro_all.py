"""
settle_oreapro_all.py
=====================
2026 の週末 bundle x kekka を全部精算して、馬単3+三連複4 prob-first 戦略の
「2026 ライブ実績」を出す。バックテスト(2024-25, ~82%)と整合するかの検証。

実行: python settle_oreapro_all.py
"""
from __future__ import annotations
import io, json, sys
from pathlib import Path
import numpy as np
from oreapro_bets import race_bets, GAP_GATE
from settle_oreapro import load_kekka, STAKE, UNIT

try:
    if not isinstance(sys.stdout, io.TextIOWrapper) or (sys.stdout.encoding or "").lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

BASE = Path(__file__).parent
DATES = ["20260426", "20260516", "20260517", "20260523", "20260524",
         "20260530", "20260531", "20260606", "20260607", "20260613", "20260614"]


def settle_date(date):
    bundle = json.load(open(BASE / f"reports/cowork_input/{date}_bundle.json", encoding="utf-8"))
    try:
        kek = load_kekka(date)
    except Exception as e:
        print(f"  {date}: kekka読込失敗 {e}"); return None
    cost = ret = nbet = uh = sh = hitR = 0
    for r in bundle["races"]:
        rb = race_bets(r.get("horses", []), STAKE)
        if rb is None or rb["gap"] > GAP_GATE:
            continue
        res = kek.get(str(r["race_id"])[:16])
        if res is None:
            continue
        nbet += 1; cost += 7 * STAKE
        um = (res["win"], res["plc"]) in rb["umatan"]
        wt = tuple(sorted((res["win"], res["plc"], res["sho"])))
        sp = wt in [tuple(sorted(t)) for t in rb["sanpuku"]]
        rr = (res["umatan"] * UNIT if um else 0) + (res["sanpuku"] * UNIT if sp else 0)
        ret += rr; uh += um; sh += sp; hitR += (rr > 0)
    return {"date": date, "n": nbet, "cost": cost, "ret": ret, "um": uh, "sp": sh, "hitR": hitR}


def main():
    print("=" * 80)
    print("2026 ライブ精算: 馬単3点+三連複4点 prob-first + gapゲート (1点1400円)")
    print("=" * 80)
    print(f"{'日付':>10s} {'R数':>4s} {'投資':>9s} {'払戻':>10s} {'ROI':>7s} {'的中R':>6s} {'馬単':>4s} {'三連複':>5s}")
    tot = {"n": 0, "cost": 0, "ret": 0.0, "um": 0, "sp": 0, "hitR": 0}
    rois = []
    for d in DATES:
        s = settle_date(d)
        if not s:
            continue
        roi = s["ret"] / s["cost"] * 100 if s["cost"] else 0
        rois.append(roi)
        for k in tot:
            tot[k] += s[k]
        print(f"{s['date']:>10s} {s['n']:>4d} {s['cost']:>9,} {s['ret']:>10,.0f} {roi:>6.1f}% "
              f"{s['hitR']:>6d} {s['um']:>4d} {s['sp']:>5d}")
    print("-" * 80)
    troi = tot["ret"] / tot["cost"] * 100 if tot["cost"] else 0
    print(f"{'合計':>10s} {tot['n']:>4d} {tot['cost']:>9,} {tot['ret']:>10,.0f} {troi:>6.1f}% "
          f"{tot['hitR']:>6d} {tot['um']:>4d} {tot['sp']:>5d}")
    print(f"\n  週次ROIレンジ: {min(rois):.0f}% 〜 {max(rois):.0f}%  (黒字週 {sum(r>=100 for r in rois)}/{len(rois)})")
    print(f"  バックテスト(2024-25)平均 ~82% との比較 -> 2026ライブ実績 {troi:.1f}%")
    print(f"  的中率(プラス収支R/全R): {tot['hitR']}/{tot['n']} = {tot['hitR']/tot['n']*100:.1f}%")


if __name__ == "__main__":
    main()
