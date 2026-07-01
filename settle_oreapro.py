"""
settle_oreapro.py
=================
oreapro_bets.py が出した買い目を data/kekka/{date}.csv の確定配当で精算。
馬単3点+三連複4点 (1点1,400円) の的中/払戻/週次ROIを出す。

実行: python settle_oreapro.py --date 20260614
"""
from __future__ import annotations
import argparse, io, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from oreapro_bets import race_bets, GAP_GATE

try:
    if not isinstance(sys.stdout, io.TextIOWrapper) or (sys.stdout.encoding or "").lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

BASE = Path(__file__).parent
STAKE = 1400; UNIT = STAKE // 100   # 1400円 = 14 単位(×100円配当)


def load_kekka(date):
    p = BASE / f"data/kekka/{date}.csv"
    dk = pd.read_csv(p, encoding="cp932")
    cols = list(dk.columns)
    def find(*keys, exclude=()):
        for c in cols:
            if any(k in c for k in keys) and not any(e in c for e in exclude):
                return c
        return None
    c_chaku = find("着")
    c_uma   = find("馬番")
    c_rid   = find("レースID")
    c_umatan = find("馬単")
    c_sanpuku = find("３連複", "3連複", "連複")
    print(f"[kekka] cols: 着順={c_chaku} 馬番={c_uma} rid={c_rid} 馬単={c_umatan} 三連複={c_sanpuku}")
    dk[c_chaku] = pd.to_numeric(dk[c_chaku], errors="coerce")
    dk[c_uma]   = pd.to_numeric(dk[c_uma], errors="coerce")
    dk["_rid"] = dk[c_rid].astype(str).str.replace(r"\D", "", regex=True).str[:16]
    for c in (c_umatan, c_sanpuku):
        dk[c] = pd.to_numeric(dk[c].astype(str).str.replace(",", ""), errors="coerce")
    out = {}
    for rid, g in dk.groupby("_rid"):
        r1 = g[g[c_chaku] == 1]; r2 = g[g[c_chaku] == 2]; r3 = g[g[c_chaku] == 3]
        if len(r1) == 0 or len(r2) == 0 or len(r3) == 0:
            continue
        out[rid] = {
            "win": int(r1.iloc[0][c_uma]), "plc": int(r2.iloc[0][c_uma]), "sho": int(r3.iloc[0][c_uma]),
            "umatan": float(r1.iloc[0][c_umatan]), "sanpuku": float(r1.iloc[0][c_sanpuku]),
        }
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--date", required=True)
    ap.add_argument("--pick", type=int, default=6)
    args = ap.parse_args()

    bundle = json.load(open(BASE / f"reports/cowork_input/{args.date}_bundle.json", encoding="utf-8"))
    kek = load_kekka(args.date)
    print(f"[load] kekka races={len(kek)}  bundle races={len(bundle['races'])}\n")

    # ゲート通過レースの買い目を生成
    rows = []
    for r in bundle["races"]:
        rb = race_bets(r.get("horses", []), STAKE)
        if rb is None or rb["gap"] > GAP_GATE:
            continue
        rid = str(r["race_id"])[:16]
        res = kek.get(rid)
        if res is None:
            continue
        meta = r.get("race_meta", {})
        rb["label"] = f"{rid} {meta.get('place','')}".strip()
        # 馬単的中?
        um_hit = (res["win"], res["plc"]) in rb["umatan"]
        # 三連複的中?
        wt = tuple(sorted((res["win"], res["plc"], res["sho"])))
        sp_hit = wt in [tuple(sorted(t)) for t in rb["sanpuku"]]
        ret = (res["umatan"] * UNIT if um_hit else 0) + (res["sanpuku"] * UNIT if sp_hit else 0)
        rows.append({"rid": rid, "label": rb["label"], "cost": 7 * STAKE, "ret": ret,
                     "um_hit": um_hit, "sp_hit": sp_hit,
                     "um_pay": res["umatan"], "sp_pay": res["sanpuku"],
                     "result": f"{res['win']}→{res['plc']}→{res['sho']}"})
    rows.sort(key=lambda x: x["label"])

    def summarize(subset, title):
        c = sum(x["cost"] for x in subset); rr = sum(x["ret"] for x in subset)
        uh = sum(x["um_hit"] for x in subset); sh = sum(x["sp_hit"] for x in subset)
        hit_r = sum(1 for x in subset if x["ret"] > 0)
        print("=" * 76)
        print(f"【{title}】 {len(subset)}R")
        for x in subset:
            tag = []
            if x["um_hit"]: tag.append(f"馬単的中 {int(x['um_pay'])}円")
            if x["sp_hit"]: tag.append(f"三連複的中 {int(x['sp_pay'])}円")
            mark = "  ".join(tag) if tag else "ハズレ"
            pl = x["ret"] - x["cost"]
            print(f"  {x['label']:<26s} 結果{x['result']:<10s} 払戻{x['ret']:>7,}  収支{pl:>+8,}  {mark}")
        roi = rr / c * 100 if c else 0
        print(f"  --- 投資 {c:,} / 払戻 {rr:,} / 収支 {rr-c:+,} / ROI {roi:.1f}% "
              f"(的中{hit_r}/{len(subset)}R 馬単{uh} 三連複{sh}) ---")
    summarize(rows[:args.pick], f"★買い {args.pick}R (函館1-{args.pick}R)")
    print()
    summarize(rows, "参考: ゲート通過 全レース")


if __name__ == "__main__":
    main()
