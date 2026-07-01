# -*- coding: utf-8 -*-
"""複数日横断: Cowork印の「穴」成績を検証する.

- 印(◎〇▲△)のオッズ帯別 複勝/連対/勝率
- AIが市場より強気(ai_vs_market=over=穴推し)な馬は本当に来るか
- race_nature(穴推奨 vs 中堅)別の買い目ROI
- 的中した買い目の配当オッズ分布(高配当=穴を取れているか)
"""
import sys
import json
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
base = r"E:\PyCaLiAI"
dates = ["20260426", "20260516", "20260517", "20260523", "20260524"]


def fnum(x):
    try:
        return float(str(x).strip())
    except Exception:
        return None


W = pd.read_parquet(rf"{base}\data\wide_payouts_2016-2025.parquet")
rows, betrows = [], []

for date in dates:
    bundle = json.load(open(rf"{base}\reports\cowork_input\{date}_bundle.json", encoding="utf-8"))
    bets_raw = json.load(open(rf"{base}\reports\cowork_output\{date}_bets.json", encoding="utf-8"))
    bet_list = bets_raw["bets"] if isinstance(bets_raw, dict) else bets_raw
    k = pd.read_csv(rf"{base}\data\kekka\{date}.csv", encoding="cp932", header=0)
    k.columns = ["hi", "place", "R", "waku", "umaban", "name", "chaku", "rid",
                 "tan", "fuku", "umaren", "wide", "umatan", "san3p", "san3t"]
    k["rkey"] = k["rid"].astype(str).str[:16]
    k["chaku_n"] = pd.to_numeric(k["chaku"], errors="coerce")

    cmap, pay, fuku = {}, {}, {}
    for rkey, g in k.groupby("rkey"):
        cmap[rkey] = {int(r["umaban"]): (int(r["chaku_n"]) if pd.notna(r["chaku_n"]) else None)
                      for _, r in g.iterrows()}
        fuku[rkey] = {int(r["umaban"]): fnum(r["fuku"]) for _, r in g.iterrows()
                      if pd.notna(r["chaku_n"]) and int(r["chaku_n"]) <= 3}
        first = g[g["chaku_n"] == 1]
        if len(first):
            fr = first.iloc[0]
            pay[rkey] = {"tansho": fnum(fr["tan"]), "umaren": fnum(fr["umaren"]),
                         "wide": fnum(fr["wide"]), "umatan": fnum(fr["umatan"])}

    Wd = W[W["date"] == int(date)]
    wmap = {}
    for _, r in Wd.iterrows():
        d = {}
        for a in "123":
            d[frozenset([int(r[f"w{a}_i"]), int(r[f"w{a}_j"])])] = float(r[f"w{a}_pay"])
        wmap[str(r["race_id"])] = d

    nature = {r["race_id"]: r.get("race_nature") for r in bet_list}

    for race in bundle["races"]:
        rid = race["race_id"]
        cm = cmap.get(rid, {})
        fs = race.get("race_meta", {}).get("field_size")
        for h in race["horses"]:
            mk = h.get("mark")
            if not mk:
                continue
            pos = cm.get(int(h["umaban"]))
            rows.append(dict(date=date, mark=mk, odds=h.get("tansho_odds"),
                             vs=h.get("ai_vs_market"), fs=fs, nature=nature.get(rid),
                             in3=(pos is not None and pos <= 3),
                             in2=(pos is not None and pos <= 2), win=(pos == 1)))

    for race in bet_list:
        if not race.get("bets"):
            continue
        rid = race["race_id"]
        cm = cmap.get(rid, {})
        pk = pay.get(rid, {})
        fk = fuku.get(rid, {})
        wm = wmap.get(rid, {})
        inv = {v: u for u, v in cm.items() if v}
        fin = {p: inv.get(p) for p in (1, 2, 3)}
        top3 = set(u for u in fin.values() if u)
        for b in race["bets"]:
            kind, sel_full, amt_total = b["馬券種"], str(b["買い目"]), b["購入額"]
            points = [p.strip() for p in sel_full.split(",") if p.strip()]
            amt = amt_total / len(points)
            for sel in points:
                nums = [int(x) for x in sel.replace("=", "-").split("-")]
                hit = unk = False
                ret = 0
                if kind == "単勝":
                    if fin[1] == nums[0]:
                        hit, ret = True, (pk.get("tansho") or 0) * amt / 100
                elif kind == "複勝":
                    if nums[0] in fk:
                        hit, ret = True, fk[nums[0]] * amt / 100
                elif kind == "馬連":
                    if fin[1] and fin[2] and {fin[1], fin[2]} == set(nums):
                        hit, ret = True, (pk.get("umaren") or 0) * amt / 100
                elif kind == "馬単":
                    if fin[1] == nums[0] and fin[2] == nums[1]:
                        hit, ret = True, (pk.get("umatan") or 0) * amt / 100
                elif kind in ("ワイド", "拡連複"):
                    if len(nums) == 2 and set(nums) <= top3:
                        hit = True
                        fs2 = frozenset(nums)
                        if fs2 in wm:
                            ret = wm[fs2] * amt / 100
                        elif pk.get("wide") and {fin[1], fin[2]} == set(nums):
                            ret = pk["wide"] * amt / 100
                        else:
                            unk = True
                betrows.append(dict(nature=race.get("race_nature"), kind=kind, amt=amt,
                                    hit=hit, ret=ret, unk=unk,
                                    payodds=(ret / amt if hit and not unk and amt else None)))

R = pd.DataFrame(rows)
B = pd.DataFrame(betrows)


def band(o):
    if o is None:
        return "NA"
    if o < 4:
        return "1_人気<4"
    if o < 7:
        return "2_中4-7"
    if o < 15:
        return "3_中穴7-15"
    return "4_大穴15+"


R["band"] = R["odds"].apply(band)
print(f"対象: {dates}  印サンプル {len(R)}  買い目 {len(B)}\n")

print("=== A. ◎ オッズ帯別成績 ===")
hon = R[R["mark"] == "◎"]
g = hon.groupby("band").agg(n=("in3", "size"), 勝率=("win", "mean"),
                            連対率=("in2", "mean"), 複勝率=("in3", "mean"))
print(g.round(3).to_string(), "\n")

print("=== B. 印別 複勝率/連対率 ===")
print(R.groupby("mark").agg(n=("in3", "size"), 複勝率=("in3", "mean"),
                            連対率=("in2", "mean")).round(3).to_string(), "\n")

print("=== C. ai_vs_market 別 印馬 複勝率 ===")
print(R.groupby("vs").agg(n=("in3", "size"), 複勝率=("in3", "mean"),
                          平均オッズ=("odds", "mean")).round(2).to_string())
print("--- over(AIが市場より強気=穴推し)馬をオッズ帯で ---")
ov = R[R["vs"] == "over"]
print(ov.groupby("band").agg(n=("in3", "size"), 複勝率=("in3", "mean")).round(3).to_string(), "\n")

print("=== D. race_nature 別 買い目ROI ===")
for nat, gg in B.groupby("nature"):
    inv, ret, hit = gg["amt"].sum(), gg["ret"].sum(), gg["hit"].sum()
    print(f"  {nat}: {len(gg)}点 的中{hit} 投資{inv} 払戻{int(ret)} "
          f"ROI{ret / inv * 100:.0f}% (配当未取得{gg.unk.sum()})")
print()

print("=== E. 的中買い目の配当オッズ分布 (高配当=穴を取れているか) ===")
hb = B[B["hit"] & ~B["unk"]].copy()


def pband(o):
    if o < 2:
        return "1_<2倍"
    if o < 5:
        return "2_2-5倍"
    if o < 15:
        return "3_5-15倍"
    return "4_15倍+"


hb["pb"] = hb["payodds"].apply(pband)
print(hb.groupby("pb").agg(的中数=("ret", "size"), 計払戻=("ret", "sum")).to_string(), "\n")

print("=== F. 馬券種別 ROI ===")
piv = B.groupby("kind").agg(n=("amt", "size"), inv=("amt", "sum"), ret=("ret", "sum"),
                            hit=("hit", "sum"), unk=("unk", "sum"))
piv["ROI%"] = (piv["ret"] / piv["inv"] * 100).round(0)
print(piv.to_string())
