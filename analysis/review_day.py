# -*- coding: utf-8 -*-
"""Cowork レース回顧ツール.

bundle.json(印・確率) + bets.json(買い目・寸評) + kekka.csv(着順・払戻) を
突き合わせ、Cowork の読みの当否とレース収支を回顧する。

usage: python analysis/review_day.py 20260524
"""
import sys
import json
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

date = sys.argv[1] if len(sys.argv) > 1 else "20260524"
base = r"E:\PyCaLiAI"

bundle = json.load(open(rf"{base}\reports\cowork_input\{date}_bundle.json", encoding="utf-8"))
bets = json.load(open(rf"{base}\reports\cowork_output\{date}_bets.json", encoding="utf-8"))

k = pd.read_csv(rf"{base}\data\kekka\{date}.csv", encoding="cp932", header=0)
k.columns = ["hi", "place", "R", "waku", "umaban", "name", "chaku", "rid",
             "tan", "fuku", "umaren", "wide", "umatan", "san3p", "san3t"]
k["rkey"] = k["rid"].astype(str).str[:16]
k["chaku_n"] = pd.to_numeric(k["chaku"], errors="coerce")


def fnum(x):
    try:
        return float(str(x).strip())
    except Exception:
        return None


kekka = {}
for rkey, g in k.groupby("rkey"):
    finish, fuku = {}, {}
    for _, r in g.iterrows():
        c = r["chaku_n"]
        if pd.notna(c) and 1 <= int(c) <= 3:
            ci = int(c)
            finish[ci] = (int(r["umaban"]), str(r["name"]))
            fv = fnum(r["fuku"])
            if fv:
                fuku[int(r["umaban"])] = fv
    first = g[g["chaku_n"] == 1]
    pay, win_odds = {}, None
    if len(first):
        fr = first.iloc[0]
        pay = {"tansho": fnum(fr["tan"]), "umaren": fnum(fr["umaren"]),
               "wide": fnum(fr["wide"]), "umatan": fnum(fr["umatan"])}
        wo = fnum(fr["tan"])
        win_odds = round(wo / 100, 1) if wo else None
    kekka[rkey] = {"finish": finish, "fuku": fuku, "pay": pay, "win_odds": win_odds}

# ワイド全3点配当
widemap = {}
try:
    w = pd.read_parquet(rf"{base}\data\wide_payouts_2016-2025.parquet")
    w = w[w["date"] == int(date)]
    for _, r in w.iterrows():
        d = {}
        for a in "123":
            d[frozenset([int(r[f"w{a}_i"]), int(r[f"w{a}_j"])])] = float(r[f"w{a}_pay"])
        widemap[str(r["race_id"])] = d
except Exception as e:
    print("wide parquet err:", e)

markmap = {}
for race in bundle["races"]:
    markmap[race["race_id"]] = {int(h["umaban"]): h for h in race["horses"]}


def mark_of(rid, uma):
    h = markmap.get(rid, {}).get(uma)
    return h.get("mark") if h else None


print(f"===== {date} Cowork 回顧 =====\n")
tot_inv = tot_ret = tot_unknown = hit_races = 0
kind_stat = {}
bet_races = [r for r in bets["bets"] if r.get("bets")]
miokuri = [r for r in bets["bets"] if not r.get("bets")]

for race in bet_races:
    rid, lbl = race["race_id"], race["race_label"]
    K = kekka.get(rid, {})
    fin, fuku, pay = K.get("finish", {}), K.get("fuku", {}), K.get("pay", {})
    wide = widemap.get(rid, {})
    top3 = set(u for u, _ in fin.values())
    res_str = " / ".join(
        f"{c}着 {fin[c][0]}{mark_of(rid, fin[c][0]) or ''} {fin[c][1]}"
        for c in (1, 2, 3) if c in fin)
    hon = [u for u, h in markmap.get(rid, {}).items() if h.get("mark") == "◎"]
    hon_pos = ">3着"
    if hon:
        sub = k[(k["rkey"] == rid) & (k["umaban"] == hon[0])]
        if len(sub) and pd.notna(sub.iloc[0]["chaku_n"]):
            hon_pos = f"{int(sub.iloc[0]['chaku_n'])}着"
    print(f"■ {lbl}  [{race.get('race_nature')}]")
    print(f"  結果: {res_str}")
    print(f"  ◎{hon[0] if hon else '?'}番 → {hon_pos}")

    rinv = rret = 0
    rhit = False
    for b in race["bets"]:
        kind, sel, amt = b["馬券種"], str(b["買い目"]), b["購入額"]
        rinv += amt
        tot_inv += amt
        nums = [int(x) for x in sel.replace("=", "-").replace("ー", "-").split("-")]
        hit = unk = False
        ret = 0
        if kind == "単勝":
            if 1 in fin and fin[1][0] == nums[0]:
                hit, ret = True, (pay.get("tansho") or 0) * amt / 100
        elif kind == "複勝":
            if nums[0] in fuku:
                hit, ret = True, fuku[nums[0]] * amt / 100
        elif kind == "馬連":
            if 1 in fin and 2 in fin and {fin[1][0], fin[2][0]} == set(nums):
                hit, ret = True, (pay.get("umaren") or 0) * amt / 100
        elif kind == "馬単":
            if 1 in fin and 2 in fin and fin[1][0] == nums[0] and fin[2][0] == nums[1]:
                hit, ret = True, (pay.get("umatan") or 0) * amt / 100
        elif kind in ("ワイド", "拡連複"):
            if len(nums) == 2 and set(nums) <= top3:
                hit = True
                fs = frozenset(nums)
                if fs in wide:
                    ret = wide[fs] * amt / 100
                elif pay.get("wide") and {fin[1][0], fin[2][0]} == set(nums):
                    ret = pay["wide"] * amt / 100
                else:
                    unk = True
                    tot_unknown += 1
        ks = kind_stat.setdefault(kind, {"n": 0, "hit": 0, "inv": 0, "ret": 0})
        ks["n"] += 1
        ks["inv"] += amt
        if hit:
            rhit = True
            ks["hit"] += 1
            if not unk:
                ks["ret"] += ret
                rret += ret
                tot_ret += ret
        mk = "".join(mark_of(rid, n) or "·" for n in nums)
        flag = "○HIT" if hit else " miss"
        tail = f" → {int(ret)}円" if (hit and not unk) else (" → 的中(配当不明)" if unk else "")
        print(f"    [{flag}] {kind} {sel}({mk}) {amt}円{tail}")
    if rhit:
        hit_races += 1
    sign = "+" if rret - rinv >= 0 else ""
    print(f"  レース収支: 投資{rinv} / 払戻{int(rret)} ({sign}{int(rret - rinv)})\n")

print("===== サマリ =====")
print(f"賭けレース {len(bet_races)} / 的中レース {hit_races} / 見送り {len(miokuri)}")
roi = tot_ret / tot_inv * 100 if tot_inv else 0
print(f"投資 {tot_inv} / 払戻 {int(tot_ret)} / 収支 {int(tot_ret - tot_inv)} / ROI {roi:.1f}%")
if tot_unknown:
    print(f"(ワイド的中で配当未取得 {tot_unknown}件 → 払戻は下限値)")
print("\n馬券種別:")
for kind, s in kind_stat.items():
    r = s["ret"] / s["inv"] * 100 if s["inv"] else 0
    print(f"  {kind}: {s['n']}点 / 的中{s['hit']} / 投資{s['inv']} / 払戻{int(s['ret'])} / ROI{r:.0f}%")

print("\n===== 見送りレース検証 =====")
for race in miokuri:
    rid, lbl = race["race_id"], race["race_label"]
    K = kekka.get(rid, {})
    fin, wo = K.get("finish", {}), K.get("win_odds")
    hon = [u for u, h in markmap.get(rid, {}).items() if h.get("mark") == "◎"]
    hon_pos = "?"
    if hon:
        sub = k[(k["rkey"] == rid) & (k["umaban"] == hon[0])]
        if len(sub) and pd.notna(sub.iloc[0]["chaku_n"]):
            hon_pos = f"{int(sub.iloc[0]['chaku_n'])}着"
    res = f"{fin[1][0]} {fin[1][1]}" if 1 in fin else "?"
    tag = "波乱" if (wo and wo >= 10) else ("順当" if (wo and wo < 5) else "中")
    print(f"  {lbl}: 1着 {res}(単勝{wo}倍/{tag}) ◎{hon[0] if hon else '?'}→{hon_pos}")
