"""
creative_strategy_v2.py
========================
creative_strategy.py の結果で +ROI がゼロだったので、最後の試み:
  - 超厳選 (odds≤1.3 かつ独走度≥X)
  - 高オッズ帯ピンポイント
  - 馬連 ◎〇▲ 3 頭 BOX (3 点)
  - 馬単 ◎→〇 1 点 (2 着固定)
  - ◎-〇-▲ ワイド 3 点 box
  - 「◎ が 2-3 着に入る」確率が高い race を選別 (複勝◎を限定)
"""
from __future__ import annotations
import io, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE = Path(__file__).parent
UNIT = 100

# 前回作った race table を再利用するか? → 3頭目(▲) の情報が無いので再構築
import warnings; warnings.filterwarnings("ignore")
from backtest_pl_ev import load_payouts, score_test, COL_RID, COL_BAN
import backtest_pl_ev as be

tag = "v4"
be.MODEL_PKL = BASE / f"models/unified_rank_{tag}.pkl"
be.CAL_PKL   = BASE / f"models/pl_calibrators_{tag}.pkl"
be.CURVE_PKL = BASE / f"data/pl_payout_curve_{tag}.pkl"

print("[setup] loading ...")
payouts = load_payouts()
te = score_test(include_valid=True)

# オッズ
odds_files = [BASE / f"data/odds/odds_{y}.csv" for y in ["20230105-20231228","20240106-20241228","20250105-20261228"]]
odds = pd.concat([pd.read_csv(f, encoding="cp932") for f in odds_files if f.exists()], ignore_index=True)
odds = odds.rename(columns={"レースID(新)":"rid","馬番":"ban","単勝オッズ":"tansho_odds"})
odds["rid_s"] = odds["rid"].astype(str).str[:16]
odds["ban"] = pd.to_numeric(odds["ban"], errors="coerce").astype("Int64")
odds["tansho_odds"] = pd.to_numeric(odds["tansho_odds"], errors="coerce")
odds_idx = odds.set_index(["rid_s","ban"])["tansho_odds"].to_dict()

# wide
wide_df = pd.read_parquet(BASE / "data/wide_payouts_2016-2025.parquet")
wide_df["rid_s"] = wide_df["race_id"].astype(str).str[:16]
wide_map = {}
for _, r in wide_df.iterrows():
    lst=[]
    for ci,cj,cp in [("w1_i","w1_j","w1_pay"),("w2_i","w2_j","w2_pay"),("w3_i","w3_j","w3_pay")]:
        try:
            i=int(r[ci]); j=int(r[cj]); p=float(r[cp])
            if p>0: lst.append((min(i,j),max(i,j),p))
        except: pass
    wide_map[r["rid_s"]] = lst

print("[build] race table ...")
rows=[]
for rid, g in te.groupby(COL_RID, sort=False):
    if len(g) < 5: continue
    rid_s = str(int(rid)) if isinstance(rid,(int,np.integer)) else str(rid)
    if rid_s not in payouts: continue
    po = payouts[rid_s]
    g = g.sort_values(COL_BAN).reset_index(drop=True)
    ban = g[COL_BAN].astype(int).values
    b2i = {int(b):i for i,b in enumerate(ban)}
    wi = b2i.get(po["win"]); pi_ = b2i.get(po["plc"])
    if wi is None or pi_ is None: continue
    order = np.argsort(-g["_score"].values)
    hon_i = int(order[0]); tai_i = int(order[1]); san_i = int(order[2])
    hon_b = int(ban[hon_i]); tai_b = int(ban[tai_i]); san_b = int(ban[san_i])
    hon_o = odds_idx.get((rid_s,hon_b), np.nan)
    tai_o = odds_idx.get((rid_s,tai_b), np.nan)
    san_o = odds_idx.get((rid_s,san_b), np.nan)
    if pd.isna(hon_o): continue

    tansho_pay = float(po.get("tansho") or 0.0)
    fuku_win = float(po.get("fuku_win") or 0.0)
    fuku_plc = float(po.get("fuku_plc") or 0.0)
    fuku_sho = float(po.get("fuku_sho") or 0.0)
    umaren = float(po.get("umaren") or 0.0)
    umatan = float(po.get("umatan") or 0.0)

    sho_i = b2i.get(po.get("sho")) if po.get("sho") is not None else None

    # 馬連ペア配当が分からないので、◎〇、◎▲、〇▲ のどれが当たった場合も umaren を使う
    # (実際は各組合せで配当が異なるが、◎〇▲3 頭 BOX の当たりは「1-2 着が 3 頭の中」なら umaren)
    hit_fuku_hon = hon_i in {wi,pi_,sho_i}
    fuku_ret_hon = fuku_win if hon_i==wi else (fuku_plc if hon_i==pi_ else (fuku_sho if sho_i is not None and hon_i==sho_i else 0.0))

    # 馬単 ◎→〇
    hit_umatan_ho = (hon_i==wi and tai_i==pi_)

    # ワイド ◎-〇, ◎-▲, 〇-▲ 配当 (lookup)
    def wide_pay(bi, bj):
        key = (min(bi,bj), max(bi,bj))
        for (a,b,p) in wide_map.get(rid_s, []):
            if (a,b)==key: return p
        return 0.0
    w_ho = wide_pay(hon_b, tai_b)
    w_hs = wide_pay(hon_b, san_b)
    w_os = wide_pay(tai_b, san_b)

    # 馬連 ◎〇▲ 3頭 BOX = 3 点
    # 的中 ⇔ {wi, pi_} ⊂ {hon_i, tai_i, san_i}
    top3 = {hon_i, tai_i, san_i}
    hit_umaren_box = {wi, pi_}.issubset(top3)
    ret_umaren_box = umaren if hit_umaren_box else 0.0

    # ワイド ◎〇▲ 3 頭 BOX (3点)
    ret_wide_box = 0.0
    for pair in [(hon_b,tai_b),(hon_b,san_b),(tai_b,san_b)]:
        ret_wide_box += wide_pay(*pair)

    year = int(g["year"].iloc[0])

    rows.append({
        "rid_s": rid_s, "year": year,
        "hon_odds": hon_o, "tai_odds": tai_o, "san_odds": san_o,
        "hit_fuku_hon": int(hit_fuku_hon), "ret_fuku_hon": fuku_ret_hon,
        "hit_umatan_ho": int(hit_umatan_ho), "ret_umatan_ho": umatan if hit_umatan_ho else 0.0,
        "hit_umaren_box": int(hit_umaren_box), "ret_umaren_box": ret_umaren_box,
        "ret_wide_box": ret_wide_box,
        "w_ho": w_ho,
    })

df = pd.DataFrame(rows)
print(f"  races: {len(df):,}")

def evl(sub, stake_per, retcol, name):
    R=len(sub)
    if R==0: return None
    stake = R * stake_per
    ret = sub[retcol].sum()
    roi = ret / stake
    hits = (sub[retcol]>0).sum()
    return dict(name=name, R=R, hits=hits, hit_pct=hits/R*100, stake=stake, ret=int(ret), roi_pct=roi*100, pl=int(ret-stake))

def show(results, header):
    print(f"\n=== {header} ===")
    print(f"  {'strategy':42s} {'R':>5s} {'hit%':>6s} {'ROI%':>7s} {'P&L':>10s}")
    for r in results:
        if r is None: continue
        flag = "✅" if r["roi_pct"]>=100 and r["R"]>=300 else ("🔶" if r["roi_pct"]>=95 and r["R"]>=300 else "  ")
        print(f"  {flag} {r['name']:40s} {r['R']:>5d} {r['hit_pct']:>5.1f}% {r['roi_pct']:>6.2f}% {r['pl']:>+10,d}")


for label, yset in [("3年", {2023,2024,2025}), ("真OOS", {2024,2025})]:
    base = df[df["year"].isin(yset)].copy()
    print(f"\n{'='*70}\n【{label}】 {len(base):,}\n{'='*70}")

    # --- I: 超厳選 複勝◎ ---
    res=[]
    for thr in [1.1, 1.2, 1.3, 1.4, 1.5]:
        res.append(evl(base[base["hon_odds"]<=thr], UNIT, "ret_fuku_hon", f"I 複勝◎ (odds≤{thr})"))
    show(res, f"{label} I: 超厳選 複勝◎")

    # --- J: 複勝◎ + 独走度 ---
    res=[]
    for thr in [1.5, 2.0, 2.5, 3.0]:
        for gap in [0.3, 0.5, 1.0, 2.0]:
            sub = base[(base["hon_odds"]<=thr) & ((base["tai_odds"]-base["hon_odds"])>=gap)]
            res.append(evl(sub, UNIT, "ret_fuku_hon", f"J 複勝◎ (odds≤{thr} gap≥{gap})"))
    show(res, f"{label} J: 複勝◎ + 独走度複合")

    # --- K: 馬連 ◎〇▲ 3頭 BOX (3点) ---
    res=[]
    for thr in [1.5, 2.0, 2.5, 3.0, 5.0, 999]:
        res.append(evl(base[base["hon_odds"]<=thr], UNIT*3, "ret_umaren_box", f"K 馬連◎〇▲BOX (odds≤{thr})"))
    show(res, f"{label} K: 馬連 ◎〇▲ 3頭 BOX")

    # --- L: ワイド ◎〇▲ 3頭 BOX (3点) ---
    res=[]
    for thr in [1.5, 2.0, 2.5, 3.0, 5.0, 999]:
        res.append(evl(base[base["hon_odds"]<=thr], UNIT*3, "ret_wide_box", f"L ワイド◎〇▲BOX (odds≤{thr})"))
    show(res, f"{label} L: ワイド ◎〇▲ 3頭 BOX")

    # --- M: 馬単 ◎→〇 1点 ---
    res=[]
    for thr in [1.5, 2.0, 2.5, 3.0, 5.0, 999]:
        res.append(evl(base[base["hon_odds"]<=thr], UNIT, "ret_umatan_ho", f"M 馬単◎→〇 (odds≤{thr})"))
    show(res, f"{label} M: 馬単 ◎→〇 1点")

    # --- N: 人気薄◎ 複勝 (穴狙い) ---
    res=[]
    for lo,hi in [(5,8),(8,15),(15,30),(30,100),(100,999)]:
        sub = base[(base["hon_odds"]>=lo)&(base["hon_odds"]<hi)]
        res.append(evl(sub, UNIT, "ret_fuku_hon", f"N 複勝◎ ({lo}≤odds<{hi})"))
    show(res, f"{label} N: 穴複勝◎")

    # --- O: ◎ が 1-2-3 着 いずれか かつ オッズ複勝払戻が高い帯 ---
    res=[]
    for lo,hi in [(3,5),(5,8),(8,15),(15,30)]:
        sub = base[(base["hon_odds"]>=lo)&(base["hon_odds"]<hi)]
        res.append(evl(sub, UNIT, "ret_fuku_hon", f"O 複勝◎ ({lo}≤odds<{hi})"))
    show(res, f"{label} O: 中オッズ帯 複勝◎")

print("\n[done]")
