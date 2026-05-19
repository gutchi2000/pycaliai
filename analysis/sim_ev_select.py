"""
EV選択シミュレーション（v3: 全頭版kekka使用）
・kekka_v2: 全着順・全馬収録（旧版は1〜3着のみ → 的中率水増しバグ）
・複勝row[0]=◎, row[1]=○ の正しいマッピング
・▲は馬連row[1]から復元（◎-▲の◎を除いた馬番）
・EV = prob × 事前単勝オッズ（非1着馬は (X.X) 形式、1着馬は実払戻/100）
"""
import pandas as pd
from pathlib import Path

BASE   = Path("E:/PyCaLiAI")
BT     = pd.read_csv(BASE/"reports/backtest_results_2024.csv", encoding="utf-8-sig")
# v2: 全頭版（15列: 日付,場所,回,枠番,馬番,馬名,確定着順,レースID(新),単勝払戻,...）
KK     = pd.read_csv(BASE/"data/kekka_20160105_20251228_v2.csv", encoding="cp932", dtype=str)
BUDGET = 10_000

c  = BT.columns.tolist()
race_col, bet_col, mark_col, prob_col = c[0], c[6], c[7], c[8]

kc = KK.columns.tolist()
# v2列インデックス: [0]日付 [1]場所 [2]回 [3]枠番 [4]馬番 [5]馬名 [6]確定着順
#                   [7]レースID(新) [8]単勝払戻 [9]複勝払戻 ...

# ── kekka: race16+馬番 → {odds, winner, payout} ──
KK["race16"] = KK[kc[7]].astype(str).str[:16]   # ID列はインデックス7
KK["ban"]    = pd.to_numeric(KK[kc[4]], errors="coerce")   # 馬番はインデックス4
KK["jyun"]   = pd.to_numeric(KK[kc[6]], errors="coerce")  # 着順はインデックス6

def parse_odds_str(v):
    s = str(v).strip()
    if s.lower() in ("nan","none",""): return None
    if s.startswith("(") and s.endswith(")"):
        try: return float(s[1:-1])
        except: return None
    try:
        fv = float(s)
        return fv / 100.0 if fv > 0 else None
    except: return None

kk_race = {}
for _, row in KK[KK["ban"].notna()].iterrows():
    r16 = row["race16"]; ban = int(row["ban"])
    odds = parse_odds_str(row[kc[8]])  # 単勝払戻はインデックス8
    if odds is None: continue
    is_win = (row["jyun"] == 1)
    kk_race.setdefault(r16, {})[ban] = {
        "odds": odds, "winner": is_win,
        "payout": odds * BUDGET if is_win else 0,
    }

print(f"kekkaオッズ辞書: {len(kk_race):,}レース")

# ── バックテストから ◎○▲ の馬番・確率を正しく復元 ──
BT["race16"] = BT[race_col].astype(str).str[:16]
marks_per_race = {}

for race16, grp in BT.groupby("race16"):
    m = {}
    # ◎ = 単勝行
    tan = grp[grp[bet_col] == "単勝"]
    if not tan.empty:
        r = tan.iloc[0]
        m["hon"] = {"ban": int(r[mark_col]), "prob": float(r[prob_col])}

    # 複勝: row[0]=◎, row[1]=○
    fuku = grp[grp[bet_col] == "複勝"].reset_index(drop=True)
    if len(fuku) >= 2:
        r = fuku.iloc[1]  # ← row[1]が○（row[0]は◎と同じ馬）
        m["taikou"] = {"ban": int(r[mark_col]), "prob": float(r[prob_col])}

    # ▲ = 馬連row[1]（◎-▲）から◎を除いた馬番
    uma = grp[grp[bet_col] == "馬連"].reset_index(drop=True)
    if len(uma) >= 2 and "hon" in m:
        pair = str(uma.iloc[1][mark_col]).split("-")
        try:
            bans = [int(x) for x in pair]
            sabo_ban = [b for b in bans if b != m["hon"]["ban"]]
            if sabo_ban:
                # ▲のprobを推定: hon_prob * 0.75 として近似（honより低い確率）
                m["sabo"] = {"ban": sabo_ban[0],
                             "prob": m["hon"]["prob"] * 0.75}
        except: pass

    marks_per_race[race16] = m

full3 = sum(1 for m in marks_per_race.values() if len(m) >= 3)
print(f"マーク復元: {len(marks_per_race):,}R  ◎○▲揃い: {full3:,}R")

# sanity check
for r16, m in list(marks_per_race.items())[:3]:
    h = kk_race.get(r16, {})
    winners = [ban for ban, v in h.items() if v["winner"]]
    print(f"  ◎={m.get('hon',{}).get('ban')} ○={m.get('taikou',{}).get('ban')} "
          f"▲={m.get('sabo',{}).get('ban')}  実1着={winners}")

# ── EV最大選択: prob × 実オッズ 最大の馬を選ぶ ──
def ev_best(m, horse_data):
    best_key, best_ev = None, -1.0
    for k in ["hon","taikou","sabo"]:
        if k not in m: continue
        kk = horse_data.get(m[k]["ban"])
        if kk is None: continue
        ev = m[k]["prob"] * kk["odds"]
        if ev > best_ev:
            best_ev, best_key = ev, k
    return best_key

# ── シミュレーション ──
def simulate(name, selector):
    tot_bet = tot_ret = hits = races = 0
    for race16, m in marks_per_race.items():
        horse_data = kk_race.get(race16, {})
        key = selector(m, horse_data)
        if not key or key not in m: continue
        info = m[key]
        kk = horse_data.get(info["ban"])
        if kk is None: continue
        tot_bet += BUDGET; races += 1
        if kk["winner"]:
            tot_ret += kk["payout"]; hits += 1
    roi   = tot_ret / tot_bet * 100 if tot_bet else 0
    hrate = hits / races * 100 if races else 0
    avg   = tot_ret / hits if hits else 0
    return {"name":name,"R":races,"hits":hits,"hrate":hrate,"avg":avg,"roi":roi}

results = [
    simulate("単勝◎（現行）",     lambda m,h: "hon"),
    simulate("単勝○",             lambda m,h: "taikou"),
    simulate("単勝▲(近似)",       lambda m,h: "sabo"),
    simulate("単勝EV最大(◎○▲)",  ev_best),
]

print("\n" + "="*70)
print(f"{'買い目':23} {'R':>5} {'的中率':>7} {'平均払戻':>10} {'ROI':>7}")
print("="*70)
for r in results:
    mark = "★" if r["roi"] >= 80 else "  "
    print(f" {mark} {r['name']:<22} {r['R']:>5}R  "
          f"{r['hrate']:>5.1f}%  {r['avg']:>9,.0f}円  {r['roi']:>6.1f}%")

# 内訳
print("\n--- EV最大選択 内訳 ---")
cnt = {"hon":0,"taikou":0,"sabo":0,"skip":0}
for race16, m in marks_per_race.items():
    h = kk_race.get(race16, {})
    k = ev_best(m, h)
    cnt[k if k else "skip"] += 1
total = sum(cnt.values())
for k,label in [("hon","◎"),("taikou","○"),("sabo","▲"),("skip","skip")]:
    print(f"  {label}: {cnt[k]:,}R ({cnt[k]/total*100:.1f}%)")
