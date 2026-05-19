"""
プラン最適買い目 統合シミュレーション
シミュレーション結果に基づく推奨ポートフォリオ:
  HAHO(92R)     : 三連複◎○▲ 1点
  HALO(231R)    : 三連複BOX4頭（◎○▲△）4点
  LALO_CQC(2769R): 複勝◎ 1点
"""
import itertools
import pandas as pd, json
from pathlib import Path

BASE      = Path("E:/PyCaLiAI")
BT_CSV    = BASE / "reports/backtest_results_2024.csv"
MARKS_CSV = BASE / "reports/backtest_marks_2024.csv"
KEKKA_CSV = BASE / "data/kekka_20160105_20251228_v2.csv"
STRAT_JSON= BASE / "data/strategy_weights.json"
BUDGET    = 10_000

CLASS_NORMALIZE = {
    "新馬":"新馬","未勝利":"未勝利","1勝":"1勝","500万":"1勝",
    "2勝":"2勝","1000万":"2勝","3勝":"3勝","1600万":"3勝",
    "OP(L)":"OP(L)","Ｇ１":"Ｇ１","Ｇ２":"Ｇ２","Ｇ３":"Ｇ３",
}

# ── 1. プラン別 (place,cls) キー ──────────────────────────────────────
with open(STRAT_JSON, encoding="utf-8") as f:
    sw = json.load(f)

plan_keys = {"HAHO": set(), "HALO": set(), "LALO_CQC": set()}
for place, classes in sw.items():
    for cls, bets in classes.items():
        bet_names = list(bets.keys())
        has_sf  = any("三連複" in b for b in bet_names)
        has_uma = any("馬連"   in b for b in bet_names)
        if has_sf and has_uma:   plan_keys["HAHO"].add((place, cls))
        elif has_sf:             plan_keys["HALO"].add((place, cls))
        else:                    plan_keys["LALO_CQC"].add((place, cls))

# ── 2. kekka 読み込み ─────────────────────────────────────────────────
print("kekka読み込み中...")
kk = pd.read_csv(KEKKA_CSV, encoding="cp932", dtype=str)
kc = kk.columns.tolist()
kk["race16"] = kk[kc[7]].astype(str).str[:16]
kk["ban"]    = pd.to_numeric(kk[kc[4]], errors="coerce")
kk["jyun"]   = pd.to_numeric(kk[kc[6]], errors="coerce")
for i in [8,9,11,12,13,14]:
    kk[kc[i]] = pd.to_numeric(kk[kc[i]], errors="coerce")
kk_dict = {r16: grp for r16, grp in kk.groupby("race16")}

def top_ban(race16, jyun):
    sub = kk_dict.get(race16)
    if sub is None: return None
    r = sub[sub["jyun"] == jyun]["ban"].dropna()
    return int(r.iloc[0]) if not r.empty else None

def get_pay(race16, combo, btype):
    sub = kk_dict.get(race16)
    if sub is None: return 0
    if btype == "複勝":
        r = sub[sub["ban"] == combo[0]]
        if r.empty: return 0
        v = r.iloc[0][kc[9]]; return float(v) if pd.notna(v) and v > 0 else 0
    elif btype == "三連複":
        b1,b2,b3 = sorted(combo)
        w1=top_ban(race16,1); w2=top_ban(race16,2); w3=top_ban(race16,3)
        if None in (w1,w2,w3): return 0
        if sorted([w1,w2,w3]) != [b1,b2,b3]: return 0
        r = sub[sub["jyun"]==1]
        v = r.iloc[0][kc[13]] if not r.empty else None
        return float(v) if pd.notna(v) and v > 0 else 0
    return 0

# ── 3. BT読み込み → plan マッピング ──────────────────────────────────
bt = pd.read_csv(BT_CSV, encoding="utf-8-sig")
c  = bt.columns.tolist()
race_col, place_col, cls_col = c[0], c[2], c[5]
bt["race16"]  = bt[race_col].astype(str).str[:16]
bt["cls_norm"] = bt[cls_col].map(CLASS_NORMALIZE).fillna(bt[cls_col])

race_to_r16 = bt.drop_duplicates(race_col).set_index(race_col)["race16"].to_dict()
race_to_plan = {}
for _, row in bt.drop_duplicates(race_col).iterrows():
    key = (row[place_col], row["cls_norm"])
    for plan, keys in plan_keys.items():
        if key in keys:
            race_to_plan[row[race_col]] = plan
            break

# ── 4. 印読み込み ─────────────────────────────────────────────────────
marks_df = pd.read_csv(MARKS_CSV, encoding="utf-8-sig")
all_marks = {}
for _, row in marks_df.iterrows():
    rid = row["race_id"]
    m = {}
    for key in ["hon","taikou","sabo","delta","batsu"]:
        v = row.get(key)
        if pd.notna(v): m[key] = int(v)
    all_marks[rid] = m

for plan in ["HAHO","HALO","LALO_CQC"]:
    n = sum(1 for rid in all_marks if race_to_plan.get(rid) == plan)
    print(f"{plan}: {n}R")

# ── 5. 最適買い目定義 ─────────────────────────────────────────────────
def make_bets(plan, m):
    """プランごとの推奨買い目を返す: [(btype, combo), ...]"""
    h1 = m.get("hon")
    h2 = m.get("taikou")
    h3 = m.get("sabo")
    h4 = m.get("delta")

    if plan == "HAHO":
        # 三連複◎○▲ 1点（バックテストで169.1%）
        if h1 and h2 and h3:
            return [("三連複", sorted([h1,h2,h3]))]

    elif plan == "HALO":
        # 三連複BOX4頭（◎○▲△）4点 / △なし時は1点
        if h1 and h2 and h3:
            if h4:
                heads = sorted([h1,h2,h3,h4])
                return [("三連複", list(c)) for c in itertools.combinations(heads, 3)]
            else:
                return [("三連複", sorted([h1,h2,h3]))]

    elif plan == "LALO_CQC":
        # 複勝◎ 1点（バックテストで85.6%）
        if h1:
            return [("複勝", [h1])]

    return []

# ── 6. シミュレーション ──────────────────────────────────────────────
print("\nシミュレーション中...")

plan_stats = {p: {"tb":0,"tr":0,"hits":0,"races":0} for p in ["HAHO","HALO","LALO_CQC"]}

for race_id, m in all_marks.items():
    plan = race_to_plan.get(race_id)
    if plan is None: continue
    r16 = race_to_r16.get(race_id, "")
    if not r16: continue

    bets = make_bets(plan, m)
    if not bets: continue

    st = plan_stats[plan]
    n   = len(bets)
    per = BUDGET // n
    race_hit = False
    for btype, combo in bets:
        pay = get_pay(r16, combo, btype)
        st["tb"] += per
        if pay > 0:
            st["tr"] += per * pay / 100
            race_hit = True
    st["races"] += 1
    if race_hit: st["hits"] += 1

# ── 7. 出力 ──────────────────────────────────────────────────────────
plan_labels = {
    "HAHO":     "三連複◎○▲ 1点",
    "HALO":     "三連複BOX4頭(◎○▲△)",
    "LALO_CQC": "複勝◎ 1点",
}

print(f"\n{'='*72}")
print(f"プラン最適買い目 統合シミュレーション（backtest 2024）")
print(f"{'='*72}")
print(f"  {'プラン':<12} {'買い目':<22} {'R':>5} {'的中率':>7} {'平均払戻':>10} {'ROI':>7}")
print(f"  {'-'*68}")

total_tb = total_tr = total_races = 0
for plan in ["HAHO","HALO","LALO_CQC"]:
    st = plan_stats[plan]
    tb,tr,hits,races = st["tb"],st["tr"],st["hits"],st["races"]
    if races == 0: continue
    roi   = tr / tb * 100 if tb else 0
    hrate = hits / races * 100
    avg   = tr / hits if hits else 0
    star  = "★" if roi >= 100 else ("  " if roi >= 80 else "NG")
    print(f"  {star} {plan:<12} {plan_labels[plan]:<22} {races:>5}R {hrate:>6.1f}% {avg:>10,.0f}円 {roi:>7.1f}%")
    total_tb += tb; total_tr += tr; total_races += races

# 合計
if total_tb:
    total_roi = total_tr / total_tb * 100
    print(f"  {'-'*68}")
    total_hits = sum(st["hits"] for st in plan_stats.values())
    total_hrate = total_hits / total_races * 100
    total_avg = total_tr / total_hits if total_hits else 0
    star = "★" if total_roi >= 100 else ("  " if total_roi >= 80 else "NG")
    print(f"  {star} {'合計/加重平均':<12} {'':<22} {total_races:>5}R {total_hrate:>6.1f}% {total_avg:>10,.0f}円 {total_roi:>7.1f}%")

print(f"\n★=100%超  空白=80%以上  NG=80%未満")
print(f"HALO条件: {plan_keys['HALO']}")
print(f"HAHO条件: {plan_keys['HAHO']}")
