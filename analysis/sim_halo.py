"""
HALO戦略対象レース限定 全買い目パターンシミュレーション
HALO = strategy_weights.json で三連複が入っている 場所×クラス
"""
import pandas as pd, json
from pathlib import Path

BASE      = Path("E:/PyCaLiAI")
BT_CSV    = BASE / "reports/backtest_results_2024.csv"
MARKS_CSV = BASE / "reports/backtest_marks_2024.csv"
KEKKA_CSV = BASE / "data/kekka_20160105_20251228_v2.csv"
STRAT_JSON= BASE / "data/strategy_weights.json"
BUDGET    = 10_000

# ── 1. HALO対象の場所×クラスを抽出 ───────────────────────────────────
with open(STRAT_JSON, encoding="utf-8") as f:
    sw = json.load(f)

CLASS_NORMALIZE = {
    "新馬":"新馬","未勝利":"未勝利","1勝":"1勝","500万":"1勝",
    "2勝":"2勝","1000万":"2勝","3勝":"3勝","1600万":"3勝",
    "OP(L)":"OP(L)","Ｇ１":"Ｇ１","Ｇ２":"Ｇ２","Ｇ３":"Ｇ３",
}

halo_keys = set()
for place, classes in sw.items():
    for cls, bets in classes.items():
        if any("三連複" in b for b in bets.keys()):
            halo_keys.add((place, cls))
print(f"HALO対象条件: {halo_keys}")

# ── 2. kekka読み込み ──────────────────────────────────────────────────
print("kekka読み込み中...")
kk = pd.read_csv(KEKKA_CSV, encoding="cp932", dtype=str)
kc = kk.columns.tolist()
kk["race16"] = kk[kc[7]].astype(str).str[:16]
kk["ban"]    = pd.to_numeric(kk[kc[4]], errors="coerce")
kk["jyun"]   = pd.to_numeric(kk[kc[6]], errors="coerce")
for i in [8,9,11,12,13,14]:
    kk[kc[i]] = pd.to_numeric(kk[kc[i]], errors="coerce")
kk_dict = {r16: grp for r16, grp in kk.groupby("race16")}
print(f"  レース数: {len(kk_dict):,}")

def top_ban(race16, jyun):
    sub = kk_dict.get(race16)
    if sub is None: return None
    r = sub[sub["jyun"] == jyun]["ban"].dropna()
    return int(r.iloc[0]) if not r.empty else None

def get_pay(race16, combo, btype):
    sub = kk_dict.get(race16)
    if sub is None: return 0
    if btype == "単勝":
        r = sub[(sub["ban"] == combo[0]) & (sub["jyun"] == 1)]
        if r.empty: return 0
        v = r.iloc[0][kc[8]]; return float(v) if pd.notna(v) and v > 0 else 0
    elif btype == "複勝":
        r = sub[sub["ban"] == combo[0]]
        if r.empty: return 0
        v = r.iloc[0][kc[9]]; return float(v) if pd.notna(v) and v > 0 else 0
    elif btype == "馬連":
        b1,b2 = sorted(combo)
        w1=top_ban(race16,1); w2=top_ban(race16,2)
        if w1 is None or w2 is None: return 0
        if sorted([w1,w2]) != [b1,b2]: return 0
        r = sub[sub["jyun"]==1]
        if r.empty: return 0
        v = r.iloc[0][kc[11]]; return float(v) if pd.notna(v) and v > 0 else 0
    elif btype == "馬単":
        w1=top_ban(race16,1); w2=top_ban(race16,2)
        if w1!=combo[0] or w2!=combo[1]: return 0
        r = sub[sub["jyun"]==1]
        if r.empty: return 0
        v = r.iloc[0][kc[12]]; return float(v) if pd.notna(v) and v > 0 else 0
    elif btype == "三連複":
        b1,b2,b3 = sorted(combo)
        w1=top_ban(race16,1); w2=top_ban(race16,2); w3=top_ban(race16,3)
        if None in (w1,w2,w3): return 0
        if sorted([w1,w2,w3]) != [b1,b2,b3]: return 0
        r = sub[sub["jyun"]==1]
        if r.empty: return 0
        v = r.iloc[0][kc[13]]; return float(v) if pd.notna(v) and v > 0 else 0
    elif btype == "三連単":
        w1=top_ban(race16,1); w2=top_ban(race16,2); w3=top_ban(race16,3)
        if w1!=combo[0] or w2!=combo[1] or w3!=combo[2]: return 0
        r = sub[sub["jyun"]==1]
        if r.empty: return 0
        v = r.iloc[0][kc[14]]; return float(v) if pd.notna(v) and v > 0 else 0
    return 0

# ── 3. BT読み込み → HALO対象レース絞り込み ────────────────────────────
print("BT読み込み・HALO絞り込み中...")
bt = pd.read_csv(BT_CSV, encoding="utf-8-sig")
c  = bt.columns.tolist()
race_col, place_col, cls_col = c[0], c[2], c[5]

bt["race16"] = bt[race_col].astype(str).str[:16]
bt["cls_norm"] = bt[cls_col].map(CLASS_NORMALIZE).fillna(bt[cls_col])

# HALO対象race_idを抽出
halo_race_ids = set()
for _, row in bt.drop_duplicates(race_col).iterrows():
    if (row[place_col], row["cls_norm"]) in halo_keys:
        halo_race_ids.add(row[race_col])
print(f"HALO対象レース: {len(halo_race_ids):,}R")

race_to_r16 = bt.drop_duplicates(race_col).set_index(race_col)["race16"].to_dict()

# ── 4. 印読み込み（HALO絞り込み）────────────────────────────────────
marks_df = pd.read_csv(MARKS_CSV, encoding="utf-8-sig")
marks = {}
for _, row in marks_df.iterrows():
    rid = row["race_id"]
    if rid not in halo_race_ids: continue
    m = {}
    for key in ["hon","taikou","sabo","delta","batsu"]:
        v = row.get(key)
        if pd.notna(v): m[key] = int(v)
    marks[rid] = m

full = sum(1 for m in marks.values() if all(k in m for k in ["hon","taikou","sabo"]))
print(f"◎○▲揃い: {full:,} / {len(marks):,} R")

# ── 5. パターン定義 ──────────────────────────────────────────────────
def h(m): return m.get("hon")
def t(m): return m.get("taikou")
def s(m): return m.get("sabo")
def d(m): return m.get("delta")
def has(*keys):
    def _f(m): return all(k in m for k in keys)
    return _f

patterns = {}
# 単勝
patterns["単勝◎"]         = ("単勝", lambda m: [[h(m)]] if has("hon")(m) else [])
patterns["単勝○"]         = ("単勝", lambda m: [[t(m)]] if has("taikou")(m) else [])
patterns["単勝▲"]         = ("単勝", lambda m: [[s(m)]] if has("sabo")(m) else [])
patterns["単勝△"]         = ("単勝", lambda m: [[d(m)]] if has("delta")(m) else [])
patterns["単勝○+▲ 2点"]   = ("単勝", lambda m: [[t(m)],[s(m)]] if has("taikou","sabo")(m) else [])
# 複勝
patterns["複勝◎"]         = ("複勝", lambda m: [[h(m)]] if has("hon")(m) else [])
patterns["複勝○"]         = ("複勝", lambda m: [[t(m)]] if has("taikou")(m) else [])
patterns["複勝▲"]         = ("複勝", lambda m: [[s(m)]] if has("sabo")(m) else [])
patterns["複勝○+▲ 2点"]   = ("複勝", lambda m: [[t(m)],[s(m)]] if has("taikou","sabo")(m) else [])
# 馬連
patterns["馬連◎-○"]       = ("馬連", lambda m: [[h(m),t(m)]] if has("hon","taikou")(m) else [])
patterns["馬連◎-▲"]       = ("馬連", lambda m: [[h(m),s(m)]] if has("hon","sabo")(m) else [])
patterns["馬連◎軸2点"]    = ("馬連", lambda m: [[h(m),t(m)],[h(m),s(m)]] if has("hon","taikou","sabo")(m) else [])
patterns["馬連BOX3頭"]    = ("馬連", lambda m: [[h(m),t(m)],[h(m),s(m)],[t(m),s(m)]] if has("hon","taikou","sabo")(m) else [])
# 馬単
patterns["馬単◎→▲"]       = ("馬単", lambda m: [[h(m),s(m)]] if has("hon","sabo")(m) else [])
patterns["馬単◎→○"]       = ("馬単", lambda m: [[h(m),t(m)]] if has("hon","taikou")(m) else [])
patterns["馬単◎1着流し2点"]= ("馬単", lambda m: [[h(m),t(m)],[h(m),s(m)]] if has("hon","taikou","sabo")(m) else [])
# 三連複
patterns["三連複◎○▲(現行)"] = ("三連複", lambda m: [[h(m),t(m),s(m)]] if has("hon","taikou","sabo")(m) else [])
patterns["三連複◎○△"]      = ("三連複", lambda m: [[h(m),t(m),d(m)]] if has("hon","taikou","delta")(m) else [])
patterns["三連複◎▲△"]      = ("三連複", lambda m: [[h(m),s(m),d(m)]] if has("hon","sabo","delta")(m) else [])
patterns["三連複◎軸3点(○▲△)"]= ("三連複", lambda m: [
    [h(m),t(m),s(m)],[h(m),t(m),d(m)],[h(m),s(m),d(m)]
] if has("hon","taikou","sabo","delta")(m) else [])
patterns["三連複◎○軸2点(▲△)"]= ("三連複", lambda m: [
    [h(m),t(m),s(m)],[h(m),t(m),d(m)]
] if has("hon","taikou","sabo","delta")(m) else [])
patterns["三連複BOX4頭"]   = ("三連複", lambda m: [
    [h(m),t(m),s(m)],[h(m),t(m),d(m)],[h(m),s(m),d(m)],[t(m),s(m),d(m)]
] if has("hon","taikou","sabo","delta")(m) else [])
# 三連単
patterns["三連単◎1着固定2点"] = ("三連単", lambda m: [[h(m),t(m),s(m)],[h(m),s(m),t(m)]] if has("hon","taikou","sabo")(m) else [])
patterns["三連単BOX6点"]      = ("三連単", lambda m: [
    [h(m),t(m),s(m)],[h(m),s(m),t(m)],[t(m),h(m),s(m)],
    [t(m),s(m),h(m)],[s(m),h(m),t(m)],[s(m),t(m),h(m)]
] if has("hon","taikou","sabo")(m) else [])

# ── 6. シミュレーション ──────────────────────────────────────────────
print("シミュレーション中...")
results = {}
for name, (btype, fn) in patterns.items():
    tb = tr = hits = races = 0
    for race_id, m in marks.items():
        combos = fn(m)
        if not combos: continue
        r16 = race_to_r16.get(race_id, "")
        if not r16: continue
        n   = len(combos)
        per = BUDGET // n
        race_hit = False
        for combo in combos:
            pay = get_pay(r16, combo, btype)
            tb += per
            if pay > 0:
                tr += per * pay / 100
                race_hit = True
        races += 1
        if race_hit: hits += 1
    roi   = tr / tb * 100 if tb else 0
    hrate = hits / races * 100 if races else 0
    avg   = tr / hits if hits else 0
    results[name] = {"btype":btype,"R":races,"hits":hits,"hrate":hrate,"avg":avg,"roi":roi}

# ── 7. 出力 ──────────────────────────────────────────────────────────
print()
print(f"{'='*68}")
print(f"HALO対象レース限定シミュレーション（{len(marks):,}R）")
print(f"{'='*68}")
for btype in ["単勝","複勝","馬連","馬単","三連複","三連単"]:
    items = [(n,r) for n,r in results.items() if r["btype"]==btype]
    if not items: continue
    print(f"\n  --- {btype} ---")
    print(f"  {'買い目':<22} {'R':>5} {'的中率':>7} {'平均払戻':>10} {'ROI':>7}")
    print(f"  {'-'*60}")
    for name, r in sorted(items, key=lambda x: x[1]["roi"], reverse=True):
        star = "★" if r["roi"] >= 100 else ("  " if r["roi"] >= 80 else "NG")
        cur  = " ←現行" if "現行" in name else ""
        print(f"  {star} {name:<22} {r['R']:>5}R {r['hrate']:>6.1f}% {r['avg']:>10,.0f}円 {r['roi']:>7.1f}%{cur}")
print()
print("★=100%超(黒字)  空白=80%以上  NG=80%未満")
print(f"HALO条件: {halo_keys}")
