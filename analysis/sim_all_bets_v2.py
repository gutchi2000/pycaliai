"""
全買い目パターン完全シミュレーション（v2 全頭kekka使用）
単勝/複勝/馬連/馬単/三連複/三連単 の流し・フォーメーション・ボックス全パターン
"""
import pandas as pd
from pathlib import Path

BASE   = Path("E:/PyCaLiAI")
BT_CSV = BASE / "reports/backtest_results_2024.csv"
# v2: 全頭版 15列 [0]日付 [1]場所 [2]回 [3]枠番 [4]馬番 [5]馬名
#                [6]確定着順 [7]レースID(新) [8]単勝払戻 [9]複勝払戻
#                [10]枠連払戻 [11]馬連払戻 [12]馬単払戻 [13]三連複払戻 [14]三連単払戻
KEKKA_CSV = BASE / "data/kekka_20160105_20251228_v2.csv"
BUDGET = 10_000

# ── 1. kekka 読み込み ───────────────────────────────────────────────
print("kekka読み込み中...")
kk = pd.read_csv(KEKKA_CSV, encoding="cp932", dtype=str)
kc = kk.columns.tolist()

kk["race16"] = kk[kc[7]].astype(str).str[:16]
kk["ban"]    = pd.to_numeric(kk[kc[4]], errors="coerce")
kk["jyun"]   = pd.to_numeric(kk[kc[6]], errors="coerce")
for i in [8,9,11,12,13,14]:
    kk[kc[i]] = pd.to_numeric(kk[kc[i]], errors="coerce")

kk_dict = {}
for race16, grp in kk.groupby("race16"):
    kk_dict[race16] = grp
print(f"  レース数: {len(kk_dict):,}  行数: {len(kk):,}")

def top_ban(race16, jyun):
    """指定着順の馬番を返す。なければNone"""
    sub = kk_dict.get(race16)
    if sub is None: return None
    r = sub[sub["jyun"] == jyun]["ban"].dropna()
    return int(r.iloc[0]) if not r.empty else None

def get_pay(race16, combo, btype):
    """払戻(円/100円票)。外れ=0  combo=馬番リスト（順序依存: 馬単・三連単は [1着,2着,...]）"""
    sub = kk_dict.get(race16)
    if sub is None: return 0

    if btype == "単勝":
        ban = combo[0]
        r = sub[(sub["ban"] == ban) & (sub["jyun"] == 1)]
        if r.empty: return 0
        v = r.iloc[0][kc[8]]; return float(v) if pd.notna(v) and v > 0 else 0

    elif btype == "複勝":
        ban = combo[0]
        r = sub[sub["ban"] == ban]
        if r.empty: return 0
        v = r.iloc[0][kc[9]]; return float(v) if pd.notna(v) and v > 0 else 0

    elif btype == "馬連":
        b1, b2 = sorted(combo)
        w1 = top_ban(race16, 1); w2 = top_ban(race16, 2)
        if w1 is None or w2 is None: return 0
        if sorted([w1, w2]) != [b1, b2]: return 0
        r = sub[sub["jyun"] == 1]
        if r.empty: return 0
        v = r.iloc[0][kc[11]]; return float(v) if pd.notna(v) and v > 0 else 0

    elif btype == "馬単":
        ban1, ban2 = combo[0], combo[1]   # combo = [1着指定, 2着指定]
        w1 = top_ban(race16, 1); w2 = top_ban(race16, 2)
        if w1 != ban1 or w2 != ban2: return 0
        r = sub[sub["jyun"] == 1]
        if r.empty: return 0
        v = r.iloc[0][kc[12]]; return float(v) if pd.notna(v) and v > 0 else 0

    elif btype == "三連複":
        b1,b2,b3 = sorted(combo)
        w1=top_ban(race16,1); w2=top_ban(race16,2); w3=top_ban(race16,3)
        if None in (w1,w2,w3): return 0
        if sorted([w1,w2,w3]) != [b1,b2,b3]: return 0
        r = sub[sub["jyun"] == 1]
        if r.empty: return 0
        v = r.iloc[0][kc[13]]; return float(v) if pd.notna(v) and v > 0 else 0

    elif btype == "三連単":
        ban1,ban2,ban3 = combo[0],combo[1],combo[2]  # [1着,2着,3着]
        w1=top_ban(race16,1); w2=top_ban(race16,2); w3=top_ban(race16,3)
        if w1!=ban1 or w2!=ban2 or w3!=ban3: return 0
        r = sub[sub["jyun"] == 1]
        if r.empty: return 0
        v = r.iloc[0][kc[14]]; return float(v) if pd.notna(v) and v > 0 else 0

    return 0

# ── 2. BT から ◎○▲△× 復元（backtest_marks_2024.csv 使用） ─────────
print("BT読み込み中...")
bt = pd.read_csv(BT_CSV, encoding="utf-8-sig")
c  = bt.columns.tolist()
race_col = c[0]

bt["race16"] = bt[race_col].astype(str).str[:16]
race_to_r16  = bt.drop_duplicates(race_col).set_index(race_col)["race16"].to_dict()

# 全印CSV（◎○▲△×）
marks_df = pd.read_csv(BASE / "reports/backtest_marks_2024.csv", encoding="utf-8-sig")
marks = {}
for _, row in marks_df.iterrows():
    rid = row["race_id"]
    m = {}
    for key in ["hon","taikou","sabo","delta","batsu"]:
        v = row.get(key)
        if pd.notna(v): m[key] = int(v)
    marks[rid] = m

full = sum(1 for m in marks.values() if all(k in m for k in ["hon","taikou","sabo"]))
full5 = sum(1 for m in marks.values() if all(k in m for k in ["hon","taikou","sabo","delta","batsu"]))
print(f"◎○▲揃い: {full:,} / {len(marks):,} R  (◎○▲△×揃い: {full5:,}R)")

# ── 3. パターン定義 ─────────────────────────────────────────────────
# combo = 馬番リスト。馬連/三連複は順不同（sorted内部処理）、馬単/三連単は順序あり
def h(m): return m.get("hon")
def t(m): return m.get("taikou")
def s(m): return m.get("sabo")

def has(*keys):
    def _f(m): return all(k in m for k in keys)
    return _f

patterns = {}

# 単勝
patterns["単勝◎(現行)"]      = ("単勝", lambda m: [[h(m)]] if has("hon")(m) else [])
patterns["単勝○"]            = ("単勝", lambda m: [[t(m)]] if has("taikou")(m) else [])
patterns["単勝▲"]            = ("単勝", lambda m: [[s(m)]] if has("sabo")(m) else [])
patterns["単勝◎+○ 2点"]      = ("単勝", lambda m: [[h(m)],[t(m)]] if has("hon","taikou")(m) else [])
patterns["単勝◎+▲ 2点"]      = ("単勝", lambda m: [[h(m)],[s(m)]] if has("hon","sabo")(m) else [])
patterns["単勝○+▲ 2点"]      = ("単勝", lambda m: [[t(m)],[s(m)]] if has("taikou","sabo")(m) else [])
patterns["単勝◎+○+▲ 3点"]    = ("単勝", lambda m: [[h(m)],[t(m)],[s(m)]] if has("hon","taikou","sabo")(m) else [])

# 複勝
patterns["複勝◎(現行)"]      = ("複勝", lambda m: [[h(m)]] if has("hon")(m) else [])
patterns["複勝○"]            = ("複勝", lambda m: [[t(m)]] if has("taikou")(m) else [])
patterns["複勝▲"]            = ("複勝", lambda m: [[s(m)]] if has("sabo")(m) else [])
patterns["複勝◎+○ 2点"]      = ("複勝", lambda m: [[h(m)],[t(m)]] if has("hon","taikou")(m) else [])
patterns["複勝◎+▲ 2点"]      = ("複勝", lambda m: [[h(m)],[s(m)]] if has("hon","sabo")(m) else [])
patterns["複勝○+▲ 2点"]      = ("複勝", lambda m: [[t(m)],[s(m)]] if has("taikou","sabo")(m) else [])
patterns["複勝◎+○+▲ 3点"]    = ("複勝", lambda m: [[h(m)],[t(m)],[s(m)]] if has("hon","taikou","sabo")(m) else [])

# 馬連
patterns["馬連◎-○ 1点"]      = ("馬連", lambda m: [[h(m),t(m)]] if has("hon","taikou")(m) else [])
patterns["馬連◎-▲ 1点"]      = ("馬連", lambda m: [[h(m),s(m)]] if has("hon","sabo")(m) else [])
patterns["馬連○-▲ 1点"]      = ("馬連", lambda m: [[t(m),s(m)]] if has("taikou","sabo")(m) else [])
patterns["馬連◎軸 2点(現行)"] = ("馬連", lambda m: [[h(m),t(m)],[h(m),s(m)]] if has("hon","taikou","sabo")(m) else [])
patterns["馬連○軸 2点"]       = ("馬連", lambda m: [[t(m),h(m)],[t(m),s(m)]] if has("hon","taikou","sabo")(m) else [])
patterns["馬連▲軸 2点"]       = ("馬連", lambda m: [[s(m),h(m)],[s(m),t(m)]] if has("hon","taikou","sabo")(m) else [])
patterns["馬連BOX3頭 3点"]    = ("馬連", lambda m: [[h(m),t(m)],[h(m),s(m)],[t(m),s(m)]] if has("hon","taikou","sabo")(m) else [])

# 馬単（順序あり: [1着,2着]）
patterns["馬単◎→○ 1点"]       = ("馬単", lambda m: [[h(m),t(m)]] if has("hon","taikou")(m) else [])
patterns["馬単◎→▲ 1点"]       = ("馬単", lambda m: [[h(m),s(m)]] if has("hon","sabo")(m) else [])
patterns["馬単○→◎ 1点"]       = ("馬単", lambda m: [[t(m),h(m)]] if has("hon","taikou")(m) else [])
patterns["馬単▲→◎ 1点"]       = ("馬単", lambda m: [[s(m),h(m)]] if has("hon","sabo")(m) else [])
patterns["馬単○→▲ 1点"]       = ("馬単", lambda m: [[t(m),s(m)]] if has("taikou","sabo")(m) else [])
patterns["馬単▲→○ 1点"]       = ("馬単", lambda m: [[s(m),t(m)]] if has("taikou","sabo")(m) else [])
patterns["馬単◎1着流し 2点"]   = ("馬単", lambda m: [[h(m),t(m)],[h(m),s(m)]] if has("hon","taikou","sabo")(m) else [])
patterns["馬単◎2着流し 2点"]   = ("馬単", lambda m: [[t(m),h(m)],[s(m),h(m)]] if has("hon","taikou","sabo")(m) else [])
patterns["馬単◎軸F 4点"]       = ("馬単", lambda m: [[h(m),t(m)],[h(m),s(m)],[t(m),h(m)],[s(m),h(m)]] if has("hon","taikou","sabo")(m) else [])
patterns["馬単BOX6点"]         = ("馬単", lambda m: [
    [h(m),t(m)],[h(m),s(m)],[t(m),h(m)],[t(m),s(m)],[s(m),h(m)],[s(m),t(m)]
] if has("hon","taikou","sabo")(m) else [])

# 三連複（◎○▲のみ）
patterns["三連複◎○▲ 1点(現行)"] = ("三連複", lambda m: [[h(m),t(m),s(m)]] if has("hon","taikou","sabo")(m) else [])
patterns["三連複◎○△ 1点"]       = ("三連複", lambda m: [[h(m),t(m),m["delta"]]] if has("hon","taikou","delta")(m) else [])
patterns["三連複◎▲△ 1点"]       = ("三連複", lambda m: [[h(m),s(m),m["delta"]]] if has("hon","sabo","delta")(m) else [])
patterns["三連複○▲△ 1点"]       = ("三連複", lambda m: [[t(m),s(m),m["delta"]]] if has("taikou","sabo","delta")(m) else [])
# ◎軸流し
patterns["三連複◎軸3点(○▲△)"]  = ("三連複", lambda m: [
    [h(m),t(m),s(m)],[h(m),t(m),m["delta"]],[h(m),s(m),m["delta"]]
] if has("hon","taikou","sabo","delta")(m) else [])
# ◎○軸2頭流し
patterns["三連複◎○軸2点(▲△)"]  = ("三連複", lambda m: [
    [h(m),t(m),s(m)],[h(m),t(m),m["delta"]]
] if has("hon","taikou","sabo","delta")(m) else [])
# ボックス
patterns["三連複BOX4頭 4点"]     = ("三連複", lambda m: [
    [h(m),t(m),s(m)],[h(m),t(m),m["delta"]],[h(m),s(m),m["delta"]],[t(m),s(m),m["delta"]]
] if has("hon","taikou","sabo","delta")(m) else [])

# 三連単（順序あり: [1着,2着,3着]）
patterns["三連単◎1着固定 2点"] = ("三連単", lambda m: [[h(m),t(m),s(m)],[h(m),s(m),t(m)]] if has("hon","taikou","sabo")(m) else [])
patterns["三連単◎○F 4点"]      = ("三連単", lambda m: [
    [h(m),t(m),s(m)],[h(m),s(m),t(m)],[t(m),h(m),s(m)],[t(m),s(m),h(m)]
] if has("hon","taikou","sabo")(m) else [])
patterns["三連単BOX6点"]       = ("三連単", lambda m: [
    [h(m),t(m),s(m)],[h(m),s(m),t(m)],[t(m),h(m),s(m)],
    [t(m),s(m),h(m)],[s(m),h(m),t(m)],[s(m),t(m),h(m)]
] if has("hon","taikou","sabo")(m) else [])

# ── 4. シミュレーション ─────────────────────────────────────────────
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
    avg   = (tr / hits) if hits else 0
    results[name] = {"btype": btype, "pts": len(fn(next(m for m in marks.values() if fn(m)))),
                     "R": races, "hits": hits, "hrate": hrate, "avg": avg, "roi": roi}

# ── 5. 出力 ─────────────────────────────────────────────────────────
print()
ORDER = ["単勝","複勝","馬連","馬単","三連複","三連単"]
for btype in ORDER:
    items = [(n,r) for n,r in results.items() if r["btype"]==btype]
    if not items: continue
    print(f"\n{'─'*72}")
    print(f"  {btype}")
    print(f"{'─'*72}")
    print(f"  {'買い目':<24} {'点数':>3} {'R':>5} {'的中率':>7} {'平均払戻':>10} {'ROI':>7}")
    print(f"  {'─'*66}")
    for name, r in sorted(items, key=lambda x: x[1]["roi"], reverse=True):
        mark = "★" if r["roi"] >= 80 else ("  " if r["roi"] >= 70 else "NG")
        cur  = " ←現行" if "現行" in name else ""
        print(f"  {mark} {name:<24} {r['pts']:>2}点 {r['R']:>5}R "
              f"{r['hrate']:>6.1f}% {r['avg']:>10,.0f}円 {r['roi']:>6.1f}%{cur}")

print("\n" + "="*72)
print("※ 1点あたり10,000円固定、複数点は均等割り")
print("※ kekka: kekka_20160105_20251228_v2.csv（全頭版）/ BT: backtest_results_2024.csv")
