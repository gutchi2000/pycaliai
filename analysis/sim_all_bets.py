"""全買い目パターンROIシミュレーション"""
import pandas as pd
import sys
from pathlib import Path

BASE = Path("E:/PyCaLiAI")
BT_CSV    = BASE / "reports/backtest_results_2024.csv"
KEKKA_CSV = BASE / "data/kekka_20130105-20251228.csv"
BUDGET = 10_000

# ── 1. kekka マスター読み込み ─────────────────────────────────────
print("kekka 読み込み中...")
kk = pd.read_csv(KEKKA_CSV, encoding="cp932")
kc = kk.columns.tolist()
# 正しいカラム順: 0=レースID, 1=馬番, 2=血統登録番号, 3=確定着順,
#                4=単勝配当, 5=複勝配当, 6=枠連, 7=馬連, 8=馬単, 9=３連複
rid_col  = kc[0]; ban_col  = kc[1]; jyun_col = kc[3]
tan_col  = kc[4]; fuku_col = kc[5]; umaren_col = kc[7]; sf_col = kc[9]

kk["race16"] = kk[rid_col].astype(str).str[:16]
for col in [ban_col, jyun_col, tan_col, fuku_col, umaren_col, sf_col]:
    kk[col] = pd.to_numeric(kk[col], errors="coerce")

# race16 → kekka rows の辞書（高速化）
kk_dict = {}
for race16, grp in kk.groupby("race16"):
    kk_dict[race16] = grp
print(f"  レース数: {len(kk_dict):,}  行数: {len(kk):,}")

def get_pay(race16, combo, btype):
    """払戻(100円あたり)。外れ=0"""
    sub = kk_dict.get(race16)
    if sub is None: return 0
    combo = sorted(combo)
    if btype == "単勝":
        row = sub[(sub[ban_col] == combo[0]) & (sub[jyun_col] == 1)]
        if row.empty: return 0
        v = row.iloc[0][tan_col]
        return float(v) if pd.notna(v) and v > 0 else 0
    elif btype == "複勝":
        row = sub[sub[ban_col] == combo[0]]
        if row.empty: return 0
        v = row.iloc[0][fuku_col]
        return float(v) if pd.notna(v) and v > 0 else 0
    elif btype == "馬連":
        w = sub[sub[jyun_col] == 1]
        if w.empty: return 0
        b1 = int(w.iloc[0][ban_col])
        b2_rows = sub[sub[jyun_col] == 2][ban_col].dropna().astype(int)
        if b2_rows.empty: return 0
        b2 = int(b2_rows.iloc[0])
        if sorted([b1, b2]) != combo: return 0
        v = w.iloc[0][umaren_col]
        return float(v) if pd.notna(v) and v > 0 else 0
    elif btype == "三連複":
        top = {}
        for jyun in [1, 2, 3]:
            rows = sub[sub[jyun_col] == jyun][ban_col].dropna().astype(int)
            if rows.empty: return 0
            top[jyun] = int(rows.iloc[0])
        actual = sorted(top.values())
        if actual != combo: return 0
        w = sub[sub[jyun_col] == 1]
        v = w.iloc[0][sf_col]
        return float(v) if pd.notna(v) and v > 0 else 0
    return 0

# ── 2. バックテストから ◎○▲ 復元 ─────────────────────────────────
bt = pd.read_csv(BT_CSV, encoding="utf-8-sig")
c  = bt.columns.tolist()
race_col, bet_col, mark_col, hit_col = c[0], c[6], c[7], c[14]

marks = {}
for race_id, grp in bt.groupby(race_col):
    m = {}
    tan = grp[grp[bet_col].str.encode("utf-8") == "単勝".encode("utf-8")]
    if not tan.empty:
        m["hon"] = int(tan.iloc[0][mark_col])
    uma = grp[grp[bet_col].str.encode("utf-8") == "馬連".encode("utf-8")].reset_index(drop=True)
    if len(uma) >= 1 and "hon" in m:
        pair = sorted(map(int, str(uma.iloc[0][mark_col]).split("-")))
        t = [x for x in pair if x != m["hon"]]
        if t: m["taikou"] = t[0]
    if len(uma) >= 2 and "hon" in m:
        pair = sorted(map(int, str(uma.iloc[1][mark_col]).split("-")))
        s = [x for x in pair if x != m["hon"]]
        if s: m["sabo"] = s[0]
    marks[race_id] = m

bt["race16"] = bt[race_col].astype(str).str[:16]
race_to_r16  = bt.drop_duplicates(race_col).set_index(race_col)["race16"].to_dict()

full_races = sum(1 for m in marks.values() if len(m) >= 3)
print(f"◎○▲揃い: {full_races} / {len(marks)} R")

# ── 3. 全パターン定義 ──────────────────────────────────────────────
patterns = {
    "単勝◎":             ("単勝",   lambda m: [[m["hon"]]]         if "hon" in m else []),
    "単勝○":             ("単勝",   lambda m: [[m["taikou"]]]      if "taikou" in m else []),
    "単勝▲":             ("単勝",   lambda m: [[m["sabo"]]]        if "sabo" in m else []),
    "複勝◎":             ("複勝",   lambda m: [[m["hon"]]]         if "hon" in m else []),
    "複勝○":             ("複勝",   lambda m: [[m["taikou"]]]      if "taikou" in m else []),
    "複勝▲":             ("複勝",   lambda m: [[m["sabo"]]]        if "sabo" in m else []),
    "複勝◎+○":           ("複勝",   lambda m: [[m["hon"]],[m["taikou"]]]         if "hon" in m and "taikou" in m else []),
    "複勝◎+○+▲":        ("複勝",   lambda m: [[m["hon"]],[m["taikou"]],[m["sabo"]]] if len(m)>=3 else []),
    "馬連◎-○":          ("馬連",   lambda m: [[m["hon"],m["taikou"]]]           if "hon" in m and "taikou" in m else []),
    "馬連◎-▲":          ("馬連",   lambda m: [[m["hon"],m["sabo"]]]             if "hon" in m and "sabo" in m else []),
    "馬連○-▲":          ("馬連",   lambda m: [[m["taikou"],m["sabo"]]]          if "taikou" in m and "sabo" in m else []),
    "馬連◎軸2点(現行)":  ("馬連",   lambda m: [[m["hon"],m["taikou"]],[m["hon"],m["sabo"]]] if len(m)>=3 else []),
    "馬連全3点":         ("馬連",   lambda m: [[m["hon"],m["taikou"]],[m["hon"],m["sabo"]],[m["taikou"],m["sabo"]]] if len(m)>=3 else []),
    "三連複◎○▲(現行)":  ("三連複", lambda m: [[m["hon"],m["taikou"],m["sabo"]]] if len(m)>=3 else []),
}

# ── 4. シミュレーション実行 ────────────────────────────────────────
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
    results[name] = {"btype": btype, "R": races, "hits": hits,
                     "hrate": hrate, "avg": avg, "roi": roi}

# ── 5. 出力 ────────────────────────────────────────────────────────
print()
print("=" * 65)
print(f"{'買い目':22s} {'R':>5} {'的中率':>6} {'平均払戻':>9} {'ROI':>7}")
print("=" * 65)
by_type = {}
for name, r in results.items():
    by_type.setdefault(r["btype"], []).append((name, r))

for btype in ["単勝", "複勝", "馬連", "三連複"]:
    print(f"  --- {btype} ---")
    for name, r in sorted(by_type.get(btype, []), key=lambda x: x[1]["roi"], reverse=True):
        mark = "[OK]" if r["roi"] >= 80 else ("[  ]" if r["roi"] >= 70 else "[NG]")
        cur = " <現行>" if "現行" in name else ""
        print(f"  {mark} {name:<22}{r['R']:>5}R  {r['hrate']:>5.1f}%  {r['avg']:>9,.0f}円  {r['roi']:>6.1f}%{cur}")
