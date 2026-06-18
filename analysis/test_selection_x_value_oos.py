# -*- coding: utf-8 -*-
"""
test_selection_x_value_oos.py — 「レース選別 × 妙味構築」で控除床/100%を超えられるか。
クリーンなレース(低エントロピー)を選んだ上で、過大評価の◎を買うのでなく
妙味馬(under×odds<15)/ワイドフォメを買う構築を、エントロピー3帯×複数構築でOOS総当たり。
2024fit閾値 → 2025eval。控除床(複勝~80/連系77.5)・100%(黒字)を超えるセルを探す。
"""
import sys, io, glob, re
import joblib, numpy as np, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
sys.path.insert(0, "E:/PyCaLiAI")
import pl_probs as PL
from backtest_pl_ev import all_fukusho_vec_fast, all_umaren_mat, load_payouts, load_wide_payouts, COL_RID, COL_BAN, COL_JYUN

ODIR = "E:/PyCaLiAI/data/Time _series_odds"
def rid16(x): return re.sub(r"\D", "", str(x))[:16]

def load_tan9(years):
    f = sorted(glob.glob(f"{ODIR}/TANPUK_*.csv"))[-1]
    tp = pd.read_csv(f, encoding="cp932", low_memory=False); cols = list(tp.columns); RID, KB, TM = cols[0], cols[1], cols[2]
    tan_c = {}
    for c in cols:
        m = re.match(r"^\s*(\d+)\s*単\s*$", str(c))
        if m: tan_c[int(m.group(1))] = c
    tp["rid16"] = tp[RID].map(rid16)
    tp = tp[(tp["rid16"].str.len() == 16) & (tp["rid16"].str[:4].astype(int).isin(years))]
    out = {}
    for rid, g in tp.groupby("rid16", sort=False):
        mmdd = int(rid[4:8]); kb = pd.to_numeric(g[KB], errors="coerce"); g1 = g[kb == 1]
        if len(g1) == 0: continue
        tm = pd.to_numeric(g1[TM], errors="coerce"); hh = tm % 10000; dd = tm // 10000
        base = g1[dd == mmdd]; base = base if len(base) > 0 else g1
        row = base.loc[(hh.loc[base.index] - 900).abs().idxmin()]
        tan9 = {}
        for ban, c in tan_c.items():
            v = pd.to_numeric(pd.Series([row[c]]), errors="coerce").iloc[0]
            if pd.notna(v) and v > 1.0: tan9[ban] = float(v)
        out[rid] = tan9
    return out

print("[load] ...")
b = joblib.load("E:/PyCaLiAI/models/unified_rank_v6.pkl"); model, feats, encs = b["model"], b["feature_cols"], b["encoders"]
cal = joblib.load("E:/PyCaLiAI/models/pl_calibrators_v6.pkl"); cal = cal.get("calibrators", cal)
df = pd.read_csv("E:/PyCaLiAI/data/master_v2_20130105-20251228.csv", encoding="utf-8-sig", low_memory=False)
df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce"); df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]); df = df[df["split"] == "test"].copy()
df["year"] = df[COL_RID].astype(str).str[:4].astype(int); df = df[df["year"].isin([2024, 2025])]
for c, le in encs.items():
    if c in df.columns:
        v = df[c].astype(str).fillna("__NaN__"); df[c] = le.transform(v.where(v.isin(set(le.classes_)), "__NaN__"))
X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values; df["_score"] = model.predict(X)
pays = load_payouts(); tan9 = load_tan9([2024, 2025]); wides = load_wide_payouts()

races = []
for rid, g in df.groupby(COL_RID, sort=False):
    if len(g) < 6: continue
    r = rid16(rid); od = tan9.get(r); po = pays.get(r)
    if od is None or po is None: continue
    g = g.sort_values(COL_BAN); ban = g[COL_BAN].astype(int).values; w = PL.pl_weights(g["_score"].values)
    pw = np.clip(cal["tansho"].predict(PL.all_tansho(w)), 1e-9, 1)
    pf = np.clip(cal["fukusho"].predict(all_fukusho_vec_fast(w)), 1e-9, 1)
    n = len(pw); order = np.argsort(-pw)
    ent = float(-(pw * np.log(pw)).sum() / np.log(n))
    hon, tai, san = int(ban[order[0]]), int(ban[order[1]]), int(ban[order[2]])
    # value pick: odds<15 かつ under(pw>1/odds) の中で pw 最大
    val = None; valbest = -1
    for i in range(n):
        bn = int(ban[i]); o = od.get(bn)
        if o and o < 15 and pw[i] > 1.0 / o and pw[i] > valbest:
            valbest = pw[i]; val = bn
    fin = {1: po["win"], 2: po["plc"], 3: po["sho"]}
    top3 = {po["win"], po["plc"], po["sho"]}
    def fuku(bn):
        if bn == po["win"]: return float(po["fuku_win"]) if po["fuku_win"] == po["fuku_win"] else 0.0
        if bn == po["plc"]: return float(po["fuku_plc"]) if po["fuku_plc"] == po["fuku_plc"] else 0.0
        if bn == po["sho"]: return float(po["fuku_sho"]) if po["fuku_sho"] == po["fuku_sho"] else 0.0
        return 0.0
    def tan(bn): return float(po["tansho"]) if (bn == po["win"] and po["tansho"] == po["tansho"]) else 0.0
    wd = {frozenset((int(i), int(j))): float(p) for i, j, p in wides.get(r, [])}
    def wide(a, c): return wd.get(frozenset((a, c)), 0.0)
    races.append(dict(year=int(g["year"].iloc[0]), ent=ent, hon=hon, tai=tai, san=san, val=val,
                      # 構築別 payout(flat100/点)
                      hon_tan=tan(hon), hon_fuku=fuku(hon),
                      tai_fuku=fuku(tai),
                      val_tan=(tan(val) if val else None), val_fuku=(fuku(val) if val else None),
                      w_hontai=wide(hon, tai),
                      w_box3=(wide(hon, tai) + wide(hon, san) + wide(tai, san)),  # ◎〇▲ box 3点
                      n=n))
R = pd.DataFrame(races); tr = R[R.year == 2024]; te = R[R.year == 2025]
q1, q2 = tr.ent.quantile(1/3), tr.ent.quantile(2/3)  # 2024fit
te = te.copy(); te["bin"] = np.where(te.ent <= q1, "clean", np.where(te.ent <= q2, "mid", "chaotic"))
print(f"races 2024fit={len(tr)} 2025eval={len(te)} | ent terciles(2024) {q1:.3f}/{q2:.3f}")

def roi_flat(d, col, pts=1):
    dd = d.dropna(subset=[col]);
    if len(dd) == 0: return None
    return round(dd[col].sum() / (len(dd) * 100 * pts) * 100, 1), len(dd)

constructs = [
    ("◎単勝", "hon_tan", 1), ("◎複勝", "hon_fuku", 1),
    ("〇複勝", "tai_fuku", 1),
    ("妙味単勝(under<15)", "val_tan", 1), ("妙味複勝(under<15)", "val_fuku", 1),
    ("◎-〇ワイド", "w_hontai", 1), ("◎〇▲boxワイド3点", "w_box3", 3),
]
print(f"\n{'構築':<22}{'全2025':>12}{'clean':>14}{'mid':>14}{'chaotic':>14}")
print("控除床: 単複80 / ワイド77.5。★=100%超(黒字)")
for name, col, pts in constructs:
    cells = []
    for sub in [te, te[te.bin=="clean"], te[te.bin=="mid"], te[te.bin=="chaotic"]]:
        rr = roi_flat(sub, col, pts)
        cells.append(f"{rr[0]:>6}%(n{rr[1]})" if rr else "  -")
    star = " ★" if any("★" for c in cells) else ""
    line = f"{name:<22}" + "".join(f"{c:>14}" for c in cells)
    print(line)
print("\n[読み] clean帯で各構築が床/100%をどこまで超えるか。◎は priced で頭打ち、妙味/ワイドが clean で伸びるか。")
