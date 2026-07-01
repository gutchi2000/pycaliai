# -*- coding: utf-8 -*-
"""
diag_baba_robustness.py — 重馬場で予想が崩れるか? (2025 OOS)
=============================================================
雛形: umami_uma_backtest.py の model/master/odds ロード部。
v6 model で各馬スコア -> レース内 ◎(スコア最上位) を特定し、
  (1) ◎ 複勝圏率(着順<=3) を 馬場状態別
  (2) ◎ 単勝的中率(着順==1) を 馬場状態別
  (3) 良 vs 稍/重/不良 で精度が落ちるか
  (4) 枠番(内枠1-4 / 外枠5-8)別の ◎複勝圏率
leak-safe: 馬場状態・枠番 は発走前確定。2025(test)のみ。
"""
import sys, io
from collections import defaultdict
from pathlib import Path
import joblib, numpy as np, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
BASE = Path(r"E:\PyCaLiAI"); sys.path.insert(0, str(BASE))
from backtest_pl_ev import COL_RID, COL_BAN, COL_JYUN

COL_BABA = "馬場状態"
COL_WAKU = "枠番"

print("[load] model/master ...")
b = joblib.load(BASE / "models/unified_rank_v6.pkl")
model, feats, encs = b["model"], b["feature_cols"], b["encoders"]

df = pd.read_csv(BASE / "data/master_v2_20130105-20251228.csv", encoding="utf-8-sig", low_memory=False)
df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
df = df.dropna(subset=[COL_JYUN, COL_RID, "split"])
df = df[df["split"] == "test"].copy()
df["year"] = df[COL_RID].astype(str).str[:4].astype(int)
df = df[df["year"] == 2025].copy()  # 2025 のみ (重い処理対策)
print(f"  2025 rows={len(df)}")

# baba は encoder で潰れる前に raw を退避
baba_raw = df[COL_BABA].astype(str).copy()
waku_raw = pd.to_numeric(df[COL_WAKU], errors="coerce").copy()

# encoders 適用 (model 入力用)
for c, le in encs.items():
    if c in df.columns:
        v = df[c].astype(str).fillna("__NaN__")
        df[c] = le.transform(v.where(v.isin(set(le.classes_)), "__NaN__"))

X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
df["_score"] = model.predict(X)
df["_baba"] = baba_raw.values
df["_waku"] = waku_raw.values

# 馬場状態の正規化 (良/稍/重/不良) -- master値は '良' '稍' '重' '不'/'不良' 等の揺れ得る
def norm_baba(s):
    s = str(s).strip()
    if s.startswith("良"): return "良"
    if s.startswith("稍"): return "稍重"
    if s.startswith("重"): return "重"
    if s.startswith("不"): return "不良"
    return "他"
df["_baba_n"] = df["_baba"].map(norm_baba)

print("  baba raw uniq:", sorted(df["_baba"].dropna().unique().tolist())[:12])
print("  baba norm dist (horse-rows):")
print(df["_baba_n"].value_counts().to_dict())

# レース単位で ◎ = スコア最上位 を特定
recs = []  # 1 row per race: 馬場, 枠, 複勝圏(<=3), 単勝(==1)
for rid, g in df.groupby(COL_RID, sort=False):
    if len(g) < 5:
        continue
    g = g.reset_index(drop=True)
    hon = g.iloc[g["_score"].values.argmax()]
    jun = hon[COL_JYUN]
    if pd.isna(jun):
        continue
    jun = int(jun)
    recs.append({
        "baba": hon["_baba_n"],
        "waku": hon["_waku"],
        "fukuken": 1 if (1 <= jun <= 3) else 0,
        "tansho": 1 if jun == 1 else 0,
    })
R = pd.DataFrame(recs)
print(f"\n  races(>=5頭, ◎着順あり) = {len(R)}")

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n
    den = 1 + z*z/n
    cen = (p + z*z/(2*n)) / den
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return (max(0, cen-half)*100, min(1, cen+half)*100)

print("\n" + "="*72)
print("(1)(2) ◎ 馬場状態別  複勝圏率(着順<=3) / 単勝的中率(==1)")
print("="*72)
print(f"{'馬場':<8}{'R数':>7}{'複勝圏率':>10}{'95%CI':>16}{'単勝率':>9}{'95%CI':>16}")
order = ["良", "稍重", "重", "不良"]
baba_summary = {}
for bb in order:
    sub = R[R["baba"] == bb]
    n = len(sub)
    if n == 0:
        print(f"{bb:<8}{0:>7}{'-':>10}")
        continue
    fk = sub["fukuken"].sum(); ts = sub["tansho"].sum()
    fr = fk/n*100; tr = ts/n*100
    flo, fhi = wilson(fk, n); tlo, thi = wilson(ts, n)
    baba_summary[bb] = {"n": int(n), "fuku": fr, "tan": tr, "fuku_ci": (flo, fhi), "tan_ci": (tlo, thi)}
    print(f"{bb:<8}{n:>7}{fr:>9.1f}%  [{flo:>4.1f},{fhi:>4.1f}]  {tr:>7.1f}%  [{tlo:>4.1f},{thi:>4.1f}]")
# 他(分類不能)があれば表示
oth = R[~R["baba"].isin(order)]
if len(oth):
    print(f"{'(他)':<8}{len(oth):>7}{oth['fukuken'].mean()*100:>9.1f}%")

print("\n" + "="*72)
print("(3) 良 vs 非良(稍重/重/不良) — 精度低下の検定")
print("="*72)
good = R[R["baba"] == "良"]
heavy = R[R["baba"].isin(["稍重", "重", "不良"])]
ng, nh = len(good), len(heavy)
fg = good["fukuken"].mean()*100 if ng else 0
fh = heavy["fukuken"].mean()*100 if nh else 0
tg = good["tansho"].mean()*100 if ng else 0
th = heavy["tansho"].mean()*100 if nh else 0
print(f"  良     : R={ng:>5}  複勝圏 {fg:.1f}%  単勝 {tg:.1f}%")
print(f"  非良   : R={nh:>5}  複勝圏 {fh:.1f}%  単勝 {th:.1f}%")
print(f"  差(良-非良): 複勝圏 {fg-fh:+.1f}pt   単勝 {tg-th:+.1f}pt")
# 2-proportion z-test (複勝圏)
kg, kh = good["fukuken"].sum(), heavy["fukuken"].sum()
if ng and nh:
    pp = (kg+kh)/(ng+nh); se = np.sqrt(pp*(1-pp)*(1/ng+1/nh))
    z = ((kg/ng)-(kh/nh))/se if se > 0 else 0
    from math import erf, sqrt
    pval = 2*(1-0.5*(1+erf(abs(z)/sqrt(2))))
    print(f"  複勝圏 2-prop z={z:+.2f}  p={pval:.3f}  ({'有意差あり' if pval<0.05 else '有意差なし'})")
# 重+不良 だけ (本当の重馬場)
deep = R[R["baba"].isin(["重", "不良"])]
nd = len(deep)
if nd:
    fd = deep["fukuken"].mean()*100; td = deep["tansho"].mean()*100
    print(f"  重+不良: R={nd:>5}  複勝圏 {fd:.1f}%  単勝 {td:.1f}%  (差 複勝圏 {fg-fd:+.1f}pt)")
    kd = deep["fukuken"].sum()
    if ng:
        pp = (kg+kd)/(ng+nd); se = np.sqrt(pp*(1-pp)*(1/ng+1/nd))
        z = ((kg/ng)-(kd/nd))/se if se > 0 else 0
        pval = 2*(1-0.5*(1+erf(abs(z)/sqrt(2))))
        print(f"  良 vs 重+不良 複勝圏 z={z:+.2f}  p={pval:.3f}  ({'有意差あり' if pval<0.05 else '有意差なし'})")

print("\n" + "="*72)
print("(4) ◎ 枠番別 複勝圏率  (内枠1-4 / 外枠5-8、および枠ごと)")
print("="*72)
R["waku_i"] = pd.to_numeric(R["waku"], errors="coerce")
inner = R[(R["waku_i"] >= 1) & (R["waku_i"] <= 4)]
outer = R[(R["waku_i"] >= 5) & (R["waku_i"] <= 8)]
waku_summary = {}
for lab, sub in [("内枠1-4", inner), ("外枠5-8", outer)]:
    n = len(sub)
    if n == 0: continue
    fk = sub["fukuken"].sum(); fr = fk/n*100; flo, fhi = wilson(fk, n)
    ts = sub["tansho"].sum(); tr = ts/n*100
    waku_summary[lab] = {"n": int(n), "fuku": fr, "tan": tr}
    print(f"  {lab:<8} R={n:>5}  複勝圏 {fr:.1f}% [{flo:.1f},{fhi:.1f}]  単勝 {tr:.1f}%")
ni, no = len(inner), len(outer)
if ni and no:
    print(f"  差(内-外): 複勝圏 {inner['fukuken'].mean()*100-outer['fukuken'].mean()*100:+.1f}pt")
print(f"  {'-'*40}")
print(f"  {'枠':<6}{'R数':>7}{'複勝圏率':>10}")
waku_each = {}
for w in range(1, 9):
    sub = R[R["waku_i"] == w]
    n = len(sub)
    if n == 0: continue
    fr = sub["fukuken"].mean()*100
    waku_each[int(w)] = {"n": int(n), "fuku": round(fr, 1)}
    print(f"  {w:<6}{n:>7}{fr:>9.1f}%")

# 馬場×枠 のクロス (内枠の重馬場で特に崩れるか)
print(f"\n  [クロス] 馬場 × 内外枠 複勝圏率")
for bb in ["良", "稍重", "重", "不良"]:
    for lab, lo, hi in [("内", 1, 4), ("外", 5, 8)]:
        sub = R[(R["baba"] == bb) & (R["waku_i"] >= lo) & (R["waku_i"] <= hi)]
        n = len(sub)
        if n >= 10:
            print(f"    {bb}×{lab}枠: R={n:>4} 複勝圏 {sub['fukuken'].mean()*100:.1f}%")

# 機械可読サマリを末尾に
print("\n[SUMMARY-JSON]")
import json
print(json.dumps({
    "n_races": len(R),
    "baba": {k: {"n": v["n"], "fuku": round(v["fuku"], 1), "tan": round(v["tan"], 1)} for k, v in baba_summary.items()},
    "good_vs_heavy": {"good_n": ng, "good_fuku": round(fg, 1), "heavy_n": nh, "heavy_fuku": round(fh, 1), "diff_fuku_pt": round(fg-fh, 1)},
    "waku_io": waku_summary,
    "waku_each": waku_each,
}, ensure_ascii=False))
