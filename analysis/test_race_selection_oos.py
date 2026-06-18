# -*- coding: utf-8 -*-
"""
test_race_selection_oos.py — 「レース選別」が効くかを正しいOOSで決定検証。
前提を疑う: 1頭予測は天井でも『このレースで◎が来るか』のレース単位選別は別タスク。
2024でゲート閾値fit → 2025で評価(in-sample artifact回避)。◎複勝/単勝のflat ROIが控除床(複勝~80%)を
レジーム別に超えるか。レース単位メタ特徴(全てleak-safe: 発走前の確率分布+9時前オッズ)で層別。
"""
import sys, io, glob, re
import joblib, numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
sys.path.insert(0, "E:/PyCaLiAI")
import pl_probs as PL
from backtest_pl_ev import all_fukusho_vec_fast, load_payouts, COL_RID, COL_BAN, COL_JYUN

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
pays = load_payouts(); tan9 = load_tan9([2024, 2025])

recs = []
for rid, g in df.groupby(COL_RID, sort=False):
    if len(g) < 6: continue
    r = rid16(rid); od = tan9.get(r); po = pays.get(r)
    if od is None or po is None: continue
    g = g.sort_values(COL_BAN); ban = g[COL_BAN].astype(int).values; w = PL.pl_weights(g["_score"].values)
    pw = np.clip(cal["tansho"].predict(PL.all_tansho(w)), 1e-9, 1)
    n = len(pw); order = np.argsort(-pw)
    hon = int(ban[order[0]])
    dom = float(pw[order[0]]); p2 = float(pw[order[1]]); gap = dom - p2
    ent = float(-(pw * np.log(pw)).sum() / np.log(n))  # 0=確信, 1=混沌
    # market agreement: model p_win vs 9時前 implied(1/odds) の順位相関
    imp = np.array([1.0 / od[int(ban[i])] if int(ban[i]) in od else np.nan for i in range(n)])
    msk = ~np.isnan(imp)
    agree = float(spearmanr(pw[msk], imp[msk]).correlation) if msk.sum() >= 4 else np.nan
    # ◎の複勝/単勝 結果
    hon_fin = 1 if hon == po["win"] else (2 if hon == po["plc"] else (3 if hon == po["sho"] else 0))
    fuku_pay = {1: po["fuku_win"], 2: po["fuku_plc"], 3: po["fuku_sho"]}.get(hon_fin, 0.0)
    fuku_pay = float(fuku_pay) if fuku_pay == fuku_pay else 0.0
    tan_pay = float(po["tansho"]) if (hon == po["win"] and po["tansho"] == po["tansho"]) else 0.0
    recs.append(dict(year=int(g["year"].iloc[0]), n=n, dom=dom, gap=gap, ent=ent, agree=agree,
                     hon_top3=int(hon_fin > 0), hon_win=int(hon_fin == 1),
                     fuku_pay=fuku_pay, tan_pay=tan_pay))
R = pd.DataFrame(recs).dropna(subset=["agree"])
tr = R[R.year == 2024]; te = R[R.year == 2025]
print(f"races: 2024(fit)={len(tr)}  2025(eval)={len(te)}")

def stats(d):
    return dict(n=len(d), top3=round(d.hon_top3.mean()*100,1),
                fuku_roi=round(d.fuku_pay.sum()/(len(d)*100)*100,1),
                tan_roi=round(d.tan_pay.sum()/(len(d)*100)*100,1))
print("\n=== ベースライン(全レース ◎ flat) ===")
print(" 2024:", stats(tr)); print(" 2025:", stats(te))

print("\n=== 各メタ特徴で 2024fit閾値 → 2025選別 (◎複勝ROI/単勝ROI/top3率) ===")
print("控除床: 複勝~80% / 単勝80%。超えれば『選別は効く』")
for feat, hi_is_clean, label in [("dom", True, "◎優勢(dom高)"), ("gap", True, "1-2着差(gap高)"),
                                  ("ent", False, "低カオス(ent低)"), ("agree", True, "市場一致(agree高)")]:
    print(f"\n--- {label} ---")
    for q in [0.5, 0.7, 0.8]:
        thr = tr[feat].quantile(q if hi_is_clean else 1 - q)
        sel = te[te[feat] >= thr] if hi_is_clean else te[te[feat] <= thr]
        s = stats(sel)
        flag = " ★複勝床超" if s["fuku_roi"] >= 80 else ""
        flag += " ◎単勝床超" if s["tan_roi"] >= 80 else ""
        keep = f"上位{int((1-q)*100)}%" if hi_is_clean else f"下位{int((1-q)*100)}%"
        print(f"  {label}{keep:>7}(thr={thr:.3f}): n{s['n']:>4} top3 {s['top3']:>5}% 複勝ROI {s['fuku_roi']:>5}% 単勝ROI {s['tan_roi']:>5}%{flag}")

# 複合: dom高 & ent低 & agree高 (2024 median 各)
print("\n=== 複合ゲート: dom>=med & ent<=med & agree>=med (2024fit→2025) ===")
md, me, ma = tr.dom.median(), tr.ent.median(), tr.agree.median()
sel = te[(te.dom >= md) & (te.ent <= me) & (te.agree >= ma)]
print(" 複合:", stats(sel), f"(全2025 {len(te)}中 {len(sel)}採用)")
