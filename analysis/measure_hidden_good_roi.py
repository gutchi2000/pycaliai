# -*- coding: utf-8 -*-
"""
measure_hidden_good_roi.py  (v2: 全頭母数で正しく計算)
====================================================
実測: kako5_hidden_good_count 保有群の「次走」単勝/複勝 ROI。
kekka は複勝圏内(in-the-money)の馬しか持たない → master(全頭) に LEFT join し、
非入着馬は払戻0 とする。ROI% = mean(payout_per_100bet)。
hidden>=1 が hidden==0 を有意に上回れば「市場に未織り込み(妙味)」、
ほぼ同水準なら「公開情報ゆえ織り込み済み」。CLV は時系列オッズ無しで N/A。
"""
import sys, io
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
rng = np.random.default_rng(42)

KEKKA  = r"E:\PyCaLiAI\data\kekka_20130105-20251228.csv"
MASTER = r"E:\PyCaLiAI\data\master_v2_20130105-20251228.csv"

def read_any(path, **kw):
    for enc in ("utf-8-sig", "cp932", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc, **kw)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"encoding fail: {path}")

print("=== load kekka (payouts, in-the-money only) ===", flush=True)
k = read_any(KEKKA, usecols=["レースID(新)", "単勝配当", "複勝配当"], low_memory=False)
for c in ["単勝配当", "複勝配当"]:
    k[c] = pd.to_numeric(k[c].astype(str).str.replace(",", "").str.replace("円", ""),
                         errors="coerce").fillna(0.0)
k = k.drop_duplicates(subset=["レースID(新)"])
print(f"kekka rows (unique 馬-key): {len(k):,}", flush=True)

print("=== load master_v2 (all runners) ===", flush=True)
m = read_any(MASTER, usecols=["レースID(新)", "日付", "着順",
                              "kako5_hidden_good_count", "kako5_race_count"],
             low_memory=False)
m["着順"] = pd.to_numeric(m["着順"], errors="coerce")
m["kako5_hidden_good_count"] = pd.to_numeric(m["kako5_hidden_good_count"], errors="coerce")
m["kako5_race_count"]        = pd.to_numeric(m["kako5_race_count"], errors="coerce")
m["日付"] = pd.to_numeric(m["日付"], errors="coerce")
print(f"master rows: {len(m):,} / unique レースID(新): {m['レースID(新)'].nunique():,} "
      f"(=> 馬単位キーか: {m['レースID(新)'].nunique()==len(m)})", flush=True)

df = m.merge(k, on="レースID(新)", how="left")
df["単勝配当"] = df["単勝配当"].fillna(0.0)
df["複勝配当"] = df["複勝配当"].fillna(0.0)
assert len(df) == len(m)
print(f"払戻付与: 単勝>0 {(df['単勝配当']>0).mean()*100:.1f}% / 複勝>0 {(df['複勝配当']>0).mean()*100:.1f}% (≈勝率/複勝率)", flush=True)

df["tan_ret"]  = df["単勝配当"]      # 勝った馬のみ>0、他0
df["fuku_ret"] = df["複勝配当"]      # 複勝圏のみ>0、他0

def period(d):
    if pd.isna(d): return "?"
    d = int(d)
    if d >= 20240101: return "test(2024-25)"
    if d >= 20230101: return "valid(2023)"
    return "train(<=2022)"
df["period"] = df["日付"].apply(period)

def boot_ci(vals, n_boot=2000):
    vals = vals.to_numpy(dtype=float)
    n = len(vals)
    if n == 0: return (np.nan, np.nan)
    means = rng.choice(vals, size=(n_boot, n), replace=True).mean(axis=1)
    return (round(np.percentile(means, 2.5), 1), round(np.percentile(means, 97.5), 1))

def roi_rows(sub):
    groups = [
        ("ALL(母集団)",          sub),
        ("hidden==0",            sub[sub.kako5_hidden_good_count == 0]),
        ("hidden>=1",            sub[sub.kako5_hidden_good_count >= 1]),
        ("hidden>=2",            sub[sub.kako5_hidden_good_count >= 2]),
    ]
    out = []
    for name, g in groups:
        n = len(g)
        if n == 0: continue
        tan_lo, tan_hi = boot_ci(g.tan_ret)
        fk_lo,  fk_hi  = boot_ci(g.fuku_ret)
        out.append({
            "group": name, "n": n,
            "勝率%": round((g.着順 == 1).mean() * 100, 1),
            "単ROI%": round(g.tan_ret.mean(), 1),
            "単95%CI": f"[{tan_lo},{tan_hi}]",
            "複勝率%": round((g.fuku_ret > 0).mean() * 100, 1),
            "複ROI%": round(g.fuku_ret.mean(), 1),
            "複95%CI": f"[{fk_lo},{fk_hi}]",
        })
    return pd.DataFrame(out)

for per in ["valid(2023)", "test(2024-25)", "train(<=2022)"]:
    sub = df[df.period == per]
    if len(sub) == 0: continue
    print(f"\n================ {per}  (n={len(sub):,}) ================", flush=True)
    print(roi_rows(sub).to_string(index=False), flush=True)

# hidden>=1 vs hidden==0 差分の有意性 (test)
print("\n=== 差分検定: hidden>=1 − hidden==0 (test 2024-25) ===", flush=True)
sub = df[df.period == "test(2024-25)"]
g1 = sub[sub.kako5_hidden_good_count >= 1]
g0 = sub[sub.kako5_hidden_good_count == 0]
for lbl, col in [("単勝", "tan_ret"), ("複勝", "fuku_ret")]:
    a = g1[col].to_numpy(float); b = g0[col].to_numpy(float)
    diffs = (rng.choice(a, size=(4000, len(a)), replace=True).mean(1)
             - rng.choice(b, size=(4000, len(b)), replace=True).mean(1))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_gt0 = (diffs > 0).mean()
    print(f" {lbl}ROI差 = {a.mean()-b.mean():+.1f}pt  95%CI[{lo:+.1f},{hi:+.1f}]  P(diff>0)={p_gt0:.3f}", flush=True)

print("\n[控除率] 単勝/複勝とも母集団ALLがおよそ控除率水準。"
      " hidden群がALL/0群を超え、かつCIが0をまたがなければ『未織り込みの妙味』。", flush=True)
