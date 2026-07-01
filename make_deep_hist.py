# -*- coding: utf-8 -*-
"""
make_deep_hist.py — 走間分析5ファイルから「深い履歴」特徴を集計 (leak-safe)
各対象レース×馬(起ＲレースID新 18桁)について、その馬の過去全走(同年+前年, 100%対象より前)を
集計。kako5(過去5走)に無い 補正/補9(スピード指数)履歴・PCI/RPCI・上3F地点差・コメント・通算深度。
出力: data/_xai/deep_hist_feats.parquet  (key=rid18)
実行: PYTHONUTF8=1 python make_deep_hist.py
"""
import sys, glob
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

# 列はインデックス指定(重複列名回避)
IDX = {2:"d", 84:"agari_chiten", 94:"ave3f", 97:"pci", 100:"rpci",
       113:"hosei", 114:"ho9", 157:"fin", 173:"kol", 177:"work",
       202:"t_odds", 249:"rid18", 253:"t_fin"}
def num(s): return pd.to_numeric(s.astype(str).str.strip().str.replace("(","" ).str.replace(")",""), errors="coerce")

parts=[]
for f in sorted(glob.glob("data/走間分析/*.csv")):
    d = pd.read_csv(f, encoding="cp932", usecols=list(IDX), low_memory=False)
    d.columns = [IDX[i] for i in sorted(IDX)]  # usecolsはソート順で返る
    parts.append(d)
    print(f"  read {f.split(chr(92))[-1][:24]} rows={len(d):,}")
df = pd.concat(parts, ignore_index=True)
print("total prior-run rows:", f"{len(df):,}")

for c in ["agari_chiten","ave3f","pci","rpci","hosei","ho9","fin","t_odds","t_fin","d"]:
    df[c] = num(df[c])
df["kol_f"] = (df["kol"].astype(str).str.strip() != "").astype(int)
df["work_f"] = (df["work"].astype(str).str.strip() != "").astype(int)
df["top3"] = (df["fin"] <= 3).astype(float)
df["win"] = (df["fin"] == 1).astype(float)
df = df.sort_values(["rid18","d"])

g = df.groupby("rid18")
def lastv(s): return s.dropna().iloc[-1] if s.notna().any() else np.nan
feat = g.agg(
    n_prior=("d","size"),
    hosei_mean=("hosei","mean"), hosei_max=("hosei","max"),
    ho9_mean=("ho9","mean"), ho9_max=("ho9","max"),
    pci_mean=("pci","mean"), rpci_mean=("rpci","mean"), ave3f_mean=("ave3f","mean"),
    agari_chiten_mean=("agari_chiten","mean"), agari_chiten_min=("agari_chiten","min"),
    prior_best_fin=("fin","min"), prior_top3_rate=("top3","mean"), prior_win_rate=("win","mean"),
    kol_cnt=("kol_f","sum"), work_cnt=("work_f","sum"),
    t_fin=("t_fin","first"), t_odds=("t_odds","first"),
    hosei_last=("hosei", lastv),
)
feat["hosei_trend"] = feat["hosei_last"] - feat["hosei_mean"]
feat = feat.reset_index()
feat["rid18"] = feat["rid18"].astype(str)
feat.to_parquet("data/_xai/deep_hist_feats.parquet", index=False)
print(f"[saved] {len(feat):,} target-horses, {feat.shape[1]} cols → data/_xai/deep_hist_feats.parquet")
print(feat[["n_prior","hosei_mean","hosei_max","prior_top3_rate","kol_cnt","t_fin","t_odds"]].describe().round(2).T[["mean","min","max"]].to_string())
PY = None
