# -*- coding: utf-8 -*-
"""
make_timeflow.py — タイム分析(レース単位ペース)を各馬の過去走に集計 (leak-safe)
走間ファイルの(対象rid18, 過去走rid18)リンクで、過去走が行われたレースの
"流れ"(補9/RPCI/PCI3/前後半バランス/F平均/秒速/ラップ分散/最速3F/A-3F差)を集約。
走間の深い履歴(馬自身の上り)に無い「レース全体のペース文脈」=純新規。
出力: data/_xai/timeflow_feats.parquet (key=対象rid18)
"""
import sys, glob
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

def num(s): return pd.to_numeric(s.astype(str).str.strip(), errors="coerce")

# --- タイム分析: レース単位の流れ特徴 (index指定) ---
TIDX={142:"rid16",22:"ho9_r",42:"rpci_r",43:"pci3_r",32:"f3",33:"b3",
      29:"favg",30:"spd",51:"max3f",57:"a3fd",60:"l1",61:"l2",62:"l3",63:"l4",64:"l5"}
t=pd.read_csv("data/タイム分析/タイム分析_2016-2025.csv",encoding="cp932",usecols=list(TIDX),low_memory=False)
t.columns=[TIDX[i] for i in sorted(TIDX)]
t["rid16"]=t["rid16"].astype(str).str.replace(r"\D","",regex=True).str[:16]
for c in ["ho9_r","rpci_r","pci3_r","f3","b3","favg","spd","max3f","a3fd","l1","l2","l3","l4","l5"]:
    t[c]=num(t[c])
t["pace_bal"]=t["b3"]-t["f3"]                       # 後半-前半 (正=前傾)
t["lap_std"]=t[["l1","l2","l3","l4","l5"]].std(axis=1)
flow=t[["rid16","ho9_r","rpci_r","pci3_r","pace_bal","favg","spd","max3f","a3fd","lap_std"]].drop_duplicates("rid16")
print("タイム分析 レース数:",len(flow))

# --- 走間: (対象rid18, 過去走rid18) のリンク ---
pairs=[]
for f in sorted(glob.glob("data/走間分析/*.csv")):
    d=pd.read_csv(f,encoding="cp932",usecols=[151,249],low_memory=False)  # 151=過去走rid18, 249=対象rid18
    d.columns=["past18","tgt18"]
    d["past16"]=d["past18"].astype(str).str.replace(r"\D","",regex=True).str[:16]
    d["tgt18"]=d["tgt18"].astype(str).str.replace(r"\D","",regex=True)
    pairs.append(d[["tgt18","past16"]])
    print(f"  link {f.split(chr(92))[-1][:20]} rows={len(d):,}")
P=pd.concat(pairs,ignore_index=True)
P=P.merge(flow,left_on="past16",right_on="rid16",how="inner")
print("リンク後(過去走×流れ):",f"{len(P):,}")

agg=P.groupby("tgt18").agg(
    tf_ho9_mean=("ho9_r","mean"), tf_rpci_mean=("rpci_r","mean"), tf_pci3_mean=("pci3_r","mean"),
    tf_pacebal_mean=("pace_bal","mean"), tf_favg_mean=("favg","mean"), tf_spd_mean=("spd","mean"),
    tf_max3f_mean=("max3f","mean"), tf_a3fd_mean=("a3fd","mean"), tf_lapstd_mean=("lap_std","mean"),
    tf_spd_max=("spd","max"), tf_n=("rid16","size"),
).reset_index()
agg=agg.rename(columns={"tgt18":"rid18"})
agg.to_parquet("data/_xai/timeflow_feats.parquet",index=False)
print(f"[saved] {len(agg):,} target-horses, {agg.shape[1]} cols → data/_xai/timeflow_feats.parquet")
print(agg.describe().round(2).T[["mean","min","max"]].to_string())
