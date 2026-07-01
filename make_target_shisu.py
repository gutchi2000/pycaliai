# -*- coding: utf-8 -*-
"""
make_target_shisu.py — PyCaLiAI の勝率指数を TARGET 外部指数(馬単位CSV)で出力
================================================================================
形式(公式FAQ): 馬単位・CSV形式 = 1行1頭 「レースID(馬番付き),指数,順位」
設定: 馬単位 / CSV形式 / 新仕様レースID / 大きいほど優位 / 順位自動付与可
指数 = v6 PL→calibrator の勝率 p_win × 100 (0〜100, 大きいほど強い)

出力:
  data/target_shisu/pycaliai_shisu_all.csv   ... 全期間1ファイル(レースID18,指数,順位)
  + 年別 data/target_shisu/PCAI_{YYYY}.csv    ... TARGET登録しやすい年分割
実行: PYTHONUTF8=1 python make_target_shisu.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
BASE = Path(__file__).parent
OUT = BASE / "data/target_shisu"; OUT.mkdir(parents=True, exist_ok=True)

import joblib, pl_probs as PL
from audit_ev_bin_roi import MASTER_CSV, COL_RID, COL_BAN

print("[load] v6 model + calibrator")
b = joblib.load(BASE/"models/unified_rank_v6.pkl")
model, feats, encs = b["model"], b["feature_cols"], b["encoders"]
cal = joblib.load(BASE/"models/pl_calibrators_v6.pkl"); cal = cal.get("calibrators", cal)

print("[load] master (全期間)")
df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
# 18桁 馬番付きレースID
RID18 = "レースID(新)"
if RID18 not in df.columns:
    RID18 = [c for c in df.columns if "レースID(新)" in c and "馬番無" not in c][0]
df = df.dropna(subset=[COL_RID, COL_BAN, RID18]).copy()
for c, le in encs.items():
    if c in df.columns:
        v = df[c].astype(str).fillna("__NaN__")
        df[c] = le.transform(v.where(v.isin(set(le.classes_)), "__NaN__"))
X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
df["_score"] = model.predict(X)

print("[calc] PL→p_win 指数 (レース毎)")
rid18, idx, rnk = [], [], []
for rid, g in df.groupby(COL_RID, sort=False):
    s = g["_score"].values
    w = PL.pl_weights(s)
    p = cal["tansho"].predict(PL.all_tansho(w))   # 勝率
    order = (-p).argsort().argsort() + 1          # 1=最強
    ids = g[RID18].astype(str).str.replace(r"\D","",regex=True).values
    for i in range(len(g)):
        rid18.append(ids[i]); idx.append(round(float(p[i])*100, 2)); rnk.append(int(order[i]))
out = pd.DataFrame({"rid18": rid18, "idx": idx, "rank": rnk})
out = out[out["rid18"].str.len() == 18]
allp = OUT/"pycaliai_shisu_all.csv"
out.to_csv(allp, header=False, index=False)
print(f"[saved] {allp}  {len(out):,}行  指数=p_win×100")
# 年分割
out["yr"] = out["rid18"].str[:4]
for yr, g in out.groupby("yr"):
    g[["rid18","idx","rank"]].to_csv(OUT/f"PCAI_{yr}.csv", header=False, index=False)
print(f"[saved] 年別ファイル {out['yr'].nunique()}本 → {OUT}")
print("\nサンプル(先頭5行 形式: レースID18,指数,順位):")
print(out[["rid18","idx","rank"]].head().to_string(index=False, header=False))
