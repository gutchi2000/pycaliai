# -*- coding: utf-8 -*-
"""
learn_bet_customloss.py — カスタム損失(利益直接最適化) leak-safe
================================================================================
標準損失(lambdarank/logloss/tweedie)は「当てる/回帰精度」を最適化するだけ。
ここでは **馬券の利益そのもの** を微分可能損失にする:
  レース内で softmax(score/τ) = stake配分 → 実現複勝回収 Σ pᵢ·roi_targetᵢ を最大化。
  (τ→0 で argmax=トップ1点買いに近づく)
= 「当てる」でなく「儲ける」を直接学習。これがカスタム損失の本命。

学習=split'train'(≤2022) / 評価=2025 OOS。roi_target=複勝配当/100=実現複勝回収。
比較対象: 着順83 / 確率84 / 回収(tweedie)82 (= 指数1位の複勝回収率)
実行: PYTHONUTF8=1 python learn_bet_customloss.py
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import joblib, torch, torch.nn as nn
from audit_ev_bin_roi import MASTER_CSV, COL_RID, COL_JYUN

torch.manual_seed(42); np.random.seed(42)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev)

b = joblib.load("models/unified_rank_v6.pkl"); feats, encs = b["feature_cols"], b["encoders"]
print("[load] master")
df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]).copy()
df["year"] = df[COL_RID].astype(str).str[:4]
df["fukusho_flag"] = pd.to_numeric(df["fukusho_flag"], errors="coerce").fillna(0)
df["roi_target"] = pd.to_numeric(df["roi_target"], errors="coerce").fillna(0)
for c, le in encs.items():
    if c in df.columns:
        v = df[c].astype(str).fillna("__NaN__")
        df[c] = le.transform(v.where(v.isin(set(le.classes_)), "__NaN__"))
X = df[feats].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
med = X[df["split"] == "train"].median()
X = X.fillna(med).fillna(0.0)
mu = X[df["split"] == "train"].mean(); sd = X[df["split"] == "train"].std().replace(0, 1)
sd = sd.fillna(1.0); mu = mu.fillna(0.0)
Xz = ((X - mu) / sd).clip(-8, 8).values.astype("float32")
Xz = np.nan_to_num(Xz, nan=0.0, posinf=8.0, neginf=-8.0)
print("NaN in Xz:", int(np.isnan(Xz).sum()), " inf:", int(np.isinf(Xz).sum()))
df = df.reset_index(drop=True)

def make_groups(mask):
    idx = np.where(mask.values)[0]
    sub = df.loc[idx]
    M = 18; F = Xz.shape[1]
    races = []
    for rid, g in sub.groupby(COL_RID, sort=False):
        ii = g.index.values[:M]
        races.append(ii)
    n = len(races)
    Xp = np.zeros((n, M, F), "float32"); roi = np.zeros((n, M), "float32")
    fuk = np.zeros((n, M), "float32"); msk = np.zeros((n, M), "float32")
    for k, ii in enumerate(races):
        m = len(ii)
        Xp[k, :m] = Xz[ii]; roi[k, :m] = df.loc[ii, "roi_target"].values
        fuk[k, :m] = df.loc[ii, "fukusho_flag"].values; msk[k, :m] = 1
    return (torch.tensor(Xp), torch.tensor(roi), torch.tensor(fuk), torch.tensor(msk))

print("[build] race tensors")
Xtr, roitr, _, mtr = make_groups(df["split"] == "train")
Xev, roiev, fukev, mev = make_groups((df["split"] == "test") & (df["year"] == "2025"))
print(f"train races={Xtr.shape[0]:,}  eval2025 races={Xev.shape[0]:,}")

class Net(nn.Module):
    def __init__(s, F):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(F, 128), nn.ReLU(), nn.Dropout(0.2),
                              nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(s, x): return s.net(x).squeeze(-1)   # (B,M)

def portfolio_loss(score, roi, mask, tau):
    score = score.masked_fill(mask == 0, -1e9)
    p = torch.softmax(score / tau, dim=1)            # レース内stake配分
    ret = (p * roi).sum(1)                            # 実現複勝回収/レース
    return -ret.mean()

def evaluate(model):
    model.eval()
    with torch.no_grad():
        sc = model(Xev.to(dev)).cpu()
        sc = sc.masked_fill(mev == 0, -1e9)
        top1 = sc.argmax(1)
        r = roiev[torch.arange(len(top1)), top1]
        f = fukev[torch.arange(len(top1)), top1]
        return f.mean().item()*100, r.mean().item()*100   # 複勝率, 複勝回収

for tau in [1.0, 0.3]:
    model = Net(Xz.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    Xtr_d, roitr_d, mtr_d = Xtr.to(dev), roitr.to(dev), mtr.to(dev)
    nB = 1024; idx = np.arange(Xtr.shape[0])
    print(f"\n[train] custom portfolio loss  tau={tau}")
    for ep in range(40):
        model.train(); np.random.shuffle(idx); tot = 0
        for s0 in range(0, len(idx), nB):
            bi = idx[s0:s0+nB]
            sc = model(Xtr_d[bi])
            loss = portfolio_loss(sc, roitr_d[bi], mtr_d[bi], tau)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += -loss.item()*len(bi)
        if ep % 10 == 9:
            fr, rr = evaluate(model)
            print(f"  ep{ep+1:2d} train複勝回収={tot/len(idx)*100:5.1f}  | 2025 1位 複勝率{fr:.1f}% 回収{rr:.0f}")
    fr, rr = evaluate(model)
    print(f"  >>> tau={tau}: 2025 指数1位 複勝率 {fr:.1f}% / 複勝回収 {rr:.0f}")

print("\n比較(指数1位 複勝回収): 着順83 / 確率84 / 回収tweedie82 / カスタム損失↑")
print("(>100 = 控除率超え=本物のエッジ)")
