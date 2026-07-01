# -*- coding: utf-8 -*-
"""
learn_bet_v5_clv.py — 「締切直前オッズ(遅い金/closing line)は超えられるか」CLV検証 (leak-safe)
================================================================================
発見: 時系列オッズCSVには当日・発走前のスナップが既存(区分4で2024-25=6,909R)。
      build_odds_traj_feats が9時で捨ててただけ。再エクスポート不要。
落とし穴: キャプチャは全レース同時刻 → 各レースの発走時刻で T-10 カット必須
          (早いレースの午後スナップは発走後=リーク)。master の発走時刻で厳密にカット。

検証(2025 OOS):
  1. 9時価格 vs 直前価格(T-10) の予測力(AUC/logloss) — closing line は鋭いか
  2. ★CLV: [p_win + 直前価格] は 直前価格単独 を超えるか = 我々は締切盤面を出し抜けるか
  3. drift=log(直前/9時) を特徴に足して増分あるか
実行: PYTHONUTF8=1 python learn_bet_v5_clv.py
"""
from __future__ import annotations
import sys, re
from pathlib import Path
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
BASE = Path(__file__).parent

def hhmm_of(x):
    d=re.sub(r"\D","",str(x))
    if len(d)>=4: return int(d[:4])
    return None

# ---- 1) 発走時刻 per race (master) ----
from audit_ev_bin_roi import MASTER_CSV, COL_RID, _rid16
m=pd.read_csv(MASTER_CSV,encoding="utf-8-sig",usecols=[COL_RID,"発走時刻"],low_memory=False)
m["rid"]=m[COL_RID].map(_rid16)
m["post"]=m["発走時刻"].map(hhmm_of)
post=m.dropna(subset=["post"]).groupby("rid")["post"].first().to_dict()
print(f"発走時刻 races={len(post):,}")

# ---- 2) 時系列オッズ → 直前(T-10前)単勝オッズ per (rid,ban) ----
tp=pd.read_csv(BASE/"data/Time _series_odds/TANPUK_20210105-20251228.csv",encoding="cp932",low_memory=False)
RID,KB,TM=tp.columns[0],tp.columns[1],tp.columns[2]
tan_cols={int(re.match(r"^\s*(\d+)\s*単\s*$",c).group(1)):c for c in tp.columns if re.match(r"^\s*(\d+)\s*単\s*$",c)}
tp["r"]=tp[RID].astype(str).str.replace(r"\D","",regex=True).str[:16]
tp["kb"]=pd.to_numeric(tp[KB],errors="coerce")
t=pd.to_numeric(tp[TM],errors="coerce")
tp["mmdd"]=(t//10000); tp["hhmm"]=(t%10000)
tp=tp[tp["r"].str.len()==16]
late={}  # (rid,ban)->odds at last snapshot <= post-10 (same day, 確定除く)
n_late_after9=0; races_done=0
for rid,g in tp.groupby("r",sort=False):
    if rid[:4] not in ("2024","2025"): continue
    p=post.get(rid)
    if p is None: continue
    rmmdd=int(rid[4:8])
    cut=p-10 if p%100>=10 else p-50  # T-10 (分繰下げ簡易)
    sd=g[(g["mmdd"]==rmmdd) & (g["kb"]!=4) & (g["hhmm"]<=cut)]
    if len(sd)==0: continue
    row=sd.sort_values("hhmm").iloc[-1]
    races_done+=1
    if int(row["hhmm"])>900: n_late_after9+=1
    for b,c in tan_cols.items():
        v=pd.to_numeric(pd.Series([row[c]]),errors="coerce").iloc[0]
        if pd.notna(v) and v>1.0: late[(rid,b)]=float(v)
print(f"直前価格を作れたレース={races_done:,}  うち9時より後の盤面={n_late_after9:,} ({n_late_after9/max(races_done,1)*100:.0f}%)")

# ---- 3) bet table に直前価格を結合 ----
R=pd.read_parquet(BASE/"data/_xai/bet_dyn_rows.parquet")  # rid,ban,p_win,odds(9時),won,tanpay
R["ban"]=R["ban"].astype(int)
R["o_late"]=[late.get((r,b)) for r,b in zip(R["rid"],R["ban"])]
cov=R["o_late"].notna().mean()
print(f"直前価格カバレッジ(馬単位)={cov*100:.0f}%")
R=R[R["o_late"].notna()].copy()
R["q9"]=(1/R["odds"]); R["q9"]/=R.groupby("rid")["q9"].transform("sum")
R["qL"]=(1/R["o_late"]); R["qL"]/=R.groupby("rid")["qL"].transform("sum")
R["drift"]=np.log(R["o_late"])-np.log(R["odds"])
tr=R[R.year==2024]; ev=R[R.year==2025]
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
def lg(p,e=1e-6): p=np.clip(p,e,1-e); return np.log(p/(1-p))
print(f"\nfit2024={len(tr):,} eval2025={len(ev):,} races2025={ev.rid.nunique():,} 勝率={ev.won.mean():.3f}")

# 1. 価格の鋭さ比較
print("\n[1] 価格単独の予測力 (2025 OOS):")
for nm,c in [("9時価格",'q9'),("直前価格(T-10)",'qL')]:
    print(f"  {nm:16s} logloss={log_loss(ev.won,np.clip(ev[c],1e-6,1-1e-6)):.4f} AUC={roc_auc_score(ev.won,ev[c]):.4f}")

# 2. CLV: モデルは直前価格を超えるか
base_ll=log_loss(ev.won,np.clip(ev.qL,1e-6,1-1e-6)); base_auc=roc_auc_score(ev.won,ev.qL)
print(f"\n[2] ★CLV検証: 締切直前盤面を超えられるか (baseline=直前価格 logloss={base_ll:.4f} AUC={base_auc:.4f})")
def benter(cols):
    X=np.column_stack([lg(tr[c]) if c.startswith('q') else tr[c] for c in cols])
    Xe=np.column_stack([lg(ev[c]) if c.startswith('q') else ev[c] for c in cols])
    clf=LogisticRegression(max_iter=2000).fit(X,tr.won); return clf,clf.predict_proba(Xe)[:,1]
for nm,cols in [("p_win+直前価格",["p_win","qL"]),
                ("p_win+直前+drift",["p_win","qL","drift"]),
                ("p_win+9時+直前",["p_win","q9","qL"])]:
    clf,p=benter(cols)
    coef=" ".join(f"{c}:{w:+.2f}" for c,w in zip(cols,clf.coef_[0]))
    print(f"  {nm:18s} logloss={log_loss(ev.won,p):.4f} vs直前={base_ll-log_loss(ev.won,p):+.4f} AUC={roc_auc_score(ev.won,p):.4f}  [{coef}]")

# 3. EV選抜ROI: 直前価格を実オッズに使った場合の単勝ROI
print("\n[3] 直前価格でEV選抜→単勝ROI(2025, 決済=確定単勝tanpay):")
clf,p=benter(["p_win","qL"])
E=ev.copy(); E["pb"]=p/ pd.Series(p,index=E.index).groupby(E["rid"]).transform("sum"); E["ev"]=E["pb"]*E["o_late"]
for thr in [1.0,1.1,1.2]:
    s=E[E["ev"]>=thr]; n=len(s)
    if n==0: print(f"  EV≥{thr}: n0"); continue
    r=s.tanpay.values; roi=r.mean()/100; se=r.std(ddof=1)/np.sqrt(n)/100 if n>1 else 0
    print(f"  EV≥{thr}: n={n} hit={(r>0).mean()*100:.0f}% ROI={roi*100:.0f}% [{(roi-1.96*se)*100:.0f},{(roi+1.96*se)*100:.0f}]")
