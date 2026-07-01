# -*- coding: utf-8 -*-
"""kaiko_demo.py — AI回顧デモ: ◎成績の突合 + 自信を持って外したレースのSHAP解剖。"""
import io,sys,joblib,numpy as np,pandas as pd,shap
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8")
B=joblib.load("models/unified_rank_v6.pkl"); model,feats,encs=B["model"],B["feature_cols"],B["encoders"]
RID="レースID(新/馬番無)"
need=list(dict.fromkeys(feats+[RID,"馬番","馬名","着順","split","クラス名","日付"]))
df=pd.read_csv("data/master_v2_20130105-20251228.csv",encoding="utf-8-sig",usecols=lambda c:c in need,low_memory=False)
df["着順"]=pd.to_numeric(df["着順"],errors="coerce")
te=df[(df["split"]=="test")&df["着順"].notna()].copy()
# 直近2000レースに絞る（高速化）
last=sorted(te[RID].unique())[-2000:]
te=te[te[RID].isin(last)].copy()
def enc(d):
    d=d.copy()
    for c,le in encs.items():
        if c in d.columns:
            v=d[c].astype(str).fillna("__NaN__"); d[c]=le.transform(v.where(v.isin(set(le.classes_)),"__NaN__"))
    return d[feats].apply(pd.to_numeric,errors="coerce").fillna(-9999)
te["_X_idx"]=range(len(te))
X=enc(te); te["score"]=model.predict(X)
# レース毎の◎=最高score
te["rank"]=te.groupby(RID)["score"].rank(ascending=False,method="first")
hon=te[te["rank"]==1].copy()
print(f"=== 回顧: test直近 {hon[RID].nunique()}レース ===")
print(f"◎(モデル最上位) 複勝率(3着内)= {(hon['着順']<=3).mean()*100:.1f}%  勝率= {(hon['着順']==1).mean()*100:.1f}%")
# クラス別 ◎複勝率（→ compute_bets/class_prior へ feed）
print("\n[クラス別 ◎複勝率]（賭け規律に反映する材料）")
g=hon.groupby("クラス名").agg(n=("着順","size"),fuku=("着順",lambda s:(s<=3).mean()))
for c,r in g[g["n"]>=20].sort_values("fuku").iterrows():
    print(f"   {c}: {r['fuku']*100:5.1f}%  (n={int(r['n'])})")
# 自信を持って外したレース = ◎scoreが高いのに着順悪い
hon["miss"]=hon["score"]-(6-hon["着順"].clip(upper=6))*0.3
worst=hon.sort_values("score",ascending=False)
worst=worst[worst["着順"]>=8].head(1).iloc[0]
rid=worst[RID]; race=te[te[RID]==rid].copy()
expl=shap.TreeExplainer(model); sv=np.array(expl.shap_values(enc(race)))
race=race.reset_index(drop=True)
hon_i=race[race["馬番"]==worst["馬番"]].index[0]
win=race[race["着順"]==1]; win_i=win.index[0] if len(win) else None
def drv(i): return " / ".join(f"{f}{'+' if v>0 else ''}{v:.2f}" for f,v in sorted(zip(feats,sv[i]),key=lambda x:-abs(x[1]))[:5])
print(f"\n[自信を持って外したレース] {rid}")
print(f"  ◎ {worst['馬番']}番 {worst['馬名']} → 実{int(worst['着順'])}着  (score {worst['score']:+.2f})")
print(f"     モデルが推した根拠: {drv(hon_i)}")
if win_i is not None:
    print(f"  実際の勝ち馬 {race.loc[win_i,'馬番']}番 {race.loc[win_i,'馬名']} (score {race.loc[win_i,'score']:+.2f})")
    print(f"     モデルが見落とした要素: {drv(win_i)}")
print("\n→ 次につなげる: この種の取りこぼしが系統的なら (a)賭けで該当条件を薄く (b)calibrator再fit (c)特徴の見直し材料")
