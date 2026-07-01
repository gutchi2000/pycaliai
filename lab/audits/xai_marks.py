# -*- coding: utf-8 -*-
"""xai_marks.py — v6印モデルの SHAP 説明。なぜこの馬が上位(=◎)かを特徴寄与で分解。"""
import io,sys,joblib,numpy as np,pandas as pd,shap
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8")
B=joblib.load("models/unified_rank_v6.pkl")
model,feats,encs=B["model"],B["feature_cols"],B["encoders"]
RID="レースID(新/馬番無)"
need=list(dict.fromkeys(feats+[RID,"馬番","馬名","着順","split"]))
df=pd.read_csv("data/master_v2_20130105-20251228.csv",encoding="utf-8-sig",
               usecols=lambda c:c in need,low_memory=False)
te=df[df["split"]=="test"].copy()
# 12-16頭の test レースを1つ
vc=te[RID].value_counts(); rid=vc[(vc>=14)&(vc<=16)].index[0]
race=te[te[RID]==rid].copy()
d=race.copy()
for c,le in encs.items():
    if c in d.columns:
        v=d[c].astype(str).fillna("__NaN__")
        d[c]=le.transform(v.where(v.isin(set(le.classes_)),"__NaN__"))
X=d[feats].apply(pd.to_numeric,errors="coerce").fillna(-9999)
expl=shap.TreeExplainer(model); sv=np.array(expl.shap_values(X))
score=model.predict(X)
order=np.argsort(-score)
MARK=["◎","〇","▲","△","△"]
print(f"=== SHAP 説明: race {rid}  {len(race)}頭 ===")
names=race["馬名"].values; bans=race["馬番"].values; jun=race["着順"].values
for rk,idx in enumerate(order[:5]):
    contrib=sorted(zip(feats,sv[idx]),key=lambda x:-abs(x[1]))[:6]
    drv=" / ".join(f"{f}{'+' if v>0 else ''}{v:.2f}" for f,v in contrib)
    print(f"\n{MARK[rk]} {bans[idx]:>2}番 {names[idx]}  (実着{int(jun[idx])}着, score {score[idx]:+.2f})")
    print(f"   駆動: {drv}")
print(f"\n[base value] {expl.expected_value:.2f}  (全特徴ゼロ寄与時のスコア)")
