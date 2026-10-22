# 構造確認のみ: スナップの時刻構成。乖離量は計算しない。
import pandas as pd, numpy as np, glob, re, sys
sys.stdout.reconfigure(encoding="utf-8")
fs=[f for f in sorted(glob.glob("data/Time _series_odds/TANPUK_20[12]*.csv")) if "2011" not in f]
d=pd.concat([pd.read_csv(f,encoding="cp932",dtype=str) for f in fs],ignore_index=True).drop_duplicates()
c=list(d.columns); RID,KB,TM=c[0],c[1],c[2]
d["rid"]=d[RID].str.strip(); d["kb"]=d[KB].astype(int); d["tm"]=d[TM].astype(int)
d["same"]=(d.tm//10000)==d.rid.str[4:8].astype(int); d["hhmm"]=d.tm%10000
k1=d[d.kb==1]
print(pd.crosstab(k1.same, pd.cut(k1.hhmm,[0,600,930,1000,1200,1700,2400])))
s=k1.groupby("rid").agg(nprev=("same",lambda x:(~x).sum()),nsame=("same","sum"))
print(s.value_counts().head(10))
