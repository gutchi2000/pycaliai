# -*- coding: utf-8 -*-
"""
build_tanpuk_all_drift.py — 全馬の 朝 9 時 / 発走前スナップ / 確定 の単勝オッズを 1 表にし、動き × オッズ帯の回収率を出す (2026-11-12)
入力: data/Time _series_odds/TANPUK_*.csv、data/_joint/mkt_horses.parquet。出力: data/_joint/tanpuk_all_drift.parquet
発走前スナップ = 当日の区分 1 のうち確定の 15 分以上前で最も遅いもの (発走の約 26-30 分前)。
実行: python -m analysis.build_tanpuk_all_drift
"""
import glob, re, sys
from pathlib import Path
import numpy as np, pandas as pd
BASE = Path(__file__).resolve().parents[1]
ND = re.compile(r"[^0-9]")
tp = pd.concat([pd.read_csv(f, encoding="cp932", low_memory=False) for f in sorted(glob.glob(str(BASE / "data" / "Time _series_odds" / "TANPUK_*.csv")))], ignore_index=True)
c = list(tp.columns)
tp["rid"] = tp[c[0]].astype(str).map(lambda s: ND.sub("", s)[:16]); tp = tp[tp.rid.str.len() == 16]
tp["year"] = tp.rid.str[:4].astype(int); tp = tp[tp.year >= 2013]
tm = pd.to_numeric(tp[c[2]], errors="coerce").fillna(0).astype(np.int64)
tp["ts"] = pd.to_datetime(dict(year=tp.year, month=tm // 1000000, day=(tm // 10000) % 100, hour=(tm // 100) % 100, minute=tm % 100), errors="coerce")
tp["kb"] = pd.to_numeric(tp[c[1]], errors="coerce"); tp["rd"] = pd.to_datetime(tp.rid.str[:8], format="%Y%m%d", errors="coerce")
fin = tp[tp.kb == 4].drop_duplicates("rid", keep="last").set_index("rid")
mid = tp[(tp.kb == 1) & (tp.ts.dt.normalize() == tp.rd)].join(fin[["ts"]].rename(columns={"ts": "tf"}), on="rid")
mid["gap"] = (mid.tf - mid.ts).dt.total_seconds() / 60
mid = mid[mid.gap >= 15].sort_values(["rid", "ts"]).drop_duplicates("rid", keep="last").set_index("rid")
rids = mid.index.intersection(fin.index)
tcols = {int(str(x).strip()[:-1]): x for x in c if str(x).strip().endswith("単") and str(x).strip()[:-1].isdigit()}
def long(df, name):
    m = df.loc[rids, [tcols[i] for i in sorted(tcols)]].apply(pd.to_numeric, errors="coerce"); m.columns = sorted(tcols)
    s = m.stack().rename(name).reset_index(); s.columns = ["rid", "ban", name]; return s
L = long(mid, "o_pre").merge(long(fin, "o_fin"), on=["rid", "ban"])
mk = pd.read_parquet(BASE / "data" / "_joint" / "mkt_horses.parquet", columns=["rid", "ban", "n", "o_win9", "is_win", "top3", "fpay", "year"])
d = L.merge(mk, on=["rid", "ban"]); d = d[(d.o_pre > 0) & (d.o_fin > 0) & (d.o_win9 > 0) & (d.n >= 8)]
d["tan_pay"] = d.is_win * d.o_fin * 100; d["fuku_pay"] = d.fpay.fillna(0)
d["drift"] = np.log(d.o_pre / d.o_win9)
d["per"] = pd.cut(d.year, [2012, 2018, 2022, 2025], labels=["13-18", "19-22", "23-25"])
d["band"] = pd.cut(d.o_pre, [0, 1.9, 2.9, 4.9, 9.9, 19.9, 49.9, 1e9], labels=["-1.9", "2-2.9", "3-4.9", "5-9.9", "10-19", "20-49", "50+"])
d["dr"] = pd.cut(d.drift, [-9, -0.3, -0.12, 0.12, 0.3, 9], labels=["大きく売れた", "売れた", "横ばい", "売れず", "大きく売れず"])
print(len(d), "horses")
for name, col in (("単勝", "tan_pay"), ("複勝", "fuku_pay")):
    g = d.groupby(["band", "dr", "per"], observed=True)[col].agg(["size", "mean"])
    t = g["mean"].unstack("per").round(1); n = g["size"].unstack("per")
    t["n(13-18/19-22/23-25)"] = n.astype(int).astype(str).agg("/".join, axis=1)
    t["min"] = g["mean"].unstack("per").min(axis=1).round(1)
    print(f"\n=== {name} 回収率: 発走前オッズ帯 × 朝からの動き ==="); print(t.to_string())
d[["rid", "ban", "o_win9", "o_pre", "o_fin", "is_win", "top3", "tan_pay", "fuku_pay", "year", "n"]].to_parquet(BASE / "data" / "_joint" / "tanpuk_all_drift.parquet")
