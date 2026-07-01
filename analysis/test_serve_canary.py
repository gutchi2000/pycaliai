# -*- coding: utf-8 -*-
"""test_serve_canary.py — 全特徴カナリアの動作検証 (健全週=合格 / 故障注入=検知)。"""
import sys, io, json
from pathlib import Path
import numpy as np, joblib
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
BASE = Path(r"E:\PyCaLiAI"); sys.path.insert(0, str(BASE))
from predict_weekly import parse_csv

_REN = {"前走補正":"prev_hosei","前走補9":"prev_hosei9",
        "trn_hanro_4f":"trnH_Time1","trn_hanro_3f":"trnH_Time2","trn_hanro_2f":"trnH_Time3","trn_hanro_1f":"trnH_Time4",
        "trn_hanro_lap1":"trnH_Lap1","trn_hanro_lap2":"trnH_Lap2","trn_hanro_lap3":"trnH_Lap3","trn_hanro_lap4":"trnH_Lap4",
        "trn_hanro_days":"trnH_days_ago",
        "trn_wc_5f":"trnW_5F","trn_wc_4f":"trnW_4F","trn_wc_3f":"trnW_3F",
        "trn_wc_lap1":"trnW_Lap1","trn_wc_lap2":"trnW_Lap2","trn_wc_lap3":"trnW_Lap3","trn_wc_days":"trnW_days_ago"}
feats = joblib.load(BASE/"models/unified_rank_v6.pkl")["feature_cols"]
base_cov = json.loads((BASE/"data/serve_feature_baseline.json").read_text(encoding="utf-8"))["baseline_cov"]

def canary(df):
    mon, deaths = 0, []
    for col in feats:
        exp = base_cov.get(col)
        if exp is None or exp < 0.40: continue
        mon += 1
        cur = float(df[col].notna().mean()) if col in df.columns else 0.0
        if cur < 0.20 and cur < exp*0.40:
            deaths.append(f"{col}({exp*100:.0f}%→{cur*100:.0f}%)")
    return mon, deaths

df = parse_csv(BASE/"data/weekly/20260613.csv")
df = df.rename(columns={k:v for k,v in _REN.items() if k in df.columns and v in feats and v not in df.columns})

# (a) 健全週
mon, deaths = canary(df)
print(f"[健全週 20260613] 監視{mon}特徴 / 無言死 {len(deaths)} → {'✅合格(push継続)' if not deaths else '❌'+str(deaths)}")

# (b) 故障注入: 補正+調教+kako5代表を全NaNに (= リネーム破綻/列ズレを模擬)
df2 = df.copy()
for c in ["prev_hosei","prev_hosei9","trnH_Time1","kako5_avg_pos","jockey_fuku30"]:
    if c in df2.columns: df2[c] = np.nan
mon2, deaths2 = canary(df2)
print(f"[故障注入 5特徴NaN] 監視{mon2}特徴 / 無言死 {len(deaths2)} → "
      f"{'✅検知(push停止)' if deaths2 else '❌見逃し'}")
for d in deaths2: print(f"      捕捉: {d}")
