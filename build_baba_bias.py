# -*- coding: utf-8 -*-
"""
build_baba_bias.py — クッション値/含水率 × 脚質・枠 の馬場バイアス表を生成。
explain_race.py / jusho_protocol.py のカードに「この馬場なら前残り/外枠有利」を常時表示するための
ルックアップ data/baba_bias.json を作る（記述統計・leak-safe・再生成可能）。

この一戦の脚質(逃げ/先行/中団/後方)・枠番は master(着順/枠番) × 外部結果(脚質) から、
cushion/含水率は data/baba_feats.parquet から。
出力: data/baba_bias.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(__file__).parent
MASTER = BASE / "data/master_v2_20130105-20251228.csv"
KEKKA_EXT = "E:/競馬過去走データ/raw_data/kekka_2010_2025_fix_raceid_v2__keyed.csv"
BABA = BASE / "data/baba_feats.parquet"
OUT = BASE / "data/baba_bias.json"

FRONT = {"逃げ", "先行"}; CLOSE = {"中団", "後方", "ﾏｸﾘ"}

# (metric列, 帯定義[(label, lo, hi)])  ※lo<=v<hi、Noneは無限
SPECS = {
    "芝_cushion":  ("cushion",  [("重い(軟)", None, 8.0), ("標準", 8.0, 9.5), ("軽い(速)", 9.5, None)]),
    "芝_moisture": ("shiba_gp", [("乾", None, 12.0), ("標準", 12.0, 15.0), ("湿", 15.0, None)]),
    "ダ_moisture": ("dirt_gp",  [("乾", None, 4.0), ("標準", 4.0, 9.0), ("湿(道悪)", 9.0, None)]),
}


def load():
    m = pd.read_csv(MASTER, encoding="utf-8-sig", low_memory=False,
                    usecols=["レースID(新/馬番無)", "馬番", "着順", "枠番", "場所", "芝・ダ"])
    m["rid"] = pd.to_numeric(m["レースID(新/馬番無)"], errors="coerce").astype("Int64").astype(str)
    m["ban"] = pd.to_numeric(m["馬番"], errors="coerce")
    m["jyun"] = pd.to_numeric(m["着順"], errors="coerce")
    m["waku"] = pd.to_numeric(m["枠番"], errors="coerce")
    m = m.dropna(subset=["ban", "jyun"])
    e = pd.read_csv(KEKKA_EXT, encoding="utf-8-sig", low_memory=False, usecols=["race_id", "馬番", "脚質"])
    e["rid"] = pd.to_numeric(e["race_id"], errors="coerce").astype("Int64").astype(str)
    e["ban"] = pd.to_numeric(e["馬番"], errors="coerce")
    k = m.merge(e[["rid", "ban", "脚質"]], on=["rid", "ban"], how="inner").dropna(subset=["脚質"])
    k["date"] = k["rid"].str[:8]
    k["win"] = (k["jyun"] == 1).astype(int); k["plc"] = (k["jyun"] <= 3).astype(int)
    k["n"] = k.groupby("rid")["ban"].transform("count")
    k["style"] = np.where(k["脚質"].isin(FRONT), "前", np.where(k["脚質"].isin(CLOSE), "差", "他"))
    k["wk"] = np.where(k["waku"] <= 3, "内", np.where(k["waku"] <= 5, "中", "外"))
    b = pd.read_parquet(BABA); b["date"] = b["日付"].astype(str)
    return k.merge(b[["場所", "date", "cushion", "shiba_gp", "dirt_gp"]], on=["場所", "date"], how="left")


def winrate(s, st):
    x = s[s["脚質"] == st]; return round(x["win"].mean() * 100, 1) if len(x) else None
def plcrate_style(s, st):
    x = s[s["style"] == st]; return x["plc"].mean() * 100 if len(x) else np.nan
def plcrate_wk(s, w):
    x = s[s["wk"] == w]; return x["plc"].mean() * 100 if len(x) else np.nan


def main():
    k = load()
    out = {}
    for key, (col, bands) in SPECS.items():
        surf = "芝" if key.startswith("芝") else "ダ"
        sub = k[(k["芝・ダ"] == surf) & k[col].notna()].copy()
        rows = []
        for label, lo, hi in bands:
            m = sub[col].between(lo if lo is not None else -np.inf, hi if hi is not None else np.inf, inclusive="left")
            s = sub[m]
            if s["rid"].nunique() < 200:
                continue
            big = s[s["n"] >= 12]
            fr, cl = plcrate_style(s, "前"), plcrate_style(s, "差")
            wi, mi, wo = plcrate_wk(big, "内"), plcrate_wk(big, "中"), plcrate_wk(big, "外")
            rows.append({
                "label": label, "lo": lo, "hi": hi, "n_races": int(s["rid"].nunique()),
                "front_edge": round(fr - cl, 1),           # 前複勝率 − 差複勝率
                "nige_win": winrate(s, "逃げ"), "senko_win": winrate(s, "先行"),
                "sashi_win": winrate(s, "中団"), "oikomi_win": winrate(s, "後方"),
                "waku_out_minus_in": round(wo - wi, 1),     # 外枠複勝率 − 内枠複勝率(頭数≥12)
            })
        out[key] = {"metric": col, "bands": rows}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[saved] {OUT}")
    for key, v in out.items():
        print(f"\n{key} ({v['metric']}):")
        for r in v["bands"]:
            print(f"  {r['label']:10} 前残り度{r['front_edge']:+5.1f}pp  逃{r['nige_win']}/先{r['senko_win']}  "
                  f"外-内{r['waku_out_minus_in']:+.1f}pp (n={r['n_races']})")


if __name__ == "__main__":
    main()
