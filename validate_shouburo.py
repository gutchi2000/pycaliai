"""
勝負所ゲート検証 — 単勝 ROI を train/valid/test で確認 (leak-safe 9時オッズ, 全期間2013-25)

問い: 実弾(6週)で見えた①EV逆相関(高EVほど低ROI)②穴の罠(高オッズ死)③◎単勝が良い、
      が 2013-25 backtest の valid/test 両方で robust に成り立つか。

leak規約: EV=当日9時オッズ(指時系2)。決済=確定単勝配当。markは v6 score 順位。
9時オッズ源: E:\競馬過去走データ\kekka_..._v2.csv (全頭全期間)
出力: reports/validate_shouburo.json
"""
from __future__ import annotations
import json
import re
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import pl_probs as PL

BASE = Path(__file__).parent
MASTER = BASE / "data/master_v2_20130105-20251228.csv"
KEKKA_V2 = Path(r"E:\競馬過去走データ\kekka_20130105-20251228_v2.csv")
OUT = BASE / "reports/validate_shouburo.json"
COL_RID = "レースID(新/馬番無)"
COL_JYUN = "着順"
COL_BAN = "馬番"

MARK = {0: "◎", 1: "〇", 2: "▲", 3: "△", 4: "△"}
EV_BINS = [0, 0.7, 0.9, 1.1, 1.5, 1e18]
EV_LAB = ["<0.7", "0.7-0.9", "0.9-1.1", "1.1-1.5", "1.5+"]
ODDS_BINS = [0, 5, 15, 1e18]
ODDS_LAB = ["人気≤5", "中5-15", "穴15+"]


def rid16(x):
    return re.sub(r"\D", "", str(x))[:16]


def band(v, bins, lab):
    return lab[int(np.clip(np.digitize([v], bins)[0] - 1, 0, len(lab) - 1))]


def load_odds9():
    """{(rid16, ban): (odds9, chaku, tanpay)} ; odds9=当日9時単勝(指時系2)."""
    df = pd.read_csv(KEKKA_V2, encoding="cp932", low_memory=False)
    c = list(df.columns)
    c_ban, c_chaku, c_rid, c_tanpay, c_o9 = c[4], c[6], c[7], c[8], c[17]
    rids = df[c_rid].map(rid16).values
    bans = pd.to_numeric(df[c_ban], errors="coerce").values
    o9s = pd.to_numeric(df[c_o9], errors="coerce").values
    cks = pd.to_numeric(df[c_chaku], errors="coerce").values
    pays = pd.to_numeric(df[c_tanpay], errors="coerce").values
    out = {}
    for rid, ban, o9, ck, pay in zip(rids, bans, o9s, cks, pays):
        if pd.isna(ban):
            continue
        out[(rid, int(ban))] = (o9, ck, pay)
    return out


def main():
    print("[load] v6 model + tansho calibrator")
    b = joblib.load(BASE / "models/unified_rank_v6.pkl")
    model, feats, encs = b["model"], b["feature_cols"], b["encoders"]
    cal = joblib.load(BASE / "models/pl_calibrators_v6.pkl")
    cal = cal.get("calibrators", cal)

    print("[load] 9時オッズ (kekka_v2, 全期間)")
    odds = load_odds9()
    print(f"  odds rows: {len(odds):,}")

    print("[load+score] master_v2 (全split)")
    df = pd.read_csv(MASTER, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"])
    for cc, le in encs.items():
        if cc not in df.columns:
            continue
        v = df[cc].astype(str).fillna("__NaN__")
        df[cc] = le.transform(v.where(v.isin(set(le.classes_)), "__NaN__"))
    X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    df["_score"] = model.predict(X)

    # agg[split][dim][cell] = {stake, ret, n, hit}
    def cell():
        return {"stake": 0.0, "ret": 0.0, "n": 0, "hit": 0}
    agg = {s: {"mark": defaultdict(cell), "ev": defaultdict(cell), "odds": defaultdict(cell)}
           for s in ["train", "valid", "test"]}

    n_races = defaultdict(int)
    for rid, g in df.groupby(COL_RID, sort=False):
        if len(g) < 5:
            continue
        split = g["split"].iloc[0]
        if split not in agg:
            continue
        g = g.sort_values("_score", ascending=False).reset_index(drop=True)  # 強い順
        w = PL.pl_weights(g["_score"].values)
        p_cal = cal["tansho"].predict(PL.all_tansho(w))
        r16 = rid16(rid)
        n_races[split] += 1
        ban = g[COL_BAN].astype(int).values
        for rank in range(len(g)):
            od = odds.get((r16, int(ban[rank])))
            if od is None or pd.isna(od[0]) or od[0] <= 1.0:
                continue
            o9, chaku, pay = od
            ev = float(p_cal[rank]) * float(o9)
            won = (not pd.isna(chaku)) and int(chaku) == 1
            ret = float(pay) if (won and not pd.isna(pay)) else 0.0
            mk = MARK.get(rank, "無")
            for dim, key in (("mark", mk), ("ev", band(ev, EV_BINS, EV_LAB)),
                             ("odds", band(o9, ODDS_BINS, ODDS_LAB))):
                c = agg[split][dim][key]
                c["stake"] += 100.0
                c["ret"] += ret
                c["n"] += 1
                c["hit"] += int(won)

    def roi(c):
        return round(c["ret"] / c["stake"] * 100, 1) if c["stake"] else 0.0

    result = {"n_races": dict(n_races), "by": {}}
    for dim, order in (("mark", ["◎", "〇", "▲", "△", "無"]),
                       ("ev", EV_LAB), ("odds", ODDS_LAB)):
        result["by"][dim] = {}
        for key in order:
            row = {}
            for s in ["train", "valid", "test"]:
                c = agg[s][dim].get(key)
                if c and c["n"] > 0:
                    row[s] = {"n": c["n"], "roi": roi(c), "hit": round(c["hit"] / c["n"] * 100, 1)}
            if row:
                result["by"][dim][key] = row

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    # console
    print(f"\n{'='*78}\n単勝 ROI: train(≤22) / valid(23) / test(24-25)  全頭9時オッズ\n{'='*78}")
    print("races:", dict(n_races))
    for dim, order in (("印(score順位)", "mark"), ("モデルEV帯", "ev"), ("9時オッズ帯", "odds")):
        key2 = order
        print(f"\n【{dim}】")
        print(f"  {'cell':<9}{'train ROI/n':>16}{'valid ROI/n':>16}{'test ROI/n':>16}")
        for k, row in result["by"][key2].items():
            def fmt(s):
                r = row.get(s)
                return f"{r['roi']:.0f}%/{r['n']}" if r else "-"
            print(f"  {k:<9}{fmt('train'):>16}{fmt('valid'):>16}{fmt('test'):>16}")
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
