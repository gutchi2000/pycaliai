"""
validate_favgate.py — 「今の印で+にできるか」を探す
◎〇 の 単勝/複勝 ROI を、細かいオッズ帯 と「◎が市場1番人気か(=一致)」で valid/test 別に。
100%超(=現行印で黒字)のセルが robust(valid&test両方)に存在するか?
9時オッズ=指時系2, 決済=確定配当(単/複). 出力: reports/validate_favgate.json
"""
from __future__ import annotations
import json
import re
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import pl_probs as PL  # noqa (import side-effect parity with other scripts)

BASE = Path(__file__).parent
MASTER = BASE / "data/master_v2_20130105-20251228.csv"
KEKKA_V2 = Path(r"E:\競馬過去走データ\kekka_20130105-20251228_v2.csv")
OUT = BASE / "reports/validate_favgate.json"
COL_RID = "レースID(新/馬番無)"
COL_JYUN = "着順"
COL_BAN = "馬番"
OB = [0, 1.5, 2, 2.5, 3.5, 5, 10, 1e18]
OL = ["~1.5", "1.5-2", "2-2.5", "2.5-3.5", "3.5-5", "5-10", "10+"]


def rid16(x):
    return re.sub(r"\D", "", str(x))[:16]


def band(v, bins, lab):
    return lab[int(np.clip(np.digitize([v], bins)[0] - 1, 0, len(lab) - 1))]


def load_odds():
    df = pd.read_csv(KEKKA_V2, encoding="cp932", low_memory=False)
    c = list(df.columns)
    rids = df[c[7]].map(rid16).values
    bans = pd.to_numeric(df[c[4]], errors="coerce").values
    o9 = pd.to_numeric(df[c[17]], errors="coerce").values
    ck = pd.to_numeric(df[c[6]], errors="coerce").values
    tpay = pd.to_numeric(df[c[8]], errors="coerce").values
    fpay = pd.to_numeric(df[c[9]], errors="coerce").values
    out = {}
    for rid, b, a, k, tp, fp in zip(rids, bans, o9, ck, tpay, fpay):
        if pd.isna(b):
            continue
        out[(rid, int(b))] = (a, k, tp, fp)
    return out


def main():
    print("[load] model + odds")
    m = joblib.load(BASE / "models/unified_rank_v6.pkl")
    model, feats, encs = m["model"], m["feature_cols"], m["encoders"]
    odds = load_odds()
    print(f"  odds rows {len(odds):,}")

    df = pd.read_csv(MASTER, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"])
    for cc, le in encs.items():
        if cc in df.columns:
            v = df[cc].astype(str).fillna("__NaN__")
            df[cc] = le.transform(v.where(v.isin(set(le.classes_)), "__NaN__"))
    df["_score"] = model.predict(df[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values)

    def C():
        return {"stake": 0.0, "ret": 0.0, "n": 0, "hit": 0}
    agg = defaultdict(lambda: defaultdict(C))  # agg[split][(mark,bet,dim,key)]

    for rid, g in df.groupby(COL_RID, sort=False):
        if len(g) < 5:
            continue
        sp = g["split"].iloc[0]
        if sp not in ("valid", "test"):
            continue
        g = g.sort_values("_score", ascending=False).reset_index(drop=True)
        ban = g[COL_BAN].astype(int).values
        r16 = rid16(rid)
        od_all = {int(ban[i]): odds.get((r16, int(ban[i]))) for i in range(len(g))}
        vod = {b: v[0] for b, v in od_all.items() if v and not pd.isna(v[0]) and v[0] > 1.0}
        fav_ban = min(vod, key=vod.get) if vod else None
        for rank, mk in ((0, "◎"), (1, "〇")):
            if rank >= len(g):
                continue
            b = int(ban[rank])
            od = od_all.get(b)
            if not od or pd.isna(od[0]) or od[0] <= 1.0:
                continue
            o9, ck, tp, fp = od
            obin = band(o9, OB, OL)
            isfav = "◎=1人気" if b == fav_ban else "◎≠1人気"
            for bet, won, pay in (("単勝", (not pd.isna(ck)) and int(ck) == 1, tp),
                                  ("複勝", (not pd.isna(ck)) and int(ck) <= 3, fp)):
                ret = float(pay) if (won and not pd.isna(pay)) else 0.0
                for dim, key in (("odds", obin), ("mktfav", isfav), ("all", "all")):
                    c = agg[sp][(mk, bet, dim, key)]
                    c["stake"] += 100
                    c["ret"] += ret
                    c["n"] += 1
                    c["hit"] += int(won)

    def roi(c):
        return round(c["ret"] / c["stake"] * 100, 1) if c["stake"] else 0.0

    res = defaultdict(dict)
    for sp in ("valid", "test"):
        for k, c in agg[sp].items():
            if c["n"] >= 30:
                res["|".join(map(str, k))][sp] = {"roi": roi(c), "n": c["n"], "hit": round(c["hit"] / c["n"] * 100, 1)}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- console ----
    def show(mk, bet, dim, order):
        print(f"\n【{mk} {bet} × {dim}】 (valid / test ROI%, n)")
        for key in order:
            cv = agg["valid"][(mk, bet, dim, key)]
            ct = agg["test"][(mk, bet, dim, key)]
            if cv["n"] < 30 and ct["n"] < 30:
                continue
            rv, rt = roi(cv), roi(ct)
            flag = "  ★+両方" if (rv >= 100 and rt >= 100) else ("  ◦+片方" if (rv >= 100 or rt >= 100) else "")
            print(f"  {key:<10} valid {rv:>6.1f}%/{cv['n']:<5}  test {rt:>6.1f}%/{ct['n']:<5}{flag}")

    print("\n" + "=" * 80)
    print("現行印で黒字(100%超)セル探索  valid(2023) & test(2024-25)")
    print("=" * 80)
    for mk in ("◎", "〇"):
        for bet in ("単勝", "複勝"):
            show(mk, bet, "odds", OL)
            show(mk, bet, "mktfav", ["◎=1人気", "◎≠1人気"])
            show(mk, bet, "all", ["all"])
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
