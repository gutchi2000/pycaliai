"""
validate_search.py — 現行印で+を生む◎選択シグナル探索
◎単勝/◎複勝 ROI を「top1_dominance(◎p - 〇p)」「field_size」「top1_dom×オッズ」で
valid/test 別に切り、100%超(=黒字)が robust に出るセルを探す。
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
OUT = BASE / "reports/validate_search.json"
COL_RID = "レースID(新/馬番無)"
COL_JYUN = "着順"
COL_BAN = "馬番"

DOM_B = [-1, 0.05, 0.10, 0.20, 1e9]
DOM_L = ["dom<.05", ".05-.10", ".10-.20", "dom.20+"]
FLD_B = [0, 10.5, 14.5, 1e9]
FLD_L = ["≤10頭", "11-14頭", "15+頭"]
OD_B = [0, 2.5, 5, 10, 1e18]
OD_L = ["≤2.5", "2.5-5", "5-10", "10+"]


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
    tp = pd.to_numeric(df[c[8]], errors="coerce").values
    fp = pd.to_numeric(df[c[9]], errors="coerce").values
    out = {}
    for rid, b, a, k, t, f in zip(rids, bans, o9, ck, tp, fp):
        if pd.isna(b):
            continue
        out[(rid, int(b))] = (a, k, t, f)
    return out


def main():
    print("[load] model + odds")
    m = joblib.load(BASE / "models/unified_rank_v6.pkl")
    model, feats, encs = m["model"], m["feature_cols"], m["encoders"]
    cal = joblib.load(BASE / "models/pl_calibrators_v6.pkl")
    cal = cal.get("calibrators", cal)
    odds = load_odds()

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
    agg = defaultdict(lambda: defaultdict(C))

    for rid, g in df.groupby(COL_RID, sort=False):
        if len(g) < 5:
            continue
        sp = g["split"].iloc[0]
        if sp not in ("valid", "test"):
            continue
        g = g.sort_values("_score", ascending=False).reset_index(drop=True)
        w = PL.pl_weights(g["_score"].values)
        p = cal["tansho"].predict(PL.all_tansho(w))
        ban = g[COL_BAN].astype(int).values
        r16 = rid16(rid)
        b0 = int(ban[0])
        od = odds.get((r16, b0))
        if not od or pd.isna(od[0]) or od[0] <= 1.0:
            continue
        o9, ck, tp, fp = od
        dom = float(p[0] - p[1])
        fld = len(g)
        won_t = (not pd.isna(ck)) and int(ck) == 1
        won_p = (not pd.isna(ck)) and int(ck) <= 3
        ret_t = float(tp) if (won_t and not pd.isna(tp)) else 0.0
        ret_p = float(fp) if (won_p and not pd.isna(fp)) else 0.0
        domb, fldb, odb = band(dom, DOM_B, DOM_L), band(fld, FLD_B, FLD_L), band(o9, OD_B, OD_L)
        for bet, won, ret in (("単", won_t, ret_t), ("複", won_p, ret_p)):
            for dim, key in (("dom", domb), ("fld", fldb),
                             ("dom×od", f"{domb}|{odb}")):
                c = agg[sp][(bet, dim, key)]
                c["stake"] += 100
                c["ret"] += ret
                c["n"] += 1
                c["hit"] += int(won)

    def roi(c):
        return round(c["ret"] / c["stake"] * 100, 1) if c["stake"] else 0.0

    res = {}
    hits = []
    for k in set(list(agg["valid"]) + list(agg["test"])):
        cv, ct = agg["valid"][k], agg["test"][k]
        if cv["n"] < 30 or ct["n"] < 30:
            continue
        rv, rt = roi(cv), roi(ct)
        res["|".join(map(str, k))] = {"valid": [rv, cv["n"]], "test": [rt, ct["n"]]}
        if rv >= 100 and rt >= 100:
            hits.append((k, rv, rt, cv["n"], ct["n"]))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    def show(bet, dim, order):
        print(f"\n【◎{bet}勝 × {dim}】 valid / test (ROI%, n)")
        for key in order:
            cv, ct = agg["valid"][(bet, dim, key)], agg["test"][(bet, dim, key)]
            if cv["n"] < 30 and ct["n"] < 30:
                continue
            rv, rt = roi(cv), roi(ct)
            fl = "  ★+両方" if (rv >= 100 and rt >= 100) else ("  ◦+片" if (rv >= 100 or rt >= 100) else "")
            print(f"  {key:<14} {rv:>6.1f}%/{cv['n']:<5}  {rt:>6.1f}%/{ct['n']:<5}{fl}")

    print("\n" + "=" * 74 + "\n◎選択シグナル探索 (valid23 / test24-25)\n" + "=" * 74)
    for bet in ("単", "複"):
        show(bet, "dom", DOM_L)
        show(bet, "fld", FLD_L)
        show(bet, "dom×od", [f"{d}|{o}" for d in DOM_L for o in OD_L])
    print(f"\n>>> valid&test 両方100%超セル: {len(hits)}個")
    for k, rv, rt, nv, nt in sorted(hits, key=lambda x: -(x[1] + x[2])):
        print(f"    {k}  valid {rv}%/{nv}  test {rt}%/{nt}")
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
