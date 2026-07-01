"""
trio_formation.py — 三連複フォーメーション(モデル順位ベース)の捕捉/ROI を valid/test で比較
列の数字=モデル順位(1=◎,2=〇,...). A/B/C=1着/2着/3着 候補順位集合。
捕捉=着順1-2-3の3頭の順位集合がフォーメーションの組に含まれる。払戻=sanrenpuku。
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
from backtest_pl_ev import load_payouts

BASE = Path(__file__).parent
MASTER = BASE / "data/master_v2_20130105-20251228.csv"
OUT = BASE / "reports/trio_formation.json"
COL_RID = "レースID(新/馬番無)"
COL_JYUN = "着順"


def rid16(x):
    return re.sub(r"\D", "", str(x))[:16]


def gen(A, B, C):
    s = set()
    for a in A:
        for b in B:
            for c in C:
                if len({a, b, c}) == 3:
                    s.add(frozenset({a, b, c}))
    return s


# 順位は 1-index (1=◎)
FORMS = {
    "boxTop3 ◎〇▲":        gen({1, 2, 3}, {1, 2, 3}, {1, 2, 3}),
    "boxTop4 ◎〇▲△":       gen({1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}),
    "boxTop5":             gen({1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}),
    "前回: ◎-〇▲△-広(2-8)": gen({1}, {2, 3, 4}, {2, 3, 4, 5, 6, 7, 8}),
    "◎軸 6頭流し":          gen({1}, {2, 3, 4, 5, 6, 7}, {2, 3, 4, 5, 6, 7}),
    "これ: ◎〇/◎〇▲△△/▲△△△": gen({1, 2}, {1, 2, 3, 4, 5}, {3, 4, 5, 6}),
    "◎軸: ◎/〇▲/〇▲△1-4":    gen({1}, {2, 3}, {2, 3, 4, 5, 6, 7}),
}


def main():
    print("[load] model + payouts")
    m = joblib.load(BASE / "models/unified_rank_v6.pkl")
    model, feats, encs = m["model"], m["feature_cols"], m["encoders"]
    payouts = load_payouts()

    df = pd.read_csv(MASTER, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"])
    for cc, le in encs.items():
        if cc in df.columns:
            v = df[cc].astype(str).fillna("__NaN__")
            df[cc] = le.transform(v.where(v.isin(set(le.classes_)), "__NaN__"))
    df["_score"] = model.predict(df[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values)

    pts = {name: len(s) for name, s in FORMS.items()}
    agg = {s: {name: {"stake": 0.0, "ret": 0.0, "cap": 0, "n": 0} for name in FORMS}
           for s in ("valid", "test")}

    for rid, g in df.groupby(COL_RID, sort=False):
        if len(g) < 8:
            continue
        sp = g["split"].iloc[0]
        if sp not in ("valid", "test"):
            continue
        po = payouts.get(rid16(rid))
        if not po:
            continue
        pay = po.get("sanrenpuku")
        if pay is None or pd.isna(pay) or pay <= 0:
            continue
        g = g.sort_values("_score", ascending=False).reset_index(drop=True)
        chaku = pd.to_numeric(g[COL_JYUN], errors="coerce").values
        ranks123 = frozenset(i + 1 for i in range(len(g)) if chaku[i] in (1, 2, 3))
        if len(ranks123) != 3:
            continue
        for name, forms in FORMS.items():
            cap = ranks123 in forms
            a = agg[sp][name]
            a["stake"] += pts[name] * 100.0
            a["ret"] += float(pay) if cap else 0.0
            a["cap"] += int(cap)
            a["n"] += 1

    def roi(a):
        return round(a["ret"] / a["stake"] * 100, 1) if a["stake"] else 0.0

    res = {"points": pts, "by_form": {}}
    for name in FORMS:
        res["by_form"][name] = {
            s: {"roi%": roi(agg[s][name]),
                "capture%": round(agg[s][name]["cap"] / agg[s][name]["n"] * 100, 1) if agg[s][name]["n"] else 0,
                "n": agg[s][name]["n"]} for s in ("valid", "test")}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n{'='*82}\n三連複フォーメーション比較 (valid23 / test24-25)\n{'='*82}")
    print(f"  {'形':<22}{'点数':>5}{'捕捉% V/T':>16}{'ROI% V/T':>16}")
    for name in FORMS:
        v, t = res["by_form"][name]["valid"], res["by_form"][name]["test"]
        cap = f"{v['capture%']}/{t['capture%']}"
        r = f"{v['roi%']}/{t['roi%']}"
        print(f"  {name:<22}{pts[name]:>5}{cap:>16}{r:>16}")
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
