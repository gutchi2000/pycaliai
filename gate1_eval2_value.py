"""
gate1_eval2_value.py
====================
ゲート1 評価2: 実現バリュー。M1上位K点 vs M0上位K点を同点数フラット買いした実現ROIを比較。

leak-safe: 選択は確率(M0 or M1)のみ。payout(確定配当)は実現リターン計算のみ。
- umaren: 全期間 kekka payout (2013-25)。winning pair=(1着,2着)。
- wide:   wide_payouts (2016-25)。top3 の 3 ペアが的中。
期間: test 2024-25 (OOS) 主、full は in-sample 注記付き。
β は eval1 (reports/gate1_corr_calibration.json) の値を再利用。

出力: reports/gate1_corr_value.json
実行: python gate1_eval2_value.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from gate1_eval1 import (build_records, load_odds_devig, joint_probs,
                         add_style, MASTER, COL_RID, COL_JYUN, COL_BAN)
from joint_calibration_v6 import apply_encoders
from backtest_pl_ev import load_payouts, load_wide_payouts
from corr_features import PAIR_FEATURE_NAMES

BASE = Path(__file__).parent
np.random.seed(42)
KS = [2, 3, 5]
UNIT = 100.0


def topk_idx(p, k):
    k = min(k, len(p))
    return np.argsort(-p)[:k]


def main():
    bundle = joblib.load(BASE / "models/unified_rank_v6.pkl")
    model, feats, encs = bundle["model"], bundle["feature_cols"], bundle["encoders"]
    beta_d = json.load(open(BASE / "reports/gate1_corr_calibration.json", encoding="utf-8"))["beta"]
    beta = np.array([beta_d[n] for n in PAIR_FEATURE_NAMES], dtype=float)
    print("[beta]", {n: round(beta_d[n], 3) for n in PAIR_FEATURE_NAMES})

    print("[load] master + style + score + odds")
    df = pd.read_csv(MASTER, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]).copy()
    df = add_style(df)
    enc = apply_encoders(df, encs)
    X = enc[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    df["_score"] = model.predict(X)
    odds_map = load_odds_devig()
    recs = build_records(df, odds_map)
    payouts = load_payouts(); wide_map = load_wide_payouts()
    print(f"  races={len(recs):,}  payout races={len(payouts):,}  wide races={len(wide_map):,}")

    # acc[bet][period][K][variant] = [stake, ret, bets, hits]
    def newacc():
        return {"stake": 0.0, "ret": 0.0, "bets": 0, "hits": 0}
    acc = {bt: {pp: {k: {"M0": newacc(), "M1": newacc()} for k in KS}
                for pp in ["full", "test"]} for bt in ["umaren", "wide"]}

    for rec in recs:
        rid16 = rec["rid16"]; po = payouts.get(rid16)
        if po is None:
            continue
        ban = rec["ban"]; b2i = {int(b): i for i, b in enumerate(ban)}
        wi = b2i.get(po["win"]); pi = b2i.get(po["plc"]); si = b2i.get(po["sho"])
        if wi is None or pi is None or si is None:
            continue
        n = len(ban)
        iu, ju = np.triu_indices(n, 1)
        periods = ["full"] + (["test"] if rec["split"] == "test" else [])

        # ---------- umaren ----------
        p0, p1, _, _ = joint_probs(rec, beta, "umaren")
        win_pair = frozenset({wi, pi})
        um_pay = float(po["umaren"]) if (po["umaren"] and not pd.isna(po["umaren"])) else 0.0
        for k in KS:
            for vr, p in (("M0", p0), ("M1", p1)):
                sel = topk_idx(p, k)
                for s in sel:
                    pr = frozenset({int(iu[s]), int(ju[s])})
                    hit = (pr == win_pair)
                    for pp in periods:
                        a = acc["umaren"][pp][k][vr]
                        a["stake"] += UNIT; a["bets"] += 1
                        if hit:
                            a["ret"] += um_pay; a["hits"] += 1

        # ---------- wide (2016+ payout) ----------
        wlist = wide_map.get(rid16)
        if wlist:
            win_wide = {}
            for (ui, uj, wp) in wlist:
                ii = b2i.get(int(ui)); jj = b2i.get(int(uj))
                if ii is not None and jj is not None:
                    win_wide[frozenset({ii, jj})] = float(wp)
            p0w, p1w, _, _ = joint_probs(rec, beta, "wide")
            for k in KS:
                for vr, p in (("M0", p0w), ("M1", p1w)):
                    sel = topk_idx(p, k)
                    for s in sel:
                        pr = frozenset({int(iu[s]), int(ju[s])})
                        pay = win_wide.get(pr, 0.0)
                        for pp in periods:
                            a = acc["wide"][pp][k][vr]
                            a["stake"] += UNIT; a["bets"] += 1
                            if pay > 0:
                                a["ret"] += pay; a["hits"] += 1

    def roi(a):
        return {"roi": round(a["ret"] / a["stake"], 4) if a["stake"] else 0.0,
                "hit_rate": round(a["hits"] / a["bets"], 4) if a["bets"] else 0.0,
                "bets": a["bets"]}

    out = {"beta": beta_d, "ks": KS,
           "note": "leak-safe: 選択はprobのみ, payoutは実現リターンのみ. full=2013-25(umaren)/2016-25(wide) in-sample込, test=2024-25 OOS.",
           "results": {}}
    print("\n実現ROI (M0 vs M1, top-K点フラット買い):")
    for bt in ["umaren", "wide"]:
        out["results"][bt] = {}
        for pp in ["test", "full"]:
            out["results"][bt][pp] = {}
            print(f"\n  [{bt} / {pp}]")
            print(f"    {'K':>2s} {'M0_ROI':>8s} {'M1_ROI':>8s} {'M0_hit':>7s} {'M1_hit':>7s} {'M1-M0':>7s}")
            for k in KS:
                r0 = roi(acc[bt][pp][k]["M0"]); r1 = roi(acc[bt][pp][k]["M1"])
                out["results"][bt][pp][f"K{k}"] = {"M0": r0, "M1": r1,
                                                   "delta_roi": round(r1["roi"] - r0["roi"], 4)}
                print(f"    {k:>2d} {r0['roi']*100:>7.1f}% {r1['roi']*100:>7.1f}% "
                      f"{r0['hit_rate']*100:>6.1f}% {r1['hit_rate']*100:>6.1f}% "
                      f"{(r1['roi']-r0['roi'])*100:>+6.1f}pt")

    outp = BASE / "reports/gate1_corr_value.json"
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {outp}")


if __name__ == "__main__":
    main()
