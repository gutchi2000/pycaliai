# -*- coding: utf-8 -*-
"""
exp_umami_vs_marks.py — 印(◎)駆動 vs UMAMI(xROI)駆動 の馬券適性比較 (leak-safe)
================================================================================
問い: 「印もいいけど UMAMI からのほうが馬券適性は高いのか？」

設計 (in-sample 自己成就を避ける):
  - UMAMI テーブル (EVビン×単勝オッズ帯 → 実現ROI) を **2024 のみで fit**
  - 戦略評価は **2025 のみ** (テーブルにとって完全 OOS)
  - EV判定 = 前売り9時オッズ / 決済 = 確定配当 (audit_ev_bin_roi と同じ leak 規約)
  - モデル = unified_rank_v6 (train〜2022 / valid 2023 なので test 2024-25 は OOS)

比較戦略 (全て 100 円固定買い):
  [単勝]
    A1 ◎単勝         : 各レース ◎ (PLスコア1位) を買う
    A2 UMAMI-A+ 単勝  : gate通過 (p_win>=0.04, 単勝<=50倍) かつ xROI>=0.80 の馬全部
    A3 UMAMI-S  単勝  : 同上 xROI>=0.85
    A4 ◎×UMAMI-A+    : ◎ のうち UMAMI A+ のレースだけ買う (実務ハイブリッド)
    A5 ◎×UMAMI-gate  : ◎ のうち gate (罠) に落ちないものだけ買う
  [複勝]
    B1 ◎複勝 / B2 UMAMI-A+ 複勝 / B4 ◎×UMAMI-A+ / B5 ◎×gate (単勝と同様)

出力: reports/exp_umami_vs_marks.json + コンソール表
実行: PYTHONUTF8=1 python exp_umami_vs_marks.py
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import pl_probs as PL
from backtest_pl_ev import all_fukusho_vec_fast, load_payouts
from audit_ev_bin_roi import (load_tanpuk, _rid16, ev_bin, fav_bin,
                              EV_LAB, FAV_LAB, MASTER_CSV, COL_RID, COL_JYUN, COL_BAN)

BASE = Path(__file__).parent
OUT_JSON = BASE / "reports/exp_umami_vs_marks.json"

FIT_YEAR, EVAL_YEAR = 2024, 2025
P_WIN_FLOOR, P_SHO_FLOOR, ODDS_CAP = 0.04, 0.12, 50.0   # umami.py と同値
XROI_A, XROI_S = 0.80, 0.85
MIN_CELL_N = 100   # 1年分 fit なのでセル下限は umami.py の 300 から緩和


def build_table(rows):
    """rows: (ev, fav_odds, stake, ret) → {key: {n, roi}} (cross + evビン単独)"""
    cross = defaultdict(lambda: [0, 0.0, 0.0])
    evonly = defaultdict(lambda: [0, 0.0, 0.0])
    for ev, fo, stake, ret in rows:
        k = f"{ev_bin(ev)}|{fav_bin(fo)}"
        for store, key in ((cross, k), (evonly, ev_bin(ev))):
            c = store[key]
            c[0] += 1; c[1] += stake; c[2] += ret
    fin = lambda c: {"n": c[0], "roi": (c[2] / c[1]) if c[1] else 0.0}
    return ({k: fin(c) for k, c in cross.items()},
            {k: fin(c) for k, c in evonly.items()})


def xroi_lookup(tables, ev, fav_odds):
    cross, evonly = tables
    c = cross.get(f"{ev_bin(ev)}|{fav_bin(fav_odds)}")
    if c and c["n"] >= MIN_CELL_N:
        return c["roi"]
    e = evonly.get(ev_bin(ev))
    return e["roi"] if e else None


def main():
    print("[load] v6 model + calibrators")
    b = joblib.load(BASE / "models/unified_rank_v6.pkl")
    model, feats, encs = b["model"], b["feature_cols"], b["encoders"]
    cal = joblib.load(BASE / "models/pl_calibrators_v6.pkl"); cal = cal.get("calibrators", cal)

    print("[load] master (test 2024-25)")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"])
    df = df[df["split"] == "test"].copy()
    for c, le in encs.items():
        if c not in df.columns: continue
        v = df[c].astype(str).fillna("__NaN__")
        df[c] = le.transform(v.where(v.isin(set(le.classes_)), "__NaN__"))
    X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    df["_score"] = model.predict(X)

    print("[load] payouts + 9時 odds")
    payouts = load_payouts()
    tanpuk = load_tanpuk()
    print(f"  単複 odds races={len(tanpuk):,}")

    # ---- pass1: 全馬の (year, ev, fav_odds, stake, ret, is_hon, p) を収集 ----
    # bet 候補行: 券種ごとに 1馬=1行。fit/eval を年で分ける。
    rows = {"tan": [], "fuku": []}   # dict: year, ev, fo, ret, hon, p, gated
    n_races = {FIT_YEAR: 0, EVAL_YEAR: 0}
    for rid, g in df.groupby(COL_RID, sort=False):
        if len(g) < 5: continue
        rid16 = _rid16(rid)
        year = int(rid16[:4])
        if year not in n_races: continue
        sp = tanpuk.get(rid16)
        if not sp: continue
        po = payouts.get(rid16)
        g = g.sort_values(COL_BAN).reset_index(drop=True)
        ban = g[COL_BAN].astype(int).values
        n = len(ban)
        w = PL.pl_weights(g["_score"].values)
        hon_i = int(np.argmax(g["_score"].values))   # ◎ = スコア1位
        n_races[year] += 1

        # 単勝
        p_tan = cal["tansho"].predict(PL.all_tansho(w))
        for i in range(n):
            o9 = sp["tan9"].get(int(ban[i]))
            if o9 is None: continue
            won = (po is not None and int(ban[i]) == po["win"])
            pay = float(po["tansho"]) if (won and po and po["tansho"] and not pd.isna(po["tansho"])) else 0.0
            gated = (p_tan[i] < P_WIN_FLOOR) or (o9 > ODDS_CAP)
            rows["tan"].append(dict(year=year, ev=p_tan[i] * o9, fo=o9, ret=pay,
                                    hon=(i == hon_i), gated=gated))

        # 複勝 (人気帯キーは単勝9時オッズ)
        if sp["fuku9"]:
            p_sho = cal["fukusho"].predict(all_fukusho_vec_fast(w))
            top3 = {po["win"], po["plc"], po["sho"]} if po else set()
            fpay = {}
            if po:
                for bn, k in [(po["win"], "fuku_win"), (po["plc"], "fuku_plc"), (po["sho"], "fuku_sho")]:
                    if po[k] and not pd.isna(po[k]): fpay[int(bn)] = float(po[k])
            for i in range(n):
                fo = sp["fuku9"].get(int(ban[i]))
                tan_o9 = sp["tan9"].get(int(ban[i]))
                if fo is None or tan_o9 is None: continue
                odds = (fo[0] + fo[1]) / 2.0
                won = (po is not None and int(ban[i]) in top3)
                pay = fpay.get(int(ban[i]), 0.0) if won else 0.0
                gated = (p_sho[i] < P_SHO_FLOOR) or (tan_o9 > ODDS_CAP)
                rows["fuku"].append(dict(year=year, ev=p_sho[i] * odds, fo=tan_o9, ret=pay,
                                         hon=(i == hon_i), gated=gated))

    print(f"  races: fit({FIT_YEAR})={n_races[FIT_YEAR]:,} eval({EVAL_YEAR})={n_races[EVAL_YEAR]:,}")

    # ---- pass2: 2024 で UMAMI テーブル fit ----
    tables = {}
    for kind in ("tan", "fuku"):
        fit = [(r["ev"], r["fo"], 100.0, r["ret"]) for r in rows[kind] if r["year"] == FIT_YEAR]
        tables[kind] = build_table(fit)
        print(f"  [fit {kind}] bets={len(fit):,} cells={len(tables[kind][0])}")

    # ---- pass3: 2025 で戦略評価 ----
    def evaluate(kind, name, pred):
        sel = [r for r in rows[kind] if r["year"] == EVAL_YEAR and pred(r)]
        n = len(sel)
        if n == 0:
            return {"strategy": name, "n": 0}
        rets = np.array([r["ret"] for r in sel])         # 100円あたり払戻
        roi = rets.mean() / 100.0
        hit = float((rets > 0).mean())
        se = rets.std(ddof=1) / np.sqrt(n) / 100.0 if n > 1 else 0.0
        return {"strategy": name, "n": n, "hit": round(hit, 4), "roi": round(roi, 4),
                "roi_ci95": [round(roi - 1.96 * se, 4), round(roi + 1.96 * se, 4)],
                "mean_odds9": round(float(np.mean([r["fo"] for r in sel])), 1)}

    def um_ok(kind, r, th):
        if r["gated"]: return False
        x = xroi_lookup(tables[kind], r["ev"], r["fo"])
        return x is not None and x >= th

    results = []
    for kind, lab in (("tan", "単勝"), ("fuku", "複勝")):
        results += [
            evaluate(kind, f"{lab} A1 ◎のみ",            lambda r: r["hon"]),
            evaluate(kind, f"{lab} A2 UMAMI-A+ 全馬",     lambda r, k=kind: um_ok(k, r, XROI_A)),
            evaluate(kind, f"{lab} A3 UMAMI-S 全馬",      lambda r, k=kind: um_ok(k, r, XROI_S)),
            evaluate(kind, f"{lab} A4 ◎×UMAMI-A+",       lambda r, k=kind: r["hon"] and um_ok(k, r, XROI_A)),
            evaluate(kind, f"{lab} A5 ◎×gate通過のみ",    lambda r: r["hon"] and not r["gated"]),
        ]

    out = {"design": f"UMAMIテーブル fit={FIT_YEAR} / 評価={EVAL_YEAR} (OOS), "
                     "EV=9時オッズ, 決済=確定配当, 100円固定",
           "gates": {"p_win_floor": P_WIN_FLOOR, "p_sho_floor": P_SHO_FLOOR,
                     "odds_cap": ODDS_CAP, "min_cell_n": MIN_CELL_N},
           "n_races": n_races, "results": results}
    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 88)
    print(f"印(◎) vs UMAMI 馬券適性比較  (評価={EVAL_YEAR} OOS, 100円固定, 控除率床80%)")
    print("=" * 88)
    print(f"  {'戦略':24s} {'n':>7s} {'hit':>7s} {'ROI':>8s} {'95%CI':>18s} {'平均odds9':>9s}")
    for r in results:
        if r["n"] == 0:
            print(f"  {r['strategy']:24s} {'0':>7s}  (該当なし)")
            continue
        ci = f"[{r['roi_ci95'][0]*100:.1f},{r['roi_ci95'][1]*100:.1f}]"
        print(f"  {r['strategy']:24s} {r['n']:>7,} {r['hit']*100:>6.1f}% {r['roi']*100:>7.1f}% "
              f"{ci:>18s} {r['mean_odds9']:>9.1f}")
    print(f"\n[saved] {OUT_JSON}")


if __name__ == "__main__":
    main()
