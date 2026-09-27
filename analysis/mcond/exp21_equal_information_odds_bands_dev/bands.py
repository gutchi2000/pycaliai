# -*- coding: utf-8 -*-
"""
bands.py — EXP21 Stage 0: 価格だけで作る帯と被覆 (outcome を一切読まない)
==========================================================================
ticket 集合 (券種・期間・価格時点ごと):
  単勝・複勝・馬連  TANPUK/UMAREN 2013-2018 (発見) / 2019-2023 (評価)、D0 = terminal (区分4)、
                    D1 = historical_pre_snapshot (区分1、確定記録の 15 分以上前の最後、約 T−28。T−10 とは呼ばない)
  枠連・ワイド・馬単  2023 91 列 (terminal と同定) と OD 2026 terminal 日 (D0 のみ)
  三連複            OD 2026 terminal 日だけ (D0 のみ)
  三連単            G0 FAIL (価格源なし) → 帯を作らない
starter = terminal 単勝 > 1.0 の馬 (0.0 は取消・除外で返還対象。0 円の外れ扱いにしない)
期待的中 mass q_j (結果を使わない):
  単勝・馬連・枠連・馬単・三連複  race 内で 1/odds を券種内正規化 (和 = 1)
  複勝  帯キー sqrt(Lo×Hi)、race 内で 1/key を和 = places(n) へ正規化 (8 頭以上 3、5〜7 頭 2、4 頭以下は発売なし)
  ワイド 帯キー sqrt(Lo×Hi)、race 内で和 = 3 へ正規化 (2013-2023 の全 race で的中 3 組)
帯:
  primary   equal expected-hit-mass 10 帯 (単勝・複勝・馬連は 20 帯感度も)。odds 昇順の累積 mass を等分
  secondary equal ticket-count 10 帯 (odds の分位)
  fixed     [1,2], (2,3], ... , (15000, inf) (境界は (lo, hi])
帯ごとに odds 範囲・ticket 数・race 数・暦日数・期待的中 mass を出す。
insufficient (結果前に固定): race 単位の較正 null 下の解析的 CI95 半幅 > 0.10、または暦日 < 30、または期待的中 mass < 30
出力: out/label_free_band_coverage.json、data/_research/mcond/exp21/tickets_*.npz (power 監査用)
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from . import loaders as L

FIXED_EDGES = [1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 25.0, 50.0, 100.0, 300.0, 600.0, 1200.0, 2500.0, 5000.0, 15000.0, np.inf]
HALF_WIDTH_MAX = 0.10
MIN_DAYS = 30
MIN_MASS = 30.0
PERIODS = {"discovery_2013_2018": (2013, 2018), "evaluation_2019_2023": (2019, 2023)}


def places(n):
    return 3 if n >= 8 else (2 if n >= 5 else 0)


def mass_bands(key, q, n_bins):
    o = np.argsort(key, kind="mergesort")
    cm = np.cumsum(q[o]) / q.sum()
    b = np.minimum((cm * n_bins - 1e-12).astype(int), n_bins - 1)
    out = np.empty(len(key), int)
    out[o] = b
    return out


def count_bands(key, n_bins):
    o = np.argsort(key, kind="mergesort")
    b = (np.arange(len(key)) * n_bins // len(key))
    out = np.empty(len(key), int)
    out[o] = b
    return out


def fixed_bands(key):
    return np.clip(np.searchsorted(FIXED_EDGES, key, side="left") - 1, 0, len(FIXED_EDGES) - 2)


def summarize(key, q, race, day, band, n_bins, pay_odds):
    """帯ごとの記述と、較正 null (hit ~ q、払戻 = pay_odds) 下の race 単位解析的 CI95 半幅"""
    rows = []
    for b in range(n_bins):
        m = band == b
        if not m.any():
            rows.append({"band": b, "tickets": 0})
            continue
        N = int(m.sum())
        # race 内 X_r = Σ hit_j o_j。Var ≈ Σ q o² − (Σ q o)² per race (排反近似。複数的中券種は独立近似)
        qr, orr, rr = q[m], pay_odds[m], race[m]
        s1 = np.bincount(np.unique(rr, return_inverse=True)[1], weights=qr * orr)
        s2 = np.bincount(np.unique(rr, return_inverse=True)[1], weights=qr * orr * orr)
        var = np.maximum(s2 - s1 * s1, 0).sum()
        hw = 1.96 * np.sqrt(var) / N
        nd = int(len(np.unique(day[m])))
        mass = float(q[m].sum())
        rows.append({"band": b, "odds_min": float(key[m].min()), "odds_max": float(key[m].max()), "tickets": N,
                     "races": int(len(np.unique(rr))), "days": nd, "expected_hit_mass": mass,
                     "null_roi_expectation": float((qr * orr).sum() / N), "null_ci95_half_width": float(hw),
                     "insufficient": bool(hw > HALF_WIDTH_MAX or nd < MIN_DAYS or mass < MIN_MASS)})
    return rows


def all_bands(key, q, race, day, pay_odds, twenty=False):
    out = {"primary_mass10": summarize(key, q, race, day, mass_bands(key, q, 10), 10, pay_odds),
           "secondary_count10": summarize(key, q, race, day, count_bands(key, 10), 10, pay_odds),
           "fixed": summarize(key, q, race, day, fixed_bands(key), len(FIXED_EDGES) - 1, pay_odds)}
    if twenty:
        out["primary_mass20"] = summarize(key, q, race, day, mass_bands(key, q, 20), 20, pay_odds)
    return out


# ---------------------------------------------------------------- TANPUK / UMAREN
def history_tickets():
    from ..exp18_cross_pool_market_tomography_dev.loaders import load_structure
    from ..exp18_cross_pool_market_tomography_dev.market_build import pool_matrices, race_arrays, snapshot_index
    st = load_structure(range(2013, 2024))
    W, LO, HI, U = pool_matrices(st["tan"], st["um"])
    idx = snapshot_index(st["tan"], st["um"], st["info"])
    T = {(t, lay): {"key": [], "q": [], "race": [], "day": [], "pay": [], "year": [], "a": [], "b": [], "null": []}
         for t in ("tansho", "fukusho", "umaren") for lay in ("D0", "D1")}
    cov = {}
    rid_list = sorted(idx.index)
    for ri, rid in enumerate(rid_list):
        r = idx.loc[rid]
        y = int(rid[:4])
        c = cov.setdefault(y, {"races_with_terminal": 0, "tan_terminal_complete": 0, "fuku_terminal_complete": 0,
                               "umaren_terminal_complete": 0, "pre_snapshot": 0, "tan_pre_complete": 0,
                               "fuku_pre_complete": 0, "umaren_pre_complete": 0})
        if not (r.get("term_tan") == r.get("term_tan")) or int(r["term_um"]) < 0:
            continue
        at = race_arrays(W, LO, HI, U, int(r["term_tan"]), int(r["term_um"]))
        n = at["n"]
        if n < 2:
            continue
        c["races_with_terminal"] += 1
        day = int(rid[:8])
        tan_ok = bool(np.all(np.isfinite(at["win"]) & (at["win"] > 1.0)))
        fk = places(n)
        fuku_ok = fk > 0 and bool(np.all(np.isfinite(at["place_lo"]) & (at["place_lo"] > 0) & np.isfinite(at["place_hi"])))
        um_ok = bool(np.all(np.isfinite(at["umaren"]) & (at["umaren"] >= 1.0)))
        c["tan_terminal_complete"] += tan_ok
        c["fuku_terminal_complete"] += fuku_ok
        c["umaren_terminal_complete"] += um_ok
        pre = None
        if r.get("pre_tan") == r.get("pre_tan") and int(r.get("pre_um", -1)) >= 0:
            c["pre_snapshot"] += 1
            pa = race_arrays(W, LO, HI, U, int(r["pre_tan"]), int(r["pre_um"]))
            wp = W[int(r["pre_tan"])][at["bans"] - 1]
            lop, hip = LO[int(r["pre_tan"])][at["bans"] - 1], HI[int(r["pre_tan"])][at["bans"] - 1]
            ia, ib = np.triu_indices(n, 1)
            from ..exp18_cross_pool_market_tomography_dev.market_build import PAIR_POS
            up = U[int(r["pre_um"]), [PAIR_POS[(int(at["bans"][x]), int(at["bans"][z]))] for x, z in zip(ia, ib)]]
            pre = {"win": wp, "lo": lop, "hi": hip, "um": up}
            c["tan_pre_complete"] += bool(np.all(np.isfinite(wp) & (wp > 1.0)))
            c["fuku_pre_complete"] += fk > 0 and bool(np.all(np.isfinite(lop) & (lop > 0) & np.isfinite(hip)))
            c["umaren_pre_complete"] += bool(np.all(np.isfinite(up) & (up >= 1.0)))

        bans = at["bans"].astype(int)
        ia2, ib2 = np.triu_indices(n, 1)

        def add(t, lay, key, q, pay, a, b, null):
            d = T[(t, lay)]
            d["key"].append(key); d["q"].append(q); d["race"].append(np.full(len(key), ri))
            d["day"].append(np.full(len(key), day)); d["pay"].append(pay); d["year"].append(np.full(len(key), y))
            d["a"].append(a); d["b"].append(b); d["null"].append(null)
        # null = race の terminal 価格だけから作る較正 null の ticket 期待値 1/overround (単勝・馬連)。複勝は定義しない (nan)
        if tan_ok:
            inv = 1 / at["win"]
            nul = np.full(n, 1.0 / inv.sum())
            add("tansho", "D0", at["win"], inv / inv.sum(), at["win"], bans, np.zeros(n, int), nul)
            if pre is not None and np.all(np.isfinite(pre["win"]) & (pre["win"] > 1.0)):
                iv = 1 / pre["win"]
                add("tansho", "D1", pre["win"], iv / iv.sum(), at["win"], bans, np.zeros(n, int), nul)
        if fuku_ok:
            kk = np.sqrt(at["place_lo"] * at["place_hi"])
            add("fukusho", "D0", kk, fk * (1 / kk) / (1 / kk).sum(), kk, bans, np.zeros(n, int), np.full(n, np.nan))
            if pre is not None and np.all(np.isfinite(pre["lo"]) & (pre["lo"] > 0) & np.isfinite(pre["hi"])):
                kp = np.sqrt(pre["lo"] * pre["hi"])
                add("fukusho", "D1", kp, fk * (1 / kp) / (1 / kp).sum(), kk, bans, np.zeros(n, int), np.full(n, np.nan))
        if um_ok:
            inv = 1 / at["umaren"]
            nul = np.full(len(inv), 1.0 / inv.sum())
            add("umaren", "D0", at["umaren"], inv / inv.sum(), at["umaren"], bans[ia2], bans[ib2], nul)
            if pre is not None and np.all(np.isfinite(pre["um"]) & (pre["um"] >= 1.0)):
                iv = 1 / pre["um"]
                add("umaren", "D1", pre["um"], iv / iv.sum(), at["umaren"], bans[ia2], bans[ib2], nul)
    out = {k: {kk: np.concatenate(vv) for kk, vv in v.items()} for k, v in T.items()}
    return out, cov, np.array(rid_list)


# ---------------------------------------------------------------- 91 列 / OD terminal
def target_tickets(df, with_trio):
    T = {t: {"key": [], "q": [], "race": [], "day": [], "pay": []}
         for t in ("wakuren", "wide", "umatan") + (("sanrenpuku",) if with_trio else ())}
    for ri, (rid, g) in enumerate(df.groupby("rid16")):
        g = g.set_index("ban")
        bans = [int(b) for b in g.index]
        V = {b: L.split_vals(g.loc[b, "vals"], with_trio) for b in bans}
        st = [b for b in bans if V[b]["tan"] > 0]
        day = int(rid[:8])
        wk, seen = [], set()
        for b in st:
            fi = int(g.loc[b, "waku"])
            for fj in range(1, 9):
                key = (min(fi, fj), max(fi, fj))
                v = V[b]["wakuren"][fj - 1]
                if key not in seen and v > 0:
                    seen.add(key)
                    wk.append(v)
        wl, wh, ut, tr = [], [], [], []
        for i in st:
            for j in st:
                if j > i:
                    wl.append(V[i]["wide_lo"][j - 1]); wh.append(V[i]["wide_hi"][j - 1])
                if j != i:
                    ut.append(V[i]["umatan"][j - 1])
            if with_trio:
                for idx, (j, k) in enumerate(L.TRIO_PAIRS[i]):
                    if i < j and j in st and k in st:
                        tr.append(V[i]["trio"][idx])

        def add(t, key, q, pay):
            d = T[t]
            d["key"].append(np.asarray(key)); d["q"].append(np.asarray(q)); d["race"].append(np.full(len(key), ri))
            d["day"].append(np.full(len(key), day)); d["pay"].append(np.asarray(pay))
        for t, arr in (("wakuren", wk), ("umatan", ut)) + ((("sanrenpuku", tr),) if with_trio else ()):
            a = np.array(arr, float)
            if len(a) and np.all(a > 0):
                add(t, a, (1 / a) / (1 / a).sum(), a)
        lo, hi = np.array(wl, float), np.array(wh, float)
        if len(lo) and np.all(lo > 0) and np.all(hi > 0):
            kk = np.sqrt(lo * hi)
            add("wide", kk, 3 * (1 / kk) / (1 / kk).sum(), kk)
    return {k: {kk: np.concatenate(vv) for kk, vv in v.items()} for k, v in T.items() if v["key"]}


def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    t0 = time.time()
    L.RESEARCH.mkdir(parents=True, exist_ok=True)
    g0 = json.loads((L.OUT / "g0_audit.json").read_text(encoding="utf-8"))
    res = {"rules": {"insufficient": {"null_ci95_half_width_gt": HALF_WIDTH_MAX, "days_lt": MIN_DAYS,
                                      "expected_hit_mass_lt": MIN_MASS},
                     "fixed_edges": [str(x) for x in FIXED_EDGES],
                     "D1_history": "historical_pre_snapshot (about T-28); never called T-10",
                     "no_outcome_columns_read": True}, "history": {}, "other_types": {}}
    H, cov, rid_list = history_tickets()
    np.save(L.RESEARCH / "rid_list.npy", rid_list)
    res["history_coverage_by_year"] = {str(y): v for y, v in sorted(cov.items())}
    for (t, lay), d in H.items():
        for pname, (y0, y1) in PERIODS.items():
            m = (d["year"] >= y0) & (d["year"] <= y1)
            key, q, race, day, pay = d["key"][m], d["q"][m], d["race"][m], d["day"][m], d["pay"][m]
            res["history"][f"{t}|{lay}|{pname}"] = {
                "tickets": int(m.sum()), "races": int(len(np.unique(race))), "days": int(len(np.unique(day))),
                "bands": all_bands(key, q, race, day, pay, twenty=True)}
            np.savez_compressed(L.RESEARCH / f"tickets_{t}_{lay}_{pname}.npz", key=key, q=q, race=race, day=day,
                                pay=pay, year=d["year"][m], a=d["a"][m], b=d["b"][m], null=d["null"][m])
        print(f"[{t} {lay}] done ({time.time()-t0:.0f}s)", flush=True)
    # 他券種 D0 (2023 terminal、OD 2026 terminal 日)
    df = L.read_target_odds(L.RAW2023, 91)
    from ..exp18_cross_pool_market_tomography_dev.loaders import load_structure
    from ..exp18_cross_pool_market_tomography_dev.market_build import snapshot_index
    st = load_structure([2023])
    idx = snapshot_index(st["tan"], st["um"], st["info"])
    keymap = {(r[8:10], int(r[10:12]), int(r[12:14]), int(r[14:16])): r[:8] for r in idx.index}
    df["rid16"] = [f"{keymap.get((v, k, n, R), 'X')}{v}{k:02d}{n:02d}{R:02d}" for v, k, n, R in
                   zip(df.venue, df.kai, df.nichi, df.R)]
    for t, d in target_tickets(df, with_trio=False).items():
        res["other_types"][f"{t}|D0|raw2023_terminal"] = {"tickets": int(len(d["key"])),
                                                          "races": int(len(np.unique(d["race"]))),
                                                          "days": int(len(np.unique(d["day"]))),
                                                          "bands": all_bands(d["key"], d["q"], d["race"], d["day"], d["pay"])}
    term_days = [k for k, v in g0["od2026_files"].items() if v["identified"] == "terminal"]
    parts = []
    for d8 in term_days:
        o = L.read_target_odds(L.OD_DIR / f"OD{d8[2:]}.CSV", 227)
        o["rid16"] = [f"{d8}{v}{k:02d}{n:02d}{R:02d}" for v, k, n, R in zip(o.venue, o.kai, o.nichi, o.R)]
        parts.append(o)
    od = pd.concat(parts, ignore_index=True)
    for t, d in target_tickets(od, with_trio=True).items():
        res["other_types"][f"{t}|D0|od2026_terminal_days"] = {"tickets": int(len(d["key"])),
                                                              "races": int(len(np.unique(d["race"]))),
                                                              "days": int(len(np.unique(d["day"]))),
                                                              "terminal_days": term_days,
                                                              "bands": all_bands(d["key"], d["q"], d["race"], d["day"], d["pay"])}
    res["not_available"] = {"sanrentan": "G0 FAIL (no price source)",
                            "D1_forward_T10_all_types": "no OD day identified as T-10 (36 pre-race exports, 5 terminal, "
                                                        "4 without payouts); D1 forward not computed"}
    res["elapsed_sec"] = round(time.time() - t0, 1)
    (L.OUT / "label_free_band_coverage.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[saved] label_free_band_coverage.json ({res['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
