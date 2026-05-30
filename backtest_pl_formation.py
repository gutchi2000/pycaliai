"""
backtest_pl_formation.py
=========================
フォーメーション戦略 backtest.
各馬券種ごとに:
  1. 全 combo を EV 降順ソート
  2. top K を候補
  3. レース単位 EV = mean(EV_top_K) / BET が閾値以上なら全部買い
  4. 均等割 100 円/combo (Kelly は次版)

閾値スイープ + K スイープ
"""
from __future__ import annotations

import io
import json
import sys
import warnings
from itertools import combinations, permutations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import pl_probs as PL
from backtest_pl_ev import (
    load_payouts, load_wide_payouts, apply_encoders, lookup_pay_vec, score_test,
    all_umaren_mat, all_umatan_mat, all_fukusho_vec_fast, all_sanrenpuku_tensor,
    all_wide_mat,
    BASE, MASTER_CSV, MODEL_PKL, CAL_PKL, CURVE_PKL,
    COL_RID, COL_JYUN, COL_BAN, BET,
)

# wide_map は遅延ロードの module-level singleton
_WIDE_MAP = None
def _get_wide_map():
    global _WIDE_MAP
    if _WIDE_MAP is None:
        print("[load] wide payouts ...")
        _WIDE_MAP = load_wide_payouts()
        print(f"  wide races: {len(_WIDE_MAP):,}")
    return _WIDE_MAP

warnings.filterwarnings("ignore")
# stdout wrapper is set by backtest_pl_ev import

OUT_JSON = BASE / "reports/backtest_pl_formation.json"

# 馬券別 K 候補
K_CONFIG = {
    "tansho":     [1, 2],
    "fukusho":    [1, 2, 3],
    "umaren":     [1, 3, 5],
    "umatan":     [1, 3, 6],
    "sanrenpuku": [1, 6, 10],
    "sanrentan":  [1, 6, 12, 20],
}

# race-level threshold (mean EV per combo / BET)
THRESHOLDS = [1.00, 1.05, 1.10, 1.20, 1.30]


def race_combos(w, po, b2i, cal, curves, rid_s=None, wide_map=None):
    """各馬券の全 combo の (combo_key, p_cal, ev, hit, actual_pay).
    wide の actual pay は wide_map (=load_wide_payouts()) + rid_s から解決。
    未指定なら wide の actual は 0 (ev/hit は常に計算)."""
    n = len(w)
    wi = b2i.get(po["win"]); pi_ = b2i.get(po["plc"]); si = b2i.get(po["sho"])
    top3 = {wi, pi_, si}
    out = {}

    # 単勝
    p_raw = w / w.sum()
    p_cal = cal["tansho"].predict(p_raw)
    pay = lookup_pay_vec(curves["tansho"], p_cal)
    ev = p_cal * pay
    hit = np.zeros(n, dtype=bool); hit[wi] = True
    actual = np.zeros(n)
    actual[wi] = float(po["tansho"]) if po["tansho"] and not pd.isna(po["tansho"]) else 0.0
    out["tansho"] = {"ev": ev, "hit": hit, "actual": actual, "n": n}

    # 複勝
    p_raw = all_fukusho_vec_fast(w)
    p_cal = cal["fukusho"].predict(p_raw)
    pay = lookup_pay_vec(curves["fukusho"], p_cal)
    ev = p_cal * pay
    hit = np.zeros(n, dtype=bool); hit[[wi, pi_, si]] = True
    actual = np.zeros(n)
    for idx, k in [(wi, "fuku_win"), (pi_, "fuku_plc"), (si, "fuku_sho")]:
        if po[k] and not pd.isna(po[k]): actual[idx] = float(po[k])
    out["fukusho"] = {"ev": ev, "hit": hit, "actual": actual, "n": n}

    # 馬連
    iu, ju = np.triu_indices(n, k=1)
    Pmat = all_umaren_mat(w)
    p_raw = Pmat[iu, ju]
    p_cal = cal["umaren"].predict(p_raw)
    pay = lookup_pay_vec(curves["umaren"], p_cal)
    ev = p_cal * pay
    hit = np.array([({int(iu[k]), int(ju[k])} == {wi, pi_}) for k in range(len(iu))])
    actual = hit.astype(float) * (float(po["umaren"]) if po["umaren"] and not pd.isna(po["umaren"]) else 0.0)
    out["umaren"] = {"ev": ev, "hit": hit, "actual": actual, "n": len(iu)}

    # ワイド ({i,j} ⊂ top3, 全 C(n,2) combos)
    top3 = {wi, pi_, si}
    Wmat = all_wide_mat(w)
    p_raw_w = Wmat[iu, ju]
    p_cal_w = cal["wide"].predict(p_raw_w)
    pay_w = lookup_pay_vec(curves["wide"], p_cal_w)
    ev_w = p_cal_w * pay_w
    hit_w = np.array([({int(iu[k]), int(ju[k])} <= top3) for k in range(len(iu))])
    actual_w = np.zeros(len(iu))
    wpairs = wide_map.get(rid_s) if (wide_map is not None and rid_s is not None) else None
    if wpairs:
        for bi, bj, pay_actual in wpairs:
            ii = b2i.get(int(bi)); jj = b2i.get(int(bj))
            if ii is None or jj is None: continue
            a, b = (ii, jj) if ii < jj else (jj, ii)
            # (a,b) に対応する triu インデックスを逆算: idx = a*(2n-a-1)/2 + (b-a-1)
            idx_w = a * (2 * n - a - 1) // 2 + (b - a - 1)
            if 0 <= idx_w < len(iu):
                actual_w[idx_w] = float(pay_actual)
    out["wide"] = {"ev": ev_w, "hit": hit_w, "actual": actual_w, "n": len(iu)}

    # 馬単
    idx_arr = np.arange(n)
    I, J = np.meshgrid(idx_arr, idx_arr, indexing="ij")
    mask = I != J
    Umat = all_umatan_mat(w)
    p_raw = Umat[mask]
    p_cal = cal["umatan"].predict(p_raw)
    pay = lookup_pay_vec(curves["umatan"], p_cal)
    ev = p_cal * pay
    I_f = I[mask]; J_f = J[mask]
    hit = (I_f == wi) & (J_f == pi_)
    actual = hit.astype(float) * (float(po["umatan"]) if po["umatan"] and not pd.isna(po["umatan"]) else 0.0)
    out["umatan"] = {"ev": ev, "hit": hit, "actual": actual, "n": int(mask.sum())}

    # 三連単 / 三連複 共通 tensor
    P3 = all_sanrenpuku_tensor(w)

    # 三連単
    I3, J3, K3 = np.meshgrid(idx_arr, idx_arr, idx_arr, indexing="ij")
    m3 = (I3 != J3) & (J3 != K3) & (I3 != K3)
    p_raw = P3[m3]
    p_cal = cal["sanrentan"].predict(p_raw)
    pay = lookup_pay_vec(curves["sanrentan"], p_cal)
    ev = p_cal * pay
    If = I3[m3]; Jf = J3[m3]; Kf = K3[m3]
    hit = (If == wi) & (Jf == pi_) & (Kf == si)
    actual = hit.astype(float) * (float(po["sanrentan"]) if po["sanrentan"] and not pd.isna(po["sanrentan"]) else 0.0)
    out["sanrentan"] = {"ev": ev, "hit": hit, "actual": actual, "n": int(m3.sum())}

    # 三連複
    tris = np.array(list(combinations(range(n), 3)))
    if len(tris) > 0:
        p_pk = np.zeros(len(tris))
        for perm in permutations(range(3)):
            p_pk += P3[tris[:, perm[0]], tris[:, perm[1]], tris[:, perm[2]]]
        p_cal = cal["sanrenpuku"].predict(p_pk)
        pay = lookup_pay_vec(curves["sanrenpuku"], p_cal)
        ev = p_cal * pay
        win_set = {wi, pi_, si}
        hit = np.array([set(t) == win_set for t in tris])
        actual = hit.astype(float) * (float(po["sanrenpuku"]) if po["sanrenpuku"] and not pd.isna(po["sanrenpuku"]) else 0.0)
        out["sanrenpuku"] = {"ev": ev, "hit": hit, "actual": actual, "n": len(tris)}
    else:
        out["sanrenpuku"] = {"ev": np.array([]), "hit": np.array([], dtype=bool),
                             "actual": np.array([]), "n": 0}

    return out


def simulate(te, payouts, cal, curves, k_config, thresholds):
    # (bet, K, threshold) → stats
    stats = {}
    for bet, Ks in k_config.items():
        for K in Ks:
            for t in thresholds:
                stats[(bet, K, t)] = {"races": 0, "bets": 0, "hits": 0, "stake": 0, "ret": 0.0}

    n_race = 0
    for rid, g in te.groupby(COL_RID, sort=False):
        if len(g) < 3: continue
        rid_s = str(int(rid)) if isinstance(rid, (int, np.integer)) else str(rid)
        if rid_s not in payouts: continue
        po = payouts[rid_s]
        g = g.sort_values(COL_BAN).reset_index(drop=True)
        ban = g[COL_BAN].astype(int).values
        b2i = {int(b): i for i, b in enumerate(ban)}
        if any(b2i.get(po[k]) is None for k in ("win","plc","sho")): continue
        w = PL.pl_weights(g["_score"].values)

        combos = race_combos(w, po, b2i, cal, curves)
        n_race += 1

        for bet, d in combos.items():
            ev = d["ev"]; hit = d["hit"]; actual = d["actual"]
            if len(ev) == 0: continue
            order = np.argsort(-ev)
            for K in k_config[bet]:
                K_use = min(K, len(ev))
                top = order[:K_use]
                ev_top = ev[top]
                hit_top = hit[top]
                actual_top = actual[top]
                # race-level: 平均 EV per combo / BET
                race_roi = ev_top.mean() / BET
                for t in thresholds:
                    key = (bet, K, t)
                    s = stats[key]
                    if race_roi >= t:
                        s["races"] += 1
                        s["bets"] += K_use
                        s["stake"] += K_use * BET
                        s["hits"] += int(hit_top.sum())
                        s["ret"] += float(actual_top.sum())

        if n_race % 500 == 0:
            print(f"  ..{n_race} races")

    return stats, n_race


def summarize(stats, n_race, label):
    print(f"\n=== {label} ({n_race} races) ===")
    # bet / K / threshold
    rows = []
    for (bet, K, t), d in stats.items():
        if d["stake"] == 0:
            continue
        roi = d["ret"] / d["stake"]
        hr  = d["hits"] / max(d["bets"], 1)
        rows.append({
            "bet": bet, "K": K, "threshold": t,
            "races_bet": d["races"], "bets": d["bets"], "hits": d["hits"],
            "stake": d["stake"], "ret": int(d["ret"]),
            "roi": round(roi, 4), "hit_rate": round(hr, 4),
        })
    rows.sort(key=lambda r: (-r["roi"], -r["bets"]))
    # 上位 ROI を印字
    print(f"  ROI 上位 30:")
    print(f"  {'馬券':10s} {'K':>3s} {'thr':>6s} {'races':>6s} {'bets':>7s} {'hits':>6s}  {'hr%':>6s}  {'stake':>10s}  {'ret':>10s}  {'ROI%':>7s}")
    for r in rows[:30]:
        print(f"  {r['bet']:10s} {r['K']:>3d} {r['threshold']:>6.2f} "
              f"{r['races_bet']:>6,} {r['bets']:>7,} {r['hits']:>6,} {r['hit_rate']*100:>6.2f}  "
              f"{r['stake']:>10,}  {r['ret']:>10,}  {r['roi']*100:>7.2f}")
    return rows


def main():
    print("=" * 70)
    print("backtest_pl_formation.py  (race-level EV gate + formation)")
    print("=" * 70)
    payouts = load_payouts()
    cal = joblib.load(CAL_PKL)["calibrators"]
    curves = joblib.load(CURVE_PKL)["curves"]
    te = score_test()

    all_results = {}
    for label, mask in [
        ("2024", te["year"] == 2024),
        ("2025", te["year"] == 2025),
        ("2024+2025", te["year"].isin([2024, 2025])),
    ]:
        print(f"\n>>>  {label}  <<<")
        sub = te[mask]
        stats, n_race = simulate(sub, payouts, cal, curves, K_CONFIG, THRESHOLDS)
        rows = summarize(stats, n_race, label)
        all_results[label] = rows

    OUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "thresholds": THRESHOLDS,
            "K_config":   K_CONFIG,
            "results":    all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {OUT_JSON}")


if __name__ == "__main__":
    main()
