# -*- coding: utf-8 -*-
"""
wide_backtest.py — ワイド / 馬連 / 3:1コンボ の OOS比較 (2025, prob-firstペア)
============================================================================
同じペア(prob-first top-k, umaren prob順)に対し:
  馬連単: 各ペア100円
  ワイド単: 各ペア100円
  3:1コンボ: ワイド300+馬連100 /ペア
を k=3〜10 で settle。ROI/的中を正面比較。
"""
import sys, io
from pathlib import Path
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
BASE = Path(r"E:\PyCaLiAI"); sys.path.insert(0, str(BASE))
import backtest_pl_ev as be
be.MODEL_PKL = BASE / "models/unified_rank_v6.pkl"
be.CAL_PKL = BASE / "models/pl_calibrators_v6.pkl"
import pl_probs as PL
from backtest_pl_ev import all_umaren_mat, load_payouts, load_wide_payouts, COL_RID, COL_BAN

print("[load] score 2025 + payouts + wide payouts")
te = be.score_test()
te = te[te["year"] == 2025].copy()
pays = load_payouts(); wides = load_wide_payouts()
print(f"  2025 races={te[COL_RID].nunique():,}  wide payout races={len(wides):,}")

KS = [3, 4, 5, 6, 7, 8, 10]
# per k: {strat: [stake, ret, hit_races, nrace]}
agg = {k: {s: [0.0, 0.0, 0, 0] for s in ("um", "wd", "cb")} for k in KS}
n_wide_match = 0; n_race = 0
for rid, g in te.groupby(COL_RID, sort=False):
    if len(g) < 5: continue
    rid_s = str(int(rid)) if isinstance(rid, (int, np.integer)) else str(rid)
    po = pays.get(rid_s)
    if po is None: continue
    g = g.sort_values(COL_BAN); ban = g[COL_BAN].astype(int).values; n = len(ban)
    w = PL.pl_weights(g["_score"].values)
    M = all_umaren_mat(w); iu, ju = np.triu_indices(n, 1); pr = M[iu, ju]
    order = np.argsort(-pr)
    pairs = [(int(ban[iu[o]]), int(ban[ju[o]])) for o in order]
    win_pair = frozenset((int(po["win"]), int(po["plc"])))
    um_pay = float(po["umaren"]) if (po["umaren"] == po["umaren"] and po["umaren"]) else 0.0
    wd = {frozenset((int(bi), int(bj))): float(pay) for bi, bj, pay in wides.get(rid_s, [])}
    if wd: n_wide_match += 1
    n_race += 1
    for k in KS:
        top = pairs[:k]
        um_r = wd_r = cb_r = 0.0; um_h = wd_h = cb_h = 0
        for (bi, bj) in top:
            fs = frozenset((bi, bj))
            uh = (fs == win_pair); wp = wd.get(fs, 0.0)
            if uh: um_r += um_pay
            if wp: wd_r += wp
            cb_r += 3 * wp + (um_pay if uh else 0.0)
            um_h += uh; wd_h += (wp > 0); cb_h += (uh or wp > 0)
        a = agg[k]
        a["um"][0] += k * 100; a["um"][1] += um_r; a["um"][2] += um_h > 0; a["um"][3] += 1
        a["wd"][0] += k * 100; a["wd"][1] += wd_r; a["wd"][2] += wd_h > 0; a["wd"][3] += 1
        a["cb"][0] += k * 400; a["cb"][1] += cb_r; a["cb"][2] += cb_h > 0; a["cb"][3] += 1

print(f"  wide払戻ヒットしたrace={n_wide_match:,}/{n_race:,} (低いとkey不一致)\n")
print(f"{'k点':>4} | {'馬連単 ROI/的中':>18} | {'ワイド単 ROI/的中':>19} | {'3:1コンボ ROI/的中':>20}")
print("-" * 72)
for k in KS:
    a = agg[k]
    def fmt(s):
        st, rt, hr, nr = a[s]
        return f"{rt/st*100:5.1f}% / {hr/nr*100:4.1f}%"
    print(f"{k:>4} | {fmt('um'):>18} | {fmt('wd'):>19} | {fmt('cb'):>20}")
print("\n[読み] コンボROI = (3×ワイド + 1×馬連)/4 の加重平均(=数学的にそうなる)。"
      "ワイド的中は馬連の数倍。ROIで勝つのは良い方の券種100%。コンボは“当てる回数↑/分散↓”の選択。")
