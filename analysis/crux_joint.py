# -*- coding: utf-8 -*-
"""
crux_joint.py — 【決定的単一検定】市場馬連は市場単勝と矛盾するか(Harville裁定の存否)
================================================================================
問: realized top2(馬連)を当てるのに Harville(単勝marginal) は 市場馬連de-vig を超えるか。
  超える → 単勝プールに馬連プールが値付けし損ねたペア情報がある = 裁定の芽。
  超えない → 馬連プールは単勝と整合(効率的) = この路も死亡。
さらに: Harville公正 で EVゲート → 9時で賭け確定で決済、CLV符号、控除床(77.5%)比較。
leak-safe: λ選定/τ選定=≤2023、評価=test 2024-25。
"""
import sys, io
import numpy as np, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
sys.path.insert(0, "E:/PyCaLiAI/analysis"); sys.path.insert(0, "E:/PyCaLiAI")
import joint_lib as J

H, P = J.load()
print(f"[data] horses={len(H):,} races={H.rid.nunique():,} pairs={len(P):,}")

# ---- λ フィット (≤2023) ----
lam, curve = J.fit_lambda(H, 2023)
print(f"\n[λ fit ≤2023] best λ={lam}  curve(λ,logloss)={curve}")

# ---- per-race pack: 馬連 ----
# pi/bans を rid→dict 化
hg = {rid: (g.ban.values.astype(int), g.pi.values.astype(float),
            int(g.ban[g.is_win == 1].iloc[0]) if (g.is_win == 1).any() else -1,
            int(g.ban[g.is_plc == 1].iloc[0]) if (g.is_plc == 1).any() else -1,
            int(g.year.iloc[0]))
      for rid, g in H.groupby("rid", sort=False)}

# 馬連odds: rid→ {(i,j): (o9, ofin, pay, iswin)}
print("[build] 馬連 per-race odds map ...")
umap = {}
for rid, g in P.groupby("rid", sort=False):
    d = {}
    for r in g.itertuples():
        d[(int(r.i), int(r.j))] = (float(r.o_um9),
                                   float(r.o_umfin) if r.o_umfin == r.o_umfin else np.nan,
                                   float(r.umaren_pay) if r.umaren_pay == r.umaren_pay else np.nan,
                                   int(r.is_winpair))
    umap[rid] = d

# ---- logloss 対決 (Harville-fair vs 市場de-vig) + backtest ----
def evaluate(yr_lo, yr_hi, lam, tau=None):
    ll_h = ll_m = 0.0; nrace = 0
    # backtest accumulators (per τ done by caller; here just gather rows if tau given)
    stake = ret = hits = bets = 0.0; clv = []
    ctrl_stake = ctrl_ret = ctrl_hits = ctrl_bets = 0.0  # market top-1 control
    for rid, (bans, pi, wb, pb, yr) in hg.items():
        if yr < yr_lo or yr > yr_hi: continue
        if rid not in umap or wb < 0 or pb < 0: continue
        wk = (min(wb, pb), max(wb, pb))
        od = umap[rid]
        # Harville fair over pairs that exist in market
        ph = J.harville_pairs(pi, bans, lam)
        keys = [k for k in ph if k in od]
        if not keys or wk not in od: continue
        fair = np.array([ph[k] for k in keys]); fair = fair / fair.sum()
        inv = np.array([1.0 / od[k][0] for k in keys]); qmkt = inv / inv.sum()
        fair_d = dict(zip(keys, fair)); q_d = dict(zip(keys, qmkt))
        # logloss of realized winpair
        ll_h += -np.log(max(fair_d.get(wk, 1e-12), 1e-12))
        ll_m += -np.log(max(q_d.get(wk, 1e-12), 1e-12))
        nrace += 1
        if tau is not None:
            # EV gate with Harville-fair × 9時odds
            for k in keys:
                o9, ofin, pay, isw = od[k]
                ev = fair_d[k] * o9
                if ev >= tau:
                    stake += 100; bets += 1
                    if isw:
                        payout = pay if pay == pay else (ofin * 100 if ofin == ofin else o9 * 100)
                        ret += payout; hits += 1
                    if ofin == ofin and ofin > 0:
                        clv.append(o9 / ofin - 1.0)  # >0 = 9時で高odds=締まった=賢い
            # control: market favorite pair (lowest 馬連odds = highest q) 1点
            kbest = min(keys, key=lambda k: od[k][0])
            o9, ofin, pay, isw = od[kbest]
            ctrl_stake += 100; ctrl_bets += 1
            if isw:
                ctrl_ret += (pay if pay == pay else o9 * 100); ctrl_hits += 1
    res = {"n_races": nrace, "logloss_harville": round(ll_h / max(nrace, 1), 4),
           "logloss_market": round(ll_m / max(nrace, 1), 4),
           "harville_beats_market": ll_h < ll_m,
           "delta_logloss": round((ll_m - ll_h) / max(nrace, 1), 5)}
    if tau is not None:
        res["bt"] = {"tau": tau, "bets": int(bets), "roi": round(ret / stake, 4) if stake else 0,
                     "hit": round(hits / bets, 4) if bets else 0,
                     "mean_clv": round(float(np.mean(clv)), 4) if clv else None,
                     "ctrl_favpair_roi": round(ctrl_ret / ctrl_stake, 4) if ctrl_stake else 0,
                     "ctrl_favpair_hit": round(ctrl_hits / ctrl_bets, 4) if ctrl_bets else 0}
    return res

print("\n===== [核心] realized top2 logloss: Harville(単勝) vs 市場馬連 =====")
r_tr = evaluate(2013, 2023, lam)
r_te = evaluate(2024, 2025, lam)
print(f"  ≤2023: Harville={r_tr['logloss_harville']} 市場={r_tr['logloss_market']} "
      f"Δ(市場-H)={r_tr['delta_logloss']:+.5f} Harville勝ち={r_tr['harville_beats_market']}")
print(f"  test : Harville={r_te['logloss_harville']} 市場={r_te['logloss_market']} "
      f"Δ(市場-H)={r_te['delta_logloss']:+.5f} Harville勝ち={r_te['harville_beats_market']}")
print("  [読み] Δ>0 かつ test で Harville勝ち = 単勝プールに馬連が取りこぼした情報 = 裁定の芽")

# ---- τ を ≤2023 で選び test 適用 (EVゲート backtest) ----
print("\n===== EVゲート(Harville-fair × 9時馬連odds) backtest =====")
print("  ≤2023 で τ sweep → ROI最良(n>=500)を test 適用:")
best_tau = None; best_roi = -1
for tau in [1.0, 1.1, 1.2, 1.3, 1.5, 2.0]:
    rr = evaluate(2013, 2023, lam, tau)
    bt = rr["bt"]
    print(f"    τ={tau}: ≤2023 ROI={bt['roi']*100:.1f}% n={bt['bets']:,} hit={bt['hit']*100:.1f}% CLV={(bt['mean_clv'] or 0)*100:+.1f}%")
    if bt["bets"] >= 500 and bt["roi"] > best_roi:
        best_roi = bt["roi"]; best_tau = tau
print(f"  → 選定 τ*={best_tau}")
te = evaluate(2024, 2025, lam, best_tau)["bt"]
print(f"  test ROI={te['roi']*100:.1f}% n={te['bets']:,} hit={te['hit']*100:.1f}% "
      f"CLV={(te['mean_clv'] or 0)*100:+.1f}%  (控除床 馬連77.5%)")
print(f"  対照 市場最人気ペア1点: ROI={te['ctrl_favpair_roi']*100:.1f}% hit={te['ctrl_favpair_hit']*100:.1f}%")
print("\n[結論メモ] test ROI が 77.5% を CLV>0 で超えれば市場内部裁定の実在。"
      "床近傍/CLV<0 なら馬連も効率的=この路も墓標。")
