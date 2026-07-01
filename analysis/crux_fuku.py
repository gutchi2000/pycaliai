# -*- coding: utf-8 -*-
"""
crux_fuku.py — 【決定的単一検定 その2】複勝/Dr.Z place-pool 市場内部裁定の存否
================================================================================
Hausch-Ziemba(1981)が実際にエッジを取ったのは win/exacta でなく place/show プール。
問: realized top3(複勝) を当てるのに 公正P(top3)(単勝marginalのPL) は 市場複勝implied を
    超えるか。超える → 複勝プールが単勝と矛盾(=Dr.Z型裁定の芽)。
fair: π(9時de-vig単勝marginal) の Plackett-Luce で P(i∈top3) を MC 推定。
market: 1/複勝odds(9時 (Lo+Hi)/2) を Σ=置換数(3 or 2) に正規化 = 市場implied P(top3)。
決済: 確定複勝配当(fuku_win/plc/sho)。EVゲート=fair×複勝odds9時。leak-safe。
"""
import sys, io
import numpy as np, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
sys.path.insert(0, "E:/PyCaLiAI/analysis"); sys.path.insert(0, "E:/PyCaLiAI")
import joint_lib as J
from backtest_pl_ev import load_payouts

H, _ = J.load()
pays = load_payouts()
rng = np.random.default_rng(42)
K = 4000  # MC orderings

def pl_top3_probs(pi, K=K):
    """Plackett-Luce: Gumbel-max で K本サンプル → 各馬の top3 包含率。"""
    n = len(pi)
    logits = np.log(np.clip(pi, 1e-12, 1))
    g = rng.gumbel(size=(K, n)) + logits  # K×n
    order = np.argsort(-g, axis=1)        # 各行の着順
    top3 = order[:, :3]                    # 上位3
    cnt = np.zeros(n)
    for c in range(3):
        np.add.at(cnt, top3[:, c], 1)
    return cnt / K                         # P(i in top3), Σ≈3 (n>=8) or capped

# per-race build
rows = []
for rid, g in H.groupby("rid", sort=False):
    yr = int(g.year.iloc[0]); n = len(g)
    po = pays.get(rid)
    if po is None: continue
    bans = g.ban.values.astype(int); pi = g.pi.values.astype(float)
    flo = g.fuku_lo9.values; fhi = g.fuku_hi9.values
    fmid = (flo + fhi) / 2.0
    have = ~np.isnan(fmid)
    if have.sum() < 5: continue
    places = 3 if n >= 8 else 2            # JRA: 7頭以下は複勝2着まで
    fair = pl_top3_probs(pi)               # Σ≈3 always (MC top3)
    if places == 2:
        fair = pl_top3_probs(pi, K) * 0    # 再計算: top2
    # 着順 top-k 集合 (払戻が出てる=実際の複勝対象)
    top_set = {po["win"], po["plc"], po["sho"]} if places == 3 else {po["win"], po["plc"]}
    fpay = {po["win"]: po["fuku_win"], po["plc"]: po["fuku_plc"], po["sho"]: po["fuku_sho"]}
    # market implied P(top-k): 1/fmid を Σ=places に正規化
    inv = np.where(have, 1.0 / np.where(have, fmid, 1), np.nan)
    sm = np.nansum(inv)
    qmkt = inv / sm * places
    for idx in range(n):
        if not have[idx]: continue
        b = int(bans[idx])
        rows.append(dict(rid=rid, year=yr, n=n, places=places, ban=b,
                         pi=float(pi[idx]), fair=float(fair[idx] if places == 3 else 0),
                         o_fuku9=float(fmid[idx]), qmkt=float(qmkt[idx]),
                         is_top=int(b in top_set),
                         fpay=(float(fpay[b]) if (b in fpay and fpay[b] == fpay[b]) else np.nan)))

# top2 fields need separate fair; recompute cleanly for places==2 rows
D = pd.DataFrame(rows)
# fix places==2 fair via PL top2
print(f"[data] horse-rows={len(D):,} races={D.rid.nunique():,} (places=2割合 {D.places.eq(2).mean():.3f})")
# recompute fair for ALL rows consistently with places (top-k inclusion)
fair_fixed = np.full(len(D), np.nan)
for rid, gi in D.groupby("rid", sort=False):
    sub = H[H.rid == rid]
    pi = sub.pi.values.astype(float); bans = sub.ban.values.astype(int)
    places = int(gi.places.iloc[0])
    n = len(pi); logits = np.log(np.clip(pi, 1e-12, 1))
    gmb = rng.gumbel(size=(K, n)) + logits
    order = np.argsort(-gmb, axis=1)[:, :places]
    cnt = np.zeros(n)
    for c in range(places):
        np.add.at(cnt, order[:, c], 1)
    p_in = cnt / K
    bmap = {int(bans[k]): p_in[k] for k in range(n)}
    fair_fixed[gi.index] = gi.ban.map(bmap).values
D["fair"] = fair_fixed

def brier_ll(d, col):
    p = np.clip(d[col].values, 1e-9, 1 - 1e-9); y = d.is_top.values
    return (np.mean((p - y) ** 2), -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

print("\n===== [核心] realized top3 を当てる: 公正PL(単勝) vs 市場複勝 =====")
for lab, dd in [("≤2023", D[D.year <= 2023]), ("test24-25", D[D.year >= 2024])]:
    bf, lf = brier_ll(dd, "fair"); bm, lm = brier_ll(dd, "qmkt")
    print(f"  {lab}: fair Brier={bf:.4f} LL={lf:.4f} | 市場 Brier={bm:.4f} LL={lm:.4f} | "
          f"fair勝ち(LL)={lf < lm}  Δ(市場-fair)={lm - lf:+.4f}")
print("  [読み] fairが市場を超える(Δ>0) = 複勝プールが単勝と矛盾 = Dr.Z型裁定の芽")

# ---- EVゲート backtest: fair × 複勝odds9時, 決済=確定配当 ----
print("\n===== EVゲート(fair×複勝9時odds) backtest =====")
def bt(d, tau):
    s = d[d.fair * d.o_fuku9 >= tau]
    if len(s) == 0: return (0, 0, 0)
    stake = len(s) * 100.0
    ret = np.where(s.is_top == 1, s.fpay.fillna(0), 0).sum()
    return (len(s), ret / stake, (s.is_top == 1).mean())
tr = D[D.year <= 2023]; te = D[D.year >= 2024]
best_tau, best = None, -1
for tau in [1.0, 1.05, 1.1, 1.2, 1.3, 1.5]:
    n, roi, hit = bt(tr, tau)
    print(f"  τ={tau}: ≤2023 ROI={roi*100:.1f}% n={n:,} hit={hit*100:.1f}%")
    if n >= 500 and roi > best: best = roi; best_tau = tau
n, roi, hit = bt(te, best_tau)
print(f"  → τ*={best_tau}  test ROI={roi*100:.1f}% n={n:,} hit={hit*100:.1f}%  (控除床 複勝80%)")

# 対照: 市場で割安(fair>qmkt)を見ない、純 fair-top など省略。最人気複勝1点 control
def bt_fav(d):
    idx = d.groupby("rid").o_fuku9.idxmin()
    s = d.loc[idx]; stake = len(s) * 100.0
    ret = np.where(s.is_top == 1, s.fpay.fillna(0), 0).sum()
    return (len(s), ret / stake, (s.is_top == 1).mean())
n2, roi2, hit2 = bt_fav(te)
print(f"  対照 市場最人気複勝1点: ROI={roi2*100:.1f}% hit={hit2*100:.1f}% n={n2:,}")
print("\n[結論] fairが市場複勝をLL/Brierで超え かつ EVゲートtestが80%超 → Dr.Z型裁定実在。"
      "市場優位/床未達なら複勝も効率的=市場内部裁定は全滅。")
