# -*- coding: utf-8 -*-
"""
_joint_m5_check.py — m5: 親の2cruxを独立再現 + ワイド市場内部裁定。
全て自前ロジック(joint_lib は load/pl_topk_probs/pl_pair_both_topk/harville_pairs のみ流用)。
leak-safe: 決定=9時odds。決済=確定。tune≤2023, eval=2024-2025。
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "E:/PyCaLiAI/analysis")
import joint_lib as J

def out(*a):
    print(*a, flush=True)

RNG = np.random.default_rng(20260619)
H, P = J.load()

def places_for_n(n):
    return 3 if n >= 8 else 2

# precompute per-race H groups once
out("indexing...")
Hg = {rid: g for rid, g in H.groupby("rid", sort=False)}
Pg = {rid: g for rid, g in P.groupby("rid", sort=False)}
RIDS = list(Pg.keys())
YR = {rid: int(Pg[rid].year.iloc[0]) for rid in RIDS}
out("races(P)=", len(RIDS))

# ---- own discount-Harville pair prob for a single winpair (cheap, no full dict) ----
def harv_winpair_prob(pi, bans, lam, wkey):
    """P_unordered(top2 = wkey) under discount-Harville, normalized over all pairs.
    norm = Σ_pairs p_ij. For Harville Σ over ordered = Σ_i π_i * (Σ_{k≠i} s_k)/(S - s_i).
    We compute the single winpair numerator and the full normalizer."""
    pi = np.asarray(pi, float); n = len(pi)
    s = np.power(np.clip(pi, 1e-12, 1), lam)
    S = s.sum()
    # normalizer over all unordered pairs = Σ_i π_i (since 2nd-place dist sums to 1 given 1st=i)
    # P(i 1st) = π_i ; Σ_j!=i P(j 2nd|i)=1 -> Σ ordered = Σπ_i. unordered same total.
    norm = pi.sum()
    a = int(np.where(bans == wkey[0])[0][0]); b = int(np.where(bans == wkey[1])[0][0])
    da = S - s[a]; db = S - s[b]
    p_ab = pi[a]*(s[b]/da if da > 0 else 0)
    p_ba = pi[b]*(s[a]/db if db > 0 else 0)
    return (p_ab + p_ba)/max(norm, 1e-12)

# ============================================================
# CRUX (a) 馬連 realized top2 logloss : 市場de-vig vs Harville(単勝)
# ============================================================
out("="*70)
out("CRUX(a) 馬連 realized top2 logloss : 市場de-vig vs Harville(単勝marginal)")
out("="*70)

# precompute winpair + arrays for races that have a winpair
PACK = {}
for rid in RIDS:
    pr = Pg[rid]
    wp = pr[pr.is_winpair == 1]
    if len(wp) == 0:
        continue
    hr = Hg.get(rid)
    if hr is None:
        continue
    wkey = (int(min(wp.i.iloc[0], wp.j.iloc[0])), int(max(wp.i.iloc[0], wp.j.iloc[0])))
    bset = set(hr.ban.values.astype(int).tolist())
    if wkey[0] not in bset or wkey[1] not in bset:
        continue
    PACK[rid] = dict(bans=hr.ban.values.astype(int), pi=hr.pi.values.astype(float),
                     wkey=wkey, o_um9=pr.o_um9.values, i=pr.i.values.astype(int),
                     j=pr.j.values.astype(int), year=YR[rid])

# λ tune on ≤2023
grid = [0.5, 0.7, 0.9, 1.0, 1.1, 1.2]
tune_rids = [r for r in PACK if PACK[r]["year"] <= 2023]
out(f"λ tune races(≤2023)={len(tune_rids)} grid={grid}")
curve = []
for lam in grid:
    tot = 0.0; c = 0
    for r in tune_rids:
        d = PACK[r]
        pw = harv_winpair_prob(d["pi"], d["bans"], lam, d["wkey"])
        tot += -np.log(max(pw, 1e-12)); c += 1
    curve.append((lam, round(tot/c, 5)))
lam_star = min(curve, key=lambda x: x[1])[0]
out("λ curve:", curve, "-> λ*=", lam_star)

test_rids = [r for r in PACK if PACK[r]["year"] >= 2024]
h_tot = m_tot = 0.0; c = 0
for r in test_rids:
    d = PACK[r]
    pw_h = harv_winpair_prob(d["pi"], d["bans"], lam_star, d["wkey"])
    inv = 1.0/d["o_um9"]; q = inv/inv.sum()
    qd = {(int(a), int(b)): float(v) for a, b, v in zip(d["i"], d["j"], q)}
    pw_m = qd.get(d["wkey"], 1e-12)
    h_tot += -np.log(max(pw_h, 1e-12)); m_tot += -np.log(max(pw_m, 1e-12)); c += 1
out(f"test 2024-25 races={c}")
out(f"  Harville(単勝) LL = {h_tot/c:.4f}")
out(f"  市場馬連de-vig LL = {m_tot/c:.4f}")
out(f"  親: 市場3.343 < H3.380 (市場優位=馬連死亡)")
A_market_wins = (m_tot/c) < (h_tot/c)
out(f"  判定: {'市場優位=親と一致(馬連死亡)' if A_market_wins else 'Harville優位=親と不一致!'}")

# ============================================================
# CRUX (b) 複勝 realized top3 per-horse logloss : fair PL(単勝) vs 市場複勝
# ============================================================
out("\n"+"="*70)
out("CRUX(b) 複勝 per-horse logloss : fair PL(単勝) vs 市場複勝implied")
out("="*70)
f_tot = m_tot = 0.0; cnt = 0
for rid, g in Hg.items():
    if YR[rid] < 2024:
        continue
    n = int(g.n.iloc[0]); k = places_for_n(n)
    pi = g.pi.values.astype(float)
    fmid = (g.fuku_lo9.values.astype(float) + g.fuku_hi9.values.astype(float))/2.0
    if np.any(~np.isfinite(fmid)) or np.any(fmid <= 0):
        continue
    y3 = g.top3.values.astype(float)
    fp = np.clip(J.pl_topk_probs(pi, k, K=4000, rng=RNG), 1e-6, 1-1e-6)
    imp = 1.0/fmid; mp = np.clip(imp/imp.sum()*k, 1e-6, 1-1e-6)
    f_tot += -np.sum(y3*np.log(fp)+(1-y3)*np.log(1-fp))
    m_tot += -np.sum(y3*np.log(mp)+(1-y3)*np.log(1-mp))
    cnt += n
out(f"test 2024-25 horse-rows={cnt}")
out(f"  fair PL(単勝) per-horse LL = {f_tot/cnt:.4f}")
out(f"  市場複勝implied per-horse LL = {m_tot/cnt:.4f}")
out(f"  親: fair0.4154 < 市場0.4287 (fair優位=Dr.Z署名)")
B_fair_wins = (f_tot/cnt) < (m_tot/cnt)
out(f"  判定: {'fair優位=親と一致(Dr.Z陽性)' if B_fair_wins else '市場優位=親と不一致!'}")

# ============================================================
# (2) ワイド: fair P(両方top3) で fair-top選択の test ROI (payout-only上限)
#   ワイドodds時系列無し → 確定wide payout(/100)を使う上限評価。
#   戦略: 各レースで fair pair prob 上位 m 点を1点100円買い、当たれば payout円。
#   ROI = Σpayout / Σ(100*m_bets)。tune(m)≤2023, eval=2024-25。
# ============================================================
out("\n"+"="*70)
out("(2) ワイド fair-top選択 ROI (payout-only上限評価)")
out("="*70)
W = pd.read_parquet("data/wide_payouts_2016-2025.parquet")
# winning wide pairs per race: dict rid -> {(min,max):pay}
wmap = {}
for row in W.itertuples(index=False):
    rid = str(row.race_id)
    d = wmap.setdefault(rid, {})
    for (a, b, pay) in [(row.w1_i, row.w1_j, row.w1_pay),
                        (row.w2_i, row.w2_j, row.w2_pay),
                        (row.w3_i, row.w3_j, row.w3_pay)]:
        d[(int(min(a, b)), int(max(a, b)))] = float(pay)
out("wide payout races=", len(wmap))

def wide_roi(rids, m):
    inv = pay = 0.0; hit = 0; nb = 0
    for rid in rids:
        g = Hg.get(rid)
        if g is None or rid not in wmap:
            continue
        n = int(g.n.iloc[0])
        if n < 3:
            continue
        pi = g.pi.values.astype(float); bans = g.ban.values.astype(int)
        fpairs = J.pl_pair_both_topk(pi, bans, k=3, K=4000, rng=RNG)
        top = sorted(fpairs.items(), key=lambda kv: -kv[1])[:m]
        win = wmap[rid]
        for key, _ in top:
            inv += 100.0; nb += 1
            if key in win:
                pay += win[key]; hit += 1
    roi = pay/inv if inv > 0 else 0
    return roi, hit, nb, inv

# tune m on ≤2023
wide_rids = [r for r in Hg if r in wmap]
tr = [r for r in wide_rids if YR[r] <= 2023]
te = [r for r in wide_rids if YR[r] >= 2024]
out(f"wide tune races={len(tr)} test races={len(te)}")
best_m = None; best_roi = -1
for m in [1, 2, 3]:
    roi, hit, nb, inv = wide_roi(tr, m)
    out(f"  ≤2023 m={m}: ROI={roi:.4f} hit={hit}/{nb} ({hit/nb*100:.1f}%)")
    if roi > best_roi:
        best_roi = roi; best_m = m
out(f"  -> m*={best_m} (≤2023 best)")
roi, hit, nb, inv = wide_roi(te, best_m)
out(f"test 2024-25 m={best_m}: ROI={roi:.4f} hit={hit}/{nb} ({hit/nb*100:.1f}%) bets={nb}")

# bootstrap CI on test ROI (per-race resample)
out("bootstrap 95%CI (per-race)...")
# build per-race (inv,pay) for test at m*
recs = []
for rid in te:
    g = Hg.get(rid)
    if g is None:
        continue
    n = int(g.n.iloc[0])
    if n < 3:
        continue
    pi = g.pi.values.astype(float); bans = g.ban.values.astype(int)
    fpairs = J.pl_pair_both_topk(pi, bans, k=3, K=4000, rng=RNG)
    top = sorted(fpairs.items(), key=lambda kv: -kv[1])[:best_m]
    win = wmap[rid]
    ri = 100.0*len(top); rp = sum(win.get(k, 0.0) for k, _ in top)
    recs.append((ri, rp))
recs = np.array(recs)
bs = []
rng2 = np.random.default_rng(7)
for _ in range(2000):
    idx = rng2.integers(0, len(recs), len(recs))
    s = recs[idx]
    bs.append(s[:, 1].sum()/s[:, 0].sum())
lo, hi = np.percentile(bs, [2.5, 97.5])
out(f"  test ROI={recs[:,1].sum()/recs[:,0].sum():.4f}  95%CI=[{lo:.4f},{hi:.4f}]")
out("DONE")
