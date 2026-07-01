# -*- coding: utf-8 -*-
"""
m2 — Dr.Z 型 複勝ダッチング(ポートフォリオ法)。
各レースで fair_p(PL top-k, 単勝marginal由来) x 複勝odds(9時中値) >= tau の +EV複勝馬を全部拾い、
(a)flat (b)EV比例 (c)fractional-Kelly(f) でレース内配分し、レース単位ポートフォリオを構築。
test=2024-25 の ROI / 資金成長(対数効用) / 破産確率 / bootstrap95%CI を出す。
比較: 単一最良EV馬 / 市場最人気複勝1点 control。
leak-safe: 決定=9時odds(pi,fuku_lo9/hi9)。決済=fpay(確定複勝配当/100, notna=的中)。
tune(tau,f)<=2023, eval=test only.
"""
import sys; sys.path.insert(0, "E:/PyCaLiAI/analysis"); import joint_lib as J
import numpy as np, pandas as pd

RNG = np.random.default_rng(7)
H, _ = J.load()

# ---- per-race pack (leak-safe decision inputs + settlement) ----
def build_races(H):
    races = []
    for rid, g in H.groupby("rid", sort=False):
        n = int(g.n.iloc[0]); year = int(g.year.iloc[0])
        pi = g.pi.values.astype(float)
        lo = g.fuku_lo9.values.astype(float); hi = g.fuku_hi9.values.astype(float)
        mid = (lo + hi) / 2.0
        bans = g.ban.values.astype(int)
        fpay = g.fpay.values.astype(float)  # NaN unless placed-paid
        # places: 3 if n>=8 else 2
        k = 3 if n >= 8 else 2
        ok = np.isfinite(mid) & (mid > 1.0) & np.isfinite(pi) & (pi > 0)
        if ok.sum() < k + 1:
            continue
        fairp = J.pl_topk_probs(pi, k, K=4000, rng=RNG)  # P(top-k incl)
        won = np.isfinite(fpay)            # placed & paid
        ret = np.where(won, np.nan_to_num(fpay) / 100.0, 0.0)  # payout multiple per 1 stake
        races.append(dict(rid=rid, year=year, n=n, k=k, bans=bans,
                          pi=pi, mid=mid, fairp=fairp, ok=ok, won=won, ret=ret))
    return races

print("building races...", flush=True)
RACES = build_races(H)
print("races:", len(RACES), flush=True)

LE23 = [r for r in RACES if r["year"] <= 2023]
TEST = [r for r in RACES if r["year"] >= 2024]
print("le23:", len(LE23), "test:", len(TEST), flush=True)


# ---- allocation methods over the +EV set in a race ----
def select_ev(r, tau):
    """indices with fair_p*mid >= tau and odds finite."""
    ev = r["fairp"] * r["mid"]
    sel = np.where(r["ok"] & (ev >= tau))[0]
    return sel, ev


def race_pnl(r, tau, method, f=0.25, budget=1.0):
    """Return (staked, returned, n_bets). budget=total stake per race (normalized)."""
    sel, ev = select_ev(r, tau)
    if len(sel) == 0:
        return 0.0, 0.0, 0
    p = r["fairp"][sel]; o = r["mid"][sel]; ret = r["ret"][sel]
    if method == "flat":
        w = np.ones(len(sel))
    elif method == "evprop":
        w = np.clip(ev[sel] - 1.0, 1e-9, None)   # proportional to edge
    elif method == "best":
        b = np.argmax(ev[sel]); w = np.zeros(len(sel)); w[b] = 1.0
    elif method.startswith("kelly"):
        # per-horse Kelly fraction for a place bet: b=o-1, edge p*o-1
        b = o - 1.0
        kf = (p * o - 1.0) / np.clip(b, 1e-9, None)
        kf = np.clip(kf, 0.0, None) * f
        w = kf
        if w.sum() <= 0:
            return 0.0, 0.0, 0
    else:
        raise ValueError(method)
    if w.sum() <= 0:
        return 0.0, 0.0, 0
    if method.startswith("kelly"):
        # kelly: stake = fraction of bankroll; here normalize to budget cap but keep relative sizing
        stake = w * budget  # w already fractional; budget is bankroll fraction cap multiplier
        # cap total at budget
        if stake.sum() > budget:
            stake = stake * (budget / stake.sum())
    else:
        stake = w / w.sum() * budget
    staked = stake.sum()
    returned = float((stake * ret).sum())
    return staked, returned, int(len(sel))


def portfolio_roi(races, tau, method, f=0.25):
    S = R = 0.0; nb = 0; nr = 0
    per_s = []; per_r = []
    for r in races:
        s, ret, k = race_pnl(r, tau, method, f=f, budget=1.0)
        if s > 0:
            S += s; R += ret; nb += k; nr += 1
            per_s.append(s); per_r.append(ret)
    roi = R / S if S > 0 else np.nan
    return dict(roi=roi, S=S, R=R, nbets=nb, nraces=nr,
                per_s=np.array(per_s), per_r=np.array(per_r))


def boot_ci(per_s, per_r, n=2000):
    """bootstrap 95%CI of ROI = sum(returned)/sum(staked), resampling races."""
    per_s = np.asarray(per_s); per_r = np.asarray(per_r)
    if len(per_s) == 0:
        return (np.nan, np.nan)
    m = len(per_s); rois = []
    for _ in range(n):
        idx = RNG.integers(0, m, m)
        ss = per_s[idx].sum()
        rois.append(per_r[idx].sum() / ss if ss > 0 else np.nan)
    lo, hi = np.nanpercentile(rois, [2.5, 97.5])
    return (lo, hi)


# ---- bankroll growth simulation (log utility / ruin) ----
def bankroll_sim(races, tau, method, f, bet_frac=0.02, start=1.0, ruin=0.05):
    """Sequential bankroll. each race stake = bet_frac*bankroll allocated by method.
    kelly uses its own fraction (f) of bankroll instead of bet_frac."""
    bk = start; lo = start; ruined = False
    log_growth = 0.0; nb = 0
    for r in races:
        sel, ev = select_ev(r, tau)
        if len(sel) == 0:
            continue
        if method.startswith("kelly"):
            # kelly fractions scale to bankroll, but cap total race exposure at 10% bk
            s, ret, k = race_pnl(r, tau, method, f=f, budget=0.10 * bk)
        else:
            cap = bet_frac * bk
            s, ret, k = race_pnl(r, tau, method, f=f, budget=cap)
        bk = bk - s + ret
        nb += k
        if bk < ruin * start and not ruined:
            ruined = True
        if bk <= 0:
            bk = 1e-9
        lo = min(lo, bk)
    return dict(final=bk, log=np.log(max(bk, 1e-9) / start), minbk=lo, ruined=ruined, nb=nb)


# ---- market favorite control: cheapest fuku odds horse, flat 1pt ----
def control_fav(races):
    S = R = 0.0; per_r = []
    for r in races:
        idx = np.where(r["ok"])[0]
        if len(idx) == 0: continue
        b = idx[np.argmin(r["mid"][idx])]
        S += 1.0; R += r["ret"][b]; per_r.append(r["ret"][b])
    return dict(roi=R/S if S>0 else np.nan, S=S, R=R,
                per_s=np.ones(len(per_r)), per_r=np.array(per_r))


# ================= TUNE on <=2023 =================
print("\n===== TUNE (<=2023) =====", flush=True)
taus = [1.00, 1.02, 1.05, 1.08, 1.10, 1.15, 1.20]
methods = ["flat", "evprop", "best"]
kf_list = [0.25, 0.5]
tune_rows = []
for tau in taus:
    for m in methods:
        res = portfolio_roi(LE23, tau, m)
        tune_rows.append((m, tau, None, res["roi"], res["nbets"], res["nraces"]))
    for f in kf_list:
        res = portfolio_roi(LE23, tau, "kelly", f=f)
        tune_rows.append(("kelly", tau, f, res["roi"], res["nbets"], res["nraces"]))
td = pd.DataFrame(tune_rows, columns=["method","tau","f","le23_roi","nbets","nraces"])
print(td.to_string(index=False), flush=True)

# pick best per method-family by le23 roi (with min coverage)
td_cov = td[td.nraces >= 500]
best_overall = td_cov.sort_values("le23_roi", ascending=False).iloc[0]
print("\nbest le23 cfg:", best_overall.to_dict(), flush=True)


# ================= EVALUATE on test (configs chosen on <=2023) =================
print("\n===== TEST (2024-25) =====", flush=True)
# evaluate a representative grid of configs but the HEADLINE = best-le23 cfg only
eval_cfgs = []
# best per family
for fam in ["flat","evprop","best","kelly"]:
    sub = td_cov[td_cov.method==fam]
    if len(sub)==0: continue
    bb = sub.sort_values("le23_roi", ascending=False).iloc[0]
    eval_cfgs.append((bb.method, float(bb.tau), bb.f, float(bb.le23_roi)))

rows = []
for (m, tau, f, le_roi) in eval_cfgs:
    res = portfolio_roi(TEST, tau, m, f=f if f==f else 0.25)
    ci = boot_ci(res["per_s"], res["per_r"])
    rows.append(dict(method=m, tau=tau, f=f, le23_roi=round(le_roi,4),
                     test_roi=round(res["roi"],4), test_n=res["nbets"], test_nraces=res["nraces"],
                     ci_lo=round(ci[0],4), ci_hi=round(ci[1],4)))
ed = pd.DataFrame(rows)
print(ed.to_string(index=False), flush=True)

# control
ctl = control_fav(TEST)
ctl_ci = boot_ci(ctl["per_s"], ctl["per_r"])
print(f"\nCONTROL market-fav fuku 1pt flat: test ROI={ctl['roi']:.4f} n={int(ctl['S'])} CI[{ctl_ci[0]:.4f},{ctl_ci[1]:.4f}]", flush=True)

# ================= Bankroll growth for headline configs =================
print("\n===== BANKROLL GROWTH (test, sequential) =====", flush=True)
for (m, tau, f, le_roi) in eval_cfgs:
    bk = bankroll_sim(TEST, tau, m, f if f==f else 0.25, bet_frac=0.02)
    print(f"{m:7s} tau={tau} f={f}: final={bk['final']:.3f} log={bk['log']:.3f} minbk={bk['minbk']:.3f} ruined(<5%)={bk['ruined']}", flush=True)

# also dump hit rate for headline
print("\n===== HIT/BET diagnostics (test) =====", flush=True)
for (m, tau, f, le_roi) in eval_cfgs:
    hits=0; bets=0
    for r in TEST:
        sel,ev = select_ev(r, tau)
        for s in sel:
            bets+=1; hits+= int(r["won"][s])
    if bets:
        print(f"{m:7s} tau={tau}: per-leg hit={hits/bets:.4f} legs={bets}", flush=True)
        break  # selection same across alloc methods given tau; print once per tau set
# print hit per tau
seen=set()
for (m,tau,f,le_roi) in eval_cfgs:
    if tau in seen: continue
    seen.add(tau)
    hits=bets=0
    for r in TEST:
        sel,ev=select_ev(r,tau)
        for s in sel: bets+=1; hits+=int(r["won"][s])
    print(f"tau={tau}: per-leg hit={hits/max(bets,1):.4f} legs={bets}", flush=True)
