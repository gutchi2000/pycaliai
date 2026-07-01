# -*- coding: utf-8 -*-
"""
m3 — π較正が楔を研ぐか殺すか。
単勝marginal π の FLB を ≤2023 で isotonic / ロジスティック較正 → レース内再正規化 π_cal。
fair P(top3) を π_cal で再計算し、crux(fair vs 市場複勝 logloss/Brier) と EVゲートROI を
test(2024-25) で 生π版と比較。「楔=生πのFLB漏れ」仮説を正直に判定。

leak-safe: 決定材料=9時odds(pi/fuku_lo9/fuku_hi9)。較正fitは≤2023のみ。
決済=fpay(確定複勝配当)。test閾値最適化は禁止(τは≤2023で選定)。
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "E:/PyCaLiAI/analysis")
import joint_lib as J
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

RNG = np.random.default_rng(2026)
np.random.seed(2026)


def places(n):
    return 3 if n >= 8 else 2


def race_iter(H, yr_filter):
    for rid, g in H.groupby("rid", sort=False):
        y = int(g.year.iloc[0])
        if not yr_filter(y):
            continue
        n = int(g.n.iloc[0])
        yield rid, y, n, g


# ---------- 較正器を ≤2023 で fit ----------
def fit_calibrators(H):
    tr = H[H.year <= 2023]
    pi = tr.pi.values.astype(float)
    yw = tr.is_win.values.astype(int)
    # isotonic on logit-ish; iso直接 pi->win
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
    iso.fit(pi, yw)
    # logistic on log(pi/(1-pi))
    x = np.log(np.clip(pi, 1e-9, 1 - 1e-9) / (1 - np.clip(pi, 1e-9, 1 - 1e-9))).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    lr.fit(x, yw)
    return iso, lr


def apply_cal(pi, cal, kind):
    pi = np.asarray(pi, float)
    if kind == "iso":
        return cal.predict(pi)
    else:
        x = np.log(np.clip(pi, 1e-9, 1 - 1e-9) / (1 - np.clip(pi, 1e-9, 1 - 1e-9))).reshape(-1, 1)
        return cal.predict_proba(x)[:, 1]


def renorm(p):
    p = np.clip(np.asarray(p, float), 1e-9, None)
    return p / p.sum()


# ---------- per-race fair P(top3) を各π版で算出 + 市場複勝implied ----------
def build_eval(H, yr_filter, iso, lr, K=4000):
    """各馬に fair_raw / fair_iso / fair_lr (P(top3)) と 市場複勝implied(devig) を付与。"""
    rows = []
    for rid, y, n, g in race_iter(H, yr_filter):
        k = places(n)
        pi = g.pi.values.astype(float)
        bans = g.ban.values.astype(int)
        # 市場複勝 implied: 中値odds -> 1/odds を「圏内」確率の生proxy。
        # 複勝プールは Σ P(top3)=k なので de-vig は Σ=k に正規化。
        fmid = (g.fuku_lo9.values + g.fuku_hi9.values) / 2.0
        q = 1.0 / np.clip(fmid, 1e-6, None)
        mkt = q / q.sum() * k  # Σ=k に正規化(複勝はk頭が圏内)
        mkt = np.clip(mkt, 1e-6, 1 - 1e-6)
        # fair raw
        fr = J.pl_topk_probs(pi, k, K=K, rng=RNG)
        # fair iso (renorm pi then PL)
        pic = renorm(apply_cal(pi, iso, "iso"))
        fi = J.pl_topk_probs(pic, k, K=K, rng=RNG)
        # fair lr
        picl = renorm(apply_cal(pi, lr, "lr"))
        fl = J.pl_topk_probs(picl, k, K=K, rng=RNG)
        top3 = g.top3.values.astype(int)
        fpay = g.fpay.values.astype(float)
        fmid_ = fmid
        for idx in range(n):
            rows.append(dict(rid=rid, year=y, n=n, k=k, ban=int(bans[idx]),
                             pi=float(pi[idx]),
                             fair_raw=float(np.clip(fr[idx], 1e-6, 1 - 1e-6)),
                             fair_iso=float(np.clip(fi[idx], 1e-6, 1 - 1e-6)),
                             fair_lr=float(np.clip(fl[idx], 1e-6, 1 - 1e-6)),
                             mkt=float(mkt[idx]),
                             fuku_mid=float(fmid_[idx]),
                             top3=int(top3[idx]),
                             fpay=float(fpay[idx]) if not np.isnan(fpay[idx]) else np.nan))
    return pd.DataFrame(rows)


def logloss(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def brier(p, y):
    return np.mean((p - y) ** 2)


def boot_roi_ci(stake, ret, n_boot=2000):
    """ベット単位 bootstrap で ROI 95%CI。"""
    stake = np.asarray(stake, float); ret = np.asarray(ret, float)
    m = len(stake)
    if m == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(7)
    rois = []
    for _ in range(n_boot):
        idx = rng.integers(0, m, m)
        s = stake[idx].sum()
        rois.append(ret[idx].sum() / s if s > 0 else np.nan)
    return (np.nanpercentile(rois, 2.5), np.nanpercentile(rois, 97.5))


def ev_gate(df, prob_col, tau):
    """fair_prob * fuku_mid_odds >= tau のベットを複勝で1点買い。決済=fpay/100*100=fpay(円/100単位)。
    fpay は確定複勝配当/100。100円賭けの払戻 = fpay (when top3) else 0。stake=100。"""
    sub = df[(df[prob_col] * df.fuku_mid) >= tau].copy()
    if len(sub) == 0:
        return None
    stake = np.full(len(sub), 100.0)
    ret = np.where(sub.top3.values == 1, np.nan_to_num(sub.fpay.values, nan=0.0), 0.0)
    roi = ret.sum() / stake.sum()
    hit = sub.top3.mean()
    lo, hi = boot_roi_ci(stake, ret)
    return dict(n=len(sub), roi=roi, hit=hit, ci_lo=lo, ci_hi=hi)


def main():
    H, _ = J.load()
    print("=== fitting calibrators on <=2023 ===")
    iso, lr = fit_calibrators(H)
    # sanity: calibration shift
    grid = np.array([0.005, 0.02, 0.05, 0.1, 0.2, 0.3])
    print("pi      iso     lr")
    for v in grid:
        print(f"{v:.3f}  {apply_cal([v],iso,'iso')[0]:.4f}  {apply_cal([v],lr,'lr')[0]:.4f}")

    print("\n=== building eval (train<=2023) ===")
    tr = build_eval(H, lambda y: y <= 2023, iso, lr)
    print("=== building eval (test 2024-25) ===")
    te = build_eval(H, lambda y: y >= 2024, iso, lr)
    tr.to_parquet("E:/PyCaLiAI/analysis/_tmp_m3_tr.parquet")
    te.to_parquet("E:/PyCaLiAI/analysis/_tmp_m3_te.parquet")

    # ---------- CRUX: realized top3 logloss/Brier ----------
    print("\n=== CRUX: realized top3 logloss / Brier (test 2024-25) ===")
    y = te.top3.values
    for name, col in [("fair_raw(生π)", "fair_raw"), ("fair_iso", "fair_iso"),
                      ("fair_lr", "fair_lr"), ("market_place", "mkt")]:
        ll = logloss(te[col].values, y)
        br = brier(te[col].values, y)
        print(f"  {name:18s} LL={ll:.4f}  Brier={br:.4f}")

    print("\n  -- ≤2023 (sanity) --")
    y0 = tr.top3.values
    for name, col in [("fair_raw", "fair_raw"), ("fair_iso", "fair_iso"),
                      ("fair_lr", "fair_lr"), ("market_place", "mkt")]:
        print(f"  {name:12s} LL={logloss(tr[col].values,y0):.4f}  Brier={brier(tr[col].values,y0):.4f}")

    # ---------- wedge sign: fair - market ----------
    print("\n=== wedge (fair_raw - market) realized top3 by wedge sign, test ===")
    for col, lab in [("fair_raw", "raw"), ("fair_iso", "iso"), ("fair_lr", "lr")]:
        w = te[col].values - te.mkt.values
        pos = te.top3.values[w > 0].mean(); neg = te.top3.values[w <= 0].mean()
        # fair says higher than market -> should hit more
        print(f"  {lab}: wedge>0 hit={pos:.4f} (n={int((w>0).sum())})  wedge<=0 hit={neg:.4f} (n={int((w<=0).sum())})")

    # ---------- EV gate: tune tau on <=2023, eval on test ----------
    print("\n=== EV gate (complete tau-sweep on <=2023), then frozen-tau test ===")
    taus = [1.00, 1.02, 1.05, 1.08, 1.10, 1.15, 1.20]
    for col, lab in [("fair_raw", "生π"), ("fair_iso", "iso較正"), ("fair_lr", "lr較正")]:
        print(f"\n  [{lab}] tune tau on <=2023:")
        best = None
        for tau in taus:
            r = ev_gate(tr, col, tau)
            if r:
                print(f"    tau={tau:.2f}  n={r['n']:6d} roi={r['roi']*100:6.1f}% hit={r['hit']*100:5.1f}%")
                if best is None or r["roi"] > best[1]["roi"]:
                    best = (tau, r)
        if best:
            tau = best[0]
            rt = ev_gate(te, col, tau)
            print(f"    --> frozen tau={tau:.2f} | TEST: n={rt['n']} roi={rt['roi']*100:.1f}% "
                  f"hit={rt['hit']*100:.1f}% CI[{rt['ci_lo']*100:.1f},{rt['ci_hi']*100:.1f}]")

    # ---------- control: 市場最人気複勝1点/race ----------
    print("\n=== control: 市場最人気複勝(最小fuku_mid)1点/race, test ===")
    fav = te.loc[te.groupby("rid").fuku_mid.idxmin()]
    stake = np.full(len(fav), 100.0)
    ret = np.where(fav.top3.values == 1, np.nan_to_num(fav.fpay.values, nan=0.0), 0.0)
    lo, hi = boot_roi_ci(stake, ret)
    print(f"  n={len(fav)} roi={ret.sum()/stake.sum()*100:.1f}% hit={fav.top3.mean()*100:.1f}% CI[{lo*100:.1f},{hi*100:.1f}]")


if __name__ == "__main__":
    main()
