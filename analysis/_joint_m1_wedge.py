# -*- coding: utf-8 -*-
"""
_joint_m1_wedge.py  (task m1)
楔(fair=PL単勝marginal由来の複勝公正P vs 市場複勝implied)の所在マップ。
odds帯(複勝中値) × 頭数帯 × 楔符号 で層別し、各セルの test複勝ROI を出す。
≤2023でセル選定 → test=2024-2025 で評価。leak-safe: 決定=9時odds、決済=fpay。

JRAルール: n>=8 で places=3, n<=7 で places=2。複勝控除20%(床80%)。
fair = J.pl_topk_probs(pi, places)  (各馬 top-places 包含率, Σ≈places)
市場implied = (1/複勝中値) を Σ=places に正規化 (de-vig & places-scale)
楔 = fair - market_implied  (符号>0 = fair が市場より「来る」と見る = Dr.Z候補)

EV(fair, 複勝中値) = fair * 複勝中値(=配当倍率近似)。実払戻=fpay/100倍率。
複勝odds中値は払戻幅の中央 → 実決済は確定 fpay/100(>=1.0) を使う(複勝圏馬のみ)。
"""
import sys, math
sys.path.insert(0, "E:/PyCaLiAI/analysis")
import numpy as np, pandas as pd
import joint_lib as J

RNG = np.random.default_rng(7)

H, _ = J.load()

# ---- per-race で fair / market implied / wedge を計算し per-horse 表に展開 ----
rows = []
for rid, g in H.groupby("rid", sort=False):
    n = int(g.n.iloc[0])
    if n < 5:
        continue
    places = 3 if n >= 8 else 2
    pi = g.pi.values.astype(float)
    bans = g.ban.values.astype(int)
    lo = g.fuku_lo9.values.astype(float)
    hi = g.fuku_hi9.values.astype(float)
    mid = (lo + hi) / 2.0          # 複勝odds中値 (倍率)
    # 無効odds(NaN/<=0)があるレースはスキップ(市場implied作れない)
    if np.any(~np.isfinite(mid)) or np.any(mid <= 0):
        continue
    # fair: PL top-places 包含率
    fair = J.pl_topk_probs(pi, places, K=4000, rng=RNG)
    # market implied: 1/mid を Σ=places に正規化
    inv = 1.0 / mid
    mk = inv / inv.sum() * places
    wedge = fair - mk
    yr = int(g.year.iloc[0])
    top3 = g.top3.values.astype(int)        # 複勝圏(1/2/3着 もしくは2着まで)
    fpay = g.fpay.values.astype(float)      # 確定複勝配当/100, 複勝圏のみ, else NaN
    for k in range(n):
        rows.append((rid, yr, n, int(bans[k]), float(pi[k]),
                     float(mid[k]), float(fair[k]), float(mk[k]),
                     float(wedge[k]), int(top3[k]), float(fpay[k])))

D = pd.DataFrame(rows, columns=["rid","year","n","ban","pi","mid","fair","mk",
                                "wedge","top3","fpay"])
print("horses(valid races):", len(D), " races:", D.rid.nunique())

# 決済 payoff: 複勝圏なら fpay(/100倍率) else 0。fpay は複勝圏のみ入っているはず。
# 1点100円賭けの net = payoff_mult - 1。ROI = Σpayoff / Σstake。
D["payoff"] = np.where(D.top3 == 1, np.where(np.isfinite(D.fpay), D.fpay, 0.0), 0.0)

# ---- 層別ビン ----
def odds_band(m):
    if m < 1.5: return "<1.5"
    if m < 2.0: return "1.5-2"
    if m < 3.0: return "2-3"
    if m < 5.0: return "3-5"
    if m < 10.0: return "5-10"
    return "10+"
def n_band(n):
    if n < 10: return "<10"
    if n <= 13: return "10-13"
    return "14+"
def wsign(w):
    return "pos" if w > 0 else "neg"

D["ob"] = D.mid.apply(odds_band)
D["nb"] = D.n.apply(n_band)
D["ws"] = D.wedge.apply(wsign)

TR = D[D.year <= 2023].copy()
TE = D[D.year >= 2024].copy()
print("train horses:", len(TR), " test horses:", len(TE))

OB = ["<1.5","1.5-2","2-3","3-5","5-10","10+"]
NB = ["<10","10-13","14+"]
WS = ["pos","neg"]

def cell_stats(df):
    n = len(df)
    if n == 0: return (0, np.nan, np.nan)
    roi = df.payoff.sum() / n
    hit = df.top3.mean()
    return (n, roi, hit)

def boot_ci(payoff, B=2000):
    if len(payoff) == 0: return (np.nan, np.nan)
    arr = payoff.values
    idx = RNG.integers(0, len(arr), size=(B, len(arr)))
    means = arr[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))

# ---- 全セル列挙(Bonferroni 母数) ----
cells = []
for ob in OB:
    for nb in NB:
        for ws in WS:
            cells.append((ob, nb, ws))
N_CELLS = len(cells)
print("N_CELLS(Bonferroni母数):", N_CELLS)

# ≤2023 でセル選定: train ROI でランク。test で評価。
results = []
for (ob, nb, ws) in cells:
    trc = TR[(TR.ob==ob)&(TR.nb==nb)&(TR.ws==ws)]
    tec = TE[(TE.ob==ob)&(TE.nb==nb)&(TE.ws==ws)]
    tr_n, tr_roi, tr_hit = cell_stats(trc)
    te_n, te_roi, te_hit = cell_stats(tec)
    lo, hi = boot_ci(tec.payoff) if te_n > 0 else (np.nan, np.nan)
    results.append(dict(ob=ob, nb=nb, ws=ws, tr_n=tr_n, tr_roi=tr_roi,
                        te_n=te_n, te_roi=te_roi, te_hit=te_hit,
                        ci_lo=lo, ci_hi=hi))

R = pd.DataFrame(results)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

print("\n=== 全セル (train ROIで選定, test評価) ===")
Rd = R.copy()
for c in ["tr_roi","te_roi","te_hit","ci_lo","ci_hi"]:
    Rd[c] = Rd[c].round(4)
print(Rd.sort_values("tr_roi", ascending=False).to_string(index=False))

# ≤2023で黒字(train ROI>1)のセルだけを「選定」とみなし test を見る
sel = R[(R.tr_roi > 1.0) & (R.tr_n >= 50)].copy()
print("\n=== ≤2023で黒字(tr_roi>1, tr_n>=50)に選定されたセル → test成績 ===")
if len(sel):
    seld = sel.copy()
    for c in ["tr_roi","te_roi","te_hit","ci_lo","ci_hi"]:
        seld[c] = seld[c].round(4)
    print(seld.sort_values("te_roi", ascending=False).to_string(index=False))
    n_profit_te = ((sel.te_roi > 1.0)).sum()
    n_profit_te_ci = ((sel.ci_lo > 1.0)).sum()
    print(f"\n選定セル中 test ROI>1.00 : {n_profit_te}/{len(sel)}")
    print(f"選定セル中 test CI下限>1.00 : {n_profit_te_ci}/{len(sel)}  (Bonferroni: N_CELLS={N_CELLS})")
else:
    print("≤2023で黒字セル無し")

# 全test黒字セル(参考: train条件無視)
print("\n=== 参考: test ROI>1 の全セル(train無視, 過学習注意) ===")
te_prof = R[(R.te_roi > 1.0) & (R.te_n >= 30)].sort_values("te_roi", ascending=False)
if len(te_prof):
    t = te_prof.copy()
    for c in ["tr_roi","te_roi","te_hit","ci_lo","ci_hi"]:
        t[c] = t[c].round(4)
    print(t.to_string(index=False))
else:
    print("無し")

# 楔符号が利益化に効くか: odds帯ごとに pos vs neg の test ROI 差
print("\n=== 楔符号 効果: odds帯 × (pos - neg) test ROI ===")
for ob in OB:
    p = TE[(TE.ob==ob)&(TE.ws=="pos")]
    ng = TE[(TE.ob==ob)&(TE.ws=="neg")]
    pr = p.payoff.mean() if len(p) else np.nan
    nr = ng.payoff.mean() if len(ng) else np.nan
    diff = (pr - nr) if (len(p) and len(ng)) else np.nan
    print(f"  {ob:>6}: pos ROI={pr:.4f}(n={len(p)})  neg ROI={nr:.4f}(n={len(ng)})  diff={diff:+.4f}")

# overall control: 市場最人気複勝1点(各レース mid 最小)を test で
print("\n=== control: 各レース 複勝最人気1点(mid最小) test ROI ===")
fav = TE.loc[TE.groupby("rid").mid.idxmin()]
print(f"  n={len(fav)}  ROI={fav.payoff.mean():.4f}  hit={fav.top3.mean():.4f}")

# control: 全複勝ベタ買い
print(f"\n=== control: test 全頭複勝ベタ買い ROI={TE.payoff.mean():.4f} (床0.80) ===")
print(f"=== control: train 全頭複勝ベタ買い ROI={TR.payoff.mean():.4f} ===")
