# -*- coding: utf-8 -*-
"""
m4 — 市場最人気複勝84.5%の解剖 (fairモデル無しの純odds帯)
複勝odds中値帯 × 人気順位 × 頭数帯 で test複勝ROIを実測。
pari-mutuel place pool 自体がどの帯で >100% 払うか (FLB直接観測)。
さらに「複勝で割安」= 複勝implied < 単勝impliedから期待される置換 を購入。

leak-safe: 決定材料は9時odds(o_win9/fuku_lo9/fuku_hi9)のみ。決済=fpay。
分割: ≤2023=tune/discovery, 2024-25=test。test閾値最適化は禁止。
"""
import sys, json
sys.path.insert(0, "E:/PyCaLiAI/analysis")
import joint_lib as J
import numpy as np, pandas as pd

H, _ = J.load()

# --- 複勝決済 ---
# 1.0 stake -> top3なら fpay 返る, else 0。複勝odds中値で参考。
H = H.copy()
H["fmid"] = (H["fuku_lo9"] + H["fuku_hi9"]) / 2.0
# fpay = 100円賭けに対する払戻(例 110=1.1倍)。stake=100 として ROI[%] = mean(fpay if top3 else 0)
# 1頭の複勝1点買い payout: top3 なら fpay, else 0
H["pl_ret"] = np.where(H["top3"] == 1, H["fpay"].fillna(0.0), 0.0)
# fmid 欠損行は除外 (複勝odds無し=決定不能)
H = H[H["fmid"].notna()].copy()

# 単勝implied (de-vig済み pi が単勝marginal)。9時単勝odds から raw implied も。
H["win_imp_raw"] = 1.0 / H["o_win9"]
# 複勝implied (raw, de-vig前): 複勝中値odds の逆数
H["plc_imp_raw"] = 1.0 / H["fmid"]

# 人気順位 (9時単勝odds 昇順 = 人気順)
H["pop_rank"] = H.groupby("rid")["o_win9"].rank(method="first", ascending=True).astype(int)

# 頭数帯
def nband(n):
    if n <= 7: return "n<=7"
    if n <= 12: return "n8-12"
    if n <= 15: return "n13-15"
    return "n16-18"
H["nband"] = H["n"].map(nband)

# 複勝odds中値帯
def oband(x):
    if x < 1.2: return "1.0-1.2"
    if x < 1.5: return "1.2-1.5"
    if x < 2.0: return "1.5-2.0"
    if x < 3.0: return "2.0-3.0"
    if x < 5.0: return "3.0-5.0"
    return "5.0+"
H["oband"] = H["fmid"].map(oband)
OB_ORDER = ["1.0-1.2", "1.2-1.5", "1.5-2.0", "2.0-3.0", "3.0-5.0", "5.0+"]

# 人気帯
def popband(r):
    if r <= 3: return "fav1-3"
    if r <= 6: return "fav4-6"
    return "fav7+"
H["popband"] = H["pop_rank"].map(popband)

H_tr = H[H.year <= 2023].copy()
H_te = H[H.year >= 2024].copy()


def roi_stats(sub):
    """複勝1点買い ROI と的中率, bootstrap 95%CI 下限。"""
    n = len(sub)
    if n == 0:
        return dict(n=0, hit=np.nan, roi=np.nan, ci_lo=np.nan)
    ret = sub["pl_ret"].values
    hit = (sub["top3"] == 1).mean()
    roi = ret.mean()  # 1.0 stake -> mean payout = ROI
    # bootstrap CI (race単位でなくbet単位; 簡易)
    rng = np.random.default_rng(7)
    B = 2000
    idx = rng.integers(0, n, size=(B, min(n, 100000)))
    if n > 100000:
        sample = ret  # fallback
        boots = np.array([rng.choice(ret, size=n, replace=True).mean() for _ in range(500)])
    else:
        boots = ret[idx].mean(axis=1)
    ci_lo = np.percentile(boots, 2.5)
    return dict(n=int(n), hit=float(round(hit, 4)), roi=float(round(roi, 4)),
                ci_lo=float(round(ci_lo, 4)))


print("=" * 70)
print("M4-A: 複勝odds帯 × 人気帯 × 頭数帯 ROIテーブル (純odds, fairモデル無)")
print("=" * 70)

cells = []
n_cells = 0
for ob in OB_ORDER:
    for pb in ["fav1-3", "fav4-6", "fav7+"]:
        for nb in ["n<=7", "n8-12", "n13-15", "n16-18"]:
            tr = H_tr[(H_tr.oband == ob) & (H_tr.popband == pb) & (H_tr.nband == nb)]
            te = H_te[(H_te.oband == ob) & (H_te.popband == pb) & (H_te.nband == nb)]
            if len(tr) < 50 and len(te) < 50:
                continue
            n_cells += 1
            st_tr = roi_stats(tr)
            st_te = roi_stats(te)
            cells.append(dict(oband=ob, popband=pb, nband=nb,
                              tr_n=st_tr["n"], tr_roi=st_tr["roi"],
                              te_n=st_te["n"], te_hit=st_te["hit"],
                              te_roi=st_te["roi"], te_ci_lo=st_te["ci_lo"]))

cdf = pd.DataFrame(cells)
pd.set_option("display.width", 200)
print(cdf.to_string(index=False))
print(f"\n総セル数 (≥50): {n_cells}  -> Bonferroni alpha=0.05/{n_cells}={0.05/max(n_cells,1):.5f}")

# >100% かつ CI下限>1.0 のセル (test利益確定)
prof = cdf[(cdf.te_roi > 1.0)].sort_values("te_roi", ascending=False)
print("\n--- test ROI>100% セル ---")
print(prof.to_string(index=False) if len(prof) else "なし")
prof_ci = cdf[(cdf.te_ci_lo > 1.0)]
print("\n--- test ROI CI下限>100% (利益が頑健) セル ---")
print(prof_ci.to_string(index=False) if len(prof_ci) else "なし")

# 控除床 (80%) 超え
floor = cdf[(cdf.te_roi > 0.80)].sort_values("te_roi", ascending=False)
print(f"\n--- test ROI>80%(控除床超え=平均より良い) セル: {len(floor)}個 ---")
print(floor.head(20).to_string(index=False) if len(floor) else "なし")


print("\n" + "=" * 70)
print("M4-B: 周辺集計 (odds帯のみ / 人気帯のみ / 頭数帯のみ)")
print("=" * 70)
for dim, order in [("oband", OB_ORDER), ("popband", ["fav1-3", "fav4-6", "fav7+"]),
                   ("nband", ["n<=7", "n8-12", "n13-15", "n16-18"])]:
    print(f"\n[{dim}]")
    rows = []
    for v in order:
        te = H_te[H_te[dim] == v]
        tr = H_tr[H_tr[dim] == v]
        st = roi_stats(te); sttr = roi_stats(tr)
        rows.append(dict(val=v, tr_n=sttr["n"], tr_roi=sttr["roi"],
                         te_n=st["n"], te_hit=st["hit"], te_roi=st["roi"], te_ci_lo=st["ci_lo"]))
    print(pd.DataFrame(rows).to_string(index=False))


print("\n" + "=" * 70)
print("M4-C: 市場最人気複勝1点 control (FLB再現)")
print("=" * 70)
# 各レース 単勝最人気 (pop_rank==1) の複勝を1点買い
fav1_tr = H_tr[H_tr.pop_rank == 1]
fav1_te = H_te[H_te.pop_rank == 1]
print("単勝最人気の複勝1点:")
print("  ≤2023:", roi_stats(fav1_tr))
print("  test :", roi_stats(fav1_te))
# 複勝最人気 (複勝odds最小) の複勝1点
H_tr["plc_rank"] = H_tr.groupby("rid")["fmid"].rank(method="first").astype(int)
H_te["plc_rank"] = H_te.groupby("rid")["fmid"].rank(method="first").astype(int)
print("\n複勝最人気(複勝odds最小)の複勝1点:")
print("  ≤2023:", roi_stats(H_tr[H_tr.plc_rank == 1]))
print("  test :", roi_stats(H_te[H_te.plc_rank == 1]))


print("\n" + "=" * 70)
print("M4-D: 複勝『割安』裁定 — 単勝implied vs 複勝implied の楔")
print("=" * 70)
# 複勝公正P(top3) を 単勝marginal pi から PL で作る (各レース)
# wedge = plc_imp_market - fair_plc_prob。負(market低<fair) = 市場で割安 = 買い。
# leak-safe: pi(9時単勝de-vig) と 複勝odds中値 のみ使用。
def add_fair_plc(df):
    out = np.full(len(df), np.nan)
    for rid, g in df.groupby("rid", sort=False):
        n = int(g.n.iloc[0]); k = 3 if n >= 8 else 2
        pi = g.pi.values.astype(float)
        probs = J.pl_topk_probs(pi, k=k, K=4000)
        out[g.index.map(lambda x: df.index.get_loc(x)).values if False else
            [df.index.get_loc(i) for i in g.index]] = probs
    return out

# fair_plc は full field (全頭, fmid欠損も含む) で計算 -> rid+ban で merge
# (filteringするとpi marginalが崩れPLが壊れるため)
def build_fair_plc(H_full):
    rows = []
    for rid, g in H_full.groupby("rid", sort=False):
        n = int(g.n.iloc[0]); k = 3 if n >= 8 else 2
        pi = g.pi.values.astype(float)
        if len(pi) < 2:
            continue
        probs = J.pl_topk_probs(pi, k=k, K=4000)
        for b, p in zip(g.ban.values, probs):
            rows.append((rid, int(b), float(p)))
    return pd.DataFrame(rows, columns=["rid", "ban", "fair_plc"])

print("fair_plc を計算中 (PL top-k, K=4000, full field)...")
Hf, _ = J.load()
fp = build_fair_plc(Hf)
H_tr2 = H_tr.merge(fp, on=["rid", "ban"], how="left")
H_te2 = H_te.merge(fp, on=["rid", "ban"], how="left")

# 複勝市場 de-vig implied: 各馬の複勝implied を pool整合に正規化 (合計 places にスケール)
def add_plc_market_imp(df):
    df = df.copy()
    imp = 1.0 / df["fmid"].values
    # 複勝は places=k本当たる -> 正規化先は k
    res = np.full(len(df), np.nan)
    for rid, g in df.groupby("rid", sort=False):
        n = int(g.n.iloc[0]); k = 3 if n >= 8 else 2
        v = 1.0 / g["fmid"].values
        v = v / v.sum() * k  # de-vig して合計=places
        res[g.index.values] = v
    df["plc_imp_devig"] = res
    return df

H_tr2 = add_plc_market_imp(H_tr2)
H_te2 = add_plc_market_imp(H_te2)

# wedge = fair_plc - plc_imp_devig。正 = 市場が過小評価 = fair的には来る確率高いのに複勝oddsが甘い = 買い妙味
H_tr2["wedge"] = H_tr2["fair_plc"] - H_tr2["plc_imp_devig"]
H_te2["wedge"] = H_te2["fair_plc"] - H_te2["plc_imp_devig"]

# ≤2023 で wedge 分位を切る -> test に適用 (test閾値最適化禁止のため ≤2023 で固定)
qs = H_tr2["wedge"].quantile([0.0, 0.5, 0.7, 0.8, 0.9, 0.95]).to_dict()
print("≤2023 wedge quantiles:", {round(k,2): round(v,4) for k,v in qs.items()})

print("\nwedge 分位 × test ROI (≤2023で閾値固定):")
rows = []
for lo, name in [(qs[0.5], "top50%"), (qs[0.7], "top30%"), (qs[0.8], "top20%"),
                 (qs[0.9], "top10%"), (qs[0.95], "top5%")]:
    tr = H_tr2[H_tr2.wedge >= lo]
    te = H_te2[H_te2.wedge >= lo]
    rows.append(dict(thresh=name, lo=round(lo,4),
                     tr_n=len(tr), tr_roi=roi_stats(tr)["roi"],
                     te_n=len(te), te_hit=roi_stats(te)["hit"],
                     te_roi=roi_stats(te)["roi"], te_ci_lo=roi_stats(te)["ci_lo"]))
print(pd.DataFrame(rows).to_string(index=False))

# wedge × odds帯 のクロス (妙味が odds帯のどこに住むか)
print("\nwedge>0(割安) × 複勝odds帯 × test ROI:")
rows = []
n_cells_d = 0
for ob in OB_ORDER:
    te = H_te2[(H_te2.wedge > 0) & (H_te2.oband == ob)]
    tr = H_tr2[(H_tr2.wedge > 0) & (H_tr2.oband == ob)]
    if len(te) < 30: continue
    n_cells_d += 1
    rows.append(dict(oband=ob, tr_n=len(tr), tr_roi=roi_stats(tr)["roi"],
                     te_n=len(te), te_hit=roi_stats(te)["hit"],
                     te_roi=roi_stats(te)["roi"], te_ci_lo=roi_stats(te)["ci_lo"]))
print(pd.DataFrame(rows).to_string(index=False))

# 逆: 単勝で本命(pop_rank<=3)を複勝で買う vs 複勝割安を買う 比較
print("\n--- 比較: 単勝本命複勝 vs 複勝割安(wedge>0) (test) ---")
print("単勝本命1-3 複勝:", roi_stats(H_te2[H_te2.pop_rank <= 3]))
print("複勝割安 wedge>0:", roi_stats(H_te2[H_te2.wedge > 0]))
print("複勝割高 wedge<0:", roi_stats(H_te2[H_te2.wedge < 0]))

# best test cell サマリ
best_cell = cdf.loc[cdf.te_roi.idxmax()] if len(cdf) else None

# 保存
summary = dict(
    n_cells_A=int(n_cells),
    n_cells_D=int(n_cells_d),
    best_cell=best_cell.to_dict() if best_cell is not None else None,
    fav1_te_roi=roi_stats(fav1_te)["roi"],
    plc_fav1_te_roi=roi_stats(H_te[H_te.plc_rank == 1])["roi"],
    wedge_pos_te_roi=roi_stats(H_te2[H_te2.wedge > 0])["roi"],
    pop13_te_roi=roi_stats(H_te2[H_te2.pop_rank <= 3])["roi"],
)
with open("E:/PyCaLiAI/analysis/_joint_m4_result.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
print("\nsaved _joint_m4_result.json")
print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
