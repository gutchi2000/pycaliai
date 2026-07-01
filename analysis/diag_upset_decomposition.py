# -*- coding: utf-8 -*-
"""
diag_upset_decomposition.py
===========================
A(荒れを予測できていない) vs B(予測はできるが馬券構築で負け) を切り分ける。
v6 を OOS(2024-25) でスコア化し、レース単位で
  スコア分布(s1/s2/s6/gap_12/gap_16/entropy/score_std/p1_max)・◎着内・各券種ROI・calibration
を集計。予測問題 / 期待値問題 / 組み合わせ問題 を定量判定する。
"""
import sys, io
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
rng = np.random.default_rng(42)
BASE = Path(r"E:\PyCaLiAI")
sys.path.insert(0, str(BASE))

import backtest_pl_ev as be
import pl_probs as PL
be.MODEL_PKL = BASE / "models/unified_rank_v6.pkl"
be.CAL_PKL   = BASE / "models/pl_calibrators_v6.pkl"
be.CURVE_PKL = BASE / "data/pl_payout_curve_v6.pkl"
from backtest_pl_ev import all_fukusho_vec_fast, all_umaren_mat, COL_RID, COL_BAN

cals = joblib.load(be.CAL_PKL)["calibrators"]
payouts = be.load_payouts()
print(f"[payouts] {len(payouts):,} races", flush=True)
te = be.score_test(include_valid=True)
print(f"[scored] rows={len(te):,} races={te[COL_RID].nunique():,}", flush=True)

def f(x):
    try: return float(x)
    except: return np.nan

rows = []
cal_tan_p, cal_tan_y = [], []   # 全頭 calibration (単勝)
cal_fuk_p, cal_fuk_y = [], []   # 全頭 calibration (複勝)
for rid, g in te.groupby(COL_RID, sort=False):
    n = len(g)
    if n < 5: continue
    rid_s = str(int(rid)) if isinstance(rid, (int, np.integer)) else str(rid)
    if rid_s not in payouts: continue
    po = payouts[rid_s]
    g = g.sort_values(COL_BAN).reset_index(drop=True)
    ban = g[COL_BAN].astype(int).values
    b2i = {int(b): i for i, b in enumerate(ban)}
    wi, pi_, si = b2i.get(po["win"]), b2i.get(po["plc"]), b2i.get(po["sho"])
    if wi is None or pi_ is None or si is None: continue
    s = g["_score"].values.astype(float)
    order = np.argsort(-s)
    hon, tai, san, d1, d2 = (int(order[k]) for k in range(5))
    marks2_5 = {tai, san, d1, d2}
    w = PL.pl_weights(s); p1 = w / w.sum()
    ent = float(-(p1 * np.log(np.clip(p1, 1e-12, 1))).sum())
    ss = np.sort(s)[::-1]
    # calibrated probs
    p_tan = np.clip(cals["tansho"].predict(p1), 0, 1)
    p_fuk = np.clip(cals["fukusho"].predict(all_fukusho_vec_fast(w)), 0, 1)
    win_vec = np.zeros(n); win_vec[wi] = 1
    top3_vec = np.zeros(n); top3_vec[[wi, pi_, si]] = 1
    cal_tan_p.append(p_tan); cal_tan_y.append(win_vec)
    cal_fuk_p.append(p_fuk); cal_fuk_y.append(top3_vec)

    hon_win  = int(hon == wi)
    hon_top2 = int(hon in {wi, pi_})
    hon_top3 = int(hon in {wi, pi_, si})
    win_pair = {wi, pi_}
    # 単勝◎ / 複勝◎ payout
    tan_ret = f(po["tansho"]) if hon_win else 0.0
    fuk_ret = 0.0
    if hon == wi:   fuk_ret = f(po["fuku_win"])
    elif hon == pi_: fuk_ret = f(po["fuku_plc"])
    elif hon == si:  fuk_ret = f(po["fuku_sho"])
    fuk_ret = 0.0 if np.isnan(fuk_ret) else fuk_ret
    # 馬連 ◎-〇 (1点) / ◎流し 〇▲△△ (4点)
    umaren_pay = f(po["umaren"]); umaren_pay = 0.0 if np.isnan(umaren_pay) else umaren_pay
    hit_hono = ({hon, tai} == win_pair)
    hit_naga = (hon in win_pair) and ((win_pair - {hon}) <= marks2_5)
    rows.append(dict(
        rid=rid_s, year=int(str(rid_s)[:4]), n=n,
        s1=ss[0], s2=ss[1], s6=(ss[5] if n >= 6 else np.nan),
        gap_12=ss[0]-ss[1], gap_16=(ss[0]-ss[5] if n >= 6 else np.nan),
        score_std=float(s.std()), entropy=ent, ent_norm=ent/np.log(n), p1_max=float(p1[hon]),
        hon_win=hon_win, hon_top2=hon_top2, hon_top3=hon_top3,
        win_pay=f(po["tansho"]),
        tan_ret=tan_ret, fuk_ret=fuk_ret,
        umaren_hono_ret=(umaren_pay if hit_hono else 0.0), hit_hono=int(hit_hono),
        naga_ret=(umaren_pay if hit_naga else 0.0), hit_naga=int(hit_naga),
        baba=str(g["馬場状態"].iloc[0]) if "馬場状態" in g else "?",
        dist=f(g["距離"].iloc[0]) if "距離" in g else np.nan,
    ))

df = pd.DataFrame(rows)
oos = df[df.year.isin([2024, 2025])].copy()
print(f"\n[OOS 2024-25] races={len(oos):,}\n", flush=True)

def roi(sub, col, stake=100):
    if len(sub) == 0: return np.nan
    return round(sub[col].sum() / (stake * len(sub)) * 100, 1)
def boot_diff(a, b, nb=3000):
    a, b = a.to_numpy(float), b.to_numpy(float)
    d = (rng.choice(a, (nb, len(a))).mean(1) - rng.choice(b, (nb, len(b))).mean(1))
    return round(np.percentile(d, 2.5), 2), round(np.percentile(d, 97.5), 2), round((d > 0).mean(), 3)

# ===== Q1: 外れ(◎飛び) vs 当たり(◎複勝圏) のスコア分布 =====
print("="*70); print("Q1  スコア分布: ◎複勝圏内(当たり) vs ◎飛び(外れ) / 荒れ(勝馬10倍+)"); print("="*70)
def dist_stats(sub, label):
    return dict(label=label, n=len(sub),
        s1=round(sub.s1.mean(),3), s2=round(sub.s2.mean(),3), s6=round(sub.s6.mean(),3),
        gap_12=round(sub.gap_12.mean(),3), gap_16=round(sub.gap_16.mean(),3),
        entropy=round(sub.entropy.mean(),3), ent_norm=round(sub.ent_norm.mean(),3),
        score_std=round(sub.score_std.mean(),3), p1max=round(sub.p1_max.mean(),3))
q1 = pd.DataFrame([
    dist_stats(oos, "ALL"),
    dist_stats(oos[oos.hon_top3==1], "◎複勝圏(当)"),
    dist_stats(oos[oos.hon_top3==0], "◎飛び(外)"),
    dist_stats(oos[oos.win_pay>=1000], "荒れ:勝馬>=10倍"),
    dist_stats(oos[oos.win_pay<300],  "堅:勝馬<3倍"),
])
print(q1.to_string(index=False), flush=True)

# ===== Q2: 馬連◎流し外れレースで ◎は複勝圏に来ているか =====
print("\n"+"="*70); print("Q2  負け分解: 馬連◎流し(4点)が外れたレースの内訳"); print("="*70)
loss = oos[oos.hit_naga==0]
print(f"全OOS {len(oos):,} / 流し的中 {oos.hit_naga.sum():,} ({oos.hit_naga.mean()*100:.1f}%) / 流し外れ {len(loss):,}")
print(f"  外れのうち ◎複勝圏内: {loss.hon_top3.mean()*100:.1f}%  (◎飛び: {(1-loss.hon_top3.mean())*100:.1f}%)")
cat_hit   = oos.hit_naga.mean()
cat_combo = ((oos.hit_naga==0)&(oos.hon_top3==1)).mean()
cat_flop  = ((oos.hit_naga==0)&(oos.hon_top3==0)).mean()
print(f"  全レース3分解: 的中={cat_hit*100:.1f}%  ◎来たが組合せ失敗={cat_combo*100:.1f}%  ◎飛び={cat_flop*100:.1f}%")

# ===== Q3: ◎複勝圏レース限定の回収率 =====
print("\n"+"="*70); print("Q3  ◎複勝圏内レース限定 ROI (能力推定◯のとき馬券は勝てるか)"); print("="*70)
ok = oos[oos.hon_top3==1]
print(f" 母数 {len(ok):,}レース")
print(f"  単勝◎  ROI={roi(ok,'tan_ret')}%  (◎勝率 {ok.hon_win.mean()*100:.1f}%)")
print(f"  複勝◎  ROI={roi(ok,'fuk_ret')}%")
print(f"  馬連◎-〇ROI={roi(ok,'umaren_hono_ret')}%  (的中 {ok.hit_hono.mean()*100:.1f}%)")
print(f"  馬連◎流しROI={roi(ok,'naga_ret',stake=400)}%  (的中 {ok.hit_naga.mean()*100:.1f}%)")
print(" 参考 全OOS:")
print(f"  単勝◎ {roi(oos,'tan_ret')}% / 複勝◎ {roi(oos,'fuk_ret')}% / 馬連◎-〇 {roi(oos,'umaren_hono_ret')}% / 流し {roi(oos,'naga_ret',stake=400)}%")

# ===== Q4: calibration =====
print("\n"+"="*70); print("Q4  Calibration (予測確率 vs 実現率, 全頭, OOS)"); print("="*70)
def reliability(P, Y, nbin=10, label=""):
    P = np.concatenate(P); Y = np.concatenate(Y)
    # 年フィルタ不可(レース順なので全データ=valid込み)。OOSのみにするためteのyearで絞れないので全期間=参考。
    bins = np.linspace(0, 1, nbin+1)
    idx = np.clip(np.digitize(P, bins)-1, 0, nbin-1)
    ece = 0.0; tot = len(P); print(f" [{label}]  pred_bin -> actual (n)")
    for b in range(nbin):
        m = idx == b
        if m.sum() < 50: continue
        mp, ac = P[m].mean(), Y[m].mean()
        ece += m.sum()/tot * abs(mp-ac)
        print(f"   {bins[b]:.2f}-{bins[b+1]:.2f}: pred={mp*100:5.1f}%  actual={ac*100:5.1f}%  n={m.sum():>6,}")
    print(f"   ECE = {ece:.4f}")
print(" 注: calibrationは全期間(valid込み)。valid=2023はcalibrator学習データ(in-sample)。")
reliability(cal_tan_p, cal_tan_y, label="単勝 p_win")
reliability(cal_fuk_p, cal_fuk_y, label="複勝 p_top3")

# ===== Q5/Q6: entropy / gap_12 五分位 × ROI =====
def quintile_table(metric):
    print("\n"+"="*70); print(f"Q5/6  {metric} 五分位 × 指標 (OOS)"); print("="*70)
    sub = oos.dropna(subset=[metric]).copy()
    sub["q"] = pd.qcut(sub[metric], 5, labels=["Q1低","Q2","Q3","Q4","Q5高"], duplicates="drop")
    out = []
    for q, gg in sub.groupby("q", observed=True):
        out.append({"quint": q, "n": len(gg),
            f"{metric}_m": round(gg[metric].mean(),3),
            "hon_top3%": round(gg.hon_top3.mean()*100,1), "win_x": round(gg.win_pay.mean()/100,1),
            "tanROI": roi(gg,'tan_ret'), "fukROI": roi(gg,'fuk_ret'),
            "umarenROI": roi(gg,'umaren_hono_ret'), "nagaROI": roi(gg,'naga_ret',stake=400)})
    print(pd.DataFrame(out).to_string(index=False))
    lo = sub[sub.q=="Q1低"]; hi = sub[sub.q=="Q5高"]
    for col, st in [("tan_ret",100),("fuk_ret",100),("naga_ret",400)]:
        ci = boot_diff(hi[col]/st*100, lo[col]/st*100)
        print(f"   {col} High-Low差 CI95=[{ci[0]},{ci[1]}] P(>0)={ci[2]}")
quintile_table("entropy")
quintile_table("gap_12")

# ===== Q8: ◎飛びレースの共通特徴 =====
print("\n"+"="*70); print("Q8  ◎飛び(外) vs ◎複勝圏(当) の特徴比較 (OOS)"); print("="*70)
flop = oos[oos.hon_top3==0]; hit = oos[oos.hon_top3==1]
for col in ["n", "entropy", "ent_norm", "gap_12", "score_std", "p1_max", "win_pay", "dist"]:
    a, b = flop[col].mean(), hit[col].mean()
    print(f"  {col:10s}: ◎飛び={a:8.3f}  ◎来={b:8.3f}  差={a-b:+.3f}")
print("\n  頭数帯別 ◎複勝圏率:")
oos["nband"] = pd.cut(oos.n, [4,9,12,14,16,18], labels=["5-9","10-12","13-14","15-16","17-18"])
print(oos.groupby("nband", observed=True).agg(n=("rid","size"), top3_rate=("hon_top3", lambda x: round(x.mean()*100,1))).to_string())
print("\n  馬場別 ◎複勝圏率:")
print(oos.groupby("baba", observed=True).agg(n=("rid","size"), top3_rate=("hon_top3", lambda x: round(x.mean()*100,1))).query("n>=100").to_string())

oos.to_parquet(BASE/"reports/_diag_upset_oos.parquet")
print("\n[saved] reports/_diag_upset_oos.parquet", flush=True)
