# -*- coding: utf-8 -*-
"""
bt_user_rule.py — ユーザー定義ルールの OOS バックテスト (test 2024-25)
ルール: 指数(model rank)高い順に
  馬連3点: 軸=1位, 相手=2~4位  (1位-2位,1位-3位,1位-4位) ×1000円
  ワイドフォメ5点: 軸{1位,2位}×相手{1~4位}  (12,13,14,23,24) ×1400円
  計1万/レース。決済=確定。指数は連続値で同率ほぼ無し。
"""
import sys, io, json
import numpy as np, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
sys.path.insert(0, "E:/PyCaLiAI")
from backtest_pl_ev import load_payouts

H = pd.read_parquet("data/_wide/oos_horses.parquet")
wpay = json.load(open("data/_wide/oos_wide_pay.json"))
pays = load_payouts()

# umaren winpair + payout per rid
umap = {}
for rid, po in pays.items():
    wp = (min(po["win"], po["plc"]), max(po["win"], po["plc"]))
    umap[rid] = (wp, po["umaren"])

UMA_STAKE, WIDE_STAKE = 1000.0, 1400.0
rng = np.random.default_rng(0)

def run(df):
    rows = []
    for rid, g in df.groupby("rid", sort=False):
        if g.ban.nunique() < 4:
            continue
        g = g.dropna(subset=["fin"])
        if len(g) < 4 or g.fin.min() != 1:
            continue
        # 指数順位(mark_rank) → 馬番. 同率は p_win 高→人気寄りで安定 (実質同率なし)
        gg = g.sort_values(["mark_rank", "p_win"], ascending=[True, False])
        rk = gg.ban.astype(int).tolist()
        r1, r2, r3, r4 = rk[0], rk[1], rk[2], rk[3]
        top = g.sort_values("fin")
        t1, t2 = int(top.ban.iloc[0]), int(top.ban.iloc[1])
        top3 = set(int(b) for b in top.ban.iloc[:3])
        winpair = (min(t1, t2), max(t1, t2))

        # 馬連3点
        uma_pairs = [(min(r1, x), max(r1, x)) for x in (r2, r3, r4)]
        uma_stake = UMA_STAKE * 3
        uma_ret = 0.0; uma_hit = 0
        if rid in umap and umap[rid][1] == umap[rid][1]:
            wpu, payu = umap[rid]
            if wpu in uma_pairs:
                uma_ret = UMA_STAKE / 100.0 * payu; uma_hit = 1
        # ワイド5点
        wide_pairs = [(min(a, b), max(a, b)) for a, b in
                      [(r1, r2), (r1, r3), (r1, r4), (r2, r3), (r2, r4)]]
        wide_stake = WIDE_STAKE * 5
        wide_ret = 0.0; wide_hit = 0
        wp = wpay.get(rid, {})
        for (a, b) in wide_pairs:
            if a in top3 and b in top3:
                pv = wp.get(f"{a}-{b}")
                if pv:
                    wide_ret += WIDE_STAKE / 100.0 * pv; wide_hit += 1
        rows.append((int(g.year.iloc[0]), uma_stake, uma_ret, uma_hit,
                     wide_stake, wide_ret, wide_hit, int(g.n.iloc[0])))
    return pd.DataFrame(rows, columns=["year", "us", "ur", "uh", "ws", "wr", "wh", "n"])

R = run(H)
print(f"対象レース: {len(R):,} (2024={len(R[R.year==2024]):,} 2025={len(R[R.year==2025]):,})\n")

def report(d, lab):
    us, ur = d.us.sum(), d.ur.sum(); ws, wr = d.ws.sum(), d.wr.sum()
    ts, tr = us + ws, ur + wr
    print(f"[{lab}] n={len(d):,}")
    print(f"  馬連3点 : ROI={ur/us*100:5.1f}% 的中{d.uh.mean()*100:4.1f}% 投{us:,.0f} 戻{ur:,.0f}")
    print(f"  ワイド5点: ROI={wr/ws*100:5.1f}% 的中(≥1){ (d.wh>0).mean()*100:4.1f}% 投{ws:,.0f} 戻{wr:,.0f}")
    print(f"  ★合計1万: ROI={tr/ts*100:5.1f}% 投{ts:,.0f} 戻{tr:,.0f} 純{tr-ts:+,.0f}")
    # race単位純損益で bootstrap CI
    net = (d.ur + d.wr) - (d.us + d.ws)
    bs = [net.sample(len(net), replace=True, random_state=i).mean() for i in range(2000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    roi_lo = (tr/ts) + lo/(ts/len(d)); roi_hi = (tr/ts) + hi/(ts/len(d))
    print(f"  合計ROI 95%CI ≈ [{roi_lo*100:.1f}%, {roi_hi*100:.1f}%]  (1レース平均純 {net.mean():+,.0f}円)")

report(R, "test 2024-25 全レース")
print()
report(R[R.year == 2024], "2024")
print()
report(R[R.year == 2025], "2025")
print("\n--- 頭数帯別 合計ROI ---")
R["nb"] = pd.cut(R.n, [0, 9, 13, 99], labels=["≤9", "10-13", "14+"])
for nb, d in R.groupby("nb", observed=True):
    ts = d.us.sum()+d.ws.sum(); tr = d.ur.sum()+d.wr.sum()
    print(f"  {nb:>6}: n={len(d):>5} 合計ROI={tr/ts*100:5.1f}%  馬連{d.ur.sum()/d.us.sum()*100:.0f}% ワイド{d.wr.sum()/d.ws.sum()*100:.0f}%")
print("\n控除床: 馬連77.5% / ワイド80% / 利益線100%")
