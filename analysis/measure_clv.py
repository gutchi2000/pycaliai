# -*- coding: utf-8 -*-
"""
measure_clv.py — CLV(Closing Line Value)計測。床超えの唯一の未踏路(roi_max_doctrine Tier2)。
9時オッズ(o9, 基盤) vs 確定(close)オッズで、AIの印が「締切までに買われる側=正CLV」か、
そして【正CLV部分集合が控除床/100%を超えるか】を測る。CLV>0 が長期収益の単一最良予測子。
entry=9時で賭けたとして close より良い価格を取れたか: clv = o9/close - 1 (>0=締切までに短縮=正)。
"""
import sys, io, glob, re
import pandas as pd, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
def rid16(x): return re.sub(r"\D", "", str(x))[:16]

H = pd.read_parquet(r"E:\PyCaLiAI\data\_wide\oos_horses.parquet")
# 確定(close)単勝オッズ: data/odds/odds_2024*,2025*
od = []
for f in glob.glob(r"E:\PyCaLiAI\data\odds\odds_2024*.csv") + glob.glob(r"E:\PyCaLiAI\data\odds\odds_2025*.csv"):
    od.append(pd.read_csv(f, encoding="cp932", dtype={"レースID(新)": str}))
O = pd.concat(od, ignore_index=True)
O["rid"] = O["レースID(新)"].map(rid16); O["ban"] = pd.to_numeric(O["馬番"], errors="coerce")
O = O.rename(columns={"単勝オッズ": "close"})[["rid", "ban", "close"]].dropna()
O["close"] = pd.to_numeric(O["close"], errors="coerce")
H = H.merge(O, on=["rid", "ban"], how="left")
cov = H["close"].notna().mean()
print(f"close結合カバレッジ: {cov:.3f} ({H['close'].notna().sum()}/{len(H)})")
H = H.dropna(subset=["close", "o9"]); H = H[(H.close > 1.0) & (H.o9 > 1.0)]
H["clv"] = H["o9"] / H["close"] - 1.0          # >0 = 9時より締切で短縮(買われた)=正CLV
H["pos_clv"] = H["clv"] > 0.0
te = H[H.year == 2025].copy()
print(f"2025 close有り: {len(te)} 頭")

def winroi(d):  # entry(9時)で単勝flat。payoutは確定(close)で近似(勝ち馬のclose×100)
    if len(d) == 0: return (0, 0, 0)
    hit = (d.fin == 1)
    ret = (hit * d.close * 100).sum()   # close=確定単勝オッズ→100円が close*100 戻る
    return (len(d), round(hit.mean() * 100, 1), round(ret / (len(d) * 100) * 100, 1))

print("\n=== (1) CLV by 印ランク (2025) ＝AIの印は締切までに買われる側か ===")
for mr, lab in [(1, "◎"), (2, "〇"), (3, "▲"), (4, "△")]:
    d = te[te.mark_rank == mr]
    print(f"  {lab}(rank{mr}): n{len(d):>5} 平均CLV {d.clv.mean()*100:+5.1f}% 正CLV率 {d.pos_clv.mean()*100:4.1f}% 複勝圏{d.top3.mean()*100:4.1f}%")
print(f"  全頭        : n{len(te):>5} 平均CLV {te.clv.mean()*100:+5.1f}% 正CLV率 {te.pos_clv.mean()*100:4.1f}%")

print("\n=== (2) ◎を CLV符号で分割 → 単勝ROI(entry9時/payout確定) 床80%/100%超えるか ===")
hon = te[te.mark_rank == 1]
for lab, d in [("◎ 全体", hon), ("◎ 正CLV(締切で短縮)", hon[hon.pos_clv]), ("◎ 負CLV(締切で延びた)", hon[~hon.pos_clv])]:
    n, h, r = winroi(d)
    print(f"  {lab:<22}: n{n:>5} 的中{h:>5}% 単勝ROI {r:>6}%")

print("\n=== (3) 印上位3頭(◎〇▲)を CLV符号で分割 → 単勝ROI ===")
top3m = te[te.mark_rank <= 3]
for lab, d in [("◎〇▲ 全体", top3m), ("◎〇▲ 正CLV", top3m[top3m.pos_clv]), ("◎〇▲ 負CLV", top3m[~top3m.pos_clv])]:
    n, h, r = winroi(d)
    print(f"  {lab:<18}: n{n:>5} 的中{h:>5}% 単勝ROI {r:>6}%")

print("\n=== (4) CLV帯別 単勝ROI (◎〇▲, 2025) ===")
top3m = top3m.copy(); top3m["clv_band"] = pd.cut(top3m.clv, [-1, -0.1, 0, 0.1, 0.3, 99],
                                                  labels=["<-10%", "-10..0%", "0..10%", "10..30%", "30%+"])
for b, d in top3m.groupby("clv_band", observed=True):
    n, h, r = winroi(d)
    print(f"  CLV {str(b):>9}: n{n:>5} 的中{h:>5}% 単勝ROI {r:>6}%")
print("\n[読み] ◎の平均CLVが+なら『市場と同方向(informed)』。正CLV◎の単勝ROIが100%超なら"
      "=CLVが床超えの実エッジ(Tier2昇格根拠)。床近傍止まりなら『観測KPI』のまま。")
