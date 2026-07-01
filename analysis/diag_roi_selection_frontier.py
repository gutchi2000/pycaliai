# -*- coding: utf-8 -*-
"""
diag_roi_selection_frontier.py
==============================
大会採点=累積ROI / 機械的見送りOK 前提で、
「信頼度信号で上位X%だけ買ったら 累積ROI はいくつか」= ROI vs 買い数フロンティア。
最適 (券種×選別信号×買い数) を 2024 で選び 2025(OOS) で検証。
入力: reports/_diag_upset_oos.parquet
"""
import sys, io
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
BASE = Path(r"E:\PyCaLiAI")
df = pd.read_parquet(BASE / "reports/_diag_upset_oos.parquet")
tr = df[df.year == 2024].copy()
te = df[df.year == 2025].copy()

BETS = [("単勝◎","tan_ret",100), ("複勝◎","fuk_ret",100),
        ("馬連◎-〇","umaren_hono_ret",100), ("馬連◎流し","naga_ret",400)]
# 信号: (名前, 列, ascending) ascending=True は「小さいほど買う」(entropy)
SIGNALS = [("p1max↓","p1_max",False), ("entropy↑","entropy",True),
           ("gap12↓","gap_12",False), ("std↓","score_std",False)]
FRACS = [0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00]

def roi_at(sub, ret_col, stake, sig_col, asc, frac):
    s = sub.sort_values(sig_col, ascending=asc)
    k = max(1, int(len(s) * frac))
    g = s.iloc[:k]
    return g[ret_col].sum() / (stake * k) * 100, k

def frontier_table(data, label):
    print(f"\n{'='*78}\n{label}  (n={len(data):,}レース)\n{'='*78}")
    for bname, rcol, stake in BETS:
        print(f"\n[{bname}]  買い数%→ " + "  ".join(f"{int(f*100):>3}%" for f in FRACS))
        for sname, scol, asc in SIGNALS:
            rois = [roi_at(data, rcol, stake, scol, asc, f)[0] for f in FRACS]
            print(f"   {sname:9s} " + "  ".join(f"{r:5.1f}" for r in rois))

frontier_table(te, "ROI vs 買い数フロンティア (OOS 2025)")

# ===== 2024で最適(券種×信号×買い数)を選び 2025で検証 =====
print(f"\n{'='*78}\n2024で選定 → 2025で検証 (選別バイアス対策)\n{'='*78}")
rows = []
for bname, rcol, stake in BETS:
    for sname, scol, asc in SIGNALS:
        for f in FRACS:
            r24, k24 = roi_at(tr, rcol, stake, scol, asc, f)
            r25, k25 = roi_at(te, rcol, stake, scol, asc, f)
            rows.append(dict(券種=bname, 信号=sname, 買い数=f, roi24=round(r24,1),
                             n24=k24, roi25=round(r25,1), n25=k25))
res = pd.DataFrame(rows)
# 実用最低買い数: 2025で n>=200 のものに限定 (小サンプル妙味を除外)
usable = res[res.n25 >= 200].copy()
print("\n--- 2024 ROI 上位10 (n25>=200) → その2025実績 ---")
top = usable.sort_values("roi24", ascending=False).head(10)
print(top.to_string(index=False))
print("\n--- 2025 単体で最も高いROI (n25>=200, 参考: 上限の目安) ---")
print(usable.sort_values("roi25", ascending=False).head(8).to_string(index=False))

# 全レース強制(見送り無し)のベースライン比較
print(f"\n{'='*78}\n見送り無し(全レース強制)ベースライン 2025\n{'='*78}")
for bname, rcol, stake in BETS:
    r, k = roi_at(te, rcol, stake, "p1_max", False, 1.0)
    print(f"   {bname:9s} ROI={r:5.1f}%  (n={k:,})")
